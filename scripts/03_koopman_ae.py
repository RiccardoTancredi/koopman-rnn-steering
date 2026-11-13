"""Koopman autoencoder vs affine DMD on the same relaxation data.

For each solved model (sine: omega = 0.30 only) and perturbation size sigma,
both models are fitted on the same 400 perturbed autonomous rollouts and
evaluated on the same 200 held-out rollouts, up to 60 steps.

    uv run python scripts/03_koopman_ae.py                       # all models, sigma 0.5 and 2.0
    uv run python scripts/03_koopman_ae.py --only flipflop_gru_h64_s0 --latent 4 8 16 32
Outputs: results/analysis/kae/<run>/<cond>_sigma<s>_L<L>.{npz,pt} and summary.json.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from krnn.koopman import KAEConfig, fit_dmd, fit_kae
from krnn.rollout import (
    autonomous_input,
    autonomous_pool,
    relaxation_data,
    snapshot_pairs,
    task_trajectories,
)
from krnn.train import load_run

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 60
SINE_OMEGA = 0.30


def solved(metrics: dict) -> bool:
    return metrics["accuracy"] >= 0.95 if "accuracy" in metrics else metrics["nmse"] <= 0.05


def nmse_curve(pred, true):
    var = ((true - true.reshape(-1, true.shape[-1]).mean(0)) ** 2).sum(-1).mean()
    return ((pred - true) ** 2).sum(-1).mean(0) / var


def compare(run_dir: Path, sigma: float, kcfg: KAEConfig):
    model, task, cfg = load_run(run_dir)
    omega = SINE_OMEGA if task.name == "sine" else None
    x = autonomous_input(task, omega)
    kw = {"omegas": omega} if omega is not None else {}
    batch, _, h = task_trajectories(model, task, 256, seed=20_000 + cfg.seed, **kw)
    pool = autonomous_pool(task, batch, h).double()
    rng = np.random.default_rng([cfg.seed, round(sigma * 1000)])
    traj = relaxation_data(model.double(), x, pool, 600, HORIZON, sigma, rng).numpy()
    train, test = traj[:400], traj[400:]

    t0 = time.time()
    kae, hist = fit_kae(train, kcfg)
    t_kae = time.time() - t0
    dmd = fit_dmd(*snapshot_pairs(train))

    W = model.readout.weight.detach().double().numpy()
    b = model.readout.bias.detach().double().numpy()
    preds = {"kae": kae.rollout(test[:, 0], HORIZON).astype(np.float64),
             "dmd": dmd.rollout(test[:, 0], HORIZON),
             "persistence": np.repeat(test[:, :1], HORIZON + 1, axis=1)}
    res = {f"curve_{k}": nmse_curve(p, test) for k, p in preds.items()}
    final_true = np.sign(test[:, -1] @ W.T + b)
    for k, p in preds.items():
        res[f"final_sign_{k}"] = np.array((np.sign(p[:, -1] @ W.T + b) == final_true).mean())
    res["basin_change"] = np.array((np.sign(test[:, 0] @ W.T + b) != final_true).any(-1).mean())
    res["kae_eigvals"] = kae.eigvals
    res["dmd_eigvals"] = dmd.eig()[0]
    res["kae_loss"] = np.array([[r["rec"], r["lin"], r["pred"]] for r in hist])
    return res, kae, t_kae, task.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=str(ROOT / "results" / "models"))
    p.add_argument("--out", default=str(ROOT / "results" / "analysis" / "kae"))
    p.add_argument("--only", default=None)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 2.0])
    p.add_argument("--latent", nargs="+", type=int, default=[16])
    p.add_argument("--n-steps", type=int, default=3000)
    args = p.parse_args()

    out = Path(args.out)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for run_dir in sorted(Path(args.models).glob("*_h*_s*")):
        if args.only and args.only not in run_dir.name:
            continue
        if not solved(json.loads((run_dir / "metrics.json").read_text())):
            continue
        cond = f"omega{SINE_OMEGA:.2f}" if run_dir.name.startswith("sine") else "auto"
        for sigma in args.sigmas:
            for L in args.latent:
                kcfg = KAEConfig(latent_dim=L, n_steps=args.n_steps)
                res, kae, secs, _ = compare(run_dir, sigma, kcfg)
                tag = f"{cond}_sigma{sigma:g}_L{L}"
                d = out / run_dir.name
                d.mkdir(parents=True, exist_ok=True)
                np.savez(d / f"{tag}.npz", **res)
                torch.save({"state_dict": kae.state_dict(), "config": asdict(kcfg),
                            "d": int(kae.mean.shape[0])}, d / f"{tag}.pt")
                s = {}
                for k in ("kae", "dmd", "persistence"):
                    s[f"{k}_h10"] = round(float(res[f"curve_{k}"][10]), 4)
                    s[f"{k}_h60"] = round(float(res[f"curve_{k}"][HORIZON]), 4)
                if run_dir.name.startswith("flipflop"):
                    s["final_sign_kae"] = round(float(res["final_sign_kae"]), 4)
                    s["final_sign_dmd"] = round(float(res["final_sign_dmd"]), 4)
                    s["basin_change"] = round(float(res["basin_change"]), 4)
                s["kae_n_eig_above_0.97"] = int((np.abs(res["kae_eigvals"]) > 0.97).sum())
                s["dmd_n_eig_above_0.97"] = int((np.abs(res["dmd_eigvals"]) > 0.97).sum())
                s["kae_seconds"] = round(secs, 1)
                summary.setdefault(run_dir.name, {})[tag] = s
                print(run_dir.name, tag, json.dumps(s), flush=True)
                summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
