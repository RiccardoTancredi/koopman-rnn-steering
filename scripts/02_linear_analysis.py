"""Fixed points, Jacobians and DMD for every trained model that solves its task.

For each model and condition (sine: three test frequencies) this computes
1. fixed points of the autonomous map and their Jacobian spectra;
2. a global affine DMD fitted on perturbed relaxation rollouts (sigma = 0.5),
   and its k-step prediction error on held-out rollouts, against persistence;
3. local DMD around each fixed point for decreasing perturbation size, and its
   spectral distance to the Jacobian at that point.

    uv run python scripts/02_linear_analysis.py
Outputs: results/analysis/linear/<run>/<condition>.npz and summary.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from krnn.fixed_points import find_fixed_points
from krnn.koopman import fit_dmd, spectral_distance
from krnn.rollout import (
    autonomous_input,
    autonomous_pool,
    relaxation_data,
    snapshot_pairs,
    task_trajectories,
)
from krnn.train import load_run

ROOT = Path(__file__).resolve().parents[1]
SINE_OMEGAS = [0.15, 0.3, 0.45]
LOCAL_SIGMAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
GLOBAL_SIGMA = 0.5
GLOBAL_SWEEP = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
HORIZON = 60
K_SLOW = 10          # eigenvalues of largest modulus used in spectral distances
MAX_LOCAL_FPS = 8


def solved(metrics: dict) -> bool:
    if "accuracy" in metrics:
        return metrics["accuracy"] >= 0.95
    return metrics["nmse"] <= 0.05


def nmse_curve(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Error at each horizon, normalised by the variance of the true states."""
    var = ((true - true.reshape(-1, true.shape[-1]).mean(0)) ** 2).sum(-1).mean()
    return ((pred - true) ** 2).sum(-1).mean(0) / var


def analyse_condition(model, task, seed: int, omega=None) -> dict:
    rng = np.random.default_rng(seed)
    x = autonomous_input(task, omega)
    kw = {"omegas": omega} if omega is not None else {}
    batch, y, h = task_trajectories(model, task, 256, seed=20_000 + seed, **kw)
    pool = autonomous_pool(task, batch, h).double()
    m64 = model.double()

    # 1. Fixed points: start from visited states, lightly and strongly perturbed.
    std = pool.std(0)
    idx = torch.as_tensor(rng.integers(len(pool), size=512))
    scales = torch.as_tensor(np.repeat([0.05, 1.0], 256))[:, None]
    noise = torch.as_tensor(rng.standard_normal((512, pool.shape[1])))
    inits = pool[idx] + scales * std * noise
    fps = find_fixed_points(model, x, inits)

    # 2. Global DMD on perturbed relaxations.
    traj = relaxation_data(m64, x, pool, 600, HORIZON, GLOBAL_SIGMA, rng).numpy()
    train, test = traj[:400], traj[400:]
    dmd_full = fit_dmd(*snapshot_pairs(train))
    dmd_r10 = fit_dmd(*snapshot_pairs(train), rank=10)
    W = model.readout.weight.detach().double().numpy()
    b = model.readout.bias.detach().double().numpy()

    def out(hh):
        return hh @ W.T + b

    curves = {}
    preds = {"dmd_full": dmd_full.rollout(test[:, 0], HORIZON),
             "dmd_r10": dmd_r10.rollout(test[:, 0], HORIZON),
             "persistence": np.repeat(test[:, :1], HORIZON + 1, axis=1)}
    for name, p in preds.items():
        curves[f"hidden_{name}"] = nmse_curve(p, test)
        curves[f"output_{name}"] = nmse_curve(out(p), out(test))

    # Same comparison for a range of perturbation sizes. For the flip-flop also:
    # does the model end in the same memory state (sign of each output bit)?
    sweep = {"sigma": np.array(GLOBAL_SWEEP), "hidden_dmd": [], "hidden_persist": [],
             "final_sign_dmd": [], "final_sign_persist": [], "basin_change": []}
    for s in GLOBAL_SWEEP:
        tr = relaxation_data(m64, x, pool, 600, HORIZON, s, rng).numpy()
        d = fit_dmd(*snapshot_pairs(tr[:400]))
        te = tr[400:]
        pr = d.rollout(te[:, 0], HORIZON)
        sweep["hidden_dmd"].append(nmse_curve(pr, te)[HORIZON])
        sweep["hidden_persist"].append(nmse_curve(np.repeat(te[:, :1], HORIZON + 1, 1), te)[HORIZON])
        final_true = np.sign(out(te[:, -1]))
        sweep["final_sign_dmd"].append(float((np.sign(out(pr[:, -1])) == final_true).mean()))
        sweep["final_sign_persist"].append(float((np.sign(out(te[:, 0])) == final_true).mean()))
        sweep["basin_change"].append(float((np.sign(out(te[:, 0])) != final_true).any(-1).mean()))
    sweep = {k: np.asarray(v) for k, v in sweep.items()}

    lam_global = dmd_full.eig()[0]
    global_vs_jac = [spectral_distance(lam_global, fps.eigvals[k], K_SLOW) for k in range(len(fps))]

    # 3. Local DMD around fixed points, shrinking perturbations.
    order = np.argsort(-fps.n_found)[:MAX_LOCAL_FPS]
    local = np.full((len(order), len(LOCAL_SIGMAS)), np.nan)
    for i, k in enumerate(order):
        center = torch.as_tensor(fps.h[k])
        for j, s in enumerate(LOCAL_SIGMAS):
            tr = relaxation_data(m64, x, pool, 200, 10, s, rng, center=center).numpy()
            d = fit_dmd(*snapshot_pairs(tr))
            local[i, j] = spectral_distance(d.eigvals, fps.eigvals[k], K_SLOW)

    # Low-dimensional view for figures: PCA of the task trajectories.
    H = h.reshape(-1, h.shape[-1]).double().numpy()
    mu = H.mean(0)
    Vt = np.linalg.svd(H - mu, full_matrices=False)[2]
    return {
        "fp_h": fps.h, "fp_q": fps.q, "fp_eigvals": fps.eigvals, "fp_n_found": fps.n_found,
        "fp_stable": fps.stable, "dmd_eigvals": lam_global, "dmd_r10_eigvals": dmd_r10.eig()[0],
        "global_vs_jac": np.array(global_vs_jac), "local_fp_index": order,
        "local_sigmas": np.array(LOCAL_SIGMAS), "local_dist": local,
        "pca_mean": mu, "pca_basis": Vt[:3].T, "traj_h": h[:20].double().numpy(),
        "traj_y": y[:20].double().numpy(), "traj_inputs": batch.inputs[:20].numpy(),
        **{f"curve_{k}": v for k, v in curves.items()},
        **{f"sweep_{k}": v for k, v in sweep.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=str(ROOT / "results" / "models"))
    p.add_argument("--out", default=str(ROOT / "results" / "analysis" / "linear"))
    p.add_argument("--only", default=None, help="substring filter on run names")
    args = p.parse_args()
    torch.set_grad_enabled(True)

    summary = {}
    for run_dir in sorted(Path(args.models).glob("*_h*_s*")):
        if args.only and args.only not in run_dir.name:
            continue
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if not solved(metrics):
            print(f"skip {run_dir.name} (task not solved)")
            summary[run_dir.name] = {"excluded": True}
            continue
        model, task, cfg = load_run(run_dir)
        conds = [(f"omega{w:.2f}", w) for w in SINE_OMEGAS] if task.name == "sine" else [("auto", None)]
        out_dir = Path(args.out) / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary[run_dir.name] = {}
        for cname, omega in conds:
            model, task, cfg = load_run(run_dir)  # fresh float32 copy each condition
            res = analyse_condition(model, task, cfg.seed, omega)
            np.savez(out_dir / f"{cname}.npz", **res)
            s = {
                "n_fixed_points": len(res["fp_h"]),
                "n_stable": int(res["fp_stable"].sum()),
                "hidden_nmse_dmd_h1": float(res["curve_hidden_dmd_full"][1]),
                "hidden_nmse_dmd_h10": float(res["curve_hidden_dmd_full"][10]),
                "hidden_nmse_dmd_h60": float(res["curve_hidden_dmd_full"][HORIZON]),
                "hidden_nmse_persist_h10": float(res["curve_hidden_persistence"][10]),
                "output_nmse_dmd_h10": float(res["curve_output_dmd_full"][10]),
                "sweep_hidden_dmd_h60": [round(float(v), 4) for v in res["sweep_hidden_dmd"]],
                "sweep_hidden_persist_h60": [round(float(v), 4) for v in res["sweep_hidden_persist"]],
                "sweep_final_sign_dmd": res["sweep_final_sign_dmd"].tolist(),
                "sweep_basin_change": res["sweep_basin_change"].tolist(),
                "min_global_vs_jac": float(np.min(res["global_vs_jac"])) if len(res["fp_h"]) else None,
                "local_dist_by_sigma": np.nanmean(res["local_dist"], 0).tolist()
                if len(res["local_dist"]) else None,
            }
            summary[run_dir.name][cname] = s
            print(run_dir.name, cname, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                   for k, v in s.items() if not k.startswith(("local", "sweep"))}),
                  flush=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
