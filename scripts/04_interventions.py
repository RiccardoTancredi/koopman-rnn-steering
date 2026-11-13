"""Interventions along Koopman modes.

Flip-flop: flip one memory bit without any input pulse. Start states are
states the network visits during the task whose bits are stable for 60 steps.
The hidden state is pushed by ``eps`` (relative to the distance between
neighbouring attractors) along a unit direction, the network then runs
autonomously for 60 steps, and success means that bit k flipped and the other
bits did not. Directions:

- kae_edit / dmd_edit: move the 3 slowest eigen-coordinates a fraction lambda
  of the way to their mean value for the target memory pattern, keep all other
  latent coordinates, decode (transport in Koopman coordinates, AcT-style);
- kae / dmd: gradient of the slow eigenfunction that best encodes bit k
  (linear readout of the 3 slowest eigen-coordinates, fitted on visited states);
- readout:   the output weight vector of bit k (what the network reads out);
- jacobian:  linear response after 60 steps at the nearest stable fixed point,
             (J^T)^60 w_k;
- meanshift: difference of the mean hidden states of the two memory patterns
             (the "mean transport" of Mean-AcT, without its strength knob);
- random:    a random unit direction.

Sine generation (omega = 0.30): rotate the slow oscillatory eigen-coordinate
by an angle, decode, run the network, and measure the phase shift of its
output; compared with the same rotation in DMD coordinates and with random
perturbations of the same norm.

    uv run python scripts/04_interventions.py
Outputs: results/analysis/interventions/<run>.npz and summary.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from krnn.interventions import (
    DMDLatent,
    KAELatent,
    eigen_coords,
    eigenfunction_grad,
    fit_readout_of_coords,
    load_kae,
    rotate_mode,
    shift_coords,
    slow_modes,
)
from krnn.koopman import fit_dmd
from krnn.rollout import (
    autonomous_input,
    autonomous_pool,
    relaxation_data,
    rollout,
    snapshot_pairs,
    task_trajectories,
)
from krnn.train import load_run

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_STEPS = 60
EPS_REL = np.geomspace(0.02, 3.0, 16)
METHODS = ["kae_edit", "dmd_edit", "kae", "dmd", "readout", "jacobian", "meanshift", "random"]
LAMBDAS = np.linspace(0.1, 1.5, 15)


def solved(metrics):
    return metrics["accuracy"] >= 0.95 if "accuracy" in metrics else metrics["nmse"] <= 0.05


def setup(run_dir, omega=None, n_pool_trials=256):
    """Model, task, autonomous input, visited states, and the DMD/KAE fitted by scripts/03_koopman_ae.py (sigma = 0.5)."""
    model, task, cfg = load_run(run_dir)
    x = autonomous_input(task, omega).double()
    kw = {"omegas": omega} if omega is not None else {}
    batch, _, h = task_trajectories(model, task, n_pool_trials, seed=20_000 + cfg.seed, **kw)
    pool = autonomous_pool(task, batch, h).double()
    m = model.double()  # in place: task trajectories above must run in float32 first
    # Same training rollouts as scripts/03_koopman_ae.py, so the DMD is the same one.
    traj = relaxation_data(m, x, pool, 600, N_STEPS, 0.5, np.random.default_rng([cfg.seed, 500]))
    dmd = fit_dmd(*snapshot_pairs(traj.numpy()[:400]))
    cond = f"omega{omega:.2f}" if omega is not None else "auto"
    kae = load_kae(RES / "analysis" / "kae" / run_dir.name / f"{cond}_sigma0.5_L16.pt")
    return m, task, x, pool, {"kae": KAELatent(kae), "dmd": DMDLatent(dmd)}


@torch.no_grad()
def predicted_bits(lmap, H, W, b):
    """Final bits predicted by iterating the latent linear model for N_STEPS."""
    z = lmap.encode(H)
    Kt = torch.as_tensor(lmap.K).T
    for _ in range(N_STEPS):
        z = z @ Kt
    return torch.sign(lmap.decode(z) @ W.T + b)


def flipflop(run_dir, rng):
    m, task, x, pool, lmaps = setup(run_dir)
    W, b = m.readout.weight.detach(), m.readout.bias.detach()

    def bits(H):
        return torch.sign(H @ W.T + b)

    cand = pool[torch.as_tensor(rng.choice(len(pool), 800, replace=False))]
    with torch.no_grad():
        stable = (bits(rollout(m, x, cand, N_STEPS)[:, -1]) == bits(cand)).all(1)
    H0 = cand[stable][:200]
    B0 = bits(H0)
    N, d, nb = len(H0), H0.shape[1], task.n_bits

    # Slow eigen-coordinates and how well they encode the bits.
    Bpool = bits(pool).numpy()
    modes, readouts, decode_acc, coord_means = {}, {}, {}, {}
    for name, lm in lmaps.items():
        modes[name] = slow_modes(lm.K, nb)
        with torch.no_grad():
            cc = eigen_coords(lm, modes[name], pool)
        coord_means[name] = {tuple(p): cc[torch.as_tensor((Bpool == p).all(1))].mean(0)
                             for p in {tuple(r) for r in Bpool.tolist()}}
        c = cc.real.numpy()
        A = fit_readout_of_coords(c, Bpool)
        readouts[name] = A
        decode_acc[name] = float((np.sign(np.hstack([c, np.ones((len(c), 1))]) @ A) == Bpool).mean())

    # Distance scale: median distance between stable fixed points that differ in one bit.
    lin = np.load(RES / "analysis" / "linear" / run_dir.name / "auto.npz")
    fps = torch.as_tensor(lin["fp_h"][lin["fp_stable"]])
    fb = bits(fps)
    pairs = [(i, j) for i in range(len(fps)) for j in range(i + 1, len(fps))
             if (fb[i] != fb[j]).sum() == 1]
    d_flip = float(np.median([torch.linalg.norm(fps[i] - fps[j]).item() for i, j in pairs]))

    # Directions, (N, nb, d), each pointing towards flipping bit k.
    D = {}
    for name in ("kae", "dmd"):
        cols = []
        for k in range(nb):
            _, g = eigenfunction_grad(lmaps[name], modes[name], readouts[name][:, k], H0)
            cols.append(-B0[:, k:k + 1] * g)
        D[name] = torch.stack(cols, 1)
    D["readout"] = -B0[:, :, None] * W[None]
    J = m.jacobian(fps, x)                                        # (n_fp, d, d)
    resp = W[None].repeat(len(fps), 1, 1).transpose(1, 2)        # (n_fp, d, nb)
    for _ in range(N_STEPS):
        resp = J.transpose(1, 2) @ resp
        resp = resp / resp.norm(dim=1, keepdim=True)
    nearest = torch.cdist(H0, fps).argmin(1)
    D["jacobian"] = -B0[:, :, None] * resp[nearest].transpose(1, 2)
    patterns = {tuple(p): pool[(bits(pool) == torch.as_tensor(p)).all(1)].mean(0)
                for p in {tuple(r) for r in bits(pool).tolist()}}
    ms = torch.zeros(N, nb, d, dtype=torch.float64)
    for i in range(N):
        for k in range(nb):
            tgt = B0[i].clone()
            tgt[k] *= -1
            ms[i, k] = patterns[tuple(tgt.tolist())] - patterns[tuple(B0[i].tolist())]
    D["meanshift"] = ms
    D["random"] = torch.as_tensor(rng.standard_normal((N, nb, d)))
    D = {k: (v / v.norm(dim=-1, keepdim=True)).detach() for k, v in D.items()}

    E = len(EPS_REL)
    assert len(LAMBDAS) <= E
    out = {k: np.full((len(METHODS), nb, E), np.nan) for k in
           ("success", "other_changed", "agree_kae", "agree_dmd", "norm_rel")}
    for mi, meth in enumerate(METHODS):
        for k in range(nb):
            if meth.endswith("_edit"):
                name = meth[:-5]
                with torch.no_grad():
                    c0 = eigen_coords(lmaps[name], modes[name], H0)
                tgt = B0.clone()
                tgt[:, k] *= -1
                ct = torch.stack([coord_means[name][tuple(r)] for r in tgt.tolist()])
                with torch.no_grad():
                    Hp = torch.stack([shift_coords(lmaps[name], modes[name], H0, lam * (ct - c0))
                                      for lam in LAMBDAS])
                Ek = len(LAMBDAS)
            else:
                Hp = (H0[None] + torch.as_tensor(EPS_REL)[:, None, None] * d_flip
                      * D[meth][None, :, k])
                Ek = E
            out["norm_rel"][mi, k, :Ek] = ((Hp - H0[None]).norm(dim=-1).median(1).values
                                           / d_flip).numpy()
            Hp = Hp.reshape(-1, d)
            with torch.no_grad():
                fin = bits(rollout(m, x, Hp, N_STEPS)[:, -1]).reshape(Ek, N, nb)
            B = B0[None].expand(Ek, -1, -1)
            others = [j for j in range(nb) if j != k]
            flipped = fin[..., k] == -B[..., k]
            kept = (fin[..., others] == B[..., others]).all(-1)
            success = flipped & kept
            out["success"][mi, k, :Ek] = success.double().mean(1).numpy()
            out["other_changed"][mi, k, :Ek] = (~kept).double().mean(1).numpy()
            for name in ("kae", "dmd"):
                pb = predicted_bits(lmaps[name], Hp, W, b).reshape(Ek, N, nb)
                p_succ = (pb[..., k] == -B[..., k]) & (pb[..., others] == B[..., others]).all(-1)
                out[f"agree_{name}"][mi, k, :Ek] = (p_succ == success).double().mean(1).numpy()

    summary = {"n_start": N, "d_flip": d_flip,
               "bit_decoding_acc_slow_coords": decode_acc,
               "slow_eigvals_kae": np.round(np.abs(modes["kae"].eigvals), 4).tolist(),
               "slow_eigvals_dmd": np.round(np.abs(modes["dmd"].eigvals), 4).tolist()}
    for mi, meth in enumerate(METHODS):
        s = out["success"][mi].mean(0)
        nr = np.nanmean(out["norm_rel"][mi], 0)
        best = int(np.nanargmax(s))
        above = np.nonzero(s >= 0.5)[0]
        summary[meth] = {
            "max_success": round(float(np.nanmax(s)), 3),
            "norm_at_max": round(float(nr[best]), 3),
            "norm50": round(float(nr[above[0]]), 3) if len(above) else None,
            "other_changed_at_50": round(float(out["other_changed"][mi].mean(0)[above[0]]), 3)
            if len(above) else None,
            "agree_kae": round(float(np.nanmean(out["agree_kae"][mi])), 3),
            "agree_dmd": round(float(np.nanmean(out["agree_dmd"][mi])), 3),
        }
    return {**out, "eps_rel": EPS_REL, "lambdas": LAMBDAS}, summary


def fit_phase(y, omega, t0=20):
    """y (N, T) ~ A sin(omega t + phi) + c on t >= t0. Returns A, phi, R^2 (each (N,))."""
    t = np.arange(y.shape[1])[t0:]
    X = np.stack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t, dtype=float)], 1)
    Y = y[:, t0:].T
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = ((Y - X @ coef) ** 2).sum(0)
    r2 = 1 - resid / ((Y - Y.mean(0)) ** 2).sum(0)
    return np.hypot(coef[0], coef[1]), np.arctan2(coef[1], coef[0]), r2


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def sine(run_dir, rng, omega=0.30):
    m, _, x, pool, lmaps = setup(run_dir, omega)
    W, b = m.readout.weight.detach(), m.readout.bias.detach()
    H0 = pool[torch.as_tensor(rng.choice(len(pool), 200, replace=False))]

    def output(H, T=140):
        with torch.no_grad():
            return (rollout(m, x, H, T) @ W.T + b)[..., 0].numpy()

    # The network's own frequency (close to, not exactly, the trained omega).
    y0 = output(H0)
    grid = np.linspace(omega - 0.02, omega + 0.02, 401)
    w_net = grid[np.argmax([fit_phase(y0, w)[2].mean() for w in grid])]
    A0, phi0, _ = fit_phase(y0, w_net)

    angles = np.arange(-6, 6) * np.pi / 6
    res = {"angles": angles, "omega_net": np.array(w_net)}
    norms = {}
    for name, lm in lmaps.items():
        modes = slow_modes(lm.K, 2)
        j = int(np.argmax(np.angle(modes.eigvals)))
        res[f"mode_angle_{name}"] = np.array(np.angle(modes.eigvals[j]))
        dphi, amp, nrm = [], [], []
        for a in angles:
            with torch.no_grad():
                Hp = rotate_mode(lm, modes, H0, j, a)
            A, phi, _ = fit_phase(output(Hp), w_net)
            dphi.append(wrap(phi - phi0))
            amp.append(A / A0)
            nrm.append((Hp - H0).norm(dim=1).numpy())
        res[f"dphi_{name}"], res[f"amp_{name}"] = np.array(dphi), np.array(amp)
        norms[name] = np.array(nrm)
        # Rotating by k * omega should land where the network is after k steps.
        with torch.no_grad():
            traj = rollout(m, x, H0, 10)
            scale = (traj - traj.mean((0, 1))).norm(dim=-1).mean()
            res[f"timeshift_err_{name}"] = np.array([
                ((rotate_mode(lm, modes, H0, j, k * w_net) - traj[:, k]).norm(dim=1).mean()
                 / scale).item() for k in range(1, 11)])
    # Random perturbations with the same norms as the KAE rotations.
    dphi, amp = [], []
    for ai in range(len(angles)):
        u = torch.as_tensor(rng.standard_normal(H0.shape))
        Hp = H0 + torch.as_tensor(norms["kae"][ai])[:, None] * u / u.norm(dim=1, keepdim=True)
        A, phi, _ = fit_phase(output(Hp), w_net)
        dphi.append(wrap(phi - phi0))
        amp.append(A / A0)
    res["dphi_random"], res["amp_random"] = np.array(dphi), np.array(amp)

    summary = {"omega_net": round(float(w_net), 4)}
    for name in ("kae", "dmd", "random"):
        err = np.abs(wrap(res[f"dphi_{name}"] - angles[:, None]))
        summary[name] = {"median_phase_error_rad": round(float(np.median(err)), 3),
                         "p90_phase_error_rad": round(float(np.percentile(err, 90)), 3),
                         "median_amp_ratio": round(float(np.median(res[f"amp_{name}"])), 3)}
        if name != "random":
            summary[name]["mode_angle"] = round(float(res[f"mode_angle_{name}"]), 4)
            summary[name]["timeshift_err_k1_k10"] = [
                round(float(res[f"timeshift_err_{name}"][i]), 3) for i in (0, 9)]
    return res, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--out", default=str(RES / "analysis" / "interventions"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for run_dir in sorted((RES / "models").glob("*_h*_s*")):
        if args.only and args.only not in run_dir.name:
            continue
        task = run_dir.name.split("_")[0]
        if task not in ("flipflop", "sine"):
            continue
        if not solved(json.loads((run_dir / "metrics.json").read_text())):
            continue
        rng = np.random.default_rng(7)
        res, s = (flipflop if task == "flipflop" else sine)(run_dir, rng)
        np.savez(out / f"{run_dir.name}.npz", **res)
        summary[run_dir.name] = s
        print(run_dir.name, json.dumps(s), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
