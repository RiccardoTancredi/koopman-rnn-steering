"""How do the DMS models hold the sample in memory during the delay?

For every solved DMS model:
- speed q = 1/2 |F(h) - h|^2 along the delay states the network visits;
- fixed points (q < 1e-12) and slow points (q < 1e-5), with the largest
  Jacobian eigenvalue modulus at the slow points;
- whether the sample is still decodable after the delay is extended to 200
  steps (5x the longest delay seen in training): nearest-centroid decoder,
  centroids from half of the trials, accuracy on the other half.

    uv run python scripts/06_dms_memory.py
Output: results/analysis/dms_memory/summary.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from krnn.fixed_points import find_fixed_points
from krnn.rollout import autonomous_input, autonomous_pool, rollout, task_trajectories
from krnn.train import load_run

ROOT = Path(__file__).resolve().parents[1]


def decode_acc(H, s, n_stim):
    """Nearest-centroid accuracy, centroids from the first half, tested on the second."""
    half = len(H) // 2
    c = np.stack([H[:half][s[:half] == k].mean(0) for k in range(n_stim)])
    pred = np.argmin(((H[half:, None] - c[None]) ** 2).sum(-1), 1)
    return float((pred == s[half:]).mean())


def main():
    out = ROOT / "results" / "analysis" / "dms_memory"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for run_dir in sorted((ROOT / "results" / "models").glob("dms_*")):
        if json.loads((run_dir / "metrics.json").read_text())["accuracy"] < 0.95:
            continue
        model, task, cfg = load_run(run_dir)
        batch, _, h = task_trajectories(model, task, 512, seed=20_000 + cfg.seed)
        pool = autonomous_pool(task, batch, h).double()
        x = autonomous_input(task)
        m = model.double()
        with torch.no_grad():
            q = 0.5 * ((m.step(pool, x.double().expand(len(pool), -1)) - pool) ** 2).sum(-1)
        rng = np.random.default_rng(cfg.seed)
        inits = pool[torch.as_tensor(rng.choice(len(pool), 256, replace=False))]
        fixed = find_fixed_points(model, x, inits, q_tol=1e-12)
        slow = find_fixed_points(model, x, inits, q_tol=1e-5)

        s = np.asarray(batch.info["sample"])
        d0 = task.t_fix + task.t_stim
        with torch.no_grad():
            long = rollout(m, x, h[:, d0 - 1].double(), 200).numpy()
        summary[run_dir.name] = {
            "q_delay_median": float(q.median()),
            "n_fixed_points": len(fixed),
            "n_stable_fixed_points": int(fixed.stable.sum()),
            "n_slow_points_q<1e-5": len(slow),
            "slow_points_median_max_abs_eig": float(np.median(np.abs(slow.eigvals[:, 0])))
            if len(slow) else None,
            "decode_acc_delay40": decode_acc(long[:, 40], s, task.n_stim),
            "decode_acc_delay200": decode_acc(long[:, 200], s, task.n_stim),
        }
        print(run_dir.name, json.dumps(summary[run_dir.name]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
