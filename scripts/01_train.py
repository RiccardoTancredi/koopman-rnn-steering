"""Train TanhRNN and GRU models on the three tasks, several seeds each.

    uv run python scripts/01_train.py                       # everything
    uv run python scripts/01_train.py --tasks flipflop --cells gru --seeds 0
"""

import argparse
import json
from pathlib import Path

from krnn.train import load_config, train

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["flipflop", "sine", "dms"])
    p.add_argument("--cells", nargs="+", default=["tanh", "gru"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-steps", type=int, default=None, help="override config (for smoke tests)")
    p.add_argument("--out", default=str(ROOT / "results" / "models"))
    p.add_argument("--force", action="store_true", help="retrain runs that already exist")
    args = p.parse_args()

    out = Path(args.out)
    for task in args.tasks:
        for cell in args.cells:
            for seed in args.seeds:
                cfg = load_config(ROOT / "configs" / f"{task}.yaml", cell=cell, seed=seed,
                                  n_steps=args.n_steps)
                if (out / cfg.run_name / "model.pt").exists() and not args.force:
                    print(f"skip {cfg.run_name} (exists)")
                    continue
                train(cfg, out)

    summary = {}
    for run in sorted(out.glob("*/metrics.json")):
        summary[run.parent.name] = json.loads(run.read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, m in summary.items():
        shown = " ".join(f"{k}={v:.4f}" for k, v in m.items()
                         if k in ("accuracy", "nmse", "mse"))
        print(f"{name:28s} {shown}  ({m['train_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
