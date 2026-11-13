"""Training loop and run I/O.

A run is a directory ``<out>/<task>_<cell>_h<n_hidden>_s<seed>/`` with
``config.json``, ``model.pt`` (state dict), ``history.json`` and ``metrics.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from .models import RNN, build_model
from .tasks import make_task


@dataclass
class TrainConfig:
    task: str
    cell: str
    seed: int = 0
    n_hidden: int = 64
    n_steps: int = 3000
    batch_size: int = 64
    lr: float = 1e-3
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batch_size: int = 512
    task_kwargs: dict = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        return f"{self.task}_{self.cell}_h{self.n_hidden}_s{self.seed}"


def load_config(path: str | Path, **overrides) -> TrainConfig:
    """Read a YAML config. An optional ``cell_overrides: {tanh: {...}, gru: {...}}``
    block sets per-cell values; explicit keyword overrides win over both."""
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    per_cell = d.pop("cell_overrides", {}) or {}
    d.update(per_cell.get(overrides.get("cell", d.get("cell")), {}))
    d.update(overrides)
    return TrainConfig(**d)


def masked_mse(y: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((y - target) ** 2) * mask).sum() / mask.sum()


def train(cfg: TrainConfig, out_root: str | Path, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    eval_rng = np.random.default_rng(10_000 + cfg.seed)

    task = make_task(cfg.task, **cfg.task_kwargs)
    model = build_model(cfg.cell, task, cfg.n_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_steps)
    eval_batch = task.sample(cfg.eval_batch_size, eval_rng)

    history = []
    t0 = time.time()
    for step in range(1, cfg.n_steps + 1):
        batch = task.sample(cfg.batch_size, rng)
        y, _ = model(batch.inputs)
        loss = masked_mse(y, batch.targets, batch.mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            with torch.no_grad():
                y_eval, _ = model(eval_batch.inputs)
            metrics = task.evaluate(y_eval, eval_batch)
            history.append({"step": step, "train_loss": loss.item(), **metrics})
            if verbose:
                shown = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[{cfg.run_name}] step {step:5d} loss={loss.item():.4f} {shown}", flush=True)

    run_dir = Path(out_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    final = {**history[-1], "train_seconds": time.time() - t0}
    for name, obj in [("config", asdict(cfg)), ("history", history), ("metrics", final)]:
        with open(run_dir / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=2)
    return final


def load_run(run_dir: str | Path) -> tuple[RNN, object, TrainConfig]:
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        cfg = TrainConfig(**json.load(f))
    task = make_task(cfg.task, **cfg.task_kwargs)
    model = build_model(cfg.cell, task, cfg.n_hidden)
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    model.eval()
    return model, task, cfg
