"""Collecting hidden states: task trajectories and autonomous relaxation data.

Koopman models describe an *autonomous* map. For each task we therefore fix
the input to its value during the autonomous stretches (zero for flip-flop and
the DMS delay, the frequency code for sine generation) and study

    h_{t+1} = F(h_t) := f(h_t, x*).

Fitting data come from *relaxation rollouts*: states visited by the network on
the task, perturbed with Gaussian noise of scale ``sigma`` (relative to the
per-unit standard deviation of those states), then iterated under F. Without
perturbations the states mostly sit on attractors and any linear model that
predicts "nothing moves" looks good; transients are what test a dynamical
description.
"""

from __future__ import annotations

import numpy as np
import torch

from .models import RNN


def autonomous_input(task, omega: float | None = None) -> torch.Tensor:
    """The fixed input x* of the autonomous map for this task (and condition)."""
    if task.name == "sine":
        if omega is None:
            raise ValueError("sine task needs omega")
        return torch.tensor([float(task.input_for(omega))])
    return torch.zeros(task.n_in)


@torch.no_grad()
def task_trajectories(model: RNN, task, n_trials: int, seed: int, **sample_kwargs):
    """Run the model on fresh trials. Returns (batch, y, h) with h: (B, T, H)."""
    batch = task.sample(n_trials, np.random.default_rng(seed), **sample_kwargs)
    y, h = model(batch.inputs)
    return batch, y, h


def autonomous_pool(task, batch, h: torch.Tensor, skip: int = 20) -> torch.Tensor:
    """States the network visits while running autonomously, (N, H).

    For sine generation the first ``skip`` steps (transient from h=0) are dropped.
    """
    mask = batch.autonomous.clone()
    if task.name == "sine":
        mask[:, :skip] = False
    return h[mask]


@torch.no_grad()
def rollout(model: RNN, x: torch.Tensor, h0: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Iterate F(h) = f(h, x) from h0 (N, H). Returns (N, n_steps + 1, H), including h0."""
    x = x.to(h0.dtype).expand(h0.shape[0], -1)
    out = [h0]
    h = h0
    for _ in range(n_steps):
        h = model.step(h, x)
        out.append(h)
    return torch.stack(out, dim=1)


def relaxation_data(model: RNN, x: torch.Tensor, pool: torch.Tensor, n: int, n_steps: int,
                    sigma: float, rng: np.random.Generator, center: torch.Tensor | None = None):
    """Perturbed autonomous rollouts.

    Initial states are ``base + sigma * std * noise`` where ``base`` is a random
    state from ``pool`` (or ``center`` if given, for local analysis around a
    fixed point) and ``std`` is the per-unit standard deviation of ``pool``.
    Returns (n, n_steps + 1, H).
    """
    dtype = pool.dtype
    std = pool.std(0)
    if center is None:
        base = pool[torch.as_tensor(rng.integers(len(pool), size=n))]
    else:
        base = center.to(dtype).expand(n, -1)
    noise = torch.as_tensor(rng.standard_normal((n, pool.shape[1])), dtype=dtype)
    return rollout(model, x, base + sigma * std * noise, n_steps)


def snapshot_pairs(traj: torch.Tensor | np.ndarray):
    """(N, T+1, H) trajectories -> (X, Y) one-step pairs, each (N*T, H)."""
    traj = np.asarray(traj)
    X = traj[:, :-1].reshape(-1, traj.shape[-1])
    Y = traj[:, 1:].reshape(-1, traj.shape[-1])
    return X, Y
