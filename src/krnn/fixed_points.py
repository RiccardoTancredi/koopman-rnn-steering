"""Fixed and slow points of the autonomous map (Sussillo & Barak, 2013).

Minimise ``q(h) = 1/2 |F(h) - h|^2`` from many initial states with Adam, then
polish with a few Newton steps on ``F(h) - h = 0`` (which also converges to
unstable points). Points with ``q < q_tol`` are fixed points; they are
de-duplicated and their Jacobian spectra computed. Computation is in float64.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from .models import RNN


@dataclass
class FixedPoints:
    h: np.ndarray           # (K, d)
    q: np.ndarray           # (K,)
    eigvals: np.ndarray     # (K, d) complex, sorted by |lambda| descending
    n_found: np.ndarray     # (K,) how many initial states converged to each point

    @property
    def stable(self) -> np.ndarray:
        return np.abs(self.eigvals[:, 0]) < 1.0

    def __len__(self) -> int:
        return len(self.h)


def _q(model, h, x):
    return 0.5 * ((model.step(h, x) - h) ** 2).sum(-1)


def find_fixed_points(model: RNN, x: torch.Tensor, h_init: torch.Tensor, n_adam: int = 2000,
                      lr: float = 1e-2, n_newton: int = 20, q_tol: float = 1e-12,
                      unique_tol: float = 1e-4) -> FixedPoints:
    m = copy.deepcopy(model).double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    x = x.double().expand(h_init.shape[0], -1)
    h = h_init.detach().double().clone().requires_grad_(True)

    opt = torch.optim.Adam([h], lr=lr)
    for _ in range(n_adam):
        opt.zero_grad()
        _q(m, h, x).sum().backward()
        opt.step()

    h = h.detach()
    eye = torch.eye(h.shape[1], dtype=h.dtype)
    for _ in range(n_newton):
        r = m.step(h, x) - h                                   # (N, d)
        J = m.jacobian(h, x) - eye                             # (N, d, d)
        dh = torch.linalg.lstsq(J, -r.unsqueeze(-1)).solution.squeeze(-1)
        h_new = h + dh
        # Keep the Newton step only where it lowers q (guards against divergence).
        better = _q(m, h_new, x) < _q(m, h, x)
        h = torch.where(better[:, None], h_new, h)

    with torch.no_grad():
        q = _q(m, h, x)
    keep = q < q_tol
    h_ok, q_ok = h[keep], q[keep]

    # Greedy de-duplication in order of increasing q.
    order = torch.argsort(q_ok)
    reps, counts = [], []
    for i in order.tolist():
        for k, j in enumerate(reps):
            if torch.linalg.norm(h_ok[i] - h_ok[j]) < unique_tol * (1 + torch.linalg.norm(h_ok[j])):
                counts[k] += 1
                break
        else:
            reps.append(i)
            counts.append(1)

    if not reps:
        d = h.shape[1]
        return FixedPoints(np.zeros((0, d)), np.zeros(0), np.zeros((0, d), complex), np.zeros(0, int))
    H = h_ok[reps]
    J = m.jacobian(H, x[: len(reps)])
    lam = torch.linalg.eigvals(J).numpy()
    lam = np.take_along_axis(lam, np.argsort(-np.abs(lam), axis=1), axis=1)
    return FixedPoints(H.numpy(), q_ok[reps].numpy(), lam, np.array(counts))
