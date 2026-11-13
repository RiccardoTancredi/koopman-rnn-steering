"""Dynamic mode decomposition on hidden states.

The model is affine in a POD subspace of the data:

    z = P^T (h - m),    z' = A z + c,    h' = m + P z'

with ``P`` the top-``rank`` principal directions of the snapshots. With the
constant observable included this is exact DMD on the dictionary
``{1, z_1, ..., z_r}``; the eigenvalues reported are those of ``A`` (the
trivial eigenvalue 1 of the constant is left out). An affine rather than
linear model matters here: a trained RNN's attractors are not at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class DMD:
    mean: np.ndarray        # (d,)
    P: np.ndarray           # (d, r) orthonormal POD basis
    A: np.ndarray           # (r, r)
    c: np.ndarray           # (r,)

    @property
    def rank(self) -> int:
        return self.A.shape[0]

    @property
    def eigvals(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    def eig(self):
        """Eigenvalues (r,), modes in state space (d, r), sorted by |lambda| descending."""
        lam, V = np.linalg.eig(self.A)
        order = np.argsort(-np.abs(lam))
        return lam[order], self.P @ V[:, order]

    def encode(self, h: np.ndarray) -> np.ndarray:
        return (h - self.mean) @ self.P

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.mean + z @ self.P.T

    def step(self, h: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(h) @ self.A.T + self.c)

    def rollout(self, h0: np.ndarray, n_steps: int) -> np.ndarray:
        """(N, d) -> (N, n_steps + 1, d); iterated in the latent space."""
        z = self.encode(h0)
        out = [z]
        for _ in range(n_steps):
            z = z @ self.A.T + self.c
            out.append(z)
        return self.decode(np.stack(out, axis=1))

    def fixed_point(self) -> np.ndarray | None:
        """Fixed point of the affine model, if I - A is invertible."""
        try:
            z = np.linalg.solve(np.eye(self.rank) - self.A, self.c)
        except np.linalg.LinAlgError:
            return None
        return self.decode(z)


def fit_dmd(X: np.ndarray, Y: np.ndarray, rank: int | None = None, ridge: float = 1e-8) -> DMD:
    """Least-squares affine DMD from one-step pairs X -> Y, each (N, d).

    ``rank=None`` keeps all directions (the best affine one-step predictor).
    ``ridge`` is a small Tikhonov term relative to the mean squared singular value.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    m = X.mean(0)
    Vt = np.linalg.svd(X - m, full_matrices=False)[2]
    r = Vt.shape[0] if rank is None else min(rank, Vt.shape[0])
    P = Vt[:r].T
    Zx = (X - m) @ P
    Zy = (Y - m) @ P
    Z1 = np.hstack([Zx, np.ones((len(Zx), 1))])
    G = Z1.T @ Z1
    G += ridge * np.trace(G) / len(G) * np.eye(len(G))
    W = np.linalg.solve(G, Z1.T @ Zy)        # (r+1, r)
    return DMD(mean=m, P=P, A=W[:r].T, c=W[r])


def spectral_distance(lam_a: np.ndarray, lam_b: np.ndarray, k: int | None = None) -> float:
    """Mean |lambda_a - lambda_b| after optimal one-to-one matching.

    If ``k`` is given, only the ``k`` eigenvalues of largest modulus of each set are
    matched (the slow, long-lived part of the spectrum).
    """
    lam_a = np.asarray(lam_a)
    lam_b = np.asarray(lam_b)
    if k is not None:
        lam_a = lam_a[np.argsort(-np.abs(lam_a))[:k]]
        lam_b = lam_b[np.argsort(-np.abs(lam_b))[:k]]
    C = np.abs(lam_a[:, None] - lam_b[None, :])
    i, j = linear_sum_assignment(C)
    return float(C[i, j].mean())
