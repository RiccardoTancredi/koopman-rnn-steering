"""Interventions along Koopman modes.

A Koopman model gives a latent map ``z = phi(h)`` with linear dynamics
``z' = K z``. With ``K = V diag(lambda) V^{-1}``, the *eigen-coordinates*
``c = V^{-1} z`` evolve independently, ``c_j' = lambda_j c_j``; each
``c_j(h) = u_j . phi(h)`` (``u_j`` a left eigenvector) is an approximate
Koopman eigenfunction of the network's autonomous map. Interventions:

- ``rotate_mode``: multiply one complex eigen-coordinate by ``exp(i * angle)``
  (and its conjugate by ``exp(-i * angle)``) and decode. For an oscillatory
  mode this is a phase shift, i.e. a jump forward in time by ``angle / arg(lambda)``.
- ``eigenfunction_grad``: gradient in hidden space of a real linear
  combination of slow eigen-coordinates; pushing ``h`` along it changes that
  combination fastest per unit norm.

The same code works for DMD (linear ``phi``) and for the Koopman autoencoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .koopman.dmd import DMD
from .koopman.kae import KAEConfig, KoopmanAE


def load_kae(path) -> KoopmanAE:
    ckpt = torch.load(path, weights_only=False)
    cfg = KAEConfig(**ckpt["config"])
    d = ckpt["d"]
    kae = KoopmanAE(d, cfg, torch.zeros(d), torch.ones(d))
    kae.load_state_dict(ckpt["state_dict"])
    return kae.double().eval()


class LatentMap:
    """Common interface: encode/decode in float64 torch, and the latent operator K."""

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    K: np.ndarray


class KAELatent(LatentMap):
    def __init__(self, kae: KoopmanAE):
        self.kae = kae.double().eval()
        self.K = kae.K.detach().double().numpy()

    def encode(self, h):
        return self.kae.encode(h)

    def decode(self, z):
        return self.kae.decode(z)


class DMDLatent(LatentMap):
    """DMD as a latent map centred on the model's fixed point: z = P^T (h - m) - z*.

    With z* solving (I - A) z* = c (least squares if I - A is singular), the
    affine model z' = A z + c becomes linear, z' = A z, in these coordinates, so
    rotating a mode rotates about the fixed point and not about the data mean.
    The KAE needs no such shift: K has no offset, so its fixed point is z = 0.
    """

    def __init__(self, dmd: DMD):
        self.m = torch.as_tensor(dmd.mean)
        self.P = torch.as_tensor(dmd.P)
        self.K = dmd.A
        I_A = np.eye(dmd.rank) - dmd.A
        self.z_star = torch.as_tensor(np.linalg.lstsq(I_A, dmd.c, rcond=None)[0])

    def encode(self, h):
        return (h - self.m) @ self.P - self.z_star

    def decode(self, z):
        return self.m + (z + self.z_star) @ self.P.T


@dataclass
class Modes:
    eigvals: np.ndarray   # (k,)
    right: np.ndarray     # (L, k) right eigenvectors
    left: np.ndarray      # (k, L) rows of V^{-1}


def slow_modes(K: np.ndarray, k: int) -> Modes:
    """The ``k`` eigenvalues of largest modulus, with right and left eigenvectors."""
    lam, V = np.linalg.eig(K)
    Vinv = np.linalg.inv(V)
    order = np.argsort(-np.abs(lam))[:k]
    return Modes(lam[order], V[:, order], Vinv[order])


def eigen_coords(lmap: LatentMap, modes: Modes, h: torch.Tensor) -> torch.Tensor:
    """Complex eigen-coordinates c(h), (N, k)."""
    z = lmap.encode(h).to(torch.complex128)
    return z @ torch.as_tensor(modes.left).T


def rotate_mode(lmap: LatentMap, modes: Modes, h: torch.Tensor, j: int, angle) -> torch.Tensor:
    """Rotate eigen-coordinate j (a complex mode) by ``angle`` (scalar or (N,)) and decode."""
    z = lmap.encode(h)
    c = (z.to(torch.complex128) @ torch.as_tensor(modes.left[j]))          # (N,)
    rot = torch.exp(1j * torch.as_tensor(angle, dtype=torch.float64)) - 1
    v = torch.as_tensor(modes.right[:, j])                                  # (L,)
    dz = 2 * torch.real((c * rot)[:, None] * v[None, :])                    # mode + conjugate
    return lmap.decode(z + dz)


def shift_coords(lmap: LatentMap, modes: Modes, h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Add ``delta`` (N, k) to the eigen-coordinates of the selected modes, keep the rest, decode.

    The selected modes must be real or closed under conjugation (with ``delta``
    conjugate-symmetric), so that the latent change is real.
    """
    z = lmap.encode(h)
    dz = torch.as_tensor(delta, dtype=torch.complex128) @ torch.as_tensor(modes.right).T
    if dz.imag.abs().max() > 1e-8 * (1 + dz.real.abs().max()):
        raise ValueError("latent change is not real: modes not closed under conjugation")
    return lmap.decode(z + dz.real)


def fit_readout_of_coords(coords: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Least squares targets ~ [coords, 1] @ A. coords real (N, k), targets (N, m) -> A (k+1, m)."""
    X = np.hstack([coords, np.ones((len(coords), 1))])
    return np.linalg.lstsq(X, targets, rcond=None)[0]


def eigenfunction_grad(lmap: LatentMap, modes: Modes, a: np.ndarray, h: torch.Tensor):
    """Value and hidden-space gradient of f(h) = Re(c(h)) @ a[:-1] + a[-1], per sample.

    ``a`` is one column returned by :func:`fit_readout_of_coords`. Returns (f (N,), grad (N, d)).
    """
    h = h.detach().clone().requires_grad_(True)
    c = eigen_coords(lmap, modes, h).real
    f = c @ torch.as_tensor(a[:-1]) + float(a[-1])
    (g,) = torch.autograd.grad(f.sum(), h)
    return f.detach(), g
