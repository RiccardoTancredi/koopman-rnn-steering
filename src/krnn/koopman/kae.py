"""Koopman autoencoder for hidden-state trajectories (after Lusch et al., 2018).

    z = phi(h),    z_{t+1} = K z_t,    h ~ psi(z)

``phi`` and ``psi`` are MLPs; ``K`` is a single dense L x L matrix, so the
model has one spectrum (``eig(K)``) that can be compared with DMD and with the
Jacobians at fixed points. Inputs are standardised per unit with statistics of
the training data.

Loss on windows ``h_0..h_S`` of autonomous rollouts (all in standardised units):

- reconstruction  |psi(phi(h_t)) - h_t|^2
- linearity       |K^k phi(h_0) - phi(h_k)|^2, divided by the latent variance
                  so that shrinking the latent space cannot reduce it; its weight
                  can be ramped up from 0 (``lin_warmup``), which avoids a collapse
                  to the mean on strongly perturbed data
- prediction      |psi(K^k phi(h_0)) - h_k|^2, k = 1..S
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class KAEConfig:
    latent_dim: int = 16
    width: int = 128
    depth: int = 2
    horizon: int = 30
    w_rec: float = 1.0
    w_lin: float = 1.0
    w_pred: float = 1.0
    # Fraction of training over which w_lin ramps up from 0. Without the ramp,
    # training collapses to predicting the mean on strongly perturbed data
    # (flip-flop, sigma = 2). A lower w_lin also avoids the collapse but blurs the
    # spectrum (toy test). Chosen on one model (flip-flop GRU, seed 0).
    lin_warmup: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_steps: int = 3000
    batch_size: int = 128
    seed: int = 0


def _mlp(d_in: int, d_out: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class KoopmanAE(nn.Module):
    def __init__(self, d: int, cfg: KAEConfig, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        L = cfg.latent_dim
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        self.encoder = _mlp(d, L, cfg.width, cfg.depth)
        self.decoder = _mlp(L, d, cfg.width, cfg.depth)
        self.K = nn.Parameter(0.9 * torch.eye(L) + 0.01 * torch.randn(L, L))

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.encoder((h - self.mean) / self.std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) * self.std + self.mean

    def rollout_latent(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        out = [z0]
        z = z0
        for _ in range(n_steps):
            z = z @ self.K.T
            out.append(z)
        return torch.stack(out, dim=-2)

    @torch.no_grad()
    def rollout(self, h0, n_steps: int) -> np.ndarray:
        """(N, d) -> (N, n_steps + 1, d), predicted by iterating K in the latent space."""
        h0 = torch.as_tensor(h0, dtype=self.mean.dtype)
        return self.decode(self.rollout_latent(self.encode(h0), n_steps)).numpy()

    @property
    def eigvals(self) -> np.ndarray:
        lam = np.linalg.eigvals(self.K.detach().double().numpy())
        return lam[np.argsort(-np.abs(lam))]

    def losses(self, win: torch.Tensor, w_lin: float | None = None) -> dict:
        """win: (B, S+1, d) windows of a trajectory."""
        S = win.shape[1] - 1
        x = (win - self.mean) / self.std
        z = self.encoder(x)
        rec = ((self.decoder(z) - x) ** 2).mean()
        zk = self.rollout_latent(z[:, 0], S)
        lin = ((zk[:, 1:] - z[:, 1:]) ** 2).mean() / (z.detach().var() + 1e-8)
        pred = ((self.decoder(zk[:, 1:]) - x[:, 1:]) ** 2).mean()
        c = self.cfg
        w_lin = c.w_lin if w_lin is None else w_lin
        total = c.w_rec * rec + w_lin * lin + c.w_pred * pred
        return {"total": total, "rec": rec, "lin": lin, "pred": pred}


def fit_kae(train: np.ndarray, cfg: KAEConfig, verbose: bool = False) -> tuple[KoopmanAE, list]:
    """Fit on trajectories (N, T+1, d); windows of length horizon+1 are sampled at random."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = torch.as_tensor(np.asarray(train), dtype=torch.float32)
    N, T1, d = data.shape
    S = min(cfg.horizon, T1 - 1)
    flat = data.reshape(-1, d)
    model = KoopmanAE(d, cfg, flat.mean(0), flat.std(0) + 1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_steps)

    history = []
    for step in range(1, cfg.n_steps + 1):
        i = torch.as_tensor(rng.integers(N, size=cfg.batch_size))
        t0 = torch.as_tensor(rng.integers(T1 - S, size=cfg.batch_size))
        idx = t0[:, None] + torch.arange(S + 1)[None]
        win = data[i[:, None], idx]                     # (B, S+1, d)
        ramp = 1.0 if cfg.lin_warmup <= 0 else min(1.0, step / (cfg.lin_warmup * cfg.n_steps))
        loss = model.losses(win, w_lin=cfg.w_lin * ramp)
        opt.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == cfg.n_steps:
            rec = {k: v.item() for k, v in loss.items()}
            history.append({"step": step, **rec})
            if verbose:
                print(f"  kae step {step:5d} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()),
                      flush=True)
    model.eval()
    return model, history


def config_dict(cfg: KAEConfig) -> dict:
    return asdict(cfg)
