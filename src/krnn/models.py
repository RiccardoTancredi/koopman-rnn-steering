"""Small recurrent networks with explicit access to their one-step map.

Training uses ``nn.RNN`` / ``nn.GRU`` for speed; analysis uses :meth:`RNN.step`,
the same map ``h_t = f(h_{t-1}, x_t)`` written out from the same weights, so
that fixed points, Jacobians and interventions are computed on exactly the
function the network was trained with (checked in ``tests/test_models.py``).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

Cell = Literal["tanh", "gru"]


class RNN(nn.Module):
    def __init__(self, cell: Cell, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.cell, self.n_in, self.n_hidden, self.n_out = cell, n_in, n_hidden, n_out
        if cell == "tanh":
            self.core = nn.RNN(n_in, n_hidden, nonlinearity="tanh", batch_first=True)
            nn.init.orthogonal_(self.core.weight_hh_l0)
        elif cell == "gru":
            self.core = nn.GRU(n_in, n_hidden, batch_first=True)
        else:
            raise ValueError(f"unknown cell {cell!r}")
        self.readout = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """x: (B, T, n_in) -> outputs (B, T, n_out), hidden states (B, T, n_hidden)."""
        h, _ = self.core(x, None if h0 is None else h0.unsqueeze(0))
        return self.readout(h), h

    def step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One step of the recurrence, h_t = f(h_{t-1}, x_t). Works on (..., n_hidden)."""
        c = self.core
        W_ih, W_hh, b_ih, b_hh = c.weight_ih_l0, c.weight_hh_l0, c.bias_ih_l0, c.bias_hh_l0
        if self.cell == "tanh":
            return torch.tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        # GRU, PyTorch gate order (r, z, n).
        gi = x @ W_ih.T + b_ih
        gh = h @ W_hh.T + b_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h

    def jacobian(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """d f(h, x) / d h. h: (N, H), x: (N, n_in) or (n_in,) -> (N, H, H)."""
        if x.dim() == 1:
            x = x.expand(h.shape[0], -1)
        return torch.func.vmap(torch.func.jacrev(self.step, argnums=0))(h, x)

    def output(self, h: torch.Tensor) -> torch.Tensor:
        return self.readout(h)


def build_model(cell: Cell, task, n_hidden: int) -> RNN:
    return RNN(cell, task.n_in, n_hidden, task.n_out)
