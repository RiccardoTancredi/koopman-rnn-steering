"""Neuroscience-style tasks for small RNNs.

Every task samples a :class:`Batch` with

- ``inputs``     (B, T, n_in)
- ``targets``    (B, T, n_out)
- ``mask``       (B, T, n_out), 1 where the output enters the loss
- ``autonomous`` (B, T) bool, True where the network receives no new
  information, i.e. its update is ``h_t = f(h_{t-1}, u)`` with ``u`` fixed.
  These are the stretches on which Koopman models are fitted.
- ``info``       task variables (memorised bits, frequency, sample identity, ...)

Convention: ``h_t = f(h_{t-1}, x_t)`` and ``y_t = W h_t + b``, with ``h_{-1} = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    autonomous: torch.Tensor
    info: dict = field(default_factory=dict)


def _to_batch(inputs, targets, mask, autonomous, info) -> Batch:
    def f(a):
        return torch.as_tensor(a, dtype=torch.float32)

    return Batch(f(inputs), f(targets), f(mask), torch.as_tensor(autonomous, dtype=torch.bool), info)


class FlipFlop:
    """N-bit flip-flop (Sussillo & Barak, 2013).

    Each input channel receives sparse +1/-1 pulses; the matching output must
    hold the sign of the last pulse on that channel. Every bit is set at t=0,
    so the target is always defined. A trained network needs 2**n_bits stable
    states.
    """

    name = "flipflop"

    def __init__(self, n_bits: int = 3, T: int = 100, p_pulse: float = 0.04):
        self.n_bits, self.T, self.p_pulse = n_bits, T, p_pulse
        self.n_in = self.n_out = n_bits

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        B, T, n = batch_size, self.T, self.n_bits
        signs = rng.choice([-1.0, 1.0], size=(B, T, n))
        pulses = rng.random((B, T, n)) < self.p_pulse
        pulses[:, 0, :] = True
        inputs = pulses * signs

        targets = np.empty_like(inputs)
        state = inputs[:, 0, :].copy()
        for t in range(T):
            state = np.where(pulses[:, t, :], inputs[:, t, :], state)
            targets[:, t, :] = state

        autonomous = ~pulses.any(axis=-1)
        mask = np.ones_like(targets)
        return _to_batch(inputs, targets, mask, autonomous, {"bits": targets})

    @staticmethod
    def evaluate(y: torch.Tensor, batch: Batch) -> dict:
        correct = (torch.sign(y) == batch.targets).float()
        return {"accuracy": correct.mean().item(), "mse": ((y - batch.targets) ** 2).mean().item()}


class SineGeneration:
    """Sine-wave generation with frequency set by a static input (Sussillo & Barak, 2013).

    The input is a constant ``u = 0.25 + (omega - omega_min) / (omega_max - omega_min)``
    and the target is ``sin(omega * t)``. Because the input is constant, the
    whole trial is autonomous; a trained network needs a limit cycle whose
    frequency depends on ``u``.
    """

    name = "sine"

    def __init__(self, T: int = 150, omega_min: float = 0.1, omega_max: float = 0.6,
                 n_transient: int = 5):
        self.T, self.omega_min, self.omega_max = T, omega_min, omega_max
        self.n_transient = n_transient
        self.n_in = self.n_out = 1

    def input_for(self, omega):
        return 0.25 + (np.asarray(omega) - self.omega_min) / (self.omega_max - self.omega_min)

    def sample(self, batch_size: int, rng: np.random.Generator, omegas=None) -> Batch:
        B, T = batch_size, self.T
        if omegas is None:
            omegas = rng.uniform(self.omega_min, self.omega_max, size=B)
        omegas = np.broadcast_to(np.asarray(omegas, dtype=float), (B,))
        t = np.arange(T)
        inputs = np.broadcast_to(self.input_for(omegas)[:, None, None], (B, T, 1)).copy()
        targets = np.sin(omegas[:, None] * t[None, :])[..., None]
        mask = np.ones_like(targets)
        mask[:, : self.n_transient] = 0.0
        autonomous = np.ones((B, T), dtype=bool)
        return _to_batch(inputs, targets, mask, autonomous, {"omega": omegas.copy()})

    @staticmethod
    def evaluate(y: torch.Tensor, batch: Batch) -> dict:
        m = batch.mask
        mse = (((y - batch.targets) ** 2) * m).sum() / m.sum()
        # Normalised by the target variance (0.5 for a unit sine): 0 is perfect, 1 is chance.
        return {"mse": mse.item(), "nmse": (mse / 0.5).item()}


class DelayedMatchToSample:
    """Delayed match-to-sample.

    Trial: fixation, sample stimulus (one of ``n_stim``, one-hot), a delay of
    random length with no input, a test stimulus, then a go cue during which
    the output must be +1 if test == sample and -1 otherwise (0 before). Steps
    after the response window are padding and are masked out. The response
    window is weighted ``resp_weight`` times more in the loss; without it the
    network can sit in the "always output 0" minimum. The delay is the
    autonomous stretch on which the memory of the sample must be held.
    """

    name = "dms"

    def __init__(self, n_stim: int = 4, t_fix: int = 5, t_stim: int = 5,
                 delay: tuple[int, int] = (20, 40), t_resp: int = 10,
                 resp_weight: float = 5.0):
        self.n_stim, self.t_fix, self.t_stim = n_stim, t_fix, t_stim
        self.delay, self.t_resp, self.resp_weight = delay, t_resp, resp_weight
        self.T = t_fix + 2 * t_stim + delay[1] + t_resp
        self.n_in = n_stim + 1  # stimuli + go cue
        self.n_out = 1

    def sample(self, batch_size: int, rng: np.random.Generator, delays=None) -> Batch:
        B, T, S = batch_size, self.T, self.n_stim
        sample_id = rng.integers(S, size=B)
        match = rng.random(B) < 0.5
        other = (sample_id + rng.integers(1, S, size=B)) % S  # uniform over the other stimuli
        test_id = np.where(match, sample_id, other)
        if delays is None:
            delays = rng.integers(self.delay[0], self.delay[1] + 1, size=B)
        delays = np.broadcast_to(np.asarray(delays, dtype=int), (B,))

        inputs = np.zeros((B, T, self.n_in))
        targets = np.zeros((B, T, 1))
        mask = np.zeros((B, T, 1))
        autonomous = np.zeros((B, T), dtype=bool)
        s0 = self.t_fix
        for b in range(B):
            d0 = s0 + self.t_stim
            t0 = d0 + delays[b]
            r0 = t0 + self.t_stim
            r1 = r0 + self.t_resp
            inputs[b, s0:d0, sample_id[b]] = 1.0
            inputs[b, t0:r0, test_id[b]] = 1.0
            inputs[b, r0:r1, S] = 1.0
            targets[b, r0:r1, 0] = 1.0 if match[b] else -1.0
            mask[b, :r1] = 1.0
            mask[b, r0:r1] = self.resp_weight
            autonomous[b, d0:t0] = True

        info = {"sample": sample_id, "test": test_id, "match": match, "delay": delays.copy()}
        return _to_batch(inputs, targets, mask, autonomous, info)

    @staticmethod
    def evaluate(y: torch.Tensor, batch: Batch) -> dict:
        go = batch.inputs[..., -1] > 0  # response window
        resp = (y[..., 0] * go).sum(1) / go.sum(1)
        want = torch.as_tensor(batch.info["match"], dtype=torch.float32) * 2 - 1
        m = (batch.mask > 0).float()
        mse = (((y - batch.targets) ** 2) * m).sum() / m.sum()
        return {"accuracy": (torch.sign(resp) == want).float().mean().item(), "mse": mse.item()}


TASKS = {cls.name: cls for cls in (FlipFlop, SineGeneration, DelayedMatchToSample)}


def make_task(name: str, **kwargs):
    return TASKS[name](**kwargs)
