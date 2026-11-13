import numpy as np

from krnn.koopman import KAEConfig, fit_dmd, fit_kae, spectral_distance
from krnn.rollout import snapshot_pairs

A, B, C = 0.9, 0.5, 1.0


def _slow_manifold(n, T, rng):
    """x' = a x, y' = b y + c x^2: linear in (x, y, x^2), Koopman eigenvalues {a, a^2, b}."""
    x, y = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    out = [np.stack([x, y], 1)]
    for _ in range(T):
        x, y = A * x, B * y + C * x**2
        out.append(np.stack([x, y], 1))
    return np.stack(out, 1)


def _nmse(pred, true):
    var = ((true - true.reshape(-1, true.shape[-1]).mean(0)) ** 2).sum(-1).mean()
    return ((pred - true) ** 2).sum(-1).mean() / var


def test_kae_recovers_koopman_spectrum_where_dmd_cannot():
    train = _slow_manifold(300, 20, np.random.default_rng(0))
    test = _slow_manifold(100, 20, np.random.default_rng(99))
    kae, history = fit_kae(train, KAEConfig(latent_dim=3, width=64, horizon=10, n_steps=2000))
    dmd = fit_dmd(*snapshot_pairs(train))

    assert spectral_distance(kae.eigvals, [A, A * A, B]) < 0.08   # ~0.03-0.05 observed
    assert _nmse(kae.rollout(test[:, 0], 20), test) < 0.01
    assert _nmse(dmd.rollout(test[:, 0], 20), test) > 0.1   # the x^2 term is out of reach
    assert history[-1]["total"] < history[0]["total"]
