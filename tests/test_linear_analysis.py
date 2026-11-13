import numpy as np
import torch

from krnn.fixed_points import find_fixed_points
from krnn.koopman import fit_dmd, spectral_distance
from krnn.models import RNN
from krnn.rollout import rollout, snapshot_pairs


def _rotation_system(rng, d_latent=4, d_embed=10):
    """Affine linear system with known eigenvalues, embedded in d_embed dims."""
    lam = np.array([0.95 * np.exp(0.3j), 0.95 * np.exp(-0.3j), 0.8 * np.exp(1.1j), 0.8 * np.exp(-1.1j)])
    B = np.zeros((4, 4))
    for k, (r, th) in enumerate([(0.95, 0.3), (0.8, 1.1)]):
        B[2 * k:2 * k + 2, 2 * k:2 * k + 2] = r * np.array([[np.cos(th), -np.sin(th)],
                                                            [np.sin(th), np.cos(th)]])
    Q = np.linalg.qr(rng.standard_normal((d_embed, d_latent)))[0]   # (d_embed, 4)
    b = rng.standard_normal(d_latent)
    offset = rng.standard_normal(d_embed)
    return lam, B, Q, b, offset


def test_dmd_recovers_affine_linear_system():
    rng = np.random.default_rng(0)
    lam, B, Q, b, offset = _rotation_system(rng)
    z = rng.standard_normal((200, 4))
    zn = z @ B.T + b
    X, Y = z @ Q.T + offset, zn @ Q.T + offset
    dmd = fit_dmd(X, Y, rank=4, ridge=0.0)
    assert spectral_distance(dmd.eigvals, lam) < 1e-8
    np.testing.assert_allclose(dmd.step(X), Y, atol=1e-8)
    traj = dmd.rollout(X[:3], 5)
    zz = z[:3]
    for t in range(1, 6):
        zz = zz @ B.T + b
        np.testing.assert_allclose(traj[:, t], zz @ Q.T + offset, atol=1e-8)


def test_spectral_distance_is_permutation_invariant():
    a = np.array([0.9, 0.5 + 0.1j, 0.5 - 0.1j])
    assert spectral_distance(a, a[::-1]) == 0.0
    assert np.isclose(spectral_distance(a, a + 0.1), 0.1)


def _diag_tanh_rnn(w, bias, d):
    m = RNN("tanh", n_in=1, n_hidden=d, n_out=1).double()
    with torch.no_grad():
        m.core.weight_hh_l0.copy_(w * torch.eye(d))
        m.core.weight_ih_l0.zero_()
        m.core.bias_ih_l0.zero_()
        m.core.bias_hh_l0.copy_(torch.as_tensor(bias))
    return m


def test_fixed_point_of_contracting_map():
    d = 5
    bias = np.linspace(-1, 1, d)
    m = _diag_tanh_rnn(0.5, bias, d)
    h_star = np.zeros(d)
    for _ in range(200):  # scalar fixed-point iteration, converges since slope < 1
        h_star = np.tanh(0.5 * h_star + bias)
    fps = find_fixed_points(m, torch.zeros(1), torch.randn(50, d, dtype=torch.float64))
    assert len(fps) == 1
    np.testing.assert_allclose(fps.h[0], h_star, atol=1e-8)
    expected = np.sort(0.5 * (1 - h_star**2))[::-1]
    np.testing.assert_allclose(np.abs(fps.eigvals[0]), expected, atol=1e-8)
    assert fps.stable.all()


def test_fixed_points_of_multistable_map():
    # h -> tanh(2h) per unit: fixed points at {-a, 0, a}^2 with a = tanh(2a); 4 are stable.
    a = 1.0
    for _ in range(200):
        a = np.tanh(2 * a)
    m = _diag_tanh_rnn(2.0, np.zeros(2), 2)
    h0 = torch.as_tensor(np.random.default_rng(0).uniform(-1.2, 1.2, (400, 2)))
    fps = find_fixed_points(m, torch.zeros(1), h0)
    assert fps.stable.sum() == 4
    assert len(fps) <= 9
    for h in fps.h:
        assert np.all(np.min(np.abs(np.abs(h)[:, None] - np.array([0.0, a])[None]), axis=1) < 1e-8)


def test_dmd_on_rnn_rollouts_matches_jacobian_near_fixed_point():
    """Small perturbations around a stable fixed point: DMD eigenvalues -> Jacobian eigenvalues."""
    d = 5
    m = _diag_tanh_rnn(0.5, np.linspace(-1, 1, d), d)
    fps = find_fixed_points(m, torch.zeros(1), torch.randn(20, d, dtype=torch.float64))
    rng = np.random.default_rng(0)
    h0 = torch.as_tensor(fps.h[0] + 1e-4 * rng.standard_normal((100, d)))
    X, Y = snapshot_pairs(rollout(m, torch.zeros(1), h0, 5))
    dmd = fit_dmd(X, Y, ridge=0.0)
    assert spectral_distance(dmd.eigvals, fps.eigvals[0]) < 1e-3
