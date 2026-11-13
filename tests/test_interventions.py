import numpy as np
import torch

from krnn.interventions import (
    DMDLatent,
    eigen_coords,
    eigenfunction_grad,
    fit_readout_of_coords,
    rotate_mode,
    slow_modes,
)
from krnn.koopman import fit_dmd

THETA = 0.3


def _rotation_plus_decay(rng, n=300):
    """Affine system in 10-D: a pure rotation (|lambda| = 1) and a decaying direction (0.5),
    centred on a fixed point away from the origin."""
    R = np.array([[np.cos(THETA), -np.sin(THETA), 0], [np.sin(THETA), np.cos(THETA), 0], [0, 0, 0.5]])
    Q = np.linalg.qr(rng.standard_normal((10, 3)))[0]
    centre = rng.standard_normal(10)
    z = rng.standard_normal((n, 3))
    X, Y = z @ Q.T + centre, (z @ R.T) @ Q.T + centre
    return R, Q, centre, X, Y


def test_rotating_the_oscillatory_mode_is_a_time_shift():
    rng = np.random.default_rng(0)
    R, Q, centre, X, Y = _rotation_plus_decay(rng)
    lmap = DMDLatent(fit_dmd(X, Y, rank=3, ridge=0.0))
    modes = slow_modes(lmap.K, 2)
    j = int(np.argmax(np.angle(modes.eigvals)))          # eigenvalue exp(+i theta)
    assert np.isclose(np.angle(modes.eigvals[j]), THETA)

    z = rng.standard_normal((5, 3))
    h = torch.as_tensor(z @ Q.T + centre)
    out = rotate_mode(lmap, modes, h, j, 3 * THETA).numpy()
    R3 = np.linalg.matrix_power(R, 3)
    R3[2, 2] = 1.0                                       # the decaying direction is untouched
    np.testing.assert_allclose(out, (z @ R3.T) @ Q.T + centre, atol=1e-8)


def test_eigenfunction_gradient_of_linear_map_is_constant_left_eigenvector():
    rng = np.random.default_rng(1)
    _, _, _, X, Y = _rotation_plus_decay(rng)
    lmap = DMDLatent(fit_dmd(X, Y, rank=3, ridge=0.0))
    modes = slow_modes(lmap.K, 3)
    h = torch.as_tensor(X[:20])
    targets = rng.standard_normal((20, 1))
    a = fit_readout_of_coords(eigen_coords(lmap, modes, h).real.numpy(), targets)[:, 0]
    f, g = eigenfunction_grad(lmap, modes, a, h)
    expected = lmap.P.numpy() @ (np.real(modes.left).T @ a[:-1])
    np.testing.assert_allclose(g.numpy(), np.broadcast_to(expected, g.shape), atol=1e-10)
    np.testing.assert_allclose(f.numpy(), eigen_coords(lmap, modes, h).real.numpy() @ a[:-1] + a[-1])
