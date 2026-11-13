import pytest
import torch

from krnn.models import RNN


@pytest.mark.parametrize("cell", ["tanh", "gru"])
def test_step_matches_forward(cell):
    torch.manual_seed(0)
    model = RNN(cell, n_in=3, n_hidden=16, n_out=2)
    x = torch.randn(5, 20, 3)
    _, h_seq = model(x)
    h = torch.zeros(5, 16)
    for t in range(20):
        h = model.step(h, x[:, t])
        torch.testing.assert_close(h, h_seq[:, t], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cell", ["tanh", "gru"])
def test_jacobian_matches_finite_differences(cell):
    torch.manual_seed(0)
    model = RNN(cell, n_in=2, n_hidden=8, n_out=1).double()
    h = torch.randn(3, 8, dtype=torch.float64) * 0.5
    x = torch.randn(2, dtype=torch.float64)
    J = model.jacobian(h, x)
    assert J.shape == (3, 8, 8)
    eps = 1e-6
    for i in range(8):
        e = torch.zeros(8, dtype=torch.float64)
        e[i] = eps
        fd = (model.step(h + e, x) - model.step(h - e, x)) / (2 * eps)
        torch.testing.assert_close(J[:, :, i], fd, rtol=1e-6, atol=1e-8)
