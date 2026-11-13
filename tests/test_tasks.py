import numpy as np
import torch

from krnn.tasks import DelayedMatchToSample, FlipFlop, SineGeneration


def test_flipflop_targets_hold_last_pulse():
    task = FlipFlop(n_bits=3, T=200, p_pulse=0.05)
    b = task.sample(16, np.random.default_rng(0))
    x, y = b.inputs.numpy(), b.targets.numpy()
    assert x.shape == y.shape == (16, 200, 3)
    assert (x[:, 0] != 0).all()  # every bit set at t=0
    for i in range(16):
        for k in range(3):
            last = None
            for t in range(200):
                if x[i, t, k] != 0:
                    last = x[i, t, k]
                assert y[i, t, k] == last
    assert (b.autonomous.numpy() == (x == 0).all(-1)).all()


def test_flipflop_perfect_output_scores_one():
    task = FlipFlop()
    b = task.sample(8, np.random.default_rng(1))
    assert task.evaluate(b.targets, b)["accuracy"] == 1.0


def test_sine_input_and_target():
    task = SineGeneration(T=50)
    b = task.sample(4, np.random.default_rng(0), omegas=[0.1, 0.2, 0.3, 0.6])
    assert b.inputs.shape == (4, 50, 1)
    assert torch.allclose(b.inputs[:, 0, 0], torch.tensor([0.25, 0.45, 0.65, 1.25]))
    assert torch.allclose(b.targets[2, :, 0], torch.sin(0.3 * torch.arange(50.0)), atol=1e-6)
    assert b.autonomous.all()
    assert task.evaluate(b.targets, b)["nmse"] == 0.0


def test_dms_layout():
    task = DelayedMatchToSample(n_stim=4, delay=(20, 40))
    b = task.sample(64, np.random.default_rng(0))
    x, y, m = b.inputs.numpy(), b.targets.numpy(), b.mask.numpy()
    info = b.info
    for i in range(64):
        assert x[i, task.t_fix, info["sample"][i]] == 1.0
        assert b.autonomous[i].sum().item() == info["delay"][i]
        assert (x[i][b.autonomous[i].numpy()] == 0).all()
        go = x[i, :, -1] > 0
        assert go.sum() == task.t_resp
        assert (y[i, go, 0] == (1.0 if info["match"][i] else -1.0)).all()
        assert (y[i, ~go, 0] == 0).all()
        assert m[i, -1, 0] == 0 or info["delay"][i] == task.delay[1]
    assert (info["match"] == (info["sample"] == info["test"])).all()
    assert task.evaluate(b.targets, b)["accuracy"] == 1.0


def test_cell_overrides_in_config(tmp_path):
    from krnn.train import load_config

    p = tmp_path / "c.yaml"
    p.write_text("task: dms\nlr: 0.001\ncell_overrides:\n  gru: {lr: 0.01}\n")
    assert load_config(p, cell="gru").lr == 0.01
    assert load_config(p, cell="tanh").lr == 0.001
    assert load_config(p, cell="gru", lr=0.5).lr == 0.5
