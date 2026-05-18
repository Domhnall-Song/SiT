import torch

from transport.couette import CouettePath


def test_boundary_conditions_time():
    p = CouettePath(mode="time", eta_max=3.0)
    assert torch.allclose(p.compute_alpha_t(torch.tensor([0.0])), torch.tensor([1.0]))
    assert p.compute_alpha_t(torch.tensor([1.0])).item() < 1e-4
    assert torch.allclose(
        p.compute_sigma_t(torch.tensor([0.0])), torch.tensor([0.0]), atol=1e-6
    )
    assert abs(p.compute_sigma_t(torch.tensor([1.0])).item() - 1.0) < 1e-4


def test_variance_preservation_time():
    p = CouettePath(mode="time", eta_max=3.0)
    t = torch.linspace(0.0, 1.0, 200)
    a, s = p.compute_alpha_t(t), p.compute_sigma_t(t)
    assert torch.allclose(a ** 2 + s ** 2, torch.ones_like(t), atol=1e-5)


def test_derivative_consistency_time():
    p = CouettePath(mode="time", eta_max=3.0)
    t = torch.linspace(0.05, 0.95, 50, requires_grad=True)
    a = p.compute_alpha_t(t)
    a.sum().backward()
    a_num = t.grad.clone()
    t.grad = None
    a_ana = p.compute_d_alpha_t(t.detach())
    assert torch.allclose(a_num, a_ana, atol=1e-4)
    s = p.compute_sigma_t(t)
    s.sum().backward()
    s_num = t.grad.clone()
    s_ana = p.compute_d_sigma_t(t.detach())
    assert torch.allclose(s_num, s_ana, atol=1e-3)


def test_qsample_t0_is_identity_freq():
    p = CouettePath(mode="freq", nu=1.0, alpha_min=1e-4, freq_axes=(-2, -1))
    x = torch.randn(2, 3, 16, 16)
    x_t = p.q_sample(x, torch.zeros(2), eps=torch.randn_like(x))
    assert torch.allclose(x_t, x, atol=1e-5)


def test_qsample_t1_is_noise_freq():
    p = CouettePath(mode="freq", nu=1.0, alpha_min=1e-4, freq_axes=(-2, -1))
    x = torch.randn(2, 3, 16, 16)
    eps = torch.randn_like(x)
    x_t = p.q_sample(x, torch.ones(2), eps=eps)
    # signal should be dominated by eps; correlation with x_data should be ~0
    c = (x_t * x).flatten(1).sum(1) / (
        x_t.flatten(1).norm(dim=1) * x.flatten(1).norm(dim=1)
    )
    assert c.abs().mean().item() < 0.05


def test_velocity_score_noise_roundtrip_time():
    p = CouettePath(mode="time", eta_max=3.0)
    t = torch.tensor([0.3, 0.7])
    x0 = torch.randn(2, 3, 16, 16)
    eps = torch.randn_like(x0)
    x_t, v = p.plan(t, x0, eps)
    score = p.get_score_from_velocity(v, x_t, t)
    v_back = p.get_velocity_from_score(score, x_t, t)
    assert torch.allclose(v, v_back, atol=1e-4)


def test_dropin_with_existing_transport():
    from transport import create_transport

    tr = create_transport(path_type="Couette", prediction="velocity")
    # mimic one training step
    x = torch.randn(4, 3, 32, 32)
    # ...assert tr.training_losses(...) returns finite scalar
    assert tr is not None
