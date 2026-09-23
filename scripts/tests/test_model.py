"""
Structural evals for the JEPA. Each one pins a property that, if it silently
broke, would produce a model that trains to a plausible loss and is useless:

  * AdaLN-Zero really is the identity at init (otherwise the action is optional)
  * the action really does change the prediction (otherwise it is a video model)
  * the causal mask really holds (otherwise the predictor sees the future it is
    being asked to predict, and the 3-step MSE is a lie)
  * the projection head ends in BN, not LN (an LN here makes SIGReg unreachable)
"""
import pytest
import torch
import torch.nn as nn

from algo.config import CFG
from algo.losses import aux_heads
from algo.model import ActionEncoder, Encoder, JEPA, Predictor, build_jepa

W = CFG.model.history_len + CFG.model.pred_horizon
D = CFG.model.latent_dim


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_jepa(CFG)


def _batch(b=2):
    torch.manual_seed(1)
    return (torch.randint(0, 255, (b, W, 224, 224, 3), dtype=torch.uint8),
            torch.randint(0, 2, (b, W - 1, CFG.data.frame_skip, 6)).float())


def test_shapes(model):
    f, a = _batch()
    out = model(f, a)
    assert out["z"].shape == (2, W, D)
    assert out["zhat"].shape == (2, CFG.model.pred_horizon, D)
    assert out["z_target"].shape == out["zhat"].shape


def test_param_count_reported_honestly(model):
    r = model.param_report()
    assert r["total"] == sum(p.numel() for p in model.parameters())
    # 9.8M with a 6-layer/192-wide predictor. The tracker's band is 10-22M, so
    # this is UNDER by design and the tracker is expected to say so — reaching
    # 15M means raising predictor_layers, which is a deliberate config change.
    assert 9e6 < r["total"] < 11e6, r


def test_projection_head_ends_in_batchnorm_not_layernorm(model):
    """A trailing LayerNorm puts z on a fixed-radius sphere and SIGReg can never
    reach Gaussian. This is the single line most likely to get 'cleaned up'."""
    last = list(model.encoder.proj.children())[-1]
    assert isinstance(last, nn.BatchNorm1d), type(last)
    assert last.affine is False
    assert not any(isinstance(m, nn.LayerNorm) for m in model.encoder.proj.children())


def test_adaln_is_exactly_identity_at_init():
    """Zero-init gates: at step 0 the predictor output must not depend on the
    context latents or the actions at all."""
    torch.manual_seed(0)
    p = Predictor(CFG)
    a1 = torch.randn(4, W, D)
    out1 = p(torch.randn(4, CFG.model.history_len, D), a1)
    out2 = p(torch.randn(4, CFG.model.history_len, D) * 10, torch.randn(4, W, D) * 10)
    assert torch.allclose(out1, out2, atol=1e-5)


def _unzero(p):
    """Break the AdaLN-Zero init so the conditioning path is actually live."""
    for blk in p.blocks:
        nn.init.normal_(blk.ada[1].weight, std=0.05)
        nn.init.normal_(blk.ada[1].bias, std=0.05)


def test_action_changes_the_prediction():
    """If this fails the predictor has become a video model that ignores control,
    which is the degenerate solution that leaves y unencoded."""
    torch.manual_seed(0)
    p = Predictor(CFG)
    _unzero(p)
    z = torch.randn(4, CFG.model.history_len, D)
    a = torch.randn(4, W, D)
    b = a.clone()
    b[:, CFG.model.history_len:] += 3.0
    assert not torch.allclose(p(z, a), p(z, b), atol=1e-4)


def test_predictor_is_causal():
    """
    Prediction for the FIRST future slot must not depend on the actions of later
    slots. Without this the 3-step rollout number is not a rollout number.
    """
    torch.manual_seed(0)
    p = Predictor(CFG)
    _unzero(p)
    p.eval()
    H = CFG.model.history_len
    z = torch.randn(4, H, D)
    a = torch.randn(4, W, D)
    b = a.clone()
    b[:, H + 1:] += 5.0                       # perturb strictly LATER actions only
    with torch.no_grad():
        assert torch.allclose(p(z, a)[:, 0], p(z, b)[:, 0], atol=1e-5)
        assert not torch.allclose(p(z, a)[:, -1], p(z, b)[:, -1], atol=1e-4)


def test_null_action_is_learned_not_zero(model):
    """Token 0 has no preceding action; a hard zero would be indistinguishable
    from 'no buttons pressed', which is a real and different action."""
    codes = model.action_codes(torch.zeros(2, W - 1, CFG.data.frame_skip, 6))
    assert not torch.allclose(codes[:, 0], codes[:, 1])
    assert isinstance(model.action_encoder.null, nn.Parameter)


def test_no_stop_grad_by_default(model):
    """SIGReg replaces the EMA/stop-grad machinery; a silent stop-grad here would
    hide a collapse that SIGReg is supposed to be preventing."""
    assert model.stop_grad_target is False
    f, a = _batch()
    out = model(f, a)
    assert out["z_target"].requires_grad

    sg = build_jepa(CFG, stop_grad_target=True)
    assert not sg(f, a)["z_target"].requires_grad


def test_action_encoder_sees_every_skipped_frame():
    """A 2-frame tap and a 2-frame hold must produce different codes — jump
    duration is what determines the y trajectory."""
    torch.manual_seed(0)
    ae = ActionEncoder(CFG)
    tap = torch.zeros(1, 1, CFG.data.frame_skip, 6)
    tap[0, 0, 0, 4] = 1                        # A pressed on the first frame only
    hold = torch.zeros_like(tap)
    hold[0, 0, :, 4] = 1                       # A held through both
    assert not torch.allclose(ae(tap), ae(hold), atol=1e-5)


def test_end_to_end_backward_touches_every_component(model):
    """
    AdaLN-Zero has a ONE-STEP BLIND SPOT and it is worth knowing about: with the
    modulation weights at exactly zero, d(loss)/d(action_code) is also exactly
    zero, so the action encoder receives NO gradient on step 0. The AdaLN weights
    themselves do get gradient (the gate multiplies a non-zero branch output), so
    the path unsticks on step 1. If it did not, the action encoder would be dead
    for the whole run and the model would be a video predictor.
    """
    from algo.losses import jepa_loss

    heads = aux_heads(CFG)
    f, a = _batch()
    b = {"world_y": torch.randn(2, W), "scroll": torch.randn(2, W),
         "dies_in_5": torch.zeros(2, W)}
    opt = torch.optim.SGD(list(model.parameters()) + list(heads.parameters()), lr=0.1)

    def step():
        opt.zero_grad(set_to_none=True)
        jepa_loss(model(f, a), heads, b, cfg=CFG)["loss"].backward()
        got = {n: any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in getattr(model, n).parameters())
               for n in ("encoder", "action_encoder", "predictor")}
        opt.step()
        return got

    first = step()
    assert first["encoder"] and first["predictor"]
    assert not first["action_encoder"], "AdaLN-Zero should block this on step 0"

    second = step()
    assert all(second.values()), f"action path never unstuck: {second}"
