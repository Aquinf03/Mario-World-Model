"""
Evals for the Stage 4 planner and the steering hook.

These do not need a trained model — they check the properties that make the
planner *correct* rather than *good*: that it optimises over the macro set and
not raw buttons, that the cost is probe-scored, that a longer plan than the
predictor's horizon is rolled in chunks rather than silently truncated, and that
steering is reversible.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from algo.config import CFG, MACRO_ACTIONS, MACRO_NAMES
from algo.model import build_jepa
from algo.plan import (MACRO_BLOCKS, MacroCEM, PlanConfig, macro_button_block,
                          rollout_cost, steer_predictor, steering_effect)

D_ = CFG.model.latent_dim


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_jepa(CFG).eval()
    # AdaLN-Zero makes the predictor input-independent at init; a planner test
    # against that is vacuous, so light the conditioning path up.
    for blk in m.predictor.blocks:
        nn.init.normal_(blk.ada[1].weight, std=0.05)
        nn.init.normal_(blk.ada[1].bias, std=0.05)
    return m


@pytest.fixture
def probes():
    torch.manual_seed(0)
    return {"world_x": nn.Linear(D_, 1), "dies_in_5": nn.Linear(D_, 1)}


def test_macro_blocks_cover_the_configured_macro_set():
    assert MACRO_BLOCKS.shape == (len(MACRO_NAMES), CFG.data.frame_skip, 6)
    for i, name in enumerate(MACRO_NAMES):
        assert (MACRO_BLOCKS[i] == np.array(MACRO_ACTIONS[name])).all()


def test_macro_block_holds_the_buttons_for_every_skipped_frame():
    """A tap and a hold are different actions; jump duration decides the y arc."""
    b = macro_button_block("run_right_jump")
    assert b.shape == (CFG.data.frame_skip, 6)
    assert (b[0] == b[-1]).all() and b[:, 4].all()      # A held throughout


def test_cost_is_probe_scored_and_rewards_progress(probes):
    """Lower cost for the plan the x-probe says goes further."""
    torch.manual_seed(0)
    z = torch.randn(2, 4, D_)
    with torch.no_grad():                               # make plan 0 read as further
        probes["world_x"].weight.zero_(); probes["world_x"].bias.zero_()
        probes["world_x"].weight[0, 0] = 1.0
        probes["dies_in_5"].weight.zero_(); probes["dies_in_5"].bias.fill_(-20.0)
    z[0, :, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3])     # advancing
    z[1, :, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0])     # stationary
    cost, diag = rollout_cost(z, probes)
    assert cost[0] < cost[1]
    assert diag["progress"][0] > diag["progress"][1]


def test_cost_penalises_predicted_death(probes):
    torch.manual_seed(0)
    z = torch.randn(2, 4, D_)
    z[1] = z[0]
    with torch.no_grad():
        probes["dies_in_5"].weight.zero_(); probes["dies_in_5"].bias.zero_()
        probes["dies_in_5"].weight[0, 1] = 10.0
    z[0, :, 1] = -1.0                                   # safe
    z[1, :, 1] = 1.0                                    # predicted dead
    cost, _ = rollout_cost(z, probes)
    assert cost[1] > cost[0]


def test_goal_residual_is_clamped_to_the_subgoal_span(probes):
    """
    Latent distance is only trustworthy over the span the aliasing gate
    certifies. A goal 3000 px away must not produce 15x the cost of one 200 px
    away — that is asking the latent a question it cannot answer.
    """
    torch.manual_seed(0)
    z = torch.zeros(1, 4, D_)
    p = PlanConfig()
    near, _ = rollout_cost(z, probes, goal_x=1600 + p.subgoal_px, pcfg=p)
    far, _ = rollout_cost(z, probes, goal_x=1600 + 3000, pcfg=p)
    assert float(far) == pytest.approx(float(near), rel=1e-4)


def test_planner_returns_a_macro_sequence_of_the_requested_horizon(model, probes):
    p = PlanConfig(horizon=8, population=32, elites=8, iters=2)
    frames = torch.randint(0, 255, (CFG.model.history_len, 224, 224, 3), dtype=torch.uint8)
    idx, names, diag = MacroCEM(model, probes, p).plan(frames)
    assert len(idx) == 8 and len(names) == 8
    assert all(n in MACRO_NAMES for n in names)
    assert set(diag) == {"final_cost", "progress_px", "risk"}


def test_planner_horizon_may_exceed_the_predictor_horizon(model, probes):
    """
    The predictor's positional embedding is fixed at history+pred_horizon (8), so
    a horizon-12 plan has to be rolled in chunks. Silently truncating to 5 would
    make the planner optimise a shorter plan than it reports.
    """
    p = PlanConfig(horizon=12, population=16, elites=4, iters=1)
    frames = torch.randint(0, 255, (CFG.model.history_len, 224, 224, 3), dtype=torch.uint8)
    idx, names, _ = MacroCEM(model, probes, p).plan(frames)
    assert len(idx) == 12 > CFG.model.pred_horizon


def test_cem_smoothing_keeps_every_macro_reachable(model, probes):
    """
    +1 smoothing on the elite counts. Without it a macro that no elite happened
    to sample gets probability exactly zero and CEM can never propose it again.
    """
    import inspect
    from algo import plan
    src = inspect.getsource(plan.MacroCEM.plan)
    assert "counts + 1.0" in src


# ---- steering ---------------------------------------------------------------
def test_steering_changes_the_rollout_and_is_reversible(model):
    W = CFG.model.history_len + CFG.model.pred_horizon
    frames = torch.randint(0, 255, (1, W, 224, 224, 3), dtype=torch.uint8)
    acts = torch.randint(0, 2, (1, W - 1, CFG.data.frame_skip, 6)).float()
    with torch.no_grad():
        base = model.rollout(frames, acts)

    col = torch.randn(CFG.model.latent_dim)
    h = steer_predictor(model, col, alpha=5.0)
    with torch.no_grad():
        steered = model.rollout(frames, acts)
    h.remove()
    with torch.no_grad():
        restored = model.rollout(frames, acts)

    assert not torch.allclose(base, steered, atol=1e-4)
    assert torch.allclose(base, restored, atol=1e-6), "hook leaked"


def test_steering_effect_sweeps_and_always_removes_its_hooks(model):
    W = CFG.model.history_len + CFG.model.pred_horizon
    frames = torch.randint(0, 255, (1, W, 224, 224, 3), dtype=torch.uint8)
    acts = torch.randint(0, 2, (1, W - 1, CFG.data.frame_skip, 6)).float()
    before = len(model.predictor.blocks[-1]._forward_hooks)
    rows = steering_effect(model, frames, acts, torch.randn(CFG.model.latent_dim),
                           alphas=(-2, 0, 2))
    assert [r["alpha"] for r in rows] == [-2.0, 0.0, 2.0]
    assert len(model.predictor.blocks[-1]._forward_hooks) == before


def test_steering_effect_reports_world_units_when_given_a_probe(model):
    W = CFG.model.history_len + CFG.model.pred_horizon
    frames = torch.randint(0, 255, (1, W, 224, 224, 3), dtype=torch.uint8)
    acts = torch.randint(0, 2, (1, W - 1, CFG.data.frame_skip, 6)).float()
    rows = steering_effect(model, frames, acts, torch.randn(CFG.model.latent_dim),
                           alphas=(-4, 4), probe=nn.Linear(CFG.model.latent_dim, 1))
    assert all("y_px" in r for r in rows)
    assert rows[0]["y_px"] != rows[1]["y_px"]
