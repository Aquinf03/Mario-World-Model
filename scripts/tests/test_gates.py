"""
Evals for the gates. A gate that cannot fail is not a gate, so most of these
construct a latent with a KNOWN defect and assert the gate catches it.

The untrained-checkpoint numbers below are measured on real shards and are the
baseline every trained result must be read against — notably x-R2 0.40 from a
randomly initialised ViT, which is why the x threshold is 0.95 and not 0.5.
"""
import numpy as np
import pytest
import torch

from algo.config import CFG
from algo.gates import (average_precision, aliasing_gate, dead_in_5_gate,
                           probe_gate, r2_score, roc_auc)


def _cache(n=800, d=192, seed=0, signal=1.0, n_eps=4):
    """Synthetic latent cache with a controllable amount of real signal in it."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n)).astype(np.float32)
    z = torch.randn(n, d, generator=g)
    z[:, 0] = torch.from_numpy(x) * signal * 10          # x is linearly readable
    z[:, 1] = torch.from_numpy(np.sin(x * 9)) * signal * 10
    return {"z": z, "world_x": x, "world_y": np.sin(x * 9).astype(np.float32),
            "scroll": (x * 0.99).astype(np.float32),
            "dies_in_5": (rng.random(n) < 0.01).astype(np.float32),
            "ep_id": np.repeat(np.arange(n_eps), n // n_eps)[:n]}


# ---- metric primitives ------------------------------------------------------
def test_r2_is_against_heldout_variance_and_can_go_negative():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert r2_score(torch.full_like(y, y.mean()), y) == pytest.approx(0.0)
    assert r2_score(torch.full_like(y, 100.0), y) < -100      # not clamped


def test_roc_auc_matches_hand_computed_cases():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert roc_auc([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert roc_auc([4, 3, 2, 1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert roc_auc([1, 1, 1, 1], [0, 0, 1, 1]) == pytest.approx(0.5)   # all ties


def test_average_precision_baseline_is_the_base_rate():
    """The number that makes dead-in-5 readable at a 1% positive rate."""
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.01).astype(int)
    ap = average_precision(rng.random(20000), y)          # random scores
    assert ap == pytest.approx(y.mean(), abs=0.006)


def test_auc_and_ap_disagree_when_accuracy_would_lie():
    """A ranking that is good but not great: AUC looks fine, AP exposes it."""
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.01).astype(int)
    s = rng.normal(0, 1, 5000) + y * 1.5
    assert roc_auc(s, y) > 0.8
    assert average_precision(s, y) < 0.15


# ---- probe gate -------------------------------------------------------------
def test_probe_gate_recovers_a_readable_signal():
    tr, va = _cache(seed=0), _cache(seed=1)
    r = probe_gate(tr, va, CFG)
    assert r["x_probe_r2"] > 0.9
    assert r["y_probe_r2"] > 0.9


def test_probe_gate_reports_near_zero_when_the_latent_has_nothing():
    """The LeMario failure, synthesised: y absent from z must NOT score 0.8."""
    tr, va = _cache(seed=0, signal=0.0), _cache(seed=1, signal=0.0)
    r = probe_gate(tr, va, CFG)
    assert r["y_probe_r2"] < 0.3, r
    assert r["y_probe_r2"] < CFG.gate.min_y_probe_r2


# ---- aliasing ---------------------------------------------------------------
def test_aliasing_gate_passes_when_far_frames_are_far_apart():
    n = 1200
    rng = np.random.default_rng(0)
    x = np.tile(np.linspace(0, 2000, n // 4), 4).astype(np.float32)
    z = torch.zeros(n, 192)
    z[:, 0] = torch.from_numpy(x / 500.0)      # latent tracks x monotonically
    z += torch.randn(n, 192, generator=torch.Generator().manual_seed(0)) * 0.01
    va = {"z": z, "world_x": np.float32(np.vectorize(
              lambda v: (v - 1600.0) / 1600.0)(x)),
          "ep_id": np.repeat(np.arange(4), n // 4)}
    r = aliasing_gate(va, CFG)
    assert r["aliasing_margin"] > CFG.gate.min_aliasing_margin, r


def test_aliasing_gate_fails_on_a_camera_aliased_latent():
    """
    The bug itself: the latent encodes SCREEN position, so frames 500+ world-px
    apart land on top of each other. x is still partly readable, the gate is not
    fooled.
    """
    n = 1200
    x = np.tile(np.linspace(0, 2000, n // 4), 4).astype(np.float32)
    z = torch.zeros(n, 192)
    z[:, 0] = torch.from_numpy((x % 256) / 128.0)        # screen-relative only
    z += torch.randn(n, 192, generator=torch.Generator().manual_seed(0)) * 0.01
    va = {"z": z, "world_x": (x - 1600.0) / 1600.0,
          "ep_id": np.repeat(np.arange(4), n // 4)}
    r = aliasing_gate(va, CFG)
    assert r["aliasing_margin"] < CFG.gate.min_aliasing_margin, r


def test_aliasing_gate_says_nan_rather_than_guessing_on_thin_data():
    """Random-policy episodes rarely span 500px, so the gate must decline to
    report rather than invent a margin from 3 pairs."""
    va = {"z": torch.randn(60, 192), "world_x": np.linspace(-1, -0.98, 60).astype(np.float32),
          "ep_id": np.zeros(60, dtype=np.int64)}
    r = aliasing_gate(va, CFG)
    assert np.isnan(r["aliasing_margin"])
    assert "PPO" in r["aliasing_note"]


# ---- dead-in-5 --------------------------------------------------------------
def test_dead_in_5_reports_base_rate_alongside_the_score():
    tr, va = _cache(seed=0), _cache(seed=1)
    r = dead_in_5_gate(tr, va, CFG)
    assert 0.0 <= r["dead_in_5_base_rate"] < 0.05
    assert set(r) == {"dead_in_5_auc", "dead_in_5_ap", "dead_in_5_base_rate"}
    assert "dead_in_5_acc" not in r          # accuracy is meaningless at 1% and is not reported


# ---- checkpoint loading -----------------------------------------------------
def test_load_jepa_refuses_a_foreign_checkpoint(tmp_path):
    """
    A silent no-load is indistinguishable from a real collapse: an untrained
    encoder measures x-R2 0.40, y-R2 -0.00, aliasing -0.94 on real shards, which
    reads exactly like a failed run.
    """
    from algo.model import load_jepa
    p = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, p)
    with pytest.raises(ValueError, match="not an aq-mario checkpoint"):
        load_jepa(p, CFG)


def test_load_jepa_is_strict_about_missing_keys(tmp_path):
    from algo.model import build_jepa, load_jepa
    sd = build_jepa(CFG).state_dict()
    sd.pop(next(iter(sd)))
    p = tmp_path / "partial.pt"
    torch.save({"jepa": sd}, p)
    with pytest.raises(RuntimeError):
        load_jepa(p, CFG)
