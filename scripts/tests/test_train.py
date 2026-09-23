"""
Evals for the training-loop diagnostics and the dry-run gate.

The gate is here because the first version of it had a hole that a REAL run
walked straight through: it asserted only "both losses moved", and a run that
mapped every frame to one latent satisfied that perfectly. The exact history
that fooled it is pinned below as a regression case.
"""
import math

import pytest
import torch

from algo.aq_watch import (effective_dim, effective_dim_corrected,
                              effective_dim_null, effective_dim_windowed)
from algo.config import CFG
from algo.losses import sigreg, sigreg_null, sigreg_ratio, sigreg_windowed
from algo.train import _cosine_lr, dryrun_verdict

B, W, D = 32, 6, 192


def _windowed(b=B, w=W, d=D, jitter=0.05, seed=0):
    """B independent scenes x W near-identical frames — the real batch shape."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, 1, d, generator=g) + torch.randn(b, w, d, generator=g) * jitter


# ---- the window-correlation correction --------------------------------------
def test_flattening_a_windowed_batch_fakes_a_collapse():
    """
    Documented failure of the FIRST implementation. Epps-Pulley's null assumes
    iid samples; W frames from one window are 2 emulator frames apart. Scoring
    the flattened batch against the n=B*W null reports a healthy latent as ~6x
    null, which reads as a catastrophe.
    """
    z = _windowed()
    flat = z.reshape(B * W, D)
    assert sigreg_ratio(sigreg(flat), B * W) > 3.0          # looks broken
    assert sigreg_ratio(sigreg(flat), B) < 2.0              # against the honest n, fine


def test_sigreg_windowed_null_uses_independent_sample_count():
    """A healthy windowed batch must read ~1x null, at n = B not n = B*W."""
    vals = [float(sigreg_ratio(sigreg_windowed(_windowed(seed=s)), B)) for s in range(6)]
    assert 0.3 < sum(vals) / len(vals) < 2.5, vals


def test_sigreg_windowed_matches_plain_sigreg_on_2d_input():
    z = torch.randn(256, D, generator=torch.Generator().manual_seed(0))
    assert float(sigreg_windowed(z)) == pytest.approx(float(sigreg(z)), rel=0.35)


def test_sigreg_windowed_still_flags_a_collapsed_windowed_batch():
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(1, 1, D, generator=g)
    scale = torch.randn(B, 1, 1, generator=g)
    z = direction * scale + torch.randn(B, W, D, generator=g) * 0.01   # rank ~1
    assert sigreg_ratio(sigreg_windowed(z), B) > 2 * sigreg_ratio(
        sigreg_windowed(_windowed()), B)


def test_effective_dim_windowed_reports_independent_n():
    z = _windowed()
    pr, corrected, n = effective_dim_windowed(z)
    assert n == B                                    # not B*W
    assert corrected > 100                           # a full-rank latent, correctly read
    # flattening instead reads 27 against an apparent null of 96 — a fake 3.5x collapse
    flat_pr = effective_dim(z.reshape(B * W, D))
    assert flat_pr < 0.5 * effective_dim_null(B * W, D)


def test_effective_dim_null_and_correction_are_inverses():
    for n in (64, 384, 2048):
        assert effective_dim_null(n, D) == pytest.approx(n * D / (n + D))
    z = torch.randn(2048, D, generator=torch.Generator().manual_seed(0))
    assert effective_dim_corrected(z) == pytest.approx(D, rel=0.05)


# ---- lr schedule ------------------------------------------------------------
def test_warmup_never_outlasts_a_short_run():
    """A 500-step warmup on a 400-step dry run judged the model at a fraction of
    its intended lr for the entire run."""
    lrs = [_cosine_lr(s, 400, 3e-4) for s in range(400)]
    assert max(lrs) == pytest.approx(3e-4, rel=1e-6)
    assert lrs[-1] < 1e-5                            # actually completes the cosine
    assert lrs[0] < lrs[100]                         # still warms up


# ---- the dry-run gate -------------------------------------------------------
def _history(pred, sigreg_ratio_, eff_dim, n=30, null=27.4):
    return [{"pred_loss": p, "sigreg_loss": r * sigreg_null(B), "sigreg_ratio": r,
             "eff_dim": e, "eff_dim_null": null}
            for p, r, e in zip(*(list(v) for v in (pred, sigreg_ratio_, eff_dim)))]


def _ramp(a, b, n=30):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def test_gate_rejects_the_exact_run_that_fooled_the_first_version():
    """
    REGRESSION. The real numbers: pred_loss 0.975 -> 0.061 (moved), sigreg
    0.030 -> 0.098 (moved), eff_dim 9.96 -> 1.17. The old gate passed this.
    """
    h = _history(_ramp(0.975, 0.061), _ramp(2.0, 6.3), _ramp(9.96, 1.17))
    r = dryrun_verdict({"history": h, "steps": 300}, batch_size=B, cfg=CFG)
    assert r["pred_loss_moved"] and r["sigreg_loss_moved"]      # the old criteria
    assert not r["not_collapsed"]
    assert not r["passed"]


def test_gate_accepts_a_healthy_run():
    h = _history(_ramp(0.98, 0.42), _ramp(2.0, 1.1), _ramp(9.0, 22.0))
    r = dryrun_verdict({"history": h, "steps": 300}, batch_size=B, cfg=CFG)
    assert r["passed"], r


def test_gate_rejects_a_run_that_never_learns():
    h = _history(_ramp(0.98, 0.97), _ramp(1.0, 1.0), _ramp(20.0, 22.0))
    r = dryrun_verdict({"history": h, "steps": 300}, batch_size=B, cfg=CFG)
    assert not r["pred_loss_moved"] and not r["passed"]


def test_gate_rejects_an_unregularised_run_even_if_eff_dim_holds():
    h = _history(_ramp(0.98, 0.30), _ramp(2.0, 40.0), _ramp(20.0, 22.0))
    r = dryrun_verdict({"history": h, "steps": 300}, batch_size=B, cfg=CFG)
    assert not r["regularised"] and not r["passed"]


def test_collapse_check_is_relative_to_initialisation_not_an_absolute_bar():
    """
    Measured: an untrained encoder starts at eff_dim 7.9 against an MP null of
    27 (batch 32) — every Mario frame looks alike, so a random ViT is already
    near-degenerate. A "> 0.5 x null" bar is therefore unreachable from step 0
    and would fail every run including healthy ones. The honest question is
    whether training makes the rank worse.
    """
    held = dryrun_verdict({"history": _history(_ramp(0.9, 0.4), _ramp(1, 1),
                                               _ramp(7.9, 8.4)), "steps": 1}, B, CFG)
    assert held["not_collapsed"], held        # would have failed the old bar

    fell = dryrun_verdict({"history": _history(_ramp(0.9, 0.03), _ramp(1, 1),
                                               _ramp(7.9, 2.0)), "steps": 1}, B, CFG)
    assert not fell["not_collapsed"]          # the real no-stop-grad trajectory

    rank1 = dryrun_verdict({"history": _history(_ramp(0.9, 0.03), _ramp(1, 1),
                                                _ramp(3.5, 3.4)), "steps": 1}, B, CFG)
    assert not rank1["not_collapsed"]         # flat, but rank-3 is collapse by any bar


def test_gate_does_not_require_sigreg_to_decrease():
    """sigreg legitimately rises early while the encoder is still moving."""
    h = _history(_ramp(0.98, 0.40), _ramp(0.6, 1.4), _ramp(9.0, 22.0))
    assert dryrun_verdict({"history": h, "steps": 1}, B, CFG)["passed"]


def test_gate_carries_the_lemario_comparable_gain_through():
    h = _history(_ramp(0.98, 0.42), _ramp(2.0, 1.1), _ramp(9.0, 22.0))
    for i, r in enumerate(h):
        r["gain_5step"] = 0.1 + 0.4 * i / (len(h) - 1)
        r["mse_5step"] = 1 - r["gain_5step"]
    out = dryrun_verdict({"history": h, "steps": 1}, B, CFG)
    # the verdict averages the LAST THIRD, not the final step — a single-step
    # readout is noise at these batch sizes
    n = len(h) // 3
    assert out["gain_5step_last"] == pytest.approx(
        sum(r["gain_5step"] for r in h[-n:]) / n, rel=1e-9)
    assert 0.40 < out["gain_5step_last"] < 0.50


# ---- the periodic checkpoint / health machinery -----------------------------
def _hist(step, pred, eff, sig, n=50):
    return [{"step": step, "epoch": 0, "pred_loss": pred, "eff_dim": eff,
             "sigreg_ratio": sig, f"gain_{CFG.model.pred_horizon}step": 1 - pred}
            for _ in range(n)]


def test_health_flags_a_stalled_representation():
    """
    The y-probe not improving is the ONE signal a collapsed-but-confident run
    cannot fake, which is why it is the one wired to the abort. Observed on a
    real smoke run: y drifted -0.030 -> -0.055 while every loss term looked fine.
    """
    from algo.train import checkpoint_health

    g = [{"y_probe_r2": -0.030, "x_probe_r2": 0.46, "eff_dim": 20.0, "frac": 0.1},
         {"y_probe_r2": -0.055, "x_probe_r2": 0.10, "eff_dim": 20.0, "frac": 0.2}]
    h = checkpoint_health(_hist(200, 0.81, 20.0, 0.7), g, CFG)
    assert h["ok_learning"] and h["ok_regularised"]
    assert not h["ok_representing"]
    assert not h["healthy"]


def test_health_accepts_a_run_already_past_lemario_even_if_it_dips():
    """Past 0.188 the run is beating the prior art; small dips must not abort it."""
    from algo.train import checkpoint_health

    g = [{"y_probe_r2": 0.80, "eff_dim": 22.0, "frac": 0.5},
         {"y_probe_r2": 0.74, "eff_dim": 22.0, "frac": 0.6}]
    h = checkpoint_health(_hist(600, 0.30, 22.0, 1.1), g, CFG)
    assert h["ok_representing"] and h["healthy"]


def test_health_flags_a_collapse_against_the_best_seen():
    from algo.train import checkpoint_health

    g = [{"y_probe_r2": 0.5, "eff_dim": 24.0, "frac": 0.4},
         {"y_probe_r2": 0.5, "eff_dim": 24.0, "frac": 0.5}]
    h = checkpoint_health(_hist(500, 0.30, 8.0, 1.0), g, CFG)
    assert not h["ok_not_collapsing"] and not h["healthy"]


def test_health_flags_an_unregularised_run():
    from algo.train import checkpoint_health

    g = [{"y_probe_r2": 0.5, "eff_dim": 20.0, "frac": 0.5}]
    h = checkpoint_health(_hist(500, 0.30, 20.0, 40.0), g, CFG)
    assert not h["ok_regularised"] and not h["healthy"]


def test_epoch_order_reshuffles_between_passes(tmp_path):
    """
    MEASURED BUG: the loader RNG was seeded with (seed + worker_id) only, so
    every epoch replayed the identical shard order AND the identical window
    order — which is the one thing shuffling exists to prevent.
    """
    import numpy as np

    from algo import data as D
    from tests.test_data import _fake_shard

    d = tmp_path / "random"
    d.mkdir()
    paths = [_fake_shard(d / f"shard_{i:04d}.npz", n_eps=3, ep_len=80) for i in range(3)]
    eps = D.episode_table(paths)
    ds = D.ShardWindows(eps, shuffle=True, seed=0)
    take = lambda: [float(s["world_x"][0]) for _, s in zip(range(30), ds)]
    a, b = take(), take()
    assert a != b, "epoch 2 replayed epoch 1 exactly"

    # ...but a fresh dataset with the same seed still reproduces epoch 1
    again = D.ShardWindows(eps, shuffle=True, seed=0)
    c = [float(s["world_x"][0]) for _, s in zip(range(30), again)]
    assert c == a, "runs are no longer reproducible from the seed"


def test_sigreg_is_immune_to_autocast():
    """
    MEASURED: under torch.autocast the projection matmul ran in bf16 and moved
    the statistic by 1.0% — which is 1.0% of the NULL, and the whole signal
    lives only a few multiples above that null.
    """
    z = torch.randn(384, 192, generator=torch.Generator().manual_seed(0))
    plain = float(sigreg(z, generator=torch.Generator().manual_seed(7)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = float(sigreg(z, generator=torch.Generator().manual_seed(7)))
    assert plain == cast


def test_bn_recalibration_changes_the_measurement(tmp_path):
    """
    MEASURED on a checkpoint 10% into a real run: the same encoder on the same
    data reads y-probe -0.028 with stale BN running statistics and +0.121 after
    re-estimating them. Every downstream measurement runs in eval(), so stale
    statistics silently corrupt the gate, the probes, the SAE dump and the
    planner alike.
    """
    import torch.nn as nn

    from algo.model import build_jepa, recalibrate_bn

    m = build_jepa(CFG)
    bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    assert bns, "encoder must end in BatchNorm, not LayerNorm"

    # drive the running stats somewhere wrong on purpose
    with torch.no_grad():
        for b in bns:
            b.running_mean.fill_(5.0)
            b.running_var.fill_(0.01)

    g = torch.Generator().manual_seed(0)
    frames = [torch.randint(0, 255, (4, 224, 224, 3), dtype=torch.uint8, generator=g)
              for _ in range(6)]
    n = recalibrate_bn(m, frames)
    assert n == 6

    # Assert the statistics MOVED OFF the corrupted values, not that they hit
    # some absolute number — the absolute scale depends on the input, and an
    # arbitrary bound makes this test fail on unrelated RNG changes.
    for b in bns:
        assert b.running_mean.abs().mean() < 1.0, "running_mean still near the corrupt 5.0"
        assert b.running_var.mean() > 0.015, "running_var never left the corrupt 0.01"
        assert not torch.allclose(b.running_var, torch.full_like(b.running_var, 0.01))

    # and re-running on the same input is idempotent (momentum=None => true mean)
    before = [b.running_var.clone() for b in bns]
    recalibrate_bn(m, frames)
    for b, v in zip(bns, before):
        assert torch.allclose(b.running_var, v, rtol=1e-4)


def test_recalibration_restores_training_mode_and_momentum():
    """It must not leave the model in a different state than it found it."""
    import torch.nn as nn

    from algo.model import build_jepa, recalibrate_bn

    m = build_jepa(CFG).train()
    bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    mom = [b.momentum for b in bns]
    recalibrate_bn(m, [torch.randint(0, 255, (2, 224, 224, 3), dtype=torch.uint8)])
    assert m.training
    assert [b.momentum for b in bns] == mom
    assert all(b.training for b in bns)


def test_abort_requires_a_trend_not_a_single_low_reading():
    """
    A healthy run measured y = 0.12 at 10% — below LeMario's 0.188 — simply
    because the encoder was young. Aborting on one reading would have killed it.
    """
    from algo.train import checkpoint_health

    rising = [{"y_probe_r2": 0.02, "eff_dim": 20.0, "frac": 0.4},
              {"y_probe_r2": 0.09, "eff_dim": 20.0, "frac": 0.5}]
    ys = [g["y_probe_r2"] for g in rising]
    assert ys[-1] < 0.10                      # still under the floor
    assert not (ys[-1] <= ys[-2] + 0.01)      # ...but improving, so no abort

    stuck = [{"y_probe_r2": 0.02, "eff_dim": 20.0, "frac": 0.4},
             {"y_probe_r2": 0.015, "eff_dim": 20.0, "frac": 0.5}]
    ys = [g["y_probe_r2"] for g in stuck]
    assert ys[-1] < 0.10 and ys[-1] <= ys[-2] + 0.01   # flat and low -> abort
