"""
SIGReg is the only thing preventing collapse (no EMA target, no stop-grad), so
these are the evals that say whether the run has a safety net at all.

Every threshold here is measured, not guessed — see the table in
src/algo/losses.py.
"""
import math

import pytest
import torch

from algo.config import CFG
from algo.losses import (AuxHeads, aux_heads, aux_loss, jepa_loss, sigreg,
                            sigreg_null, sigreg_ratio)

D = 192


def _g(n, d=D, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def test_sigreg_null_matches_closed_form():
    """E[T] = sqrt(pi)(1-1/sqrt2)/n. If this drifts, the quadrature is wrong."""
    assert sigreg_null(1) == pytest.approx(math.sqrt(math.pi) * (1 - 1 / math.sqrt(2)), rel=1e-9)
    for n in (128, 384, 1024):
        got = sum(float(sigreg(_g(n, seed=s))) for s in range(6)) / 6
        assert got == pytest.approx(sigreg_null(n), rel=0.15), f"n={n}: {got}"


def test_sigreg_null_scales_as_one_over_n():
    """The floor is batch-size dependent; raw sigreg across batch sizes is not comparable."""
    a = sum(float(sigreg(_g(128, seed=s))) for s in range(6)) / 6
    b = sum(float(sigreg(_g(1024, seed=s))) for s in range(6)) / 6
    assert a / b == pytest.approx(8.0, rel=0.25)


@pytest.mark.parametrize("name,factor,build", [
    ("all-identical", 100, lambda: _g(1).repeat(384, 1)),
    ("rank-8", 50, lambda: _g(384, 8) @ _g(8, D, seed=1)),
    ("scale x3", 50, lambda: _g(384) * 3),
    ("heavy radial", 5, lambda: torch.nn.functional.normalize(_g(384), dim=1)
     * torch.distributions.StudentT(3.0).sample((384, 1)).abs() * 8.0),
])
def test_sigreg_flags_collapse_modes(name, factor, build):
    """Each mode must sit at least `factor` x above the null it would be confused with."""
    null = sigreg_null(384)
    assert float(sigreg(build())) > factor * null, name


def test_sigreg_blind_to_projection_invisible_structure():
    """
    Documented blind spot, asserted so nobody later claims SIGReg certifies
    Gaussianity: random 1-D projections of a 192-dim shell are Gaussian, so the
    statistic cannot see the shell.
    """
    shell = torch.nn.functional.normalize(_g(384), dim=1) * math.sqrt(D)
    assert float(sigreg(shell)) < 3 * sigreg_null(384)


def test_sigreg_escapes_rank_collapse_through_bn():
    """
    THE experiment that says Stage 2's design is sound.

    Optimising a collapsed latent as a free tensor barely moves it — once BN has
    made the marginals unit-variance the statistic is nearly flat in z. What
    works is optimising the MAP that produces z, with the gradient flowing
    through the BatchNorm, which is exactly how model.Encoder is wired.
    """
    import torch.nn as nn
    from algo.aq_watch import effective_dim_corrected

    torch.manual_seed(0)
    n = 384
    h = torch.randn(n, 64)
    W = nn.Linear(64, D, bias=False)
    with torch.no_grad():                      # start it rank-8 on purpose
        U, S, V = torch.linalg.svd(W.weight, full_matrices=False)
        S[8:] = 0
        W.weight.copy_((U * S) @ V)
    bn = nn.BatchNorm1d(D, affine=False)

    def z():
        return bn(W(h))

    start_ed = effective_dim_corrected(z())
    assert start_ed < 10, start_ed
    assert float(sigreg(z())) > 5 * sigreg_null(n)      # 10x null after BN, not 254x

    opt = torch.optim.Adam(W.parameters(), lr=1e-3)
    for _ in range(400):
        opt.zero_grad()
        sigreg(z()).backward()
        opt.step()

    assert effective_dim_corrected(z()) > 4 * start_ed
    assert float(sigreg(z())) < 3 * sigreg_null(n)


def test_sigreg_gradient_is_small_in_absolute_terms():
    """
    Documented so nobody reads a flat sigreg curve as 'the term is broken'. The
    gradient norm is ~1e-3 for a 384x192 batch across every scale, which is why
    sigreg_lambda spans two orders of magnitude in the sweep.
    """
    z = (_g(384) * 2).requires_grad_(True)
    sigreg(z).backward()
    assert 1e-5 < float(z.grad.norm()) < 1e-1


def test_sigreg_ratio_is_batch_size_invariant():
    r = [sigreg_ratio(sigreg(_g(n, seed=1)), n) for n in (256, 1024)]
    assert all(0.5 < x < 2.0 for x in r), r


def test_aux_head_pos_weight_matches_measured_base_rate():
    """dies_in_5 is true on 1.04% of observations; without pos_weight the BCE
    optimum is 'always alive'."""
    h = AuxHeads(CFG)
    assert float(h.pos_weight) == pytest.approx(1 / 0.0104 - 1, rel=1e-6)
    assert float(h.pos_weight) > 90


def test_pure_variant_drops_aux_terms_but_keeps_graph():
    cfg = CFG.__class__()
    cfg.loss.variant = "pure"
    z = _g(64).requires_grad_(True)
    batch = {"world_y": torch.zeros(64), "scroll": torch.zeros(64),
             "dies_in_5": torch.zeros(64)}
    terms = aux_loss(aux_heads(cfg), z, batch, cfg=cfg)
    assert all(float(v) == 0.0 for v in terms.values())
    sum(terms.values()).backward()      # must not detach the graph


def test_jepa_loss_reports_every_term_separately():
    """The two failure signatures are invisible in the total; the split is the diagnosis."""
    out = {"z": _g(64).unsqueeze(0), "zhat": _g(8), "z_target": _g(8, seed=2)}
    out = {"z": _g(64), "zhat": _g(8), "z_target": _g(8, seed=2)}
    heads = aux_heads(CFG)
    b = {"world_y": torch.zeros(64), "scroll": torch.zeros(64), "dies_in_5": torch.zeros(64)}
    t = jepa_loss(out, heads, b, cfg=CFG)
    for k in ("pred_loss", "sigreg_loss", "sigreg_ratio", "aux_y", "aux_scroll",
              "aux_dead", "loss"):
        assert k in t
    # sigreg_ratio is a readout, NOT part of the objective
    expect = t["pred_loss"] + CFG.loss.sigreg_lambda * t["sigreg_loss"] \
        + t["aux_y"] + t["aux_scroll"] + t["aux_dead"]
    assert float(t["loss"]) == pytest.approx(float(expect), rel=1e-5)


def test_aux_scale_multiplies_every_aux_term():
    """
    aux_scale exists because the defaults are ~100x too weak to matter:
    collapsing is worth ~0.80 of pred_loss and the aux terms at scale 1 cost at
    most 0.008 in total.
    """
    from algo.config import Config

    z = _g(64)
    batch = {"world_y": torch.ones(64), "scroll": torch.ones(64),
             "dies_in_5": torch.zeros(64)}
    base, scaled = Config(), Config()
    scaled.loss.aux_scale = 20.0
    heads = aux_heads(base)
    torch.manual_seed(0)
    a = aux_loss(heads, z, batch, cfg=base)
    b = aux_loss(heads, z, batch, cfg=scaled)
    for k in a:
        assert float(b[k]) == pytest.approx(20.0 * float(a[k]), rel=1e-5), k


def test_aux_cost_is_dwarfed_by_the_collapse_incentive_at_scale_1():
    """The arithmetic that says aux_scale had to exist. Pinned so it stays true."""
    from algo.config import CFG as C
    worst = (C.loss.aux_y_lambda * 0.36 ** 2 + C.loss.aux_scroll_lambda * 0.15 ** 2)
    assert worst < 0.01                       # vs ~0.80 of pred_loss for collapsing


def test_raw_mse_targets_are_marked_not_comparable_to_lemario():
    """
    LeMario's mse_5step 0.07772 with gain_5step 0.455 pins their latent variance
    at 0.1426; ours is exactly 1.0 by BN. Equal predictive quality shows up as a
    7x larger MSE for us, so the raw-MSE rows cannot be compared and say so.
    """
    from algo.config import TARGETS
    lm = TARGETS["mse_5step"]["lemario"]
    their_var = lm / (1 - TARGETS["gain_5step"]["lemario"])
    assert their_var == pytest.approx(0.1426, abs=1e-3)
    assert TARGETS["mse_5step"]["comparable"] is False
    assert TARGETS["mse_1step"]["comparable"] is False
    # gain and R2 are scale-invariant and therefore do compare
    for k in ("gain_5step", "y_probe_r2", "x_probe_r2"):
        assert TARGETS[k]["comparable"] is True
    # and our mse target is the gain target expressed on OUR scale
    assert TARGETS["mse_5step"]["target"] == pytest.approx(
        1 - TARGETS["gain_5step"]["target"], abs=1e-9)
