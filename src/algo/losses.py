"""
Stage 2 losses.

    L = MSE(zhat, z_target)
      + sigreg_lambda * SIGReg(z)
      + 0.05*MSE(y_head, y) + 0.05*MSE(s_head, scroll) + 0.02*BCE(dead_head, dies_in_5)

SIGReg is the only thing standing between this run and collapse — there is no
EMA target encoder and no stop-gradient (see model.py). It is the sketched
isotropic-Gaussian regulariser from LeJEPA: push z's one-dimensional marginals
toward N(0,1) along many random directions, and the joint distribution has
nowhere to collapse to.

WHY EPPS-PULLEY AND NOT A MOMENT MATCH. Matching mean and covariance to (0, I)
is a weaker constraint than it looks; Epps-Pulley compares the empirical
CHARACTERISTIC FUNCTION to the Gaussian one along each direction, so it also
sees the radial law, and it is smooth in z (no sorting, no histogram bins),
which Kolmogorov-Smirnov and binned Cramer-von Mises are not.

    T = integral |phi_n(t) - exp(-t^2/2)|^2 exp(-t^2) dt

evaluated by Gauss-Hermite quadrature on `sigreg_knots` nodes. `sigreg_knots`
and `sigreg_projections` in the config are exactly these two: quadrature nodes
per direction, and number of random directions.

WHAT IT ACTUALLY CATCHES, measured at n=384, D=192 against a null of 0.00123:

    all-latents-identical      0.503     410x null
    scale wrong (z * 3)        0.395     321x
    rank-8 collapse            0.313     254x
    heavy radial tail          0.037      30x
    scale mixture              0.021      17x
    anisotropic covariance     0.012      10x
    ---------------------------------------------- blind spots:
    5% outliers at 8 sigma     0.0024      2x
    per-coordinate bimodal     0.0013      1.1x
    unit shell at radius sqrt(D) 0.0014    1.1x

The blind spots are not a bug in the statistic, they are the projection CLT: a
random 1-D projection of a 192-dim distribution is nearly Gaussian unless the
COVARIANCE or the RADIAL law is wrong. Per-coordinate structure is averaged away
by the sketch. So SIGReg is a guarantee against collapse (which is always a rank
or scale failure, and shows up at 10-400x null), NOT a certificate that z is
Gaussian in every respect. Do not claim the latter in the writeup.

THE TABLE ABOVE IS ON RAW LATENTS. In the real model z has just come out of
BatchNorm(affine=False), which pins every coordinate to unit variance for free —
so the "scale wrong" and most of the "rank-8" signal is already gone before
SIGReg sees anything. Measured on a rank-8 latent AFTER that BN, the statistic
reads 10x null, not 254x. That is still unambiguous, and it is the regime the
sweep is calibrated in, but it means the BN and SIGReg are not redundant: BN
fixes the diagonal of the covariance, SIGReg fixes the rest.

AND IT DOES ESCAPE, but only through the BN. Optimising a rank-8 latent as a
free tensor barely moves it (the statistic is nearly flat once the marginals are
unit-variance). Optimising the MAP that produces it, with the gradient flowing
through the BN as it does in the model, works:

    step   0   sigreg 0.0139  (10.0x null)   eff_dim   7.9
    step 100   sigreg 0.0051  ( 3.8x null)   eff_dim  21.5
    step 300   sigreg 0.0021  ( 1.6x null)   eff_dim  52.0
    step 600   sigreg 0.0015  ( 1.1x null)   eff_dim  69.5

This is the single experiment that says the Stage 2 design is sound, and it is
pinned as test_sigreg_escapes_rank_collapse_through_bn.

Gradient norms here are small in absolute terms (~1e-3 for 384x192 latents),
which is why sigreg_lambda is a real knob rather than a formality and why the
sweep spans {0.01, 0.1, 0.5} rather than something tighter.

THE NULL IS NOT ZERO, AND IT DEPENDS ON BATCH SIZE. Under the null,
E|phi_n - phi|^2 = (1 - e^{-t^2})/n exactly, so

    E[T | z ~ N(0,I)] = sqrt(pi) * (1 - 1/sqrt(2)) / n = 0.51914 / n

(verified: predicted/measured within 3% at n = 64..2048). A run at batch 64 with
window 6 has n=384 and therefore FLOORS at ~0.0013 no matter how Gaussian the
encoder gets. Reading raw sigreg_loss across two runs with different batch sizes
is meaningless; `sigreg_ratio = sigreg / sigreg_null(n)` is the comparable
number, and it is what train.py logs alongside it. Ratio ~1 means "Gaussian, at
the resolution this batch size can measure".

One consequence worth knowing: because the floor comes from sampling noise, the
optimiser can push T slightly BELOW 1.0x null by making the batch mildly
repulsive (more evenly spread than iid). That is benign, and it is a second
reason the target is "ratio near 1", not "loss near 0".

Two failure signatures to watch in the curves (aq_watch streams both):
  * sigreg_loss plateaus high        -> encoder cannot reach Gaussian; usually a
                                        trailing LayerNorm somewhere, or lr too low
  * pred_loss -> 0, eff_dim collapses -> collapsing anyway, sigreg_lambda too low
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.config import CFG, N_BUTTONS


# ---- SIGReg -----------------------------------------------------------------
_GH_CACHE: dict[int, tuple] = {}


def _gauss_hermite(knots: int, device, dtype):
    """Nodes/weights for integral f(t) exp(-t^2) dt. Cached; tiny."""
    if knots not in _GH_CACHE:
        _GH_CACHE[knots] = np.polynomial.hermite.hermgauss(knots)
    t, w = _GH_CACHE[knots]
    return (torch.as_tensor(t, device=device, dtype=dtype),
            torch.as_tensor(w, device=device, dtype=dtype))


def sigreg(z: torch.Tensor, n_proj: int | None = None, knots: int | None = None,
           generator: torch.Generator | None = None, cfg=CFG) -> torch.Tensor:
    """
    Sketched isotropic-Gaussian regularisation. z is (N, D); returns a scalar
    that is 0 iff every 1-D projection of z is standard normal.

    The directions are resampled EVERY call. A fixed projection matrix is a
    finite set of constraints the encoder can satisfy exactly while staying
    non-Gaussian off-sketch; fresh directions make that impossible in
    expectation, and the extra variance is what the 1024 projections absorb.
    """
    n_proj = cfg.loss.sigreg_projections if n_proj is None else n_proj
    knots = cfg.loss.sigreg_knots if knots is None else knots
    z = z.reshape(-1, z.shape[-1]).float()
    N, D = z.shape
    # AUTOCAST OFF INSIDE THIS FUNCTION. The .float() above is not enough: under
    # torch.autocast the projection matmul and the trig still run in bf16, and
    # this statistic subtracts two numbers that are both near 1 at small t.
    # MEASURED: 1.0% error on the statistic, which is 1.0% of the NULL — and the
    # entire signal lives only a few multiples above that null. The docstring
    # already claimed fp32 accumulation; this makes the code true.
    ac = torch.autocast(device_type=z.device.type, enabled=False)

    with ac:
        V = torch.randn(D, n_proj, device=z.device, dtype=z.dtype, generator=generator)
        V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-8)
        p = z @ V                                        # (N, n_proj) marginals

        t, w = _gauss_hermite(knots, z.device, z.dtype)  # (K,), (K,)
        ang = p.unsqueeze(-1) * t                        # (N, n_proj, K)
        re = torch.cos(ang).mean(0)                      # (n_proj, K)
        im = torch.sin(ang).mean(0)
        tgt = torch.exp(-0.5 * t.pow(2))                 # CF of N(0,1)
        stat = ((re - tgt).pow(2) + im.pow(2)) * w       # (n_proj, K)
        return stat.sum(-1).mean()


_NULL_C = float(np.sqrt(np.pi) * (1 - 1 / np.sqrt(2)))      # 0.51914


def sigreg_null(n: int) -> float:
    """
    E[sigreg] when z really is N(0,I), for a batch of n vectors. Closed form,
    not a fit: E|phi_n - phi|^2 = (1 - e^{-t^2})/n, and the Gauss-Hermite
    integral of (1 - e^{-t^2})e^{-t^2} is sqrt(pi)(1 - 1/sqrt(2)).
    """
    return _NULL_C / max(1, n)


def sigreg_ratio(loss, n: int) -> float:
    """sigreg expressed in units of its own batch-size-dependent floor."""
    return float(loss) / sigreg_null(n)


def sigreg_windowed(z: torch.Tensor, cfg=CFG, **kw) -> torch.Tensor:
    """
    SIGReg on a (B, W, D) training batch — the shape the loader actually
    produces — computed PER FRAME POSITION and averaged.

    THE FLATTENED VERSION IS WRONG, AND IT IS WRONG IN THE DIRECTION THAT MAKES
    A HEALTHY RUN LOOK COLLAPSED. Epps-Pulley's null (0.51914/n) assumes iid
    samples. A batch of B windows x W frames has only B independent samples: the
    W frames inside a window are 2 emulator frames apart and near-identical.
    Measured on a synthetic FULL-RANK latent batched as 32x6:

        flattened, scored against the n=192 null    6.7x   "collapsing"
        flattened, scored against the n=32 null     1.2x   fine
        per-position mean, n=32 null                1.1x   fine

    So the first run of this loop reported 35x null and read as a catastrophe
    when part of that was the batch shape.

    Averaging W statistics each computed across the batch keeps every latent,
    makes the null exact at n=B, and removes a second problem: flattened SIGReg
    penalises the within-window similarity that the predictor is supposed to
    rely on — it was pushing consecutive frames apart while pred_loss pulled
    them together.
    """
    if z.ndim == 2:
        return sigreg(z, cfg=cfg, **kw)
    return torch.stack([sigreg(z[:, w], cfg=cfg, **kw) for w in range(z.shape[1])]).mean()


def effective_dim(z: torch.Tensor) -> float:
    """Participation ratio; re-exported from aq_watch so losses.py stands alone."""
    from algo.aq_watch import effective_dim as _ed
    return _ed(z.reshape(-1, z.shape[-1]))


# ---- Aux heads --------------------------------------------------------------
class AuxHeads(nn.Module):
    """
    Small MLPs z -> {y, scroll, dies_in_5}. Present only in the "aux" variant;
    in "pure" they are built but excluded from the loss so both variants can
    share one checkpoint format and one gate script.

    x is deliberately NOT an aux head. Encoding x is the one thing an
    unregularised JEPA already does perfectly (LeMario: x-R2 0.997) — adding a
    loss term for it spends capacity on the part that was never broken.
    """

    def __init__(self, cfg=CFG):
        super().__init__()
        d = cfg.model.latent_dim

        def head(out):
            return nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, out))

        self.y = head(1)
        self.scroll = head(1)
        self.dead = head(1)
        # dies_in_5 is true on 1.04% of observations (measured, see data.py).
        # Without pos_weight the BCE optimum is "always alive" at a loss of
        # 0.05, and the dead-in-5 gate then has nothing to measure.
        self.register_buffer("pos_weight", torch.tensor(1 / 0.0104 - 1))

    def forward(self, z):
        return {"y": self.y(z).squeeze(-1),
                "scroll": self.scroll(z).squeeze(-1),
                "dead": self.dead(z).squeeze(-1)}


def aux_heads(cfg=CFG) -> AuxHeads:
    return AuxHeads(cfg)


def aux_loss(heads: AuxHeads, z: torch.Tensor, batch: dict, cfg=CFG) -> dict:
    """Aux terms, each already multiplied by its lambda. Empty in 'pure'."""
    if cfg.loss.variant != "aux":
        zero = z.sum() * 0
        return {"aux_y": zero, "aux_scroll": zero, "aux_dead": zero}
    out = heads(z)
    s = getattr(cfg.loss, "aux_scale", 1.0)
    return {
        "aux_y": s * cfg.loss.aux_y_lambda * F.mse_loss(out["y"], batch["world_y"]),
        "aux_scroll": s * cfg.loss.aux_scroll_lambda * F.mse_loss(out["scroll"], batch["scroll"]),
        "aux_dead": s * cfg.loss.aux_alive_lambda * F.binary_cross_entropy_with_logits(
            out["dead"], batch["dies_in_5"], pos_weight=heads.pos_weight),
    }


# ---- Inverse dynamics -------------------------------------------------------
class InverseDynamics(nn.Module):
    """
    (z_t, z_t+1) -> the 6 buttons that caused the transition.

    Kept OUT of AuxHeads deliberately. AuxHeads is saved under the checkpoint's
    "aux" key and reloaded with strict=True by model.load_jepa, so adding
    parameters to it would make every checkpoint trained before today fail to
    load -- including the three runs whose gate numbers are the current results.
    This is its own module under its own key.
    """

    def __init__(self, cfg=CFG, hidden: int = 256):
        super().__init__()
        d = cfg.model.latent_dim
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(),
                                 nn.Linear(hidden, N_BUTTONS))

    def forward(self, z_a, z_b):
        return self.net(torch.cat([z_a, z_b], dim=-1))


def inverse_dynamics(cfg=CFG) -> InverseDynamics:
    return InverseDynamics(cfg)


def inverse_dynamics_loss(inv: InverseDynamics, out: dict, batch: dict,
                          cfg=CFG) -> dict:
    """
    Two BCE terms, both predicting the action from the latents around it.

    INDEXING, which is the whole correctness of this function. Per DOCS.md Part V,
    `actions[i]` CAUSED `frames[i]`, and a window starting at s carries
    `actions[s+1 .. s+W-1]`. So within a window, `batch["actions"][:, j]` is the
    action that caused window frame j+1, i.e. it is the action of the transition
    j -> j+1. Getting this off by one trains the head to predict the action from
    the pair it did NOT act on, which still produces a falling loss because
    consecutive actions are correlated -- a silent near-miss, exactly the failure
    the action alignment note in DOCS.md Part V warns about.

    The buttons are reduced over the frame_skip block with amax: a button counts
    as pressed if it was held for either emulator frame in the block.
    """
    if inv is None:
        z = out["z"]
        zero = z.sum() * 0
        return {"inv_real": zero, "inv_pred": zero}

    z, zhat = out["z"], out["zhat"]
    H = cfg.model.history_len
    a = batch["actions"]                              # (B, W-1, skip, 6)
    tgt = a.amax(dim=2).float()                       # (B, W-1, 6)

    lam_r = getattr(cfg.loss, "inv_real_lambda", 0.0)
    lam_p = getattr(cfg.loss, "inv_pred_lambda", 0.0)
    terms = {}

    if lam_r > 0:
        logit = inv(z[:, :-1], z[:, 1:])              # (B, W-1, 6)
        terms["inv_real"] = lam_r * F.binary_cross_entropy_with_logits(logit, tgt)
    else:
        terms["inv_real"] = z.sum() * 0

    if lam_p > 0:
        # frames H-1 .. W-1 as (real context tail, then the predictions)
        seq = torch.cat([z[:, H - 1:H], zhat], dim=1)  # (B, P+1, D)
        logit = inv(seq[:, :-1], seq[:, 1:])           # (B, P, 6)
        terms["inv_pred"] = lam_p * F.binary_cross_entropy_with_logits(
            logit, tgt[:, H - 1:H - 1 + zhat.shape[1]])
    else:
        terms["inv_pred"] = z.sum() * 0
    return terms


# ---- Total ------------------------------------------------------------------
def jepa_loss(out: dict, heads: AuxHeads, batch: dict, cfg=CFG, inv=None) -> dict:
    """
    out: dict from JEPA.forward. Returns every term separately AND the total,
    because the two diagnostic failure signatures are only visible term by term.
    """
    pred = F.mse_loss(out["zhat"], out["z_target"])
    z = out["z"]
    sr = sigreg_windowed(z, cfg=cfg)
    # n is the number of INDEPENDENT samples (windows), not latents — see
    # sigreg_windowed. Getting this wrong inflates the ratio ~6x at W=6.
    n_indep = z.shape[0] if z.ndim == 3 else z.shape[0]
    terms = {"pred_loss": pred, "sigreg_loss": sr,
             # dimensionless: 1.0 means "as Gaussian as this batch size can show"
             "sigreg_ratio": torch.as_tensor(sigreg_ratio(sr.detach(), n_indep))}
    # Per-horizon readouts, NOT part of the objective. gain = 1 - mse/var(target)
    # is the LeMario-comparable column (their 5-step gain was 0.455); raw mse is
    # only meaningful because BN pins var(z)=1, and BOTH are meaningless unless
    # eff_dim is quoted with them — see the note above TARGETS in config.py.
    with torch.no_grad():
        zt, zh = out["z_target"], out["zhat"]
        var = zt.reshape(-1, zt.shape[-1]).var(0).mean().clamp_min(1e-8)
        for h in (1, zt.shape[1]):
            m = F.mse_loss(zh[:, :h], zt[:, :h])
            terms[f"mse_{h}step"] = m
            terms[f"gain_{h}step"] = 1 - m / var
        terms["z_var"] = var
    terms.update(aux_loss(heads, z, batch, cfg=cfg))
    terms.update(inverse_dynamics_loss(inv, out, batch, cfg=cfg))
    terms["loss"] = pred + cfg.loss.sigreg_lambda * sr + \
        terms["aux_y"] + terms["aux_scroll"] + terms["aux_dead"] + \
        terms["inv_real"] + terms["inv_pred"]
    return terms
