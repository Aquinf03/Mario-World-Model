"""
Stage 2: the JEPA world model. ~10M params, 192-dim latent.

    frames -> Encoder -> z (B,W,192)
    buttons -> ActionEncoder -> a (B,W,192)
    (z[:H], a) -> Predictor -> zhat[H:]        one parallel causal pass

Design decisions that came out of the LeMario postmortem, each one a thing that
silently breaks the y-probe if you do the obvious alternative instead:

* PROJECTION HEAD ENDS IN BATCHNORM, NOT LAYERNORM. A trailing LayerNorm puts
  every z on a fixed-radius sphere. SIGReg's whole job is to make the marginals
  N(0,1), and a sphere constraint fights that directly — the sigreg term
  plateaus high and never comes down. BN normalises across the batch instead,
  which is the same thing SIGReg is asking for.

* NO EMA TARGET ENCODER, NO STOP-GRADIENT. That machinery exists in BYOL/I-JEPA
  to stop representational collapse. SIGReg prevents collapse by construction
  (an isotropic-Gaussian embedding cannot be collapsed), so the single encoder
  is trained through both the context and the target path. `stop_grad_target`
  is exposed for ablation but defaults False — flipping it on is a real
  experiment, not a safety net.

* ACTIONS ENTER VIA AdaLN-ZERO, NOT CONCATENATION. Concatenated action tokens
  let the predictor ignore the action for the first few thousand steps and just
  learn "next frame looks like this frame", which is precisely the degenerate
  solution that leaves y unencoded. AdaLN-Zero starts each block as an exact
  identity (gates init to 0) and the action is the ONLY way to modulate it.

* THE FUTURE SLOTS ARE MASK TOKENS UNDER A CAUSAL MASK. Predicting all P future
  latents in one pass, where slot H+1 can only attend to slot H (itself a
  prediction, not a real latent), makes multi-step rollout the training
  objective rather than a thing bolted on at eval. Errors compound during
  training, where the optimiser can do something about it.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.config import CFG, N_BUTTONS


# ---- Encoder ----------------------------------------------------------------
class Encoder(nn.Module):
    """ViT-Tiny patch-14 @224 -> CLS -> MLP-BN projection -> 192-dim z."""

    def __init__(self, cfg=CFG):
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer

        d = cfg.model.latent_dim
        self.vit = VisionTransformer(
            img_size=cfg.data.frame_size, patch_size=14, in_chans=3,
            num_classes=0, global_pool="", class_token=True,
            embed_dim=192, depth=12, num_heads=3,
        )
        if getattr(cfg.model, "grad_checkpointing", False):
            self.vit.set_grad_checkpointing(True)
        # BN, not LN — see module docstring.
        self.proj = nn.Sequential(
            nn.Linear(192, 384), nn.BatchNorm1d(384), nn.GELU(),
            nn.Linear(384, d), nn.BatchNorm1d(d, affine=False),
        )
        self.latent_dim = d

    @staticmethod
    def prep(frames: torch.Tensor) -> torch.Tensor:
        """(...,H,W,3) uint8 -> (...,3,H,W) float in [-1,1]."""
        x = frames.float().div_(255.0).sub_(0.5).div_(0.5)
        return x.permute(*range(x.ndim - 3), x.ndim - 1, x.ndim - 3, x.ndim - 2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B,W,H,Wd,3) uint8 or (B,H,Wd,3) -> z (B,W,D) or (B,D)."""
        squeeze = frames.ndim == 4
        if squeeze:
            frames = frames.unsqueeze(1)
        B, W = frames.shape[:2]
        x = self.prep(frames).reshape(B * W, 3, frames.shape[2], frames.shape[3])
        tok = self.vit.forward_features(x)          # (B*W, 1+P, 192)
        z = self.proj(tok[:, 0])
        z = z.reshape(B, W, -1)
        return z.squeeze(1) if squeeze else z


# ---- Action encoder ---------------------------------------------------------
class ActionEncoder(nn.Module):
    """
    The frame_skip button vectors between two observations -> one 192-dim code.

    Input is (skip, 6), flattened. Keeping the buttons held during BOTH skipped
    emulator frames (rather than just the first) is what makes a 2-frame jump
    tap distinguishable from a 2-frame hold, and jump duration is exactly the
    thing that determines the y trajectory.
    """

    def __init__(self, cfg=CFG):
        super().__init__()
        d = cfg.model.latent_dim
        n_in = cfg.data.frame_skip * N_BUTTONS
        self.net = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(d))    # token 0 has no preceding action

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """(B,T,skip,6) float -> (B,T,D)."""
        B, T = actions.shape[:2]
        return self.net(actions.reshape(B, T, -1))


# ---- Predictor --------------------------------------------------------------
def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNBlock(nn.Module):
    """Pre-norm transformer block, AdaLN-Zero conditioned on the action code."""

    def __init__(self, dim, heads, dropout=0.0, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * mlp_ratio, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # ZERO init: every block starts as the identity, so the action is the
        # only thing that can ever move the residual stream.
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, cond, attn_mask):
        # cond is per-token (B,T,D); AdaLN params are produced per token.
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.n1(x) * (1 + sc1) + sh1
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + g1 * a
        h = self.n2(x) * (1 + sc2) + sh2
        return x + g2 * self.mlp(h)


class Predictor(nn.Module):
    """
    Causal over the whole window. Tokens 0..H-1 carry real encoder latents;
    tokens H..W-1 are a learned mask token and are what the model must fill in.
    """

    def __init__(self, cfg=CFG):
        super().__init__()
        m = cfg.model
        d, self.H, self.P = m.latent_dim, m.history_len, m.pred_horizon
        self.W = self.H + self.P
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, self.W, d) * 0.02)
        self.blocks = nn.ModuleList([
            AdaLNBlock(d, m.predictor_heads, m.predictor_dropout)
            for _ in range(m.predictor_layers)])
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(d, d)
        self.register_buffer(
            "causal", torch.triu(torch.full((self.W, self.W), float("-inf")), 1),
            persistent=False)

    def forward(self, z_ctx: torch.Tensor, a_code: torch.Tensor) -> torch.Tensor:
        """
        z_ctx  (B,H,D) encoder latents for the context frames
        a_code (B,W,D) action code per token; index 0 is the null action
        returns zhat (B,P,D) — predictions for window positions H..W-1
        """
        B = z_ctx.shape[0]
        x = torch.cat([z_ctx, self.mask_token.expand(B, self.P, -1)], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x, a_code, self.causal)
        return self.head(self.out_norm(x[:, self.H:]))


# ---- Full model -------------------------------------------------------------
class JEPA(nn.Module):
    """
    `ema_target` > 0 swaps the target branch for a momentum (EMA) teacher, the
    I-JEPA / BYOL construction, and it is a DIFFERENT experiment from
    `stop_grad_target` rather than a tidier spelling of it.

    Both stop collapse by removing the optimiser's incentive to make z_target
    constant, but they pay differently. stop_grad detaches the target, so the
    future frames contribute NO gradient to the encoder at all -- the
    representation of the thing being predicted is shaped only by whatever the
    context path and SIGReg ask for. An EMA teacher keeps a target that is a slow
    average of the student, so the target representation keeps developing while
    still being unchaseable on any single step.

    That difference is the reason to try it here. This project's measured result
    is that a pure JEPA on this recipe does not encode Mario's height (y-probe
    -0.383, worse than the 0.188 it set out to beat). If the reason is that
    stop_grad leaves the future-frame representation underdetermined, an EMA
    teacher should recover some of it WITHOUT adding a single supervised term --
    which is the only kind of fix that keeps the "fully self-supervised" claim.
    If it does not, that is a stronger negative result than the one we have.

    The gates continue to measure the STUDENT encoder, exactly as the other two
    arms do, so the three numbers stay directly comparable.
    """

    def __init__(self, cfg=CFG, stop_grad_target: bool = False,
                 ema_target: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.action_encoder = ActionEncoder(cfg)
        self.predictor = Predictor(cfg)
        self.stop_grad_target = stop_grad_target
        self.ema_target = float(ema_target)
        self.H, self.P = cfg.model.history_len, cfg.model.pred_horizon
        self.W = self.H + self.P
        self.target_encoder = None
        if self.ema_target > 0:
            import copy
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)

    def action_codes(self, actions: torch.Tensor) -> torch.Tensor:
        """(B,W-1,skip,6) -> (B,W,D), prepending the learned null action."""
        a = self.action_encoder(actions)
        null = self.action_encoder.null.expand(a.shape[0], 1, -1)
        return torch.cat([null, a], dim=1)

    def forward(self, frames: torch.Tensor, actions: torch.Tensor):
        """
        frames (B,W,H,Wd,3) uint8, actions (B,W-1,skip,6) float.
        returns dict(z, zhat, z_target).
        """
        z = self.encoder(frames)                       # (B,W,D)
        a = self.action_codes(actions)                 # (B,W,D)
        zhat = self.predictor(z[:, :self.H], a)        # (B,P,D)
        if self.target_encoder is not None:
            # Teacher sees ONLY the future frames. The student still encodes the
            # whole window, so SIGReg and the aux heads see exactly the same
            # tensor they see in the other arms -- the teacher changes what is
            # being predicted, not what is being regularised.
            with torch.no_grad():
                tgt = self.target_encoder(frames[:, self.H:])
            return {"z": z, "zhat": zhat, "z_target": tgt}
        tgt = z[:, self.H:]
        return {"z": z, "zhat": zhat,
                "z_target": tgt.detach() if self.stop_grad_target else tgt}

    @torch.no_grad()
    def ema_update(self):
        """
        One momentum step, called AFTER opt.step(). Buffers are copied, not
        lerped: BN running statistics are already an exponential average of the
        student's activations, and averaging an average again lags the teacher's
        normalisation behind its own weights, which shows up as a slow drift
        between train and eval statistics.
        """
        if self.target_encoder is None:
            return
        m = self.ema_target
        for pt, ps in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.lerp_(ps.detach(), 1 - m)
        for bt, bs in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            bt.copy_(bs)

    @torch.no_grad()
    def rollout(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Latent rollout from real context frames. Same pass as training."""
        z = self.encoder(frames[:, :self.H])
        return self.predictor(z, self.action_codes(actions))

    def param_report(self) -> dict:
        def n(m):
            return sum(p.numel() for p in m.parameters())
        rep = {
            "encoder": n(self.encoder),
            "action_encoder": n(self.action_encoder),
            "predictor": n(self.predictor),
            # n(self) would fold in the frozen EMA teacher and report a model
            # ~5.6M params larger than the one being trained, breaking the
            # "~15M band" check and every params-vs-LeMario comparison.
            "total": n(self) - (n(self.target_encoder) if self.target_encoder is not None else 0),
        }
        if self.target_encoder is not None:
            rep["ema_teacher_frozen"] = n(self.target_encoder)
        return rep


@torch.no_grad()
def recalibrate_bn(model, frame_batches, max_batches: int = 60) -> int:
    """
    Re-estimate the encoder's BatchNorm running statistics from current data.

    WHY THIS IS NOT OPTIONAL. Every downstream measurement — the per-checkpoint
    gate, the Stage 3 probes, the SAE latent dump, the planner — runs the encoder
    in eval(), where BN normalises with RUNNING statistics. Those track the
    PRE-BN activations, which drift fast while the encoder is still learning, so
    mid-training they are simply stale.

    MEASURED on a checkpoint 10% into a real run:

        BN mode                     y-probe   x-probe
        running stats (eval)         -0.028    +0.238
        batch stats (train-mode)     +0.404    +0.343

    with running_var sitting at 0.0975 instead of ~1. The gate was reporting a
    healthy encoder as dead, and the fail-closed abort — which triggers on
    y < 0.10 — was about to kill a run measuring 0.40. An end-of-run gate does
    not show this because the lr has annealed and the statistics have caught up,
    which is exactly why it survived undetected until checkpoints were taken
    mid-run.

    Resetting to momentum=None makes each BN accumulate a cumulative average
    over the batches it sees here, so the result is the true mean/var over
    `max_batches` rather than an exponentially-weighted tail.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    saved = [(b.momentum, b.training) for b in bns]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
        b.train()
    was_training = model.training
    model.eval()                      # dropout off; the BNs above stay in train mode
    for b in bns:
        b.train()

    n = 0
    for frames in frame_batches:
        model.encoder(frames)
        n += 1
        if n >= max_batches:
            break

    for b, (mom, tr) in zip(bns, saved):
        b.momentum = mom
        b.train(tr)
    model.train(was_training)
    return n


def build_jepa(cfg=CFG, **kw) -> JEPA:
    return JEPA(cfg, **kw)


def load_jepa(weights: str | Path, cfg=CFG, map_location="cpu", strict: bool = True) -> JEPA:
    """
    Rebuild from a .pt written by train.py. The aq checkpoint is JSON and only
    holds a POINTER to this file (engine/step.py json.loads the checkpoint), so
    this is the only path that reconstitutes real weights.

    STRICT ON PURPOSE. `strict=False` here is a trap: a key-name mismatch loads
    nothing, leaves a randomly-initialised encoder, and the gates then report
    x-R2 0.40 / y-R2 -0.00 / aliasing -0.94 — which is a perfectly plausible
    "the run collapsed" result, and is completely wrong. Measured: those are the
    exact numbers an UNTRAINED checkpoint produces, so a silent no-load is
    indistinguishable from a real failure. Fail loudly instead.
    """
    from algo.losses import AuxHeads

    blob = torch.load(Path(weights), map_location=map_location, weights_only=False)
    if not isinstance(blob, dict) or "jepa" not in blob:
        raise ValueError(
            f"{weights} is not an aq-mario checkpoint (expected a dict with a "
            f"'jepa' key; got {type(blob).__name__} "
            f"{sorted(blob)[:6] if isinstance(blob, dict) else ''})")

    # An EMA run's state_dict carries target_encoder.* keys; rebuilding without
    # the teacher makes a strict load fail, and strict=False here is the trap
    # documented above. Read the setting off the checkpoint instead.
    m = JEPA(cfg, ema_target=float(blob.get("ema_target", 0.0) or 0.0))
    m.load_state_dict(blob["jepa"], strict=strict)
    if "aux" in blob:
        heads = AuxHeads(cfg)
        heads.load_state_dict(blob["aux"], strict=strict)
        m.aux_heads = heads
    m.loaded_from = str(weights)
    m.loaded_epoch = blob.get("epoch")
    return m.eval()
