"""
Stage 4: plan with the world model, and steer it with a feature the SAE found.

    from algo.plan import MacroCEM, rollout_cost, steer_predictor
    planner = MacroCEM(model, probes)
    actions = planner.plan(frames, actions_so_far)

TWO DECISIONS FROM THE LeMario POSTMORTEM, both of which are about the optimiser
fighting the problem rather than the model being wrong:

* CATEGORICAL CEM OVER 5 MACRO-ACTIONS, NOT GAUSSIAN CEM OVER 6 BINARY BUTTONS.
  A Gaussian proposal over binary buttons spends most of its mass on button
  combinations that are not reachable and have no dynamics behind them — half of
  LeMario's planning failures were the optimiser, not the world model. The macro
  set in config.MACRO_ACTIONS is 5 things Mario can actually do.

* SUB-GOALS EVERY ~200 PX, NOT ONE LATENT-DISTANCE CALL TO THE FLAGPOLE. Latent
  distance is only trustworthy over the range the aliasing gate certifies. A
  single call spanning 3000 px asks the latent a question the camera-aliasing
  gate says it cannot answer, and the planner will happily walk into it.

THE COST IS PROBE-SCORED, NOT LATENT-DISTANCE-SCORED. `x_progress` comes from the
frozen x-probe applied to the PREDICTED latent, so the planner optimises a
quantity the Stage 3 gates have actually measured the reliability of. A cost
built on raw latent distance is a cost whose units nobody has validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from algo import data as D
from algo.config import CFG, MACRO_ACTIONS, MACRO_NAMES


def macro_button_block(name: str, skip: int | None = None) -> np.ndarray:
    """One macro action -> the (skip, 6) button block the action encoder expects."""
    skip = skip or CFG.data.frame_skip
    return np.tile(np.array(MACRO_ACTIONS[name], dtype=np.float32), (skip, 1))


MACRO_BLOCKS = np.stack([macro_button_block(n) for n in MACRO_NAMES])   # (5, skip, 6)


# ---- cost ------------------------------------------------------------------
@dataclass
class PlanConfig:
    horizon: int = 8              # macros per plan
    population: int = 256
    elites: int = 32
    iters: int = 4
    replan_every: int = 2         # macros executed before replanning
    death_penalty: float = 20.0
    goal_weight: float = 1.0
    subgoal_px: int = 200         # never ask the latent about a longer span
    temperature: float = 1.0

    # ---- added after the first planner runs, all three measured -------------
    # switch_penalty: px-equivalent charged per macro CHANGE within a plan. The
    # visible symptom of a weak action signal is oscillation -- the best
    # checkpoint separates run_right from run_left by +6.07 px of decoded x when
    # the true effect is ~200 px, so candidate plans are nearly tied and CEM
    # picks noise. Charging for direction changes breaks those ties toward
    # committed motion instead of toward whichever candidate the probe happened
    # to score highest.
    switch_penalty: float = 0.0
    # death_relative: rank candidates by risk instead of trusting the absolute
    # probability. MEASURED: P(dead) reads 0.000 on every macro from predicted
    # latents, so the death term contributes nothing and the planner has NO
    # hazard avoidance at all -- which is why it falls off 1-3. Standardising
    # risk across the candidate population restores the ordering without
    # pretending the probe is calibrated, which it is not.
    death_relative: bool = False
    death_px: float = 60.0        # px-equivalent charged to a +1 sigma risk


def rollout_cost(zhat, probes, goal_x=None, cfg=CFG, pcfg: PlanConfig | None = None):
    """
    Score predicted latents with the FROZEN probes.

        cost = -x_progress + death_penalty * P(dead) + goal_weight * |x - goal|

    zhat   (P, H, D) predicted latents for P candidate plans over H steps
    probes dict of the fitted probe modules from gates.probe_gate
    """
    pcfg = pcfg or PlanConfig()
    with torch.no_grad():
        flat = zhat.reshape(-1, zhat.shape[-1])
        x = probes["world_x"](flat).squeeze(-1).reshape(zhat.shape[:2])
        dead = torch.sigmoid(probes["dies_in_5"](flat).squeeze(-1)).reshape(zhat.shape[:2])

        x_px = D.denormalize("world_x", x)
        progress = x_px[:, -1] - x_px[:, 0]
        risk = dead.max(dim=1).values

        if pcfg.death_relative:
            # z-scored within the candidate population: relative, unit-free, and
            # honest about the probe being uncalibrated.
            rz = (risk - risk.mean()) / risk.std().clamp_min(1e-6)
            cost = -progress + pcfg.death_px * rz
        else:
            cost = -progress + pcfg.death_penalty * risk * 100.0
        if goal_x is not None:
            # Clamped to subgoal_px: beyond that the aliasing gate has not
            # certified the latent, so a larger residual is not more information.
            gap = (x_px[:, -1] - float(goal_x)).abs().clamp(max=pcfg.subgoal_px)
            cost = cost + pcfg.goal_weight * gap
    return cost, {"progress": progress, "risk": risk}


# ---- planner ---------------------------------------------------------------
class MacroCEM:
    """
    Cross-entropy method over macro-action sequences, in latent space.

    The distribution is a per-timestep categorical over the 5 macros — one
    (H, 5) probability table, refit from the elite set each iteration. No
    Gaussian anywhere near the action space.
    """

    def __init__(self, model, probes, pcfg: PlanConfig | None = None, cfg=CFG,
                 device=None, seed=0):
        self.model = model.eval()
        self.probes = probes
        self.p = pcfg or PlanConfig()
        self.cfg = cfg
        self.device = device or next(model.parameters()).device
        self.rng = np.random.default_rng(seed)
        self.n_macros = len(MACRO_NAMES)
        self.blocks = torch.from_numpy(MACRO_BLOCKS).to(self.device)

    @torch.no_grad()
    def _score(self, z_ctx, idx, goal_x):
        """idx (P, H) macro indices -> cost per candidate."""
        P, H = idx.shape
        # The predictor's positional embedding is fixed at history + horizon, so
        # a plan longer than pred_horizon is rolled in chunks rather than by
        # pretending the model supports an arbitrary window.
        step = self.model.P
        z = z_ctx.unsqueeze(0).expand(P, -1, -1).contiguous()
        outs = []
        for s in range(0, H, step):
            chunk = idx[:, s:s + step]
            if chunk.shape[1] < step:          # pad the tail with "wait"
                pad = torch.zeros(P, step - chunk.shape[1], dtype=chunk.dtype,
                                  device=chunk.device)
                chunk = torch.cat([chunk, pad], dim=1)
            acts = self.blocks[chunk]                      # (P, step, skip, 6)
            # tokens: history_len context + step futures, null action at index 0
            hist_pad = torch.zeros(P, self.model.H - 1, *acts.shape[2:],
                                   device=acts.device)
            a_all = self.model.action_codes(torch.cat([hist_pad, acts], dim=1))
            zhat = self.model.predictor(z[:, -self.model.H:], a_all)
            outs.append(zhat)
            z = torch.cat([z, zhat], dim=1)
        return torch.cat(outs, dim=1)[:, :H]

    @torch.no_grad()
    def plan(self, frames, goal_x=None):
        """
        frames (history_len, H, W, 3) uint8 -> (macro indices, names, diagnostics)
        """
        z_ctx = self.model.encoder(
            torch.as_tensor(frames).unsqueeze(0).to(self.device))[0]
        H, P = self.p.horizon, self.p.population
        logits = torch.zeros(H, self.n_macros, device=self.device)

        for _ in range(self.p.iters):
            probs = torch.softmax(logits / self.p.temperature, dim=-1)
            idx = torch.multinomial(probs, P, replacement=True).T.contiguous()  # (P,H)
            zhat = self._score(z_ctx, idx, goal_x)
            cost, diag = rollout_cost(zhat, self.probes, goal_x, self.cfg, self.p)
            if self.p.switch_penalty:
                switches = (idx[:, 1:] != idx[:, :-1]).float().sum(dim=1)
                cost = cost + self.p.switch_penalty * switches
            elite = idx[cost.argsort()[: self.p.elites]]                        # (E,H)
            counts = torch.zeros_like(logits)
            for h in range(H):
                counts[h] = torch.bincount(elite[:, h], minlength=self.n_macros).float()
            # +1 smoothing: a macro that no elite happened to pick must not have
            # its probability driven to exactly zero, or CEM cannot recover it.
            logits = torch.log(counts + 1.0)

        best = torch.softmax(logits, -1).argmax(-1)
        return best.cpu().numpy(), [MACRO_NAMES[i] for i in best.cpu().numpy()], {
            "final_cost": float(cost.min()),
            "progress_px": float(diag["progress"].max()),
            "risk": float(diag["risk"].min()),
        }


# ---- steering --------------------------------------------------------------
def steer_predictor(model, decoder_col: torch.Tensor, alpha: float):
    """
    Add alpha * (an SAE decoder column) into the predictor's residual stream, and
    return a handle that removes it.

    `aquin steer` cannot do this — it is prompt-in / tokens-out and assumes a
    tokeniser and an LLM residual stream. This is about twenty lines and is ours
    to write, which is the honest version of "we used the interp tooling".

    The hook lands on the LAST predictor block so the edit is not re-normalised
    by anything downstream except the final LayerNorm, which is affine-free.
    """
    v = decoder_col.detach().to(next(model.parameters()).device).view(1, 1, -1)

    def hook(_mod, _inp, out):
        return out + alpha * v

    return model.predictor.blocks[-1].register_forward_hook(hook)


def steering_effect(model, frames, actions, decoder_col, alphas=(-4, -2, 0, 2, 4),
                    probe=None):
    """
    Sweep the steering coefficient and report what it does to the ROLLOUT, in
    world units when a probe is supplied.

    The claim "the SAE found a y-feature" is only worth making if pushing that
    feature moves y and not everything else, so this returns the per-alpha
    prediction rather than a single number.
    """
    out = []
    for a in alphas:
        h = steer_predictor(model, decoder_col, float(a))
        try:
            with torch.no_grad():
                zhat = model.rollout(frames, actions)
            row = {"alpha": float(a), "z_norm": float(zhat.norm(dim=-1).mean())}
            if probe is not None:
                y = probe(zhat.reshape(-1, zhat.shape[-1])).squeeze(-1)
                row["y_px"] = float(D.denormalize("world_y", y).mean())
            out.append(row)
        finally:
            h.remove()
    return out
