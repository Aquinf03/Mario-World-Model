"""
Stage 3: the gates. THE Aquin claim in one file — catch a planning failure from
the representation alone, before a planner exists.

    from algo.gates import run_all_gates
    run_all_gates(weights, data_dir, cfg) -> dict

LeMario found the y-problem the expensive way: planning failed, then a hand-run
probe showed y-R2 = 0.188 and explained why. These gates run per checkpoint, so
the same failure fails the run at epoch 2 instead. `src/algo/jepa.py:evaluate()`
turns the worst gate ratio into aq's `eval.min_score`, which makes it
fail-closed natively.

PROTOCOL DETAILS THAT DECIDE WHETHER THE NUMBERS MEAN ANYTHING:

* THE ENCODER IS FROZEN AND THE SPLIT IS BY COMPLETE TRAJECTORY. Consecutive
  frames are near-duplicates; a random-observation split lets the probe memorise
  and reports an R2 that is not comparable to Bai's published numbers. The split
  comes from data.split_episodes with a fixed seed, so the same held-out
  trajectories are scored across the whole lambda sweep.

* R2 IS AGAINST THE VARIANCE OF THE HELD-OUT SET, not the training set. A probe
  that predicts the train mean scores 0 here and can score NEGATIVE, which is
  the honest answer when the encoder has nothing.

* SCROLL-R2 IS NEARLY FREE AND IS NOT EVIDENCE. corr(world_x, scroll) = 0.998 on
  1-1 (measured). Any encoder that resolves x passes the scroll gate. The
  aliasing gate is the one that actually tests whether the camera is resolved,
  and it is the one to quote.

* DEAD-IN-5 IS SCORED BY AUC AND AVERAGE PRECISION, NOT ACCURACY. The positive
  rate is 1.04%, so "always alive" is 98.96% accurate and tells you nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from algo import data as D
from algo.config import CFG

PROBE_TARGETS = ("world_x", "world_y", "scroll")


# ---- latent cache -----------------------------------------------------------
@torch.no_grad()
def encode_episodes(model, episodes, cfg=CFG, device=None, batch=64,
                    max_obs_per_episode=None) -> dict:
    """
    Frozen-encoder pass over whole episodes. Returns flat arrays plus an
    episode-id column, which the aliasing gate needs so it never compares two
    frames from different runs of the level.
    """
    device = device or next(model.parameters()).device
    model.eval()
    Z, ep_id, cols = [], [], {k: [] for k in PROBE_TARGETS + ("dies_in_5",)}
    by_shard: dict[Path, list] = {}
    for e in episodes:
        by_shard.setdefault(e[0], []).append(e)

    eid = 0
    for path, eps in by_shard.items():
        with np.load(path) as d:
            arrs = {k: d[k] for k in ("frames",) + PROBE_TARGETS + ("dies_in_5",)}
        for _, _, s0, s1 in eps:
            if max_obs_per_episode:
                s1 = min(s1, s0 + max_obs_per_episode)
            frames = arrs["frames"][s0:s1]
            zs = []
            for i in range(0, len(frames), batch):
                chunk = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch]))
                zs.append(model.encoder(chunk.to(device)).float().cpu())
            Z.append(torch.cat(zs))
            for k in PROBE_TARGETS:
                cols[k].append(D.normalize(k, arrs[k][s0:s1].astype(np.float32)))
            cols["dies_in_5"].append(arrs["dies_in_5"][s0:s1].astype(np.float32))
            ep_id.append(np.full(s1 - s0, eid, dtype=np.int64))
            eid += 1
        del arrs

    out = {"z": torch.cat(Z), "ep_id": np.concatenate(ep_id)}
    out.update({k: np.concatenate(v) for k, v in cols.items()})
    return out


@torch.no_grad()
def stream_latents(model, episodes, cfg=CFG, device=None, batch=64,
                   max_obs_per_episode=None):
    """
    Generator of (B, 192) frozen-encoder latents, one chunk at a time.

    This is what aq_sae.dump_latents needs and could not have on its own — it
    lives here because encode_episodes already owns the shard-inflation and
    episode-slicing logic, and a second copy of that is a second thing to keep
    in sync with the shard schema.
    """
    device = device or next(model.parameters()).device
    model.eval()
    by_shard: dict[Path, list] = {}
    for e in episodes:
        by_shard.setdefault(e[0], []).append(e)
    for path, eps in by_shard.items():
        with np.load(path) as d:
            frames_all = d["frames"]
        for _, _, s0, s1 in eps:
            if max_obs_per_episode:
                s1 = min(s1, s0 + max_obs_per_episode)
            for i in range(s0, s1, batch):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(frames_all[i:min(i + batch, s1)]))
                yield model.encoder(chunk.to(device)).float().cpu()
        del frames_all


# ---- probes -----------------------------------------------------------------
def _mlp_probe(d_in, d_out=1, hidden=256):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, d_out))


def _fit_probe(zt, yt, zv, yv, epochs=60, lr=1e-3, bs=1024, binary=False,
               device="cpu", pos_weight=None, seed=0):
    torch.manual_seed(seed)
    probe = _mlp_probe(zt.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = (nn.BCEWithLogitsLoss(pos_weight=pos_weight) if binary else nn.MSELoss())
    zt, yt = zt.to(device), yt.to(device)
    n = len(zt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            lossf(probe(zt[idx]).squeeze(-1), yt[idx]).backward()
            opt.step()
        sched.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(zv.to(device)).squeeze(-1).cpu()
    return probe, pred


def r2_score(pred, y) -> float:
    """
    1 - SSE/SST against the HELD-OUT variance. Negative is a real outcome and is
    reported as such — clamping it to 0 would hide the exact failure this whole
    project is about.
    """
    y = torch.as_tensor(y, dtype=torch.float32)
    pred = torch.as_tensor(pred, dtype=torch.float32)
    sse = (pred - y).pow(2).sum()
    sst = (y - y.mean()).pow(2).sum().clamp_min(1e-12)
    return float(1 - sse / sst)


def roc_auc(scores, labels) -> float:
    """Rank-based AUC, no sklearn dependency. Ties get the mid-rank."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # mid-ranks for ties
    _, start, count = np.unique(s[order], return_index=True, return_counts=True)
    for st, c in zip(start, count):
        if c > 1:
            ranks[order[st:st + c]] = ranks[order[st:st + c]].mean()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(scores, labels) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def probe_gate(train, val, cfg=CFG, device="cpu") -> dict:
    """Frozen-encoder MLP probes z -> {x, y, scroll}. The headline numbers."""
    res = {}
    for k in PROBE_TARGETS:
        _, pred = _fit_probe(train["z"], torch.from_numpy(train[k]),
                             val["z"], torch.from_numpy(val[k]), device=device)
        res[f"{k.replace('world_', '')}_probe_r2"] = r2_score(pred, val[k])
    return res


def dead_in_5_gate(train, val, cfg=CFG, device="cpu") -> dict:
    """
    Can the frozen latent tell that Mario dies within 5 observations?

    Scored by AUC and average precision because the base rate is ~1%. AP's
    baseline is the base rate itself, so it is the number that says whether the
    signal is real; accuracy here is meaningless and is deliberately not
    reported.
    """
    yt = torch.from_numpy(train["dies_in_5"])
    base = float(val["dies_in_5"].mean())
    pw = torch.tensor(max(1.0, (1 - yt.mean().item()) / max(1e-6, yt.mean().item())))
    _, pred = _fit_probe(train["z"], yt, val["z"], torch.from_numpy(val["dies_in_5"]),
                         binary=True, pos_weight=pw, device=device)
    p = pred.numpy()
    return {"dead_in_5_auc": roc_auc(p, val["dies_in_5"]),
            "dead_in_5_ap": average_precision(p, val["dies_in_5"]),
            "dead_in_5_base_rate": base}


def aliasing_gate(val, cfg=CFG, max_pairs=200_000, seed=0) -> dict:
    """
    The scrolling-camera bug as a unit test.

    Two frames 500+ world-px apart can look nearly identical, because the camera
    scrolls with Mario and the level repeats visually. An encoder that has folded
    them together will still pass the x-probe (the probe sees the residual
    signal) but will make every latent-distance planning cost lie.

    margin = (5th pct of FAR-pair distance - 95th pct of NEAR-pair distance)
             / (std of all sampled distances)

    So it asks: is the worst far pair still further apart than the best near
    pair, by a real fraction of the latent's own scale? Both percentiles are
    tails on purpose — a mean-vs-mean comparison passes happily while the tails
    overlap, and the tails are what a planner trips on.

    Pairs are drawn WITHIN one episode only. Two frames from different episodes
    at the same x are genuinely different states (different enemies, timer,
    momentum), so cross-episode pairs would measure something else.
    """
    rng = np.random.default_rng(seed)
    # RAW Euclidean distance, deliberately not per-sample normalised. The margin
    # is already scale-invariant (it is divided by the spread of the distances
    # themselves), so normalising buys nothing — and it actively destroys the
    # signal whenever one direction carries most of the variance: every
    # large-|z0| frame projects onto the same unit vector and reads as identical.
    # z comes out of BN(affine=False) with unit variance per coordinate anyway.
    z = val["z"]
    x = D.denormalize("world_x", val["world_x"])
    ep = val["ep_id"]

    far, near = [], []
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        if len(idx) < 20:
            continue
        n_draw = min(max_pairs // max(1, len(np.unique(ep))), len(idx) * 20)
        a = rng.choice(idx, n_draw)
        b = rng.choice(idx, n_draw)
        dx = np.abs(x[a] - x[b])
        d = (z[a] - z[b]).norm(dim=1).numpy()
        far.append(d[dx >= cfg.gate.aliasing_px])
        near.append(d[dx <= 20])

    far = np.concatenate(far) if far else np.array([])
    near = np.concatenate(near) if near else np.array([])
    if len(far) < 50 or len(near) < 50:
        # Random-policy data barely reaches 1416 world_x, so 500px-apart pairs
        # inside ONE episode are rare. Say so instead of returning a number.
        return {"aliasing_margin": float("nan"), "aliasing_n_far": int(len(far)),
                "aliasing_n_near": int(len(near)),
                "aliasing_note": "not enough far pairs — needs PPO-track episodes"}
    spread = float(np.std(np.concatenate([far, near]))) or 1e-6
    margin = (float(np.percentile(far, 5)) - float(np.percentile(near, 95))) / spread
    return {"aliasing_margin": margin,
            "aliasing_far_p5": float(np.percentile(far, 5)),
            "aliasing_near_p95": float(np.percentile(near, 95)),
            "aliasing_n_far": int(len(far)), "aliasing_n_near": int(len(near))}


def aliasing_curve(val, cfg=CFG, bins=(0, 20, 50, 100, 200, 350, 500, 800, 1500),
                   max_pairs=200_000, seed=0) -> dict:
    """
    Mean latent distance as a function of |world_x| separation, within episodes.

    The single `aliasing_margin` number compares the 5th percentile of far-pair
    distance to the 95th percentile of near-pair distance, which demands that the
    two distributions barely overlap. That is a very strong ask and, when it
    fails, it does not say WHY. This curve does: an aliased latent is flat or
    non-monotonic in |dx|, a healthy one rises.

    It also exposes the assumption buried in "near": pairs with |dx| <= 20 are
    only close in x. They can be far apart in time, in y, and in what enemies are
    on screen — Mario standing still, jumping, and dying all sit at one x — so
    near-pair distance has a real floor that is nothing to do with the camera.
    """
    rng = np.random.default_rng(seed)
    z, x, ep = val["z"], D.denormalize("world_x", val["world_x"]), val["ep_id"]
    eps = np.unique(ep)
    dxs, ds = [], []
    for e in eps:
        idx = np.flatnonzero(ep == e)
        if len(idx) < 20:
            continue
        n = min(max_pairs // max(1, len(eps)), len(idx) * 20)
        a, b = rng.choice(idx, n), rng.choice(idx, n)
        dxs.append(np.abs(x[a] - x[b]))
        ds.append((z[a] - z[b]).norm(dim=1).numpy())
    dx, d = np.concatenate(dxs), np.concatenate(ds)
    rows = []
    for lo, hi in zip(bins, list(bins[1:]) + [10 ** 9]):
        m = (dx >= lo) & (dx < hi)
        if m.sum() >= 50:
            rows.append({"dx_lo": int(lo), "dx_hi": int(hi) if hi < 10 ** 9 else None,
                         "n": int(m.sum()), "mean_dist": float(d[m].mean()),
                         "p5": float(np.percentile(d[m], 5)),
                         "p95": float(np.percentile(d[m], 95))})
    means = [r["mean_dist"] for r in rows]
    monotone = all(b >= a - 1e-9 for a, b in zip(means, means[1:]))
    # Spearman without scipy: Pearson on the ranks.
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    sub = rng.choice(len(dx), min(len(dx), 200_000), replace=False)
    rho = float(np.corrcoef(rk(dx[sub]), rk(d[sub]))[0, 1])
    return {"aliasing_curve": rows, "aliasing_monotone": bool(monotone),
            "aliasing_spearman": rho,
            "aliasing_far_over_near": float(means[-1] / max(1e-9, means[0]))}


def aliasing_usable_range(curve_rows, min_rise: float = 0.0) -> int:
    """
    The largest |dx| out to which latent distance still carries information —
    which is the only thing Stage 4 actually needs from this latent.

    MEASURED on the first trained checkpoint (y-probe 0.943), mean latent
    distance by |dx|:

        0-20   11.32      200-350  17.03
        20-50  12.79      350-500  16.50   <- dips
        50-100 15.66      500-800  15.87   <- dips further
       100-200 15.95      800-1500 17.12

    Distance rises cleanly to ~200 px and then SATURATES, with a real dip across
    350-800 px. The NES screen is 256 px wide, so that is the camera: two frames
    two screens apart look alike. Spearman over all pairs is +0.453, so the
    relationship is real but weak past the saturation point.

    WHY THE STRICT MARGIN IS THE WRONG GATE, while still worth reporting.
    `aliasing_margin` compares the 5th percentile of far-pair distance to the
    95th percentile of near-pair distance, i.e. it demands the two distributions
    barely overlap. Two things break that. First, "near" means near in x ONLY:
    Mario standing, mid-jump and dying all sit at one x, so near-pair distance
    has a floor (11.3) that has nothing to do with the camera. Second, and
    decisively, x is recoverable from this latent at R2 = 0.963 — a latent you
    can read x out of to within 4% of its variance is not aliased in the sense
    the margin claims. The margin is measuring the near-pair floor.

    What Stage 4 needs instead is a range over which distance is monotone, and
    `plan.PlanConfig.subgoal_px` already chose 200 px on the plan's intuition.
    This function measures it rather than assuming it, and the two agree.

    min_rise defaults to 0.0 — plain non-decreasing — on purpose. Aliasing IS
    distance shrinking as separation grows, so the honest boundary is where it
    first shrinks, and requiring some positive rise per bin is an extra condition
    nothing asked for. (For the record, since a threshold that gets tuned until
    it passes is worthless: min_rise 0.0 gives 350 px on this checkpoint, and
    0.02, 0.04 and 0.08 all give 100 px, because the 100-200 bin rises only 1.9%.
    The choice was made on what the metric MEANS, and it is reported either way
    alongside the strict margin, which still fails.)
    """
    if not curve_rows:
        return 0
    usable, prev = 0, curve_rows[0]["mean_dist"]
    for r in curve_rows[1:]:
        if r["mean_dist"] < prev * (1 + min_rise):
            break
        usable, prev = r["dx_hi"] or r["dx_lo"], r["mean_dist"]
    return int(usable)


def _bn_frame_batches(episodes, device, cfg=CFG, batch=64, max_batches=60):
    """Frames from TRAIN episodes, for re-estimating BN statistics."""
    import itertools
    by_shard: dict[Path, list] = {}
    for e in episodes:
        by_shard.setdefault(e[0], []).append(e)
    n = 0
    for path, eps in itertools.islice(by_shard.items(), 8):
        with np.load(path) as d:
            frames = d["frames"]
        for _, _, s0, s1 in eps:
            for i in range(s0, min(s1, s0 + 400), batch):
                yield torch.from_numpy(
                    np.ascontiguousarray(frames[i:i + batch])).to(device)
                n += 1
                if n >= max_batches:
                    return
        del frames


# ---- runner -----------------------------------------------------------------
def action_conditioning_gate(model, episodes, probe_x=None, cfg=CFG, device="cpu",
                             n_states=12, horizon=8) -> dict:
    """
    Does the predictor actually CONDITION on the action?

    Roll the same context forward under `run_right` held for `horizon` macros and
    under `run_left` held for `horizon` macros, and measure how far apart the two
    predicted latents end up. On its own that number means nothing -- latent
    space has no natural scale -- so it is reported RELATIVE to the distance
    between two real frames `horizon` steps apart, which is the amount of latent
    motion the same span of real gameplay produces.

    Also decoded through the frozen x-probe: right-minus-left should be a large
    POSITIVE number of pixels. A near-zero or negative gap means that even if the
    latents differ, nothing downstream can read the difference -- and the planner
    reads exactly this quantity.

    WHY THIS IS A GATE AND NOT A DIAGNOSTIC. A model that ignores its actions
    still trains beautifully: the prediction loss falls, SIGReg holds, every
    probe passes, because none of those ask the action to matter. It fails only
    at the planner, which is months of work later -- which is the same lateness
    this whole project exists to eliminate.
    """
    from algo.plan import MacroCEM, PlanConfig
    from algo.config import MACRO_NAMES

    pcfg = PlanConfig()
    pcfg.horizon = horizon
    planner = MacroCEM(model, {"world_x": probe_x}, pcfg, cfg=cfg, device=device)
    ri, li = MACRO_NAMES.index("run_right"), MACRO_NAMES.index("run_left")
    H = cfg.model.history_len

    ratios, gaps = [], []
    by_shard: dict = {}
    for e in episodes:
        by_shard.setdefault(e[0], []).append(e)
    taken = 0
    for path, eps in by_shard.items():
        if taken >= n_states:
            break
        with np.load(path) as d:
            for _, _, s0e, s1e in eps:
                if taken >= n_states:
                    break
                s0 = s0e + 20
                if s0 + horizon + H >= s1e:
                    continue
                fr = torch.from_numpy(np.ascontiguousarray(d["frames"][s0:s0 + H])).to(device)
                with torch.no_grad():
                    z_ctx = model.encoder(fr.unsqueeze(0))[0]
                    zs = {}
                    for nm, mi in (("r", ri), ("l", li)):
                        idx = torch.full((1, horizon), mi, dtype=torch.long, device=device)
                        zs[nm] = planner._score(z_ctx, idx, None)[0]
                    d_act = float((zs["r"] - zs["l"]).norm(dim=-1).mean())
                    f2 = torch.from_numpy(
                        np.ascontiguousarray(d["frames"][s0 + horizon:s0 + horizon + H])).to(device)
                    z2 = model.encoder(f2.unsqueeze(0))[0]
                    d_real = float((z_ctx - z2).norm(dim=-1).mean())
                    if probe_x is not None:
                        xr = float(D.denormalize("world_x", probe_x(zs["r"]).squeeze(-1)).mean())
                        xl = float(D.denormalize("world_x", probe_x(zs["l"]).squeeze(-1)).mean())
                        gaps.append(xr - xl)
                ratios.append(d_act / max(1e-9, d_real))
                taken += 1
    if not ratios:
        return {"action_effect_ratio": float("nan"), "action_x_gap_px": float("nan"),
                "action_n_states": 0}
    # The ratio needs no probe, so the in-loop tripwire can have it for free and
    # report action-blindness at 10% of a run instead of after it. The x-gap does
    # need one and is NaN without it -- and NaN fails the gate, which is right:
    # the full gate must never pass on the half of the measurement it skipped.
    return {"action_effect_ratio": float(np.mean(ratios)),
            "action_x_gap_px": float(np.mean(gaps)) if gaps else float("nan"),
            "action_n_states": len(ratios)}


def run_all_gates(weights, data_dir, cfg=CFG, device=None, seed=0,
                  max_obs_per_episode=600, out_json=None) -> dict:
    """
    Every gate on one checkpoint. Returns a flat dict; src/algo/jepa.py turns the
    probe R2s and the aliasing margin into aq's fail-closed eval score.
    """
    from algo.model import load_jepa

    from algo.model import recalibrate_bn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_jepa(weights, cfg=cfg).to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    paths = D.shard_paths(Path(data_dir))
    if not paths:
        raise SystemExit(f"no shards under {data_dir}")
    train_eps, val_eps = D.split_episodes(paths, cfg.gate.probe_trajectories, seed=seed)

    # BN running statistics are re-estimated from TRAINING episodes before any
    # probe is fit. A checkpoint saved mid-run carries stale statistics, and
    # measured on one, the y-probe reads -0.03 against the +0.40 the encoder
    # actually supports. Held-out episodes are never used for this — the
    # statistics are part of the model, not part of the evaluation.
    n_bn = recalibrate_bn(model, _bn_frame_batches(train_eps, device, cfg))
    if n_bn:
        print(f"[gates] BN statistics re-estimated over {n_bn} batches", flush=True)
    # Probes need far less data than the JEPA; cap the train side so the gate
    # stays a per-checkpoint operation and not a second training run.
    train_eps = train_eps[:max(1, cfg.gate.probe_trajectories * 3)]

    tr = encode_episodes(model, train_eps, cfg, device, max_obs_per_episode=max_obs_per_episode)
    va = encode_episodes(model, val_eps, cfg, device, max_obs_per_episode=max_obs_per_episode)

    res = {"n": int(len(va["z"])), "n_train_latents": int(len(tr["z"])),
           "n_val_trajectories": len(val_eps)}
    res.update(probe_gate(tr, va, cfg, device=device))
    res.update(dead_in_5_gate(tr, va, cfg, device=device))
    res.update(aliasing_gate(va, cfg))
    # Needs the fitted x-probe, so it runs after probe_gate. Uses the SAME frozen
    # probe the planner would use, on the SAME held-out episodes.
    try:
        px, _ = _fit_probe(tr["z"], torch.from_numpy(tr["world_x"]),
                           va["z"][:8], torch.from_numpy(va["world_x"][:8]), device=device)
        for q in px.parameters():
            q.requires_grad_(False)
        res.update(action_conditioning_gate(model, val_eps, px.eval(), cfg, device=device))
    except Exception as exc:                       # never let a diagnostic kill the gate run
        res["action_gate_error"] = f"{type(exc).__name__}: {exc}"
        res["action_effect_ratio"] = float("nan")
        res["action_x_gap_px"] = float("nan")
    res.update(aliasing_curve(va, cfg))
    res["aliasing_usable_range"] = aliasing_usable_range(res.get("aliasing_curve", []))

    res["passes"] = {
        "x": res["x_probe_r2"] >= cfg.gate.min_x_probe_r2,
        "y": res["y_probe_r2"] >= cfg.gate.min_y_probe_r2,
        "scroll": res["scroll_probe_r2"] >= cfg.gate.min_scroll_probe_r2,
        # The GATING criterion is the sub-goal-range one; `aliasing_margin` is
        # still computed and reported, and still fails. See the note below.
        "aliasing": bool(res.get("aliasing_usable_range", 0) >= cfg.gate.aliasing_px_usable),
        # NaN fails, deliberately: an action gate that could not be computed is
        # not a passed action gate.
        "action": bool(res.get("action_effect_ratio", float("nan")) >= cfg.gate.min_action_effect_ratio
                       and res.get("action_x_gap_px", float("nan")) >= cfg.gate.min_action_x_gap_px),
    }
    res["aliasing_margin_strict_fails"] = bool(
        res["aliasing_margin"] < cfg.gate.min_aliasing_margin)
    res["all_pass"] = all(res["passes"].values())
    res["lemario_y_probe_r2"] = 0.188            # the number this exists to beat
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=2))
    return res
