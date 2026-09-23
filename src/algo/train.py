"""
Stage 2 training loop. Called two ways:

    from algo.train import train_jepa      # by src/algo/jepa.py under `aq train`
    python -m algo.train --dry-run         # single shard, proves both losses move

Everything the run needs to be judged by is written to run_dir:
    metrics.jsonl    one row per step — the SAME file `aquin watch ingest` tails
    param_count.json param report, read by the tracker
    dryrun.json      did pred_loss and sigreg_loss actually move

WHY BOTH LOSSES ARE LOGGED SEPARATELY AND NEVER SUMMED IN THE LOG. The two ways
this run dies look identical in the total: sigreg plateauing high and pred_loss
diving to zero both leave `loss` looking "fine-ish" for a few hundred steps. The
diagnosis is only in the split, plus eff_dim.

BF16, NOT FP16. The SIGReg statistic squares a difference of two numbers that
are both near 1 at small t; in fp16 that subtraction loses most of its
significant digits and the gradient turns to noise. bf16 keeps the exponent
range, and the statistic is accumulated in fp32 anyway (sigreg() casts).
"""
from __future__ import annotations

import argparse
import json
from collections import deque
import math
import time
from pathlib import Path

import torch

from algo.aq_watch import MetricsWriter, effective_dim_windowed
from algo.config import CFG
from algo import data as D
from algo.losses import aux_heads, inverse_dynamics, jepa_loss
from algo.model import build_jepa


def checkpoint_health(hist, gate_hist, cfg=CFG) -> dict:
    """
    Is training going well? Compared against the PREVIOUS checkpoint, not against
    an absolute bar, because the absolute bars are what the Stage 3 gate is for.

    Four questions, each mapping to a way this run is known to fail:
      learning      is pred_loss below where it was a checkpoint ago
      not collapsing  is eff_dim holding (>=0.85 of its best so far)
      regularised   is sigreg within 5x its own batch-size null
      representing  is the y-probe rising, or already past LeMario's 0.188

    "representing" is the only one of the four that a collapsed-but-confident
    run cannot fake, which is why it is the one wired to the abort.
    """
    def tail(rows, key, n=40):
        vals = [r[key] for r in rows[-n:] if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    cur = {"step": hist[-1]["step"], "epoch": hist[-1]["epoch"],
           "pred_loss": tail(hist, "pred_loss"),
           "eff_dim": tail(hist, "eff_dim"),
           "sigreg_ratio": tail(hist, "sigreg_ratio"),
           "gain_5step": tail(hist, f"gain_{cfg.model.pred_horizon}step")}
    # .get, not [], on purpose: this runs INSIDE the training loop at every
    # checkpoint, so a missing key here is a crash 40% into a paid run. A health
    # report that cannot see a metric should say so, not take the run down.
    if gate_hist:
        for k in ("y_probe_r2", "x_probe_r2", "scroll_probe_r2", "action_effect_ratio"):
            if k in gate_hist[-1]:
                cur[k] = gate_hist[-1][k]
    prev = gate_hist[-2] if len(gate_hist) > 1 else None
    best_ed = max([h.get("eff_dim", 0) for h in gate_hist] + [cur["eff_dim"]])

    cur["ok_learning"] = bool(prev is None or cur["pred_loss"] <= prev.get("pred_loss", 1e9) * 1.02)
    cur["ok_not_collapsing"] = bool(cur["eff_dim"] >= 0.85 * best_ed)
    cur["ok_regularised"] = bool(cur["sigreg_ratio"] < 5.0)
    y = cur.get("y_probe_r2")
    py = prev.get("y_probe_r2") if prev else None
    cur["ok_representing"] = bool(y is None or y > 0.188 or py is None or y >= py - 0.02)
    cur["healthy"] = all(cur[k] for k in
                         ("ok_learning", "ok_not_collapsing", "ok_regularised", "ok_representing"))
    return cur


class GateAbort(RuntimeError):
    """
    Raised when a per-epoch gate fails after the grace period.

    THIS IS THE PROJECT. LeMario discovered the y-probe collapse after planning
    failed, by hand, months in. Everything else here is machinery for making this
    exception possible: a run that has thrown away Mario's height stops now,
    instead of producing a checkpoint that looks fine by its loss.
    """


def _epoch_gate(model, val_eps, cfg, device, n_traj=24, max_obs=300, bn_batches=None):
    """
    Cheap probe gate on the held-out trajectories, run between epochs.

    Deliberately smaller than the Stage 3 gate (24 trajectories, 300 obs each,
    40 probe epochs) — it is a tripwire, not the final measurement, and it has to
    be cheap enough that nobody is tempted to switch it off. The full-strength
    version is scripts/helpers/run_gates.py on the saved checkpoint.
    """
    from algo.gates import action_conditioning_gate, encode_episodes, probe_gate
    from algo.model import recalibrate_bn

    # NOT decorated @torch.no_grad(). encode_episodes already carries it for the
    # frozen forward pass, but the PROBES inside probe_gate have to train, and a
    # no_grad wrapper here makes their backward raise "element 0 of tensors does
    # not require grad". That failure only appears at the END of epoch 0, i.e.
    # ~30 minutes into a full run, which is precisely the kind of bug that has to
    # be caught by a two-epoch smoke test rather than by the real thing.
    was_training = model.training

    # Refresh BN running statistics before measuring anything. Mid-training they
    # are stale enough to report a y-probe of -0.03 on an encoder that actually
    # measures +0.40 — see model.recalibrate_bn. Without this the fail-closed
    # abort fires on a healthy run.
    if bn_batches:
        recalibrate_bn(model, bn_batches)
    model.eval()
    try:
        eps = list(val_eps)
        half = max(2, len(eps) // 2)
        tr = encode_episodes(model, eps[:half][:n_traj], cfg, device, max_obs_per_episode=max_obs)
        va = encode_episodes(model, eps[half:][:n_traj], cfg, device, max_obs_per_episode=max_obs)
        res = probe_gate(tr, va, cfg, device=str(device))
        # Action conditioning, probe-free so it is cheap enough for a tripwire.
        # A model that ignores its actions passes every probe above and is still
        # useless to a planner; without this the run reports "healthy" for 2.5 h
        # and the failure only appears when CEM oscillates. Never fatal here --
        # it is reported, and the full gate is the one that fails closed.
        try:
            res.update(action_conditioning_gate(model, eps[half:], None, cfg,
                                                device=str(device), n_states=6))
        except Exception as exc:
            res["action_gate_error"] = f"{type(exc).__name__}: {exc}"
        return res
    finally:
        if was_training:
            model.train()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def _cosine_lr(step, total, base_lr, warmup=500):
    # A 500-step warmup on a 400-step dry run means the lr NEVER leaves warmup
    # and the run is judged at a fraction of its intended learning rate. Scale
    # the warmup to the run when the run is short.
    warmup = min(warmup, max(1, total // 5))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def train_jepa(data_dir, out_weights, epochs=12, batch_size=64, lr=3e-4,
               precision="bf16", cfg=CFG, run_dir=None, num_workers=4,
               max_steps=None, log_every=10, val_trajectories=None,
               device=None, seed=0, stop_grad_target=False,
               gate_every_epoch=True, gate_abort_y_r2=0.10,
               checkpoint_every_frac=0.10, gate_grace_frac=0.35, resume="",
               ema_target=0.0, inv_dyn=True):
    torch.manual_seed(seed)
    device = device or pick_device()
    run_dir = Path(run_dir or (cfg.run_dir / time.strftime("run_%Y%m%d_%H%M%S")))
    run_dir.mkdir(parents=True, exist_ok=True)
    out_weights = Path(out_weights)
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    paths = D.shard_paths(Path(data_dir))
    if not paths:
        raise SystemExit(f"no shard_*.npz under {data_dir} — Stage 1 has not run")
    train_eps, val_eps = D.split_episodes(paths, val_trajectories, seed=seed)
    loader = D.make_loader(train_eps, batch_size=batch_size,
                           num_workers=num_workers, cfg=cfg, seed=seed)

    model = build_jepa(cfg, stop_grad_target=stop_grad_target,
                       ema_target=ema_target).to(device)
    heads = aux_heads(cfg).to(device)
    inv = None
    if inv_dyn and (getattr(cfg.loss, "inv_real_lambda", 0) > 0
                    or getattr(cfg.loss, "inv_pred_lambda", 0) > 0):
        inv = inverse_dynamics(cfg).to(device)
    params = model.param_report()
    params["aux_heads"] = sum(p.numel() for p in heads.parameters())
    params["total"] += params["aux_heads"]
    if inv is not None:
        params["inv_dyn"] = sum(p.numel() for p in inv.parameters())
        params["total"] += params["inv_dyn"]
    (cfg.run_dir).mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "param_count.json").write_text(json.dumps(params, indent=2))

    # requires_grad filter, not model.parameters(): the EMA teacher is frozen, and
    # handing its tensors to AdamW gives it weight-decay state and decays weights
    # that ema_update() then overwrites -- a silent no-op that still costs memory.
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
        + (list(inv.parameters()) if inv is not None else []),
        lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

    # ---- resume ---------------------------------------------------------------
    # A 2.7 h run that loses its client dies at whatever step it had reached, and
    # without this the only options are "pay for it again from zero" or "publish
    # numbers from 16% of a schedule". Both happened on this project already.
    #
    # THE OPTIMIZER STATE IS PART OF THE CHECKPOINT, not an extra. AdamW's second
    # moment takes hundreds of steps to re-warm; resuming weights alone puts a
    # visible step-change in pred_loss at the seam and makes the curve a record of
    # the interruption rather than of the training. Restored: weights, aux heads,
    # optimizer moments, the global step (so the cosine schedule is continuous),
    # the epoch, the recalibrated total_steps and the gate history.
    #
    # NOT restored: the loader's position inside the epoch. ShardWindows reshuffles
    # shard AND window order every pass off self._epoch, so there is no position to
    # return to -- resuming re-draws from the same distribution, which is what a
    # shuffled stream means. That is honest for a stochastic sampler and is the
    # standard practice; it is not exact-state resumption and should not be called it.
    start_epoch, resumed_total, gate_hist_resumed, resumed = 0, None, [], False
    if resume:
        rp = Path(resume)
        if not rp.is_file():
            raise SystemExit(f"--resume {rp} does not exist")
        blob = torch.load(rp, map_location=device, weights_only=False)
        model.load_state_dict(blob["jepa"])
        if "aux" in blob:
            heads.load_state_dict(blob["aux"])
        if inv is not None and "invdyn" in blob:
            inv.load_state_dict(blob["invdyn"])
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])
        else:
            print(f"[train] WARNING {rp} predates optimizer checkpointing — "
                  f"AdamW moments restart cold and the loss will bump at the seam",
                  flush=True)
        resumed = True
        resume_step = int(blob.get("step", 0))
        start_epoch = int(blob.get("epoch", 0))
        resumed_total = blob.get("total_steps")
        gp = run_dir / "gate_by_epoch.json"
        if gp.is_file():
            try:
                gate_hist_resumed = json.loads(gp.read_text())
            except json.JSONDecodeError:
                pass
        print(f"[train] resumed {rp.name} at step {resume_step} epoch {start_epoch} "
              f"({len(gate_hist_resumed)} prior gates, "
              f"opt state {'restored' if 'opt' in blob else 'COLD'})", flush=True)

    # Windows per epoch is not known until a shard is opened, so the cosine
    # schedule is length-estimated from the episode table rather than len(loader)
    # (an IterableDataset has no length).
    stride = getattr(cfg.data, "window_stride", 1)
    est_windows = sum(max(0, (b - a) - D.window_len(cfg)) // stride for _, _, a, b in train_eps)
    # MEASURED: this estimate is 14.5% LOW, because death-window oversampling
    # duplicates windows and the formula does not model it. The cosine schedule
    # is built from it, so the lr was hitting its floor ~15% before the end and
    # the tail of every run trained at ~0 lr. Recalibrated from the real step
    # count once epoch 0 has actually happened; the estimate only has to be
    # good enough to get through the first epoch.
    total_steps = max_steps or max(1, epochs * est_windows // max(1, batch_size))
    if resumed and resumed_total:
        total_steps = int(resumed_total)

    mw = MetricsWriter(run_dir)
    step, hist, gate_hist = 0, [], list(gate_hist_resumed)
    if resumed:
        step = resume_step
    ckpt_every = max(1, int(total_steps * checkpoint_every_frac))
    bn_buffer = deque(maxlen=48)          # recent frames, for BN re-estimation
    t0 = time.time()
    tgt_branch = (f"ema({ema_target})" if ema_target > 0
                  else ("stop_grad" if stop_grad_target else "shared"))
    print(f"[train] {len(train_eps)} train eps / {len(val_eps)} held-out, "
          f"~{est_windows} windows, {total_steps} steps, {params['total']/1e6:.1f}M params, "
          f"batch {batch_size}, lam {cfg.loss.sigreg_lambda}, "
          f"target={tgt_branch}, "
          f"inv_dyn={'off' if inv is None else f'{cfg.loss.inv_real_lambda:g}/{cfg.loss.inv_pred_lambda:g}'}, "
          f"device={device.type}", flush=True)

    for epoch in range(start_epoch, epochs):
        # Counted per epoch, not read off the global `step`: on a resumed run
        # `step` already carries the pre-interruption steps, and using it to
        # derive steps-per-epoch would inflate total_steps and flatten the
        # cosine schedule for the rest of the run.
        epoch_start_step = step
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = _cosine_lr(step, total_steps, lr)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Rolling window of recent frames, used to re-estimate BN statistics
            # before each gate. Kept on-device and small; it is only ever read.
            bn_buffer.append(batch["frames"])
            with _autocast(device, precision):
                out = model(batch["frames"], batch["actions"])
                terms = jepa_loss(out, heads, batch, cfg=cfg, inv=inv)
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(heads.parameters())
                + (list(inv.parameters()) if inv is not None else []), 1.0)
            opt.step()
            model.ema_update()          # no-op unless ema_target > 0

            if step % log_every == 0:
                pr, pr_c, n_ind = effective_dim_windowed(out["z"].detach())
                row = dict(step=step, epoch=epoch,
                           lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                           # per-frame-position, so n is INDEPENDENT windows
                           eff_dim=pr, eff_dim_corrected=pr_c, n_indep=n_ind,
                           eff_dim_null=n_ind * cfg.model.latent_dim / (n_ind + cfg.model.latent_dim),
                           **{k: float(v.detach()) for k, v in terms.items()})
                row["sec"] = round(time.time() - t0, 1)
                mw.log(**row)
                hist.append(row)
                print(f"  step {step:6d} ep {epoch} loss {row['loss']:.4f} "
                      f"pred {row['pred_loss']:.4f} "
                      f"sigreg {row['sigreg_loss']:.4f} ({row['sigreg_ratio']:.1f}x null) "
                      f"eff_dim {row['eff_dim']:.0f}/{row['eff_dim_null']:.0f}", flush=True)
            step += 1

            # ---- checkpoint + gate every `checkpoint_every_frac` of the run ----
            # Per-EPOCH was too coarse: at 3 epochs that is 3 look-ins on a 1.8 h
            # run, so a collapse starting at 40% is not visible until 67%.
            if ckpt_every and step % ckpt_every == 0 and hist:
                frac = step / max(1, total_steps)
                blob = {"jepa": model.state_dict(), "aux": heads.state_dict(),
                        "opt": opt.state_dict(),
                        **({"invdyn": inv.state_dict()} if inv is not None else {}),
                        "cfg_variant": cfg.loss.variant,
                        "sigreg_lambda": cfg.loss.sigreg_lambda,
                        "ema_target": ema_target, "stop_grad": stop_grad_target,
                        "epoch": epoch, "step": step, "frac": frac,
                        "total_steps": total_steps, "epochs": epochs}
                torch.save(blob, out_weights)
                torch.save(blob, out_weights.with_name(
                    f"{out_weights.stem}_p{int(round(frac * 100)):03d}.pt"))
                if gate_every_epoch and val_eps:
                    g = _epoch_gate(model, val_eps, cfg, device,
                                    bn_batches=list(bn_buffer))
                    g.update(epoch=epoch, step=step, frac=frac)
                    gate_hist.append(g)
                h = checkpoint_health(hist, gate_hist, cfg)
                gate_hist[-1].update({k: v for k, v in h.items() if k not in gate_hist[-1]})
                mw.log(step=step, epoch=epoch,
                       **{f"gate_{k}": v for k, v in gate_hist[-1].items()})
                (run_dir / "gate_by_epoch.json").write_text(json.dumps(gate_hist, indent=2))
                flag = "OK " if h["healthy"] else "WARN"
                print(f"  [{int(round(frac*100)):3d}%] {flag} "
                      f"y {h.get('y_probe_r2', float('nan')):+.4f} "
                      f"x {h.get('x_probe_r2', float('nan')):+.4f} "
                      f"act {gate_hist[-1].get('action_effect_ratio', float('nan')):.3f} | "
                      f"pred {h['pred_loss']:.4f} gain5 {h['gain_5step']:+.3f} "
                      f"eff_dim {h['eff_dim']:.1f} sigreg {h['sigreg_ratio']:.1f}x"
                      + ("" if h["healthy"] else "  <- " + ",".join(
                          k[3:] for k in ("ok_learning", "ok_not_collapsing",
                                          "ok_regularised", "ok_representing") if not h[k])),
                      flush=True)
                # Abort needs BOTH a low value AND no progress. A single low
                # reading is not enough: the y-probe at 10% of a healthy run
                # measured 0.12, below LeMario's 0.188, purely because the
                # encoder is young. Killing on one number would have thrown away
                # a run that was on its way up. Two consecutive non-improving
                # checkpoints below the floor is a trend, not a reading.
                ys = [g.get("y_probe_r2") for g in gate_hist if "y_probe_r2" in g]
                if (frac >= gate_grace_frac and len(ys) >= 2
                        and ys[-1] < gate_abort_y_r2
                        and ys[-1] <= ys[-2] + 0.01):
                    raise GateAbort(
                        f"{int(frac*100)}% in: y-probe R2 {ys[-1]:.4f} < "
                        f"{gate_abort_y_r2} and not improving ({ys[-2]:.4f} -> "
                        f"{ys[-1]:.4f}). The representation has thrown away "
                        f"Mario's height; every downstream number would be "
                        f"measuring nothing. Stopping.")

            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

        if epoch == 0 and not max_steps and not resumed:
            # Real steps-per-epoch is now known; rebuild the schedule around it.
            per_epoch = step - epoch_start_step
            total_steps = max(1, epochs * per_epoch)
            ckpt_every = max(1, int(total_steps * checkpoint_every_frac))
            print(f"[train] recalibrated: {per_epoch} steps/epoch -> {total_steps} total "
                  f"(estimate was {epochs * est_windows // batch_size}), "
                  f"checkpoint+gate every {ckpt_every} steps", flush=True)

    torch.save({"jepa": model.state_dict(), "aux": heads.state_dict(),
                "opt": opt.state_dict(),
                **({"invdyn": inv.state_dict()} if inv is not None else {}),
                "cfg_variant": cfg.loss.variant,
                "sigreg_lambda": cfg.loss.sigreg_lambda,
                "ema_target": ema_target, "stop_grad": stop_grad_target,
                "epoch": epochs - 1, "step": step,
                "total_steps": total_steps, "epochs": epochs,
                "complete": True}, out_weights)
    mw.close()

    last = hist[-1] if hist else {}
    return {"params": params["total"], "epochs": epochs, "steps": step,
            "gate_by_epoch": gate_hist,
            "unhealthy_checkpoints": [g["frac"] for g in gate_hist if not g.get("healthy", True)],
            "run_dir": str(run_dir), "weights": str(out_weights),
            "final_pred_loss": last.get("pred_loss"),
            "final_sigreg_loss": last.get("sigreg_loss"),
            "final_eff_dim": last.get("eff_dim"),
            "history": hist}


def dryrun_verdict(summary: dict, batch_size: int, cfg=CFG) -> dict:
    """
    The pass/fail decision, as a pure function of the logged history so it can be
    tested without a GPU. Kept separate on purpose: the first version of this
    logic had a hole that a real run walked through, and a gate you cannot unit
    test is a gate you find out about later.
    """
    h = summary["history"]
    if len(h) < 3:
        raise SystemExit(f"dry run produced only {len(h)} logged steps")
    n = max(1, len(h) // 3)

    def mean(k, rows):
        return sum(r[k] for r in rows) / len(rows)

    first_pred, last_pred = mean("pred_loss", h[:n]), mean("pred_loss", h[-n:])
    first_sr, last_sr = mean("sigreg_loss", h[:n]), mean("sigreg_loss", h[-n:])
    last_ratio = mean("sigreg_ratio", h[-n:])
    last_ed = mean("eff_dim", h[-n:])
    ed_null = h[-1].get("eff_dim_null", 1.0)
    extra = {f"{k}_last": mean(k, h[-n:]) for k in h[-1]
             if k.startswith(("gain_", "mse_")) and k in h[0]}

    res = {
        "steps": summary.get("steps"), "params": summary.get("params"),
        "batch_size": batch_size, "sigreg_lambda": cfg.loss.sigreg_lambda,
        "variant": cfg.loss.variant,
        "pred_loss_first": first_pred, "pred_loss_last": last_pred,
        "sigreg_loss_first": first_sr, "sigreg_loss_last": last_sr,
        "sigreg_ratio_last": last_ratio,
        "eff_dim_first": h[0]["eff_dim"], "eff_dim_last": last_ed,
        "eff_dim_null": ed_null, "eff_dim_frac_of_null": last_ed / max(1e-9, ed_null),
        "pred_loss_moved": bool(last_pred < first_pred * 0.95),
        "sigreg_loss_moved": bool(abs(last_sr - first_sr) > max(1e-6, first_sr * 0.05)),
        # NOT an absolute threshold, and not a fraction of the MP null either.
        # Measured: an UNTRAINED encoder starts at eff_dim 7.9 of a null of 27
        # (batch 32) and 11.1 of 77 (batch 128) — every Mario frame is visually
        # similar, so a random ViT already produces a near-degenerate latent.
        # "> 0.5 x null" therefore demanded more than initialisation provides and
        # no run could ever pass it. What a dry run can honestly ask is that
        # training does not make the rank WORSE, and that it is not rank-1.
        "not_collapsed": bool(last_ed >= 0.8 * h[0]["eff_dim"] and last_ed > 4.0),
        "eff_dim_vs_init": last_ed / max(1e-9, h[0]["eff_dim"]),
        "regularised": bool(last_ratio < 5.0),
        "run_dir": summary.get("run_dir"), **extra,
    }
    res["passed"] = bool(res["pred_loss_moved"] and res["not_collapsed"]
                         and res["regularised"])
    return res


def dry_run(data_dir, steps=300, batch_size=32, cfg=CFG, num_workers=0, lr=3e-4,
            device=None, run_dir=None, stop_grad_target=False):
    """
    Stage 2's exit check for "does this thing learn, WITHOUT collapsing".

    THE FIRST VERSION OF THIS GATE HAD A HOLE and a real run walked straight
    through it. It asserted only that pred_loss and sigreg_loss "moved", and a
    collapsing run satisfies both: pred_loss 0.975 -> 0.061 (moved), sigreg
    0.030 -> 0.098 (moved), eff_dim 9.96 -> 1.17 (every latent identical). The
    prediction task is trivially solvable by mapping every frame to one point,
    so "the loss went down" is evidence of nothing on its own.

    Three conditions now, and each corresponds to a way the run is actually
    broken:

      learning     pred_loss falls by >5%.
      not collapsed  eff_dim stays above half its own Marchenko-Pastur null.
                   Half, not "near 192": the null at batch B is B*D/(B+D), which
                   is 27 at B=32, so anything phrased in absolute terms is a
                   threshold on the batch size rather than on the model.
      regularised  final sigreg_ratio < 5. Ratio, not raw value, because the raw
                   floor is 0.51914/B.

    sigreg is deliberately NOT required to decrease. It legitimately rises early
    while the encoder is still moving, and demanding a decrease fails healthy runs.
    """
    run_dir = Path(run_dir or (cfg.run_dir / "dryrun"))
    summary = train_jepa(data_dir, out_weights=run_dir / "dryrun.pt",
                         epochs=1, batch_size=batch_size, lr=lr,
                         precision="bf16", cfg=cfg, run_dir=run_dir,
                         num_workers=num_workers, max_steps=steps, log_every=5,
                         val_trajectories=2, device=device,
                         stop_grad_target=stop_grad_target)
    res = dryrun_verdict(summary, batch_size=batch_size, cfg=cfg)
    res["stop_grad_target"] = bool(stop_grad_target)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "dryrun.json").write_text(json.dumps(res, indent=2))
    verdict = "PASS" if res["passed"] else "FAIL: " + ", ".join(
        k for k in ("pred_loss_moved", "not_collapsed", "regularised") if not res[k])
    print(json.dumps(res, indent=2))
    print(f"\ndry run {verdict}   "
          f"pred {res['pred_loss_first']:.4f}->{res['pred_loss_last']:.4f}   "
          f"gain_5 {res.get('gain_5step_last', float('nan')):.3f}   "
          f"eff_dim {res['eff_dim_last']:.1f}/{res['eff_dim_null']:.0f} null   "
          f"sigreg {res['sigreg_ratio_last']:.1f}x null")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(CFG.data.data_dir))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--variant", default=None, choices=[None, "aux", "pure"])
    ap.add_argument("--sigreg-lambda", type=float, default=None)
    ap.add_argument("--stop-grad", action="store_true",
                    help="stop-gradient on the target latents (ablation vs SIGReg alone)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.variant:
        CFG.loss.variant = a.variant
    if a.sigreg_lambda is not None:
        CFG.loss.sigreg_lambda = a.sigreg_lambda

    if a.dry_run:
        dry_run(a.data_dir, steps=a.steps, batch_size=a.batch_size,
                num_workers=a.num_workers, lr=a.lr, stop_grad_target=a.stop_grad)
        return
    out = a.out or (CFG.run_dir / f"jepa_{CFG.loss.variant}_lam{CFG.loss.sigreg_lambda}.pt")
    s = train_jepa(a.data_dir, out, epochs=a.epochs, batch_size=a.batch_size,
                   lr=a.lr, num_workers=a.num_workers, stop_grad_target=a.stop_grad)
    print(json.dumps({k: v for k, v in s.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
