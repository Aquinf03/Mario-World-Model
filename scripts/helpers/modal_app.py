"""
Modal entrypoints, one per stage. Verified against modal 1.5.0.

  modal run scripts/helpers/modal_app.py::stage1_data
  modal run scripts/helpers/modal_app.py::stage2_train --variant aux --epochs 12
  modal run scripts/helpers/modal_app.py::stage3_gates --ckpt /runs/aux/ep12.pt
  modal run scripts/helpers/modal_app.py::stage3_sweep          # 3 full runs, one per SIGReg lambda
  modal run scripts/helpers/modal_app.py::stage4_plan --ckpt ...

Compute split: data-gen is emulator-bound so it fans out across CPU containers;
training/gates/SAE/planning are GPU. Everything lands on Volumes so the stages
compose. aq/aquin run alongside as the record-and-inspect layer — they provide
no compute of their own (`aq job run --gpu` is a LOCAL subprocess runner).
"""
import os

import modal
from pathlib import Path

# Two images. Stage 1 is emulator-bound and needs NO torch — pulling the full
# CUDA stack for it made the build ~4 GB and timed out on save. Splitting cuts
# the data-gen image to the emulator plus opencv, so Stage 1 builds in seconds.
EMULATOR_PINS = [
    # Each pin found by a real failure, not caution:
    #   numpy<2      nes-py's _rom.py does uint8 * 2**10, which overflows under
    #                NumPy 2's NEP-50 scalar promotion (OverflowError)
    #   gym==0.25.2  0.26 switched step() to a 5-tuple; nes-py returns 4
    #   nes-py 8.2.1 the version that builds against the above
    "numpy<2", "gym==0.25.2", "nes-py==8.2.1", "gym-super-mario-bros==7.4.0",
]

# Repo root (this file lives at scripts/helpers/modal_app.py).
_REPO = Path(__file__).resolve().parents[2]

cpu_image = (
    modal.Image.debian_slim()
    .pip_install(*EMULATOR_PINS, "opencv-python-headless", "tqdm")
    .add_local_dir(str(_REPO / "src"), "/root/src")
    .add_local_dir(str(_REPO / "scripts"), "/root/scripts")
)

# PPO needs the emulator AND torch AND gymnasium (the API adapter in
# src/dataset/ppo.py bridges nes-py's old 4-tuple to SB3's 5-tuple), but not timm
# or aquin. opencv is the preprocessing.
ppo_image = (
    modal.Image.debian_slim()
    .pip_install(*EMULATOR_PINS, "opencv-python-headless", "tqdm",
                 "torch", "gymnasium", "stable-baselines3", "tensorboard")
    .add_local_dir(str(_REPO / "src"), "/root/src")
    .add_local_dir(str(_REPO / "scripts"), "/root/scripts")
)

gpu_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(*EMULATOR_PINS, "opencv-python-headless", "tqdm",
                 "torch", "timm", "stable-baselines3", "aquin")
    .add_local_dir(str(_REPO / "src"), "/root/src")
    .add_local_dir(str(_REPO / "scripts"), "/root/scripts")
)

# App name is overridable so a NEW experiment can be deployed while existing runs
# are still in flight. `modal deploy` replaces a deployed app's function
# definitions, and pushing new code onto the app that is currently 55% through
# two paid H100 runs is not a risk worth taking for a config flag -- deploy the
# new arm as its own app against the SAME volumes instead:
#     AQMARIO_APP=aq-mario-ema modal deploy scripts/helpers/modal_app.py
app = modal.App(os.environ.get("AQMARIO_APP", "aq-mario"), image=gpu_image)
data_vol = modal.Volume.from_name("aqmario-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("aqmario-runs", create_if_missing=True)
VOLS = {"/data": data_vol, "/runs": runs_vol}

# aquin watch ingest + sae train both call require_active_session, so Stage 2/3
# containers need the CLI token. Create with:
#   modal secret create aquin-token AQUIN_TOKEN=aq-...
# Stage 1 does not need it, and a missing Stage 2 dependency must not block
# Stage 1 — so resolve it lazily and degrade to no secret.
def _aquin_secret():
    try:
        s = modal.Secret.from_name("aquin-token")
        s.hydrate()
        return [s]
    except Exception:
        print("[modal_app] aquin-token secret absent; aquin watch/SAE sync disabled")
        return []


AQUIN = _aquin_secret()


# --- Stage 1: ground truth -- CPU, fan out ----------------------------------
@app.function(image=cpu_image, volumes=VOLS, cpu=2.0, timeout=60 * 60, max_containers=64)
def collect_shard(policy: str, seed: int, episodes: int):
    import subprocess, sys
    subprocess.run([sys.executable, "/root/scripts/helpers/collect_data.py",
                    "--policy", policy, "--episodes", str(episodes),
                    "--shard-start", str(seed * 1000),
                    "--out-dir", "/data"], check=True)
    data_vol.commit()


@app.local_entrypoint()
def stage1_data(episodes_per_shard: int = 250, shards: int = 16, policy: str = "random"):
    """
    Fan out data collection. Default is the random track only: the ppo track
    needs a trained SB3 checkpoint that scripts/helpers/train_ppo.py has not produced
    yet, and random alone does not reach the back half of 1-1 (see DOCS.md Part VI).
    """
    jobs = [(policy, s, episodes_per_shard) for s in range(shards)]
    list(collect_shard.starmap(jobs))
    print(f"stage 1: {shards * episodes_per_shard} {policy} episodes requested")


# The tracker probes local state and knows nothing about Modal Volumes, so
# without this it reported "3 shards / 48 episodes" while 4000 episodes sat in
# aqmario-data. This writes a manifest it can read.
@app.function(image=cpu_image, volumes=VOLS, cpu=2.0, timeout=30 * 60)
def _data_stats():
    import numpy as np
    from pathlib import Path as P
    out = {}
    total_bytes = 0
    for sub in sorted(P("/data").iterdir()):
        if not sub.is_dir():
            continue
        shards = sorted(sub.glob("shard_*.npz"))
        eps = obs = 0
        for s in shards:
            total_bytes += s.stat().st_size
            with np.load(s) as d:                 # ep_offsets only: no frame inflation
                off = d["ep_offsets"]
            eps += len(off) - 1
            obs += int(off[-1])
        out[sub.name] = {"shards": len(shards), "episodes": int(eps), "observations": int(obs)}
    return {"by_policy": out, "gb": total_bytes / 1e9}


@app.local_entrypoint()
def data_stats():
    import json, time
    from pathlib import Path
    res = _data_stats.remote()
    res["taken_at"] = time.time()
    res["stale_hours"] = 0.0
    d = Path("~/.aqmario/data").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    (d / "modal_manifest.json").write_text(json.dumps(res, indent=2))
    for k, v in sorted(res["by_policy"].items()):
        print(f"  {k:<8} {v['shards']:>4} shards  {v['episodes']:>5} episodes  "
              f"{v['observations']:>9} observations")
    print(f"  total {res['gb']:.1f} GB on the volume")


# The PPO explorer. A10G, not H100: PPO here is bounded by 8 emulator processes
# on the CPU side, so the GPU is mostly idle no matter which one you rent.
@app.function(image=ppo_image, volumes=VOLS, gpu="A10G", cpu=8.0, memory=16384,
              timeout=6 * 60 * 60)
def train_ppo(steps: int = 2_000_000, n_envs: int = 8, target_x: int = 2400):
    import json, subprocess, sys
    from pathlib import Path as P
    subprocess.run([sys.executable, "/root/scripts/helpers/train_ppo.py",
                    "--steps", str(steps), "--n-envs", str(n_envs),
                    "--target-x", str(target_x),
                    "--out", "/runs/ppo/ppo_mario.zip"], check=True)
    runs_vol.commit()
    cov = P("/runs/ppo/ppo_coverage.json")
    return json.loads(cov.read_text()) if cov.is_file() else {}


@app.local_entrypoint()
def stage1_ppo(steps: int = 2_000_000, n_envs: int = 8, target_x: int = 2400):
    import json
    from pathlib import Path
    res = train_ppo.remote(steps=steps, n_envs=n_envs, target_x=target_x)
    # Mirror locally: the tracker probes real state and cannot see Modal Volumes,
    # so without this it reports "PPO not trained" after a successful run.
    out = Path("~/.aqmario/runs/ppo").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "ppo_coverage.json").write_text(json.dumps(res, indent=2))
    print(res)


# collect_shard for the ppo track needs the SB3 checkpoint, so it runs on
# ppo_image (which has torch) rather than the lean cpu_image.
@app.function(image=ppo_image, volumes=VOLS, cpu=2.0, timeout=60 * 60, max_containers=64)
def collect_shard_ppo(seed: int, episodes: int, ckpt: str = "/runs/ppo/ppo_mario.zip"):
    import subprocess, sys
    subprocess.run([sys.executable, "/root/scripts/helpers/collect_data.py",
                    "--policy", "ppo", "--episodes", str(episodes),
                    "--shard-start", str(seed * 1000),
                    "--ppo-ckpt", ckpt, "--out-dir", "/data"], check=True)
    data_vol.commit()


@app.local_entrypoint()
def stage1_data_ppo(episodes_per_shard: int = 250, shards: int = 16):
    list(collect_shard_ppo.starmap([(s, episodes_per_shard) for s in range(shards)]))
    print(f"stage 1: {shards * episodes_per_shard} ppo episodes requested")


# --- Stage 2: world model -- one GPU ----------------------------------------
def _resolve_resume(resume: str, tag: str) -> str:
    """"" -> fresh, "auto" -> this tag's jepa.pt if it exists, else an explicit path."""
    from pathlib import Path as P
    if resume != "auto":
        return resume
    ck = P(f"/runs/{tag}/jepa.pt")
    if not ck.is_file():
        print(f"[train] resume=auto but no {ck} — starting fresh", flush=True)
        return ""
    import torch
    if torch.load(ck, map_location="cpu", weights_only=False).get("complete"):
        print(f"[train] resume=auto but {ck} is already a COMPLETED run — "
              f"refusing to append to a finished trajectory. Use a new run_tag.",
              flush=True)
        raise SystemExit(0)
    return str(ck)


@app.function(volumes=VOLS, gpu="H100", cpu=16.0, memory=65536,
              timeout=8 * 60 * 60, secrets=AQUIN)
def train(variant: str = "aux", epochs: int = 3, sigreg_lambda: float = 10.0,
          batch_size: int = 96, stop_grad: bool = True, aux_scale: float = 20.0,
          run_tag: str = "", gate_abort: bool = True, resume: str = "",
          ema: float = 0.0, inv_real: float = 0.0, inv_pred: float = 0.0):
    """
    cpu=16 is not optional: at 224px the loader must decompress ~200 GB/epoch,
    and one core decompresses at ~300 MB/s. Fewer cores idles the H100.
    """
    import sys
    sys.path.insert(0, "/root/src")
    from algo.config import CFG
    from algo.train import train_jepa

    # Defaults are the measured winners, not the plan's originals. stop_grad
    # True and sigreg_lambda 10 (not 0.1) is what the dry-run grid says keeps
    # eff_dim rising: at lam 0.1/no-stop-grad the latent goes to rank 2 while
    # gain_5step reads 0.97, which looks like beating LeMario and is not.
    CFG.loss.variant = variant
    CFG.loss.sigreg_lambda = sigreg_lambda
    CFG.loss.aux_scale = aux_scale
    CFG.loss.inv_real_lambda = inv_real
    CFG.loss.inv_pred_lambda = inv_pred
    tag = run_tag or (f"{variant}_lam{sigreg_lambda}_aux{aux_scale:g}_"
                      + (f"ema{ema:g}" if ema > 0 else ('sg' if stop_grad else 'nosg')))
    CFG.run_dir = Path(f"/runs/{tag}")
    s = train_jepa("/data", out_weights=f"/runs/{tag}/jepa.pt", epochs=epochs,
                   batch_size=batch_size, num_workers=12,
                   run_dir=f"/runs/{tag}", cfg=CFG,
                   stop_grad_target=stop_grad,
                   ema_target=ema,
                   inv_dyn=(inv_real > 0 or inv_pred > 0),
                   checkpoint_every_frac=0.10,   # 10 look-ins, not 3
                   gate_grace_frac=0.35,         # probes need ~a third of the run
                   # gate_abort=False keeps every 10% probe MEASURED and logged
                   # but declaws GateAbort. That is right for exactly one thing:
                   # the `pure` control arm. Its y-probe read -0.383 at 4%, so an
                   # armed gate kills it at 35% by design -- and then the open
                   # question ("does a pure JEPA catch up given a full schedule?")
                   # stays open forever, because the only run that could answer it
                   # is the one the gate keeps stopping. The abort exists to save
                   # money on a doomed PRODUCTION run; the control is not one.
                   # Left True everywhere else, where a sinking y-probe is a
                   # regression and not the measurement.
                   gate_abort_y_r2=(0.10 if gate_abort else -1e9),
                   # "auto" = pick up this tag's own last checkpoint if there is
                   # one. Safe as a default because the tag IS the run identity:
                   # a fresh tag has nothing to resume, and a repeated tag is by
                   # definition the same run being continued.
                   resume=_resolve_resume(resume, tag))
    s["ckpt"] = f"/runs/{tag}/jepa.pt"
    # The 10% health table is the thing to look at before trusting anything else.
    print(f"\n{'%':>5}{'y_probe':>10}{'x_probe':>10}{'pred':>9}{'gain5':>8}"
          f"{'eff_dim':>9}{'sigreg':>8}   health")
    for g in s.get("gate_by_epoch", []):
        print(f"{int(g['frac']*100):>4}%{g.get('y_probe_r2', float('nan')):>10.4f}"
              f"{g.get('x_probe_r2', float('nan')):>10.4f}{g.get('pred_loss', float('nan')):>9.4f}"
              f"{g.get('gain_5step', float('nan')):>8.3f}{g.get('eff_dim', float('nan')):>9.1f}"
              f"{g.get('sigreg_ratio', float('nan')):>7.1f}x   "
              f"{'ok' if g.get('healthy') else 'WARN'}")
    s.pop("history", None)
    runs_vol.commit()
    print(s)
    return s


@app.local_entrypoint()
def stage2_train(variant: str = "aux", epochs: int = 3, sigreg_lambda: float = 10.0,
                 batch_size: int = 96, stop_grad: bool = True, aux_scale: float = 20.0,
                 run_tag: str = "", gate_abort: bool = True, resume: str = "auto",
                 ema: float = 0.0, inv_real: float = 0.0, inv_pred: float = 0.0,
                 wait: bool = False):
    """
    SPAWN, NOT REMOTE. `.remote()` blocks this client for the whole run and a
    client kill cancels the call server-side: both arms of the first full run
    died that way, 11 seconds apart, at steps 6050 and 3340 of 37251, with every
    loss healthy. `--detach` does not save a blocking local entrypoint. `.spawn()`
    hands the work to Modal and returns an id, so the run outlives this process
    by construction rather than by flag.
    """
    fc = train.spawn(variant=variant, epochs=epochs, sigreg_lambda=sigreg_lambda,
                     batch_size=batch_size, stop_grad=stop_grad, aux_scale=aux_scale,
                     run_tag=run_tag, gate_abort=gate_abort, resume=resume, ema=ema,
                     inv_real=inv_real, inv_pred=inv_pred)
    tag = run_tag or f"{variant}_lam{sigreg_lambda}_aux{aux_scale:g}_{'sg' if stop_grad else 'nosg'}"
    print(f"SPAWNED {tag}  function_call_id={fc.object_id}")
    print(f"  logs:   modal app logs {app.app_id}")
    print(f"  resume: modal run scripts/helpers/modal_app.py::stage2_train --run-tag {tag} "
          f"--variant {variant}   (resume=auto picks up /runs/{tag}/jepa.pt)")
    if wait:
        s = fc.get()
        print(s)
        print(f"\nnow gate it:\n  modal run scripts/helpers/modal_app.py::stage3_gates --ckpt {s.get('ckpt')}")


# The dry run is the Stage 2 exit check for "does it learn at all", and it is
# deliberately on an A10G rather than the H100: it is 200 steps on one shard, so
# it is bounded by the data loader, not the matmuls, and the H100 would idle at
# 10x the price.
@app.function(volumes=VOLS, gpu="A10G", cpu=8.0, memory=32768, timeout=60 * 60)
def dryrun(steps: int = 300, batch_size: int = 32, variant: str = "aux",
           sigreg_lambda: float = 0.1, stop_grad: bool = False,
           aux_scale: float = 1.0):
    import json, sys
    sys.path.insert(0, "/root/src")
    from pathlib import Path as P
    from algo.config import CFG
    from algo.train import dry_run

    CFG.loss.variant = variant
    CFG.loss.sigreg_lambda = sigreg_lambda
    CFG.loss.aux_scale = aux_scale
    # PER-CELL run dir. Every cell of a sweep sharing /runs/dryrun means they
    # overwrite each other's dryrun.pt and metrics.jsonl, and whichever container
    # happens to finish last is the checkpoint you end up running gates on —
    # silently, and not necessarily the cell you meant.
    tag = (f"{variant}_b{batch_size}_lam{sigreg_lambda}_aux{aux_scale:g}"
           f"_{'sg' if stop_grad else 'nosg'}")
    rd = P(f"/runs/dryrun/{tag}")
    CFG.run_dir = rd
    res = dry_run("/data", steps=steps, batch_size=batch_size,
                  num_workers=6, cfg=CFG, run_dir=str(rd),
                  stop_grad_target=stop_grad)
    res["aux_scale"] = aux_scale
    res["tag"] = tag
    res["ckpt"] = str(rd / "dryrun.pt")
    runs_vol.commit()
    return res


@app.local_entrypoint()
def stage2_dryrun_sweep(steps: int = 500, batch_size: int = 32,
                        lambdas: str = "1.0", batches: str = "", stop_grads: str = "0",
                        aux_scales: str = "1", variants: str = "aux"):
    """
    The dry run across a grid, in parallel. This is the cheap version of the
    Stage 3 lambda sweep and it answers the only question worth answering before
    spending H100 hours: is there ANY setting at which this does not collapse?

    Three axes, because the first sweep showed lambda alone cannot fix it:
      lambdas    sigreg strength
      batches    SIGReg's collapse sensitivity is mostly a function of BATCH
                 SIZE — measured, a rank-2 latent reads 2.8x null at n=32 and
                 41.8x at n=512, against a healthy 0.4x / 1.0x. At batch 32 the
                 statistic barely separates collapse from health, so it cannot
                 push back on it either.
      stop_grads with a shared encoder and no stop-gradient, MSE(zhat, z_target)
                 is minimised by making z_target CONSTANT — the target path is
                 trainable. LeJEPA claims SIGReg alone suffices; this is the
                 ablation that says whether that holds at our batch size.
    """
    import json
    from pathlib import Path

    lams = [float(x) for x in lambdas.split(",") if x.strip()]
    bs = [int(x) for x in batches.split(",") if x.strip()] or [batch_size]
    sgs = [bool(int(x)) for x in stop_grads.split(",") if x.strip()]
    axs = [float(x) for x in aux_scales.split(",") if x.strip()]
    vrs = [v.strip() for v in variants.split(",") if v.strip()]
    # variant "pure" drops the aux heads from the LOSS entirely (they stay built,
    # for the gates). This is the control that matters for the LeMario
    # comparison: the aux variant trains an explicit MSE(y_head(z), y) term, so
    # measuring "y is decodable from z" on it is partly measuring the thing we
    # asked for. LeMario had no such term.
    grid = [(steps, b, v, l, sg, a)
            for v in vrs for l in lams for b in bs for sg in sgs for a in axs]
    print(f"{len(grid)} cells: variants={vrs} lambdas={lams} batches={bs} "
          f"stop_grads={sgs} aux_scales={axs}")
    res = list(dryrun.starmap(grid))

    out = Path("~/.aqmario/runs").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "dryrun_sweep.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'var':>6}{'lam':>6}{'aux':>6}{'batch':>7}{'sg':>6}{'pred':>18}{'gain_5':>9}"
          f"{'eff_dim':>16}{'sigreg':>10}   verdict")
    for r in sorted(res, key=lambda r: -r["eff_dim_frac_of_null"]):
        ed = f"{r['eff_dim_last']:.1f}/{r['eff_dim_null']:.0f}"
        ed = f"{r['eff_dim_first']:.1f}->{r['eff_dim_last']:.1f}"
        print(f"{r.get('variant','?')[:4]:>6}{r['sigreg_lambda']:>6.2f}{r.get('aux_scale', 1):>6.0f}{r['batch_size']:>7}"
              f"{str(r.get('stop_grad_target'))[0]:>6}"
              f"{r['pred_loss_first']:>8.3f}->{r['pred_loss_last']:<9.3f}"
              f"{r.get('gain_5step_last', float('nan')):>9.3f}"
              f"{ed:>16}{r['sigreg_ratio_last']:>9.1f}x   "
              f"{'PASS' if r['passed'] else 'FAIL ' + ','.join(k[:4] for k in ('not_collapsed','regularised','pred_loss_moved') if not r[k])}")
    # the tracker reads dryrun.json; give it the best PASSING run, else the best
    # eff_dim, so the board never claims a pass that did not happen.
    best = max(res, key=lambda r: (r["passed"], r["eff_dim_frac_of_null"]))
    (out / "dryrun.json").write_text(json.dumps(best, indent=2))
    if best.get("params"):
        (out / "param_count.json").write_text(json.dumps({"total": best["params"]}, indent=2))
    print(f"\nwrote dryrun.json from {best.get('tag')} "
          f"({'PASS' if best['passed'] else 'FAIL'})")
    if best.get("ckpt"):
        print(f"run the gates on it:\n"
              f"  modal run scripts/helpers/modal_app.py::stage3_gates --ckpt {best['ckpt']}")


@app.local_entrypoint()
def stage2_dryrun(steps: int = 400, batch_size: int = 32):
    import json
    from pathlib import Path
    res = dryrun.remote(steps=steps, batch_size=batch_size)
    # Mirror the result into the LOCAL run dir so `python -m algo.tracker`
    # sees the Modal run. The tracker probes real state and does not know about
    # Modal Volumes, so without this the board reports a dry run that happened.
    out = Path("~/.aqmario/runs").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "dryrun.json").write_text(json.dumps(res, indent=2))
    if res.get("params"):
        (out / "param_count.json").write_text(
            json.dumps({"total": res["params"]}, indent=2))
    print(json.dumps(res, indent=2))


# --- Stage 3: gates + inspect -- cheaper GPU --------------------------------
@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=2 * 60 * 60, secrets=AQUIN)
def gates(ckpt: str):
    import json, subprocess, sys
    from pathlib import Path as P
    tag = P(ckpt).parent.name
    out = P(f"/runs/{tag}/gates_{tag}.json")
    before = out.stat().st_mtime if out.is_file() else 0
    r = subprocess.run([sys.executable, "/root/scripts/helpers/run_gates.py",
                        "--ckpt", ckpt, "--data-dir", "/data",
                        "--out", str(out), "--tag", tag])
    runs_vol.commit()
    # exit 1 means a gate FAILED, which is a RESULT and must be returned.
    # Any other exit code is the script crashing, and returning the stale JSON
    # from a previous run would report last week's numbers as this run's — which
    # is exactly what happened once and is far worse than an error.
    if r.returncode not in (0, 1):
        raise RuntimeError(f"run_gates.py crashed with exit {r.returncode}")
    if not out.is_file() or out.stat().st_mtime == before:
        raise RuntimeError(f"{out} was not rewritten — refusing to return stale gate results")
    return json.loads(out.read_text())


@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=2 * 60 * 60, secrets=AQUIN)
def sae(ckpt: str, tag: str = "base", layer: int = 0, max_steps: int = 4000):
    """Dump latents (+ RAM labels) -> chunk_*.pt, then train the 192-dim SAE."""
    import json, sys
    sys.path.insert(0, "/root/src")
    from algo.aq_sae import dump_latents_with_labels, feature_report, train_latent_sae
    d = dump_latents_with_labels(ckpt, out_dir=f"/runs/lat/{tag}", data_dir="/data")
    out = f"/runs/sae/{tag}.pt"
    train_latent_sae(d, tag=tag, layer=layer, output=out, max_steps=max_steps)
    rep = feature_report(out, d)
    rep.pop("corr", None); rep.pop("live_idx", None)      # large, stays in the SAE dir
    runs_vol.commit()
    return {"sae": out, "acts": str(d), **rep}


@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=2 * 60 * 60, secrets=AQUIN)
def sae_lambda_diff(tags: str):
    """tags: comma-separated 'lambda=tag' pairs, e.g. '0.1=aux_lam0.1,10=aux_lam10'."""
    import json, sys
    sys.path.insert(0, "/root/src")
    from algo.aq_sae import lambda_diff
    pairs = dict(p.split("=", 1) for p in tags.split(","))
    saes = {k: f"/runs/sae/{v}.pt" for k, v in pairs.items()}
    acts = {k: f"/runs/lat/{v}" for k, v in pairs.items()}
    out = lambda_diff(saes, acts, output="/runs/sae/lambda_diff.json")
    runs_vol.commit()
    return json.loads(open(out).read())


@app.local_entrypoint()
def stage3_sae_diff(tags: str):
    import json
    from pathlib import Path
    res = sae_lambda_diff.remote(tags)
    o = Path("~/.aqmario/runs/sae").expanduser(); o.mkdir(parents=True, exist_ok=True)
    (o / "lambda_diff.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res.get("per_lambda", {}), indent=2))
    for p in res.get("pairs", []):
        print(f"lam {p['lambda_a']} vs {p['lambda_b']}: "
              f"{p['orphaned_a']}/{p['live_a']} features orphaned, "
              f"y-corr matched {p['mean_y_corr_matched']} vs orphaned {p['mean_y_corr_orphaned']}")


@app.local_entrypoint()
def stage3_gates(ckpt: str):
    import json
    from pathlib import Path
    res = gates.remote(ckpt)
    out = Path("~/.aqmario/runs").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates_modal.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


@app.local_entrypoint()
def stage3_sweep(epochs: int = 12):
    """
    The SIGReg lambda-sweep. NOTE: each lambda is a FULL training run
    (~3.2 h H100, ~$17), not a cheap SAE refit. Modal runs them in parallel,
    so this costs money but no extra wall clock.
    """
    from algo.config import CFG
    lams = CFG.loss.sigreg_sweep
    print(f"launching {len(lams)} full training runs: lambda={lams}")
    ckpts = list(train.starmap([("aux", epochs, l) for l in lams]))
    for c, l in zip(ckpts, lams):
        sae.spawn(c, tag=f"lam{l}", layer=int(l * 100))
    print(ckpts)


# --- Stage 4: control -------------------------------------------------------
@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=60 * 60)
def plan_diag(ckpt: str, n_states: int = 6):
    """
    Decompose the planner's cost, per macro, on REAL held-out states.

    The sanity plan advanced Mario -1 px in 60 macros and spent 50 of them on
    `jump` and `run_left`. Two explanations fit that: the risk term drowning the
    progress term, or the x-probe being too noisy on PREDICTED latents to give
    the progress term any signal. They call for opposite fixes, so this measures
    which it is instead of picking one.

    For each macro, hold it for the whole horizon and report what rollout_cost
    actually computes: progress in px, P(dead), and the two terms in the units
    they are summed in.
    """
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch
    from algo import data as D
    from algo.config import CFG, MACRO_NAMES
    from algo.model import load_jepa
    from algo.plan import MacroCEM, PlanConfig, rollout_cost
    sys.path.insert(0, "/root/src")
    from run_plan import fit_frozen_probes

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_jepa(ckpt, cfg=CFG).to(dev)
    probes = fit_frozen_probes(model, "/data", cfg=CFG, device=dev)
    pcfg = PlanConfig()
    planner = MacroCEM(model, probes, pcfg, device=dev)

    paths = D.shard_paths(Path("/data"))
    _, val = D.split_episodes(paths, CFG.gate.probe_trajectories, seed=0)
    H = CFG.model.history_len
    rows = []
    with np.load(val[0][0]) as d:
        base = val[0][2]
        for si in range(n_states):
            s0 = base + 20 + si * 40
            frames = torch.from_numpy(d["frames"][s0:s0 + H]).to(dev)
            z_ctx = model.encoder(frames.unsqueeze(0))[0]
            for mi, name in enumerate(MACRO_NAMES):
                idx = torch.full((1, pcfg.horizon), mi, dtype=torch.long, device=dev)
                zhat = planner._score(z_ctx, idx, None)
                cost, diag = rollout_cost(zhat, probes, None, CFG, pcfg)
                rows.append(dict(state=si, macro=name,
                                 progress_px=float(diag["progress"][0]),
                                 risk=float(diag["risk"][0]),
                                 death_term=float(pcfg.death_penalty * diag["risk"][0] * 100.0),
                                 cost=float(cost[0])))
    print(f"\n{'macro':<18}{'progress_px':>13}{'P(dead)':>10}{'death_term':>12}{'cost':>10}")
    for name in MACRO_NAMES:
        rs = [r for r in rows if r["macro"] == name]
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{name:<18}{m('progress_px'):>13.1f}{m('risk'):>10.3f}"
              f"{m('death_term'):>12.1f}{m('cost'):>10.1f}")
    prog = [abs(r["progress_px"]) for r in rows]
    dth = [r["death_term"] for r in rows]
    print(f"\nspread of |progress| across macros: {max(prog)-min(prog):.1f} px")
    print(f"spread of death_term across macros:  {max(dth)-min(dth):.1f} px-equivalent")
    print(f"-> the term with the LARGER spread is the one choosing the macro")
    return rows


@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=60 * 60)
def action_sensitivity(ckpt: str, n_states: int = 12):
    """
    Does the PREDICTOR respond to actions, and can the x-probe READ that response?

    plan_diag showed the planner's progress term is nearly macro-independent
    (`wait` +107 px, `run_left` +110.5 px, `run_right` +108.2 px). Two very
    different faults produce that, and they need opposite fixes:

      A. the predictor ignores the action    -> latents under run_right and
                                                run_left are nearly identical
      B. the probe cannot read the action    -> latents DIFFER but the x-probe
                                                maps them to the same number

    So measure both: the latent distance between opposite-action rollouts, in
    units of the latent's own scale, AND what the probe decodes from each.
    A real reference point is needed for "is this distance big", so it is
    compared against the distance between two DIFFERENT real frames.
    """
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch
    from algo import data as D
    from algo.config import CFG, MACRO_NAMES
    from algo.model import load_jepa
    from algo.plan import MacroCEM, PlanConfig
    sys.path.insert(0, "/root/src")
    from run_plan import fit_frozen_probes

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_jepa(ckpt, cfg=CFG).to(dev)
    probes = fit_frozen_probes(model, "/data", cfg=CFG, device=dev)
    pcfg = PlanConfig()
    planner = MacroCEM(model, probes, pcfg, device=dev)
    ri, li = MACRO_NAMES.index("run_right"), MACRO_NAMES.index("run_left")

    paths = D.shard_paths(Path("/data"))
    _, val = D.split_episodes(paths, CFG.gate.probe_trajectories, seed=0)
    H = CFG.model.history_len
    out = []
    with np.load(val[0][0]) as d:
        base = val[0][2]
        for si in range(n_states):
            s0 = base + 20 + si * 25
            fr = torch.from_numpy(d["frames"][s0:s0 + H]).to(dev)
            z_ctx = model.encoder(fr.unsqueeze(0))[0]
            zs = {}
            for nm, mi in (("right", ri), ("left", li)):
                idx = torch.full((1, pcfg.horizon), mi, dtype=torch.long, device=dev)
                zs[nm] = planner._score(z_ctx, idx, None)[0]     # (Hz, D)
            d_act = float((zs["right"] - zs["left"]).norm(dim=-1).mean())
            # reference: distance between two real consecutive-ish latents
            f2 = torch.from_numpy(d["frames"][s0 + 8:s0 + 8 + H]).to(dev)
            z2 = model.encoder(f2.unsqueeze(0))[0]
            d_real = float((z_ctx - z2).norm(dim=-1).mean())
            xr = float(D.denormalize("world_x", probes["world_x"](zs["right"]).squeeze(-1)).mean())
            xl = float(D.denormalize("world_x", probes["world_x"](zs["left"]).squeeze(-1)).mean())
            out.append(dict(state=si, latent_dist_right_vs_left=d_act,
                            latent_dist_real_frames_8_apart=d_real,
                            probe_x_right=xr, probe_x_left=xl,
                            probe_x_gap=xr - xl,
                            true_x=int(d["world_x"][s0 + H - 1])))
    return out


@app.function(volumes=VOLS, gpu="A10G", cpu=8.0, timeout=4 * 60 * 60)
def plan_sweep(ckpt: str, episodes: int = 2, max_macros: int = 400):
    """
    Planner-side ablation on ONE checkpoint. No retraining.

    The model's action signal is +6.07 px of decoded x between run_right and
    run_left, against a true effect of roughly 200. This asks how much of the
    planner's wandering is the weak signal itself and how much is the planner
    mishandling it. Probes are fit ONCE and shared, so every config sees an
    identical model, identical probes and identical seeds -- the only variable is
    the planner.

    horizon 5 matters structurally, not as a tuning knob: pred_horizon is 5, so
    an 8-macro plan is rolled in CHUNKS with predictions fed back as context, and
    the compounding lands exactly where the signal is weakest.
    """
    import sys, json
    from pathlib import Path as P
    sys.path.insert(0, "/root/src")
    import torch
    from algo.config import CFG
    from algo.model import load_jepa
    from algo.plan import PlanConfig
    from run_plan import fit_frozen_probes, rollout_episode

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_jepa(ckpt, cfg=CFG).to(dev)
    probes = fit_frozen_probes(model, "/data", cfg=CFG, device=dev)

    configs = {
        "baseline (h8)":            dict(horizon=8),
        "h5":                       dict(horizon=5),
        "h5 + switch40":            dict(horizon=5, switch_penalty=40.0),
        "h5 + switch40 + deathz":   dict(horizon=5, switch_penalty=40.0,
                                         death_relative=True, death_px=60.0),
        "h5 + switch80 + deathz":   dict(horizon=5, switch_penalty=80.0,
                                         death_relative=True, death_px=60.0),
        "h5 + sw40 + dz + big CEM": dict(horizon=5, switch_penalty=40.0,
                                         death_relative=True, death_px=60.0,
                                         population=512, elites=64, iters=6),
    }
    out = []
    for name, kw in configs.items():
        pcfg = PlanConfig(**kw)
        xs = []
        for ep in range(episodes):
            r = rollout_episode(model, probes, pcfg, max_macros=max_macros,
                                seed=ep, device=dev, goal_ahead=pcfg.subgoal_px)
            xs.append(r["max_world_x"])
        row = dict(config=name, xs=xs, best=max(xs), mean=sum(xs) / len(xs))
        out.append(row)
        print(f"[sweep] {name:<28} best {row['best']:>5}  mean {row['mean']:>7.1f}  {xs}",
              flush=True)
    tag = P(ckpt).parent.name
    (P(f"/runs/{tag}") / "plan_sweep.json").write_text(json.dumps(out, indent=2))
    runs_vol.commit()
    return out


@app.function(volumes=VOLS, gpu="A10G", cpu=8.0, timeout=4 * 60 * 60)
def plan_video(ckpt: str, levels: str = "1-1", episodes: int = 2,
               max_macros: int = 400, tag_suffix: str = ""):
    """
    Plan with the world model across several episodes and levels, and record all
    of it into ONE annotated gif.

    LEVELS OTHER THAN 1-1 ARE OUT OF DISTRIBUTION AND THE OVERLAY SAYS SO. Every
    one of the 8,000 training episodes is World 1-1 (CFG.data.level, "stay on ONE
    level until it works"), and the x / dies-in-5 probes that form the planner's
    cost were fit on 1-1 latents. A run on 1-2 is therefore testing two different
    things at once -- whether the dynamics transfer AND whether the probes do --
    and it is a generalisation probe, not a demo of the system working.
    """
    import json, subprocess, sys
    from pathlib import Path as P
    sys.path.insert(0, "/root/src")
    import numpy as np, torch
    from algo.config import CFG
    from algo.model import load_jepa
    from algo.plan import PlanConfig
    from run_plan import fit_frozen_probes, rollout_episode, FLAGPOLE_X

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_jepa(ckpt, cfg=CFG).to(dev)
    # Probes are fit ONCE, on 1-1, and reused for every level -- deliberately.
    # Refitting per level would need labelled data for that level, which does not
    # exist here, and would also hide the question being asked.
    probes = fit_frozen_probes(model, "/data", cfg=CFG, device=dev)
    pcfg = PlanConfig()

    tag = P(ckpt).parent.name
    out = P(f"/runs/{tag}")
    lv = [x.strip() for x in levels.split(",") if x.strip()]
    results, segments = [], []
    for level in lv:
        CFG.data.level = f"SuperMarioBros-{level}-v0"
        for ep in range(episodes):
            rec = []
            r = rollout_episode(model, probes, pcfg, max_macros=max_macros,
                                seed=ep, record=rec, device=dev,
                                goal_ahead=pcfg.subgoal_px)
            r.update(level=level, episode=ep, n_frames=len(rec))
            results.append(r)
            segments.append((level, ep, rec, r))
            print(f"[video] {level} ep{ep}: x {r['max_world_x']}  macros {r['macros']}",
                  flush=True)

    from PIL import Image, ImageDraw
    ims, budget = [], 1100
    total = sum(max(1, len(rec)) for _, _, rec, _ in segments) or 1
    stride = max(1, total // budget)
    for level, ep, rec, r in segments:
        for o, x, name in rec[::stride]:
            im = Image.fromarray(o).resize((384, 360), Image.NEAREST)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 316, 384, 360], fill=(0, 0, 0))
            ood = "" if level == "1-1" else "  OUT-OF-DISTRIBUTION"
            dr.text((6, 320), f"World {level}  ep{ep}{ood}", fill=(255, 220, 80))
            dr.text((6, 336), f"x={x}  macro={name}", fill=(255, 255, 255))
            ims.append(im)
    name = f"plan_video{tag_suffix}.gif"
    if ims:
        ims[0].save(out / name, save_all=True, append_images=ims[1:],
                    duration=60, loop=0)
        print(f"[video] wrote {out / name} ({len(ims)} frames)", flush=True)
    (out / f"plan_video{tag_suffix}.json").write_text(json.dumps(results, indent=2))
    runs_vol.commit()
    return results


@app.function(volumes=VOLS, gpu="A10G", cpu=4.0, timeout=4 * 60 * 60)
def plan(ckpt: str, sae_path: str = "", episodes: int = 3):
    """Emulator + GPU in one container: CEM rolls out in latent space, the
    chosen macro executes on the real env, repeat."""
    import json, subprocess, sys
    from pathlib import Path as P
    tag = P(ckpt).parent.name
    out = P(f"/runs/{tag}")
    cmd = [sys.executable, "/root/scripts/helpers/run_plan.py", "--ckpt", ckpt,
           "--data-dir", "/data", "--out-dir", str(out),
           "--episodes", str(episodes)]
    if sae_path:
        cmd += ["--sae", sae_path]
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    res = {}
    for name in ("plan_best.json", "plan_sanity.json", "steer_demo.json"):
        f = out / name
        if f.is_file():
            res[name.replace(".json", "")] = json.loads(f.read_text())
    return res


@app.local_entrypoint()
def stage4_plan(ckpt: str, sae_path: str = "", episodes: int = 3):
    import json
    from pathlib import Path
    res = plan.remote(ckpt, sae_path, episodes)
    # Mirror to the local run dir so the tracker can see Stage 4 happened.
    out = Path("~/.aqmario/runs").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        (out / f"{k}.json").write_text(json.dumps(v, indent=2))
    print(json.dumps(res, indent=2))


@app.function(image=cpu_image, volumes=VOLS, cpu=2.0, timeout=15 * 60)
def smoke():
    """Cheap preflight: does the pinned emulator stack actually work in-image?"""
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, gym, gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from dataset import ram as R
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), COMPLEX_MOVEMENT)
    env.reset()
    for _ in range(30):
        out = env.step(3)
    m = env.unwrapped.ram
    x = R._pair(m, 0x006D, 0x0086)
    env.close()
    # Everything returned across the Modal boundary must be a PLAIN python type.
    # `info["x_pos"]` is a numpy scalar, and unpickling it locally requires numpy
    # in the *modal CLI's* env (a uv tool install, not this repo's venv) — which
    # fails with DeserializationError and hides an otherwise successful run.
    return dict(numpy=str(np.__version__), gym=str(gym.__version__),
                step_tuple=int(len(out)), world_x=int(x),
                info_x=int(out[3].get("x_pos", -1)))


@app.local_entrypoint()
def preflight():
    print(smoke.remote())
