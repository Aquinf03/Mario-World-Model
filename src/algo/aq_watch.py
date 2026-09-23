"""
Stage 2 adapter: stream training curves into `aquin watch`.

This is one of exactly TWO places aquin's tooling is model-agnostic enough to
use on a 192-dim ViT-JEPA (the other is aq_sae.py). `aquin watch` is a passive
external-metrics observer — it ingests a JSONL and knows nothing about your
architecture, so it fits without any faking.

    from algo.aq_watch import MetricsWriter
    mw = MetricsWriter(run_dir)                 # writes metrics.jsonl
    mw.log(step=i, epoch=e, pred_loss=..., sigreg_loss=..., eff_dim=...)

Then, once (needs `aquin login` first):
    aquin watch init --model gpt2                  -> run_id
    aquin watch ingest --run <id> --file metrics.jsonl --follow --step-field step
    aquin watch <run_id>                            # live tail

Two live failure signatures worth watching for:
  * sigreg_loss plateaus high  -> encoder can't reach Gaussian
  * pred_loss -> 0 while eff_dim collapses -> collapsing despite SIGReg, lambda too low
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def effective_dim(z) -> float:
    """
    Participation ratio of the latent covariance spectrum: (sum L)^2 / sum(L^2).
    A collapse shows up here long before pred_loss looks wrong.

    RAW PR IS BIASED DOWN BY THE BATCH SIZE, badly. Read it without correcting
    and you will report a collapse that is not happening. For n samples of a
    genuinely full-rank D-dim Gaussian, Marchenko-Pastur gives E[L^2] = 1 + D/n,
    so

        E[PR] = n*D / (n + D)

    which is not D unless n >> D. Measured against that formula at D=192:

        n =   64   PR  47.1   formula  48.0
        n =  384   PR 127.6   formula 128.0
        n = 2048   PR 175.5   formula 175.5
        n = 8192   PR 187.6   formula 187.6

    So a training run at batch 64 (n = 64*6 = 384 latents) reads eff_dim 128 on a
    perfectly healthy encoder. Use effective_dim_corrected() to compare across
    batch sizes, or across the lambda sweep if the sweep ever changes batch size.
    """
    import torch
    z = z.detach().float()
    z = z - z.mean(0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    s = ev.sum()
    return float(s * s / ev.pow(2).sum().clamp_min(1e-12))


def effective_dim_null(n: int, d: int) -> float:
    """PR that a full-rank D-dim Gaussian actually produces from n samples."""
    return n * d / (n + d)


def effective_dim_corrected(z) -> float:
    """
    Invert the Marchenko-Pastur bias: PR = n*Deff/(n+Deff)  =>  Deff = PR*n/(n-PR).

    Verified: recovers 192.0 +- 1 from n = 128..8192 on full-rank noise, and
    stays at 7.6 on a rank-8 latent regardless of n. This is the number to put in
    the writeup; the raw PR is the number to watch live.
    """
    pr = effective_dim(z)
    n = int(z.shape[0])
    return float(pr * n / max(1e-6, n - pr))


def effective_dim_windowed(z) -> tuple[float, float, int]:
    """
    PR for a (B, W, D) training batch: computed per frame position and averaged,
    so the Marchenko-Pastur correction uses n = B independent samples rather
    than B*W correlated ones.

    Returns (raw_pr, corrected, n_independent). Measured on a synthetic
    FULL-RANK latent batched as 32 windows x 6 frames, the flattened PR reads
    27.0 against an apparent null of 96 — which looks like a 3.5x collapse and
    is not one.
    """
    import torch
    if z.ndim == 2:
        return effective_dim(z), effective_dim_corrected(z), int(z.shape[0])
    B, W, _ = z.shape
    prs = [effective_dim(z[:, w]) for w in range(W)]
    pr = float(sum(prs) / W)
    return pr, float(pr * B / max(1e-6, B - pr)), int(B)


class MetricsWriter:
    """Append-only JSONL, one object per step. Flushed so `--follow` sees it."""

    def __init__(self, run_dir: str | Path, name: str = "metrics.jsonl"):
        self.path = Path(run_dir).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.path = self.path / name
        self._fh = self.path.open("a", buffering=1)

    def log(self, **row):
        self._fh.write(json.dumps(row, default=float) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def watch_init(model_id: str = "gpt2") -> str | None:
    """
    Register a watch run and return its id. `--model` is a catalog label only —
    aquin never loads it for a watch run, it just tags the stream.
    Requires `aquin login` and an active session.
    """
    r = subprocess.run(["aquin", "watch", "init", "--model", model_id],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[aq_watch] watch init failed: {r.stderr.strip() or r.stdout.strip()}")
        return None
    for tok in r.stdout.split():
        if len(tok) >= 8 and tok.strip().isascii() and "-" in tok:
            return tok.strip()
    print(r.stdout)
    return None


def ingest(run_id: str, metrics_path: str | Path, follow: bool = True,
           step_field: str = "step") -> subprocess.Popen:
    """Start `aquin watch ingest` as a background process tailing metrics.jsonl."""
    cmd = ["aquin", "watch", "ingest", "--run", run_id,
           "--file", str(metrics_path), "--step-field", step_field]
    if follow:
        cmd.append("--follow")
    return subprocess.Popen(cmd)
