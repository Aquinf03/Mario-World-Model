"""
aq method adapter for the AQ-Mario JEPA world model.

Implements recipe.yaml `method: jepa`:
    fit(src: Path, rec: dict) -> dict          # rec carries rec["_train"]
    evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]
    predict(model: dict, X: list) -> list
    write_inspect(train: Path, model: dict) -> str

CHECKPOINT SHAPE: engine/step.py does `json.loads(ckpt.read_text())`, so the
returned dict must be JSON-serialisable. A 15M-param torch model obviously is
not, so fit() writes weights to a .pt beside the checkpoint and returns a
pointer. Every consumer here reloads from that path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

__all__ = ["fit", "predict", "evaluate", "write_inspect"]


def _cfg_from_recipe(rec: dict):
    from algo.config import CFG
    CFG.model.latent_dim = int(rec.get("latent_dim", CFG.model.latent_dim))
    CFG.model.predictor_layers = int(rec.get("predictor_layers", CFG.model.predictor_layers))
    CFG.model.history_len = int(rec.get("history_len", CFG.model.history_len))
    CFG.model.pred_horizon = int(rec.get("pred_horizon", CFG.model.pred_horizon))
    CFG.data.frame_skip = int(rec.get("frame_skip", CFG.data.frame_skip))
    CFG.loss.variant = str(rec.get("variant", CFG.loss.variant))
    CFG.loss.sigreg_lambda = float(rec.get("sigreg_lambda", CFG.loss.sigreg_lambda))
    return CFG


def fit(src: Path, rec: dict) -> dict:
    """
    Train the JEPA. Streams metrics through the kernel's own metrics.jsonl
    (protocol/metrics.py), which is also exactly the file `aquin watch ingest`
    consumes — so one write feeds both tools.
    """
    from algo.train import train_jepa          # Stage 2

    train_dir = Path(rec["_train"])
    cfg = _cfg_from_recipe(rec)
    weights = train_dir / "artifacts" / f"jepa_{cfg.loss.variant}_lam{cfg.loss.sigreg_lambda}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)

    summary = train_jepa(
        data_dir=Path(src),
        out_weights=weights,
        epochs=int(rec.get("epochs", 12)),
        batch_size=int(rec.get("batch_size", 256)),
        lr=float(rec.get("lr", 3e-4)),
        precision=str(rec.get("precision", "bf16")),
        cfg=cfg,
    )
    return {
        "backend": "torch",
        "kind": "jepa-world-model",
        "weights": str(weights.relative_to(train_dir)),
        "variant": cfg.loss.variant,
        "sigreg_lambda": cfg.loss.sigreg_lambda,
        "latent_dim": cfg.model.latent_dim,
        "params": summary.get("params"),
        "train_loss": summary.get("final_pred_loss"),
        "eff_dim": summary.get("final_eff_dim"),
        "epochs": summary.get("epochs"),
    }


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    """
    THE GATE. Returns (gate_score, n). gate_score is the worst of the three
    probe R2s expressed as a fraction of its threshold, so >= 1.0 means every
    gate passed and recipe eval.min_score: 1.0 fails the run otherwise.

    This is what turns LeMario's post-hoc postmortem into tooling: the y-probe
    collapse shows up here at epoch 2 instead of after planning fails.
    """
    from algo.gates import run_all_gates       # Stage 3

    train_dir = Path(rec.get("_train", REPO))
    cfg = _cfg_from_recipe(rec)
    res = run_all_gates(train_dir / model["weights"], data_dir=Path(src), cfg=cfg)

    ratios = [
        res["x_probe_r2"] / cfg.gate.min_x_probe_r2,
        res["y_probe_r2"] / cfg.gate.min_y_probe_r2,
        res["scroll_probe_r2"] / cfg.gate.min_scroll_probe_r2,
        res["aliasing_margin"] / cfg.gate.min_aliasing_margin,
    ]
    (train_dir / "artifacts" / "gates_last.json").write_text(json.dumps(res, indent=2))
    return float(min(ratios)), int(res.get("n", 0))


def predict(model: dict, X: list, rec: dict | None = None) -> list:
    """
    Roll the predictor forward. Used by `aq serve`.

    Two bugs lived here until this was actually executed rather than read:
    `model["weights"]` is stored RELATIVE to the train dir (fit() makes it so, on
    purpose, so a train directory is movable), and it was being opened as if it
    were absolute — FileNotFoundError the first time anyone called it. And
    JEPA.rollout takes (frames, actions); it was being handed one argument.

    X is a list of (frames, actions) pairs, or of frame arrays when the caller
    has no actions to offer, in which case the null action is used throughout.
    """
    import torch

    from algo.model import load_jepa

    train_dir = Path(rec["_train"]) if rec and "_train" in rec else REPO
    w = Path(model["weights"])
    if not w.is_absolute():
        w = train_dir / w
    m = load_jepa(w)
    n_btn = m.cfg.data.frame_skip, 6

    out = []
    for item in X:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            frames, actions = item
        else:
            frames, actions = item, None
        frames = torch.as_tensor(frames)
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        if actions is None:
            actions = torch.zeros(frames.shape[0], m.W - 1, *n_btn)
        out.append(m.rollout(frames, torch.as_tensor(actions).float()).cpu())
    return out


def write_inspect(train: Path, model: dict) -> str:
    g = train / "artifacts" / "gates_last.json"
    res = json.loads(g.read_text()) if g.is_file() else {}
    lines = [
        "# inspect — AQ-Mario JEPA", "",
        f"variant: {model.get('variant')}   sigreg_lambda: {model.get('sigreg_lambda')}",
        f"latent_dim: {model.get('latent_dim')}   params: {model.get('params')}",
        f"final pred_loss: {model.get('train_loss')}   eff_dim: {model.get('eff_dim')}",
        "", "## gates", "",
        "| probe | R2 | threshold | LeMario |",
        "|---|---|---|---|",
        f"| x | {res.get('x_probe_r2')} | 0.95 | 0.997 |",
        f"| y | {res.get('y_probe_r2')} | 0.80 | **0.188** |",
        f"| scroll | {res.get('scroll_probe_r2')} | 0.90 | unmeasured |",
        f"| aliasing margin | {res.get('aliasing_margin')} | 0.10 | — |",
        "",
    ]
    rel = "artifacts/inspect.md"
    (train / rel).write_text("\n".join(lines), encoding="utf-8")
    return rel
