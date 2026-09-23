"""
Stage 3: run every gate on one checkpoint, standalone.

    python scripts/helpers/run_gates.py --ckpt ~/.aqmario/runs/aux_lam0.1/jepa.pt
    modal run scripts/helpers/modal_app.py::stage3_gates --ckpt /runs/aux_lam0.1/jepa.pt

Standalone on purpose. The same gates run inside `aq train` through
src/algo/jepa.py:evaluate(), but a gate you can only reach by launching a
training run is a gate you will stop running.

Exit code is 1 when any gate fails, so this is usable in CI and in
`aq schedule`'s resume-on-fail without parsing the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from algo.config import CFG, TARGETS
from algo.gates import run_all_gates


def report(res: dict, cfg=CFG) -> str:
    rows = [
        ("x probe R2", res.get("x_probe_r2"), cfg.gate.min_x_probe_r2, TARGETS["x_probe_r2"]["lemario"]),
        ("y probe R2", res.get("y_probe_r2"), cfg.gate.min_y_probe_r2, TARGETS["y_probe_r2"]["lemario"]),
        ("scroll probe R2", res.get("scroll_probe_r2"), cfg.gate.min_scroll_probe_r2, None),
        ("aliasing margin", res.get("aliasing_margin"), cfg.gate.min_aliasing_margin, None),
    ]
    out = ["", f"gates on {res.get('n')} held-out latents "
               f"({res.get('n_val_trajectories')} complete trajectories)", ""]
    out.append(f"{'gate':<18}{'value':>10}{'threshold':>12}{'LeMario':>10}   ")
    for name, v, thr, lem in rows:
        mark = "PASS" if (v is not None and v == v and v >= thr) else "FAIL"
        vs = "  n/a" if v is None or v != v else f"{v:.4f}"
        out.append(f"{name:<18}{vs:>10}{thr:>12.2f}{(f'{lem:.3f}' if lem else '—'):>10}   {mark}")
    out.append("")
    out.append(f"dead-in-5: AUC {res.get('dead_in_5_auc', float('nan')):.4f}  "
               f"AP {res.get('dead_in_5_ap', float('nan')):.4f}  "
               f"(base rate {res.get('dead_in_5_base_rate', float('nan')):.4f} — "
               f"AP below that is no signal at all)")
    curve = res.get("aliasing_curve")
    if curve:
        out.append("")
        out.append("latent distance vs world-x separation (the aliasing question, unrolled):")
        out.append(f"  {'|dx| px':>12}{'pairs':>9}{'mean dist':>11}")
        for r in curve:
            lbl = f"{r['dx_lo']}-{r['dx_hi']}" if r["dx_hi"] else f"{r['dx_lo']}+"
            out.append(f"  {lbl:>12}{r['n']:>9}{r['mean_dist']:>11.2f}")
        out.append(f"  monotone={res.get('aliasing_monotone')}  "
                   f"spearman={res.get('aliasing_spearman', float('nan')):.3f}  "
                   f"far/near={res.get('aliasing_far_over_near', float('nan')):.2f}x")
    if res.get("aliasing_note"):
        out.append(f"aliasing: {res['aliasing_note']}")
    out.append("")
    out.append("ALL GATES PASS" if res.get("all_pass") else
               "FAILED: " + ", ".join(k for k, v in res.get("passes", {}).items() if not v))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=str(CFG.data.data_dir))
    ap.add_argument("--out", default=None, help="where to write gates_<tag>.json")
    ap.add_argument("--tag", default="last")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-obs", type=int, default=600)
    a = ap.parse_args()

    out = a.out or str(CFG.run_dir / f"gates_{a.tag}.json")
    res = run_all_gates(a.ckpt, a.data_dir, cfg=CFG, seed=a.seed,
                        max_obs_per_episode=a.max_obs, out_json=out)
    print(report(res))
    print(f"\nwrote {out}")
    sys.exit(0 if res.get("all_pass") else 1)


if __name__ == "__main__":
    main()
