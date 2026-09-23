"""
Stage 4: play World 1-1 with the world model, and steer it with an SAE feature.

    python scripts/helpers/run_plan.py --ckpt <jepa.pt> --episodes 3
    modal run scripts/helpers/modal_app.py::stage4_plan --ckpt /runs/<tag>/jepa.pt

The loop: encode the last `history_len` real frames, run categorical CEM over
macro-action sequences ENTIRELY in latent space, execute the first couple of
macros on the real emulator, repeat. The emulator is the referee, never the
planner's search space.

THREE THINGS THIS SCRIPT REFUSES TO FAKE, because each is a way a planning demo
can look like it works and not:

* THE PROBES ARE FIT ON HELD-OUT TRAJECTORIES AND FROZEN, using exactly the
  gates' split. The planner's cost is probe-scored, so a probe fit on the frames
  it is about to plan through would be scoring itself.

* THE NEARBY-GOAL SANITY PLAN RUNS FIRST. Before claiming anything about
  clearing the level, ask the planner for a goal ~200 px ahead — inside the
  range the aliasing gate certified — and check it gets there. A planner that
  fails that has no business being pointed at the flagpole, and a planner that
  clears the level while failing it got there by accident.

* PROGRESS IS world_x FROM RAM, not the planner's own belief. The model's
  opinion of how far it got is exactly the thing under test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from algo import data as D
from dataset import ram as R
from algo.config import CFG, MACRO_ACTIONS, MACRO_NAMES
from algo.gates import _fit_probe, encode_episodes
from algo.model import load_jepa
from algo.plan import MacroCEM, PlanConfig, steering_effect

FLAGPOLE_X = 3161


def fit_frozen_probes(model, data_dir, cfg=CFG, device="cpu", seed=0,
                      max_obs=400, n_train_traj=40):
    """Probes on held-out trajectories, using the gates' own split, then frozen."""
    paths = D.shard_paths(Path(data_dir))
    train_eps, _ = D.split_episodes(paths, cfg.gate.probe_trajectories, seed=seed)
    cache = encode_episodes(model, train_eps[:n_train_traj], cfg, device,
                            max_obs_per_episode=max_obs)
    probes = {}
    for key, binary in (("world_x", False), ("world_y", False), ("dies_in_5", True)):
        y = torch.from_numpy(cache[key])
        pw = None
        if binary:
            r = max(1e-6, float(y.mean()))
            pw = torch.tensor(max(1.0, (1 - r) / r))
        p, _ = _fit_probe(cache["z"], y, cache["z"][:8], y[:8],
                          binary=binary, pos_weight=pw, device=device, epochs=40)
        for q in p.parameters():
            q.requires_grad_(False)
        probes[key] = p.eval()
    return probes


def make_env():
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    return JoypadSpace(gym_super_mario_bros.make(CFG.data.level), COMPLEX_MOVEMENT)


def _macro_to_env_action(name, combos):
    """Nearest COMPLEX_MOVEMENT combo to one of our 5 macros."""
    want = set(b for b, on in zip(["left", "right", "up", "down", "A", "B"],
                                  MACRO_ACTIONS[name]) if on)
    best, bi = -1, 0
    for i, c in enumerate(combos):
        s = set(c) - {"NOOP"}
        score = len(want & s) - len(want ^ s)
        if score > best:
            best, bi = score, i
    return bi


def rollout_episode(model, probes, pcfg, max_macros=400, seed=0, record=None,
                    device="cpu", goal_ahead=None):
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    import cv2

    env = make_env()
    obs = env.reset()
    H, size = CFG.model.history_len, CFG.data.frame_size
    prep = lambda o: cv2.resize(o, (size, size), interpolation=cv2.INTER_AREA)
    hist = [prep(obs)] * H
    planner = MacroCEM(model, probes, pcfg, device=device, seed=seed)
    a_map = {n: _macro_to_env_action(n, COMPLEX_MOVEMENT) for n in MACRO_NAMES}

    max_x, done, macros, chosen = 0, False, 0, []
    start_x = int(R._pair(env.unwrapped.ram, 0x006D, 0x0086))
    while not done and macros < max_macros:
        goal = (max_x + goal_ahead) if goal_ahead else None
        idx, names, diag = planner.plan(
            torch.from_numpy(np.stack(hist[-H:])), goal_x=goal)
        for name in names[: pcfg.replan_every]:
            for _ in range(CFG.data.frame_skip):
                obs, _, done, info = env.step(a_map[name])
                if done:
                    break
            hist.append(prep(obs))
            # Cap raised from 900: at max_macros 2000 a 1-1 episode runs past it
            # and the recording stops a third of the way in, so the video shows
            # only the opening of a run that kept going. The gif writer already
            # subsamples to a frame budget, so a larger cap costs memory here and
            # nothing in the output.
            if record is not None and len(record) < 3000:
                record.append((obs.copy(), int(info.get("x_pos", 0)), name))
            macros += 1
            max_x = max(max_x, int(info.get("x_pos", 0)))
            if done:
                break
        chosen += names[: pcfg.replan_every]
    env.close()
    return {"max_world_x": int(max_x), "start_x": int(start_x),
            "macros": macros, "cleared": bool(max_x >= FLAGPOLE_X - 40),
            "macro_hist": {n: chosen.count(n) for n in MACRO_NAMES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=str(CFG.data.data_dir))
    ap.add_argument("--sae", default=None, help="SAE .pt for the steering demo")
    ap.add_argument("--feature", type=int, default=None, help="SAE feature index to steer")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--population", type=int, default=256)
    ap.add_argument("--out-dir", default=str(CFG.run_dir))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = load_jepa(a.ckpt, CFG).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    probes = fit_frozen_probes(model, a.data_dir, device=device)
    pcfg = PlanConfig(horizon=a.horizon, population=a.population)

    # 1. sanity: can it reach a goal INSIDE the certified aliasing range?
    t0 = time.time()
    sanity = rollout_episode(model, probes, PlanConfig(horizon=6, population=128),
                             max_macros=60, device=device, goal_ahead=pcfg.subgoal_px)
    sanity["reached"] = bool(sanity["max_world_x"] - sanity["start_x"] >= pcfg.subgoal_px * 0.6)
    sanity["goal_px"] = pcfg.subgoal_px
    (out / "plan_sanity.json").write_text(json.dumps(sanity, indent=2))
    print(f"[sanity] goal +{pcfg.subgoal_px}px -> advanced "
          f"{sanity['max_world_x'] - sanity['start_x']}px  reached={sanity['reached']}")

    # 2. the real thing
    best, frames = None, []
    for e in range(a.episodes):
        rec = [] if e == 0 else None
        r = rollout_episode(model, probes, pcfg, seed=e, record=rec, device=device,
                            goal_ahead=pcfg.subgoal_px)
        print(f"[plan] episode {e}: max world_x {r['max_world_x']}/{FLAGPOLE_X} "
              f"({r['max_world_x']/FLAGPOLE_X:.0%})  macros={r['macros']}")
        if best is None or r["max_world_x"] > best["max_world_x"]:
            best = r
            if rec:
                frames = rec
    best["seconds"] = round(time.time() - t0, 1)
    (out / "plan_best.json").write_text(json.dumps(best, indent=2))

    # 3. steering, if an SAE feature was supplied
    if a.sae and Path(a.sae).is_file():
        # aquin's SAE checkpoint is {d_model, n_features, state_dict} with
        # state_dict["W_dec"] of shape (n_features, d_model) — VERIFIED against
        # aquin.compute.sae_train, not assumed. A decoder column is therefore a
        # ROW of W_dec, and it is already in our 192-dim latent space, which is
        # exactly why the SAE had to be trained with d_model=192 explicitly.
        blob = torch.load(a.sae, map_location=device, weights_only=False)
        sd = blob.get("state_dict", blob)
        dec = sd.get("W_dec")
        if dec is not None:
            if dec.shape[-1] != CFG.model.latent_dim:
                dec = dec.T
            feat = a.feature if a.feature is not None else int(
                dec.norm(dim=-1).argmax())
            paths = D.shard_paths(Path(a.data_dir))
            _, val = D.split_episodes(paths, CFG.gate.probe_trajectories, seed=0)
            with np.load(val[0][0]) as d:
                W = CFG.model.history_len + CFG.model.pred_horizon
                s0 = val[0][2] + 1
                fr = torch.from_numpy(d["frames"][s0:s0 + W]).unsqueeze(0).to(device)
                ac = torch.from_numpy(d["actions"][s0 + 1:s0 + W]).float().unsqueeze(0).to(device)
            rows = steering_effect(model, fr, ac, dec[feat],
                                   alphas=(-6, -3, 0, 3, 6), probe=probes["world_y"])
            ys = [r["y_px"] for r in rows]
            res = {"feature": feat, "rows": rows,
                   "delta_y": float(max(ys) - min(ys)),
                   "monotone": bool(all(b >= a_ - 1e-6 for a_, b in zip(ys, ys[1:]))
                                    or all(b <= a_ + 1e-6 for a_, b in zip(ys, ys[1:]))),
                   "rollout_jumped": bool(abs(max(ys) - min(ys)) > 8.0)}
            (out / "steer_demo.json").write_text(json.dumps(res, indent=2))
            print(f"[steer] feature {feat}: y moves {res['delta_y']:.1f}px across alpha, "
                  f"monotone={res['monotone']}")

    # 4. the GIF
    if frames:
        from PIL import Image, ImageDraw
        ims = []
        for o, x, name in frames[::2]:
            im = Image.fromarray(o).resize((512, 480), Image.NEAREST)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 452, 512, 480], fill=(0, 0, 0))
            dr.text((8, 460), f"x={x}/{FLAGPOLE_X}   macro={name}", fill=(255, 255, 255))
            ims.append(im)
        gif = out / "plan_demo.gif"
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=50, loop=0)
        print(f"[gif] {gif}  ({len(ims)} frames)")

    print(json.dumps({k: v for k, v in best.items()}, indent=2))


if __name__ == "__main__":
    main()
