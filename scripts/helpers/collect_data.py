"""
Stage 1: DATA. Collect (frame, buttons, ram_state) trajectories from
gym-super-mario-bros and write them to .npz shards.

    python scripts/helpers/collect_data.py --policy random --episodes 4000
    python scripts/helpers/collect_data.py --policy ppo --episodes 4000 --ppo-ckpt path.zip

RAM state is FREE ground truth for x, y, scroll and alive — the exact quantities
LeMario had to reverse-engineer with a probe after planning failed. Extraction
lives in src/dataset/ram.py (ONE definition, shared with gates.py) and refuses to
run until scripts/helpers/validate_ram.py has locked the addresses.

Changes vs. the original draft, all load-bearing:
  * random_policy actually biases toward right/run/jump. The draft built a
    uniform distribution (`p = np.ones(n)`) despite the comment, so it would
    have collected mostly idle/left frames and undercovered jump arcs — the
    single behaviour the y-probe fix depends on.
  * shards stream to disk as flat arrays + episode offsets instead of an
    object-dtype array of dicts. Object arrays need allow_pickle at load and
    can't be memory-mapped; at 783 GB of pixels that matters.
  * shard_size dropped 256 -> 16 (see config): 256 PPO episodes is ~42 GB
    resident before the first write, which OOMs the container.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataset import ram as R
from algo.config import CFG, BUTTONS

OUTCOME_OTHER, OUTCOME_DEATH = 0, 1     # per-episode termination reason


def preprocess(frame, size):
    """
    NES native is 256x240. We resize straight to size x size, which stretches
    the aspect ratio slightly (256:240 -> 1:1). That is deliberate and constant
    across every frame, so the encoder sees one consistent geometry; cropping
    instead would cut off Mario near the screen edges.
    """
    import cv2
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def random_policy(combos):
    """
    Bias toward right + run + jump. Uniform random over COMPLEX_MOVEMENT wastes
    most steps standing still or walking left, and barely produces jump arcs —
    which is exactly the vertical coverage the y-probe needs.
    """
    w = np.ones(len(combos), dtype=np.float64) * 0.25
    for i, c in enumerate(combos):
        s = set(c)
        if "right" in s:
            w[i] = 2.0
            if "B" in s:
                w[i] += 1.5           # running right
            if "A" in s:
                w[i] += 2.5           # jumping right — the coverage we care about
        elif "A" in s:
            w[i] = 1.0                # standing jump
        elif "left" in s:
            w[i] = 0.3
    w /= w.sum()
    return lambda obs: int(np.random.choice(len(combos), p=w)), w


def load_ppo(ckpt):
    """
    The PPO track. Returns dataset.ppo.PPOPolicy, which carries the SAME
    grayscale/resize/frame-stack the policy was trained under.

    The draft here called model.predict() on the raw (240,256,3) RGB frame. SB3
    would not have raised — it reshapes whatever it is given — the policy would
    just have been reading noise, and 4000 GPU-funded episodes would have come
    back indistinguishable from the free random ones. Preprocessing has exactly
    one definition, in src/dataset/ppo.py, imported by both sides.
    """
    from dataset.ppo import PPOPolicy
    return PPOPolicy(ckpt, deterministic=False)


def save_shard(out_dir: Path, idx: int, episodes: list):
    """
    Flat concatenated arrays + ep_offsets. Memmap-friendly, no pickle.
      frames (T,H,W,3) u8 | actions (T,skip,6) u8 | world_x/world_y/scroll i32
      alive/power u8       | ep_offsets (n_eps+1,) i64
    """
    if not episodes:
        return None
    off = np.zeros(len(episodes) + 1, dtype=np.int64)
    for i, e in enumerate(episodes):
        off[i + 1] = off[i] + len(e["frames"])
    pack = dict(
        frames=np.concatenate([e["frames"] for e in episodes]),
        actions=np.concatenate([e["actions"] for e in episodes]),
        ep_offsets=off,
    )
    for k, dt in (("world_x", np.int32), ("world_y", np.int32), ("scroll", np.int32),
                  ("alive", np.uint8), ("power", np.uint8), ("dies_in_5", np.uint8)):
        pack[k] = np.concatenate([e[k] for e in episodes]).astype(dt)
    pack["outcome"] = np.array([e["outcome"] for e in episodes], dtype=np.uint8)

    path = out_dir / f"shard_{idx:04d}.npz"
    np.savez_compressed(path, **pack)
    mb = path.stat().st_size / 1e6
    print(f"  wrote {path.name}  {len(episodes)} eps, {off[-1]} obs, {mb:.0f} MB", flush=True)
    return path


def collect(args):
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from tqdm import trange

    if not R.validated():
        raise SystemExit(
            "RAM addresses are not validated. Run:\n"
            "    python scripts/helpers/validate_ram.py --lock\n"
            "Collecting now would silently corrupt every gate and probe."
        )
    chosen = {**R.CHOSEN, **(R.load_lock() or {}).get("chosen", {})}

    size, skip = CFG.data.frame_size, CFG.data.frame_skip
    # data_dir defaults to ~/.aqmario/data, which inside a Modal container is
    # the container's own filesystem, NOT the mounted Volume — shards written
    # there vanish when the container exits. --out-dir points at /data.
    base = Path(args.out_dir).expanduser() if args.out_dir else CFG.data.data_dir
    out_dir = base / args.policy
    out_dir.mkdir(parents=True, exist_ok=True)

    env = JoypadSpace(gym_super_mario_bros.make(CFG.data.level), COMPLEX_MOVEMENT)

    # map each COMPLEX_MOVEMENT combo to our 6-bit button vector (order from config)
    action_bits = np.array(
        [[1 if b in combo else 0 for b in BUTTONS] for combo in COMPLEX_MOVEMENT],
        dtype=np.uint8)

    if args.policy == "random":
        pick, w = random_policy(COMPLEX_MOVEMENT)
        top = np.argsort(-w)[:3]
        print("action bias:", ", ".join(f"{'+'.join(COMPLEX_MOVEMENT[i])}={w[i]:.2f}" for i in top))
    elif args.policy == "ppo":
        pick = load_ppo(args.ppo_ckpt)
    else:
        raise ValueError(args.policy)

    min_len = CFG.data.obs_per_sample + CFG.model.pred_horizon
    shard, shard_idx, kept, dropped = [], args.shard_start, 0, 0

    for _ in trange(args.episodes, desc=f"{args.policy}"):
        obs = env.reset()
        if hasattr(pick, "reset"):
            pick.reset(obs)          # PPO's frame stack is per-episode state
        frames, actions, states = [], [], []
        done, step = False, 0
        while not done and step <= args.max_steps:
            a = pick(obs)
            block = []
            for _ in range(skip):
                obs, _, done, _ = env.step(a)
                block.append(action_bits[a])
                if done:
                    break
            while len(block) < skip:                    # pad short final block
                block.append(np.zeros(len(BUTTONS), dtype=np.uint8))
            frames.append(preprocess(obs, size))
            actions.append(np.stack(block))
            states.append(R.ram_state(env.unwrapped.ram, chosen))
            step += 1

        if len(frames) < min_len:
            dropped += 1
            continue
        kept += 1
        alive = np.array([s["alive"] for s in states], dtype=np.uint8)
        # dies_in_5 must be computed once the whole episode is known: raw `alive`
        # is 0 on ~0.007% of frames (the episode ends the instant Mario dies), so
        # a BCE head on it learns "always alive". Rolling it 5 observations back
        # turns a coincident signal into the predictive one the gate needs.
        dies = R.dies_within(alive, 5)
        shard.append(dict(
            frames=np.stack(frames),
            actions=np.stack(actions),
            alive=alive,
            dies_in_5=dies,
            outcome=OUTCOME_DEATH if not alive[-1] else OUTCOME_OTHER,
            **{k: np.array([s[k] for s in states]) for k in
               ("world_x", "world_y", "scroll", "power")},
        ))
        if len(shard) >= CFG.data.shard_size:
            save_shard(out_dir, shard_idx, shard)
            shard, shard_idx = [], shard_idx + 1

    save_shard(out_dir, shard_idx, shard)
    env.close()
    print(f"kept {kept} episodes, dropped {dropped} (shorter than {min_len} obs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "ppo"], default="random")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--ppo-ckpt", type=str, default=None)
    ap.add_argument("--shard-start", type=int, default=0,
                    help="offset so parallel Modal containers don't collide")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="write shards here instead of CFG.data.data_dir (use /data on Modal)")
    args = ap.parse_args()
    t0 = time.time()
    collect(args)
    print(f"done in {(time.time()-t0)/60:.1f} min")
