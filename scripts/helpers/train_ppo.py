"""
Stage 1, second track: train the noisy PPO explorer that reaches the back half
of World 1-1.

    modal run scripts/helpers/modal_app.py::stage1_ppo --steps 3000000
    python scripts/helpers/train_ppo.py --steps 200000 --n-envs 8      # local sanity

This is NOT trying to be a good Mario agent. It is trying to be a DIFFERENT
distribution from the random policy: one whose episodes contain the pit
sequence, the stair and the flagpole. `--target-x` stops training as soon as the
rolling mean furthest-x crosses it, because a policy that clears the level
deterministically is worse for us than one that dies in interesting places —
it collects the same optimal trajectory 4000 times.

Preprocessing and frame-skip alignment live in src/dataset/ppo.py and are shared
with scripts/helpers/collect_data.py; do not reimplement them here (see that module's
docstring for what breaks if the two drift).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from algo.config import CFG
from dataset.ppo import make_env


class StopAtX:
    """
    SB3 callback: end training when the agent reliably gets deep enough, and
    record the furthest-x curve so the data-coverage claim is evidenced rather
    than asserted.
    """

    def __init__(self, target_x, out_json, window=50):
        from stable_baselines3.common.callbacks import BaseCallback
        self._base = BaseCallback
        self.target_x, self.out_json, self.window = target_x, Path(out_json), window

    def build(self):
        outer = self

        class _CB(outer._base):
            def __init__(self):
                super().__init__()
                self.finals = []

            def _on_step(self) -> bool:
                for info, done in zip(self.locals.get("infos", []),
                                      self.locals.get("dones", [])):
                    if done and "x_pos" in info:
                        self.finals.append(int(info["x_pos"]))
                if len(self.finals) >= outer.window:
                    recent = self.finals[-outer.window:]
                    mean_x = sum(recent) / len(recent)
                    if self.num_timesteps % 20000 < 64:
                        print(f"  [{self.num_timesteps}] mean final x over last "
                              f"{outer.window} eps: {mean_x:.0f} / {outer.target_x}",
                              flush=True)
                    if mean_x >= outer.target_x:
                        print(f"  reached target x {outer.target_x}; stopping early")
                        return False
                return True

            def _on_training_end(self):
                outer.out_json.parent.mkdir(parents=True, exist_ok=True)
                recent = self.finals[-outer.window:] or [0]
                outer.out_json.write_text(json.dumps({
                    "episodes": len(self.finals),
                    "timesteps": int(self.num_timesteps),
                    "final_x_mean": sum(recent) / len(recent),
                    "final_x_max": max(self.finals) if self.finals else 0,
                    "target_x": outer.target_x,
                }, indent=2))

        return _CB()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default=str(CFG.run_dir / "ppo" / "ppo_mario.zip"))
    ap.add_argument("--target-x", type=int, default=2400,
                    help="stop once rolling mean final x clears this (flagpole is 3161)")
    ap.add_argument("--max-steps", type=int, default=3000)
    a = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    env_cls = make_env(max_steps=a.max_steps)
    venv = VecMonitor(SubprocVecEnv([lambda: env_cls() for _ in range(a.n_envs)]))

    model = PPO(
        "CnnPolicy", venv,
        # n_steps 512 x n_envs is ~4k transitions per update: long enough that a
        # full jump-and-land is inside one rollout at frame_skip 2.
        n_steps=512, batch_size=256, n_epochs=4,
        learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.1, ent_coef=0.01,     # ent_coef 0.01, not 0: we want a NOISY policy
        verbose=1,
    )
    cb = StopAtX(a.target_x, Path(a.out).with_name("ppo_coverage.json")).build()
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=cb)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    venv.close()
    print(f"saved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
