"""
The PPO exploration policy — Stage 1's SECOND data track.

Why it exists: the biased random policy tops out at world_x = 1416 of the
flagpole's 3161 (measured on the first 40 episodes). Everything past the first
pipe complex — the pit sequence, the stair, the flagpole approach — is simply
absent from random data. A y-probe R2 measured on that data describes the first
45% of 1-1 and is not comparable to a number measured on the whole level, so
half the dataset comes from a noisy PPO agent that actually gets to the end.

THE PREPROCESSING LIVES HERE, NOT IN THE TRAINING SCRIPT, and that is the whole
point of the module. collect_data.py's --policy ppo calls model.predict() on
whatever it hands over; if that observation is not preprocessed EXACTLY as it
was during PPO training, the policy sees garbage and silently degenerates to
near-random — you get 4000 episodes that took an hour of GPU to produce and are
no better than the free ones. One definition, imported by both sides.

Two alignment details that are easy to get wrong:

* FRAME SKIP MUST MATCH COLLECTION. The PPO env repeats each action for
  CFG.data.frame_skip emulator frames, the same as scripts/helpers/collect_data.py. A
  policy trained at skip-4 and replayed at skip-2 holds every jump for half as
  long, which is exactly the vertical behaviour we are collecting for.

* GYMNASIUM, NOT GYM. nes-py speaks the old 4-tuple gym API (which is why
  gym==0.25.2 is pinned); modern stable-baselines3 speaks gymnasium's 5-tuple.
  MarioEnv is the adapter, and it is 30 lines — much less fragile than pinning
  stable-baselines3 back to 1.6.x to meet nes-py where it stands.
"""
from __future__ import annotations

import numpy as np

from algo.config import CFG

PPO_SIZE = 84            # standard Atari-scale input; the JEPA's 224 is separate
PPO_STACK = 4            # frames of history the policy sees (velocity is not in one frame)


def ppo_obs(frame: np.ndarray) -> np.ndarray:
    """Raw NES RGB (240,256,3) -> (84,84) uint8 grayscale."""
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (PPO_SIZE, PPO_SIZE), interpolation=cv2.INTER_AREA)


class FrameStacker:
    """(PPO_STACK, 84, 84) uint8, oldest first. Channel-first for SB3's CnnPolicy."""

    def __init__(self, stack: int = PPO_STACK):
        self.stack = stack
        self.buf = np.zeros((stack, PPO_SIZE, PPO_SIZE), dtype=np.uint8)

    def reset(self, frame) -> np.ndarray:
        self.buf[:] = ppo_obs(frame)
        return self.buf.copy()

    def push(self, frame) -> np.ndarray:
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = ppo_obs(frame)
        return self.buf.copy()


def make_raw_env():
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    return JoypadSpace(gym_super_mario_bros.make(CFG.data.level), COMPLEX_MOVEMENT)


def make_env(max_steps: int = 3000, seed: int = 0):
    """Gymnasium-API Mario env for stable-baselines3."""
    import gymnasium as gym
    from gymnasium import spaces

    class MarioEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.env = make_raw_env()
            self.skip = CFG.data.frame_skip
            self.stacker = FrameStacker()
            self.action_space = spaces.Discrete(self.env.action_space.n)
            self.observation_space = spaces.Box(
                0, 255, (PPO_STACK, PPO_SIZE, PPO_SIZE), dtype=np.uint8)
            self._steps = 0
            self._max_x = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs = self.env.reset()
            self._steps, self._max_x = 0, 0
            return self.stacker.reset(obs), {}

        def step(self, action):
            r_total, done, info = 0.0, False, {}
            for _ in range(self.skip):
                obs, r, done, info = self.env.step(int(action))
                r_total += float(r)
                if done:
                    break
            self._steps += 1
            # The env's own reward already is (delta-x - time - death). We add a
            # one-off bonus for each new furthest point, because PPO otherwise
            # happily farms the flat opening stretch by oscillating: delta-x
            # rewards are symmetric, a max-x bonus is not.
            x = int(info.get("x_pos", 0))
            if x > self._max_x:
                r_total += 0.5 * (x - self._max_x)
                self._max_x = x
            trunc = self._steps >= max_steps
            return self.stacker.push(obs), r_total / 10.0, bool(done), bool(trunc), info

        def close(self):
            self.env.close()

    return MarioEnv


class PPOPolicy:
    """
    Collection-time wrapper. Holds the frame stack that SB3's VecEnv held during
    training; without it the policy sees four copies of the current frame and
    cannot tell rising from falling.

        pick = PPOPolicy(ckpt)
        pick.reset(obs_after_env_reset)
        a = pick(obs)                # once per observation, same cadence as training
    """

    def __init__(self, ckpt, deterministic: bool = False):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(ckpt), device="cpu")
        self.stacker = FrameStacker()
        self.deterministic = deterministic      # False on purpose: we want coverage, not a speedrun

    def reset(self, frame):
        self.stacker.reset(frame)

    def __call__(self, frame) -> int:
        obs = self.stacker.push(frame)
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)
