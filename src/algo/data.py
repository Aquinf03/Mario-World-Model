"""
Stage 2 data path: .npz shards on disk -> batched training windows.

Shard layout (written by scripts/helpers/collect_data.py):
    frames (T,224,224,3) u8 | actions (T,skip,6) u8 | ep_offsets (n_eps+1,) i64
    world_x/world_y/scroll i32 | alive/power/dies_in_5 u8 | outcome (n_eps,) u8

Three things here are load-bearing and easy to get wrong:

1. ACTION ALIGNMENT. collect_data.py appends `actions[i]` BEFORE observing
   `frames[i]` — so actions[i] is the button block that CAUSED frames[i], not
   the one taken from it. A window frames[s..s+W-1] is therefore conditioned on
   actions[s+1..s+W-1]: W-1 transitions, off by one from the naive slice. Get
   this backwards and the predictor is asked to predict the past.

2. WINDOWS NEVER CROSS EPISODES. ep_offsets exists for exactly this reason. A
   window spanning a reset would teach the predictor that Mario teleports.

3. THE SPLIT IS BY WHOLE TRAJECTORY. Bai's protocol (~60 held-out complete
   trajectories). Splitting by observation instead leaks: consecutive frames are
   near-duplicates, so a random-observation split reports a probe R2 that is
   mostly memorisation and is not comparable to LeMario's published numbers.

MEASURED on the first 40 real episodes, and worth knowing before reading any
gate number:
  * corr(world_x, scroll) = 0.998. Scroll is very nearly an affine function of x
    on 1-1, so scroll-R2 > 0.90 is almost FREE once x is encoded. The scroll
    probe is not evidence that the model resolves the camera — the aliasing gate
    is the one that tests that.
  * random-policy world_x tops out at 1416 of the flagpole's 3161. The back half
    of the level is unseen until the PPO track exists, so probe R2 measured on
    random-only data describes the first 45% of 1-1.
  * dies_in_5 is true on 1.04% of observations. Any BCE head on it needs
    pos_weight or it converges to "always alive" (see losses.aux_heads).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.utils.data as tud
from pathlib import Path

from algo.config import CFG

# ---- Target normalisation ---------------------------------------------------
# Affine maps chosen from the PHYSICAL range of each quantity, not from batch
# statistics, so the mapping is identical for random data, PPO data and eval.
# (offset, scale): value_n = (value - offset) / scale
#
# Measured spread of each normalised target on random-policy data, for calibrating
# the aux lambdas: x_n std 0.15, y_n std 0.36, scroll_n std 0.14. x and scroll
# look "small" only because random play covers 45% of the level; PPO data widens
# them toward unit scale without changing the constants.
NORM = {
    "world_x": (1600.0, 1600.0),   # level spans 0..3161 (flagpole)
    "world_y": (410.0, 60.0),      # SCREEN coord, grows DOWNWARD: 432 ground, 346 apex
    "scroll":  (1600.0, 1600.0),   # tracks world_x on 1-1
}


def normalize(name: str, v):
    off, sc = NORM[name]
    return (v - off) / sc


def denormalize(name: str, v):
    off, sc = NORM[name]
    return v * sc + off


# ---- Shard / episode bookkeeping --------------------------------------------
def shard_paths(data_dir: Path, policy: str | None = None) -> list[Path]:
    """Every shard under data_dir, optionally restricted to one policy subdir."""
    data_dir = Path(data_dir).expanduser()
    roots = [data_dir / policy] if policy else (
        [data_dir] + [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
        if data_dir.is_dir() else [])
    out = []
    for r in roots:
        if r.is_dir():
            out += sorted(r.glob("shard_*.npz"))
    return sorted(set(out))


def episode_table(paths: list[Path], strict: bool = False
                  ) -> list[tuple[Path, int, int, int]]:
    """
    (shard, ep_idx, start, stop) for every episode, read WITHOUT decompressing
    frames — NpzFile inflates per key, so touching ep_offsets alone is cheap.

    PARTIALLY-WRITTEN SHARDS ARE SKIPPED, NOT FATAL. Data collection and training
    share one Modal Volume, so a gate run launched while a collection fan-out is
    still going WILL hit an .npz that is mid-write — np.load raises
    BadZipFile deep inside zipfile and takes the whole run down. That happened,
    and it cost a gate run. A shard that cannot be opened yet is simply not part
    of this epoch; the count of skipped files is printed so a genuinely corrupt
    shard is still visible rather than silently tolerated.
    """
    rows, skipped = [], []
    for p in paths:
        try:
            with np.load(p) as d:
                off = d["ep_offsets"]
        except Exception as e:                    # BadZipFile, EOFError, ValueError
            if strict:
                raise
            skipped.append((p.name, type(e).__name__))
            continue
        for i in range(len(off) - 1):
            rows.append((p, i, int(off[i]), int(off[i + 1])))
    if skipped:
        print(f"[data] skipped {len(skipped)} unreadable shard(s) "
              f"(likely still being written): "
              f"{', '.join(n for n, _ in skipped[:4])}"
              f"{' ...' if len(skipped) > 4 else ''}", flush=True)
    return rows


def split_episodes(paths: list[Path], val_trajectories: int | None = None, seed: int = 0):
    """
    Bai's complete-trajectory split. Returns (train_eps, val_eps) as lists of
    (shard, ep_idx, start, stop). Deterministic in `seed` so the gate always
    scores the same held-out trajectories across runs and across the lambda sweep.
    """
    eps = episode_table(paths)
    n_val = CFG.gate.probe_trajectories if val_trajectories is None else val_trajectories
    n_val = min(n_val, max(1, len(eps) // 5))
    order = np.random.default_rng(seed).permutation(len(eps))
    val = {int(i) for i in order[:n_val]}
    return ([e for i, e in enumerate(eps) if i not in val],
            [e for i, e in enumerate(eps) if i in val])


# ---- Windows ----------------------------------------------------------------
def window_len(cfg=CFG) -> int:
    """
    history_len context observations + pred_horizon targets. This is the real
    sample length; DataConfig.obs_per_sample is only collect_data.py's
    minimum-episode-length guard and is deliberately larger.
    """
    return cfg.model.history_len + cfg.model.pred_horizon


class ShardWindows(tud.IterableDataset):
    """
    IterableDataset over training windows, one shard resident at a time.

    A shard decompresses to ~700-850 MB of frames, so random access across the
    whole dataset is not an option: we shuffle shard order, shuffle windows
    inside each shard, and stream. That is why the Modal training container asks
    for cpu=16 / 64 GB — each loader worker holds one inflated shard.
    """

    def __init__(self, episodes, cfg=CFG, shuffle=True, seed=0, stride=None,
                 death_frac=None):
        super().__init__()
        self.episodes = list(episodes)
        self.cfg = cfg
        self.shuffle = shuffle
        self.seed = seed
        self.stride = stride or getattr(cfg.data, "window_stride", 1)
        self.death_frac = (getattr(cfg.data, "death_window_frac", 0.0)
                           if death_frac is None else death_frac)
        self.W = window_len(cfg)
        # MEASURED BUG: the RNG was seeded with (seed + worker_id) alone, so every
        # epoch replayed the IDENTICAL shard order and the identical window order
        # inside each shard — verified by iterating twice and diffing. The model
        # then sees the same sequence of correlated batches every epoch, which is
        # the one thing shuffling exists to prevent. _epoch advances per pass.
        self._epoch = 0

    def __iter__(self):
        info = tud.get_worker_info()
        eps = self.episodes
        # Every worker in every epoch gets its own stream, and epoch N != epoch 0.
        rng = np.random.default_rng(
            [self.seed, self._epoch, info.id if info else 0])
        self._epoch += 1
        if info:
            # Split by EPISODE, not by shard: with 16 episodes per shard and 8
            # workers, a shard-wise split leaves workers idle whenever the shard
            # count is not a multiple of the worker count.
            eps = eps[info.id::info.num_workers]

        by_shard: dict[Path, list] = {}
        for e in eps:
            by_shard.setdefault(e[0], []).append(e)
        order = list(by_shard)
        if self.shuffle:
            rng.shuffle(order)

        for path in order:
            try:
                with np.load(path) as d:
                    arrs = {k: d[k] for k in
                            ("frames", "actions", "world_x", "world_y", "scroll", "dies_in_5")}
            except Exception as e:      # same mid-write race as episode_table
                print(f"[data] skipping {path.name} mid-iteration: {type(e).__name__}",
                      flush=True)
                continue
            starts = []
            for _, _, s0, s1 in by_shard[path]:
                # start at s0+1: the window needs actions[start+1..], and
                # actions[s0] is the reset->first-frame block, which has no
                # preceding observation to condition on.
                starts += list(range(s0 + 1, s1 - self.W + 1, self.stride))
            starts = self._rebalance_deaths(starts, arrs["dies_in_5"], rng)
            if self.shuffle:
                rng.shuffle(starts)
            for s in starts:
                yield self._sample(arrs, s)
            del arrs

    def _rebalance_deaths(self, starts, dies, rng):
        """
        Duplicate death-containing windows up to `death_frac` of the stream.

        Deaths are ~0.75% of frames and clustered one-per-episode, so uniform
        sampling leaves ~45% of batches with NO positive at all — measured on
        real shards. The dead-in-5 head cannot learn from a signal that is absent
        from half its gradient steps, and the batches that do contain a death
        produce a loss spike that is 48% of the total variance.

        Oversampling, not reweighting: pos_weight already handles the per-label
        imbalance, and raising it further would make the spikes worse, not
        smaller. Windows are duplicated rather than dropped so no non-death data
        is thrown away.
        """
        if not self.death_frac or not starts:
            return starts
        W = self.W
        pos = [s for s in starts if dies[s:s + W].any()]
        if not pos:
            return starts
        want = int(self.death_frac * len(starts))
        if len(pos) >= want:
            return starts
        extra = rng.choice(pos, size=want - len(pos), replace=True)
        return starts + [int(s) for s in extra]

    def _sample(self, a, s):
        W = self.W
        out = {
            "frames": torch.from_numpy(np.ascontiguousarray(a["frames"][s:s + W])),
            # THE OFF-BY-ONE: actions[i] caused frames[i], so the W-1 transitions
            # inside this window are actions[s+1 .. s+W-1].
            "actions": torch.from_numpy(
                np.ascontiguousarray(a["actions"][s + 1:s + W])).float(),
            "dies_in_5": torch.from_numpy(a["dies_in_5"][s:s + W].astype(np.float32)),
        }
        for k in ("world_x", "world_y", "scroll"):
            out[k] = torch.from_numpy(normalize(k, a[k][s:s + W].astype(np.float32)))
        return out


def make_dataset(episodes, cfg=CFG, **kw):
    return ShardWindows(episodes, cfg=cfg, **kw)


def make_loader(episodes, batch_size=64, num_workers=4, cfg=CFG, **kw):
    ds = ShardWindows(episodes, cfg=cfg, **kw)
    return tud.DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True, drop_last=True,
                          persistent_workers=bool(num_workers))
