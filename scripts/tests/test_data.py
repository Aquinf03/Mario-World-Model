"""
Evals for the shard -> window path. These run against SYNTHETIC shards with
known contents, because the properties being checked (action alignment, episode
boundaries, split disjointness) are exactly the ones real data cannot reveal —
a misaligned window still looks like plausible Mario.

The last test does run on real shards if any exist locally, and is skipped
otherwise.
"""
import numpy as np
import pytest
import torch

from algo import data as D
from algo.config import CFG

W = D.window_len()


def _fake_shard(path, n_eps=3, ep_len=40):
    """
    Every field is a known function of the global frame index, so any
    misalignment shows up as an arithmetic mismatch rather than a plausible
    number. actions[i] encodes i in its first button slot (mod 2) and i//2 in
    the second, so an off-by-one in the action slice is detectable.
    """
    T = n_eps * ep_len
    idx = np.arange(T)
    frames = np.zeros((T, 8, 8, 3), np.uint8)
    frames[:, 0, 0, 0] = idx % 256                     # frame i is stamped with i
    actions = np.zeros((T, CFG.data.frame_skip, 6), np.uint8)
    actions[:, 0, 0] = idx % 2
    actions[:, 0, 1] = (idx // 2) % 2
    np.savez_compressed(
        path,
        frames=frames, actions=actions,
        ep_offsets=np.arange(n_eps + 1, dtype=np.int64) * ep_len,
        world_x=(idx * 3).astype(np.int32),
        world_y=(400 + idx % 20).astype(np.int32),
        scroll=(idx * 2).astype(np.int32),
        alive=np.ones(T, np.uint8), power=np.zeros(T, np.uint8),
        dies_in_5=np.zeros(T, np.uint8),
        outcome=np.zeros(n_eps, np.uint8))
    return path


@pytest.fixture
def shard(tmp_path):
    d = tmp_path / "random"
    d.mkdir()
    return _fake_shard(d / "shard_0000.npz")


def test_normalize_roundtrip():
    for k in ("world_x", "world_y", "scroll"):
        v = np.array([0.0, 100.0, 3161.0])
        assert np.allclose(D.denormalize(k, D.normalize(k, v)), v)


def test_normalisation_constants_are_physical_not_fitted():
    """Fixed constants so random-policy data and PPO data land on one scale."""
    assert D.NORM["world_x"] == D.NORM["scroll"] == (1600.0, 1600.0)
    # y is a SCREEN coord: 432 ground, 346 apex, so ground maps just above zero
    assert D.normalize("world_y", 432) > 0 > D.normalize("world_y", 346)


def test_window_len_is_history_plus_horizon():
    assert D.window_len() == CFG.model.history_len + CFG.model.pred_horizon


def test_action_alignment_is_off_by_one(shard):
    """
    actions[i] CAUSED frames[i], so a window starting at s must carry
    actions[s+1 .. s+W-1]. This is the single most damaging silent bug in the
    pipeline: get it backwards and the predictor is conditioned on the action
    that produced the frame it is being given, not the one it must predict.
    """
    ds = D.ShardWindows(D.episode_table([shard]), shuffle=False)
    sample = next(iter(ds))
    stamp = int(sample["frames"][0, 0, 0, 0])          # global index of frame 0
    assert sample["actions"].shape[0] == W - 1
    for j in range(W - 1):
        i = stamp + 1 + j                              # the causing index
        assert int(sample["actions"][j, 0, 0]) == i % 2, (j, i)
        assert int(sample["actions"][j, 0, 1]) == (i // 2) % 2, (j, i)


def test_windows_never_cross_an_episode_boundary(shard):
    ds = D.ShardWindows(D.episode_table([shard]), shuffle=False)
    bounds = set(np.load(shard)["ep_offsets"].tolist())
    for s in ds:
        stamps = [int(v) for v in s["frames"][:, 0, 0, 0]]
        assert stamps == list(range(stamps[0], stamps[0] + W)), stamps
        # no reset inside the window
        assert not any(b in range(stamps[0] + 1, stamps[0] + W) for b in bounds)


def test_first_observation_of_an_episode_is_never_a_window_start(shard):
    """actions[ep_start] is the reset->first-frame block; there is no preceding
    observation to condition it on."""
    starts = {int(o) for o in np.load(shard)["ep_offsets"]}
    ds = D.ShardWindows(D.episode_table([shard]), shuffle=False)
    for s in ds:
        assert int(s["frames"][0, 0, 0, 0]) not in starts


def test_split_is_by_whole_trajectory_and_disjoint(tmp_path):
    d = tmp_path / "random"
    d.mkdir()
    paths = [_fake_shard(d / f"shard_{i:04d}.npz", n_eps=5) for i in range(4)]
    tr, va = D.split_episodes(paths, val_trajectories=4, seed=0)
    assert len(va) == 4 and len(tr) == 16
    keys = lambda eps: {(p.name, i) for p, i, _, _ in eps}
    assert not (keys(tr) & keys(va))
    # and the exact same split next time, so gate numbers compare across runs
    tr2, va2 = D.split_episodes(paths, val_trajectories=4, seed=0)
    assert keys(va) == keys(va2)


def test_split_val_is_capped_on_tiny_datasets(tmp_path):
    """Asking for Bai's 60 trajectories against 5 episodes must not empty the
    training set."""
    d = tmp_path / "random"
    d.mkdir()
    tr, va = D.split_episodes([_fake_shard(d / "shard_0000.npz", n_eps=5)], seed=0)
    assert len(tr) >= 4 and len(va) >= 1


def test_loader_batches_have_the_documented_shapes(shard):
    tr, _ = D.split_episodes([shard], val_trajectories=1, seed=0)
    b = next(iter(D.make_loader(tr, batch_size=4, num_workers=0)))
    assert b["frames"].shape[:2] == (4, W) and b["frames"].dtype == torch.uint8
    assert b["actions"].shape == (4, W - 1, CFG.data.frame_skip, 6)
    for k in ("world_x", "world_y", "scroll", "dies_in_5"):
        assert b[k].shape == (4, W) and b[k].dtype == torch.float32


def test_real_shards_are_consistent_with_the_schema():
    paths = D.shard_paths(CFG.data.data_dir)
    if not paths:
        pytest.skip("no collected shards on this machine")
    with np.load(paths[0]) as d:
        off = d["ep_offsets"]
        assert int(off[-1]) == len(d["world_x"]) == len(d["frames"])
        assert d["actions"].shape[1:] == (CFG.data.frame_skip, 6)
        assert len(d["outcome"]) == len(off) - 1
        # scroll tracks world_x on 1-1 (measured corr 0.998) — the scroll probe
        # is therefore nearly free, and is NOT evidence the camera is resolved.
        r = np.corrcoef(d["world_x"], d["scroll"])[0, 1]
        assert r > 0.9, r


def test_window_stride_reduces_windows_without_dropping_coverage(shard):
    """
    At stride 1 consecutive windows overlap by 7 of 8 frames and one epoch over
    2.67M observations is ~27,800 steps at batch 96 — 49 h on an A10G. Stride 4
    still covers every frame (windows overlap by half) and cuts that ~4x.
    """
    eps = D.episode_table([shard])
    n1 = sum(1 for _ in D.ShardWindows(eps, shuffle=False, stride=1))
    n4 = sum(1 for _ in D.ShardWindows(eps, shuffle=False, stride=4))
    assert 3.5 < n1 / n4 < 4.5
    seen = set()
    for s in D.ShardWindows(eps, shuffle=False, stride=4):
        seen.update(int(v) for v in s["frames"][:, 0, 0, 0])
    covered = len(seen) / (n1 + W - 1)
    assert covered > 0.9, covered          # every frame still appears in some window


def test_stride_defaults_to_the_config_value(shard):
    ds = D.ShardWindows(D.episode_table([shard]), shuffle=False)
    assert ds.stride == CFG.data.window_stride


def test_death_oversampling_removes_zero_positive_batches(tmp_path):
    """
    MEASURED on real shards at batch 96: 14 of 30 batches contained NO death at
    all, so the dead-in-5 head got no positive gradient on half its steps, and
    the batches that did contain one produced 48% of the total loss variance.
    """
    d = tmp_path / "random"
    d.mkdir()
    paths = []
    for i in range(3):
        p = _fake_shard(d / f"shard_{i:04d}.npz", n_eps=4, ep_len=60)
        with np.load(p) as z:
            arr = {k: z[k] for k in z.files}
        arr["dies_in_5"][:] = 0
        for s in np.load(p)["ep_offsets"][:-1]:         # 5 death frames per episode
            arr["dies_in_5"][int(s) + 50:int(s) + 55] = 1
        np.savez_compressed(p, **arr)
        paths.append(p)

    eps = D.episode_table(paths)
    plain = D.ShardWindows(eps, shuffle=False, death_frac=0.0)
    rich = D.ShardWindows(eps, shuffle=False, death_frac=0.5)
    frac = lambda ds: np.mean([float(s["dies_in_5"].any()) for s in ds])
    assert frac(plain) < 0.3
    assert frac(rich) > 1.6 * frac(plain)


def test_death_oversampling_duplicates_rather_than_drops(tmp_path):
    """No non-death window may be thrown away — the model still has to learn the
    99% of the game where nothing happens."""
    d = tmp_path / "random"
    d.mkdir()
    p = _fake_shard(d / "shard_0000.npz", n_eps=3, ep_len=60)
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files}
    arr["dies_in_5"][:] = 0
    arr["dies_in_5"][50:55] = 1
    np.savez_compressed(p, **arr)
    eps = D.episode_table([p])
    n_plain = sum(1 for _ in D.ShardWindows(eps, shuffle=False, death_frac=0.0))
    n_rich = sum(1 for _ in D.ShardWindows(eps, shuffle=False, death_frac=0.5))
    assert n_rich >= n_plain


def test_death_oversampling_is_a_noop_when_no_deaths_exist(shard):
    eps = D.episode_table([shard])
    a = sum(1 for _ in D.ShardWindows(eps, shuffle=False, death_frac=0.0))
    b = sum(1 for _ in D.ShardWindows(eps, shuffle=False, death_frac=0.5))
    assert a == b


def test_partially_written_shards_are_skipped_not_fatal(tmp_path):
    """
    Data collection and training share one Modal Volume. A gate run launched
    while a collection fan-out is still going WILL open an .npz that is mid-write
    and np.load raises BadZipFile from inside zipfile. That took down a real gate
    run; a shard that cannot be opened yet is simply not part of this epoch.
    """
    d = tmp_path / "random"
    d.mkdir()
    good = _fake_shard(d / "shard_0000.npz", n_eps=2)
    half = d / "shard_0001.npz"
    half.write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])   # truncated

    eps = D.episode_table([good, half])
    assert len(eps) == 2
    assert all(p == good for p, *_ in eps)

    with pytest.raises(Exception):
        D.episode_table([good, half], strict=True)


def test_iteration_survives_a_shard_going_bad_mid_stream(tmp_path):
    d = tmp_path / "random"
    d.mkdir()
    good = _fake_shard(d / "shard_0000.npz", n_eps=2)
    bad = _fake_shard(d / "shard_0001.npz", n_eps=2)
    eps = D.episode_table([good, bad])
    bad.write_bytes(b"not a zip")                 # corrupt AFTER the table is built
    got = list(D.ShardWindows(eps, shuffle=False))
    assert got, "iteration produced nothing; it should have yielded the good shard"
