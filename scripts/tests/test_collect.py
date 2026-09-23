"""
Stage 1 evals: the collection layer — policy coverage and the shard contract.
"""
import numpy as np
import pytest

from dataset import ram as R
from algo.config import CFG, BUTTONS
from helpers import load_collector

CD = load_collector()
COMPLEX = [["NOOP"], ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"],
           ["A"], ["left"], ["left", "A"], ["left", "B"], ["left", "A", "B"],
           ["down"], ["up"]]


# --- policy coverage ---------------------------------------------------------
def test_random_policy_is_actually_biased_not_uniform():
    """
    Regression guard for the original bug: the draft built `p = np.ones(n)`,
    a uniform distribution, despite documenting a right/run/jump bias.
    """
    _, w = CD.random_policy(COMPLEX)
    assert w.sum() == pytest.approx(1.0)
    uniform = 1.0 / len(COMPLEX)
    right = sum(w[i] for i, c in enumerate(COMPLEX) if "right" in c)
    assert right > 0.6, f"right mass only {right:.2f}"
    assert w.std() > uniform * 0.5, "distribution is ~uniform — the bias is gone"


def test_policy_covers_jumps_which_the_y_probe_depends_on():
    _, w = CD.random_policy(COMPLEX)
    jump = sum(w[i] for i, c in enumerate(COMPLEX) if "A" in c)
    assert jump > 0.5, f"jump mass only {jump:.2f}; y-probe will starve"


def test_policy_still_explores_left_and_idle():
    """Pure right-jump would never see hazards from a standstill."""
    _, w = CD.random_policy(COMPLEX)
    assert sum(w[i] for i, c in enumerate(COMPLEX) if "left" in c) > 0.02
    assert w[0] > 0.0, "NOOP should retain some mass"


@pytest.mark.emulator
def test_policy_produces_real_jumps_in_the_emulator(random_trace):
    """Distribution mass is not enough — check Mario actually leaves the ground."""
    ys = np.array([r["y_ce"] for r in random_trace])
    ground = np.bincount(ys - ys.min()).argmax() + ys.min()
    airborne = (ys < ground - 15).mean()
    assert airborne > 0.05, f"airborne only {airborne:.1%} of observations"


@pytest.mark.emulator
def test_policy_makes_rightward_progress(random_trace):
    xs = [r["world_x"] for r in random_trace]
    assert max(xs) > min(xs) + 150, f"progress only {max(xs)-min(xs)}px"


# --- the shard contract ------------------------------------------------------
def _fake_episode(n=40, seed=0):
    rng = np.random.default_rng(seed)
    alive = np.ones(n, np.uint8)
    alive[-1] = 0
    return dict(
        frames=rng.integers(0, 255, (n, 8, 8, 3), dtype=np.uint8),
        actions=rng.integers(0, 2, (n, CFG.data.frame_skip, len(BUTTONS)), dtype=np.uint8),
        world_x=np.arange(n, dtype=np.int32) * 2,
        world_y=np.full(n, 432, np.int32),
        scroll=np.arange(n, dtype=np.int32),
        alive=alive,
        dies_in_5=R.dies_within(alive, 5),
        power=np.zeros(n, np.uint8),
        outcome=1,
    )


def test_shard_round_trips_without_pickle(tmp_path):
    """
    Regression guard: the draft stored an object array of dicts, which needs
    allow_pickle at load and cannot be memory-mapped. At 783 GB that matters.
    """
    eps = [_fake_episode(30, 0), _fake_episode(45, 1)]
    p = CD.save_shard(tmp_path, 0, eps)
    d = np.load(p)                        # NO allow_pickle
    assert d["frames"].dtype == np.uint8
    assert d["ep_offsets"].tolist() == [0, 30, 75]
    assert d["frames"].shape[0] == 75


def test_shard_offsets_reconstruct_each_episode(tmp_path):
    eps = [_fake_episode(30, 0), _fake_episode(45, 1)]
    d = np.load(CD.save_shard(tmp_path, 0, eps))
    off = d["ep_offsets"]
    for i, src in enumerate(eps):
        got = d["world_x"][off[i]:off[i + 1]]
        assert np.array_equal(got, src["world_x"]), f"episode {i} corrupted"


def test_shard_preserves_action_block_shape(tmp_path):
    d = np.load(CD.save_shard(tmp_path, 0, [_fake_episode(20)]))
    assert d["actions"].shape[1:] == (CFG.data.frame_skip, len(BUTTONS))


def test_shard_carries_the_predictive_death_label(tmp_path):
    d = np.load(CD.save_shard(tmp_path, 0, [_fake_episode(30)]))
    assert d["dies_in_5"].sum() == 6, "dies_in_5 not propagated"
    assert d["alive"].sum() == 29
    assert d["outcome"].tolist() == [1]


def test_empty_shard_writes_nothing(tmp_path):
    assert CD.save_shard(tmp_path, 0, []) is None
    assert not list(tmp_path.glob("*.npz"))


def test_shard_size_will_not_oom():
    """
    Regression guard for shard_size=256: a PPO episode at 224px/skip-2 is
    ~166 MB of frames, so 256 of them is ~42 GB resident before the first write.
    """
    d = CFG.data
    peak_gb = d.shard_size * (d.est_frames_ppo / d.frame_skip) * d.frame_size**2 * 3 / 1e9
    assert peak_gb < 8, f"peak shard RAM {peak_gb:.1f} GB will OOM the container"


# --- preprocessing -----------------------------------------------------------
@pytest.mark.emulator
def test_preprocess_produces_the_encoder_input_shape(walk_trace):
    f = CD.preprocess(walk_trace[0]["obs"], CFG.data.frame_size)
    assert f.shape == (CFG.data.frame_size, CFG.data.frame_size, 3)
    assert f.dtype == np.uint8


@pytest.mark.emulator
def test_preprocess_is_deterministic(walk_trace):
    a = CD.preprocess(walk_trace[0]["obs"], CFG.data.frame_size)
    b = CD.preprocess(walk_trace[0]["obs"], CFG.data.frame_size)
    assert np.array_equal(a, b)
