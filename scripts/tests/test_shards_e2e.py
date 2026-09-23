"""
Stage 1 evals: end-to-end over REAL collected shards.

These are the checks that would have caught a bad RAM address after the fact.
They skip if no shards exist yet, so the suite still runs on a clean checkout.
"""
import numpy as np
import pytest

from algo.config import CFG
from dataset import ram as R

SHARDS = sorted((CFG.data.data_dir / "random").glob("shard_*.npz")) if \
    (CFG.data.data_dir / "random").exists() else []
pytestmark = pytest.mark.skipif(not SHARDS, reason="no collected shards yet")


@pytest.fixture(scope="module")
def data():
    out = {k: [] for k in ("world_x", "world_y", "scroll", "alive", "dies_in_5", "power")}
    eps = []
    for f in SHARDS:
        d = np.load(f)
        off = d["ep_offsets"]
        for k in out:
            out[k].append(d[k])
        for i in range(len(off) - 1):
            eps.append({k: d[k][off[i]:off[i + 1]] for k in out})
    return {k: np.concatenate(v) for k, v in out.items()}, eps


def test_shards_load_without_pickle():
    for f in SHARDS:
        np.load(f)          # would raise if object arrays had been written


def test_every_field_has_the_same_length(data):
    flat, _ = data
    n = {k: len(v) for k, v in flat.items()}
    assert len(set(n.values())) == 1, f"ragged shard: {n}"


def test_world_x_stays_in_level_bounds(data):
    flat, _ = data
    assert 0 <= flat["world_x"].min()
    assert flat["world_x"].max() < 4000, "beyond the end of 1-1"


def test_no_episode_page_wraps(data):
    _, eps = data
    for i, e in enumerate(eps):
        d = np.diff(e["world_x"].astype(int))
        assert not (d < -50).any(), f"episode {i} page-wrap: {d[d < -50][:3]}"


def test_scroll_never_leads_mario(data):
    flat, _ = data
    assert (flat["scroll"] <= flat["world_x"] + 8).all()


def test_world_y_is_centred_on_the_measured_ground(data):
    flat, _ = data
    y = flat["world_y"]
    assert abs(np.bincount(y - y.min()).argmax() + y.min() - R.GROUND_Y_1_1) < 12


def test_vertical_coverage_is_rich_enough_for_the_y_probe(data):
    """
    The entire point of skip-2 plus a jump-biased policy. LeMario's y-probe hit
    R^2=0.188 partly because height barely varied in his data.
    """
    y = flat_y = data[0]["world_y"]
    assert y.std() > 10, f"y std only {y.std():.1f}"
    assert len(np.unique(y)) > 50, f"only {len(np.unique(y))} distinct heights"
    airborne = (y < R.GROUND_Y_1_1 - 15).mean()
    assert airborne > 0.10, f"airborne only {airborne:.1%} of observations"


def test_death_label_has_usable_positive_rate(data):
    """
    Raw `alive` carries ~one positive per episode, because the episode ends the
    instant Mario dies. dies_within(k) marks the k observations BEFORE it too,
    so by construction the rate is (k+1)x richer — that is the whole point.
    """
    flat, eps = data
    raw = 1 - flat["alive"].mean()
    rolled = flat["dies_in_5"].mean()
    deaths = sum(1 for e in eps if e["dies_in_5"].sum() > 0)
    assert raw > 0, "no deaths at all — policy is not reaching hazards"
    assert rolled == pytest.approx(6 * raw, rel=0.35), \
        f"dies_in_5 {rolled:.4%} vs raw {raw:.4%} (expected ~6x for k=5)"
    assert deaths / len(eps) > 0.5, \
        f"only {deaths}/{len(eps)} episodes end in death — poor hazard coverage"


def test_dies_in_5_only_fires_at_the_end_of_an_episode(data):
    _, eps = data
    for i, e in enumerate(eps):
        d = e["dies_in_5"]
        if d.sum() == 0:
            continue
        assert d[-1] == 1, f"episode {i}: label set but not at the end"
        assert d.sum() <= 6, f"episode {i}: {d.sum()} positives, expected <=6"


def test_power_state_is_in_range(data):
    flat, _ = data
    assert set(np.unique(flat["power"])) <= {0, 1, 2}


def test_measured_compression_matches_config():
    obs = sum(np.load(f)["ep_offsets"][-1] for f in SHARDS)
    disk = sum(f.stat().st_size for f in SHARDS)
    ratio = (CFG.data.frame_size ** 2 * 3 * obs) / disk
    assert ratio > 0.5 * CFG.data.compression_ratio, \
        f"measured {ratio:.0f}x vs config {CFG.data.compression_ratio}x"
