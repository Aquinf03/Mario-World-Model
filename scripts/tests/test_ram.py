"""
Stage 1 evals: the ground-truth layer.

Every gate, probe, aux head and planner cost downstream reads these values. A
wrong address does not crash — it silently corrupts all of them. So these tests
cross-check our extraction against an INDEPENDENT source: the env's own info
dict, and gym-super-mario-bros' own death logic.
"""
import numpy as np
import pytest

from dataset import ram as R
from algo.config import CFG

pytestmark = pytest.mark.emulator


# --- world_x -----------------------------------------------------------------
def test_world_x_matches_env_ground_truth_exactly(walk_trace):
    """Our page*256+fine formula must equal the env's own x_pos on EVERY frame."""
    pairs = [(r["world_x"], r["info_x"]) for r in walk_trace if r["info_x"] is not None]
    assert pairs, "env exposed no x_pos"
    mismatches = [(a, b) for a, b in pairs if a != b]
    assert not mismatches, f"{len(mismatches)}/{len(pairs)} mismatches, first {mismatches[:3]}"


def test_world_x_advances_monotonically_when_running_right(walk_trace):
    xs = [r["world_x"] for r in walk_trace[:60]]
    d = np.diff(xs)
    assert (d >= 0).mean() > 0.9, f"only {(d>=0).mean():.0%} of steps advanced"
    assert xs[-1] > xs[0] + 100, f"barely moved: {xs[0]} -> {xs[-1]}"


def test_world_x_never_page_wraps_backwards(walk_trace):
    """A wrong page byte shows up as a large negative jump."""
    d = np.diff([r["world_x"] for r in walk_trace])
    assert not (d < -50).any(), f"page-wrap corruption: {d[d < -50][:5]}"


# --- world_y -----------------------------------------------------------------
def test_world_y_is_exactly_affine_to_env_y(walk_trace):
    """
    The env computes y independently. If our extraction is right the two are
    related by an exact affine map, so R^2 is 1 to floating-point precision.
    R^2 is affine-invariant, which is why our different origin is harmless.
    """
    ours = np.array([r["y_ce"] for r in walk_trace], float)
    theirs = np.array([r["info_y"] for r in walk_trace], float)
    A = np.vstack([ours, np.ones_like(ours)]).T
    coef, *_ = np.linalg.lstsq(A, theirs, rcond=None)
    resid = theirs - A @ coef
    r2 = 1 - (resid**2).sum() / ((theirs - theirs.mean())**2).sum()
    assert r2 > 0.999999, f"R^2={r2:.9f}, slope={coef[0]:.4f}"
    assert coef[0] == pytest.approx(-1.0, abs=1e-6), f"slope {coef[0]}"


def test_y_candidates_are_mirrors_not_alternatives(walk_trace):
    """
    0x00CE and 0x03B8 hold the same value. Documented as a measurement so nobody
    later "fixes" ram.py by switching to the other one expecting a difference.
    """
    a = np.array([r["y_ce"] for r in walk_trace])
    b = np.array([r["y_3b8"] for r in walk_trace])
    assert (a == b).all(), f"{(a != b).sum()}/{len(a)} frames differ"


def test_world_y_decreases_when_jumping(walk_trace):
    """
    Sign convention is load-bearing: the planner's vertical cost and the SAE
    steering demo both assume it. world_y is a SCREEN coord, growing downward.
    """
    ground = walk_trace[55]["y_ce"]
    jump = [r["y_ce"] for r in walk_trace[60:105]]
    assert min(jump) < ground - 40, f"ground={ground}, jump min={min(jump)}"
    assert ground == pytest.approx(R.GROUND_Y_1_1, abs=8)


def test_jump_arc_survives_frame_skip_2(walk_trace):
    """
    The whole reason frame_skip is 2 and not 5. A jump arc must yield enough
    observations that y is prediction-relevant; at skip-5 it nearly vanishes.
    """
    jump = np.array([r["y_ce"] for r in walk_trace[60:105]])
    arc_frames = int((jump < jump[0] - 5).sum())
    assert arc_frames >= 20, f"only {arc_frames} emulator frames airborne"
    assert arc_frames // CFG.data.frame_skip >= 10, "too few observations per jump at skip-2"


# --- scroll ------------------------------------------------------------------
def test_scroll_tracks_x_and_never_exceeds_it(walk_trace):
    xs = np.array([r["world_x"] for r in walk_trace[:60]], float)
    sc = np.array([r["scroll"] for r in walk_trace[:60]], float)
    assert (sc <= xs + 8).all(), "camera left edge cannot be ahead of Mario"
    moved_x, moved_s = xs[-1] - xs[0], sc[-1] - sc[0]
    assert moved_s > 0, "camera never scrolled"
    assert 0.3 <= moved_s / moved_x <= 1.2, f"ratio {moved_s/moved_x:.2f}"


def test_scroll_is_flat_before_mid_screen_then_follows(walk_trace):
    """Camera is pinned until Mario passes mid-screen — a distinct signature."""
    sc = np.array([r["scroll"] for r in walk_trace[:60]])
    assert sc[0] == sc[1], "camera moved on the very first step"
    assert sc[-1] > sc[0], "camera never started following"


# --- alive / death -----------------------------------------------------------
def test_is_dead_matches_gym_smb_own_definition(random_trace, env):
    """Our is_dead must agree with the env's _is_dying/_is_dead, pit falls included."""
    for r in random_trace:
        expected = r["player_state"] in (0x06, 0x0B) or r["y_viewport"] > 1
        assert r["is_dead_ours"] == expected, f"disagreement at {r}"


def test_pit_fall_clause_is_present():
    """
    Regression guard. player_state ALONE misses pit deaths, which is most of
    them for a right-biased policy. Checking the constant is not enough —
    exercise the branch.
    """
    ram = np.zeros(2048, dtype=np.uint8)
    ram[R.PLAYER_STATE] = 0x08           # "normal"
    ram[R.Y_VIEWPORT] = 2                # fallen below the stage
    assert R.is_dead(ram), "pit fall not detected — player_state-only regression"
    ram[R.Y_VIEWPORT] = 1
    assert not R.is_dead(ram)


def test_alive_is_almost_always_one(random_trace):
    """
    Documents WHY dies_within exists. The episode terminates the instant Mario
    dies, so the coincident label carries almost no signal.
    """
    alive = np.array([0 if r["is_dead_ours"] else 1 for r in random_trace])
    assert alive.mean() > 0.99, "unexpectedly many dead frames"


def test_dies_within_rolls_the_label_backwards():
    alive = np.array([1] * 20 + [0])
    lab = R.dies_within(alive, 5)
    assert lab.sum() == 6, f"expected 6 positives, got {lab.sum()}"
    assert lab[:15].sum() == 0 and lab[-6:].all()


def test_dies_within_is_empty_when_episode_never_ends_in_death():
    assert R.dies_within(np.ones(50), 5).sum() == 0


def test_dies_within_handles_empty_and_short():
    assert len(R.dies_within([], 5)) == 0
    assert R.dies_within([0], 5).tolist() == [1]


# --- the extraction contract -------------------------------------------------
def test_ram_state_returns_every_field_with_sane_values(env):
    env.reset()
    for _ in range(30):
        env.step(3)
    s = R.ram_state(env.unwrapped.ram)
    assert set(s) == set(R.STATE_FIELDS), f"field drift: {set(s) ^ set(R.STATE_FIELDS)}"
    assert all(isinstance(v, int) for v in s.values()), "non-int leaked into the schema"
    assert 0 < s["world_x"] < 4000
    assert 0 < s["world_y"] < 2048
    assert 0 <= s["scroll"] <= s["world_x"] + 8
    assert s["alive"] in (0, 1)
    assert s["power"] in (0, 1, 2)


def test_addresses_are_locked_and_consistent():
    lock = R.load_lock()
    assert lock is not None, "run scripts/helpers/validate_ram.py --lock"
    assert R.validated()
    for field, name in lock["chosen"].items():
        assert name in R.CANDIDATES[field], f"lock names unknown candidate {name}"
    assert lock["level"] == CFG.data.level, "lock was taken on a different level"


def test_extraction_refuses_when_a_field_is_unvalidated(monkeypatch):
    """Fail closed: a missing choice must raise, never silently pick one."""
    monkeypatch.setattr(R, "CHOSEN", {**R.CHOSEN, "world_y": None})
    monkeypatch.setattr(R, "load_lock", lambda: None)
    assert not R.validated()
    with pytest.raises(RuntimeError, match="not validated"):
        R.ram_state(np.zeros(2048, dtype=np.uint8))
