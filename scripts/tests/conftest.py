"""
Shared fixtures. The emulator is slow to construct, so traces are session-scoped
and shared: one walk-right-and-jump trace, one random-policy trace, reused by
every test that needs real RAM.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
warnings.filterwarnings("ignore")

emulator = pytest.mark.emulator          # registered in pytest.ini


def _make_env():
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from algo.config import CFG
    return JoypadSpace(gym_super_mario_bros.make(CFG.data.level), COMPLEX_MOVEMENT)


@pytest.fixture(scope="session")
def env():
    try:
        e = _make_env()
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"emulator unavailable: {exc}")
    yield e
    e.close()


def _record(env, actions):
    """Step `actions` and record every RAM quantity plus the env's own info dict."""
    from dataset import ram as R
    env.reset()
    rows = []
    for a in actions:
        obs, _, done, info = env.step(a)
        m = env.unwrapped.ram
        rows.append(dict(
            world_x=R._pair(m, 0x006D, 0x0086),
            y_ce=R._pair(m, 0x00B5, 0x00CE),
            y_3b8=R._pair(m, 0x00B5, 0x03B8),
            scroll=R._pair(m, 0x071A, 0x071C),
            player_state=int(m[0x000E]),
            y_viewport=int(m[0x00B5]),
            info_x=info.get("x_pos"),
            info_y=info.get("y_pos"),
            is_dead_ours=R.is_dead(m),
            done=bool(done),
            obs=obs,
        ))
        if done:
            break
    return rows


@pytest.fixture(scope="session")
def walk_trace(env):
    """Run right for 60 steps, then hold a jump for 45."""
    return _record(env, [3] * 60 + [4] * 45)


@pytest.fixture(scope="session")
def random_trace(env):
    """A trace from the ACTUAL biased policy used for collection."""
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from helpers import load_collector
    pick, _ = load_collector().random_policy(COMPLEX_MOVEMENT)
    np.random.seed(0)
    return _record(env, [pick(None) for _ in range(600)])


def pytest_sessionfinish(session, exitstatus):
    """
    Write .pytest_result.json so `python -m algo.tracker` reports the eval
    suite from a REAL run rather than by counting `def test_` in the files.
    Hand-maintaining that file would be exactly the hand-ticked checkbox the
    tracker exists to avoid.
    """
    import json
    rep = getattr(session.config, "_aqmario_counts", None) or {}
    total = session.testscollected or 0
    failed = session.testsfailed or 0
    (Path(__file__).resolve().parents[2] / ".pytest_result.json").write_text(
        json.dumps({"passed": total - failed, "failed": failed,
                    "exitstatus": int(exitstatus), **rep}))
