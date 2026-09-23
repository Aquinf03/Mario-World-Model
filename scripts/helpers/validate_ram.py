"""
Stage 1, do-first: validate the SMB RAM addresses against a real trace.

A wrong address does not crash — it silently corrupts every gate, probe and
planner cost downstream. So before collecting 8k episodes, drive Mario right
(and jump), record every candidate address from src/dataset/ram.py, and check which
ones satisfy the invariants declared there.

    python scripts/helpers/validate_ram.py            # report
    python scripts/helpers/validate_ram.py --lock     # report + write ~/.aqmario/ram_lock.json

Needs the emulator:
    pip install gym-super-mario-bros==7.4.0 nes-py opencv-python-headless
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataset import ram as R
from algo.config import CFG, BUTTONS


def _combo_index(combos, want):
    """Index of the COMPLEX_MOVEMENT entry exactly matching `want`."""
    for i, c in enumerate(combos):
        if set(c) == set(want):
            return i
    return None


def trace(n_obs=90, jump_at=40):
    """Walk right, jump partway through, recording all candidate readings."""
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    env = JoypadSpace(gym_super_mario_bros.make(CFG.data.level), COMPLEX_MOVEMENT)
    a_right = _combo_index(COMPLEX_MOVEMENT, ["right", "B"]) or 3
    a_jump = _combo_index(COMPLEX_MOVEMENT, ["right", "A", "B"]) or 4

    env.reset()
    series = {f"{field}:{name}": [] for field, cands in R.CANDIDATES.items() for name in cands}
    series["alive"] = []
    skip = CFG.data.frame_skip

    for t in range(n_obs):
        a = a_jump if jump_at <= t < jump_at + 12 else a_right
        done = False
        for _ in range(skip):
            _, _, done, _ = env.step(a)
            if done:
                break
        mem = env.unwrapped.ram
        for field, cands in R.CANDIDATES.items():
            for name, addrs in cands.items():
                series[f"{field}:{name}"].append(R._pair(mem, *addrs))
        series["alive"].append(0 if int(mem[R.PLAYER_STATE]) in R.DEAD_STATES else 1)
        if done:
            break
    env.close()
    return series


def evaluate(series):
    """Run each declared invariant against each candidate. Returns results + winners."""
    results, winners = [], {}
    # world_x first — the scroll invariant needs it as context.
    order = ["world_x", "scroll", "world_y"]
    ctx = {}
    for field in order:
        inv = next((i for i in R.INVARIANTS if i.field == field), None)
        best = None
        for name in R.CANDIDATES[field]:
            s = series[f"{field}:{name}"]
            ok, detail = inv.fn(s, ctx)
            results.append((field, name, ok, detail, inv.label))
            if ok and best is None:
                best = name
        winners[field] = best
        if best:
            ctx[field] = series[f"{field}:{best}"]
    return results, winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true", help="write ~/.aqmario/ram_lock.json")
    ap.add_argument("--obs", type=int, default=90)
    a = ap.parse_args()

    print(f"tracing {a.obs} observations on {CFG.data.level} (skip {CFG.data.frame_skip})...")
    series = trace(a.obs)
    results, winners = evaluate(series)

    print()
    width = max(len(f"{f}:{n}") for f, n, *_ in results)
    for field, name, ok, detail, label in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {f'{field}:{name}':<{width}}  {detail}")
        print(f"        invariant: {label}")

    print("\nwinners:")
    for f, n in winners.items():
        print(f"  {f:9s} {n or '*** NONE PASSED — do not collect data ***'}")

    if not all(winners.values()):
        print("\nAt least one field has no passing candidate. Add candidates to "
              "src/dataset/ram.py CANDIDATES and re-run before collecting.")
        return 1

    if a.lock:
        p = Path("~/.aqmario/ram_lock.json").expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"chosen": winners, "level": CFG.data.level,
                                 "n_obs": a.obs}, indent=2))
        print(f"\nlocked -> {p}")
    else:
        print("\n(dry run — pass --lock to write the lock file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
