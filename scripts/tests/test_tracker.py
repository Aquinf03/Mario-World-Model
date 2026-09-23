"""Stage 1 evals: the tracker itself must never lie or crash."""
import json

import pytest

from algo import tracker as T


def test_every_probe_runs_without_crashing():
    """A probe that raises would silently report TODO and hide real progress."""
    for st in T.STAGES:
        for c in st.checks:
            state, detail = c.run()
            assert state in (T.DONE, T.PARTIAL, T.TODO, T.BLOCKED), f"{c.key}: {state}"
            assert isinstance(detail, str) and detail, f"{c.key} gave no detail"


def test_no_probe_swallows_an_exception_as_done():
    for st in T.STAGES:
        for c in st.checks:
            state, detail = c.run()
            assert not (state == T.DONE and "probe error" in detail)


def test_progress_is_bounded_and_weighted():
    for st in T.STAGES:
        _, pct = st.evaluate()
        assert 0.0 <= pct <= 1.0, f"stage {st.num} -> {pct}"


def test_there_are_exactly_four_stages_with_unique_numbers():
    assert [s.num for s in T.STAGES] == [1, 2, 3, 4]
    assert len({s.name for s in T.STAGES}) == 4


def test_every_stage_declares_a_goal_and_exit_criterion():
    for st in T.STAGES:
        assert st.goal.strip() and st.exit_criteria.strip()
        assert st.checks, f"stage {st.num} has no checks"


def test_check_keys_are_unique_within_a_stage():
    for st in T.STAGES:
        keys = [c.key for c in st.checks]
        assert len(keys) == len(set(keys)), f"stage {st.num}: {keys}"


def test_json_output_is_serialisable():
    d = T.as_dict(T.STAGES)
    json.dumps(d)
    assert 0.0 <= d["overall"] <= 1.0
    assert len(d["stages"]) == 4


def test_status_md_can_be_written(tmp_path):
    p = T.write_status_md(T.STAGES, tmp_path / "STATUS.md")
    txt = p.read_text()
    assert "AQ-Mario" in txt
    for st in T.STAGES:
        assert st.name in txt


def test_render_does_not_crash_without_a_tty():
    out = T.render(T.STAGES, use_color=False)
    assert "STAGE 1" in out and "OVERALL" in out


def test_stage1_ram_probe_reflects_the_lock():
    from dataset import ram as R
    state, _ = T.p_ram_validated()
    assert state == (T.DONE if R.validated() else T.TODO)
