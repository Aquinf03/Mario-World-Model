"""Stage 1 evals: config invariants that everything downstream assumes."""
import numpy as np
import pytest

from algo.config import CFG, BUTTONS, MACRO_ACTIONS, MACRO_NAMES, TARGETS


def test_button_order_is_load_bearing():
    """The action encoder and every probe index into this order. Do not reorder."""
    assert BUTTONS == ["left", "right", "up", "down", "A", "B"]


def test_macro_actions_are_valid_button_vectors():
    for name, vec in MACRO_ACTIONS.items():
        assert len(vec) == len(BUTTONS), f"{name} has {len(vec)} bits"
        assert set(vec) <= {0, 1}, f"{name} is not binary"


def test_macro_actions_cover_what_the_planner_needs():
    idx = {b: i for i, b in enumerate(BUTTONS)}
    assert MACRO_ACTIONS["run_right"][idx["right"]] == 1
    assert MACRO_ACTIONS["run_right"][idx["B"]] == 1
    assert MACRO_ACTIONS["run_right_jump"][idx["A"]] == 1
    assert sum(MACRO_ACTIONS["wait"]) == 0
    assert len(MACRO_NAMES) == 5


def test_no_macro_presses_left_and_right_together():
    idx = {b: i for i, b in enumerate(BUTTONS)}
    for name, v in MACRO_ACTIONS.items():
        assert not (v[idx["left"]] and v[idx["right"]]), f"{name} presses both"


def test_frame_skip_is_2_not_5():
    """A deliberate fix, not a default. skip-5 gives ~6 samples per jump arc."""
    assert CFG.data.frame_skip == 2


def test_encoder_patch_size_divides_the_frame():
    assert CFG.data.frame_size % 14 == 0, "ViT patch-14 needs a divisible input"


def test_gate_thresholds_beat_lemario():
    assert CFG.gate.min_y_probe_r2 > 0.188, "y gate must exceed LeMario's number"
    assert CFG.gate.min_x_probe_r2 <= 0.997


def test_targets_table_is_well_formed():
    for k, t in TARGETS.items():
        assert t["cmp"] in ("lt", "gt"), k
        assert isinstance(t["target"], float), k


def test_sae_width_is_not_aquins_llm_default():
    """
    aquin defaults n_features to 32768, which at d=192 is 170x overcomplete.
    """
    assert CFG.sae.d_model == CFG.model.latent_dim == 192
    assert CFG.sae.n_features / CFG.sae.d_model < 20


def test_sigreg_sweep_is_understood_as_full_runs():
    assert len(CFG.loss.sigreg_sweep) == 3
    assert CFG.loss.sigreg_lambda in CFG.loss.sigreg_sweep


def test_history_and_horizon_fit_in_a_sample():
    assert CFG.data.obs_per_sample >= CFG.model.history_len
    assert CFG.model.pred_horizon >= 1
