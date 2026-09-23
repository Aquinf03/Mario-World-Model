"""
The action-conditioning gate.

WHY THIS FILE EXISTS. scripts/tests/test_model.py already asserts three things about the
action path: AdaLN-Zero is the identity at init, an UNZEROED predictor responds
to a changed action embedding, and the action encoder's gradient unsticks on
step 1. All three pass on a model that, once trained, ignores its actions
completely -- measured on the first fully-trained checkpoint, where rolling
`run_right` for 8 macros vs `run_left` for 8 macros moved the predicted latent
0.759 against 18.325 between two real frames the same span apart (ratio 0.041),
and the planner stalled at x=435/3161.

Those tests check the architecture CAN use actions. This one checks a given
checkpoint DOES.
"""
import numpy as np
import torch

from algo.config import CFG
from algo.gates import action_conditioning_gate, _fit_probe, encode_episodes
from algo.model import build_jepa
from algo import data as D

from pathlib import Path

DATA = Path("~/.aqmario/data").expanduser()
HAVE_DATA = bool(D.shard_paths(DATA))

import pytest
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="no local shards")


def _tiny_setup(n_traj=4):
    paths = D.shard_paths(DATA)
    train_eps, val_eps = D.split_episodes(paths, 4, seed=0)
    model = build_jepa(CFG).eval()
    cache = encode_episodes(model, train_eps[:n_traj], CFG, "cpu", max_obs_per_episode=120)
    px, _ = _fit_probe(cache["z"], torch.from_numpy(cache["world_x"]),
                       cache["z"][:8], torch.from_numpy(cache["world_x"][:8]),
                       device="cpu", epochs=5)
    for q in px.parameters():
        q.requires_grad_(False)
    return model, val_eps, px.eval()


def test_action_blind_predictor_scores_near_zero():
    """
    A freshly built JEPA has AdaLN-Zero gates at exactly zero, so its predictor
    output cannot depend on the action at all. The gate must report that, not a
    comfortable-looking small number.
    """
    model, val_eps, px = _tiny_setup()
    res = action_conditioning_gate(model, val_eps, px, CFG, device="cpu",
                                   n_states=3, horizon=4)
    assert res["action_n_states"] > 0, "gate found no usable states"
    assert res["action_effect_ratio"] < 0.01, res
    assert res["action_effect_ratio"] < CFG.gate.min_action_effect_ratio


def test_gate_reports_no_states_rather_than_a_number():
    """Given nothing to measure, say so. A gate that invents a value is worse."""
    model, _, px = _tiny_setup()
    res = action_conditioning_gate(model, [], px, CFG, device="cpu")
    assert res["action_n_states"] == 0
    assert np.isnan(res["action_effect_ratio"])


def test_nan_fails_closed():
    """
    The pass expression in run_all_gates is `ratio >= threshold`. NaN >= x is
    False in Python, so an uncomputable action gate FAILS. Pinned because the
    obvious 'tidy-up' -- np.nan_to_num, or `or 0.0` -- silently turns a gate that
    could not be measured into a gate that passed.
    """
    assert not (float("nan") >= CFG.gate.min_action_effect_ratio)
    assert not (float("nan") >= CFG.gate.min_action_x_gap_px)
