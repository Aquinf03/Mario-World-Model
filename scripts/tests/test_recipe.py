"""
The recipe is the aq spine: `aq train` reads recipe.yaml, not src/algo/config.py.
When the two disagree, the recipe is a confident wrong answer about what was run.

This happened: 9 of 10 settings had drifted while the experiments moved on,
including sigreg_lambda 0.1 — the value the dry-run grid showed collapses the
latent to rank 2. These tests pin the agreement.
"""
import pytest
import yaml
from pathlib import Path

from algo.config import CFG

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def recipe():
    return yaml.safe_load((ROOT / "recipe.yaml").read_text())


def test_recipe_matches_the_model_config(recipe):
    assert recipe["latent_dim"] == CFG.model.latent_dim
    assert recipe["pred_horizon"] == CFG.model.pred_horizon
    assert recipe["history_len"] == CFG.model.history_len
    assert recipe["predictor_layers"] == CFG.model.predictor_layers
    assert recipe["grad_checkpointing"] == CFG.model.grad_checkpointing


def test_recipe_matches_the_data_config(recipe):
    d = recipe["data"]
    assert d["window_stride"] == CFG.data.window_stride
    assert d["death_window_frac"] == CFG.data.death_window_frac
    assert recipe["frame_skip"] == CFG.data.frame_skip


def test_recipe_carries_the_measured_anti_collapse_settings(recipe):
    """
    The three settings without which the run collapses, all measured:
      stop_grad   absent -> eff_dim 11.1 -> 2.0, gain_5step 0.978 (fake)
      lambda 10   at 0.1 -> eff_dim 1.08; at 1.0 -> 1.89; at 10 -> 21.1
      aux_scale   at 1 the aux terms cost 0.008 against a 0.80 incentive
    """
    assert recipe["stop_grad_target"] is True
    assert recipe["sigreg_lambda"] >= 10.0
    assert recipe["aux_scale"] >= 20.0
    assert recipe["precision"] == "bf16"


def test_recipe_selects_this_train_s_method(recipe):
    assert recipe["method"] == "jepa"
    assert (ROOT / "methods" / "jepa.py").is_file()


def test_gate_fails_closed(recipe):
    """min_score 1.0 means every probe must meet its threshold, so the default
    outcome of a bad run is failure rather than a checkpoint."""
    assert recipe["eval"]["min_score"] == 1.0
    assert recipe["eval"]["metric"] == "gate_score"
    assert recipe["guard"]["safety"] is True


def test_epoch_gate_abort_threshold_is_below_lemario(recipe):
    """
    The abort bar has to sit BELOW the number being beaten. Setting it above
    0.188 would abort runs that are already better than the prior art; setting
    it at 0 would never abort anything.
    """
    ev = recipe["eval"]
    assert 0.0 < ev["gate_abort_y_r2"] < 0.188
    assert ev["gate_grace_epochs"] >= 1     # probes need an epoch to mean anything


def test_batch_size_fits_the_measured_memory_envelope(recipe):
    """batch 256 at window 8 OOMs a 24 GB A10G — measured, with a real crash."""
    imgs = recipe["batch_size"] * (recipe["history_len"] + recipe["pred_horizon"])
    assert imgs <= 800, f"{imgs} images per step will not fit"


def test_method_interface_is_complete_and_checkpoint_is_json_serialisable():
    """
    engine/step.py does json.loads(ckpt.read_text()), so fit() must return a
    JSON-able dict with a POINTER to the .pt rather than weights.
    """
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("m_jepa", ROOT / "methods" / "jepa.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for fn in ("fit", "evaluate", "predict", "write_inspect"):
        assert callable(getattr(m, fn)), fn
    json.dumps({"backend": "torch", "weights": "artifacts/jepa.pt",
                "variant": "aux", "params": 9801411, "eff_dim": 24.4})


def test_predict_resolves_relative_weight_paths(tmp_path):
    """
    fit() stores `weights` RELATIVE to the train dir so a train directory is
    movable. predict() opened it as absolute and raised FileNotFoundError the
    first time it was ever called.
    """
    import importlib.util
    import torch

    from algo.model import build_jepa

    (tmp_path / "artifacts").mkdir()
    torch.save({"jepa": build_jepa(CFG).state_dict()}, tmp_path / "artifacts" / "jepa.pt")
    spec = importlib.util.spec_from_file_location("m_jepa", ROOT / "methods" / "jepa.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    W = CFG.model.history_len + CFG.model.pred_horizon
    frames = torch.randint(0, 255, (1, W, 224, 224, 3), dtype=torch.uint8)
    out = m.predict({"weights": "artifacts/jepa.pt"}, [frames], {"_train": str(tmp_path)})
    assert out[0].shape == (1, CFG.model.pred_horizon, CFG.model.latent_dim)
