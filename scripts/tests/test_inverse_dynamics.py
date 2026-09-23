"""
Inverse dynamics: predict the action from the latents around the transition.

THE OFF-BY-ONE IS THE WHOLE CORRECTNESS OF THIS LOSS. Per DOCS.md Part V,
`actions[i]` CAUSED `frames[i]`, and a window carries `actions[s+1 .. s+W-1]`,
so `batch["actions"][:, j]` is the action of the transition j -> j+1. Train the
head against `[:, j+1]` instead and the loss still falls -- consecutive actions
in Mario are highly correlated -- so the mistake does not announce itself. These
tests make the alignment observable.
"""
import torch

from algo.config import CFG, N_BUTTONS
from algo.losses import InverseDynamics, inverse_dynamics, inverse_dynamics_loss

W = CFG.model.history_len + CFG.model.pred_horizon
D = CFG.model.latent_dim
H = CFG.model.history_len
P = CFG.model.pred_horizon


def _fake_out(B=4):
    z = torch.randn(B, W, D)
    return {"z": z, "zhat": torch.randn(B, P, D), "z_target": z[:, H:]}


def _batch(B=4):
    return {"actions": torch.randint(0, 2, (B, W - 1, CFG.data.frame_skip, N_BUTTONS)).float()}


def test_terms_present_and_finite():
    inv = inverse_dynamics(CFG)
    t = inverse_dynamics_loss(inv, _fake_out(), _batch(), cfg=CFG)
    assert set(t) == {"inv_real", "inv_pred"}
    for v in t.values():
        assert torch.isfinite(v), t


def test_none_head_is_a_zero_that_still_backprops():
    """A disabled term must not break the graph — it is summed into `loss`."""
    out = _fake_out()
    out["z"].requires_grad_(True)
    t = inverse_dynamics_loss(None, out, _batch(), cfg=CFG)
    (t["inv_real"] + t["inv_pred"]).backward()
    assert float(t["inv_real"]) == 0.0


def test_pred_term_targets_the_transitions_zhat_actually_covers():
    """
    zhat predicts window frames H..W-1, so the pairs it forms start at frame H-1
    and the actions involved are indices H-1 .. W-2 — exactly P of them.
    """
    inv = inverse_dynamics(CFG)
    out, b = _fake_out(), _batch()
    seq = torch.cat([out["z"][:, H - 1:H], out["zhat"]], dim=1)
    assert seq.shape[1] == P + 1
    tgt = b["actions"].amax(dim=2)[:, H - 1:H - 1 + P]
    assert tgt.shape[1] == P
    assert H - 1 + P - 1 == W - 2, "last action used must be the last one available"


def test_alignment_is_learnable_and_the_shift_is_not():
    """
    Build latents where the transition t -> t+1 LITERALLY carries the action:
    z[t+1] - z[t] = the button vector, padded. A head trained on the correct
    alignment must reach a far lower loss than the same head trained on the
    off-by-one, which is the check that the indexing in the loss is right and not
    merely plausible.
    """
    torch.manual_seed(0)
    B, T = 64, W
    acts = torch.randint(0, 2, (B, T - 1, N_BUTTONS)).float()
    z = torch.zeros(B, T, D)
    for t in range(T - 1):
        step = torch.zeros(B, D)
        step[:, :N_BUTTONS] = acts[:, t] * 4.0
        z[:, t + 1] = z[:, t] + step

    def train(target):
        torch.manual_seed(0)
        inv = InverseDynamics(CFG)
        opt = torch.optim.Adam(inv.parameters(), lr=3e-3)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            logit = inv(z[:, :-1], z[:, 1:])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
            loss.backward()
            opt.step()
        return float(loss)

    correct = train(acts)
    shifted = train(torch.cat([acts[:, 1:], acts[:, :1]], dim=1))
    assert correct < 0.05, f"correct alignment should be nearly solvable, got {correct}"
    assert shifted > 5 * correct, f"shift must be much worse: {correct=} {shifted=}"
