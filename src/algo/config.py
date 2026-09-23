"""
Central config for AQ-Mario. Everything downstream (data schema, model dims,
eval gates, the stage tracker) reads from here so the stages stay consistent.

Design notes tied to the LeMario postmortem:
- FRAME_SKIP is 2, not 5. A jump arc is ~30 emulator frames; skip-5 gives ~6
  samples per jump and lets SIGReg throw away vertical info because it barely
  helps 1-step prediction. Skip-2 makes y matter for the prediction loss.
- We log RAM-derived state (x, y, scroll, alive) as AUX TARGETS. In the
  aux-loss variant these get small prediction heads; in the pure variant they
  are used only by the eval gates / probes, never in the training loss.
"""
from dataclasses import dataclass, field
from pathlib import Path


# ---- Action space -----------------------------------------------------------
# gym-super-mario-bros COMPLEX_MOVEMENT is 12 actions, but LeMario used the raw
# 6-button representation (Left, Right, Up, Down, A, B). We keep the 6-bit
# button vector as the ground-truth action, and ALSO expose a small macro-action
# set for the planner (categorical CEM over Gaussian is a bad fit for buttons).
BUTTONS = ["left", "right", "up", "down", "A", "B"]  # order is load-bearing; probes/action-encoder assume it
N_BUTTONS = len(BUTTONS)

MACRO_ACTIONS = {
    "wait":            [0, 0, 0, 0, 0, 0],
    "run_right":       [0, 1, 0, 0, 0, 1],   # right + B (run)
    "run_right_jump":  [0, 1, 0, 0, 1, 1],   # right + B + A
    "jump":            [0, 0, 0, 0, 1, 0],   # A only
    "run_left":        [1, 0, 0, 0, 0, 1],
}
MACRO_NAMES = list(MACRO_ACTIONS.keys())


# ---- Data schema ------------------------------------------------------------
@dataclass
class DataConfig:
    level: str = "SuperMarioBros-1-1-v0"   # stay on ONE level until it works
    frame_size: int = 224                   # matches LeMario / ViT-Tiny patch=14
    frame_skip: int = 2                     # emulator frames between observations
    obs_per_sample: int = 4                 # frames per training sample (context+targets)
    episodes_random: int = 4000             # random policy: jumps a lot, covers hazards
    episodes_ppo: int = 4000                # noisy PPO: covers "making progress"
    episodes_human: int = 0                 # optional, if you record any

    # WAS 256. A PPO episode at 224px/skip-2 is ~166 MB of frames; 256 of them
    # held in a list before writing is ~42 GB resident -> container OOM.
    # 16 keeps a shard near ~2.7 GB peak. See tracker's projected-size readout.
    shard_size: int = 16

    # Window stride. At stride 1 consecutive windows overlap by 7 of 8 frames and
    # one epoch over 2.67M observations is ~27,800 steps at batch 96 — measured
    # ~1.9 steps/s on an A10G, so 12 epochs would be 49 hours there and the
    # plan's "3.2 h H100" estimate is not reachable. Stride 4 keeps every frame
    # in the dataset (windows still overlap by half) and cuts an epoch to ~7,000
    # steps, which is what the budget was written against.
    window_stride: int = 4

    # Fraction of each batch drawn from windows that CONTAIN a death.
    #
    # Deaths are ~1 per episode: 5 labelled observations out of ~669, so 0.75% of
    # frames. Sampling windows uniformly, MEASURED over 40 real batches of 96
    # windows: 18 of 40 batches contained ZERO deaths and one contained 24. The
    # dead-in-5 head therefore gets no positive gradient on ~45% of steps, and
    # aux_dead alone accounts for 48% of the total loss variance (std/mean 0.43
    # against aux_scroll's 0.09) — the jagged loss curve is this term firing.
    # Oversampling fixes both the variance and the dead steps; pos_weight in
    # AuxHeads then corrects the remaining imbalance.
    death_window_frac: float = 0.15

    data_dir: Path = Path("~/.aqmario/data").expanduser()

    # MEASURED on the full 4000-episode random track (not estimated, and no
    # longer extrapolated from 40): 2,674,774 observations / 4000 episodes = 669
    # obs = 1338 emulator frames per episode, 3.3x the 400 originally assumed.
    # The earlier 40-episode sample said 1094 and was 20% low. PPO is still an
    # estimate until that track runs.
    est_frames_random: int = 1338
    est_frames_ppo: int = 2200
    # MEASURED compression on real shards: 5.4 KB/obs on disk vs 147 KB raw.
    # NES art deflates far better than the 4x originally assumed, which is what
    # takes training off the network-I/O bound.
    # 14.7 GB on the volume for 2.67M observations = 5.5 KB/obs against 147 KB
    # raw. Confirmed at full scale, not just on the first three shards.
    compression_ratio: float = 27.4
    sec_per_episode: float = 1.78          # measured, single core


# ---- Model ------------------------------------------------------------------
@dataclass
class ModelConfig:
    latent_dim: int = 192                   # the famous 192 numbers
    encoder: str = "vit_tiny_patch14"       # ~5M params
    predictor_layers: int = 6
    predictor_heads: int = 16
    predictor_dropout: float = 0.1          # paper ablation: 0.1 is the sweet spot
    history_len: int = 3                    # frames the predictor attends over
    action_frames: int = None               # set in __post_init__ from frame_skip
    # 5, not 3. TARGETS below compares mse_5step / gain_5step against LeMario;
    # at horizon 3 those two are simply not measurable, and rolling further at
    # eval time is not an option because the predictor's positional embedding is
    # fixed at history_len + pred_horizon. Costs 33% more per batch (window 6->8).
    pred_horizon: int = 5

    # ViT activation checkpointing. NOT optional above batch ~64: a step encodes
    # batch_size x window images (8 frames per window at horizon 5), so batch 128
    # is 1024 forward passes of a 257-token ViT and OOMs a 24 GB A10G. This
    # matters because SIGReg's ability to see a collapse is mostly a function of
    # BATCH size (measured: rank-2 reads 2.8x null at n=32, 41.8x at n=512), so
    # the regulariser and the memory budget pull in opposite directions and
    # checkpointing is what buys the batch. Costs ~30% step time.
    grad_checkpointing: bool = True

    def __post_init__(self):
        self.action_frames = DataConfig().frame_skip


# ---- Loss -------------------------------------------------------------------
@dataclass
class LossConfig:
    variant: str = "aux"                    # "pure" | "aux"
    sigreg_lambda: float = 0.1              # paper default; only effective knob
    sigreg_projections: int = 1024
    sigreg_knots: int = 17
    # MEASURED, and the reason these are a real knob rather than a garnish:
    # collapsing the latent is worth ~0.80 of pred_loss to the optimiser, while
    # the aux terms at 0.05 cost it at most 0.008 in total (the normalised
    # targets have std 0.36 and 0.15, so a mean-predictor loses 0.05*0.36^2 +
    # 0.05*0.15^2). That is 100x too weak to influence anything. aux_scale
    # multiplies all three so the ablation is one number.
    aux_y_lambda: float = 0.05              # Mario vertical position
    aux_scroll_lambda: float = 0.05         # camera scroll offset (fixes aliasing)
    aux_alive_lambda: float = 0.02          # dead-in-N signal
    aux_scale: float = 1.0                  # global multiplier on the three above

    # ---- inverse dynamics ---------------------------------------------------
    # Predict the ACTION that caused a transition, from the two latents around
    # it. Added after the first fully-trained checkpoint was measured to ignore
    # its actions entirely (effect ratio 0.041, probe x-gap -3.2 px), which made
    # every CEM candidate look identical and stalled the planner at x=435/3161.
    #
    # THIS IS NOT SUPERVISION IN THE SENSE `variant: aux` IS. The aux heads read
    # RAM state (world_y, scroll, dies_in_5) that the model is otherwise never
    # told. Inverse dynamics uses only frames and actions -- both already inputs
    # to the model -- so a `pure` run with these lambdas on is still fully
    # self-supervised and the claim survives.
    #
    # inv_real acts on consecutive REAL latents and forces the ENCODER to keep
    # action-discriminative information, which makes z_target itself differ by
    # action so the predictor has something to gain from reading `a`.
    # inv_pred acts on (z, zhat) pairs and is the direct fix: it makes the
    # PREDICTOR's output depend on the action or the term cannot be minimised.
    inv_real_lambda: float = 1.0
    inv_pred_lambda: float = 1.0

    # Stage 3 lambda-sweep. NOTE: this is SIGReg lambda, so each value is a FULL
    # training run (~3.2 h H100, ~$17), not a cheap SAE refit. Budget accordingly.
    sigreg_sweep: tuple = (0.01, 0.1, 0.5)


# ---- Eval gates -------------------------------------------------------------
@dataclass
class GateConfig:
    min_x_probe_r2: float = 0.95
    min_y_probe_r2: float = 0.80            # LeMario got 0.188 here — the target
    min_scroll_probe_r2: float = 0.90
    aliasing_px: int = 500
    min_aliasing_margin: float = 0.10       # reported; see gates.aliasing_usable_range
    # The GATE. Latent distance must stay informative out to at least this many
    # world px, because that is the span a single planner sub-goal covers.
    # plan.PlanConfig.subgoal_px is 200 and was chosen on intuition; the measured
    # saturation point on the first trained checkpoint is also ~200, so the gate
    # and the planner are asking the same question.
    aliasing_px_usable: int = 200
    # ---- action conditioning ------------------------------------------------
    # MEASURED on the first fully-trained checkpoint (41,160 steps, aux variant):
    # rolling `run_right` for 8 macros vs `run_left` for 8 macros moved the
    # predicted latent by 0.759, against 18.325 between two real frames 8 steps
    # apart -- a ratio of 0.041. The predictor was ignoring the action, and the
    # planner therefore oscillated (164 rights, 120 lefts) and stalled at
    # x=435/3161.
    #
    # Nothing caught it. scripts/tests/test_model.py asserts the AdaLN-Zero gradient path
    # UNSTICKS at step 1, which is necessary and not sufficient: gradient
    # reaching the action encoder does not mean the converged model uses it.
    # This is the sufficient version, and it is a GATE because a world model that
    # ignores actions is not a world model, however good its probes look.
    min_action_effect_ratio: float = 0.25   # ||z_a - z_b|| / ||z between real frames||
    min_action_x_gap_px: float = 20.0       # probe-decoded x, run_right minus run_left
    probe_val_frac: float = 0.2
    probe_trajectories: int = 60            # Bai's split protocol, for comparability


# ---- SAE (Stage 3 inspect) --------------------------------------------------
@dataclass
class SAEConfig:
    d_model: int = 192                      # MUST be passed explicitly to train_sae;
                                            # the `aquin sae train` CLI cannot set it
    n_features: int = 2048                  # aquin default is 32768 -> 170x overcomplete at d=192
    placeholder_model_id: str = "gpt2"      # satisfies aquin's catalog config lookup only
    chunk_vectors: int = 65536              # rows per chunk_*.pt


# ---- Targets we are trying to beat (LeMario) --------------------------------
# HOW TO READ mse_*: pred_loss is MSE in a latent space the model chooses its own
# scale for, so the raw number means nothing alone. The encoder ends in
# BatchNorm(affine=False), which pins var(z)=1 per coordinate, so a mean-predictor
# scores exactly 1.0 and gain = 1 - mse. That is why `gain` is the comparable
# column and mse is the convenience one.
#
# AND NEITHER IS VALID WITHOUT eff_dim. A rank-1 latent still has unit variance
# per coordinate after that BN (every coordinate perfectly correlated), so the
# predictor only has to predict ONE SCALAR: measured on the first real dry run,
# pred_loss 0.061 -> gain 0.939, which would "beat" LeMario's 0.455 while eff_dim
# sat at 1.17. Quote gain and eff_dim together or do not quote gain.
# RAW MSE IS NOT COMPARABLE ACROSS LATENT SCALES, and the original targets here
# were. LeMario reports mse_5step 0.07772 AND gain_5step 0.455 together, which
# pins their latent variance at 0.07772/(1-0.455) = 0.1426 (cross-checks: their
# 1-step implies gain 0.903). Ours is exactly 1.0, forced by BN(affine=False).
# Identical predictive quality therefore shows up as a 7.0x larger MSE for us,
# and the original "mse_5step < 0.05" target silently meant gain > 0.95.
# `comparable` marks the columns that survive the scale difference.
TARGETS = {
    "mse_1step":       dict(lemario=0.01377, target=0.10,   cmp="lt", comparable=False,
                            note="their scale; ours is 7.0x larger for equal quality"),
    "mse_5step":       dict(lemario=0.07772, target=0.40,   cmp="lt", comparable=False,
                            note="0.40 on our scale == gain 0.60; their 0.05 == gain 0.95"),
    "gain_5step":      dict(lemario=0.455,   target=0.60,   cmp="gt", comparable=True),
    # R2 is scale-invariant, so the probe columns compare directly. This is the
    # headline: 0.188 is the number the whole project exists to move.
    "x_probe_r2":      dict(lemario=0.997,   target=0.97,   cmp="gt", comparable=True),
    "y_probe_r2":      dict(lemario=0.188,   target=0.90,   cmp="gt", comparable=True),
    "scroll_probe_r2": dict(lemario=None,    target=0.90,   cmp="gt", comparable=True),
}


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    seed: int = 0
    run_dir: Path = Path("~/.aqmario/runs").expanduser()


CFG = Config()
