# AQ-Mario

**An implementation of [LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun) for an
action-conditioned world model — and a study of what its SIGReg objective needs in that setting.**

Run as an Aquin interpretability-in-the-loop demonstration. The claim: a model that predicts
perfectly can still fail to plan, and an interp gate catches that before you waste a training run.
Mario is the vehicle; the reusable gate harness is the product.

> LeJEPA identifies the isotropic Gaussian as the optimal embedding distribution for a JEPA and
> introduces **SIGReg** to enforce it, arguing that this removes the need for anti-collapse
> heuristics. This repository takes SIGReg out of image self-supervised learning and into a world
> model over video with actions, evaluated by whether a *control-relevant state variable* survives
> in the representation.

A 9.8M-parameter JEPA trained on 8,000 episodes of Super Mario Bros 1-1 (4.88M observations),
predicting 5 steps ahead in a 192-dimensional latent space. The train fits `method: jepa`
([`src/algo/jepa.py`](src/algo/jepa.py)); `eval.min_score: 1.0` is the gate — it fails the run when
any probe R² falls under its threshold, which is how LeMario's y-probe collapse (R²=0.188) gets
caught at epoch 2 instead of after planning fails.

| Doc | What it is |
|:---|:---|
| **[DOCS.md](DOCS.md)** | Full write-up — experiment, measurements, SIGReg study, walkthrough, [dataset](DOCS.md#part-v--dataset), [stage log](DOCS.md#part-vi--stage-log) |

---

## Findings

Six results, in decreasing order of how much they depend on our training budget.

**1. SIGReg has a closed-form null, and without it λ is not comparable across runs.**
Under the null, `E|φₙ − φ|² = (1 − e^{−t²})/n` exactly, so

```
E[T | z ~ N(0, I)] = √π · (1 − 1/√2) / n  =  0.51914 / n
```

Verified within 3% from n=64 to n=2048. At batch 64 the term **floors at 0.0013** however Gaussian
the encoder becomes — so raw SIGReg values from different batch sizes are not comparable, and
neither are their λ. Report `T / null(n)`.

**2. Its power to detect collapse scales with batch size.** Ratio to null for a rank-*r* latent
after BatchNorm, D=192:

| rank | n=32 | n=64 | n=128 | n=512 | n=2048 |
|---|---|---|---|---|---|
| 2 | 2.8× | 5.6× | 10.9× | 41.8× | 173.8× |
| 16 | 0.8× | 1.2× | 2.0× | 7.2× | 28.5× |
| 192 (healthy) | 0.4× | 0.5× | 0.5× | 1.0× | 2.8× |

At n=32 a collapsed latent is barely separable from a healthy one, so the term cannot resist what it
cannot see. A world-model step encodes `batch × window` images, so memory pushes the batch *down*,
into exactly that regime. Activation checkpointing buys statistical power, not just throughput.

**3. A trailing BatchNorm cuts the collapse signal 25×.** The projection head must end in BatchNorm
(a trailing LayerNorm puts embeddings on a fixed-radius sphere and fights the Gaussian target), but
BN pins the covariance diagonal for free: a rank-8 latent reads **254× null raw, 10× after BN**.

**4. Correlated video batches break the iid null.** A batch is `B` windows × `W` near-identical
frames, so it holds `B` independent samples, not `B×W`. Scored naively, a *healthy* latent reads
6.7× null and looks collapsed. Compute the statistic per frame position and average.

**5. The stop-gradient was still required.** 500 steps, identical otherwise:

| batch | stop-grad | effective dim (init → 500) | gain₅ |
|---|---|---|---|
| 32 | ✗ | 7.9 → **2.0** | 0.969 *(fake)* |
| 128 | ✗ | 11.1 → **2.0** | 0.978 *(fake)* |
| 128 | ✓ | 11.1 → **8.4** | 0.651 |

λ alone could not substitute: collapsing buys the optimiser 0.80 of prediction loss, while λ=5's
SIGReg term costs 0.18. With a shared encoder, `MSE(ẑ, z_target)` is *directly* minimised by making
the target constant — collapse is the objective's global optimum unless something removes the
incentive.

**6. Preventing collapse did not preserve usable state.** The SIGReg-only model did **not** collapse
and still could not report the character's height:

| probe (frozen encoder, 60 held-out trajectories) | SIGReg only | + aux heads (0.07M params) |
|---|---|---|
| **height (y)** | **−0.383** | **0.796** |
| position (x) | 0.451 | 0.870 |
| camera (scroll) | 0.464 | 0.878 |
| death within 5 steps (AUC) | 0.628 | 0.927 |

An isotropic-Gaussian embedding is a **floor, not a sufficient condition**. Which is the argument for
gating on a probe of the state the task needs, per checkpoint — it costs minutes and is the only
measurement here a collapsed-but-confident model cannot fake.

> **Read the limitations before citing any of this.** All numbers come from a 4,000-step run
> (~4% of the planned schedule) on one level of one game. Finding 6's aux column is *trained* to make
> height decodable and then measured for it — it shows a cheap head fixes the problem, not that the
> objective learned a better representation. See [DOCS §15](DOCS.md#15--limitations).

---

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH="src:scripts"
python -m algo.tracker          # stage board, probed from real state
pytest -q                       # automated checks
```

Nothing in this repository is hand-ticked: `src/algo/tracker.py` derives progress from files on disk,
shard counts, `metrics.jsonl` contents and gate JSON. Write a live board with
`python -m algo.tracker --write` → `STATUS.md`.

## Reproducing

Compute runs on [Modal](https://modal.com); the code has no other cloud dependency.

```bash
python scripts/helpers/validate_ram.py --lock
modal run scripts/helpers/modal_app.py::preflight
modal run scripts/helpers/modal_app.py::stage1_data  --shards 16 --episodes-per-shard 250
modal run scripts/helpers/modal_app.py::stage1_ppo   --steps 4000000 --target-x 2400
modal run scripts/helpers/modal_app.py::stage1_data_ppo --shards 16 --episodes-per-shard 250

modal run scripts/helpers/modal_app.py::stage2_train --variant aux  --epochs 3
modal run scripts/helpers/modal_app.py::stage2_train --variant pure --epochs 3

modal run scripts/helpers/modal_app.py::stage3_gates --ckpt /runs/<tag>/jepa.pt
modal run scripts/helpers/modal_app.py::stage4_plan  --ckpt /runs/<tag>/jepa.pt
```

Every setting lives in [`recipe.yaml`](recipe.yaml) with the measurement that justifies it, and a
test asserts the recipe never drifts from the code.

## Layout

```
DOCS.md             full project write-up (incl. dataset + stage log)
LICENSE.md          Apache License 2.0
recipe.yaml         pinned hyperparameters
assets/
  figures/          report figures (fig01–fig08, curves)
  gifs/             output_*.gif
  vids/             output_*.mp4
  episode_strip.png
src/
  algo/             model, train, losses, gates, plan, tracker, jepa
  dataset/          ram ground truth, PPO env helpers
scripts/
  helpers/          Modal app + CLI runners
  tests/            automated checks
```

## Dataset

**The 27.4 GB dataset is not in this repository.** It is hosted at
[Aquinlabs/aq-mario-smb1](https://huggingface.co/datasets/Aquinlabs/aq-mario-smb1)
and can also be regenerated on a Modal volume with the Stage 1 commands above
(deterministic given the seeds in `recipe.yaml`).

| | |
|---|---|
| 8,000 episodes | 4,000 jump-biased random + 4,000 from a trained PPO explorer |
| 4,877,368 observations | 224×224 RGB at frame-skip 2 |
| 27.4 GB compressed | 992 GB of raw pixels; NES artwork deflates 27× |
| per frame | buttons held, and `world_x`, `world_y`, `scroll`, `power`, `dies_in_5` from console RAM |

- Dataset: [Aquinlabs/aq-mario-smb1](https://huggingface.co/datasets/Aquinlabs/aq-mario-smb1)
- Model: [Aquinlabs/aq-mario](https://huggingface.co/Aquinlabs/aq-mario)
- Details: [`DOCS.md` Part V](DOCS.md#part-v--dataset)

## Licence

Code: [Apache License 2.0](LICENSE.md) © 2026 Aquin Labs Private Limited.
LeJEPA paper material (where redistributed) is under CC BY-SA 4.0, © Balestriero & LeCun.

## Citation

```bibtex
@software{aqmario2026,
  title  = {AQ-Mario: SIGReg in an action-conditioned world model},
  year   = {2026},
  url    = {https://github.com/sachin1705s/aq-mario}
}
```
