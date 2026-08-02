# Heatmap decoding: sub-pixel refinement (`ml/model.py::decode_heatmaps`)

## Why this was investigated

`docs/au_threshold_calibration.md` calibrated every per-AU threshold against CatFLW **ground-truth**
landmarks. `ml/compare_predicted_geometry.py` was then written to check whether those thresholds
still hold when the same formulas run on **predicted** landmarks, which is what production actually
scores.

The aggregate answer was reassuring — predicted-landmark geometry distributions track the
ground-truth ones closely, so the thresholds sit at roughly the same percentiles. But scoring
agreement per image was not:

| AU | agreement | majority-class baseline | Cohen's κ |
|---|---|---|---|
| ears | 91.6% | 89.1% | **0.57** |
| head | 82.3% | 79.1% | **0.52** |
| eyes | 91.6% | 93.2% | 0.10 |
| whiskers | 87.1% | 88.7% | 0.05 |
| muzzle | 64.6% | 77.2% | 0.01 |
| pain flag | 87.5% | 90.7% | 0.20 |

The high agreement percentages are an artifact of class imbalance — most images score the same
value for most AUs, so a predictor that always guessed the majority class would score comparably or
better. Cohen's κ removes that chance agreement and shows only **ears and head** carry real signal.

## Root cause: landmark precision vs. feature size

Mean per-landmark prediction error is **5.56% of face width**. Measured against the distance each
AU's formula actually needs to resolve (both normalized by face width, val split, n=311):

| measured distance | size | error | noise-to-signal |
|---|---|---|---|
| ear tip-to-tip | 1.776 | 0.045 | **2.5%** |
| eye width (denominator) | 0.319 | 0.060 | 18.9% |
| whisker span | 0.328 | 0.059 | 18.0% |
| eyelid opening (numerator) | 0.242 | 0.063 | 26.1% |
| mouth width (numerator) | 0.181 | 0.040 | 21.9% |
| nose→mouth centre (denominator) | 0.195 | 0.043 | 22.2% |

Ears work because they span nearly two face widths, so a 5.6% error is negligible. The failing AUs
measure features spanning only 0.18–0.33 face widths, so the same absolute error is 18–26% of the
quantity being measured. Muzzle is worst because `muzzle_ratio` divides one ~22%-noisy small
distance by another, compounding both.

This is a precision floor, not a calibration problem: no threshold retuning can rescue a
measurement whose noise is a quarter of the signal.

## The fix attempted

`decode_heatmaps` originally took a plain `argmax`, which can only return integer positions on the
heatmap grid. At `HEATMAP_STRIDE=4` and a 256px crop that grid is 64×64, so every point was
quantized to 1/64th of the crop regardless of model quality. Three training-free refinements were
added and measured (all applied post-hoc at decode time — no retraining):

- `quarter` — shift a quarter-pixel toward the larger neighbour (SimpleBaseline's trick)
- `parabolic` — fit a parabola through the peak and its two neighbours per axis
- `soft` — intensity-weighted centroid over a 5×5 window around the peak

### Results (val split, n=311)

| mode | landmark err | ears | eyes | muzzle | whiskers | head | pain | total *r* |
|---|---|---|---|---|---|---|---|---|
| argmax | 5.56% | 0.572 | 0.097 | 0.014 | 0.047 | 0.525 | 0.196 | 0.313 |
| quarter | 5.18% | 0.610 | -0.017 | 0.036 | -0.053 | 0.564 | 0.145 | 0.341 |
| **parabolic** | **5.07%** | 0.616 | -0.012 | 0.029 | -0.044 | 0.530 | 0.206 | 0.357 |
| soft | 5.06% | 0.637 | -0.012 | 0.040 | 0.005 | 0.610 | 0.141 | 0.401 |

Paired bootstrap (10k resamples) on landmark error: every refinement beats `argmax` by a
statistically clear margin (`argmax - parabolic` = 0.498%, 95% CI [0.472%, 0.524%]), while
`parabolic` and `soft` are indistinguishable from each other (diff 0.001%, CI [-0.012%, 0.014%]).

**`parabolic` is the default**: statistically tied with `soft` on the one rigorous metric, far
cheaper (3 samples vs a 25-point window), and principled given targets are rendered Gaussians
(`ml/dataset.py`) whose neighbourhood a parabola fits exactly.

## What this did and did not fix

It measurably improved the AUs that already worked (ears κ 0.57→0.62, head 0.53→0.61, total-score
correlation 0.31→0.36) and did **nothing** for eyes, muzzle, or whiskers, which stay at κ≈0.

That is the informative result: grid quantization was only ~0.5 of the 5.56 percentage-point error
budget. The remaining ~5% is genuine model error, so the small-distance AUs remain unresolvable.
Making them trustworthy needs a real precision improvement — a finer heatmap
(`HEATMAP_STRIDE=2`), higher input resolution, or a longer/better-regularized training run — not a
better decoder. The eyes/whiskers κ values drifting slightly negative is noise around zero, not a
regression; they were never meaningfully above zero.

## Is heatmap resolution the bottleneck? No.

The obvious next lever looked like `HEATMAP_STRIDE=2` (a 128×128 heatmap). Measuring the error in
units of heatmap cells says otherwise — median error is 0.79 cells, and only 27.9% of errors fall
below half a cell, which is the regime where a finer grid would help. The model is not saturating
the grid it already has; sub-pixel refinement already collected that win.

The decisive comparison is per-AU. The landmarks the *failing* AUs depend on are localized just as
precisely as the ones the *working* AU depends on:

| AU | median landmark error | κ |
|---|---|---|
| ears (works) | 0.71 cells | **0.62** |
| muzzle | 0.72 cells | 0.01 |
| eyes | 0.89 cells | ~0 |
| whiskers | 1.09 cells | 0.05 |

Muzzle's landmarks are as accurate as ears' and muzzle is still at chance, so "the model is worse at
finding mouths than ears" is not the explanation. Feature size is. Closing the gap to ears' 2.5%
noise-to-signal would need landmark error to drop from ~5% of face width to ~0.6% — roughly **8×**.
A finer heatmap is worth maybe 5–10%; a larger backbone with more data might plausibly reach 2×.
Nothing on the landmark-precision path closes 8×.

## Can the formulas be reformulated? No.

If short measured distances are the problem, the natural fix is formulas built on longer baselines
or on more points (averaging n independent points cuts noise by √n). Tested on the val split, with
each variant's threshold calibrated on ground truth to reproduce the shipped class proportions:

| eye formula | κ |
|---|---|
| current (eyelid opening ÷ eye width) | -0.01 |
| long baseline (eyelid opening ÷ face width) | -0.03 |
| PCA ellipse over 4 contour points | -0.02 |
| PCA ellipse over 4 contour + 4 pupil points | -0.01 |

All at chance. The reason shows up when the metrics are compared as *continuous* values, before any
thresholding — correlation between the ground-truth-computed and predicted-computed value, plus κ at
progressively stricter cutoffs:

| metric | *r* | κ@50% | κ@25% | κ@10% | κ@7% |
|---|---|---|---|---|---|
| ears angle (control) | **0.88** | 0.74 | 0.76 | 0.62 | 0.67 |
| ears tip/base ratio (control) | **0.66** | 0.59 | 0.62 | 0.45 | 0.40 |
| eyes, ÷ eye width | 0.09 | 0.00 | 0.01 | 0.03 | -0.01 |
| eyes, ÷ face width | 0.15 | 0.10 | 0.05 | -0.06 | -0.03 |
| muzzle ratio | 0.08 | 0.00 | 0.08 | 0.01 | -0.03 |
| muzzle mouth width ÷ face width | -0.02 | 0.07 | -0.08 | -0.08 | -0.04 |
| whiskers spread | 0.01 | -0.03 | 0.07 | 0.10 | 0.04 |

κ at a median split is the easiest possible version of the task, and the three failing AUs are still
at zero there — so this is not "signal too coarse for a strict cutoff", it is no signal at any
granularity. Swapping the noisy denominator for face width does help slightly (*r* 0.09 → 0.15),
exactly as the noise analysis predicts, and is still useless in absolute terms because the
numerator — an inherently short distance — dominates.

**Conclusion:** eyes, muzzle and whiskers cannot be rescued by thresholds, decoding, resolution, or
reformulation over these landmarks. The remaining options are a substantially more precise landmark
model, or dropping hand-crafted geometry for those AUs in favour of a learned image→score model —
which is what both automated-FGS papers in `README.md` actually did (Steagall et al. fed raw
landmarks to XGBoost; PMC10238514 used RF/MLP on coordinates or a ResNet on images). Neither
hand-built per-AU geometric formulas. These results are a concrete empirical argument for why.

Acted on in the meantime: `core/models.py` marks eyes/muzzle/whiskers as
`LOW_CONFIDENCE_ACTION_UNITS`, and the Kivy results panel and `scripts/smoke_analyze.py` both flag
them rather than presenting five equally-weighted scores.

## Caveats

- **The evaluation split leaked at the cat level — this is the biggest caveat here.** CatFLW stems
  are `<subject>_<shot>` and the dataset is ~339 cats with a median of 6 photos each. The split used
  for both training and this evaluation was index-level, so **310 of 311 val images had another
  photo of the same cat in training** — val measured memorization, not generalization.
  `ml/train.py::_subject_disjoint_split` now splits whole cats (verified: 289 train cats / 50 val
  cats, zero shared) and is the default, but `ml/checkpoints/*.pt` predate it and were trained
  leaky. Consequences:
  - **The ears/head κ ≈ 0.5–0.6 figures are an optimistic ceiling, not a validation.** They cannot be
    corrected by re-splitting this evaluation, because the checkpoint saw photos of essentially every
    cat in the dataset — an honest number requires retraining under the subject-disjoint split.
  - **The κ ≈ 0 results for eyes/muzzle/whiskers get stronger, not weaker.** Those AUs failed even
    with the advantage of being tested on cats the model had memorized.
- **The split assumption is also unverified in the other direction.** `val_stems` assumes the Colab
  run used the default `--val-split 0.15 --seed 42`; those arguments were never recorded.
- κ deflates under heavy class imbalance, so treat eyes and whiskers as "unproven / likely weak"
  rather than definitively dead. **Muzzle** is the unambiguous one: its raw agreement (64.6%) falls
  *below* its own majority-class baseline (77.2%).
- Per-metric hit rates in the script's output are reported one metric at a time, but `_score_ears`
  combines its two metrics with an OR, so those rows understate the true joint rate. They are still
  valid for comparing *across* landmark sources, which is what the script exists to do.
- CatFLW has no FGS pain labels, so "agreement" here means agreement with ground-truth-*landmark*
  scoring, not with veterinary pain assessment. This measures whether the model's landmarks are
  precise enough to reproduce the scores its own formulas would give on perfect landmarks — a
  necessary, not sufficient, condition for the scores being clinically meaningful.
- A multi-agent adversarial review of this methodology was run but hit a session token limit with
  14 of 21 agents incomplete, so it should not be treated as a clean validation.

## Reproducing

```bash
python -m ml.compare_predicted_geometry \
    --data-dir "ml/data/CatFLW dataset" \
    --checkpoint ml/checkpoints/landmark_net.pt \
    --bbox-checkpoint ml/checkpoints/bbox_net.pt
```
