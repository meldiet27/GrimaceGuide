# Eyes AU threshold calibration (`core/scoring.py::_score_eyes`)

## Background

`_score_eyes` computes an `eye_aspect_ratio = avg_v_dist / avg_h_dist`, where `v_dist` is the
vertical eyelid opening (top-to-bottom landmark distance) and `h_dist` is the true eye width
(outer-to-inner corner distance). This diamond-geometry formula replaced an earlier version that
averaged corner points into a fake top/bottom midpoint (see git history on `core/scoring.py` and
`core/landmarks.py`'s `LANDMARK_LABELS` comment for why — CatFLW-derived landmarks supply one top
point and one bottom point per eye, not two of each).

Unlike `PAIN_THRESHOLD = 0.39` (the actual published Feline Grimace Scale clinical cutoff — see
`README.md`'s references), there is no equivalent published threshold for a per-AU eye-openness
ratio. Checked the two automated-FGS papers cited in `README.md`:

- **"Fully automated deep learning models with smartphone applicability for prediction of pain
  using the Feline Grimace Scale"** (Steagall et al., *Sci Rep* 2023) — uses an XGBoost model
  trained end-to-end on 37 raw landmark coordinates to predict FGS scores directly. No
  intermediate per-AU geometric formula.
- **"Explainable automated pain recognition in cats"** (*Sci Rep* 2023, PMC10238514) — uses raw
  landmark XY coordinates fed into Random Forest / MLP models, or a ResNet50 on aligned images.
  Also no hand-crafted per-AU geometric formula; the paper's focus is on which facial *regions*
  (occlusion analysis) matter, not on defining eye-openness math.

So the `0.25` / `0.5` thresholds inherited from the original ported prototype were never actually
derived from FGS literature — there was nothing to carry over correctly or incorrectly. This
calibration is an empirical check against real data instead, as the next-best grounding available.

## Method

Computed `eye_aspect_ratio` for all 2079 images in the CatFLW dataset, using each image's
ground-truth landmarks (not model predictions — this measures what the *formula* does on real
faces, independent of any model's prediction accuracy).

```python
# see grimaceguide/core/landmarks.py::landmarks_from_catflw_array and
# grimaceguide/core/scoring.py::_label_landmarks for the remapping/lookup used
```

## Results

| stat | value |
|---|---|
| n | 2079 |
| mean | 0.752 |
| median | 0.768 |
| std | 0.147 |
| p1 | 0.383 |
| p5 | 0.477 |
| p10 | 0.548 |
| min | **0.271** (image `00000088_022`) |
| max | 1.183 |

Old thresholds (`OBVIOUS < 0.25`, `MODERATE < 0.5`): 0% of the dataset ever reached `OBVIOUS` — not
even the single most extreme case (0.271). Only 6.3% reached `MODERATE` or below.

Visually inspected the two lowest-ratio images (`00000088_022` at 0.271, `00000091_002` at 0.278):
both show cats with clearly, visibly squinted/narrowed eyes — confirming the formula measures the
right thing, but that even the most extreme real squinting example in a 2000+ photo general
population fell just short of the `OBVIOUS` cutoff.

## Decision

Raised `OBVIOUS` from `0.25` to **`0.30`** (2026-08-02) — just above the observed real-world
minimum, so genuinely extreme squinting (like the two examples above) registers as `OBVIOUS`
instead of maxing out at `MODERATE`. Left `MODERATE` at `0.5` unchanged: only the narrowest ~6% of
a general (non-pain-curated) photo population crosses it, which is a reasonable minority-flagging
threshold rather than one that's too permissive or too strict.

## Caveat

CatFLW is not a pain-labeled dataset — this calibration validates that the geometry is sane and
picks a threshold that isn't unreachable in practice, but it cannot validate against true FGS
squint scores (0/1/2 as assessed by a veterinary rater), since no such ground truth exists in this
dataset. If a labeled dataset with actual FGS eye scores becomes available, recalibrate against
that instead.
