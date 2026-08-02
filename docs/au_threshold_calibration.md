# Ears/muzzle/whiskers/head AU threshold calibration (`core/scoring.py`)

## Background

Like the eye-openness ratio (see `docs/eye_scoring_calibration.md`), the per-AU geometric
thresholds for ears, muzzle, whiskers, and head were ported from the original prototype without
ever being derived from FGS literature — neither automated-FGS paper cited in `README.md` defines
hand-crafted per-AU geometric formulas (both use end-to-end learned models on raw landmarks
instead). This calibrates all four against real CatFLW ground-truth data using the same method:
compute each AU's raw geometric value across all 2079 CatFLW images (using ground-truth landmarks,
not model predictions, so this measures what the *formula* does independent of any model's
accuracy), via `ml/calibrate_thresholds.py`, which imports the exact production formulas from
`core/scoring.py` (`_ear_geometry`, `_muzzle_geometry`, `_whisker_geometry`, `_head_geometry`)
rather than reimplementing them, so the calibration can't silently drift from what actually ships.

```bash
python -m ml.calibrate_thresholds --data-dir "ml/data/CatFLW dataset"
```

## Ears (`_score_ears`)

`avg_ear_angle` (vertical-angle of base-to-tip per ear, averaged) and `tip_base_ratio`
(tip-to-tip distance / base-to-base distance) combine via OR: `OBVIOUS` if either exceeds its
cutoff, else `MODERATE` if either exceeds its lower cutoff.

The original cutoffs (`OBVIOUS`: angle>45 or ratio>1.5; `MODERATE`: angle>25 or ratio>1.2) were so
loose that **50.8%** of a general (non-distressed) photo population hit `OBVIOUS`, and **98.9%**
hit `MODERATE`-or-above — the AU was flagging almost every cat as at least mildly ear-tense,
carrying near-zero discriminating power (the mirror-image failure of the original eyes bug, which
was unreachable in the other direction).

| threshold | old | new | population hit (joint OR) |
|---|---|---|---|
| OBVIOUS | angle>45 or ratio>1.5 | angle>55 or ratio>1.9 | 6.0% |
| MODERATE-or-above | angle>25 or ratio>1.2 | angle>50 or ratio>1.7 | 15.1% |

Raised (2026-08-02) so `OBVIOUS`/`MODERATE`-or-above flag roughly the top 6%/15% of a general
population instead of the majority.

## Muzzle (`_score_muzzle`)

`muzzle_ratio` (mouth width / nose-to-mouth distance), single threshold each. Checked against the
same 2079-image distribution:

| threshold | value | population hit |
|---|---|---|
| OBVIOUS | ratio > 1.5 | 4.2% |
| MODERATE-or-above | ratio > 1.1 | 25.8% |

**No change.** Both cutoffs already land at reasonable minority-flagging rates — unlike ears, this
AU wasn't actually broken; worth confirming rather than assuming a fix was needed everywhere.

## Whiskers (`_score_whiskers`)

Two separate issues, of different kinds:

**1. Fixed — the `normalized_spread` cutoff was unreachable.** In the "not straight, not
forward-pointing" branch, `ABSENT` required `normalized_spread > 0.8`, but the observed max across
all 2079 photos was **0.40** — this branch could never actually resolve to `ABSENT`, only
`MODERATE`. Restricting to that branch specifically (n=1287), the spread distribution's 80th
percentile is 0.242. Lowered the cutoff to **0.24** (2026-08-02) so the loosest-spread ~20% of
that branch now register `ABSENT` as intended.

**2. Flagged, not fixed — `OBVIOUS` (`is_straight and is_forward`) is structurally near-unreachable.**
Measuring `is_forward` (`left_whisker_5.x < left_whisker_1.x` and mirrored on the right) directly:
only 4/2079 images (0.19%) satisfy *both* sides simultaneously; each side alone is only satisfied
~3% of the time, consistent with the two sides being close to statistically independent (product of
marginals ≈ observed joint rate). This isn't a threshold that can be recalibrated — `is_forward`
and `is_straight` are boolean geometric comparisons, not continuous ratios with a movable cutoff.
Whether this reflects a genuinely rare pose (forward-swept-and-straight whiskers are a real,
uncommon "alert/tense" configuration - plausible given `OBVIOUS` is supposed to be the severe end)
or a mis-specified proxy (the `CATFLW_INDEX_FOR_LABEL` confidence notes in `core/landmarks.py`
flag whisker points 1/3/5 as ordered by vertical rank, not lateral/fan position, which is what
`is_forward`'s x-comparison assumes) is unresolved — left as a follow-up if whiskers' `OBVIOUS`
rate ever needs investigating, rather than redesigning the geometry under a calibration pass.

## Head (`_score_head`)

**Bug, not a threshold issue.** The original formula normalized the chin-to-ear-base vertical
offset by its own absolute value: `head_height_proxy = abs(diff)`, `normalized_diff = diff /
head_height_proxy`. This is self-referential — `diff / abs(diff)` is just `sign(diff)`, always
exactly `+1` or `-1` regardless of magnitude, carrying zero continuous information. Confirmed
empirically: across all 2079 photos, `normalized_diff` was already clipped to exactly `-1` (1% of
images) or `+1` (the rest) before any threshold was applied — no choice of `OBVIOUS`/`MODERATE`
cutoff strictly between -1 and 1 could have changed this, since the signal itself carries no
magnitude, only sign. `MODERATE` was consequently unreachable in practice (only hit in the
degenerate `head_height_proxy == 0` case).

Fixed (2026-08-02) by normalizing by `face_width` instead — the same pattern `_score_whiskers`
already uses for its spread ratio — giving a real continuous signal:

| stat | value |
|---|---|
| n | 2079 |
| mean | 1.096 |
| median | 1.140 |
| std | 0.267 |
| p1 | -0.123 |
| p10 | 0.861 |
| p90 | 1.322 |
| p99 | 1.527 |

Set `OBVIOUS` at `normalized_diff > 1.32` (~p90) and `MODERATE` at `> 0.86` (~p10), so the bottom
~10% (head held unusually high relative to face width) reads `ABSENT`, the top ~10% (chin unusually
low) reads `OBVIOUS`, and the middle ~80% reads `MODERATE` — extremes as minorities, matching the
calibration philosophy used for the other AUs.

## Caveat

Same as `docs/eye_scoring_calibration.md`: CatFLW is not a pain-labeled dataset. This calibration
validates that each formula's geometry is sane and that thresholds sit at real, reachable, minority
percentiles rather than being universally-hit or practically-unreachable — it cannot validate
against true FGS AU scores (0/1/2 as assessed by a veterinary rater), since no such ground truth
exists in this dataset. If a labeled dataset with actual per-AU FGS scores becomes available,
recalibrate against that instead. The head AU also has a deeper limitation worth remembering: FGS's
actual head-position AU compares head position to the *shoulder line*, but no shoulder landmarks
exist in CatFLW's 48-point facial scheme — `normalized_diff` is a face-only proxy, not a literal
head-vs-shoulder measurement, regardless of normalization.
