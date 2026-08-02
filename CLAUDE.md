# GrimaceGuide

Computer-vision app that scores feline pain via the Feline Grimace Scale (FGS). Users upload a
cat photo; facial landmarks are extracted (via a remote API or local model) and scored across 5
action units (ears, eyes, muzzle, whiskers, head), each 0-2. Total >= 4 (normalized >= 0.39) flags
likely pain. See README.md for the full FGS scoring rubric and references.

## Running

```bash
python main.py                          # launch the Kivy GUI
python scripts/smoke_analyze.py <img>   # CLI smoke test of the analysis pipeline, no GUI
pytest                                   # run tests (pure-Python, no Kivy)
```

## Architecture

Layered: `core/` (pure-Python domain, no Kivy) → `services/` (use cases) → `infrastructure/`
(concrete adapters) → `ui/` (Kivy). `container.py` is the single composition root wiring it
together; everything else depends on abstractions, not concrete implementations.

- `grimaceguide/core/` — `models.py` (dataclasses: `GrimaceResult`, `ActionUnitBreakdown`,
  `LandmarkSet`, `AnalysisOutcome`), `scoring.py` (FGS geometry/thresholds, `PAIN_THRESHOLD =
  0.39`), `landmarks.py`, `api_client.py`, `image_processing.py` (incl. `draw_landmarks_overlay`),
  `exceptions.py`. No Kivy dependency — this is what `tests/` exercises directly.
- `grimaceguide/services/analysis_service.py` — `AnalysisService.analyze()`: the single use-case
  that loads an image, calls the landmark API, scores it, renders + persists the overlay image,
  and persists the result — returning an `AnalysisOutcome`.
- `grimaceguide/infrastructure/` — concrete adapters: `repository.py` (`SQLiteResultRepository`,
  `analyses.db`), `storage.py` (`LocalBlobStorage`, rendered overlay images under `processed/`).
- `grimaceguide/container.py` — `build_service()` is the **only** place that wires concrete
  implementations (`LandmarkAPIClient`, `SQLiteResultRepository`, `LocalBlobStorage`) into
  `AnalysisService`. Env var overrides: `GG_API_URL`, `GG_ANALYSES_DB`, `GG_PROCESSED_DIR`.
- `grimaceguide/ui/` — Kivy app (`app.py`, `widgets.py`, `popups.py`, `camera_cv.py`), wired
  through `container.build_service()`.

## Conventions

- All new code goes through `core/` + `services/` + `infrastructure/`, wired via
  `container.build_service()`.
- Domain errors are typed exceptions from `core/exceptions.py`
  (`ImageLoadError`, `LandmarkAPIError`, `ScoringError`, `GrimaceGuideError`) — catch these
  specifically rather than bare `Exception` (see `ui/app.py::_do_api_processing` or
  `scripts/smoke_analyze.py` for the pattern).
- `core/` and `tests/` must stay Kivy-free so scoring logic is testable without a display/GUI
  context.
- `PAIN_THRESHOLD = 0.39` (core/scoring.py) is the FGS-validated cutoff — don't change without a
  reason tied to the FGS literature (see README references).
- The per-AU thresholds inside `_score_ears`/`_score_eyes`/etc. (unlike `PAIN_THRESHOLD`) are
  **not** from FGS literature — no cited paper defines per-AU geometric formulas. All five AUs'
  thresholds have been empirically recalibrated against real CatFLW data; see
  `docs/eye_scoring_calibration.md` (eyes) and `docs/au_threshold_calibration.md` (ears, muzzle,
  whiskers, head) before changing them again. Each `_score_*` function delegates its raw geometry
  to a paired `_*_geometry` helper (e.g. `_ear_geometry`) — `ml/calibrate_thresholds.py` imports
  those directly so any future recalibration runs against the exact production formula.
- **Eyes, muzzle and whiskers are not trustworthy from *predicted* landmarks** — they score Cohen's
  κ ≈ 0 against ground-truth-landmark results, versus ≈ 0.5–0.6 for ears/head. The landmark model's
  ~5%-of-face-width error is 18–26% of the short distances those three AUs measure. This is a
  model-precision limit, not a threshold problem, and it survives reformulation and finer decoding —
  don't try to fix it by moving cutoffs. `core/models.py::LOW_CONFIDENCE_ACTION_UNITS` marks them and
  the UI de-emphasises them. The ears/head figures are an optimistic ceiling, not a validation (see
  the split caveat below). Full derivation in `docs/heatmap_decoding.md`; re-measure with
  `ml/compare_predicted_geometry.py` after any checkpoint or decoding change.
- **Split CatFLW by cat, not by image.** Stems are `<subject>_<shot>` and the dataset is ~339 cats ×
  ~6 photos, so an index-level split leaks: the old default put a same-cat photo in training for
  310 of 311 val images. `ml/train.py::_subject_disjoint_split` (on by default) fixes this, but the
  current `ml/checkpoints/*.pt` predate it and were trained leaky — so every held-out number
  measured on them is optimistic, and an honest one needs a retrain.

## History

The app previously ran on a monolithic Kivy implementation (`database.py`, `api.py`,
`fgsScoreCalc.py`, writing to `grimace_scores.db`). That path has been fully replaced by the
layered architecture above and the legacy files were deleted; `grimace_scores.db` remains on disk
as historical data from that era but nothing reads or writes it anymore.