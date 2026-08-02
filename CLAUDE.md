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
- **Only the ears and head AUs are currently trustworthy from *predicted* landmarks.** Scored
  against ground-truth-landmark results on held-out CatFLW images, ears/head reach Cohen's κ ≈
  0.5–0.6 while eyes/whiskers/muzzle sit at κ ≈ 0 — the landmark model's ~5% -of-face-width error is
  18–26% of the short distances those three AUs measure. This is a model-precision limit, not a
  threshold problem, so don't try to fix it by moving cutoffs. See `docs/heatmap_decoding.md`;
  re-measure with `ml/compare_predicted_geometry.py` after any checkpoint or decoding change.

## History

The app previously ran on a monolithic Kivy implementation (`database.py`, `api.py`,
`fgsScoreCalc.py`, writing to `grimace_scores.db`). That path has been fully replaced by the
layered architecture above and the legacy files were deleted; `grimace_scores.db` remains on disk
as historical data from that era but nothing reads or writes it anymore.