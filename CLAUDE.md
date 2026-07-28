# GrimaceGuide

Computer-vision app that scores feline pain via the Feline Grimace Scale (FGS). Users upload a
cat photo; facial landmarks are extracted (via a remote API or local model) and scored across 5
action units (ears, eyes, muzzle, whiskers, head), each 0-2. Total >= 4 (normalized >= 0.39) flags
likely pain. See README.md for the full FGS scoring rubric and references.

## Running

```bash
python main.py                          # launch the Kivy GUI
python scripts/smoke_analyze.py <img>   # CLI smoke test of the analysis pipeline, no GUI
pytest                                   # run tests (tests/test_scoring.py — pure-Python, no Kivy)
```

Kivy requires `MPLBACKEND=Agg` to be set before import (already handled in `main.py`) to avoid
conflicting with matplotlib's GUI backend.

## Architecture — mid-refactor

The codebase is actively migrating from a monolithic Kivy app onto a layered architecture. Current
branch: `refactor/extract-core`. Expect to find **both** styles side by side; check which layer
you're touching before assuming a pattern.

- `grimaceguide/core/` — pure-Python domain layer: `models.py` (dataclasses like
  `GrimaceResult`, `ActionUnitBreakdown`, `LandmarkSet`), `scoring.py`, `landmarks.py`,
  `api_client.py`, `exceptions.py`. No Kivy dependency — this is what `tests/` exercises directly.
- `grimaceguide/services/analysis_service.py` — orchestrates api_client + scoring + repository
  for a single `analyze(image_path)` call.
- `grimaceguide/infrastructure/` — concrete adapters: `repository.py` (SQLite persistence),
  `storage.py`.
- `grimaceguide/container.py` — the **only** place that wires concrete implementations
  (`LandmarkAPIClient`, `SQLiteResultRepository`) into `AnalysisService`. Everything else should
  depend on abstractions, not construct these directly.
- `grimaceguide/fgsScoreCalc.py` — legacy pure-Python scoring implementation. `core/scoring.py`
  no longer depends on it (the raw geometry was ported into `core/scoring.py` directly, operating
  on typed `Landmark`/`LandmarkSet` objects). The file itself stays, still imported by
  `grimaceguide/api.py` for the legacy Kivy path — don't assume it's dead code.
- `grimaceguide/ui/` — Kivy app (`app.py`, `widgets.py`, `popups.py`, `camera_cv.py`) — the
  original/legacy entry point, still wired to `database.py` + `grimace_scores.db` rather than the
  new service layer.

## Two databases — don't conflate them

- `grimace_scores.db` (project root) — used by the legacy Kivy app via `grimaceguide/database.py`
  and `config.DATABASE_PATH`.
- `analyses.db` (project root) — used by the new `AnalysisService` / `SQLiteResultRepository` path
  via `container.py`, deliberately kept separate so refactor work doesn't touch the live app's data.

Override either via env vars when testing: `GG_API_URL`, `GG_ANALYSES_DB`.

## Conventions

- New code should go through `core/` + `services/` + `infrastructure/`, wired via
  `container.build_service()` — not the legacy `fgsScoreCalc`/`database.py` path directly.
- Domain errors are typed exceptions from `core/exceptions.py`
  (`ImageLoadError`, `LandmarkAPIError`, `ScoringError`, `GrimaceGuideError`) — catch these
  specifically rather than bare `Exception` (see `scripts/smoke_analyze.py` for the pattern).
- `core/` and `tests/` must stay Kivy-free so scoring logic is testable without a display/GUI
  context.
- `PAIN_THRESHOLD = 0.39` (core/scoring.py) is the FGS-validated cutoff — don't change without a
  reason tied to the FGS literature (see README references).