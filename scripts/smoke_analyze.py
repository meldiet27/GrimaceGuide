"""Smoke test: run the new AnalysisService against a local cat image.

Usage (from project root):
    python scripts/smoke_analyze.py path/to/cat.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable regardless of where this is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grimaceguide.container import build_service
from grimaceguide.core.exceptions import (
    GrimaceGuideError,
    ImageLoadError,
    LandmarkAPIError,
    ScoringError,
)
from grimaceguide.core.models import LOW_CONFIDENCE_NOTE


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_analyze.py <image_path>")
        return 2

    image_path = sys.argv[1]

    try:
        service = build_service()
        outcome = service.analyze(image_path)
    except ImageLoadError as exc:
        print(f"[ERROR] Could not load image: {exc}")
        return 1
    except LandmarkAPIError as exc:
        print(f"[ERROR] Landmark API failure: {exc}")
        return 1
    except ScoringError as exc:
        print(f"[ERROR] Scoring failure: {exc}")
        return 1
    except GrimaceGuideError as exc:
        print(f"[ERROR] {exc}")
        return 1

    breakdown = outcome.result.breakdown
    low_confidence = set(breakdown.low_confidence_units)
    per_au = "  ".join(
        f"{name}={score}{'*' if name in low_confidence else ''}"
        for name, score in breakdown.as_dict().items()
    )

    print("=" * 50)
    print(f"File:            {image_path}")
    print(f"Total score:     {breakdown.total} / 10")
    print(f"Normalized:      {breakdown.normalized:.2f}")
    print(f"Pain likely:     {outcome.result.pain_likely}")
    print(f"Per-AU:          {per_au}")
    print(f"Processing time: {outcome.result.processing_ms:.1f} ms")
    print(f"Persisted ID:    {outcome.persisted_id}")
    print("=" * 50)
    print(f"* {LOW_CONFIDENCE_NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())