"""Composition root — builds a fully-wired AnalysisService.

This module is the ONLY place that decides which concrete implementations
back the service. Everything else depends on abstractions.
"""
from __future__ import annotations

import os
from pathlib import Path

from grimaceguide.core.api_client import LandmarkAPIClient
from grimaceguide.infrastructure.repository import SQLiteResultRepository
from grimaceguide.services.analysis_service import AnalysisService

# Import the existing app config for the API URL so we don't have two sources of truth.
try:
    from grimaceguide.config import API_URL as _DEFAULT_API_URL
except Exception:
    _DEFAULT_API_URL = ""


BASE_DIR = Path(__file__).resolve().parent.parent
# Use a SEPARATE db from grimace_scores.db so the Kivy app is not affected.
DEFAULT_ANALYSES_DB = BASE_DIR / "analyses.db"


def build_service(
    api_url: str | None = None,
    db_path: str | Path | None = None,
) -> AnalysisService:
    """Wire together an AnalysisService with sensible defaults."""
    resolved_url = api_url or os.getenv("GG_API_URL", _DEFAULT_API_URL)
    if not resolved_url:
        raise RuntimeError(
            "No API URL configured. Set GG_API_URL or pass api_url= explicitly."
        )
    resolved_db = Path(db_path or os.getenv("GG_ANALYSES_DB", str(DEFAULT_ANALYSES_DB)))

    return AnalysisService(
        api_client=LandmarkAPIClient(url=resolved_url),
        repository=SQLiteResultRepository(db_path=resolved_db),
    )