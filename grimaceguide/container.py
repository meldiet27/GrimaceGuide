"""Composition root — builds a fully-wired AnalysisService.

This module is the ONLY place that decides which concrete implementations
back the service. Everything else depends on abstractions.
"""
from __future__ import annotations

import os
from pathlib import Path

from grimaceguide.core.api_client import LandmarkAPIClient
from grimaceguide.infrastructure.repository import SQLiteResultRepository
from grimaceguide.infrastructure.storage import LocalBlobStorage
from grimaceguide.services.analysis_service import AnalysisService

# Import the existing app config for the API URL so we don't have two sources of truth.
try:
    from grimaceguide.config import API_URL as _DEFAULT_API_URL
except Exception:
    _DEFAULT_API_URL = ""


BASE_DIR = Path(__file__).resolve().parent.parent
# Use a SEPARATE db from grimace_scores.db so the Kivy app is not affected.
DEFAULT_ANALYSES_DB = BASE_DIR / "analyses.db"
DEFAULT_PROCESSED_DIR = BASE_DIR / "processed"

# The checkpoints trained under the subject-disjoint split (ml/dataset.py::
# subject_disjoint_split). Deliberately NOT the plain landmark_net.pt/bbox_net.pt,
# which are the older checkpoints trained with cat-level leakage across train/val --
# equivalent in accuracy, but their held-out metrics were measured on cats they had
# already seen, so nothing about them can be trusted. These carry their own split
# config, which ml/compare_predicted_geometry.py reads back when evaluating.
# See docs/heatmap_decoding.md.
DEFAULT_LOCAL_CHECKPOINT = BASE_DIR / "ml" / "checkpoints" / "landmark_net_subjsplit.pt"
DEFAULT_LOCAL_BBOX_CHECKPOINT = BASE_DIR / "ml" / "checkpoints" / "bbox_net_subjsplit.pt"


def build_service(
    api_url: str | None = None,
    db_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    landmark_source: str | None = None,
    local_checkpoint_path: str | Path | None = None,
    local_bbox_checkpoint_path: str | Path | None = None,
) -> AnalysisService:
    """Wire together an AnalysisService with sensible defaults.

    landmark_source selects between the remote API (default) and the local
    ml/-trained model ("local", requires torch -- see
    grimaceguide/infrastructure/local_landmark_model.py). Override via
    GG_LANDMARK_SOURCE / GG_LOCAL_CHECKPOINT / GG_LOCAL_BBOX_CHECKPOINT env
    vars; the local checkpoints default to the subject-disjoint-split ones
    (DEFAULT_LOCAL_CHECKPOINT), so "local" now needs no extra configuration.
    The bbox checkpoint is optional -- without it, LocalLandmarkModel falls back
    to treating the whole input image as an already-cropped face.
    """
    resolved_source = (landmark_source or os.getenv("GG_LANDMARK_SOURCE", "remote")).lower()

    if resolved_source == "local":
        from grimaceguide.infrastructure.local_landmark_model import LocalLandmarkModel

        resolved_checkpoint = Path(
            local_checkpoint_path
            or os.getenv("GG_LOCAL_CHECKPOINT", str(DEFAULT_LOCAL_CHECKPOINT))
        )
        if not resolved_checkpoint.exists():
            raise RuntimeError(
                f"Local landmark checkpoint not found: {resolved_checkpoint}. Train one "
                "(see ml/README.md) or point GG_LOCAL_CHECKPOINT at an existing checkpoint."
            )
        resolved_bbox_checkpoint = Path(
            local_bbox_checkpoint_path
            or os.getenv("GG_LOCAL_BBOX_CHECKPOINT", str(DEFAULT_LOCAL_BBOX_CHECKPOINT))
        )
        # The bbox model stays optional: without it LocalLandmarkModel treats the whole
        # image as an already-cropped face, which still works for pre-cropped inputs.
        if not resolved_bbox_checkpoint.exists():
            resolved_bbox_checkpoint = None
        api_client = LocalLandmarkModel(
            checkpoint_path=resolved_checkpoint,
            bbox_checkpoint_path=resolved_bbox_checkpoint,
        )
    elif resolved_source == "remote":
        resolved_url = api_url or os.getenv("GG_API_URL", _DEFAULT_API_URL)
        if not resolved_url:
            raise RuntimeError(
                "No API URL configured. Set GG_API_URL or pass api_url= explicitly."
            )
        api_client = LandmarkAPIClient(url=resolved_url)
    else:
        raise RuntimeError(
            f"Unknown GG_LANDMARK_SOURCE: {resolved_source!r} (expected 'remote' or 'local')."
        )

    resolved_db = Path(db_path or os.getenv("GG_ANALYSES_DB", str(DEFAULT_ANALYSES_DB)))
    resolved_processed_dir = Path(
        processed_dir or os.getenv("GG_PROCESSED_DIR", str(DEFAULT_PROCESSED_DIR))
    )

    return AnalysisService(
        api_client=api_client,
        repository=SQLiteResultRepository(db_path=resolved_db),
        image_storage=LocalBlobStorage(root=resolved_processed_dir),
    )