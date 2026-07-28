"""Main use-case: analyze a cat image and produce a GrimaceResult."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from grimaceguide.core.api_client import LandmarkAPIClient
from grimaceguide.core.image_processing import ImageInput, draw_landmarks_overlay, encode_png, load_image
from grimaceguide.core.landmarks import landmarks_to_labeled_dict
from grimaceguide.core.models import AnalysisOutcome
from grimaceguide.core.scoring import compute_grimace_score
from grimaceguide.infrastructure.repository import SQLiteResultRepository
from grimaceguide.infrastructure.storage import BlobStorage


class AnalysisService:
    """Coordinates image loading, landmark detection, scoring and persistence."""

    def __init__(
        self,
        api_client: LandmarkAPIClient,
        repository: SQLiteResultRepository,
        image_storage: Optional[BlobStorage] = None,
    ):
        self._api = api_client
        self._repo = repository
        self._image_storage = image_storage

    def analyze(
        self,
        image_input: ImageInput,
        name: Optional[str] = None,
        persist: bool = True,
    ) -> AnalysisOutcome:
        image = load_image(image_input)
        display_name = name or (
            os.path.basename(image_input) if isinstance(image_input, str) else "image.jpg"
        )
        landmarks = self._api.detect_landmarks(image, name=display_name)
        result = compute_grimace_score(image, landmarks)

        # Only render/persist an overlay when we're also persisting the row that
        # references it — a rendered file with no DB row would just be an orphan.
        processed_path = None
        if persist and self._image_storage is not None:
            overlay = draw_landmarks_overlay(image, landmarks)
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
            key = f"{Path(display_name).stem}_{timestamp}.png"
            processed_path = self._image_storage.put(key, encode_png(overlay))

        persisted_id = None
        if persist:
            persisted_id = self._repo.save(
                result,
                filename=display_name,
                original_path=image_input if isinstance(image_input, str) else None,
                processed_path=processed_path,
                raw_landmarks=landmarks_to_labeled_dict(landmarks),
            )

        return AnalysisOutcome(
            result=result,
            raw_api_response=None,  # populated in a later phase if needed
            persisted_id=persisted_id,
            processed_path=processed_path,
        )
