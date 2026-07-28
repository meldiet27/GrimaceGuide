"""Main use-case: analyze a cat image and produce a GrimaceResult."""
from __future__ import annotations

import os
from typing import Optional

from grimaceguide.core.api_client import LandmarkAPIClient
from grimaceguide.core.image_processing import ImageInput, load_image
from grimaceguide.core.models import AnalysisOutcome
from grimaceguide.core.scoring import compute_grimace_score
from grimaceguide.infrastructure.repository import SQLiteResultRepository


class AnalysisService:
    """Coordinates image loading, landmark detection, scoring and persistence."""

    def __init__(
        self,
        api_client: LandmarkAPIClient,
        repository: SQLiteResultRepository,
    ):
        self._api = api_client
        self._repo = repository

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

        persisted_id = None
        if persist:
            persisted_id = self._repo.save(
                result,
                filename=display_name,
                original_path=image_input if isinstance(image_input, str) else None,
            )

        return AnalysisOutcome(
            result=result,
            raw_api_response=None,  # populated in a later phase if needed
            persisted_id=persisted_id,
        )
