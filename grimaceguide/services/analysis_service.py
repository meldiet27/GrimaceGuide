"""Main use-case: analyze a cat image and produce a GrimaceResult."""
from __future__ import annotations

from grimaceguide.core.api_client import LandmarkAPIClient
from grimaceguide.core.image_processing import ImageInput, load_image
from grimaceguide.core.models import GrimaceResult
from grimaceguide.core.scoring import compute_grimace_score
from grimaceguide.infrastructure.repository import ResultRepository


class AnalysisService:
    """Coordinates image loading, landmark detection, scoring and persistence."""

    def __init__(
            self,
            api_client: LandmarkAPIClient,
            repository: ResultRepository,
    ):
        self._api = api_client
        self._repo = repository

    def analyze(self, image_input: ImageInput, persist: bool = True) -> GrimaceResult:
        image = load_image(image_input)
        landmarks = self._api.detect_landmarks(image)
        result = compute_grimace_score(image, landmarks)
        if persist:
            self._repo.save(result)
        return result
