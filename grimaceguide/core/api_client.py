"""Client for the external landmark-detection API.

Returns a typed LandmarkSet instead of a raw JSON list.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional

import numpy as np
import requests

from grimaceguide.core.exceptions import LandmarkAPIError
from grimaceguide.core.image_processing import encode_jpeg
from grimaceguide.core.landmarks import landmarks_from_points
from grimaceguide.core.models import LandmarkSet


class LandmarkAPIClient:
    """Thin wrapper around the remote landmark inference endpoint."""

    def __init__(
        self,
        url: str,
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
    ):
        self.url = url
        self.timeout = timeout
        self._session = session or requests.Session()

    def detect_landmarks(
        self,
        image: np.ndarray,
        name: str = "image.jpg",
    ) -> LandmarkSet:
        """Send an image to the API and return a LandmarkSet for the first detected animal."""
        b64 = base64.b64encode(encode_jpeg(image)).decode("utf-8")
        payload = json.dumps(
            {
                "name": os.path.basename(name),
                "image": f"data:image/jpeg;base64,{b64}",
            }
        )
        try:
            response = self._session.post(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LandmarkAPIError(f"Remote API call failed: {exc}") from exc

        body = response.json()
        return self._parse_response(body, image.shape[1], image.shape[0])

    @staticmethod
    def _parse_response(body: list, image_width: int, image_height: int) -> LandmarkSet:
        if not body:
            raise LandmarkAPIError("Empty response from landmark API.")
        first_animal = body[0]
        if not first_animal:
            raise LandmarkAPIError("First animal record was empty.")
        animal_key = next(iter(first_animal))
        landmarks = first_animal[animal_key].get("landmarks", [])
        if not landmarks:
            raise LandmarkAPIError(f"No landmarks returned for '{animal_key}'.")
        return landmarks_from_points(
            points=landmarks,
            image_width=image_width,
            image_height=image_height,
        )