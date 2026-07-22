"""Client for the external landmark-detection API."""
from __future__ import annotations

import base64
from typing import Optional

import numpy as np
import requests

from grimaceguide.core.exceptions import LandmarkAPIError
from grimaceguide.core.image_processing import encode_png
from grimaceguide.core.landmarks import landmarks_from_points
from grimaceguide.core.models import LandmarkSet


class LandmarkAPIClient:
    """Thin wrapper around the remote landmark inference endpoint.

    All configuration is injected — no hidden globals, no environment reads.
    """

    def __init__(
            self,
            base_url: str,
            api_key: Optional[str] = None,
            timeout: float = 30.0,
            session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = session or requests.Session()

    def detect_landmarks(self, image: np.ndarray) -> LandmarkSet:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "image": base64.b64encode(encode_png(image)).decode("ascii"),
        }
        try:
            response = self._session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LandmarkAPIError(f"Remote API call failed: {exc}") from exc

        body = response.json()
        return landmarks_from_points(
            points=body.get("points", []),
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
