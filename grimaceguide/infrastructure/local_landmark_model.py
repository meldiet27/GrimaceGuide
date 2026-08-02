"""Runs a locally-trained ml/ LandmarkNet checkpoint instead of the remote API.

Matches LandmarkDetector's interface (core/api_client.py) so it can be swapped
in via container.py without touching AnalysisService.

Limitation: the checkpoint was trained on face crops (CatFLW's ground-truth
bounding boxes, padded 15% -- see ml/dataset.py), and there is no face-detection
step here (OpenCV 5.x dropped the bundled Haar cascades this would have used).
This treats the whole input image as already being a face-centered crop,
matching this app's documented input assumption (README: "a frontal view of
the cat's face"). If photo framing turns out to be inconsistent in practice,
add a real face-detection step before this.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from grimaceguide.core.exceptions import LandmarkAPIError
from grimaceguide.core.image_processing import to_rgb
from grimaceguide.core.landmarks import landmarks_from_catflw_array
from grimaceguide.core.models import LandmarkSet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LocalLandmarkModel:
    """Runs a locally-trained LandmarkNet checkpoint in-process."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        try:
            import torch
            from ml.model import LandmarkNet
        except ImportError as exc:
            raise LandmarkAPIError(
                "Local landmark model requires torch/torchvision (see "
                "ml/requirements.txt) and the ml/ package to be importable."
            ) from exc

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise LandmarkAPIError(f"Checkpoint not found: {checkpoint_path}")

        self._torch = torch
        self._device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self._image_size = checkpoint.get("image_size", 256)

        self._model = LandmarkNet(pretrained=False).to(self._device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()

    def detect_landmarks(self, image: np.ndarray, name: str = "image.jpg") -> LandmarkSet:
        from ml.model import decode_heatmaps

        torch = self._torch
        height, width = image.shape[:2]

        rgb = to_rgb(image)
        resized = cv2.resize(rgb, (self._image_size, self._image_size))
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = (
            torch.from_numpy(normalized.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.no_grad():
            heatmaps = self._model(tensor)
            points = decode_heatmaps(heatmaps).cpu().numpy()[0]  # (96,) normalized [0, 1]

        pixel_points = points.reshape(-1, 2).astype(np.float64, copy=True)
        pixel_points[:, 0] *= width
        pixel_points[:, 1] *= height

        return landmarks_from_catflw_array(
            pixel_points.reshape(-1).tolist(), image_width=width, image_height=height
        )
