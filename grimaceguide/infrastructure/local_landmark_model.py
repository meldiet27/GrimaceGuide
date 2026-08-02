"""Runs locally-trained ml/ models (BBoxNet + LandmarkNet) instead of the remote API.

Matches LandmarkDetector's interface (core/api_client.py) so it can be swapped
in via container.py without touching AnalysisService.

If a bbox checkpoint is supplied, BBoxNet (ml/bbox_model.py) locates the cat's
face in the raw input image first, then the crop is padded 15% (matching
ml/dataset.py's training convention) before running LandmarkNet on it -- this
is what turns an arbitrary photo into the face-centered input LandmarkNet
expects. Without a bbox checkpoint, this falls back to treating the whole
input image as already being that crop, which is a much rougher
approximation (see grimaceguide/container.py's GG_LOCAL_BBOX_CHECKPOINT).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from grimaceguide.core.exceptions import LandmarkAPIError
from grimaceguide.core.image_processing import to_rgb
from grimaceguide.core.landmarks import landmarks_from_catflw_array
from grimaceguide.core.models import LandmarkSet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BOX_PADDING = 0.15  # matches ml/dataset.py's box_padding used at training time


class LocalLandmarkModel:
    """Runs locally-trained BBoxNet (optional) + LandmarkNet checkpoints in-process."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        bbox_checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
    ):
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

        self._bbox_model = None
        self._bbox_image_size = None
        if bbox_checkpoint_path is not None:
            from ml.bbox_model import BBoxNet

            bbox_checkpoint_path = Path(bbox_checkpoint_path)
            if not bbox_checkpoint_path.exists():
                raise LandmarkAPIError(f"BBox checkpoint not found: {bbox_checkpoint_path}")
            bbox_checkpoint = torch.load(bbox_checkpoint_path, map_location=self._device)
            self._bbox_image_size = bbox_checkpoint.get("image_size", 224)
            self._bbox_model = BBoxNet(pretrained=False).to(self._device)
            self._bbox_model.load_state_dict(bbox_checkpoint["model_state"])
            self._bbox_model.eval()

    def _to_tensor(self, rgb: np.ndarray, size: int):
        torch = self._torch
        resized = cv2.resize(rgb, (size, size))
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

    def _detect_face_box(self, rgb: np.ndarray) -> tuple[float, float, float, float]:
        """Returns a padded (x_min, y_min, x_max, y_max) crop box in pixel coords."""
        height, width = rgb.shape[:2]
        if self._bbox_model is None:
            return 0.0, 0.0, float(width), float(height)

        torch = self._torch
        tensor = self._to_tensor(rgb, self._bbox_image_size)
        with torch.no_grad():
            box = self._bbox_model(tensor).cpu().numpy()[0]  # normalized [x_min,y_min,x_max,y_max]

        x_min, y_min = box[0] * width, box[1] * height
        x_max, y_max = box[2] * width, box[3] * height
        pad_x, pad_y = (x_max - x_min) * BOX_PADDING, (y_max - y_min) * BOX_PADDING
        x_min = max(0.0, x_min - pad_x)
        y_min = max(0.0, y_min - pad_y)
        x_max = min(float(width), x_max + pad_x)
        y_max = min(float(height), y_max + pad_y)
        return x_min, y_min, x_max, y_max

    def detect_landmarks(self, image: np.ndarray, name: str = "image.jpg") -> LandmarkSet:
        from ml.model import decode_heatmaps

        torch = self._torch
        rgb = to_rgb(image)
        image_height, image_width = rgb.shape[:2]

        x_min, y_min, x_max, y_max = self._detect_face_box(rgb)
        crop = rgb[int(y_min):int(y_max), int(x_min):int(x_max)]
        crop_h, crop_w = crop.shape[:2]

        tensor = self._to_tensor(crop, self._image_size)
        with torch.no_grad():
            heatmaps = self._model(tensor)
            points = decode_heatmaps(heatmaps).cpu().numpy()[0]  # (96,) normalized [0, 1] in crop

        pixel_points = points.reshape(-1, 2).astype(np.float64, copy=True)
        pixel_points[:, 0] = x_min + pixel_points[:, 0] * crop_w
        pixel_points[:, 1] = y_min + pixel_points[:, 1] * crop_h

        return landmarks_from_catflw_array(
            pixel_points.reshape(-1).tolist(), image_width=image_width, image_height=image_height
        )
