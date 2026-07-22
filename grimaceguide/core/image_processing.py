"""Image I/O and preprocessing — no UI dependencies."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from grimaceguide.core.exceptions import ImageLoadError

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


def load_image(source: ImageInput) -> np.ndarray:
    """Load an image from a path, raw bytes, or an existing ndarray.

    Always returns a BGR ndarray (as OpenCV expects).
    """
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageLoadError("Could not decode image bytes.")
        return img
    path = Path(source)
    if not path.exists():
        raise ImageLoadError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError(f"Unable to read image at: {path}")
    return img


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ImageLoadError("Failed to encode image as PNG.")
    return buf.tobytes()


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageLoadError("Failed to encode image as JPEG.")
    return buf.tobytes()
