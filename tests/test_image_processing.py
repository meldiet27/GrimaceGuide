"""Tests for core.image_processing (pure-Python, no Kivy required)."""
import numpy as np

from grimaceguide.core.image_processing import draw_landmarks_overlay
from grimaceguide.core.models import Landmark, LandmarkSet


def test_draw_landmarks_overlay_marks_pixels_and_preserves_shape_dtype():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    landmarks = LandmarkSet(
        points=(Landmark(x=10, y=10), Landmark(x=30, y=20)),
        image_width=50,
        image_height=50,
    )

    overlay = draw_landmarks_overlay(image, landmarks)

    assert overlay.shape == image.shape
    assert overlay.dtype == image.dtype
    assert not np.array_equal(overlay, image)
    assert tuple(overlay[10, 10]) == (0, 255, 0)  # (row=y, col=x) indexing
    assert tuple(overlay[20, 30]) == (0, 255, 0)
    assert np.array_equal(image, np.zeros((50, 50, 3), dtype=np.uint8))  # input untouched


def test_draw_landmarks_overlay_with_no_points_returns_unchanged_copy():
    image = np.full((20, 20, 3), 127, dtype=np.uint8)
    landmarks = LandmarkSet(points=(), image_width=20, image_height=20)

    overlay = draw_landmarks_overlay(image, landmarks)

    assert np.array_equal(overlay, image)
    assert overlay is not image
