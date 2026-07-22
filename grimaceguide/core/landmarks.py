"""Helpers for constructing / manipulating landmark sets."""
from __future__ import annotations

from typing import Iterable, Mapping

from grimaceguide.core.models import Landmark, LandmarkSet


def landmarks_from_points(
        points: Iterable[Mapping[str, float]],
        image_width: int,
        image_height: int,
) -> LandmarkSet:
    """Build a LandmarkSet from an iterable of dicts like {'x': ..., 'y': ...}."""
    parsed = tuple(
        Landmark(
            x=float(p["x"]),
            y=float(p["y"]),
            confidence=float(p.get("confidence", 1.0)),
        )
        for p in points
    )
    return LandmarkSet(points=parsed, image_width=image_width, image_height=image_height)


def landmarks_from_flat_array(
        flat: list[float],
        image_width: int,
        image_height: int,
) -> LandmarkSet:
    """Build a LandmarkSet from a flat [x0, y0, x1, y1, ...] list."""
    if len(flat) % 2 != 0:
        raise ValueError("Flat landmark array must have an even number of values.")
    points = tuple(
        Landmark(x=float(flat[i]), y=float(flat[i + 1]))
        for i in range(0, len(flat), 2)
    )
    return LandmarkSet(points=points, image_width=image_width, image_height=image_height)
