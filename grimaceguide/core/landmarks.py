"""Helpers for constructing / manipulating landmark sets."""
from __future__ import annotations

from typing import Iterable, Mapping

from grimaceguide.core.models import Landmark, LandmarkSet


# The 48-point label order used by the GrimaceGuide landmark API integration.
LANDMARK_LABELS: tuple[str, ...] = tuple(
    [f"left_ear_{i + 1}" for i in range(5)]
    + [f"right_ear_{i + 1}" for i in range(5)]
    + [f"right_eye_{i + 1}" for i in range(4)]
    + [f"right_pupil_{i + 1}" for i in range(4)]
    + [f"left_eye_{i + 1}" for i in range(4)]
    + [f"left_pupil_{i + 1}" for i in range(4)]
    + [f"nose_{i + 1}" for i in range(5)]
    + [f"mouth_{i + 1}" for i in range(6)]
    + [f"left_whisker_{i + 1}" for i in range(5)]
    + [f"right_whisker_{i + 1}" for i in range(5)]
    + ["chin_point"]
)


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


def landmarks_to_labeled_dict(landmarks: LandmarkSet) -> dict[str, dict[str, float]]:
    """Convert a LandmarkSet into a labeled dict shape.

    Produces {'left_ear_1': {'x': ..., 'y': ...}, ..., 'chin_point': {'x': ..., 'y': ...}}.
    """
    labeled: dict[str, dict[str, float]] = {}
    for i, point in enumerate(landmarks.points):
        if i < len(LANDMARK_LABELS):
            labeled[LANDMARK_LABELS[i]] = {"x": point.x, "y": point.y}
        else:
            labeled[f"extra_point_{i}"] = {"x": point.x, "y": point.y}
    return labeled


def landmarks_from_labeled_dict(
    labeled: Mapping[str, Mapping[str, float]],
    image_width: int,
    image_height: int,
) -> LandmarkSet:
    """Reverse of landmarks_to_labeled_dict — for turning a legacy dict into a LandmarkSet."""
    points: list[Landmark] = []
    for label in LANDMARK_LABELS:
        pt = labeled.get(label)
        if pt is None:
            continue
        points.append(Landmark(x=float(pt["x"]), y=float(pt["y"])))
    return LandmarkSet(points=tuple(points), image_width=image_width, image_height=image_height)