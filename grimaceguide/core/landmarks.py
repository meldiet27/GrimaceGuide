"""Helpers for constructing / manipulating landmark sets."""
from __future__ import annotations

from typing import Iterable, Mapping

from grimaceguide.core.models import Landmark, LandmarkSet


# The 48-point label order used by the GrimaceGuide landmark API integration.
# eye_1..4 is a 4-point diamond: outer corner, inner corner, top, bottom (see
# scoring.py::_score_eyes) -- not two top points + two bottom points.
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

NUM_CATFLW_LANDMARKS = 48

# Maps each LANDMARK_LABELS slot (by position) to the CatFLW dataset's own raw
# landmark index. CatFLW's point order is unrelated to LANDMARK_LABELS (it's a
# separate, undocumented-by-index annotation scheme -- see arXiv:2305.04232,
# which describes the 48 points anatomically but never numbers them) and was
# reverse-engineered empirically: for each cluster of raw indices, plotting many
# samples' coordinates and checking which specific index consistently played
# which anatomical role across varied poses (see ml/diagnose.py's per-landmark
# checks and the ear base/tip robustness check done during that investigation).
# Confidence is high for ears (base/tip verified across poses), eyes (outer/
# inner/top/bottom verified across poses), and chin (verified via a stable
# local vertical ordering). Confidence is lower for the specific sub-ordering
# within nose_1/2/4/5, mouth_2/3/5/6, and whisker points 2/4 -- but none of
# those are actually read by any function in scoring.py (only nose_3, mouth_1,
# mouth_4, chin_point, ear_1/_3/_5, and whisker_1/_3/_5 are), so their exact
# identity doesn't affect FGS scoring today.
CATFLW_INDEX_FOR_LABEL: tuple[int, ...] = (
    # left_ear_1..5 (1=base, 3=tip -- verified robust across poses)
    22, 23, 24, 25, 26,
    # right_ear_1..5 (1=base, 3=tip -- verified robust across poses)
    27, 28, 29, 30, 31,
    # right_eye_1..4: outer, inner, top, bottom (verified robust across poses)
    8, 9, 10, 11,
    # right_pupil_1..4 (unused by scoring.py; order arbitrary)
    1, 39, 40, 41,
    # left_eye_1..4: outer, inner, top, bottom (verified robust across poses)
    4, 5, 6, 7,
    # left_pupil_1..4 (unused by scoring.py; order arbitrary)
    3, 36, 37, 38,
    # nose_1..5 (nose_3 = tip, the only one scoring.py reads; verified as the
    # most centered of the nose-tip-level cluster across poses)
    12, 13, 44, 14, 15,
    # mouth_1..6 (mouth_1 = left corner, mouth_4 = right corner -- the only two
    # scoring.py reads; both verified as the lateral pair at mouth-stack height)
    20, 16, 17, 18, 0, 45,
    # left_whisker_1..5 (1/3/5 verified: 5 is robustly the lowest point across
    # poses; 1 and 3 are the topmost/middle of the fan, with 1 vs 2 occasionally
    # swapping rank by a hair between poses -- doesn't change which is read)
    32, 42, 33, 46, 21,
    # right_whisker_1..5 (mirrors left_whisker; same confidence notes)
    35, 43, 34, 47, 19,
    # chin_point (verified: robustly the lowest point of its local vertical
    # stack across poses)
    2,
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


def landmarks_from_catflw_array(
    flat: list[float],
    image_width: int,
    image_height: int,
) -> LandmarkSet:
    """Build a LandmarkSet from a flat CatFLW-order [x0, y0, x1, y1, ...] array.

    Reorders via CATFLW_INDEX_FOR_LABEL so the result lines up with
    LANDMARK_LABELS the same way the remote API's response does.
    """
    if len(flat) != NUM_CATFLW_LANDMARKS * 2:
        raise ValueError(
            f"Expected {NUM_CATFLW_LANDMARKS * 2} values, got {len(flat)}."
        )
    catflw_points = [(flat[2 * i], flat[2 * i + 1]) for i in range(NUM_CATFLW_LANDMARKS)]
    ordered = tuple(
        Landmark(x=float(catflw_points[idx][0]), y=float(catflw_points[idx][1]))
        for idx in CATFLW_INDEX_FOR_LABEL
    )
    return LandmarkSet(points=ordered, image_width=image_width, image_height=image_height)


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