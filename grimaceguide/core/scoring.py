"""Feline Grimace Scale scoring.

Per-AU geometry ported from the original grimaceguide.fgsScoreCalc prototype,
operating on typed Landmark/LandmarkSet objects instead of raw {'x':.., 'y':..}
dicts. grimaceguide.fgsScoreCalc itself remains in place for the legacy
Kivy path (grimaceguide.api -> grimaceguide.ui.app) — this module no longer
depends on it.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from grimaceguide.core.exceptions import ScoringError
from grimaceguide.core.landmarks import LANDMARK_LABELS
from grimaceguide.core.models import (
    ActionUnitBreakdown,
    ActionUnitScore,
    GrimaceResult,
    Landmark,
    LandmarkSet,
)

PAIN_THRESHOLD = 0.39


def _label_landmarks(landmarks: LandmarkSet) -> dict[str, Landmark]:
    """Pair LandmarkSet points with their FGS labels for name-based lookup."""
    return dict(zip(LANDMARK_LABELS, landmarks.points))


def _distance(p1: Optional[Landmark], p2: Optional[Landmark]) -> Optional[float]:
    if p1 is None or p2 is None:
        return None
    return float(np.hypot(p1.x - p2.x, p1.y - p2.y))


def _vertical_angle(p1: Optional[Landmark], p2: Optional[Landmark]) -> Optional[float]:
    """Angle of the line p1-p2 relative to the vertical axis, in degrees."""
    if p1 is None or p2 is None:
        return None
    delta_x = p2.x - p1.x
    delta_y = p2.y - p1.y
    if delta_y == 0:
        return 90.0
    angle_rad = np.arctan(delta_x / delta_y)
    return float(abs(np.degrees(angle_rad)))


def _estimate_face_width(labeled: dict[str, Landmark]) -> Optional[float]:
    return _distance(labeled.get("left_eye_1"), labeled.get("right_eye_1"))


def _score_ears(labeled: dict[str, Landmark]) -> ActionUnitScore:
    """0 = forward, 1 = slightly apart, 2 = flattened/rotated outwards."""
    le_base, le_tip, le_outer = labeled.get("left_ear_1"), labeled.get("left_ear_3"), labeled.get("left_ear_5")
    re_base, re_tip, re_outer = labeled.get("right_ear_1"), labeled.get("right_ear_3"), labeled.get("right_ear_5")

    # le_outer/re_outer are presence-gate only, not used in the math below (parity with legacy).
    if not (le_base and le_tip and le_outer and re_base and re_tip and re_outer):
        return ActionUnitScore.ABSENT

    left_ear_angle = _vertical_angle(le_base, le_tip)
    right_ear_angle = _vertical_angle(re_base, re_tip)
    avg_ear_angle = (
        (left_ear_angle + right_ear_angle) / 2
        if left_ear_angle is not None and right_ear_angle is not None else None
    )
    tip_dist = _distance(le_tip, re_tip)
    base_dist = _distance(le_base, re_base)
    tip_base_ratio = tip_dist / base_dist if base_dist and base_dist > 0 else None

    if avg_ear_angle is None or tip_base_ratio is None:
        return ActionUnitScore.ABSENT
    if avg_ear_angle > 45 or tip_base_ratio > 1.5:
        return ActionUnitScore.OBVIOUS
    if avg_ear_angle > 25 or tip_base_ratio > 1.2:
        return ActionUnitScore.MODERATE
    return ActionUnitScore.ABSENT


def _score_eyes(labeled: dict[str, Landmark]) -> ActionUnitScore:
    """0 = open, 1 = partially open, 2 = squinted."""
    le1, le2, le3, le4 = (labeled.get(f"left_eye_{i + 1}") for i in range(4))
    re1, re2, re3, re4 = (labeled.get(f"right_eye_{i + 1}") for i in range(4))

    if not all([le1, le2, le3, le4, re1, re2, re3, re4]):
        return ActionUnitScore.ABSENT

    left_top_mid_y = (le1.y + le2.y) / 2
    left_bot_mid_y = (le3.y + le4.y) / 2
    left_v_dist = abs(left_bot_mid_y - left_top_mid_y)
    left_h_dist = _distance(le1, le2)

    right_top_mid_y = (re1.y + re2.y) / 2
    right_bot_mid_y = (re3.y + re4.y) / 2
    right_v_dist = abs(right_bot_mid_y - right_top_mid_y)
    right_h_dist = _distance(re1, re2)

    avg_v_dist = (left_v_dist + right_v_dist) / 2
    avg_h_dist = (left_h_dist + right_h_dist) / 2 if left_h_dist and right_h_dist else None

    if not avg_h_dist or avg_h_dist <= 0:
        return ActionUnitScore.ABSENT

    eye_aspect_ratio = avg_v_dist / avg_h_dist
    if eye_aspect_ratio < 0.25:
        return ActionUnitScore.OBVIOUS
    if eye_aspect_ratio < 0.5:
        return ActionUnitScore.MODERATE
    return ActionUnitScore.ABSENT


def _score_muzzle(labeled: dict[str, Landmark]) -> ActionUnitScore:
    """0 = relaxed round shape, 1 = mildly tense, 2 = tense elliptical shape."""
    nose_tip = labeled.get("nose_3")
    mouth_left = labeled.get("mouth_1")
    mouth_right = labeled.get("mouth_4")

    if not (nose_tip and mouth_left and mouth_right):
        return ActionUnitScore.ABSENT

    mouth_width = _distance(mouth_left, mouth_right)
    mouth_center = Landmark(
        x=(mouth_left.x + mouth_right.x) / 2,
        y=(mouth_left.y + mouth_right.y) / 2,
    )
    nose_to_mouth_dist = _distance(nose_tip, mouth_center)

    if not (mouth_width and nose_to_mouth_dist and nose_to_mouth_dist > 0):
        return ActionUnitScore.ABSENT

    muzzle_ratio = mouth_width / nose_to_mouth_dist
    if muzzle_ratio > 1.5:
        return ActionUnitScore.OBVIOUS
    if muzzle_ratio > 1.1:
        return ActionUnitScore.MODERATE
    return ActionUnitScore.ABSENT


def _score_whiskers(labeled: dict[str, Landmark], face_width: Optional[float]) -> ActionUnitScore:
    """0 = loose/curved, 1 = slightly curved/straight, 2 = straight/forward-pointing."""
    lw1, lw3, lw5 = labeled.get("left_whisker_1"), labeled.get("left_whisker_3"), labeled.get("left_whisker_5")
    rw1, rw3, rw5 = labeled.get("right_whisker_1"), labeled.get("right_whisker_3"), labeled.get("right_whisker_5")
    # nose_base_l/nose_base_r are presence-gate only, not used in the math below (parity with legacy).
    nose_base_l, nose_base_r = labeled.get("nose_1"), labeled.get("nose_5")

    if not all([lw1, lw3, lw5, rw1, rw3, rw5, nose_base_l, nose_base_r]):
        return ActionUnitScore.ABSENT

    left_spread = abs(lw5.x - lw1.x)
    right_spread = abs(rw5.x - rw1.x)
    avg_spread = (left_spread + right_spread) / 2

    left_forward = lw5.x < lw1.x
    right_forward = rw5.x > rw1.x

    left_curve_proxy = lw3.y > (lw1.y + lw5.y) / 2
    right_curve_proxy = rw3.y > (rw1.y + rw5.y) / 2

    normalized_spread = avg_spread / face_width if face_width else None

    is_straight = not (left_curve_proxy or right_curve_proxy)
    is_forward = left_forward and right_forward

    if is_straight and is_forward:
        return ActionUnitScore.OBVIOUS
    if not is_straight and not is_forward:
        if normalized_spread and normalized_spread > 0.8:
            return ActionUnitScore.ABSENT
        return ActionUnitScore.MODERATE
    return ActionUnitScore.MODERATE


def _score_head(labeled: dict[str, Landmark]) -> ActionUnitScore:
    """0 = above shoulder line, 1 = aligned, 2 = below shoulder line/tilted down."""
    chin_point = labeled.get("chin_point")
    left_ear_base = labeled.get("left_ear_1")
    right_ear_base = labeled.get("right_ear_1")

    if not (chin_point and left_ear_base and right_ear_base):
        return ActionUnitScore.ABSENT

    avg_ear_base_y = (left_ear_base.y + right_ear_base.y) / 2
    chin_y = chin_point.y

    head_height_proxy = abs(chin_y - avg_ear_base_y)
    diff = chin_y - avg_ear_base_y
    normalized_diff = diff / head_height_proxy if head_height_proxy and head_height_proxy > 0 else 0

    if normalized_diff > 0.1:
        return ActionUnitScore.OBVIOUS
    if normalized_diff > -0.1:
        return ActionUnitScore.MODERATE
    return ActionUnitScore.ABSENT


def compute_grimace_score(
    image: Optional[np.ndarray],
    landmarks: LandmarkSet,
) -> GrimaceResult:
    """Compute the full FGS score from an image + landmarks."""
    if not landmarks.points:
        raise ScoringError("No landmarks supplied to scoring.")

    labeled = _label_landmarks(landmarks)

    start = time.perf_counter()
    face_width = _estimate_face_width(labeled)
    breakdown = ActionUnitBreakdown(
        ears=_score_ears(labeled),
        eyes=_score_eyes(labeled),
        muzzle=_score_muzzle(labeled),
        whiskers=_score_whiskers(labeled, face_width),
        head=_score_head(labeled),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return GrimaceResult.from_breakdown(
        breakdown=breakdown,
        landmarks=landmarks,
        processing_ms=elapsed_ms,
    )
