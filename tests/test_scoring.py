"""Smoke tests for the pure-Python core (no Kivy required)."""
from grimaceguide.core.landmarks import (
    LANDMARK_LABELS,
    landmarks_from_labeled_dict,
    landmarks_to_labeled_dict,
)
from grimaceguide.core.models import (
    ActionUnitBreakdown,
    ActionUnitScore,
    GrimaceResult,
    Landmark,
    LandmarkSet,
)
from grimaceguide.core.scoring import (
    _score_ears,
    _score_eyes,
    _score_head,
    _score_muzzle,
    _score_whiskers,
    compute_grimace_score,
)


def test_breakdown_total_and_normalized():
    b = ActionUnitBreakdown(
        ears=ActionUnitScore.OBVIOUS,
        eyes=ActionUnitScore.OBVIOUS,
        muzzle=ActionUnitScore.MODERATE,
        whiskers=ActionUnitScore.ABSENT,
        head=ActionUnitScore.ABSENT,
    )
    assert b.total == 5
    assert b.normalized == 0.5


def test_pain_flag_below_threshold():
    b = ActionUnitBreakdown(
        ears=ActionUnitScore.MODERATE,
        eyes=ActionUnitScore.MODERATE,
        muzzle=ActionUnitScore.ABSENT,
        whiskers=ActionUnitScore.ABSENT,
        head=ActionUnitScore.ABSENT,
    )
    result = GrimaceResult.from_breakdown(b)
    assert not result.pain_likely
    assert result.breakdown.total == 2


def test_pain_flag_at_threshold():
    b = ActionUnitBreakdown(
        ears=ActionUnitScore.OBVIOUS,
        eyes=ActionUnitScore.OBVIOUS,
        muzzle=ActionUnitScore.ABSENT,
        whiskers=ActionUnitScore.ABSENT,
        head=ActionUnitScore.ABSENT,
    )
    result = GrimaceResult.from_breakdown(b)
    assert result.pain_likely


def test_landmark_labels_are_48_points():
    assert len(LANDMARK_LABELS) == 48


def test_labeled_dict_round_trip():
    # Build a synthetic 48-point landmark set
    points = tuple(Landmark(x=float(i), y=float(i * 2)) for i in range(48))
    original = LandmarkSet(points=points, image_width=1000, image_height=800)

    labeled = landmarks_to_labeled_dict(original)
    assert set(labeled.keys()) == set(LANDMARK_LABELS)
    assert labeled["chin_point"] == {"x": 47.0, "y": 94.0}

    round_tripped = landmarks_from_labeled_dict(labeled, 1000, 800)
    assert len(round_tripped.points) == 48
    assert round_tripped.points[0].x == 0.0
    assert round_tripped.points[-1].x == 47.0


def _ears_landmarks(re_tip_x):
    return {
        "left_ear_1": Landmark(0, 0),
        "left_ear_3": Landmark(0, -100),
        "left_ear_5": Landmark(0, -50),
        "right_ear_1": Landmark(20, 0),
        "right_ear_3": Landmark(re_tip_x, -100),
        "right_ear_5": Landmark(20, -50),
    }


def test_score_ears_absent():
    assert _score_ears(_ears_landmarks(re_tip_x=20)) == ActionUnitScore.ABSENT  # ratio 1.0


def test_score_ears_moderate():
    assert _score_ears(_ears_landmarks(re_tip_x=36)) == ActionUnitScore.MODERATE  # ratio 1.8


def test_score_ears_obvious():
    assert _score_ears(_ears_landmarks(re_tip_x=40)) == ActionUnitScore.OBVIOUS  # ratio 2.0


def test_score_ears_missing_landmarks_defaults_absent():
    assert _score_ears({}) == ActionUnitScore.ABSENT


def _eyes_landmarks(v_dist):
    # eye_1/eye_2 = outer/inner corner (h_dist=20); eye_3/eye_4 = top/bottom
    # (same x, so distance between them is exactly v_dist) -- a diamond, not
    # two same-y pairs, matching _score_eyes' outer/inner/top/bottom geometry.
    return {
        "left_eye_1": Landmark(0, 0),
        "left_eye_2": Landmark(20, 0),
        "left_eye_3": Landmark(10, -v_dist / 2),
        "left_eye_4": Landmark(10, v_dist / 2),
        "right_eye_1": Landmark(0, 0),
        "right_eye_2": Landmark(20, 0),
        "right_eye_3": Landmark(10, -v_dist / 2),
        "right_eye_4": Landmark(10, v_dist / 2),
    }


def test_score_eyes_absent():
    assert _score_eyes(_eyes_landmarks(v_dist=12)) == ActionUnitScore.ABSENT  # ratio 0.6


def test_score_eyes_moderate():
    assert _score_eyes(_eyes_landmarks(v_dist=6)) == ActionUnitScore.MODERATE  # ratio 0.3


def test_score_eyes_obvious():
    assert _score_eyes(_eyes_landmarks(v_dist=2)) == ActionUnitScore.OBVIOUS  # ratio 0.1


def test_score_eyes_missing_landmarks_defaults_absent():
    assert _score_eyes({}) == ActionUnitScore.ABSENT


def _muzzle_landmarks(width):
    return {
        "nose_3": Landmark(width / 2, -50),
        "mouth_1": Landmark(0, 0),
        "mouth_4": Landmark(width, 0),
    }


def test_score_muzzle_absent():
    assert _score_muzzle(_muzzle_landmarks(width=40)) == ActionUnitScore.ABSENT  # ratio 0.8


def test_score_muzzle_moderate():
    assert _score_muzzle(_muzzle_landmarks(width=60)) == ActionUnitScore.MODERATE  # ratio 1.2


def test_score_muzzle_obvious():
    assert _score_muzzle(_muzzle_landmarks(width=80)) == ActionUnitScore.OBVIOUS  # ratio 1.6


def test_score_muzzle_missing_landmarks_defaults_absent():
    assert _score_muzzle({}) == ActionUnitScore.ABSENT


def _head_landmarks(chin_y):
    return {
        "chin_point": Landmark(10, chin_y),
        "left_ear_1": Landmark(0, 0),
        "right_ear_1": Landmark(20, 0),
    }


def test_score_head_absent():
    # normalized_diff = chin_y / face_width = 5 / 10 = 0.5 (<= 0.86)
    assert _score_head(_head_landmarks(chin_y=5), face_width=10) == ActionUnitScore.ABSENT


def test_score_head_moderate():
    # normalized_diff = 10 / 10 = 1.0 (between 0.86 and 1.32)
    assert _score_head(_head_landmarks(chin_y=10), face_width=10) == ActionUnitScore.MODERATE


def test_score_head_obvious():
    # normalized_diff = 15 / 10 = 1.5 (> 1.32)
    assert _score_head(_head_landmarks(chin_y=15), face_width=10) == ActionUnitScore.OBVIOUS


def test_score_head_missing_landmarks_defaults_absent():
    assert _score_head({}, face_width=None) == ActionUnitScore.ABSENT


def test_score_whiskers_straight_and_forward_is_obvious():
    labeled = {
        "left_whisker_1": Landmark(10, 0),
        "left_whisker_3": Landmark(5, -1),
        "left_whisker_5": Landmark(0, 0),
        "right_whisker_1": Landmark(10, 0),
        "right_whisker_3": Landmark(15, -1),
        "right_whisker_5": Landmark(20, 0),
        "nose_1": Landmark(5, 5),
        "nose_5": Landmark(15, 5),
    }
    assert _score_whiskers(labeled, face_width=100) == ActionUnitScore.OBVIOUS


def test_score_whiskers_mixed_is_moderate():
    labeled = {
        "left_whisker_1": Landmark(10, 0),
        "left_whisker_3": Landmark(5, -1),
        "left_whisker_5": Landmark(0, 0),
        "right_whisker_1": Landmark(10, 0),
        "right_whisker_3": Landmark(7, -1),
        "right_whisker_5": Landmark(5, 0),  # not forward: rw5.x < rw1.x
        "nose_1": Landmark(5, 5),
        "nose_5": Landmark(15, 5),
    }
    assert _score_whiskers(labeled, face_width=100) == ActionUnitScore.MODERATE


def test_score_whiskers_curved_and_wide_spread_is_absent():
    labeled = {
        "left_whisker_1": Landmark(10, 0),
        "left_whisker_3": Landmark(20, 5),  # curved
        "left_whisker_5": Landmark(30, 0),
        "right_whisker_1": Landmark(10, 0),
        "right_whisker_3": Landmark(5, 5),  # curved
        "right_whisker_5": Landmark(0, 0),
        "nose_1": Landmark(5, 5),
        "nose_5": Landmark(15, 5),
    }
    # avg_spread = 15, face_width = 15 -> normalized_spread = 1.0 > 0.8
    assert _score_whiskers(labeled, face_width=15) == ActionUnitScore.ABSENT


def test_score_whiskers_missing_landmarks_defaults_absent():
    assert _score_whiskers({}, face_width=100) == ActionUnitScore.ABSENT


def test_compute_grimace_score_missing_landmarks_falls_back_to_zero():
    landmarks = LandmarkSet(points=(Landmark(0, 0),), image_width=100, image_height=100)
    result = compute_grimace_score(image=None, landmarks=landmarks)
    assert result.breakdown.total == 0
    assert result.pain_likely is False


def test_compute_grimace_score_end_to_end_wiring():
    points = tuple(Landmark(x=float(i), y=float(i * 2)) for i in range(48))
    landmarks = LandmarkSet(points=points, image_width=1000, image_height=800)

    result = compute_grimace_score(image=None, landmarks=landmarks)

    assert isinstance(result, GrimaceResult)
    assert result.landmarks is landmarks
    assert result.processing_ms >= 0
    assert isinstance(result.breakdown, ActionUnitBreakdown)
