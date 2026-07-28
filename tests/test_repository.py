"""Tests for repository persistence and its wiring into AnalysisService.

Pure-Python (no Kivy); uses a real SQLite file in a tmp_path and a fake
LandmarkAPIClient so no network access is required.
"""
import numpy as np

from grimaceguide.core.landmarks import landmarks_to_labeled_dict
from grimaceguide.core.models import ActionUnitBreakdown, ActionUnitScore, GrimaceResult, Landmark, LandmarkSet
from grimaceguide.infrastructure.repository import SQLiteResultRepository, StoredAnalysis
from grimaceguide.services.analysis_service import AnalysisService


def _make_result() -> GrimaceResult:
    breakdown = ActionUnitBreakdown(
        ears=ActionUnitScore.OBVIOUS,
        eyes=ActionUnitScore.MODERATE,
        muzzle=ActionUnitScore.ABSENT,
        whiskers=ActionUnitScore.ABSENT,
        head=ActionUnitScore.MODERATE,
    )
    return GrimaceResult.from_breakdown(breakdown)


def test_save_and_list_recent_round_trips_metadata(tmp_path):
    repo = SQLiteResultRepository(db_path=tmp_path / "analyses.db")
    result = _make_result()
    raw_landmarks = {"left_ear_1": {"x": 1.0, "y": 2.0}}

    saved_id = repo.save(
        result,
        filename="cat.jpg",
        original_path="/images/cat.jpg",
        raw_landmarks=raw_landmarks,
    )

    [stored] = repo.list_recent()
    assert isinstance(stored, StoredAnalysis)
    assert stored.id == saved_id
    assert stored.filename == "cat.jpg"
    assert stored.original_path == "/images/cat.jpg"
    assert stored.processed_path is None
    assert stored.raw_landmarks == raw_landmarks
    assert stored.result.breakdown == result.breakdown
    assert stored.result.pain_likely == result.pain_likely


def test_list_recent_leaves_raw_landmarks_none_when_not_saved(tmp_path):
    repo = SQLiteResultRepository(db_path=tmp_path / "analyses.db")
    repo.save(_make_result())

    [stored] = repo.list_recent()
    assert stored.raw_landmarks is None
    assert stored.filename is None


class _FakeLandmarkAPIClient:
    def __init__(self, landmarks: LandmarkSet):
        self._landmarks = landmarks

    def detect_landmarks(self, image, name: str = "image.jpg") -> LandmarkSet:
        return self._landmarks


def test_analyze_persists_raw_landmarks(tmp_path):
    points = tuple(Landmark(x=float(i), y=float(i * 2)) for i in range(48))
    landmarks = LandmarkSet(points=points, image_width=1000, image_height=800)

    repo = SQLiteResultRepository(db_path=tmp_path / "analyses.db")
    service = AnalysisService(api_client=_FakeLandmarkAPIClient(landmarks), repository=repo)

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    outcome = service.analyze(image, name="synthetic.jpg")

    [stored] = repo.list_recent()
    assert stored.id == outcome.persisted_id
    assert stored.filename == "synthetic.jpg"
    assert stored.raw_landmarks == landmarks_to_labeled_dict(landmarks)
