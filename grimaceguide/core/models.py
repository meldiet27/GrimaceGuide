"""Immutable domain models — plain data, no side effects."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Optional


class ActionUnitScore(IntEnum):
    ABSENT = 0
    MODERATE = 1
    OBVIOUS = 2


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class LandmarkSet:
    """Facial landmarks detected on a single cat image."""
    points: tuple[Landmark, ...]
    image_width: int
    image_height: int


@dataclass(frozen=True)
class ActionUnitBreakdown:
    ears: ActionUnitScore
    eyes: ActionUnitScore
    muzzle: ActionUnitScore
    whiskers: ActionUnitScore
    head: ActionUnitScore

    @property
    def total(self) -> int:
        return (
            int(self.ears)
            + int(self.eyes)
            + int(self.muzzle)
            + int(self.whiskers)
            + int(self.head)
        )

    @property
    def normalized(self) -> float:
        return self.total / 10.0

    def as_dict(self) -> dict[str, int]:
        return {
            "ears": int(self.ears),
            "eyes": int(self.eyes),
            "muzzle": int(self.muzzle),
            "whiskers": int(self.whiskers),
            "head": int(self.head),
        }


@dataclass(frozen=True)
class GrimaceResult:
    breakdown: ActionUnitBreakdown
    pain_likely: bool
    landmarks: Optional[LandmarkSet] = None
    processing_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def from_breakdown(
        breakdown: ActionUnitBreakdown,
        landmarks: Optional[LandmarkSet] = None,
        processing_ms: float = 0.0,
    ) -> "GrimaceResult":
        return GrimaceResult(
            breakdown=breakdown,
            pain_likely=breakdown.normalized >= 0.39,
            landmarks=landmarks,
            processing_ms=processing_ms,
        )


@dataclass(frozen=True)
class AnalysisOutcome:
    """The full result of running the analysis pipeline on one image."""
    result: GrimaceResult
    raw_api_response: Any = None
    persisted_id: Optional[int] = None