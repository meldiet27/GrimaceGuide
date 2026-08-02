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


ACTION_UNIT_NAMES: tuple[str, ...] = ("ears", "eyes", "muzzle", "whiskers", "head")

# Of the five FGS action units, only these two show usable agreement (Cohen's kappa
# 0.5-0.6) with ground-truth-landmark scoring when run on *model-predicted* landmarks.
# The other three come out at kappa ~= 0 -- not because their thresholds are wrong,
# but because their formulas measure facial features short enough that the landmark
# model's error is 18-26% of the quantity being measured.
#
# Treat the ears/head figures as an optimistic ceiling rather than a validation: the
# checkpoint they were measured on was trained with a cat-level-leaky split, so the
# evaluation images were of cats the model had already seen. That leak makes the
# kappa ~= 0 result for the other three *more* damning, not less -- they failed even
# with that advantage. See docs/heatmap_decoding.md for the full derivation.
#
# The FGS total and pain flag are still computed from all five, because the scale is
# clinically defined that way -- this only records which inputs are trustworthy, so
# callers can avoid presenting three noisy numbers as if they carried equal weight.
VALIDATED_ACTION_UNITS: frozenset[str] = frozenset({"ears", "head"})
LOW_CONFIDENCE_ACTION_UNITS: frozenset[str] = frozenset({"eyes", "muzzle", "whiskers"})

LOW_CONFIDENCE_NOTE = (
    "Eyes, muzzle and whiskers are low-confidence: the landmark model is not precise "
    "enough to measure them reliably, so those three scores -- and the total that "
    "includes them -- should be treated as indicative only, not diagnostic."
)


@dataclass(frozen=True)
class ActionUnitBreakdown:
    ears: ActionUnitScore
    eyes: ActionUnitScore
    muzzle: ActionUnitScore
    whiskers: ActionUnitScore
    head: ActionUnitScore

    @property
    def low_confidence_units(self) -> tuple[str, ...]:
        """AU names whose value isn't trustworthy from predicted landmarks."""
        return tuple(n for n in ACTION_UNIT_NAMES if n in LOW_CONFIDENCE_ACTION_UNITS)

    @property
    def validated_units(self) -> tuple[str, ...]:
        """AU names empirically validated to track ground-truth-landmark scoring."""
        return tuple(n for n in ACTION_UNIT_NAMES if n in VALIDATED_ACTION_UNITS)

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
    processed_path: Optional[str] = None