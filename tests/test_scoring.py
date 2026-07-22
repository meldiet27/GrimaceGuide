"""Smoke tests for the pure-Python core (no Kivy required)."""
from grimaceguide.core.models import (
    ActionUnitBreakdown,
    ActionUnitScore,
    GrimaceResult,
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
