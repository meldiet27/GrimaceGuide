"""Pure-Python core of GrimaceGuide.

This package MUST NOT import from any UI framework (kivy, tkinter, matplotlib.pyplot)
or from `grimaceguide.ui`. It can be safely used from FastAPI, CLI, or tests.
"""
from grimaceguide.core.models import (
    ActionUnitBreakdown,
    ActionUnitScore,
    GrimaceResult,
    Landmark,
    LandmarkSet,
)

__all__ = [
    "ActionUnitBreakdown",
    "ActionUnitScore",
    "GrimaceResult",
    "Landmark",
    "LandmarkSet",
]