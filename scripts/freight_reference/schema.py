"""Pydantic schema for the structured shipment extraction target.

The schema defines the five fields the model must extract from freeform freight
text, plus canonicalization helpers shared by dataset generation and evaluation so
that scoring compares values consistently (e.g. trimmed/title-cased city names).
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, field_validator

VALID_MODES = ["LTL", "FTL", "Parcel", "Intermodal"]


class Mode(StrEnum):
    LTL = "LTL"
    FTL = "FTL"
    Parcel = "Parcel"
    Intermodal = "Intermodal"


class ShipmentExtraction(BaseModel):
    """Structured fields extracted from a freeform shipment sentence."""

    origin: str
    destination: str
    weight_lbs: int
    mode: Mode
    on_time: bool

    @field_validator("origin", "destination")
    @classmethod
    def _canon_city(cls, v: str) -> str:
        return canonicalize_city(v)

    def to_json(self) -> str:
        """Compact, key-ordered JSON string (the model's target output)."""
        return json.dumps(
            {
                "origin": self.origin,
                "destination": self.destination,
                "weight_lbs": self.weight_lbs,
                "mode": self.mode.value,
                "on_time": self.on_time,
            }
        )


def canonicalize_city(value: str) -> str:
    """Normalize a city name for stable comparison: trim and title-case."""
    return " ".join(str(value).strip().split()).title()


def canonicalize_mode(value: str) -> str | None:
    """Map a free-text mode to a canonical mode, or None if unrecognized."""
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    for mode in VALID_MODES:
        if cleaned == mode.lower():
            return mode
    return None
