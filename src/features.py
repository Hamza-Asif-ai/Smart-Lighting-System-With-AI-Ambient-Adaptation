"""Shared feature encoding, target encoding and input validation.

Training, inference, the dashboard and the tests all go through this module so
that the sensor -> model contract is defined in exactly one place.
"""

import pandas as pd

from src.config import (
    AMBIENT_LEVELS,
    AMBIENT_TO_CODE,
    BRIGHTNESS_LEVELS,
    BRIGHTNESS_TO_CODE,
    FEATURE_NAMES,
)


def validate_readings(readings):
    """Raise ``ValueError`` with a clear message if any reading is invalid."""
    missing = {"ambient", "occupancy", "hour"} - set(readings.columns)
    if missing:
        raise ValueError(f"Missing sensor column(s): {sorted(missing)}.")

    unknown = sorted(set(readings["ambient"].astype(str)) - set(AMBIENT_LEVELS))
    if unknown:
        raise ValueError(
            f"Unknown ambient level(s) {unknown}; expected one of {AMBIENT_LEVELS}."
        )

    occupancy = pd.to_numeric(readings["occupancy"], errors="coerce")
    if occupancy.isna().any() or not occupancy.isin([0, 1]).all():
        raise ValueError("occupancy must be 0 (empty) or 1 (occupied).")

    hour = pd.to_numeric(readings["hour"], errors="coerce")
    if hour.isna().any() or ((hour % 1) != 0).any() or not hour.between(0, 23).all():
        raise ValueError("hour must be an integer between 0 and 23.")


def encode_features(readings):
    """Convert raw sensor readings into the numeric model input frame."""
    validate_readings(readings)
    return pd.DataFrame(
        {
            "ambient_level": readings["ambient"].astype(str).map(AMBIENT_TO_CODE).astype(int),
            "occupancy": pd.to_numeric(readings["occupancy"]).astype(int),
            "hour": pd.to_numeric(readings["hour"]).astype(int),
        }
    )[FEATURE_NAMES]


def encode_target(labels):
    """Map brightness labels (e.g. '50%') to ordinal integer codes."""
    labels = pd.Series(labels).astype(str)
    unknown = sorted(set(labels) - set(BRIGHTNESS_LEVELS))
    if unknown:
        raise ValueError(
            f"Unknown brightness label(s) {unknown}; expected {BRIGHTNESS_LEVELS}."
        )
    return labels.map(BRIGHTNESS_TO_CODE).astype(int).to_numpy()


def decode_target(codes):
    """Map ordinal integer codes back to brightness labels."""
    return [BRIGHTNESS_LEVELS[int(code)] for code in codes]
