"""Synthetic IoT sensor data generators for the smart lighting simulation.

Two generators live here:

* ``generate_dataset`` - independent (hour, ambient, occupancy) readings with a
  deterministic control rule plus injected label noise. Used for TRAINING.
* ``simulate_days`` - a chronological, time-continuous stream (5-minute steps)
  with a daylight curve, cloud variation and persistent occupancy. Used for
  INFERENCE demos and the ENERGY analysis, where time must be consistent.
"""

import math
import random

import numpy as np
import pandas as pd

# Make `python src/<script>.py` work as well as `python -m src.<script>`.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from src.config import (
    AMBIENT_HIGH_MIN,
    AMBIENT_LEVELS,
    AMBIENT_LOW_MAX,
    BRIGHTNESS_LEVELS,
    DATA_PATH,
    N_SAMPLES,
    NIGHT_FLOOR,
    NOISE_RATIO,
    OCCUPANCY_LEAVE_PROB,
    RANDOM_SEED,
    SIM_DAYS,
    SIM_SEED,
    STEP_MINUTES,
    SUNRISE_HOUR,
    SUNSET_HOUR,
    create_directories,
)

NIGHT_HOURS = set(range(18, 24)) | set(range(0, 7))
PEAK_HOURS = set(range(10, 17))
MID_HOURS = set(range(7, 10)) | {17}


def set_seed(seed=RANDOM_SEED):
    """Seed both Python and NumPy random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ambient_probabilities(hour):
    """Return the probability of each ambient level for a given hour."""
    if hour in NIGHT_HOURS:
        return {"Low": 0.70, "Medium": 0.25, "High": 0.05}
    if hour in PEAK_HOURS:
        return {"Low": 0.10, "Medium": 0.30, "High": 0.60}
    return {"Low": 0.20, "Medium": 0.60, "High": 0.20}


def occupancy_probability(hour):
    """Return the probability that the room is occupied at a given hour."""
    return 0.85 if 8 <= hour <= 18 else 0.20


def sample_reading(rng):
    """Sample one synthetic (hour, ambient, occupancy) sensor reading."""
    hour = int(rng.integers(0, 24))
    probs = ambient_probabilities(hour)
    ambient = str(rng.choice(AMBIENT_LEVELS, p=[probs[level] for level in AMBIENT_LEVELS]))
    occupancy = int(rng.random() < occupancy_probability(hour))
    return ambient, occupancy, hour


def rule_brightness(ambient, occupancy, hour):
    """Compute the ideal brightness decision from the deterministic IoT rule."""
    if occupancy == 0:
        return "OFF"
    if ambient == "High":
        return "20%"
    if ambient == "Medium":
        return "50%"
    if hour >= 18 or hour <= 6:
        return "90%"
    return "80%"


def flip_label(label, rng):
    """Flip one brightness label to a neighbouring level."""
    level_index = BRIGHTNESS_LEVELS.index(label)
    neighbours = []
    if level_index > 0:
        neighbours.append(level_index - 1)
    if level_index < len(BRIGHTNESS_LEVELS) - 1:
        neighbours.append(level_index + 1)
    return BRIGHTNESS_LEVELS[int(rng.choice(neighbours))]


def add_label_noise(labels, rng, noise_ratio=NOISE_RATIO):
    """Flip a fraction of labels to an adjacent brightness level."""
    noisy = list(labels)
    flip_indices = rng.choice(
        len(noisy), size=int(round(noise_ratio * len(noisy))), replace=False
    )
    for index in flip_indices:
        noisy[index] = flip_label(noisy[index], rng)
    return noisy


def generate_dataset(n_samples=N_SAMPLES, noise_ratio=NOISE_RATIO, seed=RANDOM_SEED):
    """Build the synthetic training dataset as a pandas DataFrame.

    Columns: ``hour, ambient, occupancy`` (sensors), ``brightness`` (observed,
    noisy label) and ``ideal_brightness`` (clean rule output, kept so models can
    also be scored against the noise-free truth).
    """
    rng = set_seed(seed)
    readings = [sample_reading(rng) for _ in range(n_samples)]
    hours = [reading[2] for reading in readings]
    ambients = [reading[0] for reading in readings]
    occupancies = [reading[1] for reading in readings]
    ideal = [rule_brightness(a, o, h) for a, o, h in readings]
    brightness = add_label_noise(ideal, rng, noise_ratio)
    return pd.DataFrame(
        {
            "hour": hours,
            "ambient": ambients,
            "occupancy": occupancies,
            "brightness": brightness,
            "ideal_brightness": ideal,
        }
    )


def all_states():
    """Enumerate the COMPLETE input space (3 ambient x 2 occupancy x 24 hours).

    Because the sensor space is finite (144 states), a model can be verified
    exhaustively against the clean control rule - something real-world ML
    problems rarely allow.
    """
    states = pd.DataFrame(
        [(a, o, h) for a in AMBIENT_LEVELS for o in (0, 1) for h in range(24)],
        columns=["ambient", "occupancy", "hour"],
    )
    states["ideal_brightness"] = [
        rule_brightness(a, o, h)
        for a, o, h in zip(states["ambient"], states["occupancy"], states["hour"])
    ]
    return states


# ---------------------------------------------------------------------------
# Chronological day simulation
# ---------------------------------------------------------------------------
def classify_ambient(daylight_index):
    """Discretise the continuous daylight index into Low / Medium / High."""
    if daylight_index < AMBIENT_LOW_MAX:
        return "Low"
    if daylight_index < AMBIENT_HIGH_MIN:
        return "Medium"
    return "High"


def _sun_height(time_hours):
    """Idealised sun elevation in [0, 1]: 0 outside daylight, 1 at solar noon."""
    if not SUNRISE_HOUR <= time_hours <= SUNSET_HOUR:
        return 0.0
    span = SUNSET_HOUR - SUNRISE_HOUR
    return max(0.0, math.sin(math.pi * (time_hours - SUNRISE_HOUR) / span))


def simulate_days(n_days=SIM_DAYS, seed=SIM_SEED, step_minutes=STEP_MINUTES):
    """Simulate a chronological sensor stream (one row per ``step_minutes``).

    * Daylight follows a sine curve between sunrise and sunset, scaled by a
      per-day "clearness" (cloudy vs sunny day) plus slowly varying cloud noise.
    * Occupancy is a two-state Markov chain whose long-run occupied fraction at
      each hour matches ``occupancy_probability(hour)``; people stay for a
      realistic stretch of time instead of flickering every sample.

    Returns columns: day, step, time_h, hour, daylight_index, ambient, occupancy.
    """
    if 24 * 60 % step_minutes:
        raise ValueError("step_minutes must divide a day evenly.")
    rng = np.random.default_rng(seed)
    steps_per_day = 24 * 60 // step_minutes
    leave = OCCUPANCY_LEAVE_PROB
    occupied = False
    rows = []
    for day in range(n_days):
        clearness = float(rng.beta(5, 2))  # sunny vs overcast day
        cloud = 0.0
        for step in range(steps_per_day):
            time_h = step * step_minutes / 60
            hour = int(time_h)
            sun = _sun_height(time_h)
            cloud = 0.9 * cloud + float(rng.normal(0, 0.05))
            index = float(np.clip(NIGHT_FLOOR + sun * clearness + 0.5 * sun * cloud, 0, 1))

            target = occupancy_probability(hour)
            enter = min(1.0, target * leave / (1 - target))
            if occupied:
                occupied = bool(rng.random() > leave)
            else:
                occupied = bool(rng.random() < enter)

            rows.append(
                {
                    "day": day,
                    "step": step,
                    "time_h": round(time_h, 4),
                    "hour": hour,
                    "daylight_index": round(index, 4),
                    "ambient": classify_ambient(index),
                    "occupancy": int(occupied),
                }
            )
    return pd.DataFrame(rows)


def main():
    """Generate, describe and persist the synthetic sensor dataset."""
    create_directories()
    data = generate_dataset()
    data.to_csv(DATA_PATH, index=False)

    print("=" * 68)
    print("SMART LIGHTING - SYNTHETIC SENSOR DATA GENERATION")
    print("=" * 68)
    print(f"Saved to: {DATA_PATH}")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head().to_string(index=False))
    print("\nBrightness value counts:")
    print(data["brightness"].value_counts().to_string())
    print("\nClass balance:")
    balance = data["brightness"].value_counts(normalize=True).mul(100).round(2)
    print(balance.to_string())
    agreement = (data["brightness"] == data["ideal_brightness"]).mean()
    print(f"\nNoise ratio applied: {NOISE_RATIO:.0%} "
          f"(observed labels match the clean rule on {agreement:.1%} of rows)")
    print("=" * 68)


if __name__ == "__main__":
    main()
