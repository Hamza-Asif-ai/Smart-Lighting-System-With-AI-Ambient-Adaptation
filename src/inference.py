"""Real-time IoT inference for the smart lighting simulator.

Provides reusable prediction functions for the Streamlit dashboard and the
tests, a logged random-reading loop, and a chronological day-simulation helper.
"""

import argparse
import json
import sys
import time
import warnings
from functools import lru_cache

import joblib
import pandas as pd
import sklearn

# Make `python src/<script>.py` work as well as `python -m src.<script>`.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from src.config import (
    BRIGHTNESS_LEVELS,
    META_PATH,
    MODEL_PATH,
    NOISE_RATIO,
    SCHEMA_VERSION,
    SIM_SEED,
    SIMULATION_LOG_CSV,
    create_directories,
)
from src.features import decode_target, encode_features
from src.sensor_simulator import (
    flip_label,
    rule_brightness,
    sample_reading,
    set_seed,
    simulate_days,
)

LOG_COLUMNS = [
    "step",
    "ambient",
    "occupancy",
    "hour",
    "predicted_brightness",
    "ideal_brightness",
    "ground_truth",
]


def _try_load_artifacts():
    """Return (model, meta) if usable artefacts exist, otherwise None."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            return None
        if meta.get("sklearn_version") != sklearn.__version__:
            return None  # pickles are only guaranteed within one sklearn version
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # treat pickle-version warnings as failure
            model = joblib.load(MODEL_PATH)
        return model, meta
    except Exception:  # corrupt / incompatible artefact -> rebuild
        return None


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the trained model and its metadata, rebuilding them if necessary.

    Artefacts are rebuilt automatically when they are missing, corrupt, from an
    older schema or were pickled by a different scikit-learn version, so a fresh
    clone (or an upgraded environment) never crashes the dashboard.
    """
    loaded = _try_load_artifacts()
    if loaded is None:
        from src.train_model import run_training  # local import avoids a cycle

        print("[inference] Model artefacts missing or incompatible - retraining...",
              file=sys.stderr)
        run_training(verbose=False, make_plots=False)
        loaded = _try_load_artifacts()
        if loaded is None:
            raise RuntimeError("Could not create usable model artefacts.")
    return loaded


def predict_actions(readings):
    """Predict brightness for many readings at once (vectorised).

    ``readings`` is a DataFrame with ``ambient``, ``occupancy`` and ``hour``.
    Returns a list of brightness labels aligned with the input rows.
    """
    model, _ = load_artifacts()
    return decode_target(model.predict(encode_features(readings)))


def predict_action(ambient, occupancy, hour):
    """Predict the brightness level for one (ambient, occupancy, hour) reading."""
    reading = pd.DataFrame([{"ambient": ambient, "occupancy": occupancy, "hour": hour}])
    return predict_actions(reading)[0]


def generate_simulation(n_steps=100, seed=SIM_SEED, verbose=False, delay=0.0):
    """Simulate ``n_steps`` independent sensor readings and log the decisions.

    ``ideal_brightness`` is the clean rule output; ``ground_truth`` is the same
    value passed through the label-noise process, mimicking imperfect observed
    labels.
    """
    rng = set_seed(seed)
    readings, ideals, observed = [], [], []
    for _ in range(n_steps):
        ambient, occupancy, hour = sample_reading(rng)
        ideal = rule_brightness(ambient, occupancy, hour)
        truth = flip_label(ideal, rng) if rng.random() < NOISE_RATIO else ideal
        readings.append({"ambient": ambient, "occupancy": occupancy, "hour": hour})
        ideals.append(ideal)
        observed.append(truth)

    frame = pd.DataFrame(readings)
    predictions = predict_actions(frame)

    for step, row in enumerate(frame.itertuples(index=False), start=1):
        if verbose:
            print(
                f"[{step:>3}] ambient={row.ambient:<6} occ={row.occupancy} "
                f"hour={row.hour:>2} -> Predicted: {predictions[step - 1]:<3} "
                f"(rule: {ideals[step - 1]}, observed: {observed[step - 1]})"
            )
        if delay:
            time.sleep(delay)

    frame.insert(0, "step", range(1, n_steps + 1))
    frame["predicted_brightness"] = predictions
    frame["ideal_brightness"] = ideals
    frame["ground_truth"] = observed
    return frame[LOG_COLUMNS]


def simulate_day_decisions(n_days=1, seed=SIM_SEED):
    """Run the model over a chronological multi-day sensor stream."""
    stream = simulate_days(n_days=n_days, seed=seed)
    stream["predicted_brightness"] = predict_actions(stream)
    stream["ideal_brightness"] = [
        rule_brightness(a, o, h)
        for a, o, h in zip(stream["ambient"], stream["occupancy"], stream["hour"])
    ]
    return stream


def main():
    """Run the live-style inference loop and report its accuracy."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100, help="number of readings")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds between readings (e.g. 0.05 for a live feel)")
    args = parser.parse_args()

    create_directories()
    print("=" * 68)
    print(f"REAL-TIME INFERENCE SIMULATION ({args.steps} readings)")
    print("=" * 68)
    log = generate_simulation(n_steps=args.steps, verbose=True, delay=args.delay)
    log.to_csv(SIMULATION_LOG_CSV, index=False)

    vs_rule = (log["predicted_brightness"] == log["ideal_brightness"]).mean()
    vs_observed = (log["predicted_brightness"] == log["ground_truth"]).mean()
    print("-" * 68)
    print(f"Agreement with the control rule      : {vs_rule:.2%}")
    print(f"Agreement with noisy observed labels : {vs_observed:.2%} "
          f"(capped near {1 - NOISE_RATIO:.0%} by label noise)")
    print(f"Simulation log saved to: {SIMULATION_LOG_CSV}")
    print("=" * 68)


if __name__ == "__main__":
    main()
