"""Automated checks for the smart lighting pipeline.

Run with:  python -m pytest -q
"""

import json

import numpy as np
import pandas as pd
import pytest

import energy_analysis
import inference
import train_model
from config import (
    BRIGHTNESS_LEVELS,
    BULB_FULL_WATT,
    NOISE_RATIO,
    N_SAMPLES,
    SIM_DAYS,
    STEP_MINUTES,
)
from features import decode_target, encode_features, encode_target
from sensor_simulator import (
    all_states,
    generate_dataset,
    rule_brightness,
    simulate_days,
)


# ---------------------------------------------------------------------------
# Control rule and dataset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ambient, occupancy, hour, expected",
    [
        ("High", 0, 12, "OFF"),
        ("Low", 0, 22, "OFF"),
        ("High", 1, 12, "20%"),
        ("Medium", 1, 9, "50%"),
        ("Low", 1, 12, "80%"),   # dark in the daytime
        ("Low", 1, 22, "90%"),   # dark at night
        ("Low", 1, 3, "90%"),
        ("Low", 1, 6, "90%"),    # boundary: hour <= 6 is still "night"
        ("Low", 1, 7, "80%"),
        ("Low", 1, 18, "90%"),   # boundary: hour >= 18 is "night"
    ],
)
def test_rule_truth_table(ambient, occupancy, hour, expected):
    assert rule_brightness(ambient, occupancy, hour) == expected


def test_dataset_shape_and_validity():
    data = generate_dataset()
    assert len(data) == N_SAMPLES
    assert set(data["brightness"]) <= set(BRIGHTNESS_LEVELS)
    assert data["hour"].between(0, 23).all()
    assert set(data["occupancy"]) <= {0, 1}


def test_dataset_is_reproducible():
    pd.testing.assert_frame_equal(generate_dataset(seed=7), generate_dataset(seed=7))


def test_label_noise_is_exact_and_adjacent():
    data = generate_dataset()
    changed = data["brightness"] != data["ideal_brightness"]
    assert changed.sum() == round(NOISE_RATIO * N_SAMPLES)
    gap = (
        data.loc[changed, "brightness"].map(BRIGHTNESS_LEVELS.index)
        - data.loc[changed, "ideal_brightness"].map(BRIGHTNESS_LEVELS.index)
    ).abs()
    assert (gap == 1).all(), "noise must only move a label to a neighbouring level"


def test_ideal_column_matches_rule():
    data = generate_dataset()
    expected = [rule_brightness(a, o, h)
                for a, o, h in zip(data["ambient"], data["occupancy"], data["hour"])]
    assert data["ideal_brightness"].tolist() == expected


# ---------------------------------------------------------------------------
# Chronological day simulation
# ---------------------------------------------------------------------------
def test_day_simulation_structure():
    stream = simulate_days()
    assert len(stream) == SIM_DAYS * 24 * 60 // STEP_MINUTES
    assert stream["hour"].between(0, 23).all()
    assert set(stream["ambient"]) == {"Low", "Medium", "High"}
    assert set(stream["occupancy"]) <= {0, 1}
    assert stream["daylight_index"].between(0, 1).all()
    pd.testing.assert_frame_equal(stream, simulate_days())  # deterministic


def test_day_simulation_is_physically_plausible():
    stream = simulate_days()
    night = stream[stream["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23])]
    noon = stream[stream["hour"].isin([11, 12, 13])]
    assert (night["ambient"] == "Low").mean() > 0.95
    assert (noon["ambient"] == "High").mean() > 0.5
    work = stream[stream["hour"].between(9, 16)]["occupancy"].mean()
    sleep = stream[stream["hour"].between(0, 5)]["occupancy"].mean()
    assert work > 0.7 > 0.4 > sleep


def test_occupancy_is_persistent_not_flickering():
    occupancy = simulate_days(n_days=5)["occupancy"].to_numpy()
    switches = np.count_nonzero(np.diff(occupancy))
    assert switches < 0.2 * len(occupancy), "occupancy should stay put for a while"


# ---------------------------------------------------------------------------
# Feature encoding and validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ambient, occupancy, hour",
    [("Dark", 1, 5), ("low", 1, 5), ("Low", 2, 5), ("Low", -1, 5),
     ("Low", 1, 24), ("Low", 1, -1), ("Low", 1, 5.5), ("Low", None, 5)],
)
def test_invalid_readings_are_rejected(ambient, occupancy, hour):
    with pytest.raises(ValueError):
        inference.predict_action(ambient, occupancy, hour)


def test_valid_edge_readings_are_accepted():
    assert inference.predict_action("Low", True, 0) in BRIGHTNESS_LEVELS
    assert inference.predict_action("High", 0, 23) == "OFF"
    assert inference.predict_action("Medium", 1, np.int64(12)) == "50%"


def test_target_round_trip():
    codes = encode_target(BRIGHTNESS_LEVELS)
    assert decode_target(codes) == BRIGHTNESS_LEVELS
    with pytest.raises(ValueError):
        encode_target(["75%"])


def test_ambient_encoding_is_ordinal():
    frame = pd.DataFrame({"ambient": ["Low", "Medium", "High"],
                          "occupancy": [1, 1, 1], "hour": [1, 2, 3]})
    assert encode_features(frame)["ambient_level"].tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Model quality
# ---------------------------------------------------------------------------
def test_deployed_model_matches_rule_on_every_possible_state():
    """The sensor space is finite (144 states), so verify it exhaustively."""
    states = all_states()
    assert len(states) == 144
    predictions = inference.predict_actions(states)
    wrong = states[np.array(predictions) != states["ideal_brightness"].to_numpy()]
    assert wrong.empty, f"model disagrees with the rule on:\n{wrong}"


def test_safety_never_off_when_occupied_and_off_when_empty():
    states = all_states()
    states["predicted"] = inference.predict_actions(states)
    assert (states.loc[states["occupancy"] == 1, "predicted"] != "OFF").all()
    assert (states.loc[states["occupancy"] == 0, "predicted"] == "OFF").all()


def test_batch_prediction_matches_single_prediction():
    states = all_states().sample(25, random_state=1)
    batch = inference.predict_actions(states)
    single = [inference.predict_action(a, o, h)
              for a, o, h in zip(states["ambient"], states["occupancy"], states["hour"])]
    assert batch == single


def test_model_selection_prefers_smallest_among_near_ties():
    table = pd.DataFrame(
        {
            "Model": ["Big", "Small", "Worse"],
            "CV F1 (mean)": [0.9300, 0.9280, 0.8000],
            "Size (KB)": [2000.0, 6.0, 1.0],
        }
    )
    assert train_model.select_best(table, tolerance=0.005) == "Small"
    assert train_model.select_best(table, tolerance=0.0) == "Big"


def test_training_holds_out_test_data_and_reports_cv():
    data = generate_dataset()
    x_tr, x_te, y_tr, y_te, _, c_te = train_model.split_data(data)
    assert len(x_tr) + len(x_te) == len(data)
    comparison, fitted = train_model.evaluate_models(
        train_model.build_models(), x_tr, y_tr, x_te, y_te, c_te
    )
    assert set(comparison["Model"]) == {"DecisionTree", "KNeighbors", "RandomForest"}
    # accuracy against noisy labels cannot meaningfully exceed the noise ceiling
    assert (comparison["Test Accuracy"] < 0.97).all()
    assert (comparison["Clean-Label Accuracy"] > 0.95).all()


def test_inference_log_is_reproducible_and_accurate():
    first = inference.generate_simulation(n_steps=200, seed=5)
    second = inference.generate_simulation(n_steps=200, seed=5)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == inference.LOG_COLUMNS
    assert (first["predicted_brightness"] == first["ideal_brightness"]).mean() > 0.99


# ---------------------------------------------------------------------------
# Energy maths (regression tests for the original unit bug)
# ---------------------------------------------------------------------------
def _constant_day(occupancy, level):
    steps = 24 * 60 // STEP_MINUTES
    return pd.DataFrame(
        {
            "hour": np.repeat(np.arange(24), steps // 24),
            "occupancy": occupancy,
            "ideal_brightness": level,
            "predicted_brightness": level,
        }
    )


def test_energy_units_are_physical():
    """60 W for 24 h is 1.44 kWh/day -> 43.2 kWh over 30 days."""
    result = energy_analysis.compute_energy(_constant_day(1, "50%"))
    assert result["monthly_kwh"]["always_on"] == pytest.approx(43.2)
    assert result["monthly_kwh"]["occupancy_only"] == pytest.approx(43.2)
    assert result["monthly_kwh"]["ai"] == pytest.approx(21.6)  # 30 W * 720 h
    assert result["saving_vs_occupancy_pct"] == pytest.approx(50.0)
    assert result["monthly_cost_pkr"]["ai"] == pytest.approx(21.6 * 25)


def test_empty_room_uses_no_energy_with_control():
    result = energy_analysis.compute_energy(_constant_day(0, "OFF"))
    assert result["monthly_kwh"]["ai"] == 0
    assert result["monthly_kwh"]["occupancy_only"] == 0
    assert result["monthly_kwh"]["always_on"] == pytest.approx(43.2)
    # no occupancy -> baseline is 0 kWh; savings must be 0, never NaN / inf
    assert result["saving_vs_occupancy_pct"] == 0.0
    assert result["rule_saving_vs_occupancy_pct"] == 0.0


def test_energy_is_independent_of_stream_length():
    one_day = energy_analysis.compute_energy(_constant_day(1, "80%"))
    three_days = energy_analysis.compute_energy(
        pd.concat([_constant_day(1, "80%")] * 3, ignore_index=True)
    )
    assert one_day["monthly_kwh"]["ai"] == pytest.approx(three_days["monthly_kwh"]["ai"])


def test_real_simulation_never_exceeds_physical_maximum():
    stream = inference.simulate_day_decisions(n_days=SIM_DAYS)
    result = energy_analysis.compute_energy(stream)
    physical_max = BULB_FULL_WATT * 24 * 30 / 1000
    assert result["monthly_kwh"]["always_on"] == pytest.approx(physical_max)
    for key, value in result["monthly_kwh"].items():
        assert 0 <= value <= physical_max + 1e-9, key
    assert result["monthly_kwh"]["ai"] <= result["monthly_kwh"]["occupancy_only"]
    assert 0 < result["saving_vs_occupancy_pct"] < 100
    assert result["occupied_but_off_steps"] == 0


# ---------------------------------------------------------------------------
# Artefact auto-recovery
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    """Point every artefact path at a temp dir so tests never touch real files."""
    paths = {
        "MODEL_PATH": tmp_path / "model.pkl",
        "META_PATH": tmp_path / "meta.json",
        "COMPARISON_CSV": tmp_path / "comparison.csv",
        "RULES_TXT": tmp_path / "rules.txt",
    }
    for name, path in paths.items():
        if hasattr(inference, name):
            monkeypatch.setattr(inference, name, path)
        if hasattr(train_model, name):
            monkeypatch.setattr(train_model, name, path)
    inference.load_artifacts.cache_clear()
    yield paths
    inference.load_artifacts.cache_clear()


def test_missing_artifacts_are_rebuilt_automatically(isolated_artifacts):
    assert not isolated_artifacts["MODEL_PATH"].exists()
    assert inference.predict_action("Low", 1, 22) == "90%"
    assert isolated_artifacts["MODEL_PATH"].exists()
    meta = json.loads(isolated_artifacts["META_PATH"].read_text())
    assert meta["metrics"]["rule_coverage"] == 1.0


def test_incompatible_artifacts_trigger_rebuild(isolated_artifacts):
    inference.predict_action("Low", 1, 22)  # builds artefacts
    inference.load_artifacts.cache_clear()
    meta = json.loads(isolated_artifacts["META_PATH"].read_text())
    meta["sklearn_version"] = "0.0.0-other"
    isolated_artifacts["META_PATH"].write_text(json.dumps(meta))
    assert inference._try_load_artifacts() is None

    isolated_artifacts["MODEL_PATH"].write_bytes(b"corrupt")
    assert inference._try_load_artifacts() is None
    assert inference.predict_action("High", 1, 12) == "20%"  # rebuilt and working
