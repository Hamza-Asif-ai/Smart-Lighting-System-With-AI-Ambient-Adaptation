"""Headless tests of the Streamlit dashboard (real widgets, no browser needed).

Run with:  python -m pytest -q
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def _walk(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _walk(child)


def _iframes(app):
    return [
        element.proto
        for element in _walk(app.main)
        if getattr(element, "proto", None) is not None
        and element.proto.__class__.__name__ == "IFrame"
    ]


def _live_fade(app):
    """(from, to, fade_seconds) embedded in the live-bulb iframe."""
    live = [frame for frame in _iframes(app) if "__lightingPlayer" not in frame.srcdoc][0]
    return tuple(float(v) for v in re.search(
        r"var from=([\d.]+),to=([\d.]+),fade=([\d.]+)", live.srcdoc).groups())


def _metrics(app):
    return {metric.label: metric.value for metric in app.metric}


@pytest.fixture()
def app():
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def test_app_loads_without_errors_and_has_five_tabs(app):
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Live Prediction", "Virtual Bulb", "Day Simulation", "Model Performance", "About"
    ]


def test_default_sensor_values_drive_the_live_and_actuator_tabs(app):
    metrics = _metrics(app)
    # defaults: Medium ambient, occupied, 22:00 -> 50 %
    assert metrics["AI Predicted Brightness"] == "50%"
    assert metrics["Control-rule Brightness"] == "50%"
    assert metrics["AI command"] == "50%"
    assert metrics["PWM duty"] == "128 / 255"
    assert metrics["Bulb power"] == "30 W"
    assert metrics["Fade time"] == "1.0 s"          # first render fades up from dark


def test_virtual_bulb_and_day_player_are_embedded(app):
    frames = _iframes(app)
    assert len(frames) == 2
    kinds = sorted("player" if "__lightingPlayer" in f.srcdoc else "bulb" for f in frames)
    assert kinds == ["bulb", "player"]
    assert all(f.srcdoc.startswith("<!DOCTYPE html>") for f in frames)


def test_changing_sensors_updates_command_pwm_and_fade_from_previous_level(app):
    assert _live_fade(app) == (0.0, 50.0, 1.0)
    app.sidebar.selectbox[0].select("High")
    app.sidebar.slider[0].set_value(12)
    app.run()
    assert not app.exception
    metrics = _metrics(app)
    assert (metrics["AI command"], metrics["PWM duty"], metrics["Bulb power"]) == (
        "20%", "51 / 255", "12 W")
    assert _live_fade(app) == (50.0, 20.0, pytest.approx(0.6))   # fades from the previous 50 %
    assert metrics["Fade time"] == "0.6 s"


def test_empty_room_turns_the_bulb_off(app):
    app.sidebar.radio[0].set_value(0)
    app.run()
    assert not app.exception
    metrics = _metrics(app)
    assert metrics["AI command"] == "OFF"
    assert metrics["PWM duty"] == "0 / 255"
    assert metrics["Bulb power"] == "0 W"
    # the bulb was at 50 % after the first render, so it fades 50 -> 0 in 1.0 s
    assert _live_fade(app) == (50.0, 0.0, 1.0)
    assert metrics["Fade time"] == "1.0 s"


def test_unchanged_sensors_do_not_replay_the_fade(app):
    app.sidebar.slider[0].set_value(22)        # same value as before -> same state
    app.run()
    assert _live_fade(app) == (50.0, 50.0, 0.0)
    assert _metrics(app)["Fade time"] == "0.0 s"


def test_changing_the_seed_changes_the_simulated_day(app):
    before = _metrics(app)["Energy that day"]
    app.sidebar.number_input[0].set_value(7)
    app.run()
    assert not app.exception
    after = _metrics(app)
    assert after["Energy that day"] != before
    assert int(after["Commands sent"]) > 0


def test_actuator_energy_matches_day_tab_ai_energy(app):
    """Both tabs simulate the same day, so their energy must agree."""
    wh = float(_metrics(app)["Energy that day"].split()[0])
    day_tab_kwh = float([m.value for m in app.metric if m.label == "AI model"][0].split()[0])
    assert wh / 1000 == pytest.approx(day_tab_kwh, abs=0.006)  # day tab shows 2 decimals


def test_every_sidebar_combination_runs_cleanly():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    for ambient in ("Low", "Medium", "High"):
        for occupancy in (0, 1):
            for hour in (0, 6, 12, 18, 23):
                app.sidebar.selectbox[0].select(ambient)
                app.sidebar.radio[0].set_value(occupancy)
                app.sidebar.slider[0].set_value(hour)
                app.run()
                assert not app.exception, (ambient, occupancy, hour)
                metrics = _metrics(app)
                assert metrics["AI command"] == metrics["AI Predicted Brightness"]
                if occupancy == 0:
                    assert metrics["AI command"] == "OFF"
                else:
                    assert metrics["AI command"] != "OFF"


def test_about_tab_is_accurate_and_complete(app):
    """Every claim in the About tab must match the saved model and a fresh calculation."""
    import json

    import numpy as np

    from src.config import META_PATH, WATT_MAP
    from src.inference import simulate_day_decisions

    text = "\n".join(item.value for item in app.tabs[4].markdown)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    metrics = meta["metrics"]

    # headline model numbers come straight from the saved metadata
    assert meta["model_name"] in text
    assert f"F1 {metrics['cv_f1_mean']:.3f}" in text
    assert f"held-out accuracy {metrics['test_accuracy']:.1%}" in text
    assert "144 possible sensor states" in text and f"{metrics['rule_coverage']:.0%}" in text

    # energy numbers recomputed independently from a 30-day stream
    stream = simulate_day_decisions(n_days=30)
    hours = 5 / 60
    always = 60 * len(stream) * hours / 1000
    occupancy_only = 60 * stream["occupancy"].sum() * hours / 1000
    ai = stream["predicted_brightness"].map(WATT_MAP).sum() * hours / 1000
    assert f"always-on {always:.2f} kWh" in text
    assert f"occupancy-only {occupancy_only:.2f} kWh" in text
    assert f"AI {ai:.2f} kWh" in text
    assert f"{(occupancy_only - ai) / occupancy_only * 100:.1f}% saved" in text
    assert f"{(always - ai) / always * 100:.1f}% versus always-on" in text
    occupied = int(stream["occupancy"].sum())
    unsafe = int(((stream["occupancy"] == 1) & (stream["predicted_brightness"] == "OFF")).sum())
    assert f"{unsafe} of {occupied:,} occupied steps" in text

    # method description matches the real configuration
    for fragment in ("3,000 reproducible", "7% label noise", "5-fold", "80/20",
                     "30-day simulation", "5-minute steps", "8-bit PWM", "0-255",
                     "50 %/s", "PKR 25/kWh", "60 W-equivalent", "18:00-06:59",
                     "07:00-17:59", "Limitations", "synthetic"):
        assert fragment in text, fragment

    # every tab is described, and no template placeholder leaked into the page
    for tab_name in ("Live Prediction", "Virtual Bulb", "Day Simulation",
                     "Model Performance", "About"):
        assert f"**{tab_name}**" in text
    assert "{" not in text and "}" not in text
    assert any("23-AI-93" in caption.value for caption in app.tabs[4].caption)
