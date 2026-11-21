"""Tests for the simulated actuator (dimmable LED driver).

Run with:  python -m pytest -q
"""

import json
import re

import matplotlib
import numpy as np
import pandas as pd
import pytest

from src import actuator
from src import inference
from src.actuator import (
    DimmableBulb,
    bulb_html,
    clock_label,
    command_message,
    day_player_html,
    fade_closeup,
    fade_time_s,
    level_to_pct,
    pwm_duty,
    simulate_actuator,
)
from src.config import (
    BRIGHTNESS_LEVELS,
    BULB_FULL_WATT,
    FADE_RATE_PCT_PER_S,
    MQTT_TOPIC,
    PWM_BITS,
    STEP_MINUTES,
    WATT_MAP,
)

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_level_to_pct_covers_every_label_and_rejects_unknown():
    assert [level_to_pct(label) for label in BRIGHTNESS_LEVELS] == [0, 20, 50, 80, 90]
    for bad in ("75%", "100%", "on", "", None):
        with pytest.raises(ValueError):
            level_to_pct(bad)


@pytest.mark.parametrize("label", BRIGHTNESS_LEVELS)
def test_bulb_power_matches_the_energy_models_watt_map(label):
    """The actuator and energy_analysis must agree on watts per level."""
    assert DimmableBulb(initial=label).power_w == pytest.approx(WATT_MAP[label])


def test_pwm_duty_values_and_monotonicity():
    maximum = 2 ** PWM_BITS - 1
    assert pwm_duty(0) == 0
    assert pwm_duty(100) == maximum
    assert {pct: pwm_duty(pct) for pct in (20, 50, 80, 90)} == {20: 51, 50: 128, 80: 204, 90: 230}
    duties = [pwm_duty(pct) for pct in np.linspace(0, 100, 201)]
    assert duties == sorted(duties)
    assert pwm_duty(-5) == 0 and pwm_duty(500) == maximum  # clamped, never out of range


def test_fade_time_is_delta_over_rate():
    assert fade_time_s(0, 90, 50) == pytest.approx(1.8)
    assert fade_time_s(90, 20, 50) == pytest.approx(1.4)
    assert fade_time_s(30, 30, 50) == 0
    with pytest.raises(ValueError):
        fade_time_s(0, 10, 0)


def test_clock_label():
    assert clock_label(0) == "Day 1 00:00"
    assert clock_label(100) == "Day 1 08:20"
    assert clock_label(287) == "Day 1 23:55"
    assert clock_label(288) == "Day 2 00:00"


def test_mqtt_message_is_valid_json_with_consistent_fields():
    message = command_message("80%", from_pct=0.0)
    assert message["topic"] == MQTT_TOPIC
    payload = json.loads(message["payload"])
    assert payload == {"state": "ON", "brightness": 80, "pwm": 204, "fade_s": 1.6}
    off = json.loads(command_message("OFF", from_pct=90.0)["payload"])
    assert off == {"state": "OFF", "brightness": 0, "pwm": 0, "fade_s": 1.8}
    with pytest.raises(ValueError):
        command_message("42%")


# ---------------------------------------------------------------------------
# DimmableBulb physics
# ---------------------------------------------------------------------------
def test_ramp_up_energy_is_exact():
    """OFF -> 90 % at 50 %/s: 1.8 s ramp, average 45 % of 60 W = 27 W."""
    bulb = DimmableBulb()
    bulb.command("90%")
    assert bulb.state == "FADING"
    power = bulb.advance(1.8)
    assert power == pytest.approx(27.0)
    assert bulb.energy_wh == pytest.approx(27.0 * 1.8 / 3600)
    assert bulb.level_pct == pytest.approx(90.0)
    assert bulb.state == "ON"
    assert bulb.pwm_duty == 230


def test_mid_fade_state_and_level():
    bulb = DimmableBulb()
    bulb.command("90%")
    bulb.advance(0.9)
    assert bulb.level_pct == pytest.approx(45.0)
    assert bulb.state == "FADING"
    assert bulb.pwm_duty == pwm_duty(45.0)


def test_ramp_then_hold_within_one_interval():
    """OFF -> 20 % (0.4 s ramp, avg 6 W) then 12 W for the remaining 9.6 s."""
    bulb = DimmableBulb()
    bulb.command("20%")
    average = bulb.advance(10.0)
    expected_wh = (6.0 * 0.4 + 12.0 * 9.6) / 3600
    assert bulb.energy_wh == pytest.approx(expected_wh)
    assert average == pytest.approx(expected_wh * 3600 / 10.0)
    assert bulb.state == "ON"


def test_ramp_down_is_symmetric():
    bulb = DimmableBulb(initial="90%")
    bulb.command("OFF")
    bulb.advance(1.8)
    assert bulb.level_pct == pytest.approx(0.0)
    assert bulb.state == "OFF"
    assert bulb.energy_wh == pytest.approx(27.0 * 1.8 / 3600)


def test_steady_energy_is_power_times_time():
    bulb = DimmableBulb(initial="50%")
    bulb.advance(3600)
    assert bulb.energy_wh == pytest.approx(30.0)  # 30 W for one hour
    assert bulb.elapsed_s == pytest.approx(3600)


def test_new_command_mid_fade_continues_from_current_level():
    bulb = DimmableBulb()
    bulb.command("90%")
    bulb.advance(0.9)                 # level 45 %
    bulb.command("20%")               # reverse direction
    bulb.advance(0.3)                 # 45 -> 30 %
    assert bulb.level_pct == pytest.approx(30.0)
    bulb.advance(5)
    assert bulb.level_pct == pytest.approx(20.0)
    assert bulb.state == "ON"


def test_zero_and_negative_intervals():
    bulb = DimmableBulb(initial="50%")
    assert bulb.advance(0) == pytest.approx(30.0)
    assert bulb.energy_wh == 0
    with pytest.raises(ValueError):
        bulb.advance(-1)


def test_level_never_leaves_0_to_100():
    bulb = DimmableBulb()
    for label in ["90%", "OFF", "80%", "20%", "OFF", "90%"]:
        bulb.command(label)
        for _ in range(7):
            bulb.advance(0.37)
            assert 0.0 <= bulb.level_pct <= 100.0


def test_redundant_and_invalid_commands():
    bulb = DimmableBulb(initial="50%")
    assert bulb.command("50%") is None           # unchanged target -> no message
    assert bulb.events == [] and bulb.level_changes == 0
    with pytest.raises(ValueError):
        bulb.command("75%")
    assert bulb.target_label == "50%"            # a rejected command changes nothing


def test_counters_and_events():
    bulb = DimmableBulb()
    for label in ["20%", "OFF", "50%", "50%", "OFF"]:
        bulb.command(label, time_label=label)
        bulb.advance(5)
    assert bulb.level_changes == 4                # the repeated 50% was ignored
    assert bulb.turn_on_events == 2               # OFF -> 20 %, OFF -> 50 %
    assert [event["to"] for event in bulb.events] == ["20%", "OFF", "50%", "OFF"]
    assert [event["from"] for event in bulb.events] == ["OFF", "20%", "OFF", "50%"]
    assert all(event["topic"] == MQTT_TOPIC for event in bulb.events)
    assert json.loads(bulb.events[0]["payload"])["pwm"] == 51


def test_constructor_validation():
    with pytest.raises(ValueError):
        DimmableBulb(fade_rate=0)
    with pytest.raises(ValueError):
        DimmableBulb(rated_watt=-1)
    with pytest.raises(ValueError):
        DimmableBulb(initial="bogus")


def test_custom_fade_rate_and_rating():
    bulb = DimmableBulb(rated_watt=100, fade_rate=10)  # slow 10 %/s dimmer, 100 W lamp
    bulb.command("50%")
    bulb.advance(5)
    assert bulb.level_pct == pytest.approx(50.0)
    # 5 s linear ramp averaging 25 % of 100 W = 25 W  ->  125 W*s
    assert bulb.energy_wh == pytest.approx(25.0 * 5 / 3600)
    assert bulb.power_w == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Driving the actuator with a day of AI decisions
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def day_run():
    stream = inference.simulate_day_decisions(n_days=1)
    return stream, simulate_actuator(stream)


def test_simulation_log_structure(day_run):
    stream, (log, events, summary) = day_run
    assert len(log) == 24 * 60 // STEP_MINUTES == 288
    for column in ["commanded", "actual_pct", "pwm_duty", "avg_power_w",
                   "energy_wh_cum", "state", "command_sent"]:
        assert column in log.columns
    for column in stream.columns:                      # sensor columns are carried through
        assert column in log.columns
    assert list(events.columns) == ["time", "from", "to", "topic", "payload"]


def test_bulb_reaches_commanded_level_within_every_step(day_run):
    """The fade takes seconds, a step is 5 minutes: level must match the command."""
    _, (log, _, _) = day_run
    expected = log["commanded"].map(level_to_pct)
    assert np.allclose(log["actual_pct"], expected)
    assert (log["pwm_duty"] == expected.map(pwm_duty)).all()
    assert set(log["state"]) <= {"ON", "OFF"}           # never stuck mid-fade at step end


def test_safety_lit_when_occupied_dark_when_empty(day_run):
    _, (log, _, _) = day_run
    assert (log.loc[log["occupancy"] == 0, "actual_pct"] == 0).all()
    assert (log.loc[log["occupancy"] == 1, "actual_pct"] > 0).all()


def test_events_match_decision_changes(day_run):
    _, (log, events, summary) = day_run
    labels = ["OFF"] + log["commanded"].tolist()        # the bulb starts OFF
    changes = sum(1 for before, after in zip(labels, labels[1:]) if before != after)
    assert len(events) == changes == summary["commands_sent"] == summary["level_changes"]
    assert log["command_sent"].sum() == changes
    assert summary["turn_on_events"] == int(((events["from"] == "OFF")).sum())
    assert events["time"].str.startswith("Day 1").all()


def test_energy_is_cumulative_and_consistent_with_instant_model(day_run):
    _, (log, _, summary) = day_run
    assert log["energy_wh_cum"].is_monotonic_increasing
    assert log["energy_wh_cum"].iloc[-1] == pytest.approx(summary["energy_wh"], rel=1e-6)
    # fades last seconds in 5-minute steps, so the instant-switch energy model holds
    assert abs(summary["fade_effect_pct"]) < 0.5
    watts = log["commanded"].map(WATT_MAP).sum() * STEP_MINUTES / 60.0
    assert summary["energy_wh"] == pytest.approx(watts, rel=0.005)
    assert (log["avg_power_w"] <= BULB_FULL_WATT + 1e-9).all()


def test_multi_day_stream_keeps_state_and_labels_days():
    stream = inference.simulate_day_decisions(n_days=2)
    log, events, summary = simulate_actuator(stream)
    assert len(log) == 2 * 288
    assert summary["days"] == pytest.approx(2.0)
    assert events["time"].str.startswith("Day 2").any()


def test_simulate_actuator_requires_decisions_column():
    with pytest.raises(ValueError):
        simulate_actuator(pd.DataFrame({"hour": [1, 2]}))


def test_simulate_actuator_rejects_invalid_label():
    with pytest.raises(ValueError):
        simulate_actuator(pd.DataFrame({"predicted_brightness": ["OFF", "75%"]}))


def test_fade_closeup_trace():
    trace = fade_closeup()
    assert trace["level_pct"].between(0, 100).all()
    assert trace["level_pct"].iloc[0] == 0
    at = lambda t: float(trace.loc[(trace["t_s"] - t).abs().idxmin(), "level_pct"])  # noqa: E731
    assert at(1.9) == pytest.approx(45.0, abs=2.6)      # halfway up the 0 -> 90 ramp
    assert at(4.9) == pytest.approx(90.0)               # settled before the next command
    assert at(7.9) == pytest.approx(50.0)
    assert at(11.9) == pytest.approx(0.0)
    assert trace["pwm"].between(0, 2 ** PWM_BITS - 1).all()
    ramp = trace[(trace["t_s"] >= 1.0) & (trace["t_s"] <= 2.8)]["level_pct"]
    assert ramp.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Figure and HTML rendering
# ---------------------------------------------------------------------------
def test_actuator_figure_has_four_panels(day_run):
    import matplotlib.pyplot as plt

    _, (log, _, _) = day_run
    figure = actuator.build_actuator_figure(log)
    assert len([ax for ax in figure.axes if ax.get_ylabel()]) >= 4
    plt.close(figure)


def test_bulb_html_embeds_correct_fade_and_values():
    html = bulb_html(50, "50%", from_pct=0)
    assert html.startswith("<!DOCTYPE html>")
    assert "from=0.0,to=50.0,fade=1.0" in html
    assert "128 / 255" in html and "30 W" in html
    assert "__" not in html                              # no unreplaced placeholders
    steady = bulb_html(0, "OFF")
    assert "from=0.0,to=0.0,fade=0.0" in steady and "OFF" in steady


def test_day_player_html_is_self_consistent(day_run):
    _, (log, _, summary) = day_run
    html = day_player_html(log)
    assert html.count("</script>") == 1                  # data cannot break out of the script
    assert "__" not in re.sub(r"__lightingPlayer", "", html)  # no unreplaced placeholders
    raw = re.search(r"var D=(\[.*?\]),\s*N=D\.length", html, re.S).group(1)
    frames = json.loads(raw)
    assert len(frames) == 288
    assert frames[0]["t"] == "00:00" and frames[-1]["t"] == "23:55"
    assert {"t", "amb", "occ", "dl", "cmd", "pct", "pwm", "p", "wh", "ev"} <= set(frames[0])
    assert frames[-1]["wh"] == pytest.approx(summary["energy_wh"], rel=1e-6)
    assert sum(frame["ev"] for frame in frames) == summary["commands_sent"]
    start = int(re.search(r"idx=(\d+)", html).group(1))
    assert frames[start]["pct"] > 0 and all(f["pct"] == 0 for f in frames[:start])


def test_day_player_rejects_empty_log():
    with pytest.raises(ValueError):
        day_player_html(pd.DataFrame(columns=["day"]))
