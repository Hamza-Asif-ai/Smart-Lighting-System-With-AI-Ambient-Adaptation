"""Streamlit live dashboard for the AI ambient-adaptive lighting simulator."""

import io

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from actuator import (
    build_actuator_figure,
    bulb_html,
    command_message,
    day_player_html,
    level_to_pct,
    pwm_duty,
    simulate_actuator,
)
from config import (
    AMBIENT_LEVELS,
    BULB_FULL_WATT,
    COMPARISON_CSV,
    CONFUSION_PNG,
    FADE_RATE_PCT_PER_S,
    FEATURE_IMPORTANCE_PNG,
    PRICE_PER_KWH_PKR,
    PWM_BITS,
    RULES_TXT,
    SIM_SEED,
    WATT_MAP,
    create_directories,
)
from energy_analysis import compute_energy
from inference import generate_simulation, load_artifacts, predict_action, simulate_day_decisions
from sensor_simulator import rule_brightness
from visualize import build_day_figure

matplotlib.use("Agg")

st.set_page_config(
    page_title="Smart Lighting AI - Live Simulator",
    page_icon="💡",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    """Load (or rebuild) the trained model once per Streamlit server."""
    return load_artifacts()


@st.cache_data(show_spinner=False)
def cached_day(seed):
    """Run one simulated day for a seed (cached so widgets stay responsive)."""
    return simulate_day_decisions(n_days=1, seed=int(seed))


@st.cache_data(show_spinner=False)
def cached_actuator_day(seed):
    """Drive the simulated bulb through one day of AI decisions (cached by seed)."""
    return simulate_actuator(cached_day(int(seed)))


@st.cache_data(show_spinner=False)
def cached_figure_png(kind, seed):
    """Render a matplotlib figure once per seed and cache it as PNG bytes.

    Rebuilding multi-panel figures on every widget change made the dashboard
    sluggish, even while the figure was hidden in a collapsed expander.
    """
    if kind == "actuator":
        figure = build_actuator_figure(cached_actuator_day(seed)[0])
    else:
        figure = build_day_figure(cached_day(seed))
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()


def show_html(html, height):
    """Embed trusted, self-generated HTML on both old and new Streamlit versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # Streamlit < 1.5x
        import streamlit.components.v1 as components

        components.html(html, height=height)


def watt_saving(baseline_watt, ai_watt):
    """Return the percentage reduction relative to the baseline."""
    if baseline_watt == 0:
        return 0.0
    return (baseline_watt - ai_watt) / baseline_watt * 100


def sidebar_inputs():
    """Render the sensor controls and return the chosen reading."""
    st.sidebar.title("Sensor Inputs")
    st.sidebar.caption("Simulated IoT readings (used by the Live Prediction tab)")
    ambient = st.sidebar.selectbox("Ambient Light", AMBIENT_LEVELS, index=1)
    occupancy = st.sidebar.radio(
        "Occupancy", [0, 1], index=1, horizontal=True,
        format_func=lambda value: "Occupied" if value else "Empty",
    )
    hour = st.sidebar.slider("Hour of day", 0, 23, 22)
    st.sidebar.divider()
    st.sidebar.caption("Used by the Virtual Bulb replay and Day Simulation tabs")
    seed = st.sidebar.number_input("Simulated day (random seed)", min_value=0,
                                   max_value=10_000, value=SIM_SEED, step=1)
    return ambient, occupancy, hour, int(seed)


def render_live_tab(ambient, occupancy, hour):
    """Show the model decision for the sensor values chosen in the sidebar."""
    st.header("Live Prediction")
    st.write("Change the simulated sensor readings in the sidebar - the trained "
             "model chooses the brightness instantly.")

    prediction = predict_action(ambient, occupancy, hour)
    rule = rule_brightness(ambient, occupancy, hour)
    baseline_watt = BULB_FULL_WATT if occupancy == 1 else 0
    ai_watt = WATT_MAP[prediction]
    saving = watt_saving(baseline_watt, ai_watt)

    top1, top2 = st.columns(2)
    top1.metric("AI Predicted Brightness", prediction)
    top2.metric("Control-rule Brightness", rule,
                delta="matches AI" if rule == prediction else "differs from AI",
                delta_color="off" if rule == prediction else "inverse")

    col1, col2, col3 = st.columns(3)
    col1.metric("Occupancy-only Power", f"{baseline_watt} W")
    col2.metric("AI Power", f"{ai_watt} W", delta=f"-{baseline_watt - ai_watt} W",
                delta_color="inverse")
    col3.metric("Saving vs Occupancy-only", f"{saving:.1f}%")

    st.bar_chart(pd.DataFrame(
        {"Power (W)": {"Occupancy-only": baseline_watt, "AI-controlled": ai_watt}}
    ))
    st.caption(f"Input: ambient={ambient}, occupancy={occupancy}, hour={hour:02d}:00 | "
               f"Tariff assumption: PKR {PRICE_PER_KWH_PKR}/kWh")


def render_day_tab(seed):
    """Simulate a full chronological day and show decisions plus energy."""
    st.header("Full-Day Simulation")
    st.write("A chronological day in 5-minute steps: daylight rises and falls, "
             "clouds drift, and people come and go. Each step is decided by the model.")
    st.caption(f"Simulated day: seed {seed} (change it in the sidebar).")
    day = cached_day(seed)

    st.image(cached_figure_png("day", seed))

    results = compute_energy(day)
    kwh = results["monthly_kwh"]
    daily = {key: value / 30 for key, value in kwh.items()}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Always-on", f"{daily['always_on']:.2f} kWh")
    col2.metric("Occupancy-only", f"{daily['occupancy_only']:.2f} kWh")
    col3.metric("AI model", f"{daily['ai']:.2f} kWh")
    col4.metric("Saving vs occupancy-only", f"{results['saving_vs_occupancy_pct']:.1f}%")
    st.caption("Energy for this single day, one bulb. Occupied steps switched OFF by "
               f"the AI: {results['occupied_but_off_steps']} of {results['occupied_steps']}.")

    with st.expander("Random independent readings (batch mode)"):
        count = st.slider("Number of readings", 10, 500, 100, step=10)
        log = generate_simulation(n_steps=count, seed=int(seed))
        st.dataframe(log, height=300)
        agreement = (log["predicted_brightness"] == log["ideal_brightness"]).mean()
        st.metric("Agreement with control rule", f"{agreement:.1%}")


def render_actuator_tab(ambient, occupancy, hour, seed):
    """Show the simulated LED driver: command message, PWM, soft fade and a day replay."""
    st.header("Virtual Bulb - Actuator Simulation")
    st.write("The AI only chooses a brightness label. The actuator turns it into what a "
             "real fixture does: an MQTT-style command, an 8-bit PWM duty value for the LED "
             f"driver and a soft fade ({FADE_RATE_PCT_PER_S:.0f} %/s) instead of a hard jump.")

    command = predict_action(ambient, occupancy, hour)
    target = level_to_pct(command)
    previous = float(st.session_state.get("bulb_level_pct", 0.0))
    message = command_message(command, previous)

    st.subheader("Live actuator (follows the sidebar sensors)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AI command", command)
    col2.metric("PWM duty", f"{pwm_duty(target)} / {2 ** PWM_BITS - 1}")
    col3.metric("Bulb power", f"{BULB_FULL_WATT * target / 100:.0f} W")
    col4.metric("Fade time", f"{abs(target - previous) / FADE_RATE_PCT_PER_S:.1f} s")
    show_html(bulb_html(target, command, from_pct=previous), 290)
    st.caption("Simulated command published to the light controller (no real broker):")
    st.code(f"topic   : {message['topic']}\npayload : {message['payload']}", language="text")
    st.session_state["bulb_level_pct"] = target

    log, events, summary = cached_actuator_day(seed)
    st.subheader("Full-day replay")
    st.write("Press Play to watch the AI and the bulb through a whole simulated day, or "
             "drag the slider to any time.")
    show_html(day_player_html(log), 480)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Energy that day", f"{summary['energy_wh']:.0f} Wh")
    col2.metric("Commands sent", f"{summary['commands_sent']}")
    col3.metric("Turn-on events", f"{summary['turn_on_events']}")
    col4.metric("0-100 % fade", f"{summary['full_range_fade_s']:.1f} s")
    st.caption(f"The soft fade changes the energy by only {summary['fade_effect_pct']:+.3f} % "
               "compared with an instant-switch model, so the energy analysis stays valid.")

    with st.expander("Actuator response figure (command, PWM, power, fade close-up)"):
        st.image(cached_figure_png("actuator", seed))
    with st.expander(f"Command log - {len(events)} simulated MQTT messages"):
        st.dataframe(events, height=300)


def render_model_tab():
    """Show the training results, the confusion matrix and the learned rules."""
    st.header("Model Performance")
    _, meta = get_artifacts()
    metrics = meta["metrics"]
    st.write(f"**Deployed model:** {meta['model_name']} "
             f"({metrics['size_kb']:.1f} KB) - trained on {meta['n_train']} samples, "
             f"evaluated on {meta['n_test']} held-out samples.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cross-validated F1", f"{metrics['cv_f1_mean']:.3f}")
    col2.metric("Held-out accuracy", f"{metrics['test_accuracy']:.1%}",
                help=f"Labels contain {meta['label_noise_ratio']:.0%} noise, so the best "
                     f"possible score is about {meta['noise_ceiling_accuracy']:.0%}.")
    col3.metric("Rule coverage (all 144 states)", f"{metrics['rule_coverage']:.1%}")

    if COMPARISON_CSV.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(COMPARISON_CSV))
    left, right = st.columns(2)
    if CONFUSION_PNG.exists():
        left.image(str(CONFUSION_PNG), caption="Confusion matrix (held-out test set)")
    if FEATURE_IMPORTANCE_PNG.exists():
        right.image(str(FEATURE_IMPORTANCE_PNG), caption="Feature importance")
    if RULES_TXT.exists():
        with st.expander("Learned decision-tree rules (interpretable)"):
            st.code(RULES_TXT.read_text(encoding="utf-8"))


def render_about_tab():
    """Render the project background and methodology."""
    st.header("About This Project")
    st.markdown(
        f"""
        **A8 - Smart Lighting System with AI Ambient Adaptation** is a fully
        simulated AIoT project. No hardware is needed: synthetic sensors produce
        ambient-light, occupancy and time-of-day readings, and a trained
        classifier selects an energy-efficient brightness level in real time.

        ### Methodology
        1. Generate 3,000 reproducible sensor samples with realistic hourly
           distributions and 7% label noise.
        2. Compare Decision Tree, k-NN and Random Forest with 5-fold
           cross-validation on a stratified 80/20 split; pick the best CV score
           (smallest model wins ties).
        3. Verify the winner against the control rule on all 144 possible
           sensor states.
        4. Run it on a chronological multi-day simulation to estimate energy.
        5. Drive a simulated dimmable LED driver (PWM, soft fade, MQTT-style
           commands) with the decisions - the actuator end of the AIoT loop.

        ### Control rule (ground truth)
        Unoccupied -> OFF; bright ambient -> 20%; medium ambient -> 50%;
        dark ambient at night -> 90%; dark ambient in daytime -> 80%.

        ### Energy model
        One {BULB_FULL_WATT} W-equivalent bulb, power linear in brightness,
        tariff PKR {PRICE_PER_KWH_PKR}/kWh (an assumption - use your own bill).
        The conservative baseline is a motion-sensor switch (100% while occupied).
        """
    )


def main():
    """Configure the page and render all dashboard sections."""
    create_directories()
    st.title("💡 Smart Lighting AI - Live Simulator")
    try:
        get_artifacts()
    except Exception as error:  # surface any rebuild failure in the UI
        st.error(f"Could not load or build the model: {error}")
        st.stop()

    ambient, occupancy, hour, seed = sidebar_inputs()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Live Prediction", "Virtual Bulb", "Day Simulation", "Model Performance", "About"]
    )
    with tab1:
        render_live_tab(ambient, occupancy, hour)
    with tab2:
        render_actuator_tab(ambient, occupancy, hour, seed)
    with tab3:
        render_day_tab(seed)
    with tab4:
        render_model_tab()
    with tab5:
        render_about_tab()


main()
