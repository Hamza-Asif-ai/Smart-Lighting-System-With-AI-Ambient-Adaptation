# A8 – Smart Lighting System with AI Ambient Adaptation

A fully simulated **AIoT** project: synthetic sensors (ambient light, occupancy,
time of day) feed a trained classifier that chooses an energy-efficient bulb
brightness in real time. No hardware or external API is needed – everything runs
in pure Python with fixed seeds, so every number below is reproducible.

**Headline results** (30-day simulation, one 60 W-equivalent bulb)

| | |
|---|---|
| Deployed model | Decision Tree, **6.4 KB**, CV F1 **0.930**, held-out accuracy **92.3 %** (ceiling with 7 % label noise: 93 %) |
| Correctness | Matches the control rule on **144 / 144** possible sensor states; **0** occupied time-steps switched OFF |
| Energy saving | **41.8 %** vs a motion-sensor switch · **66.8 %** vs an always-on bulb |
| Actuator | Simulated 8-bit PWM LED driver, 50 %/s soft fade, MQTT-style commands; fading changes energy by only +0.002 % |
| Cost (assumed tariff) | PKR 358 / month vs PKR 615 (occupancy-only) vs PKR 1,080 (always-on) |

---

## Problem Statement

Conventional lighting runs at full power whenever a room is occupied (or all the
time), wasting energy when daylight already provides enough illumination. A smart
system must adapt brightness to ambient conditions and occupancy without human
input. This project models that decision as a supervised classification problem
and measures the resulting energy and cost savings.

## Architecture

```
 +-------------------+     +--------------------+     +---------------------+
 |  Sensor Simulator |     |  Training Pipeline |     |   Inference Engine  |
 |  hour/ambient/occ | --> |  3 models, 5-fold  | --> |  predict_action()   |
 |  + 30-day stream  |     |  CV, 144-state     |     |  vectorised batch   |
 +---------+---------+     |  verification      |     |  auto-recovery      |
           |               +---------+----------+     +----------+----------+
           v                         v                           |
   data/lighting_data.csv   models/best_model.pkl                v
                            models/model_meta.json   +----------------------+
                            outputs/decision_tree_   |  ACTUATOR            |
                            rules.txt                |  MQTT-style command  |
                                                     |  -> 8-bit PWM driver |
                                                     |  -> soft fade -> bulb|
                                                     +----------+-----------+
                                                                |
                                         +----------------------+-----------+
                                         | Energy analysis | Visualisations |
                                         | Streamlit dashboard (Virtual Bulb)|
                                         +----------------------------------+
```

## Folder Structure

```
Smart-Lighting-System-With-AI-Ambient-Adaptation/
├── README.md
├── requirements.txt
├── run_all.py                # one-command pipeline  (--live, --test)
├── src/
│   ├── config.py             # all constants and paths (single source of truth)
│   ├── features.py           # feature/target encoding + input validation
│   ├── sensor_simulator.py   # training-data generator + chronological day simulator
│   ├── train_model.py        # CV, model selection, exhaustive rule verification
│   ├── inference.py          # predict_action(), batch prediction, auto-recovery
│   ├── visualize.py          # exploratory + decision plots
│   ├── energy_analysis.py    # physically consistent energy / cost model
│   └── actuator.py           # simulated LED driver: PWM, soft fade, MQTT-style commands, virtual bulb
├── dashboard/
│   └── app.py                # Streamlit dashboard (5 tabs incl. Virtual Bulb)
├── tests/                    # 97 pytest checks (pipeline, actuator, dashboard, repo layout)
├── data/lighting_data.csv
├── models/                   # best_model.pkl, model_meta.json
└── outputs/                  # figures, CSVs, energy report, learned rules
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## How to Run

```bash
python run_all.py            # data -> train -> inference -> plots -> energy -> actuator
python run_all.py --test     # ...and run the test suite afterwards
python run_all.py --live     # slow the inference stage down to feel like a live feed
python src/actuator.py       # actuator only: one simulated day (--seed N for another)
streamlit run dashboard/app.py   # interactive dashboard with the Virtual Bulb tab
python -m pytest -q          # tests only
```

Every script in `src/` also runs standalone, either as `python src/train_model.py`
or as a module from the repository root (`python -m src.train_model`). If the model
files are missing, corrupt, or were created with a different scikit-learn
version, the dashboard and `inference.py` **retrain automatically** instead of
crashing.

## Methodology

1. **Training data** – 3,000 readings; hours, ambient levels and occupancy are
   sampled from time-of-day distributions. A deterministic rule assigns the ideal
   brightness, then 7 % of labels are flipped to a neighbouring level to mimic
   measurement noise. Both the noisy label (`brightness`) and the clean one
   (`ideal_brightness`) are stored.
2. **Features** – `ambient_level` (ordinal: Low=0, Medium=1, High=2),
   `occupancy` (0/1) and `hour` (0–23).
3. **Model comparison** – Decision Tree (depth 5, `min_samples_leaf=10`),
   k-NN (scaled, k=11) and Random Forest (100 trees). Stratified 80/20 split;
   5-fold CV runs on the training part only. The model is chosen by CV F1 – the
   test set is never used for selection – and near-ties go to the smallest model.
4. **Exhaustive verification** – the sensor space is finite (3 × 2 × 24 = 144
   states), so each model is checked against the control rule on *every* state
   ("rule coverage").
5. **Day simulation** – a chronological stream in 5-minute steps: sine-shaped
   daylight scaled by a per-day cloudiness plus drifting cloud noise, and
   occupancy as a Markov chain (people stay for a realistic stretch instead of
   flickering every sample).
6. **Energy analysis** – the model drives one bulb through 30 simulated days
   (8,640 steps). Energy = Watts × (5 / 60) h, summed and divided by 1,000.

7. **Actuator** – the AI decision is turned into what a real fixture does
   (`src/actuator.py`): an MQTT-style command (`topic` + JSON payload), an 8-bit PWM
   duty value for the LED driver, and a **soft fade** limited to 50 %/s (0 → 100 %
   in 2 s) instead of a hard jump. `DimmableBulb` integrates power over the ramp
   exactly (trapezoid rule), counts commands and turn-on events, and ignores
   redundant commands like a real controller would.

## Results

### Model comparison (`outputs/model_comparison.csv`)

| Model | CV F1 | Test acc. | Clean-label acc. | Rule coverage (144 states) | Size |
|---|---|---|---|---|---|
| **DecisionTree** (deployed) | **0.930** ± 0.009 | 92.3 % | 100 % | **100 %** | **6.4 KB** |
| KNeighbors | 0.928 ± 0.009 | 92.3 % | 100 % | 99.3 % | 162 KB |
| RandomForest | 0.928 ± 0.009 | 92.2 % | 99.8 % | 95.8 % | 1,925 KB |

* Because 7 % of labels are deliberately wrong, **no model can exceed ≈ 93 %**
  accuracy against the noisy labels; 92–93 % is therefore essentially perfect.
  The "clean-label accuracy" column scores the same predictions against the
  noise-free rule and shows what the models actually learned.
* The three models are statistically tied on accuracy, but the tree is 300×
  smaller than the forest **and** the only one that is exactly right on every
  state – the forest misreads 6 rare states because it memorises noisy labels.
  For an IoT / microcontroller target the tree is the clear choice, and its
  rules are human-readable (`outputs/decision_tree_rules.txt`).

### Energy (`outputs/energy_report.txt`)

| Strategy | kWh / month | PKR / month |
|---|---|---|
| Always-on (100 %) | 43.20 | 1,080 |
| Occupancy-only (100 % when occupied) | 24.60 | 615 |
| Rule-based (reference) | 14.33 | 358 |
| **AI model** | **14.33** | **358** |

Saving of the AI model: **41.75 %** vs occupancy-only (PKR 257 / month) and
**66.83 %** vs always-on (PKR 722 / month). The occupancy-only baseline is the
conservative comparison; "always-on" is the no-control worst case.

### Actuator (`outputs/actuator_log.csv`, `actuator_events.csv`, `actuator_response.png`)

One simulated day (seed 43), one bulb:

| | |
|---|---|
| Commands sent | 27 MQTT-style messages (12 turn-on events) |
| Energy | 245.0 Wh – only **+0.002 %** different from the instant-switch model |
| Safety | bulb dark whenever the room is empty, lit whenever occupied |

The fade lasts seconds while a simulation step is 5 minutes, so the bulb always
reaches the commanded level inside the step and the energy analysis above stays
valid. `actuator_response.png` shows command vs bulb level, PWM duty, power and
cumulative energy over the day, plus a 50 ms-resolution close-up of the fade.

**Virtual Bulb tab (dashboard):** a live bulb that fades from its previous level
to the new one when you change the sensors, with the PWM value, power, fade time
and the exact command message; plus a full-day player (Play / Pause / slider /
speed) that replays the day with the bulb, sensors, command badge, energy counter
and a timeline. The player runs entirely client-side, so playback is smooth.

### Figures (`outputs/`)

`model_comparison.png`, `confusion_matrix.png`, `feature_importance.png`,
`decision_timeline.png` (one full simulated day), `energy_comparison.png`,
`energy_by_hour.png`, `saving_pie.png`, `brightness_distribution.png`,
`ambient_vs_brightness.png`, `hourly_usage.png`, `occupancy_breakdown.png`,
`actuator_response.png`.

## Testing

`python -m pytest -q` runs 97 checks (about 30 s), including: the control-rule truth table,
dataset reproducibility and exact noise rate, plausibility of the day simulator,
input validation, **model correctness on all 144 states**, safety (never OFF
when occupied), energy-unit regression tests (a 60 W bulb can never use more
than 43.2 kWh in 30 days), and automatic recovery from missing/incompatible
model files. The actuator tests check the physics against hand-calculated values
(e.g. a 0 → 90 % fade draws exactly 27 W on average for 1.8 s), the PWM mapping,
command counters, the MQTT payloads, and that the actuator's energy matches the
energy model. The dashboard tests drive the real widgets headlessly (Streamlit
`AppTest`), including all 30 sensor combinations, and re-compute every number
shown on the About tab. The layout tests check that each script runs directly from
any directory and that every command in this README points to a file that exists.

## Assumptions and Limitations

* **Synthetic data.** Labels come from a hand-written rule, so the model's
  100 % agreement with that rule demonstrates a correct ML workflow, **not**
  real-world performance. The value of the ML approach is that the same pipeline
  can be retrained on logged real sensor data and user overrides, where no
  clean rule exists.
* **Energy figures are estimates** for one bulb: power is assumed linear in
  brightness (20 % = 12 W … 90 % = 54 W) and the tariff of **PKR 25 / kWh is an
  assumption** – replace `PRICE_PER_KWH_PKR` in `config.py` with the rate from
  your own electricity bill. Multiply by the number of fixtures for a building.
* The control rule's occupancy and daylight patterns are plausible but invented;
  there is no sensor drift, hardware failure, or per-person comfort preference.
* The actuator is a model, not a driver: PWM resolution, the 50 %/s fade rate and
  the linear brightness-to-power relation are typical assumptions, and the MQTT
  messages are only formatted, not sent to a broker. Inrush current, thermal
  behaviour and non-linear perceived brightness are not modelled.
* No vacancy time-out or dimming smoothing: a real fixture would delay switching
  OFF for a few minutes and avoid rapid brightness changes when clouds pass.

## Future Work

* Replace the simulator with real telemetry (ESP32 + LDR + PIR/mmWave sensor)
  over MQTT, publish the already-formatted actuator commands to a real broker
  (e.g. Mosquitto), and export the 6 KB tree to C for on-device inference.
* Add vacancy time-out and hysteresis to avoid flicker; measure switching events.
* Learn personal comfort preferences from manual overrides (contextual bandit /
  reinforcement learning).
* Multi-zone / multi-bulb control and a real lux-sensor calibration model.

## Author and Course

Author: **Hamza Asif**  
Roll Number: **23-AI-93**  
Course: AIoT – Artificial Intelligence and Internet of Things  
Assignment: A8 – Smart Lighting System with AI Ambient Adaptation  
Institution: Dawood University of Engineering and Technology (DUET), Karachi
