"""Central configuration for the A8 Smart Lighting AIoT simulation.

All paths are derived from this file's location so that every script works
regardless of the current working directory. No hard-coded absolute paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42          # dataset generation + model training
SIM_SEED = RANDOM_SEED + 1  # independent stream for inference / day simulations
N_SAMPLES = 3000
NOISE_RATIO = 0.07
TEST_SIZE = 0.2
CV_FOLDS = 5
# Models whose CV F1 is within this margin of the best are considered tied;
# the smallest (most edge-friendly) one wins the tie.
SELECTION_TOLERANCE = 0.005

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
BRIGHTNESS_LEVELS = ["OFF", "20%", "50%", "80%", "90%"]  # ordered low -> high
AMBIENT_LEVELS = ["Low", "Medium", "High"]               # ordered dark -> bright

# Ordinal codes (tree / kNN friendly, and human-readable in exported rules).
AMBIENT_TO_CODE = {level: code for code, level in enumerate(AMBIENT_LEVELS)}
BRIGHTNESS_TO_CODE = {level: code for code, level in enumerate(BRIGHTNESS_LEVELS)}

# Electrical power drawn by one 60 W-equivalent LED bulb at each dim level
# (assumes power scales linearly with brightness).
WATT_MAP = {
    "OFF": 0,
    "20%": 12,
    "50%": 30,
    "80%": 48,
    "90%": 54,
    "100%": 60,
}
BULB_FULL_WATT = 60

FEATURE_NAMES = ["ambient_level", "occupancy", "hour"]

# Artefact schema version: bump when the model/meta format changes so that old
# artefacts are rebuilt automatically instead of crashing.
SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Energy report assumptions (ASSUMPTIONS - replace with your own bill / bulb)
# ---------------------------------------------------------------------------
PRICE_PER_KWH_PKR = 25
STEP_MINUTES = 5        # one simulated sensor reading every 5 minutes
DAYS_PER_MONTH = 30
SIM_DAYS = 30           # length of the day-by-day stream used for energy analysis

# ---------------------------------------------------------------------------
# Actuator (simulated dimmable LED driver)
# ---------------------------------------------------------------------------
PWM_BITS = 8                  # 8-bit PWM -> duty cycle 0..255
PWM_FREQUENCY_HZ = 1000       # informational: well above the flicker-visible range
FADE_RATE_PCT_PER_S = 50.0    # soft-start/stop: 0 -> 100 % takes 2 s
MQTT_TOPIC = "smartlighting/room1/light/set"   # simulated command topic (no real broker)

# ---------------------------------------------------------------------------
# Day-simulation physics (normalised daylight index, 0 = dark, 1 = bright)
# ---------------------------------------------------------------------------
SUNRISE_HOUR = 6
SUNSET_HOUR = 18
NIGHT_FLOOR = 0.04          # residual ambient light at night
AMBIENT_LOW_MAX = 0.25      # index <  0.25 -> "Low"
AMBIENT_HIGH_MIN = 0.60     # index >= 0.60 -> "High", otherwise "Medium"
OCCUPANCY_LEAVE_PROB = 0.05  # per step; mean stay ~ 100 min at 5-min steps

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, resolved once)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATA_PATH = DATA_DIR / "lighting_data.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
META_PATH = MODELS_DIR / "model_meta.json"

COMPARISON_CSV = OUTPUTS_DIR / "model_comparison.csv"
COMPARISON_PNG = OUTPUTS_DIR / "model_comparison.png"
CONFUSION_PNG = OUTPUTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PNG = OUTPUTS_DIR / "feature_importance.png"
RULES_TXT = OUTPUTS_DIR / "decision_tree_rules.txt"
SIMULATION_LOG_CSV = OUTPUTS_DIR / "simulation_log.csv"
ENERGY_REPORT_TXT = OUTPUTS_DIR / "energy_report.txt"
ENERGY_COMPARISON_PNG = OUTPUTS_DIR / "energy_comparison.png"
ENERGY_BY_HOUR_PNG = OUTPUTS_DIR / "energy_by_hour.png"
SAVING_PIE_PNG = OUTPUTS_DIR / "saving_pie.png"
BRIGHTNESS_DIST_PNG = OUTPUTS_DIR / "brightness_distribution.png"
AMBIENT_VS_BRIGHTNESS_PNG = OUTPUTS_DIR / "ambient_vs_brightness.png"
HOURLY_USAGE_PNG = OUTPUTS_DIR / "hourly_usage.png"
DECISION_TIMELINE_PNG = OUTPUTS_DIR / "decision_timeline.png"
OCCUPANCY_BREAKDOWN_PNG = OUTPUTS_DIR / "occupancy_breakdown.png"
ACTUATOR_LOG_CSV = OUTPUTS_DIR / "actuator_log.csv"
ACTUATOR_EVENTS_CSV = OUTPUTS_DIR / "actuator_events.csv"
ACTUATOR_RESPONSE_PNG = OUTPUTS_DIR / "actuator_response.png"


def create_directories():
    """Create the data, models and outputs directories if they are missing."""
    for directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
