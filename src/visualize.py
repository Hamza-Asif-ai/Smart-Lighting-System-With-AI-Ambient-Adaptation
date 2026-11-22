"""Exploratory and decision visualisations for the smart lighting project.

Generates publication-style PNG figures in the outputs directory using a clean
seaborn theme. Every figure has a title, axis labels, tight layout and 150 dpi.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Make `python src/<script>.py` work as well as `python -m src.<script>`.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from src.config import (  # noqa: E402
    AMBIENT_LEVELS,
    AMBIENT_VS_BRIGHTNESS_PNG,
    BRIGHTNESS_DIST_PNG,
    BRIGHTNESS_LEVELS,
    DATA_PATH,
    DECISION_TIMELINE_PNG,
    HOURLY_USAGE_PNG,
    OCCUPANCY_BREAKDOWN_PNG,
    SIM_SEED,
    create_directories,
)
from src.inference import simulate_day_decisions  # noqa: E402
from src.train_model import load_dataset  # noqa: E402

PALETTE = "viridis"


def plot_brightness_distribution(data):
    """Bar chart showing how often each brightness level occurs in the data."""
    counts = data["brightness"].value_counts().reindex(BRIGHTNESS_LEVELS, fill_value=0)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=PALETTE, hue=counts.index, legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    plt.title("Brightness Label Distribution")
    plt.xlabel("Brightness level")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(BRIGHTNESS_DIST_PNG, dpi=150)
    plt.close()


def plot_ambient_vs_brightness(data):
    """Stacked bar chart of brightness choices for every ambient level."""
    table = pd.crosstab(data["ambient"], data["brightness"])
    table = table.reindex(index=AMBIENT_LEVELS, columns=BRIGHTNESS_LEVELS, fill_value=0)
    ax = table.plot(kind="bar", stacked=True, figsize=(9, 5.5), colormap=PALETTE)
    ax.set_title("Ambient Light vs Selected Brightness")
    ax.set_xlabel("Ambient light level")
    ax.set_ylabel("Number of samples")
    ax.legend(title="Brightness", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(AMBIENT_VS_BRIGHTNESS_PNG, dpi=150)
    plt.close()


def plot_hourly_usage(data):
    """Line plot of ON events aggregated across the 24 hours of the day."""
    on_events = data.loc[data["brightness"] != "OFF", "hour"].value_counts()
    on_events = on_events.reindex(range(24), fill_value=0).sort_index()
    plt.figure(figsize=(10, 5))
    plt.plot(on_events.index, on_events.values, marker="o", color="#2a6f97", label="ON events")
    plt.fill_between(on_events.index, on_events.values, alpha=0.18, color="#2a6f97")
    plt.title("Hourly Lighting Usage (ON Events)")
    plt.xlabel("Hour of day")
    plt.ylabel("Count of ON events")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(HOURLY_USAGE_PNG, dpi=150)
    plt.close()


def build_day_figure(day):
    """Build a one-day timeline figure from a chronological decision stream.

    Shows the daylight index (right axis), occupied periods (shaded) and the
    brightness chosen by the model over the course of the day.
    """
    levels = day["predicted_brightness"].map(BRIGHTNESS_LEVELS.index)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.step(day["time_h"], levels, where="post", color="#264653", linewidth=2,
            label="AI brightness decision")
    ax.fill_between(day["time_h"], 0, 1, where=day["occupancy"].to_numpy() == 1,
                    transform=ax.get_xaxis_transform(), color="#e9c46a", alpha=0.25,
                    step="post", label="Room occupied")
    ax.set_yticks(range(len(BRIGHTNESS_LEVELS)))
    ax.set_yticklabels(BRIGHTNESS_LEVELS)
    ax.set_ylim(-0.3, len(BRIGHTNESS_LEVELS) - 0.7)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Time of day (hour)")
    ax.set_ylabel("Brightness chosen by AI")

    twin = ax.twinx()
    twin.plot(day["time_h"], day["daylight_index"], color="#f4a261", linewidth=1.6,
              alpha=0.9, label="Daylight index")
    twin.set_ylim(0, 1.05)
    twin.set_ylabel("Daylight index (0 = dark, 1 = bright)")
    twin.grid(False)

    handles = ax.get_legend_handles_labels()
    handles_twin = twin.get_legend_handles_labels()
    ax.legend(handles[0] + handles_twin[0], handles[1] + handles_twin[1],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title("AI Brightness Decisions Over One Simulated Day")
    fig.tight_layout()
    return fig


def plot_decision_timeline(day):
    """Save the one-day decision timeline figure."""
    fig = build_day_figure(day)
    fig.savefig(DECISION_TIMELINE_PNG, dpi=150)
    plt.close(fig)


def plot_occupancy_breakdown(data):
    """Pie chart of unoccupied versus occupied samples."""
    counts = [int((data["occupancy"] == 0).sum()), int((data["occupancy"] == 1).sum())]
    labels = ["Unoccupied (lights OFF)", "Occupied (lights ON)"]
    plt.figure(figsize=(7, 6))
    plt.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#adb5bd", "#457b9d"],
        wedgeprops={"edgecolor": "white"},
    )
    plt.title("Occupancy Breakdown: Unoccupied vs Occupied")
    plt.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.tight_layout()
    plt.savefig(OCCUPANCY_BREAKDOWN_PNG, dpi=150)
    plt.close()


def main():
    """Load the dataset, run a day simulation and render all figures."""
    create_directories()
    sns.set_theme(style="whitegrid")
    data = load_dataset()
    day = simulate_day_decisions(n_days=1, seed=SIM_SEED)

    plot_brightness_distribution(data)
    plot_ambient_vs_brightness(data)
    plot_hourly_usage(data)
    plot_decision_timeline(day)
    plot_occupancy_breakdown(data)

    print("=" * 68)
    print("VISUALISATION GENERATION")
    print("=" * 68)
    print(f"Source dataset: {DATA_PATH} ({len(data)} rows)")
    print(f"Day simulation: {len(day)} steps (seed {SIM_SEED})")
    print("Generated figures:")
    for path in (
        BRIGHTNESS_DIST_PNG,
        AMBIENT_VS_BRIGHTNESS_PNG,
        HOURLY_USAGE_PNG,
        DECISION_TIMELINE_PNG,
        OCCUPANCY_BREAKDOWN_PNG,
    ):
        print(f"  - {path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
