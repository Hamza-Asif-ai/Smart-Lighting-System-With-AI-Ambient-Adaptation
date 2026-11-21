"""Energy consumption and cost-savings analysis for the lighting simulation.

A chronological 30-day sensor stream (5-minute steps) is passed through the
trained model and the resulting power draw is compared with conventional
strategies. Because every step is exactly ``STEP_MINUTES`` long, energy is
computed physically:  Wh = Watts x (STEP_MINUTES / 60).

Strategies
----------
* Always-on        : bulb at 100 % around the clock (worst case, no control).
* Occupancy-only   : 100 % while the room is occupied, off otherwise
                     (a plain motion-sensor switch - the CONSERVATIVE baseline).
* Rule (reference) : the hand-written ambient/occupancy rule.
* AI (model)       : brightness chosen by the trained classifier.
"""

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config import (  # noqa: E402
    BULB_FULL_WATT,
    DAYS_PER_MONTH,
    ENERGY_BY_HOUR_PNG,
    ENERGY_COMPARISON_PNG,
    ENERGY_REPORT_TXT,
    PRICE_PER_KWH_PKR,
    SAVING_PIE_PNG,
    SIM_DAYS,
    SIM_SEED,
    STEP_MINUTES,
    WATT_MAP,
    create_directories,
)
from inference import simulate_day_decisions  # noqa: E402

STRATEGY_LABELS = {
    "always_on": "Always-on (100%)",
    "occupancy_only": "Occupancy-only (100% when occupied)",
    "rule": "Rule-based (reference)",
    "ai": "AI model",
}


def strategy_watts(stream):
    """Return a DataFrame of instantaneous power (W) per step for each strategy."""
    occupied = stream["occupancy"].astype(int)
    return pd.DataFrame(
        {
            "always_on": BULB_FULL_WATT,
            "occupancy_only": occupied * BULB_FULL_WATT,
            "rule": stream["ideal_brightness"].map(WATT_MAP),
            "ai": stream["predicted_brightness"].map(WATT_MAP),
        },
        index=stream.index,
    )


def percent_saving(baseline, value):
    """Percentage reduction of ``value`` versus ``baseline`` (0 if baseline is 0)."""
    if baseline == 0:
        return 0.0
    return float((baseline - value) / baseline * 100)


def compute_energy(stream, step_minutes=STEP_MINUTES, tariff=PRICE_PER_KWH_PKR):
    """Compute energy, cost and savings for every strategy.

    ``stream`` needs ``occupancy``, ``ideal_brightness`` and
    ``predicted_brightness`` columns, one row per ``step_minutes``.
    """
    watts = strategy_watts(stream)
    step_hours = step_minutes / 60
    total_days = len(stream) * step_minutes / (24 * 60)
    to_month = DAYS_PER_MONTH / total_days  # 1.0 for a 30-day stream

    kwh = watts.sum() * step_hours / 1000  # W x h / 1000 = kWh over the stream
    monthly_kwh = kwh * to_month

    base = monthly_kwh["occupancy_only"]
    peak = monthly_kwh["always_on"]
    ai = monthly_kwh["ai"]

    occupied = stream["occupancy"].astype(int) == 1
    occupied_off = int((occupied & (stream["predicted_brightness"] == "OFF")).sum())

    return {
        "steps": len(stream),
        "days": total_days,
        "step_minutes": step_minutes,
        "tariff": tariff,
        "monthly_kwh": {key: float(value) for key, value in monthly_kwh.items()},
        "monthly_cost_pkr": {key: float(value * tariff) for key, value in monthly_kwh.items()},
        "saving_vs_occupancy_pct": percent_saving(base, ai),
        "saving_vs_always_on_pct": percent_saving(peak, ai),
        "saving_vs_occupancy_kwh": float(base - ai),
        "saving_vs_occupancy_pkr": float((base - ai) * tariff),
        "saving_vs_always_on_pkr": float((peak - ai) * tariff),
        "rule_saving_vs_occupancy_pct": percent_saving(base, monthly_kwh["rule"]),
        "ai_rule_agreement_pct": float(
            (stream["predicted_brightness"] == stream["ideal_brightness"]).mean() * 100
        ),
        "occupied_steps": int(occupied.sum()),
        "occupied_but_off_steps": occupied_off,
        "hourly_avg_watts": watts.groupby(stream["hour"].to_numpy()).mean(),
    }


def build_summary_frame(results):
    """Return a tidy strategy comparison table for printing and reporting."""
    rows = []
    for key, label in STRATEGY_LABELS.items():
        rows.append(
            {
                "Strategy": label,
                "Monthly energy (kWh)": f"{results['monthly_kwh'][key]:,.2f}",
                "Monthly cost (PKR)": f"{results['monthly_cost_pkr'][key]:,.0f}",
            }
        )
    return pd.DataFrame(rows)


def write_report(results, summary):
    """Write a human-readable energy report to the outputs directory."""
    lines = [
        "=" * 68,
        "SMART LIGHTING - ENERGY ANALYSIS REPORT",
        "=" * 68,
        "",
        "Method:",
        f"  - Chronological simulation: {results['days']:.0f} days x "
        f"{24 * 60 // results['step_minutes']} steps/day ({results['steps']:,} steps, "
        f"{results['step_minutes']} min each).",
        "  - Energy per step = Watts x step_minutes / 60 (Wh); summed and /1000 -> kWh.",
        f"  - One {BULB_FULL_WATT} W-equivalent LED bulb; power assumed linear in brightness "
        "(20%=12 W, 50%=30 W, 80%=48 W, 90%=54 W).",
        f"  - Tariff ASSUMPTION: PKR {results['tariff']} per kWh (replace with your bill).",
        "",
        summary.to_string(index=False),
        "",
        "Savings of the AI model:",
        f"  vs occupancy-only baseline : {results['saving_vs_occupancy_pct']:.2f}%  "
        f"({results['saving_vs_occupancy_kwh']:.2f} kWh, PKR {results['saving_vs_occupancy_pkr']:,.0f} per month)",
        f"  vs always-on bulb          : {results['saving_vs_always_on_pct']:.2f}%  "
        f"(PKR {results['saving_vs_always_on_pkr']:,.0f} per month)",
        f"  (the ideal rule would save {results['rule_saving_vs_occupancy_pct']:.2f}% "
        "vs occupancy-only)",
        "",
        "Reliability / comfort check:",
        f"  AI decision matches the control rule on {results['ai_rule_agreement_pct']:.2f}% of steps.",
        f"  Occupied steps switched OFF by the AI: {results['occupied_but_off_steps']} "
        f"of {results['occupied_steps']:,} occupied steps.",
        "",
        "Note: figures are for ONE bulb in a simulated room. Multiply by the number",
        "of fixtures for a whole building; real savings depend on real usage data.",
        "=" * 68,
    ]
    ENERGY_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_energy_comparison(results):
    """Bar chart of monthly energy for each strategy."""
    keys = ["always_on", "occupancy_only", "ai"]
    comparison = pd.DataFrame(
        {
            "Strategy": ["Always-on", "Occupancy-only", "AI model"],
            "Energy (kWh / month)": [results["monthly_kwh"][key] for key in keys],
        }
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=comparison, x="Strategy", y="Energy (kWh / month)",
                     palette=["#e07a5f", "#f2cc8f", "#81b29a"], hue="Strategy", legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)
    plt.title("Monthly Energy per Bulb: Strategy Comparison")
    plt.xlabel("Lighting strategy")
    plt.ylabel("Energy (kWh per 30 days)")
    plt.tight_layout()
    plt.savefig(ENERGY_COMPARISON_PNG, dpi=150)
    plt.close()


def plot_saving_pie(results):
    """Pie chart of energy saved versus consumed relative to occupancy-only."""
    saved = results["saving_vs_occupancy_pct"]
    plt.figure(figsize=(7, 6))
    plt.pie(
        [saved, 100 - saved],
        labels=[f"Energy saved ({saved:.1f}%)", f"Energy used ({100 - saved:.1f}%)"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#81b29a", "#e07a5f"],
        wedgeprops={"edgecolor": "white"},
    )
    plt.title("AI Energy Saving vs Occupancy-only Baseline")
    plt.tight_layout()
    plt.savefig(SAVING_PIE_PNG, dpi=150)
    plt.close()


def plot_energy_by_hour(results):
    """Average power draw per hour of day for baseline versus AI control."""
    hourly = results["hourly_avg_watts"].reindex(range(24))
    plt.figure(figsize=(10, 5))
    plt.plot(hourly.index, hourly["occupancy_only"], marker="o", color="#e07a5f",
             label="Occupancy-only")
    plt.plot(hourly.index, hourly["ai"], marker="o", color="#2a9d8f", label="AI model")
    plt.fill_between(hourly.index, hourly["ai"], hourly["occupancy_only"],
                     color="#81b29a", alpha=0.25, label="Energy saved")
    plt.title("Average Power Draw by Hour of Day (30-day simulation)")
    plt.xlabel("Hour of day")
    plt.ylabel("Average power (W)")
    plt.xticks(range(24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(ENERGY_BY_HOUR_PNG, dpi=150)
    plt.close()


def main():
    """Run the energy analysis and export the report and charts."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    create_directories()
    sns.set_theme(style="whitegrid")

    stream = simulate_day_decisions(n_days=SIM_DAYS, seed=SIM_SEED)
    results = compute_energy(stream)
    summary = build_summary_frame(results)

    write_report(results, summary)
    plot_energy_comparison(results)
    plot_saving_pie(results)
    plot_energy_by_hour(results)

    print("=" * 68)
    print("ENERGY ANALYSIS SUMMARY (30-day simulation, one bulb)")
    print("=" * 68)
    print(summary.to_string(index=False))
    print("-" * 68)
    print(f"AI saving vs occupancy-only : {results['saving_vs_occupancy_pct']:.2f}% "
          f"(PKR {results['saving_vs_occupancy_pkr']:,.0f}/month)")
    print(f"AI saving vs always-on      : {results['saving_vs_always_on_pct']:.2f}% "
          f"(PKR {results['saving_vs_always_on_pkr']:,.0f}/month)")
    print(f"Occupied steps switched OFF : {results['occupied_but_off_steps']} "
          f"of {results['occupied_steps']:,}")
    print(f"Report saved to: {ENERGY_REPORT_TXT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
