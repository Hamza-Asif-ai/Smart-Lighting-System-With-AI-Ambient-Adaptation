"""One-command orchestrator for the complete smart lighting pipeline.

Usage:
    python run_all.py            # data -> train -> inference -> plots -> energy -> actuator
    python run_all.py --live     # same, with a 50 ms delay per inference step
    python run_all.py --test     # run the pytest suite after the pipeline
"""

import argparse
import json
import os
import subprocess
import sys

from config import (
    BASE_DIR,
    DATA_PATH,
    ACTUATOR_RESPONSE_PNG,
    ENERGY_REPORT_TXT,
    META_PATH,
    MODEL_PATH,
    OUTPUTS_DIR,
    SIMULATION_LOG_CSV,
)

SCRIPTS = [
    ["sensor_simulator.py"],
    ["train_model.py"],
    ["inference.py"],
    ["visualize.py"],
    ["energy_analysis.py"],
    ["actuator.py"],
]


def run_script(command):
    """Execute one pipeline stage in an isolated Python process."""
    print("\n" + "=" * 68)
    print(f"RUNNING: {' '.join(command)}")
    print("=" * 68, flush=True)
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(
        [sys.executable, str(BASE_DIR / command[0]), *command[1:]],
        cwd=str(BASE_DIR),
        env=environment,
        check=True,
    )


def main():
    """Run every stage in order and report the generated artefacts."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Run the full smart lighting pipeline.")
    parser.add_argument("--live", action="store_true",
                        help="slow the inference stage down to feel like a live feed")
    parser.add_argument("--test", action="store_true", help="run the pytest suite afterwards")
    args = parser.parse_args()

    stages = [list(stage) for stage in SCRIPTS]
    if args.live:
        stages[2] += ["--delay", "0.05"]

    print("#" * 68)
    print("A8 - SMART LIGHTING SYSTEM WITH AI AMBIENT ADAPTATION")
    print("#" * 68)
    try:
        for stage in stages:
            run_script(stage)
        if args.test:
            print("\n" + "=" * 68 + "\nRUNNING: pytest\n" + "=" * 68, flush=True)
            subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(BASE_DIR), check=True)
    except subprocess.CalledProcessError as error:
        print(f"\n[ERROR] Pipeline stopped: {error.cmd} exited with code {error.returncode}.")
        sys.exit(1)
    except FileNotFoundError as error:
        print(f"\n[ERROR] Could not start a pipeline stage: {error}")
        sys.exit(1)

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    png_files = sorted(OUTPUTS_DIR.glob("*.png"))
    print("\n" + "#" * 68)
    print("PIPELINE COMPLETE")
    print("#" * 68)
    print(f"  Deployed model : {meta['model_name']} "
          f"(CV F1 {meta['metrics']['cv_f1_mean']:.3f}, "
          f"rule coverage {meta['metrics']['rule_coverage']:.1%})")
    print(f"  Dataset        : {DATA_PATH}")
    print(f"  Model          : {MODEL_PATH}")
    print(f"  Inference log  : {SIMULATION_LOG_CSV}")
    print(f"  Energy report  : {ENERGY_REPORT_TXT}")
    print(f"  Actuator figure: {ACTUATOR_RESPONSE_PNG}")
    print(f"  PNG figures    : {len(png_files)} in {OUTPUTS_DIR}")
    for path in png_files:
        print(f"    - {path.name}")
    print("  Dashboard      : streamlit run app.py")
    print("#" * 68)


if __name__ == "__main__":
    main()
