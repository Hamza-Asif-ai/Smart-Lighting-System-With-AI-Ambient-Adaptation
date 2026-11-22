"""Repository layout checks: scripts run directly and the README stays truthful.

These guard against the project being moved into src/ and dashboard/ while the
documentation (or direct `python src/<script>.py` runs) silently break.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ["sensor_simulator", "train_model", "inference", "visualize",
           "energy_analysis", "actuator"]


def _clean_env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # prove the scripts do not rely on it
    return env


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_imports_when_run_directly(name, tmp_path):
    """Mimic `python src/<name>.py`: only src/ on sys.path, no PYTHONPATH, other cwd."""
    code = (
        "import runpy, sys; "
        f"sys.path = [{str(SRC)!r}] + [p for p in sys.path if p not in ('', {str(ROOT)!r})]; "
        f"runpy.run_path({str(SRC / (name + '.py'))!r}, run_name='direct_run_check')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=_clean_env(),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-800:]


@pytest.mark.parametrize("target", ["src/inference.py", "src/actuator.py", "run_all.py"])
def test_cli_help_works_from_any_directory(target, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / target), "--help"], cwd=tmp_path,
                            env=_clean_env(), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-800:]
    assert "usage" in result.stdout.lower()


def test_module_form_works_from_repo_root():
    result = subprocess.run([sys.executable, "-m", "src.actuator", "--help"], cwd=ROOT,
                            env=_clean_env(), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-800:]


def test_readme_commands_point_to_existing_files():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"^(?:python|streamlit run)\s+([\w./-]+\.py)", text, re.M)
    assert {"run_all.py", "dashboard/app.py"} <= set(targets)
    for target in targets:
        assert (ROOT / target).exists(), f"README tells users to run a missing file: {target}"


def test_readme_mentions_every_source_file():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for path in sorted(SRC.glob("*.py")) + [ROOT / "dashboard" / "app.py", ROOT / "run_all.py"]:
        if path.name != "__init__.py":
            assert path.name in text, f"README does not mention {path.relative_to(ROOT)}"
