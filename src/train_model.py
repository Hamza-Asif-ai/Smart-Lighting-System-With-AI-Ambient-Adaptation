"""Train, cross-validate and compare classifiers for ambient-adaptive lighting.

Workflow
--------
1. Stratified 80/20 train/test split (the test set is never used for selection).
2. 5-fold stratified cross-validation on the TRAIN split for every candidate.
3. Final scores on the held-out test set - both against the noisy observed
   labels and against the clean rule output (what the model really learned).
4. Model selection: best CV F1; if several are within ``SELECTION_TOLERANCE``,
   the smallest model wins (edge / microcontroller friendly).
5. Exhaustive check of every model against the clean rule on all 144 possible
   sensor states ("rule coverage").
6. Persist the model + a JSON metadata file, and export interpretable rules.
"""

import io
import json
import re
import sys
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

# Make `python src/<script>.py` work as well as `python -m src.<script>`.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from src.config import (  # noqa: E402
    AMBIENT_LEVELS,
    BRIGHTNESS_LEVELS,
    COMPARISON_CSV,
    COMPARISON_PNG,
    CONFUSION_PNG,
    CV_FOLDS,
    DATA_PATH,
    FEATURE_IMPORTANCE_PNG,
    FEATURE_NAMES,
    META_PATH,
    MODEL_PATH,
    NOISE_RATIO,
    RANDOM_SEED,
    RULES_TXT,
    SCHEMA_VERSION,
    SELECTION_TOLERANCE,
    TEST_SIZE,
    create_directories,
)
from src.features import decode_target, encode_features, encode_target  # noqa: E402
from src.sensor_simulator import all_states, generate_dataset, set_seed  # noqa: E402


def load_dataset(path=DATA_PATH):
    """Load the dataset from disk, generating it first if it does not exist."""
    if not path.exists():
        create_directories()
        generate_dataset().to_csv(path, index=False)
    return pd.read_csv(path)


def build_models():
    """Return the three classifiers compared in this experiment."""
    return {
        # min_samples_leaf stops the tree from memorising a handful of noisy
        # labels in rare sensor states (chosen via cross-validation).
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, random_state=RANDOM_SEED
        ),
        # kNN is distance based, so features must be on a comparable scale.
        "KNeighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=11, weights="distance")
        ),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }


def model_size_kb(model):
    """Serialised size of a model in kilobytes (relevant for edge deployment)."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return len(buffer.getvalue()) / 1024


def split_data(data):
    """Encode the dataset and return a stratified train/test split."""
    features = encode_features(data)
    target = encode_target(data["brightness"])
    clean_target = encode_target(data["ideal_brightness"])
    return train_test_split(
        features,
        target,
        clean_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )


def evaluate_models(models, x_train, y_train, x_test, y_test, clean_test):
    """Cross-validate on train, score on test; return a table and fitted models."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    states = all_states()
    state_features = encode_features(states)
    state_truth = encode_target(states["ideal_brightness"])
    rows, fitted = [], {}
    for name, model in models.items():
        cv_scores = cross_val_score(
            clone(model), x_train, y_train, cv=cv, scoring="f1_weighted"
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append(
            {
                "Model": name,
                "CV F1 (mean)": cv_scores.mean(),
                "CV F1 (std)": cv_scores.std(),
                "Test Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "F1-Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
                "Clean-Label Accuracy": accuracy_score(clean_test, predictions),
                "Rule Coverage (144 states)": accuracy_score(
                    state_truth, model.predict(state_features)
                ),
                "Size (KB)": model_size_kb(model),
            }
        )
        fitted[name] = model
    comparison = (
        pd.DataFrame(rows)
        .sort_values("CV F1 (mean)", ascending=False)
        .reset_index(drop=True)
    )
    return comparison, fitted


def select_best(comparison, tolerance=SELECTION_TOLERANCE):
    """Pick the best model by CV; among near-ties prefer the smallest model."""
    top = comparison["CV F1 (mean)"].max()
    eligible = comparison[comparison["CV F1 (mean)"] >= top - tolerance]
    winner = eligible.sort_values(
        ["Size (KB)", "CV F1 (mean)"], ascending=[True, False]
    ).iloc[0]
    return str(winner["Model"])


def export_tree_rules(tree_model):
    """Return the decision tree as human-readable IF/ELSE text."""
    text = export_text(tree_model, feature_names=FEATURE_NAMES)
    text = re.sub(r"class: (\d+)", lambda m: f"brightness: {BRIGHTNESS_LEVELS[int(m.group(1))]}", text)
    legend = [
        "Decision-tree control rules learned from the sensor data",
        "ambient_level: " + ", ".join(f"{i}={lvl}" for i, lvl in enumerate(AMBIENT_LEVELS)),
        "occupancy   : 0 = empty, 1 = occupied | hour: 0-23",
        "",
    ]
    return "\n".join(legend) + text


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_comparison(comparison, noise_ceiling):
    """Grouped bars: CV F1, held-out accuracy and clean-label accuracy."""
    long = comparison.melt(
        id_vars="Model",
        value_vars=["CV F1 (mean)", "Test Accuracy", "Clean-Label Accuracy",
                    "Rule Coverage (144 states)"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=long, x="Model", y="Score", hue="Metric", palette="viridis")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.85})
    ax.axhline(noise_ceiling, color="crimson", linestyle="--", linewidth=1.2, zorder=1,
               label=f"Noise ceiling vs noisy labels ({noise_ceiling:.0%})")
    ax.set_ylim(0.7, 1.04)
    ax.set_title("Classifier Comparison (cross-validated and held-out)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score (y-axis starts at 0.70)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=150)
    plt.close()


def plot_confusion_matrix(model, x_test, y_test, best_name):
    """Labelled confusion matrix with brightness levels in natural order."""
    order = list(range(len(BRIGHTNESS_LEVELS)))
    matrix = confusion_matrix(y_test, model.predict(x_test), labels=order)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=BRIGHTNESS_LEVELS, yticklabels=BRIGHTNESS_LEVELS)
    plt.title(f"Confusion Matrix - {best_name} (held-out test set)")
    plt.xlabel("Predicted brightness")
    plt.ylabel("Observed (noisy) brightness")
    plt.tight_layout()
    plt.savefig(CONFUSION_PNG, dpi=150)
    plt.close()


def plot_feature_importance(model, best_name):
    """Save a feature-importance bar chart when the model supports it."""
    if not hasattr(model, "feature_importances_"):
        return
    importance = pd.DataFrame(
        {"Feature": FEATURE_NAMES, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=importance, x="Importance", y="Feature",
                     palette="magma", hue="Feature", legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    plt.title(f"Feature Importance - {best_name}")
    plt.xlabel("Relative importance")
    plt.ylabel("Sensor feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_training(verbose=True, make_plots=True):
    """Train all models, persist the winner and return a summary dict."""
    create_directories()
    set_seed()
    sns.set_theme(style="whitegrid")

    data = load_dataset()
    noise_ceiling = float((data["brightness"] == data["ideal_brightness"]).mean())
    x_train, x_test, y_train, y_test, _, clean_test = split_data(data)

    comparison, fitted = evaluate_models(
        build_models(), x_train, y_train, x_test, y_test, clean_test
    )
    best_name = select_best(comparison)
    best_model = fitted[best_name]
    best_row = comparison[comparison["Model"] == best_name].iloc[0]

    comparison.to_csv(COMPARISON_CSV, index=False)
    RULES_TXT.write_text(export_tree_rules(fitted["DecisionTree"]) + "\n", encoding="utf-8")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_name": best_name,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURE_NAMES,
        "brightness_levels": BRIGHTNESS_LEVELS,
        "ambient_levels": AMBIENT_LEVELS,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "label_noise_ratio": NOISE_RATIO,
        "noise_ceiling_accuracy": noise_ceiling,
        "metrics": {
            "cv_f1_mean": float(best_row["CV F1 (mean)"]),
            "cv_f1_std": float(best_row["CV F1 (std)"]),
            "test_accuracy": float(best_row["Test Accuracy"]),
            "clean_label_accuracy": float(best_row["Clean-Label Accuracy"]),
            "rule_coverage": float(best_row["Rule Coverage (144 states)"]),
            "size_kb": float(best_row["Size (KB)"]),
        },
    }
    joblib.dump(best_model, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if make_plots:
        plot_comparison(comparison, noise_ceiling)
        plot_confusion_matrix(best_model, x_test, y_test, best_name)
        plot_feature_importance(best_model, best_name)

    if verbose:
        print("=" * 68)
        print("MODEL TRAINING & COMPARISON")
        print("=" * 68)
        print(f"Training samples: {len(x_train)} | Test samples: {len(x_test)}")
        print(f"Features: {', '.join(FEATURE_NAMES)}")
        print(f"Label-noise ceiling (accuracy of the perfect rule vs noisy labels): {noise_ceiling:.2%}")
        print(f"\nModel comparison ({CV_FOLDS}-fold CV on train + held-out test):")
        print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nBest model: {best_name} "
              f"(CV F1 {best_row['CV F1 (mean)']:.4f}, size {best_row['Size (KB)']:.1f} KB; "
              f"ties within {SELECTION_TOLERANCE} resolved by smallest size)")
        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved metadata: {META_PATH}")
        print(f"Saved tree rules: {RULES_TXT}")
        print("\nDetailed classification report (best model, vs observed labels):")
        print(classification_report(
            y_test, best_model.predict(x_test),
            labels=list(range(len(BRIGHTNESS_LEVELS))),
            target_names=BRIGHTNESS_LEVELS, zero_division=0,
        ))
        print("=" * 68)

    return {"comparison": comparison, "best_name": best_name, "meta": meta}


def main():
    """CLI entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    run_training()


if __name__ == "__main__":
    main()
