from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    "data/conflict/conflict_formulation_comparison_v1"
)

OUT = BASE / "bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
SEED = 42


# ============================================================
# LOAD
# ============================================================

y_bin = np.load(BASE / "binary_target.npy")
y_cont = np.load(BASE / "continuous_target.npy")

predictions = {}

for name in [
    "C1_absolute",
    "C2_bernoulli_JS",
    "C3_signed",
]:
    predictions[name] = {
        "binary": np.load(
            BASE / f"{name}_binary_predictions.npy"
        ),
        "continuous": np.load(
            BASE / f"{name}_continuous_predictions.npy"
        ),
    }


assert y_bin.shape == y_cont.shape

for name in predictions:
    assert predictions[name]["binary"].shape == y_bin.shape
    assert predictions[name]["continuous"].shape == y_cont.shape

n_proteins, n_labels = y_bin.shape


print("=" * 100)
print("CONFLICT FORMULATION — PAIRED PROTEIN BOOTSTRAP")
print("=" * 100)

print(f"Proteins:              {n_proteins:,}")
print(f"Labels:                {n_labels}")
print(f"Bootstrap replicates:  {N_BOOT:,}")
print(f"Seed:                  {SEED}")
print("Resampling unit:       whole protein")
print("Models refitted:       NO")
print("TEST USED:             NO")
print()


# ============================================================
# METRICS
# ============================================================

def binary_metrics(y, p):

    y = y.reshape(-1)
    p = p.reshape(-1)

    return {
        "roc_auc":
            roc_auc_score(y, p),

        "average_precision":
            average_precision_score(y, p),

        "log_loss":
            log_loss(
                y,
                p,
                labels=[0, 1],
            ),

        "brier":
            brier_score_loss(y, p),
    }


def continuous_metrics(y, p):

    y = y.reshape(-1)
    p = p.reshape(-1)

    return {
        "mae":
            mean_absolute_error(y, p),

        "rmse":
            np.sqrt(
                mean_squared_error(y, p)
            ),

        "correlation":
            np.corrcoef(y, p)[0, 1],
    }


# ============================================================
# OBSERVED DIRECT COMPARISONS
# ============================================================

def compare_models(name_a, name_b, indices=None):
    """
    Returns metrics for B minus A.

    Example:
        A = C1
        B = C2

    Positive delta is favorable for:
        ROC-AUC, AP, correlation

    Negative delta is favorable for:
        log-loss, Brier, MAE, RMSE
    """

    if indices is None:
        yb = y_bin
        yc = y_cont

        pa_bin = predictions[name_a]["binary"]
        pb_bin = predictions[name_b]["binary"]

        pa_cont = predictions[name_a]["continuous"]
        pb_cont = predictions[name_b]["continuous"]

    else:
        yb = y_bin[indices]
        yc = y_cont[indices]

        pa_bin = predictions[name_a]["binary"][indices]
        pb_bin = predictions[name_b]["binary"][indices]

        pa_cont = predictions[name_a]["continuous"][indices]
        pb_cont = predictions[name_b]["continuous"][indices]

    A_bin = binary_metrics(yb, pa_bin)
    B_bin = binary_metrics(yb, pb_bin)

    A_cont = continuous_metrics(yc, pa_cont)
    B_cont = continuous_metrics(yc, pb_cont)

    return {
        "roc_auc":
            B_bin["roc_auc"] - A_bin["roc_auc"],

        "average_precision":
            B_bin["average_precision"]
            - A_bin["average_precision"],

        "log_loss":
            B_bin["log_loss"] - A_bin["log_loss"],

        "brier":
            B_bin["brier"] - A_bin["brier"],

        "mae":
            B_cont["mae"] - A_cont["mae"],

        "rmse":
            B_cont["rmse"] - A_cont["rmse"],

        "correlation":
            B_cont["correlation"]
            - A_cont["correlation"],
    }


comparisons = [
    ("C1_absolute", "C2_bernoulli_JS"),
    ("C1_absolute", "C3_signed"),
    ("C3_signed", "C2_bernoulli_JS"),
]


observed = {}

for a, b in comparisons:
    key = f"{b}_minus_{a}"
    observed[key] = compare_models(a, b)


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(SEED)

records = []

for rep in range(N_BOOT):

    # Whole-protein bootstrap:
    # all 494 labels for each sampled protein remain together.
    idx = rng.integers(
        0,
        n_proteins,
        size=n_proteins,
    )

    for a, b in comparisons:

        key = f"{b}_minus_{a}"

        delta = compare_models(
            a,
            b,
            indices=idx,
        )

        row = {
            "replicate": rep,
            "comparison": key,
        }

        row.update(delta)

        records.append(row)

    if (rep + 1) % 100 == 0:
        print(
            f"Completed {rep + 1}/{N_BOOT}"
        )


boot = pd.DataFrame(records)

boot.to_csv(
    OUT / "paired_bootstrap_deltas.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

higher_better = {
    "roc_auc",
    "average_precision",
    "correlation",
}

summary_rows = []


print()
print("=" * 100)
print("PAIRED BOOTSTRAP RESULTS")
print("=" * 100)


for a, b in comparisons:

    key = f"{b}_minus_{a}"

    print()
    print("-" * 100)
    print(key)
    print("-" * 100)

    sub = boot[
        boot["comparison"] == key
    ]

    for metric in [
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier",
        "mae",
        "rmse",
        "correlation",
    ]:

        vals = sub[metric].to_numpy()

        obs = float(
            observed[key][metric]
        )

        mean = float(
            np.mean(vals)
        )

        lo = float(
            np.percentile(vals, 2.5)
        )

        hi = float(
            np.percentile(vals, 97.5)
        )

        if metric in higher_better:
            improvement_fraction = float(
                np.mean(vals > 0)
            )
        else:
            improvement_fraction = float(
                np.mean(vals < 0)
            )

        summary_rows.append({
            "comparison": key,
            "metric": metric,
            "observed_delta": obs,
            "bootstrap_mean_delta": mean,
            "ci_2.5": lo,
            "ci_97.5": hi,
            "improvement_fraction":
                improvement_fraction,
        })

        print(
            f"{metric:<20} "
            f"observed={obs:+.8f} | "
            f"mean={mean:+.8f} | "
            f"95% CI=[{lo:+.8f}, {hi:+.8f}] | "
            f"B better={improvement_fraction:.3f}"
        )


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT / "paired_bootstrap_summary.csv",
    index=False,
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "analysis":
        "conflict_formulation_paired_bootstrap_v1",

    "bootstrap_replicates":
        N_BOOT,

    "seed":
        SEED,

    "resampling_unit":
        "whole protein",

    "proteins":
        int(n_proteins),

    "labels":
        int(n_labels),

    "candidate_formulations": [
        "C1_absolute",
        "C2_bernoulli_JS",
        "C3_signed",
    ],

    "primary_selection_metric":
        "average precision",

    "primary_direct_comparison":
        "C2_bernoulli_JS_minus_C1_absolute",

    "models_refitted_per_bootstrap":
        False,

    "test_used":
        False,

    "interpretation":
        (
            "Intervals quantify CAL-EVAL sampling uncertainty "
            "conditional on the already fitted reliability models. "
            "All labels for a sampled protein remain together."
        ),
}

with open(
    OUT / "bootstrap_manifest.json",
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


print()
print("=" * 100)
print("INTEGRITY")
print("=" * 100)

print("Whole proteins resampled:       YES")
print("All 494 labels kept together:   YES")
print("Models refitted:                NO")
print("Candidate set changed:          NO")
print("TEST USED:                      NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)