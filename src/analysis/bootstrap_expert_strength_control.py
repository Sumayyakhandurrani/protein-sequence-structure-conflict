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
    "data/conflict/expert_strength_control_v1"
)

OUT = BASE / "bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
SEED = 42


# ============================================================
# LOAD SAVED CAL-EVAL DATA
# ============================================================

y_bin = np.load(
    BASE / "matched_binary_target.npy"
)

pA_bin = np.load(
    BASE / "matched_binary_MA_predictions.npy"
)

pB_bin = np.load(
    BASE / "matched_binary_MB_predictions.npy"
)

y_cont = np.load(
    BASE / "matched_continuous_target.npy"
)

pA_cont = np.load(
    BASE / "matched_continuous_MA_predictions.npy"
)

pB_cont = np.load(
    BASE / "matched_continuous_MB_predictions.npy"
)


assert (
    y_bin.shape
    == pA_bin.shape
    == pB_bin.shape
    == y_cont.shape
    == pA_cont.shape
    == pB_cont.shape
)

n_proteins, n_labels = y_bin.shape

assert n_proteins == 2373
assert n_labels == 92

print("=" * 100)
print("EXPERT-STRENGTH CONTROL — PROTEIN-LEVEL BOOTSTRAP")
print("=" * 100)

print(f"Proteins:              {n_proteins:,}")
print(f"Matched labels:        {n_labels}")
print(f"Bootstrap replicates:  {N_BOOT:,}")
print(f"Seed:                  {SEED}")
print("Resampling unit:       whole protein")
print("Models refitted:       NO")
print("Labels reselected:     NO")
print("TEST USED:             NO")
print()


# ============================================================
# METRIC FUNCTIONS
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

    if (
        np.std(y) > 0
        and np.std(p) > 0
    ):
        corr = np.corrcoef(y, p)[0, 1]
    else:
        corr = np.nan

    return {
        "mae":
            mean_absolute_error(y, p),

        "rmse":
            np.sqrt(
                mean_squared_error(y, p)
            ),

        "correlation":
            corr,
    }


# ============================================================
# OBSERVED DELTAS
# ============================================================

A_bin_obs = binary_metrics(
    y_bin,
    pA_bin,
)

B_bin_obs = binary_metrics(
    y_bin,
    pB_bin,
)

A_cont_obs = continuous_metrics(
    y_cont,
    pA_cont,
)

B_cont_obs = continuous_metrics(
    y_cont,
    pB_cont,
)


observed = {
    "roc_auc":
        B_bin_obs["roc_auc"]
        - A_bin_obs["roc_auc"],

    "average_precision":
        B_bin_obs["average_precision"]
        - A_bin_obs["average_precision"],

    "log_loss":
        B_bin_obs["log_loss"]
        - A_bin_obs["log_loss"],

    "brier":
        B_bin_obs["brier"]
        - A_bin_obs["brier"],

    "mae":
        B_cont_obs["mae"]
        - A_cont_obs["mae"],

    "rmse":
        B_cont_obs["rmse"]
        - A_cont_obs["rmse"],

    "correlation":
        B_cont_obs["correlation"]
        - A_cont_obs["correlation"],
}


# ============================================================
# WHOLE-PROTEIN BOOTSTRAP
# ============================================================

rng = np.random.default_rng(SEED)

records = []

for b in range(N_BOOT):

    # Resample entire proteins.
    # All 92 labels for a sampled protein stay together.
    idx = rng.integers(
        0,
        n_proteins,
        size=n_proteins,
    )

    yb = y_bin[idx]
    pA_b = pA_bin[idx]
    pB_b = pB_bin[idx]

    yc = y_cont[idx]
    pA_c = pA_cont[idx]
    pB_c = pB_cont[idx]

    A_bin = binary_metrics(
        yb,
        pA_b,
    )

    B_bin = binary_metrics(
        yb,
        pB_b,
    )

    A_cont = continuous_metrics(
        yc,
        pA_c,
    )

    B_cont = continuous_metrics(
        yc,
        pB_c,
    )

    records.append({
        "replicate": b,

        "roc_auc":
            B_bin["roc_auc"]
            - A_bin["roc_auc"],

        "average_precision":
            B_bin["average_precision"]
            - A_bin["average_precision"],

        "log_loss":
            B_bin["log_loss"]
            - A_bin["log_loss"],

        "brier":
            B_bin["brier"]
            - A_bin["brier"],

        "mae":
            B_cont["mae"]
            - A_cont["mae"],

        "rmse":
            B_cont["rmse"]
            - A_cont["rmse"],

        "correlation":
            B_cont["correlation"]
            - A_cont["correlation"],
    })

    if (b + 1) % 100 == 0:
        print(
            f"Completed {b + 1}/{N_BOOT}"
        )


boot = pd.DataFrame(records)

boot.to_csv(
    OUT / "bootstrap_deltas.csv",
    index=False,
)


# ============================================================
# SUMMARIZE
# ============================================================

# Direction of improvement:
# higher is better for ROC/AP/correlation
# lower is better for loss/Brier/MAE/RMSE.

higher_better = {
    "roc_auc",
    "average_precision",
    "correlation",
}

summary_rows = []

print()
print("=" * 100)
print("BOOTSTRAP RESULTS: MB - MA")
print("=" * 100)

for metric in [
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
    "mae",
    "rmse",
    "correlation",
]:

    vals = boot[metric].to_numpy()

    lo = float(
        np.percentile(vals, 2.5)
    )

    hi = float(
        np.percentile(vals, 97.5)
    )

    mean = float(
        np.mean(vals)
    )

    obs = float(
        observed[metric]
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
        f"improved={improvement_fraction:.3f}"
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT / "bootstrap_summary.csv",
    index=False,
)


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = {
    "analysis":
        "expert_strength_control_v1_bootstrap",

    "bootstrap_replicates":
        N_BOOT,

    "seed":
        SEED,

    "resampling_unit":
        "whole protein",

    "proteins":
        int(n_proteins),

    "matched_labels":
        int(n_labels),

    "comparison":
        "MB_minus_MA",

    "MA_features": [
        "U_structure",
        "U_sequence",
    ],

    "MB_features": [
        "U_structure",
        "U_sequence",
        "absolute_disagreement",
    ],

    "models_refitted_per_bootstrap":
        False,

    "labels_reselected_per_bootstrap":
        False,

    "interpretation":
        (
            "Bootstrap intervals quantify CAL-EVAL "
            "evaluation-set uncertainty conditional "
            "on the already fitted reliability models "
            "and frozen matched-label subset."
        ),

    "test_used":
        False,
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
print("All 92 labels kept together:    YES")
print("Reliability models refitted:    NO")
print("Matched labels reselected:      NO")
print("TEST USED:                      NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)