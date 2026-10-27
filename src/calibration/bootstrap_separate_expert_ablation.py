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
# CONFIGURATION
# ============================================================

BASE = Path(
    "data/conflict/"
    "separate_expert_reliability_ablation_v1"
)

OUT = BASE / "bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

N_PROTEINS = 2373
N_LABELS = 494
N_BOOTSTRAP = 1000
SEED = 42


# ============================================================
# HELPERS
# ============================================================

def load_and_reshape(filename):
    x = np.load(BASE / filename)

    expected = N_PROTEINS * N_LABELS

    if x.size != expected:
        raise RuntimeError(
            f"{filename}: expected {expected:,} values, "
            f"got {x.size:,}"
        )

    return x.reshape(
        N_PROTEINS,
        N_LABELS,
    )


def binary_metrics(y, p):
    yf = y.reshape(-1)
    pf = p.reshape(-1)

    return {
        "roc_auc":
            float(roc_auc_score(yf, pf)),

        "average_precision":
            float(
                average_precision_score(
                    yf,
                    pf,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    yf,
                    pf,
                    labels=[0, 1],
                )
            ),

        "brier":
            float(
                brier_score_loss(
                    yf,
                    pf,
                )
            ),
    }


def continuous_metrics(y, pred):
    yf = y.reshape(-1)
    pf = pred.reshape(-1)

    return {
        "mae":
            float(
                mean_absolute_error(
                    yf,
                    pf,
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        yf,
                        pf,
                    )
                )
            ),

        "correlation":
            float(
                np.corrcoef(
                    yf,
                    pf,
                )[0, 1]
            ),
    }


def percentile_ci(values):
    values = np.asarray(values)

    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("=" * 100)
print("PROTEIN-LEVEL BOOTSTRAP: SEPARATE-EXPERT MA -> MB ABLATION")
print("=" * 100)

print(f"Proteins:              {N_PROTEINS:,}")
print(f"Labels per protein:    {N_LABELS}")
print(f"Bootstrap replicates:  {N_BOOTSTRAP:,}")
print(f"Seed:                  {SEED}")
print("Resampling unit:       whole protein")
print("Models refit:          NO")
print("Primary comparison:    MB - MA")
print("TEST USED:             NO")
print()


data = {}

for expert in [
    "sequence",
    "structure",
]:

    data[expert] = {
        "binary_target":
            load_and_reshape(
                f"{expert}_binary_target.npy"
            ),

        "binary_MA":
            load_and_reshape(
                f"{expert}_binary_MA_predictions.npy"
            ),

        "binary_MB":
            load_and_reshape(
                f"{expert}_binary_MB_predictions.npy"
            ),

        "continuous_target":
            load_and_reshape(
                f"{expert}_continuous_target.npy"
            ),

        "continuous_MA":
            load_and_reshape(
                f"{expert}_continuous_MA_predictions.npy"
            ),

        "continuous_MB":
            load_and_reshape(
                f"{expert}_continuous_MB_predictions.npy"
            ),
    }


# ============================================================
# OBSERVED DELTAS
# ============================================================

observed = {}


for expert in [
    "sequence",
    "structure",
]:

    d = data[expert]

    binary_MA = binary_metrics(
        d["binary_target"],
        d["binary_MA"],
    )

    binary_MB = binary_metrics(
        d["binary_target"],
        d["binary_MB"],
    )

    continuous_MA = continuous_metrics(
        d["continuous_target"],
        d["continuous_MA"],
    )

    continuous_MB = continuous_metrics(
        d["continuous_target"],
        d["continuous_MB"],
    )

    observed[expert] = {
        "roc_auc":
            binary_MB["roc_auc"]
            - binary_MA["roc_auc"],

        "average_precision":
            binary_MB["average_precision"]
            - binary_MA["average_precision"],

        "log_loss":
            binary_MB["log_loss"]
            - binary_MA["log_loss"],

        "brier":
            binary_MB["brier"]
            - binary_MA["brier"],

        "mae":
            continuous_MB["mae"]
            - continuous_MA["mae"],

        "rmse":
            continuous_MB["rmse"]
            - continuous_MA["rmse"],

        "correlation":
            continuous_MB["correlation"]
            - continuous_MA["correlation"],
    }


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(SEED)

bootstrap_rows = []


for b in range(N_BOOTSTRAP):

    # Sample proteins WITH replacement.
    # All 494 labels for a sampled protein move together.
    idx = rng.integers(
        0,
        N_PROTEINS,
        size=N_PROTEINS,
    )


    for expert in [
        "sequence",
        "structure",
    ]:

        d = data[expert]

        yb = d["binary_target"][idx]
        ma_b = d["binary_MA"][idx]
        mb_b = d["binary_MB"][idx]

        yc = d["continuous_target"][idx]
        ma_c = d["continuous_MA"][idx]
        mb_c = d["continuous_MB"][idx]


        bm_a = binary_metrics(
            yb,
            ma_b,
        )

        bm_b = binary_metrics(
            yb,
            mb_b,
        )

        cm_a = continuous_metrics(
            yc,
            ma_c,
        )

        cm_b = continuous_metrics(
            yc,
            mb_c,
        )


        bootstrap_rows.append({
            "replicate": b + 1,
            "expert": expert,

            "delta_roc_auc":
                bm_b["roc_auc"]
                - bm_a["roc_auc"],

            "delta_average_precision":
                bm_b["average_precision"]
                - bm_a["average_precision"],

            "delta_log_loss":
                bm_b["log_loss"]
                - bm_a["log_loss"],

            "delta_brier":
                bm_b["brier"]
                - bm_a["brier"],

            "delta_mae":
                cm_b["mae"]
                - cm_a["mae"],

            "delta_rmse":
                cm_b["rmse"]
                - cm_a["rmse"],

            "delta_correlation":
                cm_b["correlation"]
                - cm_a["correlation"],
        })


    if (
        (b + 1) % 100 == 0
        or b == 0
    ):
        print(
            f"Completed bootstrap "
            f"{b + 1:,}/{N_BOOTSTRAP:,}"
        )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_df.to_csv(
    OUT / "bootstrap_deltas.csv",
    index=False,
)


# ============================================================
# SUMMARIZE
# ============================================================

metric_info = {
    "roc_auc": {
        "column": "delta_roc_auc",
        "better": "positive",
    },

    "average_precision": {
        "column":
            "delta_average_precision",
        "better": "positive",
    },

    "log_loss": {
        "column": "delta_log_loss",
        "better": "negative",
    },

    "brier": {
        "column": "delta_brier",
        "better": "negative",
    },

    "mae": {
        "column": "delta_mae",
        "better": "negative",
    },

    "rmse": {
        "column": "delta_rmse",
        "better": "negative",
    },

    "correlation": {
        "column":
            "delta_correlation",
        "better": "positive",
    },
}


summary = {
    "analysis":
        "protein_level_bootstrap_separate_expert_ablation_v1",

    "bootstrap_replicates":
        N_BOOTSTRAP,

    "seed":
        SEED,

    "resampling_unit":
        "protein",

    "models_refit":
        False,

    "primary_comparison":
        "MB minus MA",

    "conditional_on_fitted_models":
        True,

    "test_used":
        False,

    "experts": {},
}


print()
print("=" * 100)
print("BOOTSTRAP RESULTS: MB - MA")
print("=" * 100)


for expert in [
    "sequence",
    "structure",
]:

    print()
    print(expert.upper())

    expert_df = bootstrap_df[
        bootstrap_df["expert"] == expert
    ]

    summary["experts"][expert] = {}


    for metric, info in metric_info.items():

        values = expert_df[
            info["column"]
        ].to_numpy()

        lo, hi = percentile_ci(values)

        obs = float(
            observed[expert][metric]
        )

        mean_delta = float(
            values.mean()
        )


        if info["better"] == "positive":
            fraction_improved = float(
                np.mean(values > 0)
            )
        else:
            fraction_improved = float(
                np.mean(values < 0)
            )


        summary[
            "experts"
        ][expert][metric] = {
            "observed_delta":
                obs,

            "bootstrap_mean_delta":
                mean_delta,

            "ci_2.5":
                lo,

            "ci_97.5":
                hi,

            "fraction_improved":
                fraction_improved,

            "desired_direction":
                info["better"],
        }


        print(
            f"  {metric:<18} "
            f"observed={obs:+.8f} | "
            f"mean={mean_delta:+.8f} | "
            f"95% CI "
            f"[{lo:+.8f}, {hi:+.8f}] | "
            f"improved={fraction_improved:.3f}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    OUT / "bootstrap_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 100)
print("FINAL STATUS")
print("=" * 100)

print("Whole-protein resampling:          YES")
print("All 494 labels kept together:      YES")
print("Models refit per bootstrap:         NO")
print("Primary comparison:                MB - MA")
print("Bootstrap replicates:              1,000")
print("Threshold optimization:            NO")
print("pLDDT used:                        NO")
print("Final TEST confirmation:           NO")
print("TEST USED:                         NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)