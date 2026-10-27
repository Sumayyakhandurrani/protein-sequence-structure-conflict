from pathlib import Path
import json
import hashlib

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
    "data/analysis/backbone_sensitivity_v1/"
    "final_test_v1"
)

OUT = BASE / "bootstrap"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

N_PROTEINS = 4731
N_LABELS = 494
N_BOOTSTRAP = 1000
SEED = 42


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_and_reshape(filename):

    path = BASE / filename
    x = np.load(path)

    expected = (
        N_PROTEINS * N_LABELS
    )

    if x.size != expected:
        raise RuntimeError(
            f"{filename}: expected "
            f"{expected:,} values, "
            f"got {x.size:,}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{filename}: non-finite values."
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
            float(
                roc_auc_score(
                    yf,
                    pf,
                )
            ),

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

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print(
    "E12 BACKBONE SENSITIVITY — "
    "WHOLE-PROTEIN BOOTSTRAP"
)
print("=" * 100)

print(
    f"Proteins:              "
    f"{N_PROTEINS:,}"
)

print(
    f"Labels per protein:    "
    f"{N_LABELS}"
)

print(
    f"Bootstrap replicates:  "
    f"{N_BOOTSTRAP:,}"
)

print(
    f"Seed:                  "
    f"{SEED}"
)

print(
    "Resampling unit:       whole protein"
)

print(
    "Models refit:          NO"
)

print(
    "Primary comparison:    MB - MA"
)

print(
    "Structure backbone:    alternative GCN"
)

print(
    "TEST USED FOR EVALUATION: YES"
)

print()


# ============================================================
# LOAD FIXED FINAL TEST PREDICTIONS
# ============================================================

files = {
    "binary_target":
        "structure_binary_target.npy",

    "binary_MA":
        "structure_binary_MA_predictions.npy",

    "binary_MB":
        "structure_binary_MB_predictions.npy",

    "continuous_target":
        "structure_continuous_target.npy",

    "continuous_MA":
        "structure_continuous_MA_predictions.npy",

    "continuous_MB":
        "structure_continuous_MB_predictions.npy",
}


data = {
    key:
        load_and_reshape(filename)
    for key, filename in files.items()
}


# ============================================================
# OBSERVED DELTAS
# ============================================================

binary_MA = binary_metrics(
    data["binary_target"],
    data["binary_MA"],
)

binary_MB = binary_metrics(
    data["binary_target"],
    data["binary_MB"],
)


continuous_MA = continuous_metrics(
    data["continuous_target"],
    data["continuous_MA"],
)

continuous_MB = continuous_metrics(
    data["continuous_target"],
    data["continuous_MB"],
)


observed = {
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
# WHOLE-PROTEIN BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    SEED
)

bootstrap_rows = []


for b in range(N_BOOTSTRAP):

    # Whole-protein resampling:
    # all 494 labels for each sampled protein
    # move together.
    idx = rng.integers(
        0,
        N_PROTEINS,
        size=N_PROTEINS,
    )

    yb = data[
        "binary_target"
    ][idx]

    ma_b = data[
        "binary_MA"
    ][idx]

    mb_b = data[
        "binary_MB"
    ][idx]


    yc = data[
        "continuous_target"
    ][idx]

    ma_c = data[
        "continuous_MA"
    ][idx]

    mb_c = data[
        "continuous_MB"
    ][idx]


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
        "replicate":
            b + 1,

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
            f"{b + 1:,}/"
            f"{N_BOOTSTRAP:,}"
        )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_csv = (
    OUT /
    "bootstrap_deltas.csv"
)

bootstrap_df.to_csv(
    bootstrap_csv,
    index=False,
)


# ============================================================
# SUMMARIZE
# ============================================================

metric_info = {
    "roc_auc": {
        "column":
            "delta_roc_auc",

        "better":
            "positive",
    },

    "average_precision": {
        "column":
            "delta_average_precision",

        "better":
            "positive",
    },

    "log_loss": {
        "column":
            "delta_log_loss",

        "better":
            "negative",
    },

    "brier": {
        "column":
            "delta_brier",

        "better":
            "negative",
    },

    "mae": {
        "column":
            "delta_mae",

        "better":
            "negative",
    },

    "rmse": {
        "column":
            "delta_rmse",

        "better":
            "negative",
    },

    "correlation": {
        "column":
            "delta_correlation",

        "better":
            "positive",
    },
}


summary = {
    "analysis":
        "e12_backbone_sensitivity_bootstrap_v1",

    "scientific_role":
        (
            "uncertainty analysis for "
            "alternative structure-backbone "
            "reliability sensitivity"
        ),

    "bootstrap_replicates":
        N_BOOTSTRAP,

    "seed":
        SEED,

    "resampling_unit":
        "whole protein",

    "labels_per_protein":
        N_LABELS,

    "models_refit":
        False,

    "conditional_on_fitted_models":
        True,

    "primary_comparison":
        "MB minus MA",

    "primary_metric":
        "average_precision",

    "alternative_structure_backbone":
        (
            "GCN-style normalized-neighbor "
            "propagation"
        ),

    "sequence_expert_modified":
        False,

    "primary_structure_model_modified":
        False,

    "test_used":
        False,

    "metrics": {},

    "input_sha256": {
        key:
            sha256_file(
                BASE / filename
            )
        for key, filename
        in files.items()
    },
}


print()
print("=" * 100)
print(
    "E12 BOOTSTRAP RESULTS: MB - MA"
)
print("=" * 100)


for metric, info in metric_info.items():

    values = bootstrap_df[
        info["column"]
    ].to_numpy()

    lo, hi = percentile_ci(
        values
    )

    obs = float(
        observed[metric]
    )

    mean_delta = float(
        values.mean()
    )


    if info["better"] == "positive":

        fraction_improved = float(
            np.mean(
                values > 0
            )
        )

    else:

        fraction_improved = float(
            np.mean(
                values < 0
            )
        )


    summary["metrics"][metric] = {
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


summary_json = (
    OUT /
    "bootstrap_summary.json"
)


with open(
    summary_json,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("Models refit: NO")
print("Sequence expert modified: NO")
print("Primary structure model modified: NO")
print("TEST USED FOR EVALUATION: YES")

print()
print("Saved:")
print(" ", bootstrap_csv)
print(" ", summary_json)

print("=" * 100)
