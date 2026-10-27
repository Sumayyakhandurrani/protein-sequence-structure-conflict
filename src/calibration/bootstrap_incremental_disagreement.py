from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(
    "data/conflict/incremental_reliability_v1"
)

SPLIT = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

OUT = Path(
    "data/conflict/"
    "incremental_reliability_v1/"
    "bootstrap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

N_BOOTSTRAP = 1000
SEED = 42

N_PROTEINS = 2373
N_LABELS = 494


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


def binary_log_loss(y, p):
    eps = 1e-7

    p = np.clip(
        p,
        eps,
        1.0 - eps,
    )

    return float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y)
            * np.log(1.0 - p)
        )
    )


def brier(y, p):
    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def mae(y, p):
    return float(
        np.mean(
            np.abs(p - y)
        )
    )


def rmse(y, p):
    return float(
        np.sqrt(
            np.mean(
                (p - y) ** 2
            )
        )
    )


def correlation(y, p):

    y_std = np.std(y)
    p_std = np.std(p)

    if y_std == 0 or p_std == 0:
        return np.nan

    return float(
        np.corrcoef(
            y,
            p,
        )[0, 1]
    )


def percentile_ci(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    return {
        "mean":
            float(np.mean(values)),

        "median":
            float(np.median(values)),

        "ci95_lower":
            float(
                np.quantile(
                    values,
                    0.025,
                )
            ),

        "ci95_upper":
            float(
                np.quantile(
                    values,
                    0.975,
                )
            ),

        "fraction_positive":
            float(
                np.mean(values > 0)
            ),

        "fraction_negative":
            float(
                np.mean(values < 0)
            ),
    }


# ============================================================
# LOAD
# ============================================================

print("=" * 88)
print("PAIRED PROTEIN-LEVEL BOOTSTRAP V1")
print("=" * 88)

print(f"Bootstrap replicates:      {N_BOOTSTRAP:,}")
print(f"Random seed:               {SEED}")
print("Resampling unit:           whole protein")
print(f"Labels carried/protein:    {N_LABELS}")
print("M0/M1 resampling paired:   YES")
print("Models refitted:           NO")
print("TEST USED:                 NO")
print()


eval_ids = read_ids(
    SPLIT / "structure_cal_eval_ids.txt"
)

if len(eval_ids) != N_PROTEINS:
    raise RuntimeError(
        f"Expected {N_PROTEINS} proteins, "
        f"got {len(eval_ids)}"
    )


binary_y = np.load(
    BASE / "binary_eval_target.npy"
).reshape(
    N_PROTEINS,
    N_LABELS,
).astype(np.float64)


binary_M0 = np.load(
    BASE / "binary_eval_M0_predictions.npy"
).reshape(
    N_PROTEINS,
    N_LABELS,
).astype(np.float64)


binary_M1 = np.load(
    BASE / "binary_eval_M1_predictions.npy"
).reshape(
    N_PROTEINS,
    N_LABELS,
).astype(np.float64)


cont_y = np.load(
    BASE / "continuous_eval_target.npy"
).reshape(
    N_PROTEINS,
    N_LABELS,
).astype(np.float64)


cont_M0 = np.load(
    BASE / "continuous_eval_M0_predictions.npy"
).reshape(
    N_PROTEINS,
    N_LABELS,
).astype(np.float64)


cont_M1 = np.load(
    BASE / "continuous_eval_M1_predictions.npy"
).reshape(
    N_PROTEINS,
    N_LABELS,
).astype(np.float64)


# ============================================================
# OBSERVED DELTAS
# ============================================================

yb = binary_y.ravel()
p0 = binary_M0.ravel()
p1 = binary_M1.ravel()

yc = cont_y.ravel()
c0 = cont_M0.ravel()
c1 = cont_M1.ravel()


observed = {
    "roc_auc":
        float(
            roc_auc_score(yb, p1)
            - roc_auc_score(yb, p0)
        ),

    "average_precision":
        float(
            average_precision_score(yb, p1)
            - average_precision_score(yb, p0)
        ),

    "log_loss":
        float(
            binary_log_loss(yb, p1)
            - binary_log_loss(yb, p0)
        ),

    "brier":
        float(
            brier(yb, p1)
            - brier(yb, p0)
        ),

    "mae":
        float(
            mae(yc, c1)
            - mae(yc, c0)
        ),

    "rmse":
        float(
            rmse(yc, c1)
            - rmse(yc, c0)
        ),

    "correlation":
        float(
            correlation(yc, c1)
            - correlation(yc, c0)
        ),
}


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(SEED)

records = []


for b in range(N_BOOTSTRAP):

    protein_idx = rng.integers(
        0,
        N_PROTEINS,
        size=N_PROTEINS,
    )

    boot_yb = binary_y[
        protein_idx
    ].reshape(-1)

    boot_p0 = binary_M0[
        protein_idx
    ].reshape(-1)

    boot_p1 = binary_M1[
        protein_idx
    ].reshape(-1)


    # With this dataset every protein-level bootstrap
    # should contain both binary classes, but guard anyway.
    if (
        np.unique(boot_yb).size < 2
    ):
        auc_delta = np.nan
        ap_delta = np.nan

    else:
        auc_delta = (
            roc_auc_score(
                boot_yb,
                boot_p1,
            )
            - roc_auc_score(
                boot_yb,
                boot_p0,
            )
        )

        ap_delta = (
            average_precision_score(
                boot_yb,
                boot_p1,
            )
            - average_precision_score(
                boot_yb,
                boot_p0,
            )
        )


    logloss_delta = (
        binary_log_loss(
            boot_yb,
            boot_p1,
        )
        - binary_log_loss(
            boot_yb,
            boot_p0,
        )
    )


    brier_delta = (
        brier(
            boot_yb,
            boot_p1,
        )
        - brier(
            boot_yb,
            boot_p0,
        )
    )


    boot_yc = cont_y[
        protein_idx
    ].reshape(-1)

    boot_c0 = cont_M0[
        protein_idx
    ].reshape(-1)

    boot_c1 = cont_M1[
        protein_idx
    ].reshape(-1)


    mae_delta = (
        mae(
            boot_yc,
            boot_c1,
        )
        - mae(
            boot_yc,
            boot_c0,
        )
    )


    rmse_delta = (
        rmse(
            boot_yc,
            boot_c1,
        )
        - rmse(
            boot_yc,
            boot_c0,
        )
    )


    corr_delta = (
        correlation(
            boot_yc,
            boot_c1,
        )
        - correlation(
            boot_yc,
            boot_c0,
        )
    )


    records.append({
        "bootstrap":
            b + 1,

        "roc_auc_delta":
            auc_delta,

        "average_precision_delta":
            ap_delta,

        "log_loss_delta":
            logloss_delta,

        "brier_delta":
            brier_delta,

        "mae_delta":
            mae_delta,

        "rmse_delta":
            rmse_delta,

        "correlation_delta":
            corr_delta,
    })


    if (
        (b + 1) % 100 == 0
        or b == 0
    ):
        print(
            f"Completed "
            f"{b + 1:4d}/"
            f"{N_BOOTSTRAP}"
        )


bootstrap_df = pd.DataFrame(
    records
)


bootstrap_df.to_csv(
    OUT / "bootstrap_deltas.csv",
    index=False,
)


# ============================================================
# CONFIDENCE INTERVALS
# ============================================================

metrics = {
    "roc_auc":
        "roc_auc_delta",

    "average_precision":
        "average_precision_delta",

    "log_loss":
        "log_loss_delta",

    "brier":
        "brier_delta",

    "mae":
        "mae_delta",

    "rmse":
        "rmse_delta",

    "correlation":
        "correlation_delta",
}


summary_metrics = {}

for metric, column in metrics.items():

    stats = percentile_ci(
        bootstrap_df[column].values
    )

    stats["observed_delta"] = (
        observed[metric]
    )

    if metric in [
        "roc_auc",
        "average_precision",
        "correlation",
    ]:
        stats[
            "improvement_direction"
        ] = "positive"

        stats[
            "fraction_improved"
        ] = stats[
            "fraction_positive"
        ]

    else:
        stats[
            "improvement_direction"
        ] = "negative"

        stats[
            "fraction_improved"
        ] = stats[
            "fraction_negative"
        ]

    summary_metrics[
        metric
    ] = stats


summary = {
    "analysis":
        "paired_protein_bootstrap_v1",

    "bootstrap_replicates":
        N_BOOTSTRAP,

    "seed":
        SEED,

    "resampling_unit":
        "protein",

    "proteins_per_replicate":
        N_PROTEINS,

    "labels_per_protein":
        N_LABELS,

    "paired_M0_M1_resampling":
        True,

    "models_refitted_per_bootstrap":
        False,

    "confidence_interval":
        "percentile_95",

    "metrics":
        summary_metrics,

    "interpretation":
        (
            "Bootstrap quantifies uncertainty of "
            "held-out evaluation differences conditional "
            "on the already-fitted reliability models."
        ),

    "test_used":
        False,
}


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


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 88)
print("BOOTSTRAP RESULTS — DELTA = M1 - M0")
print("=" * 88)


for metric in [
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
    "mae",
    "rmse",
    "correlation",
]:

    s = summary_metrics[metric]

    direction = (
        "higher better"
        if s[
            "improvement_direction"
        ] == "positive"
        else "lower better"
    )

    print()
    print(
        f"{metric.upper()} "
        f"({direction})"
    )

    print(
        f"  Observed delta:   "
        f"{s['observed_delta']:+.8f}"
    )

    print(
        f"  Bootstrap mean:   "
        f"{s['mean']:+.8f}"
    )

    print(
        f"  95% CI:           "
        f"[{s['ci95_lower']:+.8f}, "
        f"{s['ci95_upper']:+.8f}]"
    )

    print(
        f"  Fraction improved:"
        f" {s['fraction_improved']:.4f}"
    )


print()
print("=" * 88)
print("INTERPRETATION GUARDRAILS")
print("=" * 88)

print(
    "Resampling unit:                    PROTEIN"
)

print(
    "All 494 labels carried together:    YES"
)

print(
    "M0/M1 evaluated on same resamples:  YES"
)

print(
    "Reliability models refitted:        NO"
)

print(
    "CI type:                            percentile 95%"
)

print(
    "This is development-stage VAL:      YES"
)

print(
    "Final TEST confirmation:            NO"
)

print(
    "TEST USED:                          NO"
)

print()
print(
    f"Saved under:\n  {OUT}"
)

print("=" * 88)