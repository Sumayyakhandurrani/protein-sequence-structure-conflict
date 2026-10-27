from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# PATHS
# ============================================================

BASE = Path("data/conflict/paired_experts_v1")

SPLIT = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

OUT = Path(
    "data/conflict/incremental_reliability_v1"
)

OUT.mkdir(parents=True, exist_ok=True)


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


def safe_log_loss(y_true, p):
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return float(log_loss(y_true, p))


# ============================================================
# LOAD PAIRED DATA
# ============================================================

print("=" * 88)
print("INCREMENTAL DISAGREEMENT RELIABILITY TEST V1")
print("=" * 88)

print("Fit population:       structure-eligible calibration proteins")
print("Evaluation population: structure-eligible calibration-eval proteins")
print("Protein-disjoint:      YES")
print("TEST USED:             NO")
print()


paired_ids = read_ids(
    BASE / "paired_val_uniprot_ids.txt"
)

y = np.load(
    BASE / "paired_val_labels.npy"
).astype(np.float64)

p_seq = np.load(
    BASE / "sequence_probs.npy"
).astype(np.float64)

p_struct = np.load(
    BASE / "structure_probs.npy"
).astype(np.float64)


cal_ids = read_ids(
    SPLIT / "structure_calibration_ids.txt"
)

eval_ids = read_ids(
    SPLIT / "structure_cal_eval_ids.txt"
)


if y.shape != (4742, 494):
    raise RuntimeError(
        f"Unexpected labels shape: {y.shape}"
    )

if p_seq.shape != y.shape:
    raise RuntimeError(
        "Sequence probability shape mismatch."
    )

if p_struct.shape != y.shape:
    raise RuntimeError(
        "Structure probability shape mismatch."
    )


index = {
    uid: i
    for i, uid in enumerate(paired_ids)
}

cal_rows = np.asarray(
    [index[uid] for uid in cal_ids],
    dtype=np.int64,
)

eval_rows = np.asarray(
    [index[uid] for uid in eval_ids],
    dtype=np.int64,
)


# ============================================================
# FEATURES
#
# U = 0 at maximum confidence near 0/1
# U = 1 at maximum uncertainty near 0.5
#
# D = absolute sequence-structure disagreement
# ============================================================

U_seq = 1.0 - 2.0 * np.abs(
    p_seq - 0.5
)

U_struct = 1.0 - 2.0 * np.abs(
    p_struct - 0.5
)

D = np.abs(
    p_seq - p_struct
)


# Guard against tiny floating-point excursions
U_seq = np.clip(U_seq, 0.0, 1.0)
U_struct = np.clip(U_struct, 0.0, 1.0)
D = np.clip(D, 0.0, 1.0)


# ============================================================
# TARGETS
#
# Primary binary reliability target:
# at least one expert lies on wrong side of 0.5.
#
# Continuous target:
# average absolute probabilistic error of the experts.
# ============================================================

seq_abs_error = np.abs(
    y - p_seq
)

struct_abs_error = np.abs(
    y - p_struct
)

mean_abs_error = (
    seq_abs_error + struct_abs_error
) / 2.0


seq_large_error = (
    seq_abs_error >= 0.5
)

struct_large_error = (
    struct_abs_error >= 0.5
)

either_large_error = (
    seq_large_error
    | struct_large_error
).astype(np.uint8)


# ============================================================
# BUILD CALIBRATION / EVALUATION MATRICES
# ============================================================

def make_features(rows, include_disagreement):

    us = U_seq[rows].reshape(-1)
    ut = U_struct[rows].reshape(-1)

    if include_disagreement:
        d = D[rows].reshape(-1)

        return np.column_stack([
            us,
            ut,
            d,
        ])

    return np.column_stack([
        us,
        ut,
    ])


X0_cal = make_features(
    cal_rows,
    include_disagreement=False,
)

X1_cal = make_features(
    cal_rows,
    include_disagreement=True,
)

X0_eval = make_features(
    eval_rows,
    include_disagreement=False,
)

X1_eval = make_features(
    eval_rows,
    include_disagreement=True,
)


y_binary_cal = (
    either_large_error[
        cal_rows
    ].reshape(-1)
)

y_binary_eval = (
    either_large_error[
        eval_rows
    ].reshape(-1)
)


y_cont_cal = (
    mean_abs_error[
        cal_rows
    ].reshape(-1)
)

y_cont_eval = (
    mean_abs_error[
        eval_rows
    ].reshape(-1)
)


print(
    f"Calibration proteins:        "
    f"{len(cal_rows):,}"
)

print(
    f"Calibration pairs:           "
    f"{len(y_binary_cal):,}"
)

print(
    f"Calibration error events:    "
    f"{int(y_binary_cal.sum()):,} "
    f"({y_binary_cal.mean():.4%})"
)

print()

print(
    f"Evaluation proteins:         "
    f"{len(eval_rows):,}"
)

print(
    f"Evaluation pairs:            "
    f"{len(y_binary_eval):,}"
)

print(
    f"Evaluation error events:     "
    f"{int(y_binary_eval.sum()):,} "
    f"({y_binary_eval.mean():.4%})"
)

print()


# ============================================================
# BINARY LARGE-ERROR MODELS
#
# M0: uncertainty only
# M1: uncertainty + disagreement
#
# No class weighting.
# Same regularization for both models.
# ============================================================

binary_M0 = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
)

binary_M1 = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
)


binary_M0.fit(
    X0_cal,
    y_binary_cal,
)

binary_M1.fit(
    X1_cal,
    y_binary_cal,
)


pred_M0 = binary_M0.predict_proba(
    X0_eval
)[:, 1]

pred_M1 = binary_M1.predict_proba(
    X1_eval
)[:, 1]


binary_metrics_M0 = {
    "roc_auc":
        float(
            roc_auc_score(
                y_binary_eval,
                pred_M0,
            )
        ),

    "average_precision":
        float(
            average_precision_score(
                y_binary_eval,
                pred_M0,
            )
        ),

    "log_loss":
        safe_log_loss(
            y_binary_eval,
            pred_M0,
        ),

    "brier":
        float(
            brier_score_loss(
                y_binary_eval,
                pred_M0,
            )
        ),
}


binary_metrics_M1 = {
    "roc_auc":
        float(
            roc_auc_score(
                y_binary_eval,
                pred_M1,
            )
        ),

    "average_precision":
        float(
            average_precision_score(
                y_binary_eval,
                pred_M1,
            )
        ),

    "log_loss":
        safe_log_loss(
            y_binary_eval,
            pred_M1,
        ),

    "brier":
        float(
            brier_score_loss(
                y_binary_eval,
                pred_M1,
            )
        ),
}


binary_delta = {
    "roc_auc":
        (
            binary_metrics_M1["roc_auc"]
            - binary_metrics_M0["roc_auc"]
        ),

    "average_precision":
        (
            binary_metrics_M1["average_precision"]
            - binary_metrics_M0["average_precision"]
        ),

    # Negative is improvement for loss metrics
    "log_loss":
        (
            binary_metrics_M1["log_loss"]
            - binary_metrics_M0["log_loss"]
        ),

    "brier":
        (
            binary_metrics_M1["brier"]
            - binary_metrics_M0["brier"]
        ),
}


# ============================================================
# CONTINUOUS ERROR MODELS
#
# Ridge chosen as a simple low-capacity diagnostic.
#
# M0: uncertainty only
# M1: uncertainty + disagreement
# ============================================================

continuous_M0 = Ridge(
    alpha=1.0
)

continuous_M1 = Ridge(
    alpha=1.0
)


continuous_M0.fit(
    X0_cal,
    y_cont_cal,
)

continuous_M1.fit(
    X1_cal,
    y_cont_cal,
)


cont_pred_M0 = continuous_M0.predict(
    X0_eval
)

cont_pred_M1 = continuous_M1.predict(
    X1_eval
)


continuous_metrics_M0 = {
    "mae":
        float(
            mean_absolute_error(
                y_cont_eval,
                cont_pred_M0,
            )
        ),

    "rmse":
        float(
            np.sqrt(
                mean_squared_error(
                    y_cont_eval,
                    cont_pred_M0,
                )
            )
        ),

    "correlation":
        float(
            np.corrcoef(
                y_cont_eval,
                cont_pred_M0,
            )[0, 1]
        ),
}


continuous_metrics_M1 = {
    "mae":
        float(
            mean_absolute_error(
                y_cont_eval,
                cont_pred_M1,
            )
        ),

    "rmse":
        float(
            np.sqrt(
                mean_squared_error(
                    y_cont_eval,
                    cont_pred_M1,
                )
            )
        ),

    "correlation":
        float(
            np.corrcoef(
                y_cont_eval,
                cont_pred_M1,
            )[0, 1]
        ),
}


continuous_delta = {
    # Negative = improvement
    "mae":
        (
            continuous_metrics_M1["mae"]
            - continuous_metrics_M0["mae"]
        ),

    # Negative = improvement
    "rmse":
        (
            continuous_metrics_M1["rmse"]
            - continuous_metrics_M0["rmse"]
        ),

    # Positive = improvement
    "correlation":
        (
            continuous_metrics_M1["correlation"]
            - continuous_metrics_M0["correlation"]
        ),
}


# ============================================================
# COEFFICIENTS
# ============================================================

binary_coefficients = {
    "M0_intercept":
        float(binary_M0.intercept_[0]),

    "M0_U_sequence":
        float(binary_M0.coef_[0, 0]),

    "M0_U_structure":
        float(binary_M0.coef_[0, 1]),

    "M1_intercept":
        float(binary_M1.intercept_[0]),

    "M1_U_sequence":
        float(binary_M1.coef_[0, 0]),

    "M1_U_structure":
        float(binary_M1.coef_[0, 1]),

    "M1_disagreement":
        float(binary_M1.coef_[0, 2]),
}


continuous_coefficients = {
    "M0_intercept":
        float(continuous_M0.intercept_),

    "M0_U_sequence":
        float(continuous_M0.coef_[0]),

    "M0_U_structure":
        float(continuous_M0.coef_[1]),

    "M1_intercept":
        float(continuous_M1.intercept_),

    "M1_U_sequence":
        float(continuous_M1.coef_[0]),

    "M1_U_structure":
        float(continuous_M1.coef_[1]),

    "M1_disagreement":
        float(continuous_M1.coef_[2]),
}


# ============================================================
# SAVE EVALUATION PREDICTIONS
# ============================================================

np.save(
    OUT / "binary_eval_target.npy",
    y_binary_eval,
)

np.save(
    OUT / "binary_eval_M0_predictions.npy",
    pred_M0.astype(np.float32),
)

np.save(
    OUT / "binary_eval_M1_predictions.npy",
    pred_M1.astype(np.float32),
)

np.save(
    OUT / "continuous_eval_target.npy",
    y_cont_eval.astype(np.float32),
)

np.save(
    OUT / "continuous_eval_M0_predictions.npy",
    cont_pred_M0.astype(np.float32),
)

np.save(
    OUT / "continuous_eval_M1_predictions.npy",
    cont_pred_M1.astype(np.float32),
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame([
    {
        "endpoint":
            "either_expert_large_error",

        "metric":
            "ROC_AUC",

        "baseline_M0":
            binary_metrics_M0["roc_auc"],

        "plus_disagreement_M1":
            binary_metrics_M1["roc_auc"],

        "delta_M1_minus_M0":
            binary_delta["roc_auc"],
    },
    {
        "endpoint":
            "either_expert_large_error",

        "metric":
            "Average_Precision",

        "baseline_M0":
            binary_metrics_M0[
                "average_precision"
            ],

        "plus_disagreement_M1":
            binary_metrics_M1[
                "average_precision"
            ],

        "delta_M1_minus_M0":
            binary_delta[
                "average_precision"
            ],
    },
    {
        "endpoint":
            "either_expert_large_error",

        "metric":
            "Log_Loss",

        "baseline_M0":
            binary_metrics_M0["log_loss"],

        "plus_disagreement_M1":
            binary_metrics_M1["log_loss"],

        "delta_M1_minus_M0":
            binary_delta["log_loss"],
    },
    {
        "endpoint":
            "either_expert_large_error",

        "metric":
            "Brier",

        "baseline_M0":
            binary_metrics_M0["brier"],

        "plus_disagreement_M1":
            binary_metrics_M1["brier"],

        "delta_M1_minus_M0":
            binary_delta["brier"],
    },
    {
        "endpoint":
            "mean_absolute_expert_error",

        "metric":
            "MAE",

        "baseline_M0":
            continuous_metrics_M0["mae"],

        "plus_disagreement_M1":
            continuous_metrics_M1["mae"],

        "delta_M1_minus_M0":
            continuous_delta["mae"],
    },
    {
        "endpoint":
            "mean_absolute_expert_error",

        "metric":
            "RMSE",

        "baseline_M0":
            continuous_metrics_M0["rmse"],

        "plus_disagreement_M1":
            continuous_metrics_M1["rmse"],

        "delta_M1_minus_M0":
            continuous_delta["rmse"],
    },
    {
        "endpoint":
            "mean_absolute_expert_error",

        "metric":
            "Correlation",

        "baseline_M0":
            continuous_metrics_M0[
                "correlation"
            ],

        "plus_disagreement_M1":
            continuous_metrics_M1[
                "correlation"
            ],

        "delta_M1_minus_M0":
            continuous_delta[
                "correlation"
            ],
    },
])


comparison.to_csv(
    OUT / "heldout_model_comparison.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "incremental_disagreement_reliability_v1",

    "fit_proteins":
        len(cal_rows),

    "evaluation_proteins":
        len(eval_rows),

    "fit_pairs":
        len(y_binary_cal),

    "evaluation_pairs":
        len(y_binary_eval),

    "evaluation_large_error_events":
        int(y_binary_eval.sum()),

    "evaluation_large_error_prevalence":
        float(y_binary_eval.mean()),

    "baseline_features": [
        "sequence_uncertainty",
        "structure_uncertainty",
    ],

    "augmented_features": [
        "sequence_uncertainty",
        "structure_uncertainty",
        "absolute_sequence_structure_disagreement",
    ],

    "binary_endpoint":
        "either expert absolute error >= 0.5",

    "continuous_endpoint":
        "mean absolute probabilistic error of sequence and structure experts",

    "binary_M0":
        binary_metrics_M0,

    "binary_M1":
        binary_metrics_M1,

    "binary_delta_M1_minus_M0":
        binary_delta,

    "binary_coefficients":
        binary_coefficients,

    "continuous_M0":
        continuous_metrics_M0,

    "continuous_M1":
        continuous_metrics_M1,

    "continuous_delta_M1_minus_M0":
        continuous_delta,

    "continuous_coefficients":
        continuous_coefficients,

    "protein_disjoint_fit_evaluation":
        True,

    "conflict_threshold_optimized":
        False,

    "plddt_used":
        False,

    "test_used":
        False,

    "interpretation_policy":
        (
            "This is a development-stage held-out "
            "incremental-value analysis within the "
            "validation partition. It is not the "
            "final locked-test confirmation."
        ),
}


with open(
    OUT / "incremental_reliability_summary.json",
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

print("=" * 88)
print("BINARY ENDPOINT: EITHER EXPERT LARGE ERROR")
print("=" * 88)

print(
    f"M0 ROC-AUC:              "
    f"{binary_metrics_M0['roc_auc']:.6f}"
)

print(
    f"M1 ROC-AUC:              "
    f"{binary_metrics_M1['roc_auc']:.6f}"
)

print(
    f"Delta:                   "
    f"{binary_delta['roc_auc']:+.6f}"
)

print()

print(
    f"M0 Average Precision:    "
    f"{binary_metrics_M0['average_precision']:.6f}"
)

print(
    f"M1 Average Precision:    "
    f"{binary_metrics_M1['average_precision']:.6f}"
)

print(
    f"Delta:                   "
    f"{binary_delta['average_precision']:+.6f}"
)

print()

print(
    f"M0 Log Loss:             "
    f"{binary_metrics_M0['log_loss']:.6f}"
)

print(
    f"M1 Log Loss:             "
    f"{binary_metrics_M1['log_loss']:.6f}"
)

print(
    f"Delta (negative better): "
    f"{binary_delta['log_loss']:+.6f}"
)

print()

print(
    f"M0 Brier:                "
    f"{binary_metrics_M0['brier']:.6f}"
)

print(
    f"M1 Brier:                "
    f"{binary_metrics_M1['brier']:.6f}"
)

print(
    f"Delta (negative better): "
    f"{binary_delta['brier']:+.6f}"
)

print()

print(
    "M1 disagreement coefficient: "
    f"{binary_coefficients['M1_disagreement']:.6f}"
)


print()
print("=" * 88)
print("CONTINUOUS ENDPOINT: MEAN ABSOLUTE EXPERT ERROR")
print("=" * 88)

print(
    f"M0 MAE:                  "
    f"{continuous_metrics_M0['mae']:.6f}"
)

print(
    f"M1 MAE:                  "
    f"{continuous_metrics_M1['mae']:.6f}"
)

print(
    f"Delta (negative better): "
    f"{continuous_delta['mae']:+.6f}"
)

print()

print(
    f"M0 RMSE:                 "
    f"{continuous_metrics_M0['rmse']:.6f}"
)

print(
    f"M1 RMSE:                 "
    f"{continuous_metrics_M1['rmse']:.6f}"
)

print(
    f"Delta (negative better): "
    f"{continuous_delta['rmse']:+.6f}"
)

print()

print(
    f"M0 correlation:          "
    f"{continuous_metrics_M0['correlation']:.6f}"
)

print(
    f"M1 correlation:          "
    f"{continuous_metrics_M1['correlation']:.6f}"
)

print(
    f"Delta (positive better): "
    f"{continuous_delta['correlation']:+.6f}"
)

print()

print(
    "M1 disagreement coefficient: "
    f"{continuous_coefficients['M1_disagreement']:.6f}"
)


print()
print("=" * 88)
print("INTERPRETATION GUARDRAILS")
print("=" * 88)

print("Held-out within VAL:                  YES")
print("Protein-disjoint fit/evaluation:      YES")
print("Disagreement added to baseline:       YES")
print("Conflict threshold optimized:         NO")
print("pLDDT used:                           NO")
print("Final TEST confirmation performed:    NO")
print("TEST USED:                            NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 88)