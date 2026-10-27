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

BASE = Path(
    "data/conflict/paired_experts_v1"
)

SPLIT = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

OUT = Path(
    "data/conflict/"
    "separate_expert_reliability_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


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


def binary_metrics(y, p):
    return {
        "roc_auc":
            float(
                roc_auc_score(y, p)
            ),

        "average_precision":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y,
                    p,
                    labels=[0, 1],
                )
            ),

        "brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),
    }


def continuous_metrics(y, pred):
    return {
        "mae":
            float(
                mean_absolute_error(
                    y,
                    pred,
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y,
                        pred,
                    )
                )
            ),

        "correlation":
            float(
                np.corrcoef(
                    y,
                    pred,
                )[0, 1]
            ),
    }


# ============================================================
# LOAD PAIRED EXPERT PREDICTIONS
# ============================================================

print("=" * 96)
print("SEPARATE-EXPERT INCREMENTAL DISAGREEMENT RELIABILITY V1")
print("=" * 96)

print("Fit population:        structure-eligible calibration proteins")
print("Evaluation population: structure-eligible calibration-eval proteins")
print("Sequence endpoint:     sequence expert error")
print("Structure endpoint:    structure expert error")
print("Disagreement:          |p_sequence - p_structure|")
print("pLDDT used:            NO")
print("Threshold optimized:   NO")
print("TEST USED:             NO")
print()


paired_ids = read_ids(
    BASE /
    "paired_val_uniprot_ids.txt"
)

y_all = np.load(
    BASE /
    "paired_val_labels.npy"
).astype(np.float64)

p_seq_all = np.load(
    BASE /
    "sequence_probs.npy"
).astype(np.float64)

p_struct_all = np.load(
    BASE /
    "structure_probs.npy"
).astype(np.float64)


if not (
    y_all.shape
    == p_seq_all.shape
    == p_struct_all.shape
):
    raise RuntimeError(
        "Paired array shapes do not match."
    )


if y_all.shape[1] != 494:
    raise RuntimeError(
        "Expected 494 labels."
    )


index = {
    uid: i
    for i, uid in enumerate(paired_ids)
}


# ============================================================
# CALIBRATION / EVALUATION SPLITS
# ============================================================

cal_ids = read_ids(
    SPLIT /
    "structure_calibration_ids.txt"
)

eval_ids = read_ids(
    SPLIT /
    "structure_cal_eval_ids.txt"
)


if set(cal_ids) & set(eval_ids):
    raise RuntimeError(
        "Calibration/evaluation overlap detected."
    )


cal_rows = np.asarray(
    [index[x] for x in cal_ids],
    dtype=np.int64,
)

eval_rows = np.asarray(
    [index[x] for x in eval_ids],
    dtype=np.int64,
)


y_cal = y_all[cal_rows]
seq_cal = p_seq_all[cal_rows]
struct_cal = p_struct_all[cal_rows]

y_eval = y_all[eval_rows]
seq_eval = p_seq_all[eval_rows]
struct_eval = p_struct_all[eval_rows]


print(
    f"Calibration proteins: {len(cal_ids):,}"
)

print(
    f"Evaluation proteins:  {len(eval_ids):,}"
)

print(
    f"Calibration pairs:    {y_cal.size:,}"
)

print(
    f"Evaluation pairs:     {y_eval.size:,}"
)

print()


# ============================================================
# FEATURES
#
# U = maximum at p=.5, minimum at p=0/1.
#
# M0 for each expert:
#     own uncertainty only
#
# M1:
#     own uncertainty
#     other expert uncertainty
#     explicit disagreement D
#
# This directly tests incremental utility of disagreement
# beyond individual uncertainty summaries.
# ============================================================

def uncertainty(p):
    return (
        1.0
        - 2.0 * np.abs(
            p - 0.5
        )
    )


U_seq_cal = uncertainty(
    seq_cal
)

U_struct_cal = uncertainty(
    struct_cal
)

D_cal = np.abs(
    seq_cal - struct_cal
)


U_seq_eval = uncertainty(
    seq_eval
)

U_struct_eval = uncertainty(
    struct_eval
)

D_eval = np.abs(
    seq_eval - struct_eval
)


# Flatten label pairs
U_seq_cal_f = U_seq_cal.reshape(-1)
U_struct_cal_f = U_struct_cal.reshape(-1)
D_cal_f = D_cal.reshape(-1)

U_seq_eval_f = U_seq_eval.reshape(-1)
U_struct_eval_f = U_struct_eval.reshape(-1)
D_eval_f = D_eval.reshape(-1)


# ============================================================
# TARGETS
# ============================================================

seq_abs_cal = np.abs(
    y_cal - seq_cal
).reshape(-1)

struct_abs_cal = np.abs(
    y_cal - struct_cal
).reshape(-1)


seq_abs_eval = np.abs(
    y_eval - seq_eval
).reshape(-1)

struct_abs_eval = np.abs(
    y_eval - struct_eval
).reshape(-1)


# Fixed diagnostic definition.
# NOT optimized.
seq_binary_cal = (
    seq_abs_cal >= 0.5
).astype(np.uint8)

struct_binary_cal = (
    struct_abs_cal >= 0.5
).astype(np.uint8)


seq_binary_eval = (
    seq_abs_eval >= 0.5
).astype(np.uint8)

struct_binary_eval = (
    struct_abs_eval >= 0.5
).astype(np.uint8)


# ============================================================
# MODEL FUNCTION
# ============================================================

def run_expert(
    expert_name,
    own_U_cal,
    other_U_cal,
    own_U_eval,
    other_U_eval,
    binary_cal,
    binary_eval,
    continuous_cal,
    continuous_eval,
):

    print("=" * 96)
    print(
        f"{expert_name.upper()} EXPERT"
    )
    print("=" * 96)


    # --------------------------------------------------------
    # Binary models
    # --------------------------------------------------------

    X0_cal = own_U_cal.reshape(
        -1, 1
    )

    X0_eval = own_U_eval.reshape(
        -1, 1
    )


    X1_cal = np.column_stack([
        own_U_cal,
        other_U_cal,
        D_cal_f,
    ])

    X1_eval = np.column_stack([
        own_U_eval,
        other_U_eval,
        D_eval_f,
    ])


    binary_M0 = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    binary_M1 = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )


    binary_M0.fit(
        X0_cal,
        binary_cal,
    )

    binary_M1.fit(
        X1_cal,
        binary_cal,
    )


    pred0 = binary_M0.predict_proba(
        X0_eval
    )[:, 1]

    pred1 = binary_M1.predict_proba(
        X1_eval
    )[:, 1]


    bm0 = binary_metrics(
        binary_eval,
        pred0,
    )

    bm1 = binary_metrics(
        binary_eval,
        pred1,
    )


    # --------------------------------------------------------
    # Continuous absolute-error models
    # --------------------------------------------------------

    cont_M0 = Ridge(
        alpha=1.0
    )

    cont_M1 = Ridge(
        alpha=1.0
    )


    cont_M0.fit(
        X0_cal,
        continuous_cal,
    )

    cont_M1.fit(
        X1_cal,
        continuous_cal,
    )


    cpred0 = cont_M0.predict(
        X0_eval
    )

    cpred1 = cont_M1.predict(
        X1_eval
    )


    cm0 = continuous_metrics(
        continuous_eval,
        cpred0,
    )

    cm1 = continuous_metrics(
        continuous_eval,
        cpred1,
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    np.save(
        OUT /
        f"{expert_name}_binary_eval_M0.npy",
        pred0.astype(np.float32),
    )

    np.save(
        OUT /
        f"{expert_name}_binary_eval_M1.npy",
        pred1.astype(np.float32),
    )

    np.save(
        OUT /
        f"{expert_name}_binary_eval_target.npy",
        binary_eval,
    )

    np.save(
        OUT /
        f"{expert_name}_continuous_eval_M0.npy",
        cpred0.astype(np.float32),
    )

    np.save(
        OUT /
        f"{expert_name}_continuous_eval_M1.npy",
        cpred1.astype(np.float32),
    )

    np.save(
        OUT /
        f"{expert_name}_continuous_eval_target.npy",
        continuous_eval.astype(
            np.float32
        ),
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        f"Calibration error events: "
        f"{int(binary_cal.sum()):,} "
        f"({binary_cal.mean():.4%})"
    )

    print(
        f"Evaluation error events:  "
        f"{int(binary_eval.sum()):,} "
        f"({binary_eval.mean():.4%})"
    )

    print()

    print("BINARY ERROR PREDICTION")

    print(
        f"  M0 ROC-AUC:  "
        f"{bm0['roc_auc']:.6f}"
    )

    print(
        f"  M1 ROC-AUC:  "
        f"{bm1['roc_auc']:.6f}"
    )

    print(
        f"  Delta:       "
        f"{bm1['roc_auc'] - bm0['roc_auc']:+.6f}"
    )

    print()

    print(
        f"  M0 AP:       "
        f"{bm0['average_precision']:.6f}"
    )

    print(
        f"  M1 AP:       "
        f"{bm1['average_precision']:.6f}"
    )

    print(
        f"  Delta:       "
        f"{bm1['average_precision'] - bm0['average_precision']:+.6f}"
    )

    print()

    print(
        f"  M0 logloss:  "
        f"{bm0['log_loss']:.6f}"
    )

    print(
        f"  M1 logloss:  "
        f"{bm1['log_loss']:.6f}"
    )

    print(
        f"  Delta:       "
        f"{bm1['log_loss'] - bm0['log_loss']:+.6f}"
    )

    print()

    print(
        f"  M0 Brier:    "
        f"{bm0['brier']:.6f}"
    )

    print(
        f"  M1 Brier:    "
        f"{bm1['brier']:.6f}"
    )

    print(
        f"  Delta:       "
        f"{bm1['brier'] - bm0['brier']:+.6f}"
    )

    print()

    print(
        "  M1 coefficients "
        "[own_U, other_U, D]:"
    )

    print(
        " ",
        [
            float(x)
            for x in
            binary_M1.coef_[0]
        ]
    )

    print()

    print("CONTINUOUS ABSOLUTE ERROR")

    print(
        f"  M0 MAE:      "
        f"{cm0['mae']:.6f}"
    )

    print(
        f"  M1 MAE:      "
        f"{cm1['mae']:.6f}"
    )

    print(
        f"  Delta:       "
        f"{cm1['mae'] - cm0['mae']:+.6f}"
    )

    print()

    print(
        f"  M0 RMSE:     "
        f"{cm0['rmse']:.6f}"
    )

    print(
        f"  M1 RMSE:     "
        f"{cm1['rmse']:.6f}"
    )

    print(
        f"  Delta:       "
        f"{cm1['rmse'] - cm0['rmse']:+.6f}"
    )

    print()

    print(
        f"  M0 corr:     "
        f"{cm0['correlation']:.6f}"
    )

    print(
        f"  M1 corr:     "
        f"{cm1['correlation']:.6f}"
    )

    print(
        f"  Delta:       "
        f"{cm1['correlation'] - cm0['correlation']:+.6f}"
    )

    print()

    print(
        "  M1 coefficients "
        "[own_U, other_U, D]:"
    )

    print(
        " ",
        [
            float(x)
            for x in
            cont_M1.coef_
        ]
    )

    print()


    result = {
        "expert":
            expert_name,

        "calibration_error_events":
            int(binary_cal.sum()),

        "calibration_error_prevalence":
            float(binary_cal.mean()),

        "evaluation_error_events":
            int(binary_eval.sum()),

        "evaluation_error_prevalence":
            float(binary_eval.mean()),

        "binary_M0":
            bm0,

        "binary_M1":
            bm1,

        "binary_delta_M1_minus_M0": {
            key:
                bm1[key] - bm0[key]
            for key in bm0
        },

        "binary_M1_coefficients": {
            "own_uncertainty":
                float(
                    binary_M1.coef_[0][0]
                ),

            "other_uncertainty":
                float(
                    binary_M1.coef_[0][1]
                ),

            "disagreement":
                float(
                    binary_M1.coef_[0][2]
                ),
        },

        "continuous_M0":
            cm0,

        "continuous_M1":
            cm1,

        "continuous_delta_M1_minus_M0": {
            key:
                cm1[key] - cm0[key]
            for key in cm0
        },

        "continuous_M1_coefficients": {
            "own_uncertainty":
                float(
                    cont_M1.coef_[0]
                ),

            "other_uncertainty":
                float(
                    cont_M1.coef_[1]
                ),

            "disagreement":
                float(
                    cont_M1.coef_[2]
                ),
        },
    }

    return result


# ============================================================
# SEQUENCE EXPERT
# ============================================================

sequence_result = run_expert(
    expert_name="sequence",

    own_U_cal=
        U_seq_cal_f,

    other_U_cal=
        U_struct_cal_f,

    own_U_eval=
        U_seq_eval_f,

    other_U_eval=
        U_struct_eval_f,

    binary_cal=
        seq_binary_cal,

    binary_eval=
        seq_binary_eval,

    continuous_cal=
        seq_abs_cal,

    continuous_eval=
        seq_abs_eval,
)


# ============================================================
# STRUCTURE EXPERT
# ============================================================

structure_result = run_expert(
    expert_name="structure",

    own_U_cal=
        U_struct_cal_f,

    other_U_cal=
        U_seq_cal_f,

    own_U_eval=
        U_struct_eval_f,

    other_U_eval=
        U_seq_eval_f,

    binary_cal=
        struct_binary_cal,

    binary_eval=
        struct_binary_eval,

    continuous_cal=
        struct_abs_cal,

    continuous_eval=
        struct_abs_eval,
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_rows = []


for result in [
    sequence_result,
    structure_result,
]:

    expert = result["expert"]

    comparison_rows.append({
        "expert":
            expert,

        "binary_event_prevalence":
            result[
                "evaluation_error_prevalence"
            ],

        "M0_ROC_AUC":
            result[
                "binary_M0"
            ]["roc_auc"],

        "M1_ROC_AUC":
            result[
                "binary_M1"
            ]["roc_auc"],

        "delta_ROC_AUC":
            result[
                "binary_delta_M1_minus_M0"
            ]["roc_auc"],

        "M0_AP":
            result[
                "binary_M0"
            ]["average_precision"],

        "M1_AP":
            result[
                "binary_M1"
            ]["average_precision"],

        "delta_AP":
            result[
                "binary_delta_M1_minus_M0"
            ]["average_precision"],

        "M0_logloss":
            result[
                "binary_M0"
            ]["log_loss"],

        "M1_logloss":
            result[
                "binary_M1"
            ]["log_loss"],

        "delta_logloss":
            result[
                "binary_delta_M1_minus_M0"
            ]["log_loss"],

        "M0_brier":
            result[
                "binary_M0"
            ]["brier"],

        "M1_brier":
            result[
                "binary_M1"
            ]["brier"],

        "delta_brier":
            result[
                "binary_delta_M1_minus_M0"
            ]["brier"],

        "M1_D_binary_coefficient":
            result[
                "binary_M1_coefficients"
            ]["disagreement"],

        "M0_continuous_MAE":
            result[
                "continuous_M0"
            ]["mae"],

        "M1_continuous_MAE":
            result[
                "continuous_M1"
            ]["mae"],

        "delta_continuous_MAE":
            result[
                "continuous_delta_M1_minus_M0"
            ]["mae"],

        "M0_continuous_RMSE":
            result[
                "continuous_M0"
            ]["rmse"],

        "M1_continuous_RMSE":
            result[
                "continuous_M1"
            ]["rmse"],

        "delta_continuous_RMSE":
            result[
                "continuous_delta_M1_minus_M0"
            ]["rmse"],

        "M0_continuous_corr":
            result[
                "continuous_M0"
            ]["correlation"],

        "M1_continuous_corr":
            result[
                "continuous_M1"
            ]["correlation"],

        "delta_continuous_corr":
            result[
                "continuous_delta_M1_minus_M0"
            ]["correlation"],

        "M1_D_continuous_coefficient":
            result[
                "continuous_M1_coefficients"
            ]["disagreement"],
    })


comparison = pd.DataFrame(
    comparison_rows
)

comparison.to_csv(
    OUT /
    "separate_expert_comparison.csv",
    index=False,
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "analysis":
        "separate_expert_incremental_reliability_v1",

    "fit_population":
        "structure-eligible calibration",

    "evaluation_population":
        "structure-eligible calibration-eval",

    "calibration_proteins":
        len(cal_ids),

    "evaluation_proteins":
        len(eval_ids),

    "labels":
        494,

    "M0":
        "own expert uncertainty only",

    "M1":
        (
            "own uncertainty + other expert uncertainty "
            "+ absolute disagreement"
        ),

    "binary_endpoint":
        "expert-specific absolute error >= 0.5",

    "continuous_endpoint":
        "expert-specific absolute probabilistic error",

    "sequence":
        sequence_result,

    "structure":
        structure_result,

    "threshold_optimized":
        False,

    "plddt_used":
        False,

    "test_used":
        False,
}


with open(
    OUT /
    "separate_expert_reliability_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print("=" * 96)
print("FINAL STATUS")
print("=" * 96)

print(
    "Separate sequence-error endpoint:   YES"
)

print(
    "Separate structure-error endpoint:  YES"
)

print(
    "Models fit on calibration subset:   YES"
)

print(
    "Evaluated on cal-eval subset:       YES"
)

print(
    "pLDDT used:                         NO"
)

print(
    "Threshold optimized:                NO"
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

print("=" * 96)