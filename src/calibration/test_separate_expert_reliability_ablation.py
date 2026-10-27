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
    "data/conflict/"
    "separate_expert_reliability_ablation_v1"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def uncertainty(p):
    return 1.0 - 2.0 * np.abs(p - 0.5)


def binary_metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(
            average_precision_score(y, p)
        ),
        "log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
    }


def continuous_metrics(y, pred):
    return {
        "mae": float(
            mean_absolute_error(y, pred)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y, pred)
            )
        ),
        "correlation": float(
            np.corrcoef(y, pred)[0, 1]
        ),
    }


def delta(a, b):
    """
    Return metric-wise a - b.
    """
    return {
        key: float(a[key] - b[key])
        for key in a
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("SEPARATE-EXPERT RELIABILITY ABLATION V1")
print("=" * 100)

print("M0: own uncertainty")
print("MA: own uncertainty + other-expert uncertainty")
print("MB: own uncertainty + other-expert uncertainty + disagreement")
print()
print("Primary ablation: MB versus MA")
print("Disagreement:     D = |p_sequence - p_structure|")
print("pLDDT used:       NO")
print("Threshold optimized: NO")
print("TEST USED:        NO")
print()


paired_ids = read_ids(
    BASE / "paired_val_uniprot_ids.txt"
)

y_all = np.load(
    BASE / "paired_val_labels.npy"
).astype(np.float64)

p_seq_all = np.load(
    BASE / "sequence_probs.npy"
).astype(np.float64)

p_struct_all = np.load(
    BASE / "structure_probs.npy"
).astype(np.float64)


if not (
    y_all.shape
    == p_seq_all.shape
    == p_struct_all.shape
):
    raise RuntimeError(
        "Paired arrays have inconsistent shapes."
    )


if y_all.shape[1] != 494:
    raise RuntimeError(
        f"Expected 494 labels, got {y_all.shape[1]}"
    )


paired_index = {
    uid: i
    for i, uid in enumerate(paired_ids)
}


cal_ids = read_ids(
    SPLIT / "structure_calibration_ids.txt"
)

eval_ids = read_ids(
    SPLIT / "structure_cal_eval_ids.txt"
)


if set(cal_ids) & set(eval_ids):
    raise RuntimeError(
        "Calibration/evaluation overlap detected."
    )


missing_cal = [
    uid for uid in cal_ids
    if uid not in paired_index
]

missing_eval = [
    uid for uid in eval_ids
    if uid not in paired_index
]


if missing_cal or missing_eval:
    raise RuntimeError(
        "Some split IDs are missing from paired predictions."
    )


cal_rows = np.asarray(
    [paired_index[x] for x in cal_ids],
    dtype=np.int64,
)

eval_rows = np.asarray(
    [paired_index[x] for x in eval_ids],
    dtype=np.int64,
)


y_cal = y_all[cal_rows]
y_eval = y_all[eval_rows]

seq_cal = p_seq_all[cal_rows]
seq_eval = p_seq_all[eval_rows]

struct_cal = p_struct_all[cal_rows]
struct_eval = p_struct_all[eval_rows]


print(f"Calibration proteins: {len(cal_ids):,}")
print(f"Evaluation proteins:  {len(eval_ids):,}")
print(f"Calibration pairs:    {y_cal.size:,}")
print(f"Evaluation pairs:     {y_eval.size:,}")
print()


# ============================================================
# FEATURES
# ============================================================

U_seq_cal = uncertainty(seq_cal).reshape(-1)
U_seq_eval = uncertainty(seq_eval).reshape(-1)

U_struct_cal = uncertainty(struct_cal).reshape(-1)
U_struct_eval = uncertainty(struct_eval).reshape(-1)

D_cal = np.abs(
    seq_cal - struct_cal
).reshape(-1)

D_eval = np.abs(
    seq_eval - struct_eval
).reshape(-1)


# ============================================================
# EXPERT-SPECIFIC TARGETS
# ============================================================

seq_abs_cal = np.abs(
    y_cal - seq_cal
).reshape(-1)

seq_abs_eval = np.abs(
    y_eval - seq_eval
).reshape(-1)

struct_abs_cal = np.abs(
    y_cal - struct_cal
).reshape(-1)

struct_abs_eval = np.abs(
    y_eval - struct_eval
).reshape(-1)


seq_binary_cal = (
    seq_abs_cal >= 0.5
).astype(np.uint8)

seq_binary_eval = (
    seq_abs_eval >= 0.5
).astype(np.uint8)

struct_binary_cal = (
    struct_abs_cal >= 0.5
).astype(np.uint8)

struct_binary_eval = (
    struct_abs_eval >= 0.5
).astype(np.uint8)


# ============================================================
# RUN ONE EXPERT
# ============================================================

def run_expert(
    name,
    own_cal,
    other_cal,
    own_eval,
    other_eval,
    binary_cal,
    binary_eval,
    continuous_cal,
    continuous_eval,
):

    print("=" * 100)
    print(f"{name.upper()} EXPERT")
    print("=" * 100)


    # --------------------------------------------------------
    # Three nested feature sets
    # --------------------------------------------------------

    X0_cal = own_cal.reshape(-1, 1)
    X0_eval = own_eval.reshape(-1, 1)

    XA_cal = np.column_stack([
        own_cal,
        other_cal,
    ])

    XA_eval = np.column_stack([
        own_eval,
        other_eval,
    ])

    XB_cal = np.column_stack([
        own_cal,
        other_cal,
        D_cal,
    ])

    XB_eval = np.column_stack([
        own_eval,
        other_eval,
        D_eval,
    ])


    # --------------------------------------------------------
    # Binary endpoint
    # --------------------------------------------------------

    binary_models = {
        "M0": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
        "MA": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
        "MB": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
    }


    binary_X_cal = {
        "M0": X0_cal,
        "MA": XA_cal,
        "MB": XB_cal,
    }

    binary_X_eval = {
        "M0": X0_eval,
        "MA": XA_eval,
        "MB": XB_eval,
    }


    binary_results = {}
    binary_predictions = {}


    for model_name in ["M0", "MA", "MB"]:

        model = binary_models[model_name]

        model.fit(
            binary_X_cal[model_name],
            binary_cal,
        )

        pred = model.predict_proba(
            binary_X_eval[model_name]
        )[:, 1]

        binary_predictions[model_name] = pred

        binary_results[model_name] = (
            binary_metrics(
                binary_eval,
                pred,
            )
        )


    # --------------------------------------------------------
    # Continuous endpoint
    # --------------------------------------------------------

    continuous_models = {
        "M0": Ridge(alpha=1.0),
        "MA": Ridge(alpha=1.0),
        "MB": Ridge(alpha=1.0),
    }


    continuous_X_cal = {
        "M0": X0_cal,
        "MA": XA_cal,
        "MB": XB_cal,
    }

    continuous_X_eval = {
        "M0": X0_eval,
        "MA": XA_eval,
        "MB": XB_eval,
    }


    continuous_results = {}
    continuous_predictions = {}


    for model_name in ["M0", "MA", "MB"]:

        model = continuous_models[model_name]

        model.fit(
            continuous_X_cal[model_name],
            continuous_cal,
        )

        pred = model.predict(
            continuous_X_eval[model_name]
        )

        continuous_predictions[
            model_name
        ] = pred

        continuous_results[
            model_name
        ] = continuous_metrics(
            continuous_eval,
            pred,
        )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    for model_name in ["M0", "MA", "MB"]:

        np.save(
            OUT /
            f"{name}_binary_{model_name}_predictions.npy",
            binary_predictions[
                model_name
            ].astype(np.float32),
        )

        np.save(
            OUT /
            f"{name}_continuous_{model_name}_predictions.npy",
            continuous_predictions[
                model_name
            ].astype(np.float32),
        )


    np.save(
        OUT /
        f"{name}_binary_target.npy",
        binary_eval,
    )

    np.save(
        OUT /
        f"{name}_continuous_target.npy",
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
    print()

    for metric in [
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier",
    ]:

        m0 = binary_results["M0"][metric]
        ma = binary_results["MA"][metric]
        mb = binary_results["MB"][metric]

        print(
            f"  {metric:<18} "
            f"M0={m0:.6f} | "
            f"MA={ma:.6f} | "
            f"MB={mb:.6f}"
        )

        print(
            f"  {'MB - MA':<18} "
            f"{mb - ma:+.6f}"
        )

        print()


    print(
        "  MA coefficients "
        "[own_U, other_U]:"
    )

    print(
        " ",
        [
            float(x)
            for x in
            binary_models[
                "MA"
            ].coef_[0]
        ]
    )

    print()

    print(
        "  MB coefficients "
        "[own_U, other_U, D]:"
    )

    print(
        " ",
        [
            float(x)
            for x in
            binary_models[
                "MB"
            ].coef_[0]
        ]
    )

    print()
    print("CONTINUOUS ABSOLUTE ERROR")
    print()

    for metric in [
        "mae",
        "rmse",
        "correlation",
    ]:

        m0 = continuous_results[
            "M0"
        ][metric]

        ma = continuous_results[
            "MA"
        ][metric]

        mb = continuous_results[
            "MB"
        ][metric]

        print(
            f"  {metric:<18} "
            f"M0={m0:.6f} | "
            f"MA={ma:.6f} | "
            f"MB={mb:.6f}"
        )

        print(
            f"  {'MB - MA':<18} "
            f"{mb - ma:+.6f}"
        )

        print()


    print(
        "  MA coefficients "
        "[own_U, other_U]:"
    )

    print(
        " ",
        [
            float(x)
            for x in
            continuous_models[
                "MA"
            ].coef_
        ]
    )

    print()

    print(
        "  MB coefficients "
        "[own_U, other_U, D]:"
    )

    print(
        " ",
        [
            float(x)
            for x in
            continuous_models[
                "MB"
            ].coef_
        ]
    )

    print()


    result = {
        "expert": name,

        "calibration_error_events":
            int(binary_cal.sum()),

        "calibration_error_prevalence":
            float(binary_cal.mean()),

        "evaluation_error_events":
            int(binary_eval.sum()),

        "evaluation_error_prevalence":
            float(binary_eval.mean()),

        "binary": binary_results,

        "binary_delta_MB_minus_MA":
            delta(
                binary_results["MB"],
                binary_results["MA"],
            ),

        "continuous":
            continuous_results,

        "continuous_delta_MB_minus_MA":
            delta(
                continuous_results["MB"],
                continuous_results["MA"],
            ),

        "binary_MA_coefficients":
            [
                float(x)
                for x in
                binary_models["MA"].coef_[0]
            ],

        "binary_MB_coefficients":
            [
                float(x)
                for x in
                binary_models["MB"].coef_[0]
            ],

        "continuous_MA_coefficients":
            [
                float(x)
                for x in
                continuous_models["MA"].coef_
            ],

        "continuous_MB_coefficients":
            [
                float(x)
                for x in
                continuous_models["MB"].coef_
            ],
    }

    return result


# ============================================================
# SEQUENCE
# ============================================================

sequence_result = run_expert(
    name="sequence",

    own_cal=U_seq_cal,
    other_cal=U_struct_cal,

    own_eval=U_seq_eval,
    other_eval=U_struct_eval,

    binary_cal=seq_binary_cal,
    binary_eval=seq_binary_eval,

    continuous_cal=seq_abs_cal,
    continuous_eval=seq_abs_eval,
)


# ============================================================
# STRUCTURE
# ============================================================

structure_result = run_expert(
    name="structure",

    own_cal=U_struct_cal,
    other_cal=U_seq_cal,

    own_eval=U_struct_eval,
    other_eval=U_seq_eval,

    binary_cal=struct_binary_cal,
    binary_eval=struct_binary_eval,

    continuous_cal=struct_abs_cal,
    continuous_eval=struct_abs_eval,
)


# ============================================================
# COMPARISON TABLE
# ============================================================

rows = []

for result in [
    sequence_result,
    structure_result,
]:

    name = result["expert"]

    b = result["binary"]
    c = result["continuous"]

    rows.append({
        "expert": name,

        "event_prevalence":
            result[
                "evaluation_error_prevalence"
            ],

        "M0_ROC_AUC":
            b["M0"]["roc_auc"],

        "MA_ROC_AUC":
            b["MA"]["roc_auc"],

        "MB_ROC_AUC":
            b["MB"]["roc_auc"],

        "delta_MB_MA_ROC_AUC":
            b["MB"]["roc_auc"]
            - b["MA"]["roc_auc"],

        "M0_AP":
            b["M0"]["average_precision"],

        "MA_AP":
            b["MA"]["average_precision"],

        "MB_AP":
            b["MB"]["average_precision"],

        "delta_MB_MA_AP":
            b["MB"]["average_precision"]
            - b["MA"]["average_precision"],

        "M0_logloss":
            b["M0"]["log_loss"],

        "MA_logloss":
            b["MA"]["log_loss"],

        "MB_logloss":
            b["MB"]["log_loss"],

        "delta_MB_MA_logloss":
            b["MB"]["log_loss"]
            - b["MA"]["log_loss"],

        "M0_brier":
            b["M0"]["brier"],

        "MA_brier":
            b["MA"]["brier"],

        "MB_brier":
            b["MB"]["brier"],

        "delta_MB_MA_brier":
            b["MB"]["brier"]
            - b["MA"]["brier"],

        "M0_MAE":
            c["M0"]["mae"],

        "MA_MAE":
            c["MA"]["mae"],

        "MB_MAE":
            c["MB"]["mae"],

        "delta_MB_MA_MAE":
            c["MB"]["mae"]
            - c["MA"]["mae"],

        "M0_RMSE":
            c["M0"]["rmse"],

        "MA_RMSE":
            c["MA"]["rmse"],

        "MB_RMSE":
            c["MB"]["rmse"],

        "delta_MB_MA_RMSE":
            c["MB"]["rmse"]
            - c["MA"]["rmse"],

        "M0_corr":
            c["M0"]["correlation"],

        "MA_corr":
            c["MA"]["correlation"],

        "MB_corr":
            c["MB"]["correlation"],

        "delta_MB_MA_corr":
            c["MB"]["correlation"]
            - c["MA"]["correlation"],
    })


comparison = pd.DataFrame(rows)

comparison.to_csv(
    OUT /
    "ablation_comparison.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "separate_expert_reliability_ablation_v1",

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

    "models": {
        "M0":
            "own uncertainty",

        "MA":
            (
                "own uncertainty + "
                "other expert uncertainty"
            ),

        "MB":
            (
                "own uncertainty + "
                "other expert uncertainty + "
                "absolute disagreement"
            ),
    },

    "primary_comparison":
        "MB versus MA",

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
    "ablation_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("=" * 100)
print("PRIMARY ABLATION SUMMARY: MB - MA")
print("=" * 100)

for result in [
    sequence_result,
    structure_result,
]:

    name = result["expert"]

    bd = result[
        "binary_delta_MB_minus_MA"
    ]

    cd = result[
        "continuous_delta_MB_minus_MA"
    ]

    print()
    print(name.upper())

    print(
        f"  ROC-AUC:      "
        f"{bd['roc_auc']:+.6f}"
    )

    print(
        f"  AP:           "
        f"{bd['average_precision']:+.6f}"
    )

    print(
        f"  Log-loss:     "
        f"{bd['log_loss']:+.6f}"
    )

    print(
        f"  Brier:        "
        f"{bd['brier']:+.6f}"
    )

    print(
        f"  MAE:          "
        f"{cd['mae']:+.6f}"
    )

    print(
        f"  RMSE:         "
        f"{cd['rmse']:+.6f}"
    )

    print(
        f"  Correlation:  "
        f"{cd['correlation']:+.6f}"
    )


print()
print("=" * 100)

print("Primary comparison MB vs MA:       YES")
print("Explicit disagreement isolated:    YES")
print("Fit only on calibration subset:    YES")
print("Evaluation only on cal-eval:       YES")
print("pLDDT used:                        NO")
print("Threshold optimized:               NO")
print("Final TEST confirmation:           NO")
print("TEST USED:                         NO")

print()

print(
    f"Saved under:\n  {OUT}"
)

print("=" * 100)
