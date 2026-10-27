from pathlib import Path
import json
import hashlib

import numpy as np

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)


EPS = 1e-7


# ============================================================
# PATHS
# ============================================================

# Frozen VAL-side inputs used ONLY to construct CAL fitting data.
PAIRED_DIR = Path(
    "data/conflict/paired_experts_v1"
)

GCN_VAL_DIR = Path(
    "data/analysis/backbone_sensitivity_v1/"
    "gcn_val_inference"
)

CALIB_DIR = Path(
    "data/analysis/backbone_sensitivity_v1/"
    "gcn_calibration"
)

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

# Frozen TEST-side prediction inputs.
GCN_TEST_DIR = Path(
    "data/analysis/backbone_sensitivity_v1/"
    "gcn_test_inference"
)

SEQ_TEST_DIR = Path(
    "data/final_evaluation/"
    "sequence_expert_v1_test"
)

TEST_IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/"
    "test_uniprot_ids.txt"
)

TEST_LABELS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/"
    "test_labels.npy"
)

OUT_DIR = Path(
    "data/analysis/backbone_sensitivity_v1/"
    "final_test_v1"
)

OUT_DIR.mkdir(
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


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def uncertainty(p):
    return (
        1.0
        - 2.0 * np.abs(p - 0.5)
    )


def bernoulli_js(ps, pt):

    ps = np.clip(
        ps,
        EPS,
        1.0 - EPS,
    )

    pt = np.clip(
        pt,
        EPS,
        1.0 - EPS,
    )

    m = 0.5 * (
        ps + pt
    )

    return 0.5 * (
        ps * np.log(ps / m)
        + (1.0 - ps)
        * np.log(
            (1.0 - ps)
            / (1.0 - m)
        )
        + pt * np.log(pt / m)
        + (1.0 - pt)
        * np.log(
            (1.0 - pt)
            / (1.0 - m)
        )
    )


def binary_metrics(y, p):
    return {
        "roc_auc":
            float(
                roc_auc_score(y, p)
            ),

        "average_precision":
            float(
                average_precision_score(y, p)
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
                brier_score_loss(y, p)
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
# HEADER
# ============================================================

print("=" * 92)
print("E12 BACKBONE SENSITIVITY — STRUCTURE RELIABILITY")
print("=" * 92)

print("Alternative structure expert: GCN-style")
print("Sequence expert modified: NO")
print("Conflict: Bernoulli JS, natural log, EPS=1e-7")
print("Primary comparison: MB versus MA")
print("MA: U_structure + U_sequence")
print("MB: U_structure + U_sequence + Bernoulli_JS")
print("Reliability fit: CAL")
print("Reliability evaluation: FINAL FROZEN TEST")
print("Threshold optimized: NO")
print("TEST used for evaluation: YES")
print()


# ============================================================
# LOAD FROZEN CAL-SIDE INPUTS
# ============================================================

paired_ids_path = (
    PAIRED_DIR /
    "paired_val_uniprot_ids.txt"
)

gcn_val_ids_path = (
    GCN_VAL_DIR /
    "val_structure_eligible_ids.txt"
)

sequence_probs_path = (
    PAIRED_DIR /
    "sequence_probs.npy"
)

labels_path = (
    PAIRED_DIR /
    "paired_val_labels.npy"
)

gcn_val_logits_path = (
    GCN_VAL_DIR /
    "val_raw_logits.npy"
)

temperature_path = (
    CALIB_DIR /
    "temperature_fit.json"
)

cal_ids_path = (
    SPLIT_DIR /
    "structure_calibration_ids.txt"
)


paired_ids = read_ids(
    paired_ids_path
)

gcn_val_ids = read_ids(
    gcn_val_ids_path
)

cal_ids = read_ids(
    cal_ids_path
)


if paired_ids != gcn_val_ids:
    raise RuntimeError(
        "GCN and paired-expert VAL ID order mismatch."
    )


y_val_all = np.load(
    labels_path
).astype(np.float64)

p_seq_val_all = np.load(
    sequence_probs_path
).astype(np.float64)

gcn_val_logits = np.load(
    gcn_val_logits_path
).astype(np.float64)


with open(
    temperature_path,
    encoding="utf-8",
) as f:
    temperature_info = json.load(f)


T = float(
    temperature_info["temperature"]
)


# ============================================================
# VALIDATE CAL-SIDE INPUTS
# ============================================================

expected_val_shape = (
    4742,
    494,
)

for name, x in [
    ("VAL labels", y_val_all),
    ("VAL sequence probabilities", p_seq_val_all),
    ("VAL GCN logits", gcn_val_logits),
]:
    if x.shape != expected_val_shape:
        raise RuntimeError(
            f"{name} shape mismatch: {x.shape}"
        )


if len(cal_ids) != 2369:
    raise RuntimeError(
        f"Expected 2,369 CAL proteins, "
        f"got {len(cal_ids)}."
    )


if (
    not np.isfinite(T)
    or T <= 0
):
    raise RuntimeError(
        f"Invalid temperature: {T}"
    )


if not np.isfinite(
    gcn_val_logits
).all():
    raise RuntimeError(
        "Non-finite GCN VAL logits."
    )


# ============================================================
# APPLY FROZEN GCN CALIBRATION TO VAL
# ============================================================

scaled_val_logits = (
    gcn_val_logits / T
)

p_struct_val_all = (
    1.0
    / (
        1.0
        + np.exp(
            -np.clip(
                scaled_val_logits,
                -80.0,
                80.0,
            )
        )
    )
)


# ============================================================
# CONSTRUCT CAL FITTING POPULATION
# ============================================================

val_index = {
    uid: i
    for i, uid in enumerate(paired_ids)
}


if not set(cal_ids).issubset(val_index):
    raise RuntimeError(
        "CAL IDs missing from VAL predictions."
    )


cal_rows = np.asarray(
    [
        val_index[x]
        for x in cal_ids
    ],
    dtype=np.int64,
)


y_cal = y_val_all[cal_rows]
seq_cal = p_seq_val_all[cal_rows]
struct_cal = p_struct_val_all[cal_rows]


# ============================================================
# LOAD FROZEN TEST PREDICTIONS — NO LABELS YET
# ============================================================

test_ids = read_ids(
    TEST_IDS_PATH
)

seq_test_ids = read_ids(
    SEQ_TEST_DIR /
    "test_uniprot_ids.txt"
)

struct_test_ids = read_ids(
    GCN_TEST_DIR /
    "test_structure_eligible_ids.txt"
)


if len(test_ids) != 4754:
    raise RuntimeError(
        f"Expected 4,754 TEST proteins, "
        f"got {len(test_ids)}."
    )


if seq_test_ids != test_ids:
    raise RuntimeError(
        "Sequence TEST ID order does not exactly "
        "match frozen TEST order."
    )


if len(struct_test_ids) != 4731:
    raise RuntimeError(
        f"Expected 4,731 structure-eligible TEST "
        f"proteins, got {len(struct_test_ids)}."
    )


p_seq_test_all = np.load(
    SEQ_TEST_DIR /
    "test_calibrated_probabilities.npy"
).astype(np.float64)

p_struct_test = np.load(
    GCN_TEST_DIR /
    "test_calibrated_probabilities.npy"
).astype(np.float64)


if p_seq_test_all.shape != (
    4754,
    494,
):
    raise RuntimeError(
        "Sequence TEST probability shape mismatch."
    )


if p_struct_test.shape != (
    4731,
    494,
):
    raise RuntimeError(
        "Alternative-GCN TEST probability "
        "shape mismatch."
    )


if not np.isfinite(
    p_seq_test_all
).all():
    raise RuntimeError(
        "Non-finite sequence TEST probabilities."
    )


if not np.isfinite(
    p_struct_test
).all():
    raise RuntimeError(
        "Non-finite alternative-GCN TEST "
        "probabilities."
    )


# ============================================================
# PAIRED TEST ORDER
# ============================================================

test_index = {
    uid: i
    for i, uid in enumerate(test_ids)
}


if any(
    uid not in test_index
    for uid in struct_test_ids
):
    raise RuntimeError(
        "Structure TEST ID missing from frozen TEST."
    )


paired_rows = np.asarray(
    [
        test_index[uid]
        for uid in struct_test_ids
    ],
    dtype=np.int64,
)


seq_eval = p_seq_test_all[
    paired_rows
]

struct_eval = p_struct_test


# ============================================================
# FIRST DELIBERATE E12 TEST GROUND-TRUTH LOAD
# ============================================================

y_test_all = np.load(
    TEST_LABELS_PATH
).astype(np.uint8)


if y_test_all.shape != (
    4754,
    494,
):
    raise RuntimeError(
        "Frozen TEST label shape mismatch."
    )


y_eval = y_test_all[
    paired_rows
].astype(np.float64)


if y_eval.shape != (
    4731,
    494,
):
    raise RuntimeError(
        "Paired TEST label shape mismatch."
    )


if not np.all(
    (y_eval == 0)
    | (y_eval == 1)
):
    raise RuntimeError(
        "TEST labels are not binary."
    )


print(
    "FIRST DELIBERATE E12 TEST GROUND-TRUTH "
    "LOAD: YES"
)
print(
    f"Frozen TEST proteins:       "
    f"{len(test_ids):,}"
)
print(
    f"Paired TEST proteins:       "
    f"{len(struct_test_ids):,}"
)
print(
    f"Excluded (no structure):    "
    f"{len(test_ids) - len(struct_test_ids):,}"
)
print()


# ============================================================
# FROZEN RELIABILITY FEATURES
# ============================================================

U_seq_cal = uncertainty(
    seq_cal
).reshape(-1)

U_seq_eval = uncertainty(
    seq_eval
).reshape(-1)

U_struct_cal = uncertainty(
    struct_cal
).reshape(-1)

U_struct_eval = uncertainty(
    struct_eval
).reshape(-1)


JS_cal = bernoulli_js(
    seq_cal,
    struct_cal,
).reshape(-1)

JS_eval = bernoulli_js(
    seq_eval,
    struct_eval,
).reshape(-1)


# ============================================================
# STRUCTURE ERROR TARGETS
# ============================================================

struct_abs_cal = np.abs(
    y_cal - struct_cal
).reshape(-1)

struct_abs_eval = np.abs(
    y_eval - struct_eval
).reshape(-1)


struct_binary_cal = (
    struct_abs_cal >= 0.5
).astype(np.uint8)

struct_binary_eval = (
    struct_abs_eval >= 0.5
).astype(np.uint8)


# ============================================================
# FEATURE MATRICES
# ============================================================

XA_cal = np.column_stack([
    U_struct_cal,
    U_seq_cal,
])

XA_eval = np.column_stack([
    U_struct_eval,
    U_seq_eval,
])


XB_cal = np.column_stack([
    U_struct_cal,
    U_seq_cal,
    JS_cal,
])

XB_eval = np.column_stack([
    U_struct_eval,
    U_seq_eval,
    JS_eval,
])


# ============================================================
# BINARY RELIABILITY MODELS
# ============================================================

binary_MA = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)

binary_MB = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)


binary_MA.fit(
    XA_cal,
    struct_binary_cal,
)

binary_MB.fit(
    XB_cal,
    struct_binary_cal,
)


pred_MA = binary_MA.predict_proba(
    XA_eval
)[:, 1]

pred_MB = binary_MB.predict_proba(
    XB_eval
)[:, 1]


binary_ma_metrics = binary_metrics(
    struct_binary_eval,
    pred_MA,
)

binary_mb_metrics = binary_metrics(
    struct_binary_eval,
    pred_MB,
)


# ============================================================
# CONTINUOUS RELIABILITY MODELS
# ============================================================

continuous_MA = Ridge(
    alpha=1.0
)

continuous_MB = Ridge(
    alpha=1.0
)


continuous_MA.fit(
    XA_cal,
    struct_abs_cal,
)

continuous_MB.fit(
    XB_cal,
    struct_abs_cal,
)


continuous_pred_MA = (
    continuous_MA.predict(
        XA_eval
    )
)

continuous_pred_MB = (
    continuous_MB.predict(
        XB_eval
    )
)


continuous_ma_metrics = (
    continuous_metrics(
        struct_abs_eval,
        continuous_pred_MA,
    )
)

continuous_mb_metrics = (
    continuous_metrics(
        struct_abs_eval,
        continuous_pred_MB,
    )
)


# ============================================================
# DELTAS
# ============================================================

binary_delta = {
    key:
        float(
            binary_mb_metrics[key]
            - binary_ma_metrics[key]
        )
    for key in binary_ma_metrics
}


continuous_delta = {
    key:
        float(
            continuous_mb_metrics[key]
            - continuous_ma_metrics[key]
        )
    for key in continuous_ma_metrics
}


# ============================================================
# SAVE PREDICTIONS FOR BOOTSTRAP
# ============================================================

np.save(
    OUT_DIR /
    "structure_binary_target.npy",
    struct_binary_eval,
)

np.save(
    OUT_DIR /
    "structure_binary_MA_predictions.npy",
    pred_MA.astype(np.float32),
)

np.save(
    OUT_DIR /
    "structure_binary_MB_predictions.npy",
    pred_MB.astype(np.float32),
)

np.save(
    OUT_DIR /
    "structure_continuous_target.npy",
    struct_abs_eval.astype(np.float32),
)

np.save(
    OUT_DIR /
    "structure_continuous_MA_predictions.npy",
    continuous_pred_MA.astype(np.float32),
)

np.save(
    OUT_DIR /
    "structure_continuous_MB_predictions.npy",
    continuous_pred_MB.astype(np.float32),
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "e12_backbone_sensitivity_final_test_v1",

    "scientific_role":
        (
            "pre-specified final TEST confirmation of "
            "alternative structure-backbone sensitivity"
        ),

    "alternative_structure_backbone":
        "GCN-style normalized-neighbor propagation",

    "calibration_temperature":
        T,

    "fit_population":
        "structure-eligible CAL only",

    "evaluation_population":
        "structure-eligible frozen Graph-V2 TEST",

    "calibration_proteins":
        len(cal_ids),

    "evaluation_proteins":
        len(struct_test_ids),

    "labels":
        494,

    "calibration_pairs":
        int(y_cal.size),

    "evaluation_pairs":
        int(y_eval.size),

    "structure_error_definition":
        "absolute error >= 0.5",

    "calibration_error_events":
        int(struct_binary_cal.sum()),

    "calibration_error_prevalence":
        float(struct_binary_cal.mean()),

    "evaluation_error_events":
        int(struct_binary_eval.sum()),

    "evaluation_error_prevalence":
        float(struct_binary_eval.mean()),

    "uncertainty_definition":
        "1 - 2*abs(p - 0.5)",

    "conflict_definition":
        (
            "per-label Bernoulli Jensen-Shannon "
            "divergence using natural logarithms"
        ),

    "conflict_eps":
        EPS,

    "MA_features": [
        "U_structure",
        "U_sequence",
    ],

    "MB_features": [
        "U_structure",
        "U_sequence",
        "Bernoulli_JS",
    ],

    "binary_model": {
        "type": "LogisticRegression",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    },

    "continuous_model": {
        "type": "Ridge",
        "alpha": 1.0,
    },

    "binary_MA":
        binary_ma_metrics,

    "binary_MB":
        binary_mb_metrics,

    "binary_delta_MB_minus_MA":
        binary_delta,

    "continuous_MA":
        continuous_ma_metrics,

    "continuous_MB":
        continuous_mb_metrics,

    "continuous_delta_MB_minus_MA":
        continuous_delta,

    "binary_MA_coefficients": [
        float(x)
        for x in binary_MA.coef_[0]
    ],

    "binary_MB_coefficients": [
        float(x)
        for x in binary_MB.coef_[0]
    ],

    "continuous_MA_coefficients": [
        float(x)
        for x in continuous_MA.coef_
    ],

    "continuous_MB_coefficients": [
        float(x)
        for x in continuous_MB.coef_
    ],

    "sequence_expert_modified":
        False,

    "primary_structure_model_modified":
        False,

    "new_conflict_definition_selected":
        False,

    "threshold_optimized":
        False,

    "test_ground_truth_opened":
        True,

    "test_used_for_evaluation":
        True,

    "test_used_for_training":
        False,

    "test_used_for_model_selection":
        False,

    "test_used_for_calibration":
        False,

    "test_used_for_conflict_selection":
        False,

    "test_used_for_threshold_optimization":
        False,

    "test_used_for_reliability_fitting":
        False,

    "reliability_models_fitted_on":
        "CAL only",

    "input_sha256": {
        "val_paired_ids":
            sha256_file(paired_ids_path),

        "val_sequence_probs":
            sha256_file(sequence_probs_path),

        "val_labels":
            sha256_file(labels_path),

        "gcn_val_ids":
            sha256_file(gcn_val_ids_path),

        "gcn_val_logits":
            sha256_file(gcn_val_logits_path),

        "temperature_fit":
            sha256_file(temperature_path),

        "cal_ids":
            sha256_file(cal_ids_path),

        "frozen_test_ids":
            sha256_file(TEST_IDS_PATH),

        "test_labels":
            sha256_file(TEST_LABELS_PATH),

        "sequence_test_ids":
            sha256_file(
                SEQ_TEST_DIR /
                "test_uniprot_ids.txt"
            ),

        "sequence_test_probabilities":
            sha256_file(
                SEQ_TEST_DIR /
                "test_calibrated_probabilities.npy"
            ),

        "gcn_test_ids":
            sha256_file(
                GCN_TEST_DIR /
                "test_structure_eligible_ids.txt"
            ),

        "gcn_test_calibrated_probabilities":
            sha256_file(
                GCN_TEST_DIR /
                "test_calibrated_probabilities.npy"
            ),
    },
}


summary_path = (
    OUT_DIR /
    "reliability_summary.json"
)

with open(
    summary_path,
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

print(
    f"Temperature T:            {T:.8f}"
)

print(
    f"CAL proteins:             {len(cal_ids):,}"
)

print(
    f"TEST proteins (paired):   "
    f"{len(struct_test_ids):,}"
)

print(
    f"CAL pairs:                {y_cal.size:,}"
)

print(
    f"TEST pairs:               {y_eval.size:,}"
)

print()

print("Structure error events:")

print(
    f"  CAL:  "
    f"{int(struct_binary_cal.sum()):,} "
    f"({struct_binary_cal.mean():.4%})"
)

print(
    f"  TEST: "
    f"{int(struct_binary_eval.sum()):,} "
    f"({struct_binary_eval.mean():.4%})"
)

print()

print("BINARY STRUCTURE-ERROR RELIABILITY")
print()

for metric in [
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
]:
    ma = binary_ma_metrics[metric]
    mb = binary_mb_metrics[metric]

    print(
        f"  {metric:<18} "
        f"MA={ma:.8f} | "
        f"MB={mb:.8f} | "
        f"delta={mb-ma:+.8f}"
    )

print()

print("CONTINUOUS STRUCTURE ABSOLUTE ERROR")
print()

for metric in [
    "mae",
    "rmse",
    "correlation",
]:
    ma = continuous_ma_metrics[metric]
    mb = continuous_mb_metrics[metric]

    print(
        f"  {metric:<18} "
        f"MA={ma:.8f} | "
        f"MB={mb:.8f} | "
        f"delta={mb-ma:+.8f}"
    )

print()

print(
    "MB Bernoulli-JS binary coefficient:",
    float(binary_MB.coef_[0][-1]),
)

print(
    "MB Bernoulli-JS continuous coefficient:",
    float(continuous_MB.coef_[-1]),
)

print()

print("Sequence expert modified:             NO")
print("Primary structure model modified:     NO")
print("New conflict definition selected:     NO")
print("Threshold optimized on TEST:          NO")
print("TEST ground truth opened:             YES")
print("TEST used for evaluation:             YES")
print("TEST used for training:               NO")
print("TEST used for model selection:        NO")
print("TEST used for calibration:            NO")
print("TEST used for conflict selection:     NO")
print("TEST used for reliability fitting:    NO")
print("Reliability models fitted on:         CAL ONLY")

print()
print("Saved:", summary_path)
print("=" * 92)
