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
# CONFIG
# ============================================================

BASE = Path("data/conflict/go_frequency_v1_inputs")

MATCH_FILE = Path(
    "data/conflict/expert_strength_control_v1/"
    "primary_matched_labels.csv"
)

OUT = Path(
    "data/conflict/expert_strength_control_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

N_LABELS = 494


# ============================================================
# LOAD DATA
# ============================================================

p_seq_cal = np.load(BASE / "p_seq_cal.npy")
p_struct_cal = np.load(BASE / "p_struct_cal.npy")
y_cal = np.load(BASE / "y_cal.npy")

p_seq_eval = np.load(BASE / "p_seq_eval.npy")
p_struct_eval = np.load(BASE / "p_struct_eval.npy")
y_eval = np.load(BASE / "y_eval.npy")

matched = pd.read_csv(MATCH_FILE)

cols = np.sort(
    matched["label_index"].to_numpy(dtype=int)
)

assert len(cols) == 92
assert len(np.unique(cols)) == 92
assert cols.min() >= 0
assert cols.max() < N_LABELS

assert (
    p_seq_cal.shape
    == p_struct_cal.shape
    == y_cal.shape
)

assert (
    p_seq_eval.shape
    == p_struct_eval.shape
    == y_eval.shape
)


# ============================================================
# SELECT PREDEFINED MATCHED LABELS
# ============================================================

ps_c = p_seq_cal[:, cols]
pt_c = p_struct_cal[:, cols]
yc = y_cal[:, cols]

ps_e = p_seq_eval[:, cols]
pt_e = p_struct_eval[:, cols]
ye = y_eval[:, cols]


# ============================================================
# FEATURES
# ============================================================

def uncertainty(p):
    return 1.0 - 2.0 * np.abs(p - 0.5)


def make_features(ps, pt):
    u_seq = uncertainty(ps)
    u_struct = uncertainty(pt)

    disagreement = np.abs(
        ps - pt
    )

    # MA:
    # own structure uncertainty
    # + other expert uncertainty
    XA = np.stack(
        [
            u_struct,
            u_seq,
        ],
        axis=-1,
    )

    # MB:
    # same baseline + explicit disagreement
    XB = np.stack(
        [
            u_struct,
            u_seq,
            disagreement,
        ],
        axis=-1,
    )

    return (
        XA.reshape(-1, 2),
        XB.reshape(-1, 3),
    )


XA_cal, XB_cal = make_features(
    ps_c,
    pt_c,
)

XA_eval, XB_eval = make_features(
    ps_e,
    pt_e,
)


# ============================================================
# STRUCTURE ERROR TARGETS
# ============================================================

struct_abs_error_cal = np.abs(
    yc - pt_c
)

struct_abs_error_eval = np.abs(
    ye - pt_e
)

# Same diagnostic failure definition used previously.
binary_cal = (
    struct_abs_error_cal >= 0.5
).astype(np.uint8).reshape(-1)

binary_eval = (
    struct_abs_error_eval >= 0.5
).astype(np.uint8).reshape(-1)

continuous_cal = (
    struct_abs_error_cal
    .reshape(-1)
)

continuous_eval = (
    struct_abs_error_eval
    .reshape(-1)
)


# ============================================================
# FIT BINARY RELIABILITY MODELS ON CAL
# ============================================================

model_A_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

model_B_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

model_A_bin.fit(
    XA_cal,
    binary_cal,
)

model_B_bin.fit(
    XB_cal,
    binary_cal,
)

pred_A_bin = model_A_bin.predict_proba(
    XA_eval
)[:, 1]

pred_B_bin = model_B_bin.predict_proba(
    XB_eval
)[:, 1]


# ============================================================
# FIT CONTINUOUS RELIABILITY MODELS ON CAL
# ============================================================

model_A_cont = Ridge(
    alpha=1.0
)

model_B_cont = Ridge(
    alpha=1.0
)

model_A_cont.fit(
    XA_cal,
    continuous_cal,
)

model_B_cont.fit(
    XB_cal,
    continuous_cal,
)

pred_A_cont = model_A_cont.predict(
    XA_eval
)

pred_B_cont = model_B_cont.predict(
    XB_eval
)


# ============================================================
# METRICS
# ============================================================

def binary_metrics(y, p):
    return {
        "roc_auc":
            float(roc_auc_score(y, p)),

        "average_precision":
            float(average_precision_score(y, p)),

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


def continuous_metrics(y, p):

    if (
        np.std(y) > 0
        and np.std(p) > 0
    ):
        corr = float(
            np.corrcoef(y, p)[0, 1]
        )
    else:
        corr = np.nan

    return {
        "mae":
            float(
                mean_absolute_error(y, p)
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(y, p)
                )
            ),

        "correlation":
            corr,
    }


A_bin = binary_metrics(
    binary_eval,
    pred_A_bin,
)

B_bin = binary_metrics(
    binary_eval,
    pred_B_bin,
)

A_cont = continuous_metrics(
    continuous_eval,
    pred_A_cont,
)

B_cont = continuous_metrics(
    continuous_eval,
    pred_B_cont,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print("EXPERT-STRENGTH CONTROL V1")
print("=" * 100)

print("Matching source:             CAL only")
print("Matching metric:             per-label Average Precision")
print("Primary AP gap:              <= 0.05")
print("Minimum CAL positives:       >= 5")
print(f"Frozen matched labels:       {len(cols)}")
print()

print(
    f"CAL pairs:                   "
    f"{len(binary_cal):,}"
)

print(
    f"CAL-EVAL pairs:              "
    f"{len(binary_eval):,}"
)

print(
    f"CAL failure events:          "
    f"{int(binary_cal.sum()):,} "
    f"({binary_cal.mean():.4%})"
)

print(
    f"CAL-EVAL failure events:     "
    f"{int(binary_eval.sum()):,} "
    f"({binary_eval.mean():.4%})"
)

print()

print(
    "MA = [U_structure, U_sequence]"
)

print(
    "MB = [U_structure, U_sequence, D]"
)

print(
    "D  = |p_sequence - p_structure|"
)

print()


print("-" * 100)
print("BINARY STRUCTURE-FAILURE RELIABILITY")
print("-" * 100)

for metric in [
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
]:
    delta = (
        B_bin[metric]
        - A_bin[metric]
    )

    print(
        f"{metric:<20} "
        f"MA={A_bin[metric]:.8f} | "
        f"MB={B_bin[metric]:.8f} | "
        f"delta={delta:+.8f}"
    )


print()
print("-" * 100)
print("CONTINUOUS STRUCTURE ABSOLUTE ERROR")
print("-" * 100)

for metric in [
    "mae",
    "rmse",
    "correlation",
]:
    delta = (
        B_cont[metric]
        - A_cont[metric]
    )

    print(
        f"{metric:<20} "
        f"MA={A_cont[metric]:.8f} | "
        f"MB={B_cont[metric]:.8f} | "
        f"delta={delta:+.8f}"
    )


print()
print("-" * 100)
print("MODEL COEFFICIENTS")
print("-" * 100)

print(
    "MA binary:",
    model_A_bin.coef_[0].tolist()
)

print(
    "MB binary:",
    model_B_bin.coef_[0].tolist()
)

print(
    "MA continuous:",
    model_A_cont.coef_.tolist()
)

print(
    "MB continuous:",
    model_B_cont.coef_.tolist()
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    OUT / "matched_binary_target.npy",
    binary_eval.reshape(
        len(y_eval),
        len(cols),
    ),
)

np.save(
    OUT / "matched_binary_MA_predictions.npy",
    pred_A_bin.reshape(
        len(y_eval),
        len(cols),
    ),
)

np.save(
    OUT / "matched_binary_MB_predictions.npy",
    pred_B_bin.reshape(
        len(y_eval),
        len(cols),
    ),
)

np.save(
    OUT / "matched_continuous_target.npy",
    continuous_eval.reshape(
        len(y_eval),
        len(cols),
    ),
)

np.save(
    OUT / "matched_continuous_MA_predictions.npy",
    pred_A_cont.reshape(
        len(y_eval),
        len(cols),
    ),
)

np.save(
    OUT / "matched_continuous_MB_predictions.npy",
    pred_B_cont.reshape(
        len(y_eval),
        len(cols),
    ),
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "analysis":
        "expert_strength_control_v1",

    "matching_population":
        "CAL only",

    "matched_labels":
        int(len(cols)),

    "matching_rule": {
        "metric":
            "per-label average precision",

        "absolute_gap_max":
            0.05,

        "minimum_cal_positives":
            5,
    },

    "cal_pairs":
        int(len(binary_cal)),

    "eval_pairs":
        int(len(binary_eval)),

    "cal_failure_events":
        int(binary_cal.sum()),

    "eval_failure_events":
        int(binary_eval.sum()),

    "eval_failure_prevalence":
        float(binary_eval.mean()),

    "binary": {
        "MA": A_bin,
        "MB": B_bin,

        "delta_MB_minus_MA": {
            k: float(
                B_bin[k] - A_bin[k]
            )
            for k in A_bin
        },
    },

    "continuous": {
        "MA": A_cont,
        "MB": B_cont,

        "delta_MB_minus_MA": {
            k: float(
                B_cont[k] - A_cont[k]
            )
            for k in A_cont
        },
    },

    "binary_coefficients": {
        "MA":
            model_A_bin.coef_[0].tolist(),

        "MB":
            model_B_bin.coef_[0].tolist(),
    },

    "continuous_coefficients": {
        "MA":
            model_A_cont.coef_.tolist(),

        "MB":
            model_B_cont.coef_.tolist(),
    },

    "models_refit_on_cal":
        True,

    "cal_eval_used_for_matching":
        False,

    "threshold_optimized":
        False,

    "test_used":
        False,
}

with open(
    OUT / "expert_strength_control_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 100)
print("INTEGRITY")
print("=" * 100)

print("Matched labels selected using CAL:       YES")
print("CAL-EVAL used for label matching:        NO")
print("Reliability models fitted on CAL:        YES")
print("Reliability evaluated on CAL-EVAL:       YES")
print("Matching cutoff changed after outcome:   NO")
print("Threshold optimization:                  NO")
print("TEST USED:                               NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)