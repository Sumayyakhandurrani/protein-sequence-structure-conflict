
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

OUT = Path(
    "data/conflict/conflict_formulation_comparison_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-7


# ============================================================
# LOAD
# ============================================================

ps_cal = np.load(BASE / "p_seq_cal.npy").astype(np.float64)
pt_cal = np.load(BASE / "p_struct_cal.npy").astype(np.float64)
y_cal = np.load(BASE / "y_cal.npy").astype(np.float64)

ps_eval = np.load(BASE / "p_seq_eval.npy").astype(np.float64)
pt_eval = np.load(BASE / "p_struct_eval.npy").astype(np.float64)
y_eval = np.load(BASE / "y_eval.npy").astype(np.float64)

assert ps_cal.shape == pt_cal.shape == y_cal.shape
assert ps_eval.shape == pt_eval.shape == y_eval.shape

print("=" * 100)
print("CONFLICT FORMULATION COMPARISON V1")
print("=" * 100)

print(f"CAL shape:       {y_cal.shape}")
print(f"CAL-EVAL shape:  {y_eval.shape}")
print("TEST USED:       NO")
print()


# ============================================================
# BASIC FEATURES
# ============================================================

def uncertainty(p):
    return 1.0 - 2.0 * np.abs(p - 0.5)


def absolute_disagreement(ps, pt):
    return np.abs(ps - pt)


def signed_disagreement(ps, pt):
    return ps - pt


def bernoulli_js(ps, pt):
    """
    Jensen-Shannon divergence between:
        Bernoulli(ps)
        Bernoulli(pt)

    Natural logarithm is used.
    """

    ps = np.clip(ps, EPS, 1.0 - EPS)
    pt = np.clip(pt, EPS, 1.0 - EPS)

    # distributions:
    # [p, 1-p]

    m1 = 0.5 * (ps + pt)
    m0 = 1.0 - m1

    kl_s = (
        ps * np.log(ps / m1)
        +
        (1.0 - ps)
        * np.log((1.0 - ps) / m0)
    )

    kl_t = (
        pt * np.log(pt / m1)
        +
        (1.0 - pt)
        * np.log((1.0 - pt) / m0)
    )

    return 0.5 * (kl_s + kl_t)


# ============================================================
# CONFLICT SIGNALS
# ============================================================

conflict_cal = {
    "C1_absolute":
        absolute_disagreement(ps_cal, pt_cal),

    "C2_bernoulli_JS":
        bernoulli_js(ps_cal, pt_cal),

    "C3_signed":
        signed_disagreement(ps_cal, pt_cal),
}

conflict_eval = {
    "C1_absolute":
        absolute_disagreement(ps_eval, pt_eval),

    "C2_bernoulli_JS":
        bernoulli_js(ps_eval, pt_eval),

    "C3_signed":
        signed_disagreement(ps_eval, pt_eval),
}


# ============================================================
# SANITY CHECK SIGNALS
# ============================================================

for name in conflict_cal:

    c_cal = conflict_cal[name]
    c_eval = conflict_eval[name]

    assert c_cal.shape == y_cal.shape
    assert c_eval.shape == y_eval.shape

    assert np.isfinite(c_cal).all()
    assert np.isfinite(c_eval).all()

    print(
        f"{name:<20} "
        f"CAL range=[{c_cal.min():+.6f}, "
        f"{c_cal.max():+.6f}] | "
        f"EVAL range=[{c_eval.min():+.6f}, "
        f"{c_eval.max():+.6f}]"
    )

print()


# ============================================================
# BASELINE UNCERTAINTIES
# ============================================================

us_cal = uncertainty(ps_cal)
ut_cal = uncertainty(pt_cal)

us_eval = uncertainty(ps_eval)
ut_eval = uncertainty(pt_eval)

XA_cal = np.stack(
    [ut_cal, us_cal],
    axis=-1,
).reshape(-1, 2)

XA_eval = np.stack(
    [ut_eval, us_eval],
    axis=-1,
).reshape(-1, 2)


# ============================================================
# STRUCTURE ERROR TARGETS
# ============================================================

err_cal = np.abs(
    y_cal - pt_cal
)

err_eval = np.abs(
    y_eval - pt_eval
)

binary_cal = (
    err_cal >= 0.5
).astype(np.uint8).reshape(-1)

binary_eval = (
    err_eval >= 0.5
).astype(np.uint8).reshape(-1)

continuous_cal = err_cal.reshape(-1)
continuous_eval = err_eval.reshape(-1)


print(
    f"CAL structure failures:      "
    f"{binary_cal.sum():,}/"
    f"{len(binary_cal):,} "
    f"({binary_cal.mean():.4%})"
)

print(
    f"CAL-EVAL structure failures: "
    f"{binary_eval.sum():,}/"
    f"{len(binary_eval):,} "
    f"({binary_eval.mean():.4%})"
)

print()


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
            float(log_loss(y, p, labels=[0, 1])),

        "brier":
            float(brier_score_loss(y, p)),
    }


def continuous_metrics(y, p):

    corr = float(
        np.corrcoef(y, p)[0, 1]
    )

    return {
        "mae":
            float(mean_absolute_error(y, p)),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(y, p)
                )
            ),

        "correlation":
            corr,
    }


# ============================================================
# FIT COMMON BASELINE MA
# ============================================================

base_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

base_cont = Ridge(alpha=1.0)

base_bin.fit(
    XA_cal,
    binary_cal,
)

base_cont.fit(
    XA_cal,
    continuous_cal,
)

base_bin_eval = base_bin.predict_proba(
    XA_eval
)[:, 1]

base_cont_eval = base_cont.predict(
    XA_eval
)

base_bin_metrics = binary_metrics(
    binary_eval,
    base_bin_eval,
)

base_cont_metrics = continuous_metrics(
    continuous_eval,
    base_cont_eval,
)


# ============================================================
# SAVE BASELINE PREDICTIONS
# ============================================================

np.save(
    OUT / "binary_target.npy",
    binary_eval.reshape(y_eval.shape),
)

np.save(
    OUT / "continuous_target.npy",
    continuous_eval.reshape(y_eval.shape),
)

np.save(
    OUT / "MA_binary_predictions.npy",
    base_bin_eval.reshape(y_eval.shape),
)

np.save(
    OUT / "MA_continuous_predictions.npy",
    base_cont_eval.reshape(y_eval.shape),
)


# ============================================================
# FIT EACH CONFLICT FORMULATION
# ============================================================

results = {}

for name in [
    "C1_absolute",
    "C2_bernoulli_JS",
    "C3_signed",
]:

    print("=" * 100)
    print(name)
    print("=" * 100)

    c_cal = conflict_cal[name].reshape(-1)
    c_eval = conflict_eval[name].reshape(-1)

    XB_cal = np.column_stack(
        [
            XA_cal,
            c_cal,
        ]
    )

    XB_eval = np.column_stack(
        [
            XA_eval,
            c_eval,
        ]
    )

    # ---------------- Binary ----------------

    model_bin = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    model_bin.fit(
        XB_cal,
        binary_cal,
    )

    pred_bin = model_bin.predict_proba(
        XB_eval
    )[:, 1]

    B_bin = binary_metrics(
        binary_eval,
        pred_bin,
    )

    # ---------------- Continuous ----------------

    model_cont = Ridge(
        alpha=1.0
    )

    model_cont.fit(
        XB_cal,
        continuous_cal,
    )

    pred_cont = model_cont.predict(
        XB_eval
    )

    B_cont = continuous_metrics(
        continuous_eval,
        pred_cont,
    )

    # ---------------- Deltas ----------------

    delta_bin = {
        k:
            B_bin[k]
            - base_bin_metrics[k]
        for k in B_bin
    }

    delta_cont = {
        k:
            B_cont[k]
            - base_cont_metrics[k]
        for k in B_cont
    }

    print("Binary:")

    for metric in [
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier",
    ]:
        print(
            f"  {metric:<20} "
            f"MA={base_bin_metrics[metric]:.8f} | "
            f"MB={B_bin[metric]:.8f} | "
            f"delta={delta_bin[metric]:+.8f}"
        )

    print()

    print("Continuous:")

    for metric in [
        "mae",
        "rmse",
        "correlation",
    ]:
        print(
            f"  {metric:<20} "
            f"MA={base_cont_metrics[metric]:.8f} | "
            f"MB={B_cont[metric]:.8f} | "
            f"delta={delta_cont[metric]:+.8f}"
        )

    print()

    print(
        "Binary coefficients:",
        model_bin.coef_[0].tolist()
    )

    print(
        "Continuous coefficients:",
        model_cont.coef_.tolist()
    )

    # Save predictions for later bootstrap/freeze.
    np.save(
        OUT / f"{name}_binary_predictions.npy",
        pred_bin.reshape(y_eval.shape),
    )

    np.save(
        OUT / f"{name}_continuous_predictions.npy",
        pred_cont.reshape(y_eval.shape),
    )

    results[name] = {
        "binary_metrics": B_bin,
        "continuous_metrics": B_cont,
        "binary_delta_vs_MA": delta_bin,
        "continuous_delta_vs_MA": delta_cont,
        "binary_coefficients":
            model_bin.coef_[0].tolist(),
        "continuous_coefficients":
            model_cont.coef_.tolist(),
    }


# ============================================================
# COMPARISON TABLE
# ============================================================

rows = []

for name, r in results.items():

    rows.append({
        "conflict":
            name,

        "delta_roc_auc":
            r["binary_delta_vs_MA"]["roc_auc"],

        "delta_average_precision":
            r["binary_delta_vs_MA"][
                "average_precision"
            ],

        "delta_log_loss":
            r["binary_delta_vs_MA"]["log_loss"],

        "delta_brier":
            r["binary_delta_vs_MA"]["brier"],

        "delta_mae":
            r["continuous_delta_vs_MA"]["mae"],

        "delta_rmse":
            r["continuous_delta_vs_MA"]["rmse"],

        "delta_correlation":
            r["continuous_delta_vs_MA"][
                "correlation"
            ],
    })

comparison = pd.DataFrame(rows)

comparison.to_csv(
    OUT / "comparison_table.csv",
    index=False,
)

print("=" * 100)
print("SUMMARY — PRIMARY METRIC")
print("=" * 100)

print(
    comparison[
        [
            "conflict",
            "delta_average_precision",
            "delta_log_loss",
            "delta_brier",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = {
    "analysis":
        "conflict_formulation_comparison_v1",

    "candidate_formulations": {
        "C1_absolute":
            "|p_sequence - p_structure|",

        "C2_bernoulli_JS":
            (
                "Jensen-Shannon divergence between "
                "Bernoulli(p_sequence) and "
                "Bernoulli(p_structure)"
            ),

        "C3_signed":
            "p_sequence - p_structure",
    },

    "baseline_features": [
        "U_structure",
        "U_sequence",
    ],

    "primary_endpoint":
        "structure failure defined as "
        "|p_structure-y| >= 0.5",

    "primary_selection_metric":
        "CAL-EVAL delta average precision versus MA",

    "secondary_metrics": [
        "ROC-AUC",
        "log-loss",
        "Brier",
        "continuous MAE",
        "continuous RMSE",
        "continuous correlation",
    ],

    "fit_population":
        "CAL",

    "evaluation_population":
        "CAL-EVAL",

    "test_used":
        False,

    "post_outcome_candidate_addition_allowed":
        False,
}

with open(
    OUT / "comparison_manifest.json",
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

print("Candidate formulations predefined: YES")
print("Models fitted on CAL:              YES")
print("Compared on CAL-EVAL:              YES")
print("Primary metric predefined:         delta AP")
print("TEST USED:                         NO")
print("Additional candidates after result:NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)