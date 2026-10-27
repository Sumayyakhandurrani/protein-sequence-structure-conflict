"""
Global structural-confidence control for sequence-structure disagreement.

Question
--------
Does explicit sequence-structure disagreement D add information for
detecting STRUCTURE-expert failures after controlling for:

    1. structure-expert uncertainty
    2. sequence-expert uncertainty
    3. AlphaFold global confidence ("global_metric")

Nested models
-------------
M0 = [U_struct, U_seq]

MA = [U_struct, U_seq, Q]

MB = [U_struct, U_seq, Q, D]

where:
    U(p) = 1 - 2*|p - 0.5|
    Q    = global_metric / 100
    D    = |p_seq - p_struct|

Primary comparison:
    MB - MA

Primary metric:
    Average Precision (AP)

Bootstrap:
    Whole proteins are resampled.
    Reliability models are NOT refitted within bootstrap replicates.

TEST is never accessed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)


SEED = 42
N_BOOTSTRAP = 1000


# ============================================================
# Paths
# ============================================================

INPUT_DIR = Path(
    "data/conflict/go_frequency_v1_inputs"
)

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

AF_PATH = Path(
    "data/processed/alphafold_availability.csv"
)

OUTPUT_DIR = Path(
    "data/conflict/"
    "global_confidence_incremental_reliability_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Utilities
# ============================================================

def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


def uncertainty(p):
    return 1.0 - 2.0 * np.abs(p - 0.5)


def structure_failure(y, p_struct):
    return (
        np.abs(y - p_struct) >= 0.5
    ).astype(np.uint8)


def fit_logistic(X, y):

    if np.unique(y).size != 2:
        raise ValueError(
            "Training target does not contain "
            "both classes."
        )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )

    model.fit(X, y)

    return model


def calculate_metrics(y, p):

    return {
        "ap": float(
            average_precision_score(y, p)
        ),

        "roc_auc": float(
            roc_auc_score(y, p)
        ),

        "log_loss": float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),

        "brier": float(
            brier_score_loss(y, p)
        ),
    }


# ============================================================
# Load frozen paired CAL / CAL-EVAL arrays
# ============================================================

print("=" * 72)
print("GLOBAL STRUCTURAL-CONFIDENCE CONTROL")
print("=" * 72)

Y_cal = np.load(
    INPUT_DIR / "y_cal.npy"
)

Y_eval = np.load(
    INPUT_DIR / "y_eval.npy"
)

P_seq_cal = np.load(
    INPUT_DIR / "p_seq_cal.npy"
)

P_seq_eval = np.load(
    INPUT_DIR / "p_seq_eval.npy"
)

P_struct_cal = np.load(
    INPUT_DIR / "p_struct_cal.npy"
)

P_struct_eval = np.load(
    INPUT_DIR / "p_struct_eval.npy"
)

cal_ids = read_ids(
    SPLIT_DIR /
    "structure_calibration_ids.txt"
)

eval_ids = read_ids(
    SPLIT_DIR /
    "structure_cal_eval_ids.txt"
)


# ============================================================
# Integrity checks
# ============================================================

assert Y_cal.shape == (2369, 494)
assert Y_eval.shape == (2373, 494)

assert P_seq_cal.shape == Y_cal.shape
assert P_struct_cal.shape == Y_cal.shape

assert P_seq_eval.shape == Y_eval.shape
assert P_struct_eval.shape == Y_eval.shape

assert len(cal_ids) == Y_cal.shape[0]
assert len(eval_ids) == Y_eval.shape[0]

assert not (set(cal_ids) & set(eval_ids))

for name, arr in {
    "P_seq_cal": P_seq_cal,
    "P_seq_eval": P_seq_eval,
    "P_struct_cal": P_struct_cal,
    "P_struct_eval": P_struct_eval,
}.items():

    assert np.isfinite(arr).all(), name

    assert arr.min() >= 0
    assert arr.max() <= 1

print(
    f"CAL:      {Y_cal.shape}"
)

print(
    f"CAL-EVAL: {Y_eval.shape}"
)

print("Prediction integrity: PASS")


# ============================================================
# Load global structural confidence
# ============================================================

af = pd.read_csv(AF_PATH)

required = {
    "uniprot_id",
    "global_metric",
}

assert required.issubset(
    af.columns
)

assert not af["uniprot_id"].duplicated().any()

af = af.set_index("uniprot_id")


def extract_confidence(ids):

    missing_ids = [
        uid
        for uid in ids
        if uid not in af.index
    ]

    assert not missing_ids, (
        f"Missing AlphaFold IDs: "
        f"{len(missing_ids)}"
    )

    q = (
        af.loc[ids, "global_metric"]
        .to_numpy(dtype=np.float64)
    )

    assert np.isfinite(q).all()

    return q


Q_cal_raw = extract_confidence(cal_ids)
Q_eval_raw = extract_confidence(eval_ids)

print()
print("Global confidence:")
print(
    "CAL      "
    f"mean={Q_cal_raw.mean():.4f}, "
    f"median={np.median(Q_cal_raw):.4f}, "
    f"range=[{Q_cal_raw.min():.2f}, "
    f"{Q_cal_raw.max():.2f}]"
)

print(
    "CAL-EVAL "
    f"mean={Q_eval_raw.mean():.4f}, "
    f"median={np.median(Q_eval_raw):.4f}, "
    f"range=[{Q_eval_raw.min():.2f}, "
    f"{Q_eval_raw.max():.2f}]"
)

# Scale from approximately 0-100 to 0-1.
# This is purely numerical scaling for logistic regression.
Q_cal = Q_cal_raw / 100.0
Q_eval = Q_eval_raw / 100.0


# ============================================================
# Construct targets
# ============================================================

failure_cal = structure_failure(
    Y_cal,
    P_struct_cal,
)

failure_eval = structure_failure(
    Y_eval,
    P_struct_eval,
)

print()
print("Structure-failure events:")

print(
    f"CAL:      {failure_cal.sum():,} / "
    f"{failure_cal.size:,} "
    f"({failure_cal.mean():.6%})"
)

print(
    f"CAL-EVAL: {failure_eval.sum():,} / "
    f"{failure_eval.size:,} "
    f"({failure_eval.mean():.6%})"
)


# ============================================================
# Features
# ============================================================

U_struct_cal = uncertainty(P_struct_cal)
U_seq_cal = uncertainty(P_seq_cal)

U_struct_eval = uncertainty(P_struct_eval)
U_seq_eval = uncertainty(P_seq_eval)

D_cal = np.abs(
    P_seq_cal - P_struct_cal
)

D_eval = np.abs(
    P_seq_eval - P_struct_eval
)


# ------------------------------------------------------------
# Protein-level Q must be repeated across all 494 labels.
# ------------------------------------------------------------

Q_cal_matrix = np.repeat(
    Q_cal[:, None],
    Y_cal.shape[1],
    axis=1,
)

Q_eval_matrix = np.repeat(
    Q_eval[:, None],
    Y_eval.shape[1],
    axis=1,
)


# ============================================================
# Nested models
#
# M0 = U_struct + U_seq
# MA = U_struct + U_seq + global confidence
# MB = U_struct + U_seq + global confidence + disagreement
# ============================================================

X0_cal = np.column_stack([
    U_struct_cal.ravel(),
    U_seq_cal.ravel(),
])

XA_cal = np.column_stack([
    U_struct_cal.ravel(),
    U_seq_cal.ravel(),
    Q_cal_matrix.ravel(),
])

XB_cal = np.column_stack([
    U_struct_cal.ravel(),
    U_seq_cal.ravel(),
    Q_cal_matrix.ravel(),
    D_cal.ravel(),
])


X0_eval = np.column_stack([
    U_struct_eval.ravel(),
    U_seq_eval.ravel(),
])

XA_eval = np.column_stack([
    U_struct_eval.ravel(),
    U_seq_eval.ravel(),
    Q_eval_matrix.ravel(),
])

XB_eval = np.column_stack([
    U_struct_eval.ravel(),
    U_seq_eval.ravel(),
    Q_eval_matrix.ravel(),
    D_eval.ravel(),
])


y_cal = failure_cal.ravel()
y_eval = failure_eval.ravel()


# ============================================================
# Fit only on CAL
# ============================================================

print()
print("Fitting reliability models on CAL...")

model_M0 = fit_logistic(
    X0_cal,
    y_cal,
)

model_MA = fit_logistic(
    XA_cal,
    y_cal,
)

model_MB = fit_logistic(
    XB_cal,
    y_cal,
)


# ============================================================
# Evaluate only on CAL-EVAL
# ============================================================

pred_M0 = model_M0.predict_proba(
    X0_eval
)[:, 1]

pred_MA = model_MA.predict_proba(
    XA_eval
)[:, 1]

pred_MB = model_MB.predict_proba(
    XB_eval
)[:, 1]


metrics_M0 = calculate_metrics(
    y_eval,
    pred_M0,
)

metrics_MA = calculate_metrics(
    y_eval,
    pred_MA,
)

metrics_MB = calculate_metrics(
    y_eval,
    pred_MB,
)


# ============================================================
# Results
# ============================================================

comparison = pd.DataFrame([
    {
        "model": "M0",
        "features":
            "U_struct + U_seq",
        **metrics_M0,
    },
    {
        "model": "MA",
        "features":
            "U_struct + U_seq + global_confidence",
        **metrics_MA,
    },
    {
        "model": "MB",
        "features":
            "U_struct + U_seq + global_confidence + D",
        **metrics_MB,
    },
])

comparison.to_csv(
    OUTPUT_DIR /
    "model_comparison.csv",
    index=False,
)


def delta(a, b):
    """
    b - a
    """
    return {
        key: b[key] - a[key]
        for key in [
            "ap",
            "roc_auc",
            "log_loss",
            "brier",
        ]
    }


delta_MA_M0 = delta(
    metrics_M0,
    metrics_MA,
)

delta_MB_MA = delta(
    metrics_MA,
    metrics_MB,
)


print()
print("=" * 72)
print("HELD-OUT CAL-EVAL RESULTS")
print("=" * 72)

print(
    comparison.to_string(
        index=False
    )
)

print()
print("Effect of GLOBAL CONFIDENCE: MA - M0")

for k, v in delta_MA_M0.items():
    print(
        f"{k:10s}: {v:+.6f}"
    )

print()
print(
    "PRIMARY COMPARISON — "
    "effect of DISAGREEMENT after confidence: "
    "MB - MA"
)

for k, v in delta_MB_MA.items():
    print(
        f"{k:10s}: {v:+.6f}"
    )


# ============================================================
# Whole-protein bootstrap
#
# IMPORTANT:
# Predictions are reshaped back to:
# proteins x labels.
#
# Reliability models are NOT refitted.
# ============================================================

pred_MA_matrix = pred_MA.reshape(
    failure_eval.shape
)

pred_MB_matrix = pred_MB.reshape(
    failure_eval.shape
)

rng = np.random.default_rng(SEED)

n_proteins = failure_eval.shape[0]

records = []

print()
print(
    f"Running {N_BOOTSTRAP} "
    "whole-protein bootstrap replicates..."
)


for b in range(N_BOOTSTRAP):

    idx = rng.integers(
        0,
        n_proteins,
        size=n_proteins,
    )

    y_b = failure_eval[idx].ravel()

    pa = pred_MA_matrix[idx].ravel()
    pb = pred_MB_matrix[idx].ravel()

    if np.unique(y_b).size < 2:
        continue

    ap_a = average_precision_score(
        y_b,
        pa,
    )

    ap_b = average_precision_score(
        y_b,
        pb,
    )

    roc_a = roc_auc_score(
        y_b,
        pa,
    )

    roc_b = roc_auc_score(
        y_b,
        pb,
    )

    ll_a = log_loss(
        y_b,
        pa,
        labels=[0, 1],
    )

    ll_b = log_loss(
        y_b,
        pb,
        labels=[0, 1],
    )

    br_a = brier_score_loss(
        y_b,
        pa,
    )

    br_b = brier_score_loss(
        y_b,
        pb,
    )

    records.append({
        "bootstrap": b,

        "delta_ap":
            ap_b - ap_a,

        "delta_roc_auc":
            roc_b - roc_a,

        "delta_log_loss":
            ll_b - ll_a,

        "delta_brier":
            br_b - br_a,
    })


boot = pd.DataFrame(records)

boot.to_csv(
    OUTPUT_DIR /
    "bootstrap_deltas.csv",
    index=False,
)


# ============================================================
# Bootstrap summary
# ============================================================

summary = {}

for metric in [
    "delta_ap",
    "delta_roc_auc",
    "delta_log_loss",
    "delta_brier",
]:

    values = boot[metric].to_numpy()

    low, high = np.percentile(
        values,
        [2.5, 97.5],
    )

    summary[metric] = {
        "observed": float(
            delta_MB_MA[
                metric.replace(
                    "delta_",
                    ""
                )
            ]
        ),

        "bootstrap_mean":
            float(values.mean()),

        "ci_low":
            float(low),

        "ci_high":
            float(high),
    }


summary["delta_ap"][
    "improvement_fraction"
] = float(
    (boot["delta_ap"] > 0).mean()
)

summary["bootstrap"] = {
    "replicates_requested":
        N_BOOTSTRAP,

    "replicates_valid":
        int(len(boot)),

    "unit":
        "whole protein",

    "models_refitted":
        False,

    "seed":
        SEED,
}


# ============================================================
# Save metadata
# ============================================================

run_summary = {
    "analysis":
        "global_confidence_incremental_reliability_v1",

    "population":
        "structure-eligible Graph-V2 "
        "calibration/calibration-eval",

    "calibration_proteins":
        int(Y_cal.shape[0]),

    "calibration_eval_proteins":
        int(Y_eval.shape[0]),

    "labels":
        int(Y_eval.shape[1]),

    "failure_definition":
        "abs(y - p_struct) >= 0.5",

    "global_confidence_source":
        str(AF_PATH),

    "global_confidence_field":
        "global_metric",

    "global_confidence_publication_name":
        "AlphaFold global confidence; "
        "exact metadata semantics pending verification",

    "M0":
        "[U_struct, U_seq]",

    "MA":
        "[U_struct, U_seq, global_confidence]",

    "MB":
        "[U_struct, U_seq, global_confidence, D]",

    "D":
        "abs(p_seq - p_struct)",

    "primary_comparison":
        "MB - MA",

    "primary_metric":
        "average_precision",

    "test_used":
        False,

    "metrics": {
        "M0":
            metrics_M0,

        "MA":
            metrics_MA,

        "MB":
            metrics_MB,
    },

    "delta_MA_minus_M0":
        delta_MA_M0,

    "delta_MB_minus_MA":
        delta_MB_MA,

    "bootstrap":
        summary,
}


with open(
    OUTPUT_DIR /
    "analysis_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        run_summary,
        f,
        indent=2,
    )


# ============================================================
# Final report
# ============================================================

print()
print("=" * 72)
print("BOOTSTRAP — PRIMARY MB - MA COMPARISON")
print("=" * 72)

for metric in [
    "delta_ap",
    "delta_roc_auc",
    "delta_log_loss",
    "delta_brier",
]:

    r = summary[metric]

    print(
        f"{metric:16s} "
        f"observed={r['observed']:+.6f} "
        f"95% CI=[{r['ci_low']:+.6f}, "
        f"{r['ci_high']:+.6f}]"
    )

print()
print(
    "AP improvement fraction: "
    f"{summary['delta_ap']['improvement_fraction']:.3f}"
)

print()
print(
    f"Saved to: {OUTPUT_DIR}"
)

print("TEST DATA ACCESSED: NO")