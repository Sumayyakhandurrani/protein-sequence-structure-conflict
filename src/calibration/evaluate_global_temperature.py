from pathlib import Path
import json

import numpy as np
from sklearn.metrics import average_precision_score


# ============================================================
# PATHS
# ============================================================

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2"
)

LABEL_DIR = SPLIT_DIR / "labels"
CAL_SPLIT_DIR = SPLIT_DIR / "calibration_v1"

CAL_MODEL_DIR = Path(
    "data/calibration/sequence_mlp_v1_global_temperature"
)

OUTPUT_DIR = CAL_MODEL_DIR


# ============================================================
# CONFIG
# ============================================================

N_BINS = 15


# ============================================================
# HELPERS
# ============================================================

def binary_nll(y, p, eps=1e-7):

    p = np.clip(
        p.astype(np.float64),
        eps,
        1.0 - eps
    )

    y = y.astype(np.float64)

    return float(
        -np.mean(
            y * np.log(p)
            +
            (1.0 - y)
            * np.log(1.0 - p)
        )
    )


def brier_score(y, p):

    y = y.astype(np.float64)
    p = p.astype(np.float64)

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def ece_equal_width(y, p, n_bins=15):

    y = y.reshape(-1).astype(np.float64)
    p = p.reshape(-1).astype(np.float64)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    total = len(p)

    ece = 0.0
    rows = []

    for i in range(n_bins):

        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (p >= lower)
                & (p <= upper)
            )

        else:

            mask = (
                (p >= lower)
                & (p < upper)
            )

        count = int(mask.sum())

        if count == 0:

            mean_conf = np.nan
            empirical_rate = np.nan
            gap = np.nan

        else:

            mean_conf = float(
                p[mask].mean()
            )

            empirical_rate = float(
                y[mask].mean()
            )

            gap = abs(
                mean_conf
                - empirical_rate
            )

            ece += (
                count / total
            ) * gap

        rows.append(
            {
                "bin": i,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": mean_conf,
                "empirical_positive_rate":
                    empirical_rate,
                "absolute_gap": gap,
            }
        )

    return float(ece), rows


def per_label_ap(y, p):

    support = y.sum(axis=0)

    supported = (
        support > 0
    )

    ap = np.full(
        y.shape[1],
        np.nan,
        dtype=np.float64
    )

    for j in np.where(supported)[0]:

        ap[j] = average_precision_score(
            y[:, j],
            p[:, j]
        )

    return ap, supported


# ============================================================
# LOAD
# ============================================================

print("=" * 78)
print("GLOBAL TEMPERATURE — HELD-OUT CALIBRATION EVALUATION")
print("=" * 78)

Y_val = np.load(
    LABEL_DIR / "val_labels.npy"
)

eval_idx = np.load(
    CAL_SPLIT_DIR
    / "calibration_eval_val_indices.npy"
)

P_before = np.load(
    CAL_MODEL_DIR
    / "calibration_eval_probabilities_before.npy"
)

P_after = np.load(
    CAL_MODEL_DIR
    / "calibration_eval_probabilities_after.npy"
)

with open(
    CAL_MODEL_DIR
    / "temperature_metadata.json"
) as f:

    temperature_metadata = json.load(f)

temperature = float(
    temperature_metadata["temperature"]
)


# ============================================================
# SAFETY
# ============================================================

Y_eval = Y_val[eval_idx]

assert Y_eval.shape == (2377, 494)
assert P_before.shape == (2377, 494)
assert P_after.shape == (2377, 494)

assert np.isfinite(P_before).all()
assert np.isfinite(P_after).all()

assert (
    (P_before >= 0).all()
    and
    (P_before <= 1).all()
)

assert (
    (P_after >= 0).all()
    and
    (P_after <= 1).all()
)

print(
    "\nTemperature:",
    f"{temperature:.8f}"
)

print(
    "Calibration-eval proteins:",
    len(Y_eval)
)

print(
    "Labels:",
    Y_eval.shape[1]
)

print(
    "Positive annotations:",
    int(Y_eval.sum())
)

print(
    "Labels with >=1 positive:",
    int(
        (
            Y_eval.sum(axis=0) > 0
        ).sum()
    )
)

print(
    "Labels with zero positives:",
    int(
        (
            Y_eval.sum(axis=0) == 0
        ).sum()
    )
)

print("\nIntegrity checks: PASS")
print("CALIBRATION-EVAL USED TO FIT T: NO")
print("TEST SET USED: NO")


# ============================================================
# NLL
# ============================================================

nll_before = binary_nll(
    Y_eval,
    P_before
)

nll_after = binary_nll(
    Y_eval,
    P_after
)


# ============================================================
# BRIER
# ============================================================

brier_before = brier_score(
    Y_eval,
    P_before
)

brier_after = brier_score(
    Y_eval,
    P_after
)


# ============================================================
# ECE
# ============================================================

ece_before, bins_before = (
    ece_equal_width(
        Y_eval,
        P_before,
        N_BINS
    )
)

ece_after, bins_after = (
    ece_equal_width(
        Y_eval,
        P_after,
        N_BINS
    )
)


# ============================================================
# AUPRC
# ============================================================

micro_ap_before = (
    average_precision_score(
        Y_eval,
        P_before,
        average="micro"
    )
)

micro_ap_after = (
    average_precision_score(
        Y_eval,
        P_after,
        average="micro"
    )
)

ap_before, supported = (
    per_label_ap(
        Y_eval,
        P_before
    )
)

ap_after, supported_after = (
    per_label_ap(
        Y_eval,
        P_after
    )
)

assert np.array_equal(
    supported,
    supported_after
)

macro_ap_before = float(
    np.nanmean(
        ap_before
    )
)

macro_ap_after = float(
    np.nanmean(
        ap_after
    )
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 78)
print("HELD-OUT CALIBRATION METRICS")
print("=" * 78)

print(
    f"{'Metric':<20}"
    f"{'Before':>14}"
    f"{'After':>14}"
    f"{'Delta':>14}"
)

print("-" * 62)

metrics = [
    (
        "NLL/BCE",
        nll_before,
        nll_after,
    ),
    (
        "Brier",
        brier_before,
        brier_after,
    ),
    (
        "ECE-15",
        ece_before,
        ece_after,
    ),
    (
        "Micro AUPRC",
        micro_ap_before,
        micro_ap_after,
    ),
    (
        "Macro AUPRC",
        macro_ap_before,
        macro_ap_after,
    ),
]

for name, before, after in metrics:

    print(
        f"{name:<20}"
        f"{before:>14.8f}"
        f"{after:>14.8f}"
        f"{after - before:>+14.8f}"
    )


# ============================================================
# RANKING INVARIANCE
# ============================================================

max_ap_difference = float(
    np.nanmax(
        np.abs(
            ap_before
            - ap_after
        )
    )
)

print("\n" + "=" * 78)
print("RANKING INVARIANCE CHECK")
print("=" * 78)

print(
    "Max absolute per-label AP difference:",
    f"{max_ap_difference:.12f}"
)

print(
    "Micro AUPRC difference:",
    f"{micro_ap_after - micro_ap_before:+.12f}"
)

print(
    "Macro AUPRC difference:",
    f"{macro_ap_after - macro_ap_before:+.12f}"
)

print(
    "\nNOTE:"
)

print(
    "Per-label ranking should remain unchanged "
    "under positive global temperature scaling."
)

print(
    "Micro AUPRC can change because global scaling "
    "changes cross-label score ordering."
)


# ============================================================
# SAVE RELIABILITY TABLES
# ============================================================

import pandas as pd

pd.DataFrame(
    bins_before
).to_csv(
    OUTPUT_DIR
    / "calibration_eval_reliability_before.csv",
    index=False
)

pd.DataFrame(
    bins_after
).to_csv(
    OUTPUT_DIR
    / "calibration_eval_reliability_after.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "method":
        "global_temperature_scaling",

    "temperature":
        temperature,

    "evaluation_subset":
        "calibration_eval",

    "evaluation_proteins":
        int(len(Y_eval)),

    "supported_labels":
        int(supported.sum()),

    "zero_positive_labels":
        int((~supported).sum()),

    "nll": {
        "before": nll_before,
        "after": nll_after,
        "delta": nll_after - nll_before,
    },

    "brier": {
        "before": brier_before,
        "after": brier_after,
        "delta": brier_after - brier_before,
    },

    "ece_15": {
        "before": ece_before,
        "after": ece_after,
        "delta": ece_after - ece_before,
    },

    "micro_auprc": {
        "before":
            float(micro_ap_before),

        "after":
            float(micro_ap_after),

        "delta":
            float(
                micro_ap_after
                - micro_ap_before
            ),
    },

    "macro_auprc": {
        "before":
            macro_ap_before,

        "after":
            macro_ap_after,

        "delta":
            macro_ap_after
            - macro_ap_before,
    },

    "max_absolute_per_label_ap_difference":
        max_ap_difference,

    "calibration_eval_used_to_fit_temperature":
        False,

    "test_used":
        False,
}


with open(
    OUTPUT_DIR
    / "calibration_eval_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\nSaved:")

print(
    OUTPUT_DIR
    / "calibration_eval_reliability_before.csv"
)

print(
    OUTPUT_DIR
    / "calibration_eval_reliability_after.csv"
)

print(
    OUTPUT_DIR
    / "calibration_eval_summary.json"
)

print("\nIMPORTANT:")
print(
    "Temperature was fitted ONLY on calibration."
)
print(
    "Metrics above are ONLY calibration-eval."
)
print(
    "No threshold optimization performed."
)
print(
    "TEST SET USED: NO"
)

print("=" * 78)