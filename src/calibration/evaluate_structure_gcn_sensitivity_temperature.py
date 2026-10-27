from pathlib import Path
import json

import numpy as np
from sklearn.metrics import average_precision_score


# ============================================================
# PATHS
# ============================================================

PRED_DIR = Path(
    "data/analysis/backbone_sensitivity_v1/"
    "gcn_val_inference"
)

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

CAL_DIR = Path(
    "data/analysis/backbone_sensitivity_v1/"
    "gcn_calibration"
)

OUT_DIR = CAL_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


VAL_IDS_PATH = (
    PRED_DIR
    / "val_structure_eligible_ids.txt"
)

LOGITS_PATH = (
    PRED_DIR
    / "val_raw_logits.npy"
)

LABELS_PATH = (
    SPLIT_DIR
    / "structure_cal_eval_labels.npy"
)

EVAL_IDS_PATH = (
    SPLIT_DIR
    / "structure_cal_eval_ids.txt"
)

TEMP_PATH = (
    CAL_DIR
    / "temperature_fit.json"
)

OUT_JSON = (
    OUT_DIR
    / "calibration_eval_metrics.json"
)


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)

    out = np.empty_like(x)

    positive = x >= 0
    negative = ~positive

    out[positive] = (
        1.0
        / (1.0 + np.exp(-x[positive]))
    )

    exp_x = np.exp(x[negative])

    out[negative] = (
        exp_x
        / (1.0 + exp_x)
    )

    return out


def binary_nll_from_probs(y, p):
    eps = 1e-12

    p = np.clip(
        p,
        eps,
        1.0 - eps,
    )

    return float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y) * np.log(1.0 - p)
        )
    )


def brier_score(y, p):
    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def ece_binary_flat(y, p, n_bins=15):
    """
    Global binary ECE after flattening all
    protein-label predictions.
    """

    y = y.reshape(-1)
    p = p.reshape(-1)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    total = len(p)

    for i in range(n_bins):

        lo = edges[i]
        hi = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (p >= lo)
                & (p <= hi)
            )
        else:
            mask = (
                (p >= lo)
                & (p < hi)
            )

        n = int(mask.sum())

        if n == 0:
            continue

        confidence = float(
            p[mask].mean()
        )

        accuracy = float(
            y[mask].mean()
        )

        ece += (
            n / total
        ) * abs(
            confidence - accuracy
        )

    return float(ece)


# ============================================================
# LOAD
# ============================================================

print("=" * 84)
print("E12 ALTERNATIVE GCN — HELD-OUT GLOBAL TEMPERATURE EVALUATION")
print("=" * 84)

print("Temperature fitting subset used here: NO")
print("Calibration-eval subset: YES")
print("Threshold optimization: NO")
print("Sequence temperature reused: NO")
print("TEST USED: NO")
print()


val_ids = read_ids(
    VAL_IDS_PATH
)

eval_ids = read_ids(
    EVAL_IDS_PATH
)

logits = np.load(
    LOGITS_PATH
)

labels = np.load(
    LABELS_PATH
)


with open(
    TEMP_PATH,
    encoding="utf-8",
) as f:

    temp_info = json.load(f)


temperature = float(
    temp_info["temperature"]
)


# ============================================================
# VALIDATION
# ============================================================

if len(val_ids) != 4742:
    raise RuntimeError(
        f"Expected 4,742 VAL IDs, "
        f"got {len(val_ids)}"
    )


if len(eval_ids) != 2373:
    raise RuntimeError(
        f"Expected 2,373 cal-eval IDs, "
        f"got {len(eval_ids)}"
    )


if logits.shape != (4742, 494):
    raise RuntimeError(
        f"Unexpected logits shape: "
        f"{logits.shape}"
    )


if labels.shape != (2373, 494):
    raise RuntimeError(
        f"Unexpected CAL-EVAL labels shape: "
        f"{labels.shape}"
    )


if (
    not np.isfinite(temperature)
    or temperature <= 0
):
    raise RuntimeError(
        f"Invalid temperature: "
        f"{temperature}"
    )


index = {
    uid: i
    for i, uid in enumerate(val_ids)
}


missing = [
    uid
    for uid in eval_ids
    if uid not in index
]

if missing:
    raise RuntimeError(
        f"{len(missing)} cal-eval IDs "
        f"missing from predictions."
    )


rows = np.asarray(
    [
        index[uid]
        for uid in eval_ids
    ],
    dtype=np.int64,
)


eval_logits = np.asarray(
    logits[rows],
    dtype=np.float64,
)

eval_labels = np.asarray(
    labels,
    dtype=np.uint8,
)


# ============================================================
# PROBABILITIES
# ============================================================

probs_before = sigmoid(
    eval_logits
)

probs_after = sigmoid(
    eval_logits / temperature
)


# ============================================================
# RELIABILITY METRICS
# ============================================================

nll_before = binary_nll_from_probs(
    eval_labels,
    probs_before,
)

nll_after = binary_nll_from_probs(
    eval_labels,
    probs_after,
)


brier_before = brier_score(
    eval_labels,
    probs_before,
)

brier_after = brier_score(
    eval_labels,
    probs_after,
)


ece_before = ece_binary_flat(
    eval_labels,
    probs_before,
    n_bins=15,
)

ece_after = ece_binary_flat(
    eval_labels,
    probs_after,
    n_bins=15,
)


# ============================================================
# DISCRIMINATION METRICS
# ============================================================

support = eval_labels.sum(axis=0)

supported_mask = (
    support > 0
)

n_supported = int(
    supported_mask.sum()
)

n_zero = int(
    (~supported_mask).sum()
)


micro_before = float(
    average_precision_score(
        eval_labels.ravel(),
        probs_before.ravel(),
    )
)

micro_after = float(
    average_precision_score(
        eval_labels.ravel(),
        probs_after.ravel(),
    )
)


per_label_before = np.full(
    494,
    np.nan,
    dtype=np.float64,
)

per_label_after = np.full(
    494,
    np.nan,
    dtype=np.float64,
)


for j in np.where(
    supported_mask
)[0]:

    per_label_before[j] = (
        average_precision_score(
            eval_labels[:, j],
            probs_before[:, j],
        )
    )

    per_label_after[j] = (
        average_precision_score(
            eval_labels[:, j],
            probs_after[:, j],
        )
    )


macro_before = float(
    np.nanmean(
        per_label_before
    )
)

macro_after = float(
    np.nanmean(
        per_label_after
    )
)


max_ap_difference = float(
    np.nanmax(
        np.abs(
            per_label_before
            - per_label_after
        )
    )
)


# ============================================================
# SAVE
# ============================================================

np.save(
    OUT_DIR
    / "cal_eval_probs_before.npy",
    probs_before.astype(np.float32),
)

np.save(
    OUT_DIR
    / "cal_eval_probs_after.npy",
    probs_after.astype(np.float32),
)

np.save(
    OUT_DIR
    / "cal_eval_per_label_ap_before.npy",
    per_label_before,
)

np.save(
    OUT_DIR
    / "cal_eval_per_label_ap_after.npy",
    per_label_after,
)


results = {
    "model":
        "structure_gcn_sensitivity_v1_full",

    "calibration_method":
        "global temperature scaling",

    "temperature":
        temperature,

    "calibration_eval_proteins":
        len(eval_ids),

    "positive_annotations":
        int(eval_labels.sum()),

    "supported_labels":
        n_supported,

    "zero_positive_labels":
        n_zero,

    "nll_before":
        nll_before,

    "nll_after":
        nll_after,

    "nll_delta_after_minus_before":
        nll_after - nll_before,

    "brier_before":
        brier_before,

    "brier_after":
        brier_after,

    "brier_delta_after_minus_before":
        brier_after - brier_before,

    "ece_15_before":
        ece_before,

    "ece_15_after":
        ece_after,

    "ece_15_delta_after_minus_before":
        ece_after - ece_before,

    "micro_auprc_before":
        micro_before,

    "micro_auprc_after":
        micro_after,

    "macro_auprc_before":
        macro_before,

    "macro_auprc_after":
        macro_after,

    "max_per_label_ap_difference":
        max_ap_difference,

    "calibration_eval_used_for_temperature_fitting":
        False,

    "threshold_optimization":
        False,

    "sequence_temperature_reused":
        False,

    "sequence_expert_modified":
        False,

    "test_used":
        False,
}


with open(
    OUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print(
    f"Temperature T:       "
    f"{temperature:.8f}"
)

print(
    f"Proteins:            "
    f"{len(eval_ids):,}"
)

print(
    f"Positive annotations:"
    f" {int(eval_labels.sum()):,}"
)

print(
    f"Supported labels:    "
    f"{n_supported}/494"
)

print(
    f"Zero-positive labels:"
    f" {n_zero}"
)

print()

print("Reliability metrics:")
print(
    f"  NLL/BCE: "
    f"{nll_before:.8f}"
    f" -> {nll_after:.8f}"
    f"  delta="
    f"{nll_after - nll_before:+.8f}"
)

print(
    f"  Brier:   "
    f"{brier_before:.8f}"
    f" -> {brier_after:.8f}"
    f"  delta="
    f"{brier_after - brier_before:+.8f}"
)

print(
    f"  ECE-15:  "
    f"{ece_before:.8f}"
    f" -> {ece_after:.8f}"
    f"  delta="
    f"{ece_after - ece_before:+.8f}"
)

print()

print("Discrimination metrics:")
print(
    f"  Micro AUPRC: "
    f"{micro_before:.8f}"
    f" -> {micro_after:.8f}"
)

print(
    f"  Macro AUPRC: "
    f"{macro_before:.8f}"
    f" -> {macro_after:.8f}"
)

print(
    f"  Max per-label AP difference: "
    f"{max_ap_difference:.12f}"
)

print()

print(
    "Calibration-eval used for temperature fitting: NO"
)

print(
    "Threshold optimization: NO"
)

print(
    "Sequence temperature reused: NO"
)

print(
    "Sequence expert modified: NO"
)

print(
    "TEST USED: NO"
)

print()

print(
    f"Saved:\n  {OUT_JSON}"
)

print("=" * 84)