from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

PRED_DIR = Path(
    "data/models/structure_gnn_v1_full/val_predictions"
)

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

OUT_DIR = Path(
    "data/calibration/"
    "structure_gnn_v1_global_temperature"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VAL_IDS_PATH = (
    PRED_DIR
    / "val_structure_eligible_ids.txt"
)

VAL_LOGITS_PATH = (
    PRED_DIR
    / "val_logits.npy"
)

VAL_LABELS_PATH = (
    PRED_DIR
    / "val_labels.npy"
)

CAL_IDS_PATH = (
    SPLIT_DIR
    / "structure_calibration_ids.txt"
)

CALEVAL_IDS_PATH = (
    SPLIT_DIR
    / "structure_cal_eval_ids.txt"
)


OUT_JSON = (
    OUT_DIR
    / "temperature_fit.json"
)


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):

    with open(
        path,
        encoding="utf-8",
    ) as f:

        return [
            line.strip()
            for line in f
            if line.strip()
        ]


# ============================================================
# LOAD
# ============================================================

print("=" * 82)
print("STRUCTURE GNN V1 — GLOBAL TEMPERATURE FIT")
print("=" * 82)

print("Calibration method: one global scalar temperature")
print("Calibration-eval labels used for fitting: NO")
print("Sequence temperature reused: NO")
print("TEST USED: NO")
print()


val_ids = read_ids(
    VAL_IDS_PATH
)

cal_ids = read_ids(
    CAL_IDS_PATH
)

caleval_ids = read_ids(
    CALEVAL_IDS_PATH
)


logits = np.load(
    VAL_LOGITS_PATH
)

labels = np.load(
    VAL_LABELS_PATH
)


if len(val_ids) != 4742:
    raise RuntimeError(
        "Expected 4,742 eligible VAL IDs."
    )


if logits.shape != (4742, 494):
    raise RuntimeError(
        f"Unexpected logits shape: "
        f"{logits.shape}"
    )


if labels.shape != (4742, 494):
    raise RuntimeError(
        f"Unexpected labels shape: "
        f"{labels.shape}"
    )


if len(cal_ids) != 2369:
    raise RuntimeError(
        f"Expected 2,369 calibration proteins, "
        f"got {len(cal_ids)}"
    )


if len(caleval_ids) != 2373:
    raise RuntimeError(
        f"Expected 2,373 calibration-eval proteins, "
        f"got {len(caleval_ids)}"
    )


if set(cal_ids) & set(caleval_ids):
    raise RuntimeError(
        "Calibration/cal-eval overlap."
    )


index = {
    uid: i
    for i, uid in enumerate(val_ids)
}


if not set(cal_ids).issubset(index):
    raise RuntimeError(
        "Calibration IDs missing from VAL predictions."
    )


if not set(caleval_ids).issubset(index):
    raise RuntimeError(
        "Calibration-eval IDs missing from VAL predictions."
    )


cal_rows = np.asarray(
    [
        index[uid]
        for uid in cal_ids
    ],
    dtype=np.int64,
)


# ============================================================
# CALIBRATION DATA ONLY
# ============================================================

cal_logits_np = np.asarray(
    logits[cal_rows],
    dtype=np.float32,
)

cal_labels_np = np.asarray(
    labels[cal_rows],
    dtype=np.float32,
)


if cal_logits_np.shape != (
    2369,
    494,
):
    raise RuntimeError(
        "Calibration logits shape mismatch."
    )


if not np.isfinite(
    cal_logits_np
).all():

    raise RuntimeError(
        "Non-finite calibration logits."
    )


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


cal_logits = torch.from_numpy(
    cal_logits_np
).to(device)


cal_labels = torch.from_numpy(
    cal_labels_np
).to(device)


# ============================================================
# BEFORE CALIBRATION
# ============================================================

with torch.no_grad():

    bce_before = (
        F.binary_cross_entropy_with_logits(
            cal_logits,
            cal_labels,
        ).item()
    )


# ============================================================
# FIT log(T)
#
# T = exp(log_temperature), guaranteeing T > 0.
# Only calibration subset is used.
# ============================================================

log_temperature = torch.nn.Parameter(
    torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    )
)


optimizer = torch.optim.LBFGS(
    [log_temperature],
    lr=0.1,
    max_iter=100,
    tolerance_grad=1e-10,
    tolerance_change=1e-12,
    line_search_fn="strong_wolfe",
)


def closure():

    optimizer.zero_grad()

    temperature = torch.exp(
        log_temperature
    )

    scaled_logits = (
        cal_logits
        / temperature
    )

    loss = (
        F.binary_cross_entropy_with_logits(
            scaled_logits,
            cal_labels,
        )
    )

    loss.backward()

    return loss


optimizer.step(
    closure
)


temperature = float(
    torch.exp(
        log_temperature
    ).detach().cpu()
)


if (
    not np.isfinite(temperature)
    or temperature <= 0
):
    raise RuntimeError(
        "Invalid fitted temperature."
    )


# ============================================================
# AFTER CALIBRATION
# ============================================================

with torch.no_grad():

    scaled = (
        cal_logits
        / temperature
    )

    bce_after = (
        F.binary_cross_entropy_with_logits(
            scaled,
            cal_labels,
        ).item()
    )


delta = (
    bce_after
    - bce_before
)


# ============================================================
# SAVE
# ============================================================

result = {
    "model":
        "structure_gnn_v1_full",

    "calibration_method":
        "global temperature scaling",

    "parameterization":
        "T = exp(log_temperature)",

    "temperature":
        temperature,

    "calibration_proteins":
        len(cal_ids),

    "labels":
        494,

    "positive_annotations":
        int(cal_labels_np.sum()),

    "calibration_bce_before":
        float(bce_before),

    "calibration_bce_after":
        float(bce_after),

    "calibration_bce_delta_after_minus_before":
        float(delta),

    "calibration_eval_used_for_fitting":
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
        result,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print(
    f"Device:                  "
    f"{device}"
)

print(
    f"Calibration proteins:    "
    f"{len(cal_ids):,}"
)

print(
    f"Positive annotations:    "
    f"{int(cal_labels_np.sum()):,}"
)

print()

print(
    f"BCE before:              "
    f"{bce_before:.8f}"
)

print(
    f"Fitted temperature T:    "
    f"{temperature:.8f}"
)

print(
    f"BCE after:               "
    f"{bce_after:.8f}"
)

print(
    f"Delta (after - before):  "
    f"{delta:+.8f}"
)

print()

print(
    "Calibration-eval used for fitting: NO"
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

print("=" * 82)