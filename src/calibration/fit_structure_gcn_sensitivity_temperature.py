from pathlib import Path
import json
import hashlib

import numpy as np
import torch
import torch.nn.functional as F


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

OUT_DIR = Path(
    "data/analysis/backbone_sensitivity_v1/"
    "gcn_calibration"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VAL_IDS_PATH = (
    PRED_DIR / "val_structure_eligible_ids.txt"
)

VAL_LOGITS_PATH = (
    PRED_DIR / "val_raw_logits.npy"
)

CAL_IDS_PATH = (
    SPLIT_DIR / "structure_calibration_ids.txt"
)

CAL_LABELS_PATH = (
    SPLIT_DIR / "structure_calibration_labels.npy"
)

CALEVAL_IDS_PATH = (
    SPLIT_DIR / "structure_cal_eval_ids.txt"
)

OUT_JSON = (
    OUT_DIR / "temperature_fit.json"
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


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# LOAD
# ============================================================

print("=" * 84)
print("E12 ALTERNATIVE GCN — GLOBAL TEMPERATURE FIT")
print("=" * 84)

print("Calibration method: one global scalar temperature")
print("Frozen CAL assignment reused: YES")
print("CAL-EVAL labels used for fitting: NO")
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

cal_labels_np = np.load(
    CAL_LABELS_PATH
).astype(np.float32)


# ============================================================
# VALIDATION
# ============================================================

if len(val_ids) != 4742:
    raise RuntimeError(
        f"Expected 4,742 VAL IDs, got {len(val_ids)}."
    )

if logits.shape != (4742, 494):
    raise RuntimeError(
        f"Unexpected logits shape: {logits.shape}"
    )

if len(cal_ids) != 2369:
    raise RuntimeError(
        f"Expected 2,369 CAL IDs, got {len(cal_ids)}."
    )

if len(caleval_ids) != 2373:
    raise RuntimeError(
        f"Expected 2,373 CAL-EVAL IDs, "
        f"got {len(caleval_ids)}."
    )

if cal_labels_np.shape != (2369, 494):
    raise RuntimeError(
        f"Unexpected CAL labels shape: "
        f"{cal_labels_np.shape}"
    )

if set(cal_ids) & set(caleval_ids):
    raise RuntimeError(
        "CAL/CAL-EVAL overlap detected."
    )

if not np.isfinite(logits).all():
    raise RuntimeError(
        "Non-finite VAL logits detected."
    )

index = {
    uid: i
    for i, uid in enumerate(val_ids)
}

if not set(cal_ids).issubset(index):
    raise RuntimeError(
        "CAL IDs missing from GCN VAL predictions."
    )

if not set(caleval_ids).issubset(index):
    raise RuntimeError(
        "CAL-EVAL IDs missing from GCN VAL predictions."
    )


cal_rows = np.asarray(
    [
        index[uid]
        for uid in cal_ids
    ],
    dtype=np.int64,
)

cal_logits_np = np.asarray(
    logits[cal_rows],
    dtype=np.float32,
)

if cal_logits_np.shape != (2369, 494):
    raise RuntimeError(
        "CAL logits shape mismatch."
    )


# ============================================================
# FIT GLOBAL TEMPERATURE ON CAL ONLY
# ============================================================

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


with torch.no_grad():
    bce_before = (
        F.binary_cross_entropy_with_logits(
            cal_logits,
            cal_labels,
        ).item()
    )


# T = exp(log_temperature), so T > 0.
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
        cal_logits / temperature
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


with torch.no_grad():

    bce_after = (
        F.binary_cross_entropy_with_logits(
            cal_logits / temperature,
            cal_labels,
        ).item()
    )


delta = (
    bce_after - bce_before
)


# ============================================================
# SAVE
# ============================================================

result = {
    "model":
        "structure_gcn_sensitivity_v1_full",

    "scientific_role":
        "E12 alternative-structure-backbone sensitivity",

    "calibration_method":
        "global temperature scaling",

    "parameterization":
        "T = exp(log_temperature)",

    "temperature":
        temperature,

    "calibration_proteins":
        2369,

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

    "frozen_calibration_assignment_reused":
        True,

    "new_calibration_split_created":
        False,

    "sequence_temperature_reused":
        False,

    "sequence_expert_modified":
        False,

    "test_used":
        False,

    "val_ids_sha256":
        sha256_file(VAL_IDS_PATH),

    "val_logits_sha256":
        sha256_file(VAL_LOGITS_PATH),

    "cal_ids_sha256":
        sha256_file(CAL_IDS_PATH),

    "cal_labels_sha256":
        sha256_file(CAL_LABELS_PATH),
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

print("Device:", device)
print("Calibration proteins:", len(cal_ids))
print("CAL-EVAL proteins held out:", len(caleval_ids))
print("Positive annotations:", int(cal_labels_np.sum()))
print()

print(f"BCE before:           {bce_before:.8f}")
print(f"Fitted temperature T: {temperature:.8f}")
print(f"BCE after:            {bce_after:.8f}")
print(f"Delta after-before:   {delta:+.8f}")

print()
print("CAL-EVAL used for fitting: NO")
print("New calibration split created: NO")
print("Sequence expert modified: NO")
print("TEST USED: NO")
print()
print("Saved:", OUT_JSON)
print("=" * 84)
