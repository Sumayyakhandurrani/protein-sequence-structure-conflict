from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path(
    "data/models/sequence_mlp_v1"
)

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2"
)

LABEL_DIR = SPLIT_DIR / "labels"
CAL_DIR = SPLIT_DIR / "calibration_v1"

OUTPUT_DIR = (
    Path("data/calibration")
    / "sequence_mlp_v1_global_temperature"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

print("=" * 76)
print("SEQUENCE MLP V1 — GLOBAL TEMPERATURE SCALING")
print("=" * 76)

val_logits = np.load(
    MODEL_DIR / "val_logits.npy"
).astype(np.float32)

Y_val = np.load(
    LABEL_DIR / "val_labels.npy"
).astype(np.float32)

cal_idx = np.load(
    CAL_DIR / "calibration_val_indices.npy"
)

eval_idx = np.load(
    CAL_DIR / "calibration_eval_val_indices.npy"
)


# ============================================================
# SAFETY CHECKS
# ============================================================

assert val_logits.shape == (4754, 494)
assert Y_val.shape == (4754, 494)

assert len(cal_idx) == 2377
assert len(eval_idx) == 2377

assert np.isfinite(val_logits).all()

assert np.intersect1d(
    cal_idx,
    eval_idx
).size == 0

assert np.array_equal(
    np.sort(
        np.concatenate(
            [cal_idx, eval_idx]
        )
    ),
    np.arange(4754)
)

print("\nVAL logits:", val_logits.shape)
print("VAL labels:", Y_val.shape)

print(
    "Calibration proteins:",
    len(cal_idx)
)

print(
    "Calibration-eval proteins:",
    len(eval_idx)
)

print("Integrity checks: PASS")
print("TEST SET USED: NO")


# ============================================================
# CALIBRATION DATA ONLY
# ============================================================

cal_logits = torch.tensor(
    val_logits[cal_idx],
    dtype=torch.float32
)

cal_targets = torch.tensor(
    Y_val[cal_idx],
    dtype=torch.float32
)


# ============================================================
# BASELINE CALIBRATION NLL
# ============================================================

with torch.no_grad():

    nll_before = F.binary_cross_entropy_with_logits(
        cal_logits,
        cal_targets
    ).item()


print("\n" + "=" * 76)
print("FIT GLOBAL TEMPERATURE")
print("=" * 76)

print(
    f"Calibration BCE/NLL before: "
    f"{nll_before:.8f}"
)


# ============================================================
# OPTIMIZE LOG TEMPERATURE
#
# T = exp(log_temperature), therefore T is always positive.
# ============================================================

log_temperature = torch.nn.Parameter(
    torch.zeros(
        1,
        dtype=torch.float32
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

    loss = F.binary_cross_entropy_with_logits(
        scaled_logits,
        cal_targets
    )

    loss.backward()

    return loss


optimizer.step(closure)


# ============================================================
# RESULT
# ============================================================

temperature = float(
    torch.exp(
        log_temperature
    ).detach().cpu().item()
)

assert np.isfinite(temperature)
assert temperature > 0


with torch.no_grad():

    calibrated_cal_logits = (
        cal_logits
        / temperature
    )

    nll_after = (
        F.binary_cross_entropy_with_logits(
            calibrated_cal_logits,
            cal_targets
        )
        .item()
    )


print(
    f"Learned temperature T: "
    f"{temperature:.8f}"
)

print(
    f"Calibration BCE/NLL after:  "
    f"{nll_after:.8f}"
)

print(
    f"Change: "
    f"{nll_after - nll_before:+.8f}"
)


# ============================================================
# APPLY TO CAL-EVAL
#
# Important:
# We APPLY here only so that the arrays are ready for the next
# evaluation script. We do NOT use cal-eval labels to fit T.
# ============================================================

eval_logits = (
    val_logits[eval_idx]
)

eval_logits_calibrated = (
    eval_logits
    / temperature
).astype(np.float32)


def sigmoid(x):

    return (
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    )


eval_prob_before = sigmoid(
    eval_logits
).astype(np.float32)

eval_prob_after = sigmoid(
    eval_logits_calibrated
).astype(np.float32)

assert np.isfinite(
    eval_prob_before
).all()

assert np.isfinite(
    eval_prob_after
).all()


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_DIR
    / "calibration_eval_logits_before.npy",
    eval_logits.astype(np.float32)
)

np.save(
    OUTPUT_DIR
    / "calibration_eval_logits_after.npy",
    eval_logits_calibrated
)

np.save(
    OUTPUT_DIR
    / "calibration_eval_probabilities_before.npy",
    eval_prob_before
)

np.save(
    OUTPUT_DIR
    / "calibration_eval_probabilities_after.npy",
    eval_prob_after
)


metadata = {

    "model":
        "sequence_mlp_v1",

    "method":
        "global_temperature_scaling",

    "temperature":
        temperature,

    "fit_subset":
        "calibration",

    "fit_proteins":
        int(len(cal_idx)),

    "evaluation_subset":
        "calibration_eval",

    "evaluation_proteins":
        int(len(eval_idx)),

    "objective":
        "binary_cross_entropy_with_logits",

    "calibration_nll_before":
        float(nll_before),

    "calibration_nll_after":
        float(nll_after),

    "calibration_nll_change":
        float(
            nll_after
            - nll_before
        ),

    "calibration_eval_labels_used_for_fit":
        False,

    "test_used":
        False,
}


with open(
    OUTPUT_DIR
    / "temperature_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 76)
print("TEMPERATURE FIT COMPLETE")
print("=" * 76)

print(
    "Temperature:",
    f"{temperature:.8f}"
)

print(
    "Calibration fit proteins:",
    len(cal_idx)
)

print(
    "Calibration-eval proteins used to fit T: NO"
)

print(
    "TEST SET USED: NO"
)

print("\nSaved to:")
print(OUTPUT_DIR)

print("\nNEXT:")
print(
    "Evaluate before-vs-after calibration "
    "ONLY on calibration-eval."
)

print("=" * 76)