from pathlib import Path
import json
import numpy as np
import pandas as pd


BASE = Path("data/splits/mmseqs30_graph_v2")
LABEL_DIR = BASE / "labels"
CAL_DIR = BASE / "calibration_v1"

print("=" * 76)
print("CALIBRATION SUBDIVISION — GO LABEL COVERAGE ANALYSIS")
print("=" * 76)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

Y_val = np.load(
    LABEL_DIR / "val_labels.npy"
)

cal_idx = np.load(
    CAL_DIR / "calibration_val_indices.npy"
)

eval_idx = np.load(
    CAL_DIR / "calibration_eval_val_indices.npy"
)

label_map = pd.read_csv(
    LABEL_DIR / "primary_label_map.csv"
)


# ------------------------------------------------------------
# SAFETY
# ------------------------------------------------------------

assert Y_val.shape == (4754, 494)
assert len(cal_idx) == 2377
assert len(eval_idx) == 2377

assert len(label_map) == 494

assert np.intersect1d(
    cal_idx,
    eval_idx
).size == 0

assert np.array_equal(
    np.sort(
        np.concatenate([cal_idx, eval_idx])
    ),
    np.arange(4754)
)

print("\nVAL labels:", Y_val.shape)
print("Calibration rows:", len(cal_idx))
print("Calibration-eval rows:", len(eval_idx))
print("Index integrity: PASS")
print("TEST SET USED: NO")


# ------------------------------------------------------------
# SUBSETS
# ------------------------------------------------------------

Y_cal = Y_val[cal_idx]
Y_eval = Y_val[eval_idx]

val_support = Y_val.sum(axis=0).astype(int)
cal_support = Y_cal.sum(axis=0).astype(int)
eval_support = Y_eval.sum(axis=0).astype(int)


# ------------------------------------------------------------
# BASIC COUNTS
# ------------------------------------------------------------

def count_positive_labels(x):
    return int((x > 0).sum())


print("\n" + "=" * 76)
print("LABEL COVERAGE")
print("=" * 76)

print(
    "Labels with >=1 positive in full VAL:",
    count_positive_labels(val_support)
)

print(
    "Labels with >=1 positive in calibration:",
    count_positive_labels(cal_support)
)

print(
    "Labels with >=1 positive in calibration-eval:",
    count_positive_labels(eval_support)
)

print(
    "Zero-positive labels in calibration:",
    int((cal_support == 0).sum())
)

print(
    "Zero-positive labels in calibration-eval:",
    int((eval_support == 0).sum())
)


# ------------------------------------------------------------
# POSITIVE COUNTS
# ------------------------------------------------------------

print("\n" + "=" * 76)
print("POSITIVE ANNOTATIONS")
print("=" * 76)

print(
    "Full VAL positives:",
    int(Y_val.sum())
)

print(
    "Calibration positives:",
    int(Y_cal.sum())
)

print(
    "Calibration-eval positives:",
    int(Y_eval.sum())
)

print()

print(
    "Full VAL labels/protein:",
    f"{Y_val.sum(axis=1).mean():.3f}"
)

print(
    "Calibration labels/protein:",
    f"{Y_cal.sum(axis=1).mean():.3f}"
)

print(
    "Calibration-eval labels/protein:",
    f"{Y_eval.sum(axis=1).mean():.3f}"
)


# ------------------------------------------------------------
# LABEL SUPPORT BANDS
# ------------------------------------------------------------

bands = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20-49", 20, 49),
    (">=50", 50, None),
]


def band_count(support, low, high):

    if high is None:
        return int(
            (support >= low).sum()
        )

    return int(
        (
            (support >= low)
            & (support <= high)
        ).sum()
    )


print("\n" + "=" * 76)
print("NUMBER OF GO LABELS BY POSITIVE SUPPORT")
print("=" * 76)

print(
    f"{'Support':<12}"
    f"{'Full VAL':>12}"
    f"{'Calibration':>14}"
    f"{'Cal-Eval':>12}"
)

print("-" * 50)

for name, low, high in bands:

    print(
        f"{name:<12}"
        f"{band_count(val_support, low, high):>12}"
        f"{band_count(cal_support, low, high):>14}"
        f"{band_count(eval_support, low, high):>12}"
    )


# ------------------------------------------------------------
# LABELS LOST FROM ONE HALF
# ------------------------------------------------------------

full_supported = val_support > 0

lost_cal = (
    full_supported
    & (cal_support == 0)
)

lost_eval = (
    full_supported
    & (eval_support == 0)
)

print("\n" + "=" * 76)
print("SUPPORTED-IN-FULL-VAL BUT ZERO IN ONE SUBSET")
print("=" * 76)

print(
    "Lost from calibration:",
    int(lost_cal.sum())
)

print(
    "Lost from calibration-eval:",
    int(lost_eval.sum())
)


# ------------------------------------------------------------
# SAVE LABEL-LEVEL TABLE
# ------------------------------------------------------------

result = label_map.copy()

result["full_val_support"] = val_support
result["calibration_support"] = cal_support
result["calibration_eval_support"] = eval_support

result["present_in_calibration"] = (
    cal_support > 0
)

result["present_in_calibration_eval"] = (
    eval_support > 0
)

result.to_csv(
    CAL_DIR / "calibration_label_coverage.csv",
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary = {
    "full_val_proteins": int(len(Y_val)),
    "calibration_proteins": int(len(Y_cal)),
    "calibration_eval_proteins": int(len(Y_eval)),

    "full_val_positive_annotations":
        int(Y_val.sum()),

    "calibration_positive_annotations":
        int(Y_cal.sum()),

    "calibration_eval_positive_annotations":
        int(Y_eval.sum()),

    "full_val_supported_labels":
        int((val_support > 0).sum()),

    "calibration_supported_labels":
        int((cal_support > 0).sum()),

    "calibration_eval_supported_labels":
        int((eval_support > 0).sum()),

    "full_val_supported_but_zero_calibration":
        int(lost_cal.sum()),

    "full_val_supported_but_zero_calibration_eval":
        int(lost_eval.sum()),

    "test_used": False,
}

with open(
    CAL_DIR / "calibration_label_coverage_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\nSaved:")
print(
    CAL_DIR / "calibration_label_coverage.csv"
)
print(
    CAL_DIR / "calibration_label_coverage_summary.json"
)

print("\nIMPORTANT:")
print("No calibration has been fitted yet.")
print("No threshold optimization performed.")
print("TEST SET USED: NO")
print("=" * 76)