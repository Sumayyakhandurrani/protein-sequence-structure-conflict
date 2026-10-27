from pathlib import Path
import json

import numpy as np


# ============================================================
# PATHS
# ============================================================

FROZEN_LABEL_DIR = Path(
    "data/splits/mmseqs30_graph_v2/labels"
)

ELIGIBILITY_DIR = Path(
    "data/structures/structure_protocol_v1"
)

OUTPUT_DIR = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_full_train_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FROZEN_TRAIN_IDS = (
    FROZEN_LABEL_DIR
    / "train_uniprot_ids.txt"
)

FROZEN_TRAIN_LABELS = (
    FROZEN_LABEL_DIR
    / "train_labels.npy"
)

ELIGIBLE_IDS = (
    ELIGIBILITY_DIR
    / "train_structure_eligible_ids.txt"
)


OUT_IDS = (
    OUTPUT_DIR
    / "structure_full_train_uniprot_ids.txt"
)

OUT_LABELS = (
    OUTPUT_DIR
    / "structure_full_train_labels.npy"
)

OUT_SUMMARY = (
    OUTPUT_DIR
    / "structure_full_train_label_summary.json"
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
# LOAD FROZEN TRAIN
# ============================================================

print("=" * 80)
print("STRUCTURE FULL TRAIN V1 — LABEL CONSTRUCTION")
print("=" * 80)

frozen_ids = read_ids(
    FROZEN_TRAIN_IDS
)

frozen_labels = np.load(
    FROZEN_TRAIN_LABELS,
    mmap_mode="r",
)

eligible_ids = read_ids(
    ELIGIBLE_IDS
)


print(
    f"Frozen TRAIN proteins: "
    f"{len(frozen_ids):,}"
)

print(
    f"Frozen TRAIN labels: "
    f"{frozen_labels.shape}"
)

print(
    f"Structure-eligible TRAIN: "
    f"{len(eligible_ids):,}"
)


# ============================================================
# VALIDATE
# ============================================================

if len(frozen_ids) != 38031:
    raise RuntimeError(
        "Unexpected frozen TRAIN size."
    )

if frozen_labels.shape != (
    38031,
    494,
):
    raise RuntimeError(
        "Unexpected frozen TRAIN label shape."
    )

if len(eligible_ids) != 37919:
    raise RuntimeError(
        "Unexpected eligible TRAIN size."
    )

if len(set(frozen_ids)) != len(
    frozen_ids
):
    raise RuntimeError(
        "Duplicate frozen TRAIN IDs."
    )

if len(set(eligible_ids)) != len(
    eligible_ids
):
    raise RuntimeError(
        "Duplicate eligible IDs."
    )


frozen_index = {
    uid: i
    for i, uid
    in enumerate(frozen_ids)
}


missing = [
    uid
    for uid in eligible_ids
    if uid not in frozen_index
]

if missing:
    raise RuntimeError(
        f"{len(missing)} eligible IDs "
        "missing from frozen TRAIN."
    )


# ============================================================
# SUBSET LABEL MATRIX
# ============================================================

rows = np.array(
    [
        frozen_index[uid]
        for uid in eligible_ids
    ],
    dtype=np.int64,
)

eligible_labels = np.asarray(
    frozen_labels[rows],
    dtype=np.uint8,
)


if eligible_labels.shape != (
    37919,
    494,
):
    raise RuntimeError(
        "Eligible label shape mismatch."
    )


# ============================================================
# EXACT RECONSTRUCTION CHECK
# ============================================================

reconstruction_pass = True

for out_i, frozen_i in enumerate(
    rows
):
    if not np.array_equal(
        eligible_labels[out_i],
        frozen_labels[frozen_i],
    ):
        reconstruction_pass = False
        break


if not reconstruction_pass:
    raise RuntimeError(
        "Label reconstruction failed."
    )


# ============================================================
# STATISTICS
# ============================================================

positive_annotations = int(
    eligible_labels.sum()
)

support = eligible_labels.sum(
    axis=0
)

labels_with_positive = int(
    (support > 0).sum()
)

zero_positive_labels = int(
    (support == 0).sum()
)


# ============================================================
# SAVE
# ============================================================

with open(
    OUT_IDS,
    "w",
    encoding="utf-8",
) as f:
    for uid in eligible_ids:
        f.write(uid + "\n")


np.save(
    OUT_LABELS,
    eligible_labels,
)


summary = {
    "name":
        "structure_full_train_v1",

    "source":
        "frozen Graph-V2 TRAIN labels",

    "proteins":
        len(eligible_ids),

    "label_matrix_shape":
        list(
            eligible_labels.shape
        ),

    "dtype":
        str(
            eligible_labels.dtype
        ),

    "positive_annotations":
        positive_annotations,

    "labels_with_at_least_one_positive":
        labels_with_positive,

    "zero_positive_labels":
        zero_positive_labels,

    "exact_row_reconstruction":
        reconstruction_pass,

    "label_columns_modified":
        False,

    "frozen_graph_v2_membership_modified":
        False,

    "frozen_val_used":
        False,

    "test_used":
        False,
}


with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print(
    f"Output matrix: "
    f"{eligible_labels.shape}"
)

print(
    f"Dtype: "
    f"{eligible_labels.dtype}"
)

print(
    f"Positive annotations: "
    f"{positive_annotations:,}"
)

print(
    f"Labels with >=1 positive: "
    f"{labels_with_positive}/494"
)

print(
    f"Zero-positive labels: "
    f"{zero_positive_labels}"
)

print()

print(
    "Exact row reconstruction:",
    "PASS"
)

print(
    "494-column ordering changed: NO"
)

print(
    "Frozen Graph-V2 membership changed: NO"
)

print(
    "Frozen VAL used: NO"
)

print(
    "TEST USED: NO"
)

print()

print(
    f"Saved IDs:\n  {OUT_IDS}"
)

print(
    f"Saved labels:\n  {OUT_LABELS}"
)

print(
    f"Saved summary:\n  {OUT_SUMMARY}"
)

print("=" * 80)