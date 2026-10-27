from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

LABEL_DIR = Path(
    "data/splits/mmseqs30_graph_v2/labels"
)

INTERNAL_DIR = Path(
    "data/splits/mmseqs30_graph_v2/structure_internal_v1"
)

OUTPUT_DIR = (
    INTERNAL_DIR
    / "labels"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FROZEN_TRAIN_IDS = (
    LABEL_DIR
    / "train_uniprot_ids.txt"
)

FROZEN_TRAIN_LABELS = (
    LABEL_DIR
    / "train_labels.npy"
)

PRIMARY_LABEL_MAP = (
    LABEL_DIR
    / "primary_label_map.csv"
)


INTERNAL_TRAIN_IDS = (
    INTERNAL_DIR
    / "structure_internal_train_ids.txt"
)

INTERNAL_DEV_IDS = (
    INTERNAL_DIR
    / "structure_internal_dev_ids.txt"
)


OUT_TRAIN_IDS = (
    OUTPUT_DIR
    / "structure_internal_train_uniprot_ids.txt"
)

OUT_DEV_IDS = (
    OUTPUT_DIR
    / "structure_internal_dev_uniprot_ids.txt"
)

OUT_TRAIN_LABELS = (
    OUTPUT_DIR
    / "structure_internal_train_labels.npy"
)

OUT_DEV_LABELS = (
    OUTPUT_DIR
    / "structure_internal_dev_labels.npy"
)

OUT_SUMMARY = (
    OUTPUT_DIR
    / "structure_internal_label_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):

    ids = [
        x.strip()
        for x in
        path.read_text().splitlines()
        if x.strip()
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"Duplicate IDs in {path}"
        )

    return ids


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# LOAD FROZEN TRAIN LABEL DATA
# ============================================================

frozen_train_ids = read_ids(
    FROZEN_TRAIN_IDS
)

train_labels = np.load(
    FROZEN_TRAIN_LABELS,
    mmap_mode="r",
)


print(
    "\n"
    + "=" * 80
)

print(
    "STRUCTURE INTERNAL V1 — "
    "LABEL ALIGNMENT"
)

print(
    "=" * 80
)

print(
    f"Frozen TRAIN IDs: "
    f"{len(frozen_train_ids):,}"
)

print(
    f"Frozen TRAIN labels: "
    f"{train_labels.shape}"
)


if len(frozen_train_ids) != 38031:
    raise RuntimeError(
        "Expected 38,031 frozen TRAIN IDs."
    )

if train_labels.shape != (
    38031,
    494,
):
    raise RuntimeError(
        "Unexpected frozen TRAIN "
        f"label shape: {train_labels.shape}"
    )

if train_labels.dtype != np.uint8:
    raise RuntimeError(
        f"Unexpected label dtype: "
        f"{train_labels.dtype}"
    )


# ============================================================
# VERIFY LABEL MAP
# ============================================================

label_map = pd.read_csv(
    PRIMARY_LABEL_MAP
)

print(
    f"Primary label map rows: "
    f"{len(label_map):,}"
)

print(
    f"Primary label map columns: "
    f"{label_map.columns.tolist()}"
)


if len(label_map) != 494:
    raise RuntimeError(
        "Primary label map does not "
        "contain exactly 494 labels."
    )


if "label_index" in label_map.columns:

    expected_indices = np.arange(
        494
    )

    actual_indices = (
        label_map[
            "label_index"
        ]
        .to_numpy()
    )

    if not np.array_equal(
        actual_indices,
        expected_indices,
    ):
        raise RuntimeError(
            "Frozen label_index ordering "
            "is not exactly 0..493."
        )


# ============================================================
# LOAD STRUCTURE INTERNAL IDS
# ============================================================

internal_train_ids = read_ids(
    INTERNAL_TRAIN_IDS
)

internal_dev_ids = read_ids(
    INTERNAL_DEV_IDS
)


print()

print(
    f"Structure internal TRAIN IDs: "
    f"{len(internal_train_ids):,}"
)

print(
    f"Structure internal DEV IDs:   "
    f"{len(internal_dev_ids):,}"
)


if len(internal_train_ids) != 34127:
    raise RuntimeError(
        "Expected 34,127 structure "
        "internal TRAIN IDs."
    )

if len(internal_dev_ids) != 3792:
    raise RuntimeError(
        "Expected 3,792 structure "
        "internal DEV IDs."
    )


train_set = set(
    internal_train_ids
)

dev_set = set(
    internal_dev_ids
)


if train_set & dev_set:
    raise RuntimeError(
        "Internal TRAIN/DEV ID overlap."
    )


# ============================================================
# BUILD FROZEN TRAIN ID -> ROW INDEX
# ============================================================

id_to_row = {
    uid: i
    for i, uid
    in enumerate(
        frozen_train_ids
    )
}


missing_internal_train = [
    uid
    for uid in internal_train_ids
    if uid not in id_to_row
]

missing_internal_dev = [
    uid
    for uid in internal_dev_ids
    if uid not in id_to_row
]


if missing_internal_train:
    raise RuntimeError(
        f"{len(missing_internal_train)} "
        "internal TRAIN IDs missing "
        "from frozen TRAIN label matrix."
    )


if missing_internal_dev:
    raise RuntimeError(
        f"{len(missing_internal_dev)} "
        "internal DEV IDs missing "
        "from frozen TRAIN label matrix."
    )


# ============================================================
# SUBSET LABELS BY UNIPROT ID
#
# IMPORTANT:
# We subset ROWS only.
# The 494 columns are NEVER reordered or modified.
# ============================================================

internal_train_rows = np.asarray(
    [
        id_to_row[uid]
        for uid in internal_train_ids
    ],
    dtype=np.int64,
)


internal_dev_rows = np.asarray(
    [
        id_to_row[uid]
        for uid in internal_dev_ids
    ],
    dtype=np.int64,
)


internal_train_labels = np.asarray(
    train_labels[
        internal_train_rows
    ],
    dtype=np.uint8,
)


internal_dev_labels = np.asarray(
    train_labels[
        internal_dev_rows
    ],
    dtype=np.uint8,
)


# ============================================================
# VALIDATE MATRICES
# ============================================================

if internal_train_labels.shape != (
    34127,
    494,
):
    raise RuntimeError(
        "Internal TRAIN label "
        f"shape incorrect: "
        f"{internal_train_labels.shape}"
    )


if internal_dev_labels.shape != (
    3792,
    494,
):
    raise RuntimeError(
        "Internal DEV label "
        f"shape incorrect: "
        f"{internal_dev_labels.shape}"
    )


if not np.all(
    (
        internal_train_labels == 0
    )
    |
    (
        internal_train_labels == 1
    )
):
    raise RuntimeError(
        "Internal TRAIN labels "
        "are not binary."
    )


if not np.all(
    (
        internal_dev_labels == 0
    )
    |
    (
        internal_dev_labels == 1
    )
):
    raise RuntimeError(
        "Internal DEV labels "
        "are not binary."
    )


# ============================================================
# EXACT ROW-BY-ROW RECONSTRUCTION CHECK
# ============================================================

for local_i, original_i in enumerate(
    internal_train_rows
):

    if not np.array_equal(
        internal_train_labels[
            local_i
        ],
        train_labels[
            original_i
        ],
    ):
        raise RuntimeError(
            "TRAIN row reconstruction "
            "check failed."
        )


for local_i, original_i in enumerate(
    internal_dev_rows
):

    if not np.array_equal(
        internal_dev_labels[
            local_i
        ],
        train_labels[
            original_i
        ],
    ):
        raise RuntimeError(
            "DEV row reconstruction "
            "check failed."
        )


# ============================================================
# LABEL STATISTICS
# ============================================================

train_positive_annotations = int(
    internal_train_labels.sum()
)

dev_positive_annotations = int(
    internal_dev_labels.sum()
)


train_support = (
    internal_train_labels.sum(
        axis=0
    )
)

dev_support = (
    internal_dev_labels.sum(
        axis=0
    )
)


train_labels_with_positive = int(
    np.sum(
        train_support > 0
    )
)

dev_labels_with_positive = int(
    np.sum(
        dev_support > 0
    )
)


train_zero_labels = int(
    np.sum(
        train_support == 0
    )
)

dev_zero_labels = int(
    np.sum(
        dev_support == 0
    )
)


# ============================================================
# SAVE
# ============================================================

np.save(
    OUT_TRAIN_LABELS,
    internal_train_labels,
)

np.save(
    OUT_DEV_LABELS,
    internal_dev_labels,
)


OUT_TRAIN_IDS.write_text(
    "\n".join(
        internal_train_ids
    )
    + "\n"
)

OUT_DEV_IDS.write_text(
    "\n".join(
        internal_dev_ids
    )
    + "\n"
)


# ============================================================
# RELOAD SAVED FILES
# ============================================================

saved_train = np.load(
    OUT_TRAIN_LABELS,
    mmap_mode="r",
)

saved_dev = np.load(
    OUT_DEV_LABELS,
    mmap_mode="r",
)


if not np.array_equal(
    saved_train,
    internal_train_labels,
):
    raise RuntimeError(
        "Saved TRAIN matrix "
        "reload check failed."
    )


if not np.array_equal(
    saved_dev,
    internal_dev_labels,
):
    raise RuntimeError(
        "Saved DEV matrix "
        "reload check failed."
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "name":
        "structure_internal_v1_labels",

    "source_label_matrix":
        str(
            FROZEN_TRAIN_LABELS
        ),

    "source_label_map":
        str(
            PRIMARY_LABEL_MAP
        ),

    "frozen_label_count":
        494,

    "label_columns_modified":
        False,

    "label_columns_reordered":
        False,

    "structure_internal_train_proteins":
        len(
            internal_train_ids
        ),

    "structure_internal_dev_proteins":
        len(
            internal_dev_ids
        ),

    "structure_internal_train_shape":
        list(
            internal_train_labels.shape
        ),

    "structure_internal_dev_shape":
        list(
            internal_dev_labels.shape
        ),

    "train_positive_annotations":
        train_positive_annotations,

    "dev_positive_annotations":
        dev_positive_annotations,

    "train_labels_with_positive":
        train_labels_with_positive,

    "dev_labels_with_positive":
        dev_labels_with_positive,

    "train_zero_positive_labels":
        train_zero_labels,

    "dev_zero_positive_labels":
        dev_zero_labels,

    "train_labels_sha256":
        sha256_file(
            OUT_TRAIN_LABELS
        ),

    "dev_labels_sha256":
        sha256_file(
            OUT_DEV_LABELS
        ),

    "train_ids_sha256":
        sha256_file(
            OUT_TRAIN_IDS
        ),

    "dev_ids_sha256":
        sha256_file(
            OUT_DEV_IDS
        ),

    "frozen_graph_v2_modified":
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
# PRINT RESULTS
# ============================================================

print()

print(
    f"Internal TRAIN matrix: "
    f"{internal_train_labels.shape} "
    f"{internal_train_labels.dtype}"
)

print(
    f"Internal DEV matrix:   "
    f"{internal_dev_labels.shape} "
    f"{internal_dev_labels.dtype}"
)

print()

print(
    f"Internal TRAIN positives: "
    f"{train_positive_annotations:,}"
)

print(
    f"Internal DEV positives:   "
    f"{dev_positive_annotations:,}"
)

print()

print(
    f"TRAIN labels with >=1 positive: "
    f"{train_labels_with_positive}/494"
)

print(
    f"DEV labels with >=1 positive:   "
    f"{dev_labels_with_positive}/494"
)

print(
    f"TRAIN zero-positive labels: "
    f"{train_zero_labels}"
)

print(
    f"DEV zero-positive labels:   "
    f"{dev_zero_labels}"
)

print()

print(
    "Exact row reconstruction: PASS"
)

print(
    "494-column ordering preserved: PASS"
)

print(
    "Frozen Graph-V2 modified: NO"
)

print(
    "Frozen VAL used: NO"
)

print(
    "TEST USED: NO"
)

print()

print("Saved:")
print(OUT_TRAIN_LABELS)
print(OUT_DEV_LABELS)
print(OUT_TRAIN_IDS)
print(OUT_DEV_IDS)
print(OUT_SUMMARY)

print(
    "=" * 80
)