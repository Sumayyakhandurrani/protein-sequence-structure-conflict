from pathlib import Path
import json
import numpy as np


BASE = Path(
    "data/splits/mmseqs30_graph_v2/calibration_v1"
)

PRED_DIR = Path(
    "data/models/structure_gnn_v1_full/val_predictions"
)

OUT_DIR = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


def find_id_file(base, candidates):
    for name in candidates:
        p = base / name
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find any of:\n"
        + "\n".join(
            str(base / x)
            for x in candidates
        )
    )


# ------------------------------------------------------------
# Locate existing calibration-v1 ID files
# ------------------------------------------------------------

cal_path = find_id_file(
    BASE,
    [
        "calibration_ids.txt",
        "calibration_uniprot_ids.txt",
        "cal_ids.txt",
    ],
)

eval_path = find_id_file(
    BASE,
    [
        "cal_eval_ids.txt",
        "calibration_eval_uniprot_ids.txt",
        "cal_eval_uniprot_ids.txt",
    ],
)


eligible_path = (
    PRED_DIR
    / "val_structure_eligible_ids.txt"
)

labels_path = (
    PRED_DIR
    / "val_labels.npy"
)


cal_ids = read_ids(cal_path)
eval_ids = read_ids(eval_path)
eligible_ids = read_ids(eligible_path)

labels = np.load(labels_path)


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

if len(cal_ids) != 2377:
    raise RuntimeError(
        f"Expected 2,377 calibration IDs, "
        f"got {len(cal_ids)}"
    )

if len(eval_ids) != 2377:
    raise RuntimeError(
        f"Expected 2,377 cal-eval IDs, "
        f"got {len(eval_ids)}"
    )

if len(eligible_ids) != 4742:
    raise RuntimeError(
        f"Expected 4,742 structure-eligible IDs, "
        f"got {len(eligible_ids)}"
    )

if labels.shape != (4742, 494):
    raise RuntimeError(
        f"Unexpected labels shape: {labels.shape}"
    )

if set(cal_ids) & set(eval_ids):
    raise RuntimeError(
        "Calibration/cal-eval overlap detected."
    )


eligible_set = set(eligible_ids)

structure_cal_ids = [
    uid
    for uid in cal_ids
    if uid in eligible_set
]

structure_eval_ids = [
    uid
    for uid in eval_ids
    if uid in eligible_set
]


if (
    set(structure_cal_ids)
    & set(structure_eval_ids)
):
    raise RuntimeError(
        "Structure calibration/eval overlap."
    )


combined = (
    structure_cal_ids
    + structure_eval_ids
)

if set(combined) != eligible_set:
    raise RuntimeError(
        "Eligible VAL coverage mismatch."
    )


if len(combined) != 4742:
    raise RuntimeError(
        "Expected exactly 4,742 combined IDs."
    )


# ------------------------------------------------------------
# Extract labels in subset order
# ------------------------------------------------------------

eligible_index = {
    uid: i
    for i, uid in enumerate(eligible_ids)
}


cal_rows = np.asarray(
    [
        eligible_index[uid]
        for uid in structure_cal_ids
    ],
    dtype=np.int64,
)

eval_rows = np.asarray(
    [
        eligible_index[uid]
        for uid in structure_eval_ids
    ],
    dtype=np.int64,
)


cal_labels = np.asarray(
    labels[cal_rows],
    dtype=np.uint8,
)

eval_labels = np.asarray(
    labels[eval_rows],
    dtype=np.uint8,
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

cal_out = (
    OUT_DIR
    / "structure_calibration_ids.txt"
)

eval_out = (
    OUT_DIR
    / "structure_cal_eval_ids.txt"
)


cal_labels_out = (
    OUT_DIR
    / "structure_calibration_labels.npy"
)

eval_labels_out = (
    OUT_DIR
    / "structure_cal_eval_labels.npy"
)


with open(
    cal_out,
    "w",
    encoding="utf-8",
) as f:

    for uid in structure_cal_ids:
        f.write(uid + "\n")


with open(
    eval_out,
    "w",
    encoding="utf-8",
) as f:

    for uid in structure_eval_ids:
        f.write(uid + "\n")


np.save(
    cal_labels_out,
    cal_labels,
)

np.save(
    eval_labels_out,
    eval_labels,
)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

cal_support = cal_labels.sum(axis=0)
eval_support = eval_labels.sum(axis=0)


summary = {
    "original_frozen_val":
        4754,

    "structure_eligible_val":
        4742,

    "structure_ineligible_val":
        12,

    "original_calibration":
        len(cal_ids),

    "original_cal_eval":
        len(eval_ids),

    "structure_calibration":
        len(structure_cal_ids),

    "structure_cal_eval":
        len(structure_eval_ids),

    "removed_from_calibration":
        len(cal_ids)
        - len(structure_cal_ids),

    "removed_from_cal_eval":
        len(eval_ids)
        - len(structure_eval_ids),

    "calibration_positive_annotations":
        int(cal_labels.sum()),

    "cal_eval_positive_annotations":
        int(eval_labels.sum()),

    "calibration_supported_labels":
        int((cal_support > 0).sum()),

    "cal_eval_supported_labels":
        int((eval_support > 0).sum()),

    "calibration_zero_positive_labels":
        int((cal_support == 0).sum()),

    "cal_eval_zero_positive_labels":
        int((eval_support == 0).sum()),

    "component_aware_split_preserved":
        True,

    "new_split_created":
        False,

    "frozen_graph_v2_modified":
        False,

    "sequence_expert_modified":
        False,

    "test_used":
        False,
}


summary_path = (
    OUT_DIR
    / "structure_calibration_subset_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("=" * 78)
print("STRUCTURE CALIBRATION V1 — ELIGIBLE VAL SUBSET")
print("=" * 78)

print(
    f"Original frozen VAL:       "
    f"{4754:,}"
)

print(
    f"Structure eligible:        "
    f"{len(eligible_ids):,}"
)

print(
    f"Structure ineligible:      "
    f"{12:,}"
)

print()

print(
    f"Original calibration:      "
    f"{len(cal_ids):,}"
)

print(
    f"Structure calibration:     "
    f"{len(structure_cal_ids):,}"
)

print(
    f"Removed (ineligible):      "
    f"{len(cal_ids) - len(structure_cal_ids):,}"
)

print()

print(
    f"Original cal-eval:         "
    f"{len(eval_ids):,}"
)

print(
    f"Structure cal-eval:        "
    f"{len(structure_eval_ids):,}"
)

print(
    f"Removed (ineligible):      "
    f"{len(eval_ids) - len(structure_eval_ids):,}"
)

print()

print(
    f"Calibration positives:     "
    f"{int(cal_labels.sum()):,}"
)

print(
    f"Calibration labels >0:     "
    f"{int((cal_support > 0).sum())}/494"
)

print(
    f"Cal-eval positives:        "
    f"{int(eval_labels.sum()):,}"
)

print(
    f"Cal-eval labels >0:        "
    f"{int((eval_support > 0).sum())}/494"
)

print()

print(
    "Calibration/cal-eval overlap: 0"
)

print(
    "All eligible VAL covered: YES"
)

print(
    "Existing component split preserved: YES"
)

print(
    "New split created: NO"
)

print(
    "Frozen Graph-V2 modified: NO"
)

print(
    "Sequence expert modified: NO"
)

print(
    "TEST USED: NO"
)

print()

print(
    f"Saved under:\n  {OUT_DIR}"
)

print("=" * 78)