from pathlib import Path
import hashlib
import json

import numpy as np


# ============================================================
# PATHS
# ============================================================

FROZEN_VAL_IDS = Path(
    "data/splits/mmseqs30_graph_v2/labels/val_uniprot_ids.txt"
)

FROZEN_VAL_LABELS = Path(
    "data/splits/mmseqs30_graph_v2/labels/val_labels.npy"
)

SEQ_LOGITS_PATH = Path(
    "data/models/sequence_mlp_v1/val_logits.npy"
)

SEQ_MANIFEST_PATH = Path(
    "data/models/sequence_expert_v1/"
    "sequence_expert_v1_manifest.json"
)

STRUCT_IDS_PATH = Path(
    "data/models/structure_gnn_v1_full/"
    "val_predictions/val_structure_eligible_ids.txt"
)

STRUCT_LOGITS_PATH = Path(
    "data/models/structure_gnn_v1_full/"
    "val_predictions/val_logits.npy"
)

STRUCT_LABELS_PATH = Path(
    "data/models/structure_gnn_v1_full/"
    "val_predictions/val_labels.npy"
)

OUT_DIR = Path(
    "data/conflict/paired_experts_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

    pos = x >= 0
    neg = ~pos

    out[pos] = (
        1.0 /
        (1.0 + np.exp(-x[pos]))
    )

    ex = np.exp(x[neg])

    out[neg] = (
        ex /
        (1.0 + ex)
    )

    return out


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# LOAD IDS
# ============================================================

print("=" * 84)
print("PAIRED SEQUENCE–STRUCTURE EXPERT ALIGNMENT V1")
print("=" * 84)

print("Conflict metric calculated: NO")
print("Threshold optimization: NO")
print("TEST USED: NO")
print()


val_ids = read_ids(
    FROZEN_VAL_IDS
)

struct_ids = read_ids(
    STRUCT_IDS_PATH
)


if len(val_ids) != 4754:
    raise RuntimeError(
        f"Expected 4,754 frozen VAL IDs, "
        f"got {len(val_ids)}"
    )


if len(struct_ids) != 4742:
    raise RuntimeError(
        f"Expected 4,742 structure IDs, "
        f"got {len(struct_ids)}"
    )


if len(set(val_ids)) != len(val_ids):
    raise RuntimeError(
        "Duplicate frozen VAL IDs."
    )


if len(set(struct_ids)) != len(struct_ids):
    raise RuntimeError(
        "Duplicate structure VAL IDs."
    )


val_set = set(val_ids)

if not set(struct_ids).issubset(val_set):
    raise RuntimeError(
        "Structure IDs are not a subset "
        "of frozen VAL IDs."
    )


# ============================================================
# LOAD ARRAYS
# ============================================================

seq_logits = np.load(
    SEQ_LOGITS_PATH
)

struct_logits = np.load(
    STRUCT_LOGITS_PATH
)

struct_labels = np.load(
    STRUCT_LABELS_PATH
)

frozen_labels = np.load(
    FROZEN_VAL_LABELS
)


expected_full = (4754, 494)
expected_struct = (4742, 494)


if seq_logits.shape != expected_full:
    raise RuntimeError(
        f"Sequence logits shape mismatch: "
        f"{seq_logits.shape}"
    )


if frozen_labels.shape != expected_full:
    raise RuntimeError(
        f"Frozen labels shape mismatch: "
        f"{frozen_labels.shape}"
    )


if struct_logits.shape != expected_struct:
    raise RuntimeError(
        f"Structure logits shape mismatch: "
        f"{struct_logits.shape}"
    )


if struct_labels.shape != expected_struct:
    raise RuntimeError(
        f"Structure labels shape mismatch: "
        f"{struct_labels.shape}"
    )


# ============================================================
# ALIGN SEQUENCE/FROZEN LABELS TO STRUCTURE IDS
# ============================================================

val_index = {
    uid: i
    for i, uid in enumerate(val_ids)
}


rows = np.asarray(
    [
        val_index[uid]
        for uid in struct_ids
    ],
    dtype=np.int64,
)


paired_seq_logits = np.asarray(
    seq_logits[rows],
    dtype=np.float32,
)

paired_frozen_labels = np.asarray(
    frozen_labels[rows],
    dtype=np.uint8,
)


# ============================================================
# STRICT LABEL EQUALITY
# ============================================================

labels_equal = np.array_equal(
    paired_frozen_labels,
    struct_labels,
)


if not labels_equal:

    differences = np.argwhere(
        paired_frozen_labels
        != struct_labels
    )

    first = differences[0]

    raise RuntimeError(
        "Label alignment failure. "
        f"First mismatch protein row="
        f"{first[0]}, label={first[1]}"
    )


# ============================================================
# LOAD FROZEN SEQUENCE TEMPERATURE
# ============================================================

with open(
    SEQ_MANIFEST_PATH,
    encoding="utf-8",
) as f:

    seq_manifest = json.load(f)


seq_temperature = float(
    seq_manifest[
        "calibration"
    ][
        "temperature"
    ]
)


expected_seq_temperature = (
    1.106977105140686
)


if abs(
    seq_temperature
    - expected_seq_temperature
) > 1e-10:

    raise RuntimeError(
        "Unexpected frozen sequence "
        f"temperature: {seq_temperature}"
    )


structure_temperature = 1.0


# ============================================================
# DEPLOYED PROBABILITIES
# ============================================================

seq_probs = sigmoid(
    paired_seq_logits.astype(
        np.float64
    )
    / seq_temperature
).astype(np.float32)


struct_probs = sigmoid(
    struct_logits.astype(
        np.float64
    )
    / structure_temperature
).astype(np.float32)


# ============================================================
# VALIDATE PROBABILITIES
# ============================================================

for name, arr in [
    ("sequence", seq_probs),
    ("structure", struct_probs),
]:

    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"{name} probabilities contain "
            "non-finite values."
        )

    if (
        arr.min() < 0.0
        or arr.max() > 1.0
    ):
        raise RuntimeError(
            f"{name} probabilities outside "
            "[0,1]."
        )


# ============================================================
# SAVE PAIRED ARTIFACTS
# ============================================================

ids_out = (
    OUT_DIR
    / "paired_val_uniprot_ids.txt"
)

labels_out = (
    OUT_DIR
    / "paired_val_labels.npy"
)

seq_logits_out = (
    OUT_DIR
    / "sequence_logits.npy"
)

seq_probs_out = (
    OUT_DIR
    / "sequence_probs.npy"
)

struct_logits_out = (
    OUT_DIR
    / "structure_logits.npy"
)

struct_probs_out = (
    OUT_DIR
    / "structure_probs.npy"
)


with open(
    ids_out,
    "w",
    encoding="utf-8",
) as f:

    for uid in struct_ids:
        f.write(uid + "\n")


np.save(
    labels_out,
    paired_frozen_labels,
)

np.save(
    seq_logits_out,
    paired_seq_logits,
)

np.save(
    seq_probs_out,
    seq_probs,
)

np.save(
    struct_logits_out,
    np.asarray(
        struct_logits,
        dtype=np.float32,
    ),
)

np.save(
    struct_probs_out,
    struct_probs,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "dataset":
        "paired_experts_v1",

    "population":
        (
            "structure-eligible frozen "
            "Graph-V2 validation proteins"
        ),

    "proteins":
        4742,

    "labels":
        494,

    "prediction_pairs":
        int(4742 * 494),

    "positive_annotations":
        int(
            paired_frozen_labels.sum()
        ),

    "sequence_expert":
        "sequence_expert_v1",

    "sequence_temperature":
        seq_temperature,

    "sequence_probability":
        "sigmoid(sequence_logit / T_seq)",

    "structure_expert":
        "structure_expert_v1",

    "structure_temperature":
        structure_temperature,

    "structure_probability":
        "sigmoid(structure_logit)",

    "sequence_logits_aligned_by":
        (
            "frozen Graph-V2 VAL UniProt "
            "ID ordering"
        ),

    "paired_order":
        (
            "structure-eligible VAL "
            "UniProt ID ordering"
        ),

    "labels_exact_match":
        bool(labels_equal),

    "sequence_probabilities_finite":
        bool(
            np.isfinite(
                seq_probs
            ).all()
        ),

    "structure_probabilities_finite":
        bool(
            np.isfinite(
                struct_probs
            ).all()
        ),

    "conflict_metric_calculated":
        False,

    "threshold_optimization":
        False,

    "test_used":
        False,
}


summary_path = (
    OUT_DIR
    / "paired_prediction_summary.json"
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


# ============================================================
# CHECKSUMS
# ============================================================

checksum_paths = [
    ids_out,
    labels_out,
    seq_logits_out,
    seq_probs_out,
    struct_logits_out,
    struct_probs_out,
    summary_path,
]


checksums = {
    p.name: sha256(p)
    for p in checksum_paths
}


checksums_path = (
    OUT_DIR
    / "SHA256SUMS.json"
)


with open(
    checksums_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        checksums,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print(
    f"Frozen VAL proteins:          "
    f"{len(val_ids):,}"
)

print(
    f"Structure-eligible paired:    "
    f"{len(struct_ids):,}"
)

print(
    f"Excluded structure-ineligible:"
    f" {len(val_ids) - len(struct_ids):,}"
)

print()

print(
    f"Labels:                       "
    f"{paired_frozen_labels.shape[1]}"
)

print(
    f"Prediction pairs:             "
    f"{paired_frozen_labels.size:,}"
)

print(
    f"Positive annotations:         "
    f"{int(paired_frozen_labels.sum()):,}"
)

print()

print(
    "Frozen labels == structure "
    f"labels: {labels_equal}"
)

print(
    f"Sequence temperature:         "
    f"{seq_temperature:.12f}"
)

print(
    f"Structure temperature:        "
    f"{structure_temperature:.1f}"
)

print()

print(
    f"Sequence probability range:   "
    f"[{seq_probs.min():.8f}, "
    f"{seq_probs.max():.8f}]"
)

print(
    f"Structure probability range:  "
    f"[{struct_probs.min():.8f}, "
    f"{struct_probs.max():.8f}]"
)

print()

print(
    "Sequence probabilities finite: YES"
)

print(
    "Structure probabilities finite: YES"
)

print(
    "Conflict metric calculated:     NO"
)

print(
    "Threshold optimization:         NO"
)

print(
    "TEST USED:                      NO"
)

print()

print(
    f"Saved under:\n  {OUT_DIR}"
)

print("=" * 84)