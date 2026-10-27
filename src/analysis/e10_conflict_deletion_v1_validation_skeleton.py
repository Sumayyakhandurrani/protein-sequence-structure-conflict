from pathlib import Path
import hashlib

import numpy as np


# ============================================================
# FROZEN CONSTANTS
# ============================================================

NUM_LABELS = 494
EXPECTED_POPULATION_SIZE = 2447
EXPECTED_TOTAL_TARGETS = 23087

EXPECTED_PROTOCOL_SHA = (
    "c62bb7952afeab57870718dff013068c"
    "11709b15262f094174b838b4a7329bf9"
)

EXPECTED_IMPLEMENTATION_SHA = (
    "ce6576a5662dcea187d0a450caf31948"
    "f4b988cdcb40594b81a40a6c0fe485a5"
)

EXPECTED_RANDOM_GENERATOR_SHA = (
    "471f64b5c73ee51d61bb570ed179ff2a"
    "7047d998006b5f822fa3d1869933cc63"
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path("data/xai/functional_sites_v1")

PROTOCOL_PATH = (
    ROOT / "CONFLICT_DELETION_PROTOCOL_V1.txt"
)

IMPLEMENTATION_PATH = (
    ROOT / "CONFLICT_DELETION_IMPLEMENTATION_V1.txt"
)

POPULATION_PATH = (
    ROOT / "xai_attribution_eligible_ids.txt"
)

SEQ_ATTR_DIR = (
    ROOT / "sequence_attributions_v1/per_protein"
)

STRUCT_ATTR_DIR = (
    ROOT / "structure_attributions_v1/per_protein"
)

TEST_IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "labels/test_uniprot_ids.txt"
)

TEST_LABELS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "labels/test_labels.npy"
)

PAIRED_IDS_PATH = Path(
    "data/final_evaluation/final_test_primary_v1/"
    "paired_test_uniprot_ids.txt"
)

JS_PATH = Path(
    "data/final_evaluation/final_test_primary_v1/"
    "bernoulli_js.npy"
)

RANDOM_GENERATOR_PATH = Path(
    "src/analysis/e10_deletion_random_sets_v1.py"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def verify_sha(path, expected, name):
    observed = sha256_file(path)

    if observed != expected:
        raise RuntimeError(
            f"{name} SHA mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )

    print(f"{name}: SHA VERIFIED")


def read_ids(path):
    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


# ============================================================
# VALIDATION
# ============================================================

print("=" * 78)
print("E10 CONFLICT-DELETION V1 — VALIDATION SKELETON")
print("=" * 78)

required = [
    PROTOCOL_PATH,
    IMPLEMENTATION_PATH,
    POPULATION_PATH,
    SEQ_ATTR_DIR,
    STRUCT_ATTR_DIR,
    TEST_IDS_PATH,
    TEST_LABELS_PATH,
    PAIRED_IDS_PATH,
    JS_PATH,
    RANDOM_GENERATOR_PATH,
]

for path in required:
    if not path.exists():
        raise RuntimeError(
            f"Missing required artifact: {path}"
        )

print("Required paths: PRESENT")

verify_sha(
    PROTOCOL_PATH,
    EXPECTED_PROTOCOL_SHA,
    "conflict-deletion protocol",
)

verify_sha(
    IMPLEMENTATION_PATH,
    EXPECTED_IMPLEMENTATION_SHA,
    "conflict-deletion implementation specification",
)

verify_sha(
    RANDOM_GENERATOR_PATH,
    EXPECTED_RANDOM_GENERATOR_SHA,
    "random-set generator",
)

population_ids = read_ids(
    POPULATION_PATH
)

test_ids = read_ids(
    TEST_IDS_PATH
)

paired_ids = read_ids(
    PAIRED_IDS_PATH
)

if len(population_ids) != EXPECTED_POPULATION_SIZE:
    raise RuntimeError(
        "Unexpected XAI population size: "
        f"{len(population_ids)}"
    )

if len(set(population_ids)) != len(population_ids):
    raise RuntimeError(
        "Duplicate XAI population IDs."
    )

if len(set(test_ids)) != len(test_ids):
    raise RuntimeError(
        "Duplicate TEST IDs."
    )

if len(set(paired_ids)) != len(paired_ids):
    raise RuntimeError(
        "Duplicate paired TEST IDs."
    )

test_row_by_id = {
    uid: i
    for i, uid in enumerate(test_ids)
}

paired_row_by_id = {
    uid: i
    for i, uid in enumerate(paired_ids)
}

for uid in population_ids:
    if uid not in test_row_by_id:
        raise RuntimeError(
            f"{uid}: missing from TEST IDs."
        )

    if uid not in paired_row_by_id:
        raise RuntimeError(
            f"{uid}: missing from paired TEST IDs."
        )

    seq_path = (
        SEQ_ATTR_DIR / f"{uid}.npz"
    )

    struct_path = (
        STRUCT_ATTR_DIR / f"{uid}.npz"
    )

    if not seq_path.exists():
        raise RuntimeError(
            f"{uid}: missing sequence attribution."
        )

    if not struct_path.exists():
        raise RuntimeError(
            f"{uid}: missing structure attribution."
        )


Y = np.load(
    TEST_LABELS_PATH,
    mmap_mode="r",
)

JS = np.load(
    JS_PATH,
    mmap_mode="r",
)

if Y.shape != (
    len(test_ids),
    NUM_LABELS,
):
    raise RuntimeError(
        f"Unexpected TEST label shape: {Y.shape}"
    )

if JS.shape != (
    len(paired_ids),
    NUM_LABELS,
):
    raise RuntimeError(
        f"Unexpected frozen JS shape: {JS.shape}"
    )

if not np.isfinite(JS).all():
    raise RuntimeError(
        "Frozen JS contains non-finite values."
    )


total_targets = 0
total_residues = 0

for uid in population_ids:

    test_row = test_row_by_id[uid]

    positive_labels = (
        np.flatnonzero(
            Y[test_row] == 1
        )
        .astype(np.int32)
    )

    if len(positive_labels) == 0:
        raise RuntimeError(
            f"{uid}: zero positive labels."
        )

    seq_path = (
        SEQ_ATTR_DIR / f"{uid}.npz"
    )

    struct_path = (
        STRUCT_ATTR_DIR / f"{uid}.npz"
    )

    with np.load(
        seq_path,
        allow_pickle=False,
    ) as s:

        seq_uid = str(
            s["uniprot_id"].item()
        )

        seq_length = int(
            s["length"].item()
        )

        seq_labels = np.asarray(
            s["positive_label_indices"],
            dtype=np.int32,
        )

        seq_attr_shape = (
            s["attribution_logit_drop"].shape
        )

    with np.load(
        struct_path,
        allow_pickle=False,
    ) as t:

        struct_uid = str(
            t["uniprot_id"].item()
        )

        struct_length = int(
            t["length"].item()
        )

        struct_labels = np.asarray(
            t["positive_label_indices"],
            dtype=np.int32,
        )

        struct_attr_shape = (
            t["attribution_logit_drop"].shape
        )

    if seq_uid != uid:
        raise RuntimeError(
            f"{uid}: sequence attribution ID mismatch."
        )

    if struct_uid != uid:
        raise RuntimeError(
            f"{uid}: structure attribution ID mismatch."
        )

    if seq_length != struct_length:
        raise RuntimeError(
            f"{uid}: sequence/structure length mismatch."
        )

    if not np.array_equal(
        seq_labels,
        positive_labels,
    ):
        raise RuntimeError(
            f"{uid}: sequence label mismatch."
        )

    if not np.array_equal(
        struct_labels,
        positive_labels,
    ):
        raise RuntimeError(
            f"{uid}: structure label mismatch."
        )

    expected_shape = (
        len(positive_labels),
        seq_length,
    )

    if seq_attr_shape != expected_shape:
        raise RuntimeError(
            f"{uid}: sequence attribution "
            f"shape mismatch {seq_attr_shape}."
        )

    if struct_attr_shape != expected_shape:
        raise RuntimeError(
            f"{uid}: structure attribution "
            f"shape mismatch {struct_attr_shape}."
        )

    total_targets += len(
        positive_labels
    )

    total_residues += seq_length


if total_targets != EXPECTED_TOTAL_TARGETS:
    raise RuntimeError(
        "Unexpected total target count: "
        f"{total_targets}"
    )


print()
print("Frozen XAI proteins:", len(population_ids))
print("Positive protein-label targets:", total_targets)
print("Population residues:", total_residues)
print("Paired TEST proteins:", len(paired_ids))
print("Frozen JS shape:", JS.shape)

print()
print("VALIDATION PASSED")
print("No model loaded.")
print("No perturbation performed.")
print("No conflict-deletion outcome computed.")
print("=" * 78)
