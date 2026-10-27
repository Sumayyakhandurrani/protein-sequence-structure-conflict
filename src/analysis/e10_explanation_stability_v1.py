from pathlib import Path
import argparse
import hashlib
import sys

import numpy as np
import pandas as pd
import torch


# ============================================================
# FROZEN STABILITY V1 CONFIGURATION
# ============================================================

NUM_LABELS = 494
GLOBAL_SEED = 42
PRESERVATION_TOLERANCE = 0.01

EXPECTED_POPULATION_SIZE = 200
EXPECTED_TARGET_COUNT = 1770

EXPECTED_PROTOCOL_SHA = (
    "8f58e93e159da1d8698edf62720a78da4c6b9c6b"
    "64414dfb3b651890d012563f"
)

EXPECTED_IMPLEMENTATION_SPEC_SHA = (
    "563c830c8c4d783e5bdab9b39ddf617851f9bb2dd"
    "259a2ea87a6378406d8e0de"
)

EXPECTED_POPULATION_SHA = (
    "b0fbf944af605d61c3fb501709a9e7aa55b188b30"
    "cae2e354cd541dc37476877"
)

EXPECTED_SEQUENCE_SOURCE_SHA = (
    "705e232ae4da685581be79d2c03965df871a45c378"
    "007391a6d0e159c43776b8"
)

EXPECTED_STRUCTURE_ATTR_SOURCE_SHA = (
    "9db5ebe15d7c52f82c860f6af8e4955f348e0f905"
    "f4047514a7bda4d1486adfc"
)

EXPECTED_STRUCTURE_MODEL_SOURCE_SHA = (
    "51be5d9d361f2cb3fd4df873d9aef292b0865a813"
    "9a6b97cfb84941f216c467e"
)


# ============================================================
# PATHS
# ============================================================

BASE = Path("data")

STABILITY_ROOT = (
    BASE
    / "xai/functional_sites_v1/"
      "explanation_stability_v1"
)

PROTOCOL_PATH = (
    STABILITY_ROOT
    / "EXPLANATION_STABILITY_PROTOCOL_V1.txt"
)

IMPLEMENTATION_SPEC_PATH = (
    STABILITY_ROOT
    / "IMPLEMENTATION_SPEC_V1.txt"
)

POPULATION_PATH = (
    STABILITY_ROOT
    / "stability_population_ids.txt"
)

TEST_PKL = (
    BASE
    / "splits/mmseqs30_graph_v2/test.pkl"
)

TEST_IDS_PATH = (
    BASE
    / "splits/mmseqs30_graph_v2/"
      "labels/test_uniprot_ids.txt"
)

TEST_LABELS_PATH = (
    BASE
    / "splits/mmseqs30_graph_v2/"
      "labels/test_labels.npy"
)

STRUCTURE_IDS_PATH = (
    BASE
    / "structures/structure_protocol_v1/"
      "test_structure_eligible_ids.txt"
)

GRAPH_DIR = (
    BASE
    / "structures/structure_protocol_v1/"
      "graphs/test"
)

SEQ_ATTR_DIR = (
    BASE
    / "xai/functional_sites_v1/"
      "sequence_attributions_v1/per_protein"
)

STRUCT_ATTR_DIR = (
    BASE
    / "xai/functional_sites_v1/"
      "structure_attributions_v1/per_protein"
)

SEQUENCE_SOURCE = Path(
    "src/analysis/run_e10_sequence_attribution_full.py"
)

STRUCTURE_ATTR_SOURCE = Path(
    "src/analysis/e10_structure_attribution_population.py"
)

STRUCTURE_MODEL_SOURCE = Path(
    "src/models/structure_gnn.py"
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


def read_ids(path):

    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def require_sha(
    path,
    expected,
    label,
):

    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} SHA mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print(
        f"{label} SHA256: VERIFIED"
    )


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--validate-only",
    action="store_true",
)

parser.add_argument(
    "--limit",
    type=int,
    default=None,
)

args = parser.parse_args()

if (
    args.limit is not None
    and args.limit <= 0
):
    raise RuntimeError(
        "--limit must be positive."
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("E10 — EXPLANATION STABILITY V1")
print("=" * 78)

print(
    "Frozen proteins:",
    EXPECTED_POPULATION_SIZE,
)

print(
    "Expected protein-label targets:",
    EXPECTED_TARGET_COUNT,
)

print(
    "Prediction-preserving tolerance:",
    PRESERVATION_TOLERANCE,
)

print()


# ============================================================
# VERIFY FROZEN ARTIFACT HASHES
# ============================================================

require_sha(
    PROTOCOL_PATH,
    EXPECTED_PROTOCOL_SHA,
    "Scientific protocol",
)

require_sha(
    IMPLEMENTATION_SPEC_PATH,
    EXPECTED_IMPLEMENTATION_SPEC_SHA,
    "Implementation specification",
)

require_sha(
    POPULATION_PATH,
    EXPECTED_POPULATION_SHA,
    "Stability population",
)

require_sha(
    SEQUENCE_SOURCE,
    EXPECTED_SEQUENCE_SOURCE_SHA,
    "Sequence attribution source",
)

require_sha(
    STRUCTURE_ATTR_SOURCE,
    EXPECTED_STRUCTURE_ATTR_SOURCE_SHA,
    "Structure attribution source",
)

require_sha(
    STRUCTURE_MODEL_SOURCE,
    EXPECTED_STRUCTURE_MODEL_SOURCE_SHA,
    "Structure model source",
)


# ============================================================
# LOAD FROZEN POPULATIONS
# ============================================================

population_ids = read_ids(
    POPULATION_PATH
)

test_ids = read_ids(
    TEST_IDS_PATH
)

structure_ids = read_ids(
    STRUCTURE_IDS_PATH
)


if len(population_ids) != EXPECTED_POPULATION_SIZE:

    raise RuntimeError(
        "Unexpected stability population size: "
        f"{len(population_ids)}"
    )


if len(set(population_ids)) != len(
    population_ids
):

    raise RuntimeError(
        "Duplicate IDs in stability population."
    )


test_row_by_id = {
    uid: i
    for i, uid in enumerate(test_ids)
}

structure_row_by_id = {
    uid: i
    for i, uid in enumerate(structure_ids)
}


# ============================================================
# LOAD TEST DATA
# ============================================================

test = pd.read_pickle(
    TEST_PKL
)

Y = np.load(
    TEST_LABELS_PATH,
    mmap_mode="r",
)


if Y.shape != (
    len(test_ids),
    NUM_LABELS,
):

    raise RuntimeError(
        f"Unexpected TEST label shape: {Y.shape}"
    )


test_pkl_ids = (
    test["uniprot_id"]
    .astype(str)
    .tolist()
)


if test_pkl_ids != test_ids:

    raise RuntimeError(
        "TEST PKL/ID order mismatch."
    )


sequence_by_id = dict(
    zip(
        test["uniprot_id"].astype(str),
        test["sequence"].astype(str),
    )
)


# ============================================================
# FULL POPULATION QC
# ============================================================

total_targets = 0

for uid in population_ids:

    if uid not in test_row_by_id:

        raise RuntimeError(
            f"{uid}: missing from TEST."
        )

    if uid not in structure_row_by_id:

        raise RuntimeError(
            f"{uid}: not structure eligible."
        )

    graph_path = (
        GRAPH_DIR / f"{uid}.npz"
    )

    seq_attr_path = (
        SEQ_ATTR_DIR / f"{uid}.npz"
    )

    struct_attr_path = (
        STRUCT_ATTR_DIR / f"{uid}.npz"
    )


    for required in [
        graph_path,
        seq_attr_path,
        struct_attr_path,
    ]:

        if not required.exists():

            raise RuntimeError(
                f"{uid}: missing required file "
                f"{required}"
            )


    test_row = test_row_by_id[
        uid
    ]

    positive_labels = (
        np.flatnonzero(
            np.asarray(
                Y[test_row]
            ) == 1
        )
        .astype(np.int32)
    )


    if len(positive_labels) <= 0:

        raise RuntimeError(
            f"{uid}: zero positive labels."
        )


    total_targets += len(
        positive_labels
    )


    sequence = (
        sequence_by_id[
            uid
        ]
        .strip()
        .upper()
    )


    # --------------------------------------------------------
    # Structure graph consistency
    # --------------------------------------------------------

    with np.load(
        graph_path,
        allow_pickle=False,
    ) as g:

        graph_uid = str(
            g["uniprot_id"].item()
        )

        graph_sequence = str(
            g["sequence"].item()
        ).strip().upper()

        graph_length = int(
            g["x"].shape[0]
        )


    if graph_uid != uid:

        raise RuntimeError(
            f"{uid}: graph ID mismatch."
        )


    if graph_sequence != sequence:

        raise RuntimeError(
            f"{uid}: TEST/graph sequence mismatch."
        )


    if graph_length != len(
        sequence
    ):

        raise RuntimeError(
            f"{uid}: graph/sequence length mismatch."
        )


    # --------------------------------------------------------
    # Sequence attribution consistency
    # --------------------------------------------------------

    with np.load(
        seq_attr_path,
        allow_pickle=False,
    ) as z:

        if str(
            z["uniprot_id"].item()
        ) != uid:

            raise RuntimeError(
                f"{uid}: sequence attribution ID mismatch."
            )

        if int(
            z["length"].item()
        ) != len(sequence):

            raise RuntimeError(
                f"{uid}: sequence attribution "
                "length mismatch."
            )

        seq_labels = np.asarray(
            z["positive_label_indices"],
            dtype=np.int32,
        )


    if not np.array_equal(
        seq_labels,
        positive_labels,
    ):

        raise RuntimeError(
            f"{uid}: sequence attribution "
            "target-label mismatch."
        )


    # --------------------------------------------------------
    # Structure attribution consistency
    # --------------------------------------------------------

    with np.load(
        struct_attr_path,
        allow_pickle=False,
    ) as z:

        if str(
            z["uniprot_id"].item()
        ) != uid:

            raise RuntimeError(
                f"{uid}: structure attribution "
                "ID mismatch."
            )

        if int(
            z["length"].item()
        ) != len(sequence):

            raise RuntimeError(
                f"{uid}: structure attribution "
                "length mismatch."
            )

        struct_labels = np.asarray(
            z["positive_label_indices"],
            dtype=np.int32,
        )


    if not np.array_equal(
        struct_labels,
        positive_labels,
    ):

        raise RuntimeError(
            f"{uid}: structure attribution "
            "target-label mismatch."
        )


if total_targets != EXPECTED_TARGET_COUNT:

    raise RuntimeError(
        "Unexpected stability target count: "
        f"{total_targets}"
    )


print()
print(
    "Population QC proteins:",
    len(population_ids),
)

print(
    "Population QC targets:",
    total_targets,
)

print(
    "Sequence/structure/reference alignment: PASS"
)


# ============================================================
# VALIDATE ONLY
# ============================================================

if args.validate_only:

    print()
    print("=" * 78)
    print("VALIDATION ONLY: PASS")
    print("=" * 78)

    print(
        "No sequence model loaded."
    )

    print(
        "No structure model loaded."
    )

    print(
        "No perturbation generated."
    )

    print(
        "No stability outcome generated."
    )

    print(
        "No output files written."
    )

    raise SystemExit(0)


raise RuntimeError(
    "Computation stage is intentionally not implemented yet. "
    "Run --validate-only first."
)
