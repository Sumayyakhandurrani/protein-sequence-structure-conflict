from pathlib import Path
import json
import random

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SEED = 42
DEV_FRACTION = 0.10

EXPECTED_ELIGIBLE_TRAIN = 37919

ASSIGNMENTS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/split_assignments.csv"
)

ELIGIBLE_IDS_PATH = Path(
    "data/structures/structure_protocol_v1/"
    "train_structure_eligible_ids.txt"
)

OUTPUT_DIR = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_internal_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INTERNAL_TRAIN_PATH = (
    OUTPUT_DIR
    / "structure_internal_train_ids.txt"
)

INTERNAL_DEV_PATH = (
    OUTPUT_DIR
    / "structure_internal_dev_ids.txt"
)

ASSIGNMENT_OUTPUT = (
    OUTPUT_DIR
    / "structure_internal_assignments.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "structure_internal_split_summary.json"
)


# ============================================================
# LOAD FROZEN GRAPH-V2 ASSIGNMENTS
# ============================================================

df = pd.read_csv(
    ASSIGNMENTS_PATH,
    dtype={
        "cluster_id": str,
        "uniprot_id": str,
        "split": str,
    },
)

required_columns = {
    "cluster_id",
    "uniprot_id",
    "split",
}

if not required_columns.issubset(df.columns):
    raise RuntimeError(
        "Required Graph-V2 columns missing."
    )


# ============================================================
# LOAD STRUCTURE-ELIGIBLE TRAIN IDS
# ============================================================

eligible_ids = [
    x.strip()
    for x in
    ELIGIBLE_IDS_PATH.read_text().splitlines()
    if x.strip()
]

if len(eligible_ids) != EXPECTED_ELIGIBLE_TRAIN:
    raise RuntimeError(
        f"Expected {EXPECTED_ELIGIBLE_TRAIN:,} "
        f"eligible TRAIN proteins, found "
        f"{len(eligible_ids):,}"
    )

if len(set(eligible_ids)) != len(eligible_ids):
    raise RuntimeError(
        "Duplicate eligible IDs detected."
    )

eligible_set = set(eligible_ids)


# ============================================================
# RESTRICT TO FROZEN TRAIN ONLY
# ============================================================

frozen_train = df[
    df["split"] == "train"
].copy()

frozen_train_ids = set(
    frozen_train["uniprot_id"]
)

missing = (
    eligible_set
    -
    frozen_train_ids
)

if missing:
    raise RuntimeError(
        f"{len(missing)} structure-eligible IDs "
        "are not in frozen Graph-V2 TRAIN."
    )


structure_df = frozen_train[
    frozen_train["uniprot_id"].isin(
        eligible_set
    )
].copy()


if len(structure_df) != EXPECTED_ELIGIBLE_TRAIN:
    raise RuntimeError(
        "Structure-eligible reconciliation failed."
    )


# ============================================================
# COMPONENT TABLE
# ============================================================

component_sizes = (
    structure_df
    .groupby("cluster_id")
    .size()
    .reset_index(
        name="eligible_proteins"
    )
)


num_components = len(
    component_sizes
)

largest_component = int(
    component_sizes[
        "eligible_proteins"
    ].max()
)


# ============================================================
# DETERMINISTIC COMPONENT-AWARE SPLIT
#
# Target:
#     90% internal TRAIN
#     10% internal DEV
#
# Components remain intact.
#
# Strategy:
# 1. deterministic random key using seed 42
# 2. largest components assigned first
# 3. choose assignment minimizing absolute
#    distance from target counts
# ============================================================

rng = random.Random(
    SEED
)

component_sizes[
    "random_key"
] = [
    rng.random()
    for _ in range(
        len(component_sizes)
    )
]


component_sizes = (
    component_sizes
    .sort_values(
        [
            "eligible_proteins",
            "random_key",
            "cluster_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


total = len(
    structure_df
)

target_dev = (
    total
    * DEV_FRACTION
)

target_train = (
    total
    - target_dev
)


train_count = 0
dev_count = 0

component_assignment = {}


for _, row in component_sizes.iterrows():

    cluster_id = row[
        "cluster_id"
    ]

    size = int(
        row[
            "eligible_proteins"
        ]
    )


    # ---------------------------------------------
    # Option 1: assign this component to TRAIN
    # ---------------------------------------------

    train_if_train = (
        train_count
        + size
    )

    dev_if_train = (
        dev_count
    )

    score_train = (
        abs(
            train_if_train
            -
            target_train
        )
        +
        abs(
            dev_if_train
            -
            target_dev
        )
    )


    # ---------------------------------------------
    # Option 2: assign this component to DEV
    # ---------------------------------------------

    train_if_dev = (
        train_count
    )

    dev_if_dev = (
        dev_count
        + size
    )

    score_dev = (
        abs(
            train_if_dev
            -
            target_train
        )
        +
        abs(
            dev_if_dev
            -
            target_dev
        )
    )


    if score_dev < score_train:

        assignment = "internal_dev"
        dev_count += size

    elif score_train < score_dev:

        assignment = "internal_train"
        train_count += size

    else:

        # Deterministic tie rule:
        # assign to side with larger
        # relative deficit.

        train_deficit = (
            target_train
            -
            train_count
        ) / target_train

        dev_deficit = (
            target_dev
            -
            dev_count
        ) / target_dev


        if dev_deficit > train_deficit:

            assignment = "internal_dev"
            dev_count += size

        else:

            assignment = "internal_train"
            train_count += size


    component_assignment[
        cluster_id
    ] = assignment


# ============================================================
# MAP BACK TO PROTEINS
# ============================================================

structure_df[
    "structure_internal_split"
] = (
    structure_df[
        "cluster_id"
    ].map(
        component_assignment
    )
)


if structure_df[
    "structure_internal_split"
].isna().any():

    raise RuntimeError(
        "Some proteins received no "
        "internal assignment."
    )


internal_train = (
    structure_df[
        structure_df[
            "structure_internal_split"
        ]
        ==
        "internal_train"
    ]
    .copy()
)

internal_dev = (
    structure_df[
        structure_df[
            "structure_internal_split"
        ]
        ==
        "internal_dev"
    ]
    .copy()
)


# ============================================================
# LEAKAGE AUDIT
# ============================================================

train_components = set(
    internal_train[
        "cluster_id"
    ]
)

dev_components = set(
    internal_dev[
        "cluster_id"
    ]
)

component_overlap = (
    train_components
    &
    dev_components
)

if component_overlap:
    raise RuntimeError(
        f"LEAKAGE: {len(component_overlap)} "
        "components cross internal TRAIN/DEV."
    )


train_ids = set(
    internal_train[
        "uniprot_id"
    ]
)

dev_ids = set(
    internal_dev[
        "uniprot_id"
    ]
)

protein_overlap = (
    train_ids
    &
    dev_ids
)

if protein_overlap:
    raise RuntimeError(
        "Protein overlap detected."
    )


if (
    len(train_ids)
    +
    len(dev_ids)
    != EXPECTED_ELIGIBLE_TRAIN
):
    raise RuntimeError(
        "Internal split does not cover "
        "all eligible TRAIN proteins."
    )


# ============================================================
# PRESERVE ORIGINAL ELIGIBLE-ID ORDER
#
# This is useful for reproducibility and label alignment.
# ============================================================

internal_train_ordered = [
    uid
    for uid in eligible_ids
    if uid in train_ids
]

internal_dev_ordered = [
    uid
    for uid in eligible_ids
    if uid in dev_ids
]


INTERNAL_TRAIN_PATH.write_text(
    "\n".join(
        internal_train_ordered
    )
    + "\n"
)

INTERNAL_DEV_PATH.write_text(
    "\n".join(
        internal_dev_ordered
    )
    + "\n"
)


# ============================================================
# SAVE ASSIGNMENTS
# ============================================================

structure_df[
    [
        "cluster_id",
        "uniprot_id",
        "split",
        "structure_internal_split",
    ]
].to_csv(
    ASSIGNMENT_OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

actual_train = len(
    internal_train_ordered
)

actual_dev = len(
    internal_dev_ordered
)

summary = {
    "name":
        "structure_internal_v1",

    "parent_benchmark":
        "mmseqs30_graph_v2",

    "seed":
        SEED,

    "eligible_frozen_train_proteins":
        EXPECTED_ELIGIBLE_TRAIN,

    "target_internal_dev_fraction":
        DEV_FRACTION,

    "internal_train_proteins":
        actual_train,

    "internal_dev_proteins":
        actual_dev,

    "actual_internal_train_fraction":
        actual_train
        / EXPECTED_ELIGIBLE_TRAIN,

    "actual_internal_dev_fraction":
        actual_dev
        / EXPECTED_ELIGIBLE_TRAIN,

    "total_components":
        num_components,

    "internal_train_components":
        len(train_components),

    "internal_dev_components":
        len(dev_components),

    "largest_eligible_component":
        largest_component,

    "crossing_components":
        len(component_overlap),

    "protein_overlap":
        len(protein_overlap),

    "all_eligible_train_proteins_covered":
        (
            actual_train
            + actual_dev
            ==
            EXPECTED_ELIGIBLE_TRAIN
        ),

    "frozen_graph_v2_modified":
        False,

    "frozen_val_used_for_assignment":
        False,

    "test_used":
        False,
}


with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# PRINT
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STRUCTURE INTERNAL V1 — "
    "COMPONENT-AWARE SPLIT"
)

print(
    "=" * 80
)

print(
    f"Eligible frozen TRAIN: "
    f"{EXPECTED_ELIGIBLE_TRAIN:,}"
)

print(
    f"Total components: "
    f"{num_components:,}"
)

print(
    f"Largest eligible component: "
    f"{largest_component:,}"
)

print()

print(
    f"Internal TRAIN proteins: "
    f"{actual_train:,} "
    f"({actual_train / EXPECTED_ELIGIBLE_TRAIN:.2%})"
)

print(
    f"Internal DEV proteins:   "
    f"{actual_dev:,} "
    f"({actual_dev / EXPECTED_ELIGIBLE_TRAIN:.2%})"
)

print()

print(
    f"Internal TRAIN components: "
    f"{len(train_components):,}"
)

print(
    f"Internal DEV components:   "
    f"{len(dev_components):,}"
)

print()

print(
    f"Crossing components: "
    f"{len(component_overlap)}"
)

print(
    f"Protein overlap: "
    f"{len(protein_overlap)}"
)

print(
    "All eligible TRAIN covered: "
    "YES"
)

print(
    "Frozen Graph-V2 modified: NO"
)

print(
    "Frozen VAL used for assignment: NO"
)

print(
    "TEST USED: NO"
)

print()

print("Saved:")
print(INTERNAL_TRAIN_PATH)
print(INTERNAL_DEV_PATH)
print(ASSIGNMENT_OUTPUT)
print(SUMMARY_OUTPUT)

print(
    "=" * 80
)