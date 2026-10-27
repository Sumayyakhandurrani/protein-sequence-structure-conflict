from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
TARGET_CAL_FRACTION = 0.50

BASE = Path("data")

SPLIT_DIR = (
    BASE
    / "splits"
    / "mmseqs30_graph_v2"
)

LABEL_DIR = SPLIT_DIR / "labels"

OUTPUT_DIR = (
    SPLIT_DIR
    / "calibration_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# LOAD FROZEN VALIDATION IDS
# ============================================================

print("=" * 76)
print("GRAPH-V2 VALIDATION CALIBRATION SUBDIVISION")
print("=" * 76)

val_ids_path = (
    LABEL_DIR
    / "val_uniprot_ids.txt"
)

with open(val_ids_path) as f:
    val_ids = [
        line.strip()
        for line in f
        if line.strip()
    ]

assert len(val_ids) == 4754
assert len(set(val_ids)) == 4754

print(
    "\nFrozen VAL proteins:",
    len(val_ids)
)


# ============================================================
# LOAD ORIGINAL GRAPH-V2 ASSIGNMENTS
# ============================================================

assignments = pd.read_csv(
    SPLIT_DIR
    / "split_assignments.csv"
)

required_columns = {
    "cluster_id",
    "uniprot_id",
    "split",
}

assert required_columns.issubset(
    assignments.columns
)

assert (
    assignments["uniprot_id"]
    .is_unique
)

val_assignments = (
    assignments[
        assignments["split"] == "val"
    ]
    .copy()
)

print(
    "VAL proteins in split_assignments:",
    len(val_assignments)
)

assert len(val_assignments) == 4754

assert set(
    val_assignments["uniprot_id"]
) == set(val_ids)

print(
    "VAL ID reconciliation: PASS"
)


# ============================================================
# COMPONENT TABLE
# ============================================================

component_table = (
    val_assignments
    .groupby(
        "cluster_id",
        as_index=False
    )
    .agg(
        n_proteins=(
            "uniprot_id",
            "size"
        )
    )
)

assert (
    component_table[
        "n_proteins"
    ].sum()
    == 4754
)

print(
    "VAL components:",
    len(component_table)
)

print(
    "Largest VAL component:",
    int(
        component_table[
            "n_proteins"
        ].max()
    )
)


# ============================================================
# REPRODUCIBLE COMPONENT-AWARE 50/50 SPLIT
#
# Random key provides deterministic tie-breaking.
# Components are processed largest-first.
# Each component is assigned to whichever subset is currently
# furthest below its target protein count.
# ============================================================

rng = np.random.default_rng(SEED)

component_table[
    "random_key"
] = rng.random(
    len(component_table)
)

component_table = (
    component_table
    .sort_values(
        [
            "n_proteins",
            "random_key",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)


target_cal = (
    len(val_ids)
    * TARGET_CAL_FRACTION
)

target_eval = (
    len(val_ids)
    - target_cal
)

cal_count = 0
eval_count = 0

subset_assignment = {}


for row in component_table.itertuples():

    cal_deficit = (
        target_cal
        - cal_count
    )

    eval_deficit = (
        target_eval
        - eval_count
    )

    if cal_deficit > eval_deficit:

        subset = "calibration"
        cal_count += row.n_proteins

    elif eval_deficit > cal_deficit:

        subset = "calibration_eval"
        eval_count += row.n_proteins

    else:
        # deterministic tie rule
        subset = "calibration"

        cal_count += row.n_proteins

    subset_assignment[
        row.cluster_id
    ] = subset


component_table[
    "calibration_subset"
] = component_table[
    "cluster_id"
].map(
    subset_assignment
)


# ============================================================
# MAP PROTEINS TO SUBSETS
# ============================================================

val_assignments[
    "calibration_subset"
] = val_assignments[
    "cluster_id"
].map(
    subset_assignment
)

assert (
    val_assignments[
        "calibration_subset"
    ]
    .notna()
    .all()
)


# ============================================================
# CRITICAL COMPONENT-LEAKAGE AUDIT
# ============================================================

component_subset_counts = (
    val_assignments
    .groupby("cluster_id")[
        "calibration_subset"
    ]
    .nunique()
)

cross_subset_components = int(
    (
        component_subset_counts > 1
    ).sum()
)

print(
    "\nComponents crossing calibration subsets:",
    cross_subset_components
)

assert cross_subset_components == 0

print(
    "Component leakage audit: PASS"
)


# ============================================================
# CREATE ID LISTS IN ORIGINAL VAL ROW ORDER
#
# This is important because val embeddings and val_labels.npy
# follow val_uniprot_ids.txt ordering.
# ============================================================

protein_to_subset = dict(
    zip(
        val_assignments[
            "uniprot_id"
        ],
        val_assignments[
            "calibration_subset"
        ],
    )
)

calibration_ids = [
    uid
    for uid in val_ids
    if protein_to_subset[uid]
    == "calibration"
]

calibration_eval_ids = [
    uid
    for uid in val_ids
    if protein_to_subset[uid]
    == "calibration_eval"
]


# ============================================================
# ROW INDICES INTO ORIGINAL VAL ARRAYS
# ============================================================

calibration_indices = np.array(
    [
        i
        for i, uid in enumerate(val_ids)
        if protein_to_subset[uid]
        == "calibration"
    ],
    dtype=np.int64,
)

calibration_eval_indices = np.array(
    [
        i
        for i, uid in enumerate(val_ids)
        if protein_to_subset[uid]
        == "calibration_eval"
    ],
    dtype=np.int64,
)


# ============================================================
# AUDITS
# ============================================================

assert (
    len(calibration_ids)
    + len(calibration_eval_ids)
    == 4754
)

assert set(calibration_ids).isdisjoint(
    calibration_eval_ids
)

assert (
    set(calibration_ids)
    | set(calibration_eval_ids)
) == set(val_ids)

assert len(
    calibration_indices
) == len(calibration_ids)

assert len(
    calibration_eval_indices
) == len(calibration_eval_ids)

assert np.intersect1d(
    calibration_indices,
    calibration_eval_indices
).size == 0

assert np.array_equal(
    np.sort(
        np.concatenate(
            [
                calibration_indices,
                calibration_eval_indices,
            ]
        )
    ),
    np.arange(
        4754,
        dtype=np.int64
    ),
)

print(
    "\nProtein coverage audit: PASS"
)

print(
    "Row-index audit: PASS"
)


# ============================================================
# COMPONENT COUNTS
# ============================================================

cal_components = int(
    (
        component_table[
            "calibration_subset"
        ]
        == "calibration"
    ).sum()
)

eval_components = int(
    (
        component_table[
            "calibration_subset"
        ]
        == "calibration_eval"
    ).sum()
)


# ============================================================
# SAVE FILES
# ============================================================

component_output = (
    component_table[
        [
            "cluster_id",
            "n_proteins",
            "calibration_subset",
        ]
    ]
    .sort_values(
        "cluster_id"
    )
)

component_output.to_csv(
    OUTPUT_DIR
    / "component_calibration_assignments.csv",
    index=False
)


protein_output = (
    val_assignments[
        [
            "cluster_id",
            "uniprot_id",
            "calibration_subset",
        ]
    ]
    .copy()
)

order_map = {
    uid: i
    for i, uid in enumerate(
        val_ids
    )
}

protein_output[
    "val_row_index"
] = protein_output[
    "uniprot_id"
].map(
    order_map
)

protein_output = (
    protein_output
    .sort_values(
        "val_row_index"
    )
)

protein_output.to_csv(
    OUTPUT_DIR
    / "protein_calibration_assignments.csv",
    index=False
)


with open(
    OUTPUT_DIR
    / "calibration_uniprot_ids.txt",
    "w"
) as f:

    for uid in calibration_ids:
        f.write(
            uid + "\n"
        )


with open(
    OUTPUT_DIR
    / "calibration_eval_uniprot_ids.txt",
    "w"
) as f:

    for uid in calibration_eval_ids:
        f.write(
            uid + "\n"
        )


np.save(
    OUTPUT_DIR
    / "calibration_val_indices.npy",
    calibration_indices
)

np.save(
    OUTPUT_DIR
    / "calibration_eval_val_indices.npy",
    calibration_eval_indices
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "name":
        "graph_v2_validation_calibration_v1",

    "parent_benchmark":
        "mmseqs30_graph_v2",

    "purpose":
        "post-hoc probability calibration development",

    "seed":
        SEED,

    "target_calibration_fraction":
        TARGET_CAL_FRACTION,

    "assignment_unit":
        "Graph-v2 connected component",

    "assignment_algorithm":
        (
            "largest-first greedy balancing "
            "with seeded random tie ordering"
        ),

    "original_validation_proteins":
        len(val_ids),

    "original_validation_components":
        int(
            len(component_table)
        ),

    "calibration_proteins":
        len(calibration_ids),

    "calibration_eval_proteins":
        len(calibration_eval_ids),

    "calibration_components":
        cal_components,

    "calibration_eval_components":
        eval_components,

    "cross_subset_components":
        cross_subset_components,

    "train_modified":
        False,

    "test_modified":
        False,

    "test_used":
        False,
}


manifest_path = (
    OUTPUT_DIR
    / "calibration_split_manifest.json"
)

with open(
    manifest_path,
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# CHECKSUMS
# ============================================================

checksum_files = [
    "component_calibration_assignments.csv",
    "protein_calibration_assignments.csv",
    "calibration_uniprot_ids.txt",
    "calibration_eval_uniprot_ids.txt",
    "calibration_val_indices.npy",
    "calibration_eval_val_indices.npy",
    "calibration_split_manifest.json",
]

checksums = {}

for filename in checksum_files:

    path = (
        OUTPUT_DIR
        / filename
    )

    checksums[
        filename
    ] = sha256_file(
        path
    )


with open(
    OUTPUT_DIR
    / "SHA256SUMS.json",
    "w"
) as f:

    json.dump(
        checksums,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 76)
print("CALIBRATION SUBDIVISION COMPLETE")
print("=" * 76)

print(
    "Original VAL proteins:",
    len(val_ids)
)

print(
    "Original VAL components:",
    len(component_table)
)

print()

print(
    "Calibration proteins:",
    len(calibration_ids)
)

print(
    "Calibration components:",
    cal_components
)

print()

print(
    "Calibration-eval proteins:",
    len(calibration_eval_ids)
)

print(
    "Calibration-eval components:",
    eval_components
)

print()

print(
    "Components crossing subsets:",
    cross_subset_components
)

print(
    "Protein overlap:",
    len(
        set(calibration_ids)
        &
        set(calibration_eval_ids)
    )
)

print(
    "Total proteins recovered:",
    len(calibration_ids)
    + len(calibration_eval_ids)
)

print()

print(
    "TRAIN MODIFIED: NO"
)

print(
    "TEST MODIFIED: NO"
)

print(
    "TEST SET USED: NO"
)

print("\nSaved to:")
print(OUTPUT_DIR)

print("=" * 76)