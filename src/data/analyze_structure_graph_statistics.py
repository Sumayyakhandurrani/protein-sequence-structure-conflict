from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PDB_DIR = Path(
    "data/structures/alphafold_v6/train/pdb"
)

ELIGIBLE_IDS_PATH = Path(
    "data/structures/structure_protocol_v1/"
    "train_structure_eligible_ids.txt"
)

PRIMARY_COHORT_PATH = Path(
    "data/processed/primary_mf_cohort.csv"
)

OUTPUT_DIR = Path(
    "data/structures/structure_protocol_v1/graph_statistics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42

SAMPLE_SIZE = 500

SPATIAL_CUTOFF = 10.0


# ============================================================
# PDB PARSER
# ============================================================

def parse_ca_coordinates(pdb_path):

    coords = []
    seen = set()

    with open(
        pdb_path,
        "r",
        errors="replace",
    ) as f:

        for line in f:

            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()

            if atom_name != "CA":
                continue

            altloc = line[16].strip()

            if altloc not in ("", "A"):
                continue

            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()

            key = (
                chain,
                resseq,
                icode,
            )

            if key in seen:
                continue

            seen.add(key)

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            coords.append(
                [x, y, z]
            )

    return np.asarray(
        coords,
        dtype=np.float32,
    )


# ============================================================
# SPATIAL GRAPH STATISTICS
# ============================================================

def graph_stats(coords, cutoff):

    n = len(coords)

    if n == 0:
        raise RuntimeError(
            "Protein has zero residues."
        )

    # For <= 2000 residues this dense distance
    # calculation is acceptable for this small
    # TRAIN-only diagnostic sample.

    diff = (
        coords[:, None, :]
        -
        coords[None, :, :]
    )

    dist2 = np.sum(
        diff * diff,
        axis=-1,
    )

    within = (
        dist2 <= cutoff ** 2
    )

    # Remove self-connections.
    np.fill_diagonal(
        within,
        False,
    )

    degree = within.sum(
        axis=1
    )

    # Number of unique undirected spatial edges.
    undirected_edges = int(
        np.triu(
            within,
            k=1,
        ).sum()
    )

    return {
        "n_residues":
            int(n),

        "spatial_edges":
            undirected_edges,

        "mean_spatial_degree":
            float(
                degree.mean()
            ),

        "median_spatial_degree":
            float(
                np.median(degree)
            ),

        "max_spatial_degree":
            int(
                degree.max()
            ),

        "p95_spatial_degree":
            float(
                np.percentile(
                    degree,
                    95,
                )
            ),

        "p99_spatial_degree":
            float(
                np.percentile(
                    degree,
                    99,
                )
            ),
    }


# ============================================================
# LOAD TRAIN ELIGIBLE IDS
# ============================================================

eligible_ids = [
    x.strip()
    for x in
    ELIGIBLE_IDS_PATH.read_text().splitlines()
    if x.strip()
]

if len(eligible_ids) != 37919:
    raise RuntimeError(
        "Expected 37,919 TRAIN "
        f"structure-eligible proteins, "
        f"found {len(eligible_ids)}"
    )


# ============================================================
# LOAD LENGTHS
# ============================================================

cohort = pd.read_csv(
    PRIMARY_COHORT_PATH,
    usecols=[
        "uniprot_id",
        "length",
    ],
)

cohort["uniprot_id"] = (
    cohort["uniprot_id"]
    .astype(str)
)

cohort = cohort[
    cohort["uniprot_id"].isin(
        set(eligible_ids)
    )
].copy()

if len(cohort) != 37919:
    raise RuntimeError(
        "TRAIN eligible IDs did not "
        "reconcile with primary cohort."
    )


# ============================================================
# STRATIFIED SAMPLE BY LENGTH
# ============================================================

bins = [
    50,
    250,
    500,
    1000,
    1500,
    2001,
]

labels = [
    "50-249",
    "250-499",
    "500-999",
    "1000-1499",
    "1500-2000",
]

cohort["length_bin"] = pd.cut(
    cohort["length"],
    bins=bins,
    labels=labels,
    right=False,
)

rng = random.Random(
    SEED
)

sampled_ids = []


# Target roughly equal representation
# across length groups.

per_bin = (
    SAMPLE_SIZE
    // len(labels)
)


print("\nTRAIN eligible proteins by length bin:")

for label in labels:

    group = cohort[
        cohort["length_bin"]
        == label
    ]

    ids = (
        group["uniprot_id"]
        .tolist()
    )

    print(
        f"{label:12s}: "
        f"{len(ids):,}"
    )

    take = min(
        per_bin,
        len(ids),
    )

    sampled = rng.sample(
        ids,
        take,
    )

    sampled_ids.extend(
        sampled
    )


# If a bin contained fewer proteins than requested,
# fill the remaining sample from other eligible TRAIN
# proteins not already selected.

if len(sampled_ids) < SAMPLE_SIZE:

    remaining = sorted(
        set(eligible_ids)
        -
        set(sampled_ids)
    )

    needed = (
        SAMPLE_SIZE
        -
        len(sampled_ids)
    )

    sampled_ids.extend(
        rng.sample(
            remaining,
            needed,
        )
    )


if len(sampled_ids) != SAMPLE_SIZE:
    raise RuntimeError(
        "Could not construct requested sample."
    )


if len(set(sampled_ids)) != SAMPLE_SIZE:
    raise RuntimeError(
        "Duplicate proteins in sample."
    )


# ============================================================
# ANALYZE
# ============================================================

length_lookup = dict(
    zip(
        cohort["uniprot_id"],
        cohort["length"],
    )
)

records = []


print(
    "\nAnalyzing "
    f"{SAMPLE_SIZE} TRAIN structures..."
)


for i, uid in enumerate(
    sampled_ids,
    start=1,
):

    pdb_path = (
        PDB_DIR
        / f"{uid}.pdb"
    )

    if not pdb_path.exists():
        raise FileNotFoundError(
            pdb_path
        )

    coords = parse_ca_coordinates(
        pdb_path
    )

    expected_length = int(
        length_lookup[uid]
    )

    if len(coords) != expected_length:
        raise RuntimeError(
            f"{uid}: expected "
            f"{expected_length} residues "
            f"but parsed {len(coords)}"
        )

    stats = graph_stats(
        coords,
        SPATIAL_CUTOFF,
    )

    stats["uniprot_id"] = uid
    stats["expected_length"] = (
        expected_length
    )

    records.append(
        stats
    )

    if (
        i % 50 == 0
        or i == SAMPLE_SIZE
    ):
        print(
            f"[{i}/{SAMPLE_SIZE}]"
        )


result = pd.DataFrame(
    records
)


# ============================================================
# SAVE PER-PROTEIN RESULTS
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "train_500_graph_statistics.csv"
)

result.to_csv(
    csv_path,
    index=False,
)


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

summary = {
    "protocol_stage":
        "pre-freeze graph statistics",

    "split":
        "train",

    "test_used":
        False,

    "seed":
        SEED,

    "sample_size":
        SAMPLE_SIZE,

    "spatial_cutoff_angstrom":
        SPATIAL_CUTOFF,

    "residue_count": {
        "median":
            float(
                result[
                    "n_residues"
                ].median()
            ),

        "max":
            int(
                result[
                    "n_residues"
                ].max()
            ),
    },

    "spatial_degree": {
        "mean_of_protein_means":
            float(
                result[
                    "mean_spatial_degree"
                ].mean()
            ),

        "median_of_protein_means":
            float(
                result[
                    "mean_spatial_degree"
                ].median()
            ),

        "p95_of_protein_p95":
            float(
                np.percentile(
                    result[
                        "p95_spatial_degree"
                    ],
                    95,
                )
            ),

        "p99_of_protein_p99":
            float(
                np.percentile(
                    result[
                        "p99_spatial_degree"
                    ],
                    99,
                )
            ),

        "largest_observed_degree":
            int(
                result[
                    "max_spatial_degree"
                ].max()
            ),
    },

    "spatial_edges": {
        "median_per_protein":
            float(
                result[
                    "spatial_edges"
                ].median()
            ),

        "p95_per_protein":
            float(
                np.percentile(
                    result[
                        "spatial_edges"
                    ],
                    95,
                )
            ),

        "max_per_protein":
            int(
                result[
                    "spatial_edges"
                ].max()
            ),
    },
}


json_path = (
    OUTPUT_DIR
    / "train_500_graph_statistics_summary.json"
)

with open(
    json_path,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "TRAIN GRAPH STATISTICS — "
    "10 Å SPATIAL CONTACTS"
)

print(
    "=" * 80
)

print(
    f"Sample size: "
    f"{SAMPLE_SIZE}"
)

print(
    f"Spatial cutoff: "
    f"{SPATIAL_CUTOFF:.1f} Å"
)

print(
    "\nSpatial degree:"
)

print(
    "Mean of protein means: "
    f"{summary['spatial_degree']['mean_of_protein_means']:.2f}"
)

print(
    "Median of protein means: "
    f"{summary['spatial_degree']['median_of_protein_means']:.2f}"
)

print(
    "95th percentile of "
    "protein p95 degree: "
    f"{summary['spatial_degree']['p95_of_protein_p95']:.2f}"
)

print(
    "99th percentile of "
    "protein p99 degree: "
    f"{summary['spatial_degree']['p99_of_protein_p99']:.2f}"
)

print(
    "Largest observed residue degree: "
    f"{summary['spatial_degree']['largest_observed_degree']}"
)

print(
    "\nSpatial edges per protein:"
)

print(
    "Median: "
    f"{summary['spatial_edges']['median_per_protein']:.0f}"
)

print(
    "95th percentile: "
    f"{summary['spatial_edges']['p95_per_protein']:.0f}"
)

print(
    "Maximum: "
    f"{summary['spatial_edges']['max_per_protein']:,}"
)

print(
    "\nTEST USED: NO"
)

print(
    "\nSaved:"
)

print(
    csv_path
)

print(
    json_path
)

print(
    "=" * 80
)