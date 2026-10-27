from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path("data/splits/mmseqs30_graph_v2")

CLUSTER_FILE = BASE / "cluster_cluster.tsv"
COHORT_FILE = Path("data/processed/primary_mf_cohort.pkl")

SEED = 42

TARGET_FRACTIONS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MAKE GRAPH-V2 CLUSTER SPLITS")
print("=" * 70)

print("\nLoading component membership...")

clusters = pd.read_csv(
    CLUSTER_FILE,
    sep="\t",
    header=None,
    names=["cluster_id", "uniprot_id"],
)

clusters["cluster_id"] = clusters["cluster_id"].astype(str)
clusters["uniprot_id"] = clusters["uniprot_id"].astype(str)

print("Loading primary cohort...")

cohort = pd.read_pickle(COHORT_FILE)

cohort["uniprot_id"] = cohort["uniprot_id"].astype(str)

if len(cohort) != 47539:
    raise ValueError(
        f"Expected 47,539 cohort proteins, found {len(cohort):,}"
    )

if clusters["uniprot_id"].nunique() != len(cohort):
    raise ValueError(
        "Cluster membership count does not match cohort."
    )

if set(clusters["uniprot_id"]) != set(cohort["uniprot_id"]):
    raise ValueError(
        "Cluster proteins do not match cohort proteins."
    )


# ============================================================
# COMPONENT SIZES
# ============================================================

cluster_sizes = (
    clusters
    .groupby("cluster_id")
    .size()
    .reset_index(name="n_proteins")
)

total_proteins = len(clusters)

print(f"\nProteins: {total_proteins:,}")
print(f"Components: {len(cluster_sizes):,}")

if total_proteins != 47539:
    raise ValueError("Unexpected total protein count.")

if len(cluster_sizes) != 19749:
    raise ValueError("Unexpected component count.")


targets = {
    split: total_proteins * fraction
    for split, fraction in TARGET_FRACTIONS.items()
}


print("\nTarget sizes:")

for split in ["train", "val", "test"]:
    print(
        f"{split:5s}: {targets[split]:,.1f}"
    )


# ============================================================
# ORIGINAL SPLITTING POLICY
#
# Randomize ties with seed=42, then process larger components
# first.
# ============================================================

rng = np.random.default_rng(SEED)

cluster_sizes["random_key"] = rng.random(
    len(cluster_sizes)
)

cluster_sizes = (
    cluster_sizes
    .sort_values(
        ["n_proteins", "random_key"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)


# ============================================================
# GREEDY RELATIVE-DEFICIT ASSIGNMENT
# ============================================================

current = {
    "train": 0,
    "val": 0,
    "test": 0,
}

assignments = {}


for row in cluster_sizes.itertuples(index=False):

    n = int(row.n_proteins)

    deficits = {}

    for split in TARGET_FRACTIONS:

        deficits[split] = (
            targets[split] - current[split]
        ) / targets[split]

    chosen = max(
        deficits,
        key=deficits.get,
    )

    assignments[row.cluster_id] = chosen

    current[chosen] += n


cluster_sizes["split"] = (
    cluster_sizes["cluster_id"]
    .map(assignments)
)


# ============================================================
# MAP COMPONENT ASSIGNMENTS TO PROTEINS
# ============================================================

protein_assignments = clusters.copy()

protein_assignments["split"] = (
    protein_assignments["cluster_id"]
    .map(assignments)
)

if protein_assignments["split"].isna().any():
    raise ValueError(
        "Some proteins were not assigned to a split."
    )

if not protein_assignments["uniprot_id"].is_unique:
    raise ValueError(
        "Duplicate proteins detected."
    )


# ============================================================
# COMPONENT-LEAKAGE CHECK
# ============================================================

cluster_split_counts = (
    protein_assignments
    .groupby("cluster_id")["split"]
    .nunique()
)

if not (cluster_split_counts == 1).all():
    raise ValueError(
        "A homology component appears in multiple splits."
    )


# ============================================================
# MERGE WITH COHORT
# ============================================================

cohort_with_split = cohort.merge(
    protein_assignments[
        ["uniprot_id", "cluster_id", "split"]
    ],
    on="uniprot_id",
    how="left",
    validate="one_to_one",
)

if cohort_with_split["split"].isna().any():
    raise ValueError(
        "Some cohort proteins are missing split assignments."
    )


train = cohort_with_split[
    cohort_with_split["split"] == "train"
].copy()

val = cohort_with_split[
    cohort_with_split["split"] == "val"
].copy()

test = cohort_with_split[
    cohort_with_split["split"] == "test"
].copy()


# ============================================================
# DISJOINTNESS
# ============================================================

train_ids = set(train["uniprot_id"])
val_ids = set(val["uniprot_id"])
test_ids = set(test["uniprot_id"])

if train_ids & val_ids:
    raise ValueError("Train/validation overlap.")

if train_ids & test_ids:
    raise ValueError("Train/test overlap.")

if val_ids & test_ids:
    raise ValueError("Validation/test overlap.")

if len(train_ids | val_ids | test_ids) != 47539:
    raise ValueError(
        "Split union does not contain all 47,539 proteins."
    )


# ============================================================
# SAVE
# ============================================================

protein_assignments.to_csv(
    BASE / "split_assignments.csv",
    index=False,
)

cluster_sizes[
    [
        "cluster_id",
        "n_proteins",
        "split",
    ]
].to_csv(
    BASE / "component_split_assignments.csv",
    index=False,
)

train.to_pickle(
    BASE / "train.pkl"
)

val.to_pickle(
    BASE / "val.pkl"
)

test.to_pickle(
    BASE / "test.pkl"
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("GRAPH-V2 SPLIT SUMMARY")
print("=" * 70)

print(f"Total proteins:      {len(cohort_with_split):,}")
print(f"Train proteins:      {len(train):,}")
print(f"Validation proteins: {len(val):,}")
print(f"Test proteins:       {len(test):,}")


cluster_counts = (
    cluster_sizes["split"]
    .value_counts()
)

print("\nComponents per split:")

for split in ["train", "val", "test"]:

    print(
        f"{split:5s}: "
        f"{cluster_counts.get(split, 0):,}"
    )


print("\nLargest component per split:")

for split in ["train", "val", "test"]:

    maximum = cluster_sizes.loc[
        cluster_sizes["split"] == split,
        "n_proteins",
    ].max()

    print(
        f"{split:5s}: {int(maximum):,}"
    )


print(
    "\nComponents crossing boundaries: 0"
)

print("\nSaved:")
print(BASE / "split_assignments.csv")
print(BASE / "component_split_assignments.csv")
print(BASE / "train.pkl")
print(BASE / "val.pkl")
print(BASE / "test.pkl")

print("\nDONE")