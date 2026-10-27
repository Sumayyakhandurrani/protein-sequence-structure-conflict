from pathlib import Path
import numpy as np
import pandas as pd


CLUSTER_FILE = Path(
    "data/splits/mmseqs30_graph/cluster_cluster.tsv"
)

COHORT_FILE = Path(
    "data/processed/primary_mf_cohort.pkl"
)

OUTPUT_DIR = Path(
    "data/splits/mmseqs30_graph"
)

SEED = 42

TARGET_FRACTIONS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}


# =========================================================
# LOAD DATA
# =========================================================

print("Loading MMseqs2 clusters...")

clusters = pd.read_csv(
    CLUSTER_FILE,
    sep="\t",
    header=None,
    names=["cluster_id", "uniprot_id"],
)

print("Loading primary cohort...")

cohort = pd.read_pickle(
    COHORT_FILE
)

assert clusters["uniprot_id"].nunique() == len(cohort)

assert set(clusters["uniprot_id"]) == set(cohort["uniprot_id"])


# =========================================================
# CLUSTER SIZES
# =========================================================

cluster_sizes = (
    clusters.groupby("cluster_id")
    .size()
    .reset_index(name="n_proteins")
)


total_proteins = len(clusters)

targets = {
    split: total_proteins * frac
    for split, frac in TARGET_FRACTIONS.items()
}


print("\n" + "=" * 60)
print("TARGET SPLIT SIZES")
print("=" * 60)

for split in TARGET_FRACTIONS:

    print(
        f"{split:5s}: "
        f"{targets[split]:,.1f}"
    )


# =========================================================
# RANDOMIZE TIES, THEN PROCESS LARGE CLUSTERS FIRST
# =========================================================

rng = np.random.default_rng(SEED)

cluster_sizes["random_key"] = rng.random(
    len(cluster_sizes)
)

cluster_sizes = cluster_sizes.sort_values(
    ["n_proteins", "random_key"],
    ascending=[False, True]
).reset_index(drop=True)


# =========================================================
# GREEDY SIZE-BALANCED CLUSTER ASSIGNMENT
# =========================================================

current = {
    "train": 0,
    "val": 0,
    "test": 0,
}

assignments = {}


for row in cluster_sizes.itertuples():

    n = row.n_proteins

    # Relative remaining capacity
    deficits = {}

    for split in TARGET_FRACTIONS:

        deficits[split] = (
            targets[split] - current[split]
        ) / targets[split]

    # Split with greatest relative deficit
    chosen = max(
        deficits,
        key=deficits.get
    )

    assignments[row.cluster_id] = chosen

    current[chosen] += n


cluster_sizes["split"] = cluster_sizes[
    "cluster_id"
].map(assignments)


# =========================================================
# MAP SPLITS BACK TO PROTEINS
# =========================================================

protein_assignments = clusters.copy()

protein_assignments["split"] = protein_assignments[
    "cluster_id"
].map(assignments)


assert protein_assignments["split"].notna().all()

assert protein_assignments["uniprot_id"].is_unique


# =========================================================
# LEAKAGE SANITY CHECK
# =========================================================

cluster_split_counts = (
    protein_assignments.groupby("cluster_id")["split"]
    .nunique()
)

assert (
    cluster_split_counts == 1
).all(), "A cluster appears in multiple splits!"


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("FINAL PROTEIN SPLIT")
print("=" * 60)

protein_counts = (
    protein_assignments["split"]
    .value_counts()
)

for split in ["train", "val", "test"]:

    n = protein_counts.get(split, 0)

    pct = 100 * n / total_proteins

    print(
        f"{split:5s}: "
        f"{n:6,d} proteins "
        f"({pct:6.2f}%)"
    )


print("\n" + "=" * 60)
print("CLUSTERS PER SPLIT")
print("=" * 60)

cluster_counts = (
    cluster_sizes["split"]
    .value_counts()
)

for split in ["train", "val", "test"]:

    n = cluster_counts.get(split, 0)

    print(
        f"{split:5s}: "
        f"{n:6,d} clusters"
    )


print("\nLargest cluster in each split:")

for split in ["train", "val", "test"]:

    maximum = cluster_sizes.loc[
        cluster_sizes["split"] == split,
        "n_proteins"
    ].max()

    print(
        f"{split:5s}: "
        f"{maximum:,}"
    )


# =========================================================
# SAVE MASTER ASSIGNMENTS
# =========================================================

protein_assignments.to_csv(
    OUTPUT_DIR / "split_assignments.csv",
    index=False
)

cluster_sizes[
    [
        "cluster_id",
        "n_proteins",
        "split"
    ]
].to_csv(
    OUTPUT_DIR / "cluster_split_assignments.csv",
    index=False
)


# =========================================================
# MERGE SPLITS WITH COHORT
# =========================================================

cohort_with_split = cohort.merge(
    protein_assignments[
        [
            "uniprot_id",
            "cluster_id",
            "split"
        ]
    ],
    on="uniprot_id",
    how="inner"
)

assert len(cohort_with_split) == len(cohort)


# =========================================================
# SAVE TRAIN / VAL / TEST DATASETS
# =========================================================

for split in ["train", "val", "test"]:

    subset = cohort_with_split[
        cohort_with_split["split"] == split
    ].copy()

    subset.to_pickle(
        OUTPUT_DIR / f"{split}.pkl"
    )

    subset.to_csv(
        OUTPUT_DIR / f"{split}.csv",
        index=False
    )

    subset[
        ["uniprot_id"]
    ].to_csv(
        OUTPUT_DIR / f"{split}_accessions.txt",
        index=False,
        header=False
    )


# =========================================================
# FINAL VERIFICATION
# =========================================================

train_ids = set(
    cohort_with_split.loc[
        cohort_with_split["split"] == "train",
        "uniprot_id"
    ]
)

val_ids = set(
    cohort_with_split.loc[
        cohort_with_split["split"] == "val",
        "uniprot_id"
    ]
)

test_ids = set(
    cohort_with_split.loc[
        cohort_with_split["split"] == "test",
        "uniprot_id"
    ]
)


assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

assert (
    len(train_ids)
    + len(val_ids)
    + len(test_ids)
    == len(cohort)
)


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    OUTPUT_DIR / "split_assignments.csv"
)

print(
    OUTPUT_DIR / "cluster_split_assignments.csv"
)

print(
    OUTPUT_DIR / "train.pkl"
)

print(
    OUTPUT_DIR / "val.pkl"
)

print(
    OUTPUT_DIR / "test.pkl"
)

print("\nNo MMseqs2 cluster crosses between splits.")
print("Cluster-level split successfully created.")