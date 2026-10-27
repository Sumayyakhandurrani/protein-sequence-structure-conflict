from pathlib import Path
import pandas as pd


CLUSTER_FILE = Path(
    "data/splits/mmseqs30/cluster_cluster.tsv"
)

COHORT_FILE = Path(
    "data/processed/primary_mf_cohort.pkl"
)

OUTPUT_CLUSTER_STATS = Path(
    "data/splits/mmseqs30/cluster_stats.csv"
)


print("Loading cluster assignments...")

clusters = pd.read_csv(
    CLUSTER_FILE,
    sep="\t",
    header=None,
    names=[
        "cluster_id",
        "uniprot_id"
    ]
)


print("Loading primary cohort...")

cohort = pd.read_pickle(
    COHORT_FILE
)


# ---------------------------------------------------------
# Basic checks
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLUSTER INPUT CHECK")
print("=" * 60)

print(
    f"Cluster rows: "
    f"{len(clusters):,}"
)

print(
    f"Unique cluster IDs: "
    f"{clusters['cluster_id'].nunique():,}"
)

print(
    f"Unique proteins in clusters: "
    f"{clusters['uniprot_id'].nunique():,}"
)

print(
    f"Proteins in cohort: "
    f"{len(cohort):,}"
)


assert clusters["uniprot_id"].nunique() == len(cohort), \
    "Cluster protein count does not match cohort."

assert set(
    clusters["uniprot_id"]
) == set(
    cohort["uniprot_id"]
), \
    "Cluster/cohort protein IDs do not match."


# ---------------------------------------------------------
# Merge with cohort metadata
# ---------------------------------------------------------

df = clusters.merge(
    cohort[
        [
            "uniprot_id",
            "length",
            "global_metric",
            "n_propagated_labels",
        ]
    ],
    on="uniprot_id",
    how="left"
)


# ---------------------------------------------------------
# Cluster-level statistics
# ---------------------------------------------------------

cluster_stats = (
    df.groupby("cluster_id")
    .agg(
        n_proteins=(
            "uniprot_id",
            "size"
        ),
        mean_length=(
            "length",
            "mean"
        ),
        median_length=(
            "length",
            "median"
        ),
        mean_global_metric=(
            "global_metric",
            "mean"
        ),
        mean_labels_per_protein=(
            "n_propagated_labels",
            "mean"
        ),
    )
    .reset_index()
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLUSTER SIZE DISTRIBUTION")
print("=" * 60)

print(
    cluster_stats[
        "n_proteins"
    ].describe()
)


print("\nClusters by size:")

bins = [
    (1, 1),
    (2, 2),
    (3, 5),
    (6, 10),
    (11, 20),
    (21, 50),
    (51, 100),
    (101, 1000000),
]

for low, high in bins:

    mask = (
        (cluster_stats["n_proteins"] >= low)
        &
        (cluster_stats["n_proteins"] <= high)
    )

    n_clusters = mask.sum()

    n_proteins = cluster_stats.loc[
        mask,
        "n_proteins"
    ].sum()

    if high >= 1000000:
        label = f">= {low}"
    elif low == high:
        label = f"{low}"
    else:
        label = f"{low}-{high}"

    print(
        f"Size {label:>6}: "
        f"{n_clusters:6,d} clusters | "
        f"{n_proteins:6,d} proteins"
    )


# ---------------------------------------------------------
# Largest clusters
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TOP 20 LARGEST CLUSTERS")
print("=" * 60)

largest = cluster_stats.nlargest(
    20,
    "n_proteins"
)

print(
    largest.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

cluster_stats.to_csv(
    OUTPUT_CLUSTER_STATS,
    index=False
)

print("\nSaved:")

print(
    OUTPUT_CLUSTER_STATS
)