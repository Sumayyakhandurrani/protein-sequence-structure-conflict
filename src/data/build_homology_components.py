from pathlib import Path
import pandas as pd


COHORT_FILE = Path(
    "data/processed/primary_mf_cohort.pkl"
)

EDGE_FILE = Path(
    "data/splits/mmseqs30_graph/search/all_vs_all.tsv"
)

OUTPUT_DIR = Path(
    "data/splits/mmseqs30_graph"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD COHORT
# =========================================================

print("Loading primary cohort...")

cohort = pd.read_pickle(COHORT_FILE)

protein_ids = cohort["uniprot_id"].astype(str).tolist()

assert len(protein_ids) == 47539
assert len(protein_ids) == len(set(protein_ids))

known = set(protein_ids)

print(f"Proteins: {len(protein_ids):,}")


# =========================================================
# UNION-FIND
# =========================================================

parent = {protein: protein for protein in protein_ids}
rank = {protein: 0 for protein in protein_ids}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra = find(a)
    rb = find(b)

    if ra == rb:
        return

    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


# =========================================================
# READ ALL-vs-ALL RESULTS
# =========================================================

columns = [
    "query",
    "target",
    "fident",
    "alnlen",
    "qcov",
    "tcov",
    "evalue",
    "bits",
]

print("\nLoading all-vs-all MMseqs2 results...")

edges = pd.read_csv(
    EDGE_FILE,
    sep="\t",
    header=None,
    names=columns,
)

print(f"Rows in search output: {len(edges):,}")


# =========================================================
# SAFETY CHECKS
# =========================================================

unknown_queries = set(edges["query"]) - known
unknown_targets = set(edges["target"]) - known

assert not unknown_queries, (
    f"Unknown query proteins found: {len(unknown_queries)}"
)

assert not unknown_targets, (
    f"Unknown target proteins found: {len(unknown_targets)}"
)

# Explicitly enforce our benchmark criterion again.
qualifying = edges[
    (edges["query"] != edges["target"])
    & (edges["fident"] >= 0.30)
    & (edges["qcov"] >= 0.80)
    & (edges["tcov"] >= 0.80)
].copy()

print(
    f"Qualifying non-self edges: {len(qualifying):,}"
)


# =========================================================
# BUILD GRAPH COMPONENTS
# =========================================================

print("\nBuilding connected components...")

for row in qualifying.itertuples(index=False):
    union(row.query, row.target)


# =========================================================
# EXTRACT COMPONENTS
# =========================================================

components = {}

for protein in protein_ids:
    root = find(protein)
    components.setdefault(root, []).append(protein)

component_lists = sorted(
    components.values(),
    key=lambda members: (-len(members), min(members)),
)

print(
    f"Connected components: {len(component_lists):,}"
)


# =========================================================
# CREATE COMPONENT TABLE
# =========================================================

rows = []

for i, members in enumerate(component_lists):

    component_id = f"HC_{i:06d}"

    for protein in members:
        rows.append({
            "cluster_id": component_id,
            "uniprot_id": protein,
        })

component_df = pd.DataFrame(rows)

assert len(component_df) == len(protein_ids)
assert component_df["uniprot_id"].nunique() == len(protein_ids)


# =========================================================
# COMPONENT STATISTICS
# =========================================================

stats = (
    component_df
    .groupby("cluster_id")
    .size()
    .reset_index(name="n_proteins")
)

print("\n" + "=" * 70)
print("HOMOLOGY COMPONENT SUMMARY")
print("=" * 70)

print(stats["n_proteins"].describe())

print("\nLargest 20 components:")

print(
    stats.sort_values(
        "n_proteins",
        ascending=False
    ).head(20).to_string(index=False)
)

singletons = (stats["n_proteins"] == 1).sum()

print(
    f"\nSingleton components: {singletons:,}"
)

print(
    f"Proteins in largest component: "
    f"{stats['n_proteins'].max():,}"
)


# =========================================================
# SAVE
# =========================================================

cluster_file = OUTPUT_DIR / "cluster_cluster.tsv"

component_df[
    ["cluster_id", "uniprot_id"]
].to_csv(
    cluster_file,
    sep="\t",
    index=False,
    header=False,
)

stats.to_csv(
    OUTPUT_DIR / "component_stats.csv",
    index=False,
)

print("\nSaved:")
print(cluster_file)
print(OUTPUT_DIR / "component_stats.csv")

print(
    "\nGraph-based homology components successfully created."
)