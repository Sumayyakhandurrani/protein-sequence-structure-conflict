from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

COHORT_PATH = Path("data/processed/primary_mf_cohort.pkl")

SEARCH_PATH = Path(
    "data/splits/mmseqs30_graph_v2/search/all_vs_all.tsv"
)

OUT_DIR = Path("data/splits/mmseqs30_graph_v2")

CLUSTER_PATH = OUT_DIR / "cluster_cluster.tsv"
STATS_PATH = OUT_DIR / "component_stats.csv"
EDGES_PATH = OUT_DIR / "qualifying_primary_edges.csv"


# ============================================================
# FINAL GRAPH-V2 EDGE CRITERION
# ============================================================

MIN_IDENTITY = 0.30
MIN_QCOV = 0.80
MIN_TCOV = 0.80


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a

        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# NORMALIZE MMSEQS VALUES
# ============================================================

def normalize_fraction(series, name):
    """
    MMseqs normally returns fident/qcov/tcov as fractions.
    This also safely handles percentage-form values if present.
    """

    series = pd.to_numeric(series, errors="coerce")

    valid = series.dropna()

    if len(valid) == 0:
        raise ValueError(f"No valid values found in {name}")

    if valid.max() > 1.0:
        print(
            f"{name}: values appear to be percentages. "
            f"Converting to fractions."
        )
        series = series / 100.0

    return series


# ============================================================
# LOAD PRIMARY COHORT
# ============================================================

print("=" * 70)
print("BUILD HOMOLOGY COMPONENTS V2")
print("=" * 70)

print("\nLoading primary cohort...")

cohort = pd.read_pickle(COHORT_PATH)

if "uniprot_id" not in cohort.columns:
    raise ValueError("primary_mf_cohort.pkl has no 'uniprot_id' column")

protein_ids = cohort["uniprot_id"].astype(str).tolist()

n_proteins = len(protein_ids)
n_unique = len(set(protein_ids))

print(f"Proteins: {n_proteins:,}")
print(f"Unique IDs: {n_unique:,}")

if n_proteins != 47539:
    raise ValueError(
        f"Expected 47,539 proteins, found {n_proteins:,}"
    )

if n_unique != n_proteins:
    raise ValueError("Duplicate UniProt IDs detected in cohort")


protein_set = set(protein_ids)


# ============================================================
# LOAD MMSEQS SEARCH RESULTS
# ============================================================

print("\nLoading broad MMseqs candidate search...")

if not SEARCH_PATH.exists():
    raise FileNotFoundError(
        f"\nMissing:\n{SEARCH_PATH}\n\n"
        "Run the graph-v2 all-vs-all MMseqs search first."
    )


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

hits = pd.read_csv(
    SEARCH_PATH,
    sep="\t",
    header=None,
    names=columns,
)

print(f"Candidate rows: {len(hits):,}")


# ============================================================
# CLEAN / NORMALIZE
# ============================================================

hits["query"] = hits["query"].astype(str)
hits["target"] = hits["target"].astype(str)

hits["fident"] = normalize_fraction(
    hits["fident"],
    "fident",
)

hits["qcov"] = normalize_fraction(
    hits["qcov"],
    "qcov",
)

hits["tcov"] = normalize_fraction(
    hits["tcov"],
    "tcov",
)


# ============================================================
# VERIFY IDS
# ============================================================

search_ids = set(hits["query"]) | set(hits["target"])

unknown_ids = search_ids - protein_set

if unknown_ids:
    examples = sorted(list(unknown_ids))[:10]

    raise ValueError(
        "MMseqs results contain IDs not present in primary cohort.\n"
        f"Examples: {examples}"
    )

print("MMseqs IDs match primary cohort: PASS")


# ============================================================
# REMOVE SELF HITS
# ============================================================

nonself = hits[hits["query"] != hits["target"]].copy()

print(f"Non-self candidate rows: {len(nonself):,}")


# ============================================================
# APPLY FINAL PRIMARY HOMOLOGY CRITERION
#
# Candidate discovery:
#   identity >= 0.30
#   coverage >= 0.50
#
# Actual graph edge:
#   identity >= 0.30
#   qcov >= 0.80
#   tcov >= 0.80
# ============================================================

edges = nonself[
    (nonself["fident"] >= MIN_IDENTITY)
    & (nonself["qcov"] >= MIN_QCOV)
    & (nonself["tcov"] >= MIN_TCOV)
].copy()

print(
    "\nPrimary qualifying directed non-self edges: "
    f"{len(edges):,}"
)


# ============================================================
# SAVE QUALIFYING EDGES
# ============================================================

edges.to_csv(
    EDGES_PATH,
    index=False,
)

print(f"Saved edges -> {EDGES_PATH}")


# ============================================================
# BUILD CONNECTED COMPONENTS
# ============================================================

print("\nBuilding connected components...")

uf = UnionFind(protein_ids)

for row in edges.itertuples(index=False):
    uf.union(row.query, row.target)


# ============================================================
# COLLECT COMPONENT MEMBERS
# ============================================================

components = {}

for protein in protein_ids:

    root = uf.find(protein)

    if root not in components:
        components[root] = []

    components[root].append(protein)


# ============================================================
# DETERMINISTIC COMPONENT ORDER
#
# Largest components first.
# Ties broken by lexicographically smallest UniProt ID.
#
# This gives stable IDs:
# HC_000000
# HC_000001
# ...
# ============================================================

component_lists = list(components.values())

for members in component_lists:
    members.sort()

component_lists.sort(
    key=lambda members: (
        -len(members),
        members[0],
    )
)


# ============================================================
# BUILD COMPONENT TABLES
# ============================================================

cluster_rows = []
stats_rows = []

for i, members in enumerate(component_lists):

    cluster_id = f"HC_{i:06d}"

    stats_rows.append(
        {
            "cluster_id": cluster_id,
            "n_proteins": len(members),
        }
    )

    for protein in members:
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "uniprot_id": protein,
            }
        )


cluster_df = pd.DataFrame(cluster_rows)

component_stats = pd.DataFrame(stats_rows)


# ============================================================
# VALIDATION
# ============================================================

if len(cluster_df) != n_proteins:
    raise ValueError(
        "Component membership does not contain all proteins"
    )

if cluster_df["uniprot_id"].nunique() != n_proteins:
    raise ValueError(
        "Protein duplication detected in component membership"
    )

if component_stats["n_proteins"].sum() != n_proteins:
    raise ValueError(
        "Component sizes do not sum to total cohort size"
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

# Keep the historical filename used by graph-v2.
# No header, matching a simple two-column clustering TSV.
cluster_df[
    ["cluster_id", "uniprot_id"]
].to_csv(
    CLUSTER_PATH,
    sep="\t",
    index=False,
    header=False,
)


component_stats.to_csv(
    STATS_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

sizes = component_stats["n_proteins"]

print("\n" + "=" * 70)
print("GRAPH-V2 COMPONENT SUMMARY")
print("=" * 70)

print(f"Proteins:                 {n_proteins:,}")
print(f"Candidate rows:           {len(hits):,}")
print(
    "Primary qualifying "
    f"directed edges:           {len(edges):,}"
)

print(f"Connected components:     {len(component_stats):,}")

print(f"Mean component size:      {sizes.mean():.6f}")
print(f"Std component size:       {sizes.std():.6f}")
print(f"Median component size:    {sizes.median():.0f}")
print(f"75th percentile:          {sizes.quantile(0.75):.0f}")
print(f"Largest component:        {sizes.max():,}")

singletons = int((sizes == 1).sum())

print(f"Singleton components:     {singletons:,}")

print("\nLargest 20 components:")

print(
    component_stats.head(20).to_string(
        index=False
    )
)

print("\nSaved:")
print(CLUSTER_PATH)
print(STATS_PATH)
print(EDGES_PATH)

print("\nDONE")