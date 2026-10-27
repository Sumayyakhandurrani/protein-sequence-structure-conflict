from pathlib import Path
import pandas as pd
import obonet
import networkx as nx


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

GO_FILE = Path("data/raw/go-basic.obo")
INPUT_FILE = Path("data/processed/go_mf_experimental.csv")

OUTPUT_FILE = Path(
    "data/processed/go_mf_propagated.csv"
)

ROOT_MF = "GO:0003674"


# ---------------------------------------------------------
# LOAD GO ONTOLOGY
# ---------------------------------------------------------

print("Loading Gene Ontology...")

graph = obonet.read_obo(GO_FILE)

print(
    f"GO terms loaded: {graph.number_of_nodes():,}"
)

print(
    f"GO relationships loaded: {graph.number_of_edges():,}"
)


# ---------------------------------------------------------
# LOAD EXPERIMENTAL MF ANNOTATIONS
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(
    f"\nOriginal annotations: {len(df):,}"
)

print(
    f"Original proteins: "
    f"{df['uniprot_id'].nunique():,}"
)

print(
    f"Original GO terms: "
    f"{df['go_id'].nunique():,}"
)


# ---------------------------------------------------------
# FIND MF TERMS
# ---------------------------------------------------------

# The GO graph contains MF, BP and CC.
#
# We only want terms whose namespace is:
# molecular_function

mf_terms = {
    node
    for node, data in graph.nodes(data=True)
    if data.get("namespace") == "molecular_function"
}

print(
    f"\nMF ontology terms: {len(mf_terms):,}"
)


# ---------------------------------------------------------
# FUNCTION: GET ANCESTORS
# ---------------------------------------------------------

def get_mf_ancestors(go_id):

    if go_id not in graph:
        return set()

    # In obonet GO graphs, edges point from
    # child -> parent.
    #
    # Therefore descendants() follows
    # upward toward more general GO terms.

    ancestors = nx.descendants(
        graph,
        go_id
    )

    # Keep Molecular Function only
    ancestors = ancestors.intersection(
        mf_terms
    )

    return ancestors


# ---------------------------------------------------------
# PROPAGATE
# ---------------------------------------------------------

print("\nPropagating GO ancestors...")

records = []


for row in df.itertuples(index=False):

    protein_id = row.uniprot_id
    go_id = row.go_id

    # Keep original direct annotation
    records.append({
        "uniprot_id": protein_id,
        "go_id": go_id,
        "annotation_type": "direct"
    })

    ancestors = get_mf_ancestors(go_id)

    for ancestor in ancestors:

        records.append({
            "uniprot_id": protein_id,
            "go_id": ancestor,
            "annotation_type": "propagated"
        })


# ---------------------------------------------------------
# DATAFRAME
# ---------------------------------------------------------

propagated = pd.DataFrame(records)

# Same protein/GO combination should appear only once.
# If a direct annotation exists, prefer "direct".

propagated["priority"] = (
    propagated["annotation_type"]
    .map({
        "direct": 0,
        "propagated": 1
    })
)

propagated = (
    propagated
    .sort_values("priority")
    .drop_duplicates(
        subset=[
            "uniprot_id",
            "go_id"
        ],
        keep="first"
    )
    .drop(columns="priority")
)


# ---------------------------------------------------------
# REMOVE MF ROOT FROM MODEL LABELS
# ---------------------------------------------------------

# GO:0003674 = molecular_function itself.
#
# Nearly every MF-annotated protein would inherit it,
# so it is not useful as a prediction label.

root_count = (
    propagated["go_id"] == ROOT_MF
).sum()

print(
    f"\nMF root annotations found: {root_count:,}"
)

propagated = propagated[
    propagated["go_id"] != ROOT_MF
].copy()


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

print("\n==============================")
print("PROPAGATED MF DATASET")
print("==============================")

print(
    f"Annotations after propagation: "
    f"{len(propagated):,}"
)

print(
    f"Proteins: "
    f"{propagated['uniprot_id'].nunique():,}"
)

print(
    f"Unique GO terms: "
    f"{propagated['go_id'].nunique():,}"
)

print("\nAnnotation type:")

print(
    propagated["annotation_type"]
    .value_counts()
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

propagated.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\nDone.")