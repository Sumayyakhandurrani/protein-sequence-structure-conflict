from pathlib import Path

import pandas as pd
import obonet
import networkx as nx


GO_FILE = Path(
    "data/raw/go-basic.obo"
)

INPUT_FILE = Path(
    "data/processed/full_mf_evidence_raw.csv"
)

OUTPUT_FILE = Path(
    "data/processed/full_mf_propagated.csv"
)

ROOT_MF = "GO:0003674"


print("Loading GO ontology...")

graph = obonet.read_obo(GO_FILE)

mf_terms = {
    node
    for node, data in graph.nodes(data=True)
    if data.get("namespace")
    == "molecular_function"
}


print(
    f"MF ontology terms: "
    f"{len(mf_terms):,}"
)


df = pd.read_csv(INPUT_FILE)

print(
    f"Direct annotations: "
    f"{len(df):,}"
)

print(
    f"Proteins: "
    f"{df['uniprot_id'].nunique():,}"
)


# Cache ancestors because the same GO term
# occurs for many proteins.
ancestor_cache = {}


def get_ancestors(go_id):

    if go_id in ancestor_cache:
        return ancestor_cache[go_id]

    if go_id not in graph:
        ancestor_cache[go_id] = set()
        return set()

    ancestors = nx.descendants(
        graph,
        go_id
    )

    ancestors = (
        ancestors
        .intersection(mf_terms)
    )

    ancestor_cache[go_id] = ancestors

    return ancestors


records = []


print("\nPropagating hierarchy...")


for row in df.itertuples(index=False):

    protein = row.uniprot_id
    go_id = row.go_id

    records.append({
        "uniprot_id": protein,
        "go_id": go_id,
        "annotation_type": "direct"
    })

    for ancestor in get_ancestors(go_id):

        records.append({
            "uniprot_id": protein,
            "go_id": ancestor,
            "annotation_type": "propagated"
        })


propagated = pd.DataFrame(records)


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
    .drop(
        columns="priority"
    )
)


root_count = (
    propagated["go_id"]
    == ROOT_MF
).sum()


print(
    f"\nMF root annotations: "
    f"{root_count:,}"
)


propagated = propagated[
    propagated["go_id"]
    != ROOT_MF
].copy()


print("\n==============================")
print("FULL PROPAGATED MF DATASET")
print("==============================")

print(
    f"Annotations: "
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


print("\nAnnotation types:")

print(
    propagated[
        "annotation_type"
    ].value_counts()
)


propagated.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)