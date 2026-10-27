from pathlib import Path
import ast
import pandas as pd

PROTEIN_FILE = Path(
    "data/processed/full_mf_proteins.pkl"
)

GO_FILE = Path(
    "data/processed/full_mf_propagated.csv"
)

AF_FILE = Path(
    "data/processed/alphafold_availability.csv"
)

OUTPUT_PKL = Path(
    "data/processed/structure_ready_proteins.pkl"
)

OUTPUT_CSV = Path(
    "data/processed/structure_ready_proteins.csv"
)


print("Loading protein dataset...")
proteins = pd.read_pickle(PROTEIN_FILE)

print("Loading propagated GO annotations...")
go = pd.read_csv(GO_FILE)

print("Loading AlphaFold availability...")
af = pd.read_csv(AF_FILE)


# ---------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------

print("\n==============================")
print("INPUT SUMMARY")
print("==============================")

print(f"Sequence proteins: {len(proteins):,}")
print(f"GO proteins: {go['uniprot_id'].nunique():,}")
print(f"AlphaFold records: {len(af):,}")

print(
    f"AlphaFold available: "
    f"{af['alphafold_available'].sum():,}"
)


# ---------------------------------------------------------
# Restrict GO labels to proteins that actually have sequences
# ---------------------------------------------------------

sequence_ids = set(
    proteins["uniprot_id"].astype(str)
)

go = go[
    go["uniprot_id"].astype(str).isin(sequence_ids)
].copy()

print("\nAfter sequence/GO reconciliation:")

print(
    f"GO proteins retained: "
    f"{go['uniprot_id'].nunique():,}"
)

print(
    f"GO annotations retained: "
    f"{len(go):,}"
)

print(
    f"Unique GO terms retained: "
    f"{go['go_id'].nunique():,}"
)


# ---------------------------------------------------------
# Group propagated GO labels by protein
# ---------------------------------------------------------

go_grouped = (
    go.groupby("uniprot_id")
    .agg(
        propagated_go_terms=(
            "go_id",
            lambda x: sorted(set(x))
        )
    )
    .reset_index()
)


# ---------------------------------------------------------
# Merge protein + propagated GO
# ---------------------------------------------------------

master = proteins.merge(
    go_grouped,
    on="uniprot_id",
    how="inner"
)


# ---------------------------------------------------------
# Merge AlphaFold metadata
# ---------------------------------------------------------

af_cols = [
    "uniprot_id",
    "alphafold_available",
    "model_id",
    "cif_url",
    "pdb_url",
    "global_metric",
]

available_cols = [
    c for c in af_cols
    if c in af.columns
]

master = master.merge(
    af[available_cols],
    on="uniprot_id",
    how="left"
)


# ---------------------------------------------------------
# Keep proteins with confirmed AlphaFold structures
# ---------------------------------------------------------

structure_ready = master[
    master["alphafold_available"] == True
].copy()


# ---------------------------------------------------------
# Label counts
# ---------------------------------------------------------

structure_ready[
    "n_propagated_labels"
] = structure_ready[
    "propagated_go_terms"
].apply(len)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n==============================")
print("STRUCTURE-READY DATASET")
print("==============================")

print(
    f"Proteins after all intersections: "
    f"{len(structure_ready):,}"
)

print(
    f"Removed due to no AlphaFold structure: "
    f"{len(master) - len(structure_ready):,}"
)


print("\nLength statistics:")

print(
    structure_ready["length"].describe()
)


print("\nPropagated labels per protein:")

print(
    structure_ready[
        "n_propagated_labels"
    ].describe()
)


if "global_metric" in structure_ready.columns:

    metric = pd.to_numeric(
        structure_ready["global_metric"],
        errors="coerce"
    )

    print("\nAlphaFold global metric statistics:")

    print(
        metric.describe()
    )

    print(
        "\nMissing global metric:"
    )

    print(
        metric.isna().sum()
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

structure_ready.to_pickle(
    OUTPUT_PKL
)

structure_ready.to_csv(
    OUTPUT_CSV,
    index=False
)


print("\nSaved:")

print(OUTPUT_PKL)
print(OUTPUT_CSV)