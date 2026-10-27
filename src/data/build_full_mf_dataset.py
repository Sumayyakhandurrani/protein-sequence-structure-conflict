from pathlib import Path
import pandas as pd


SEQ_FILE = Path(
    "data/processed/swissprot_sequences.csv"
)

GO_FILE = Path(
    "data/processed/full_mf_evidence_raw.csv"
)

OUTPUT_FILE = Path(
    "data/processed/full_mf_proteins.csv"
)


# ---------------------------------------------
# Load data
# ---------------------------------------------

seq = pd.read_csv(SEQ_FILE)
go = pd.read_csv(GO_FILE)


print("Sequence proteins:", len(seq))
print(
    "Experimental MF proteins:",
    go["uniprot_id"].nunique()
)


# ---------------------------------------------
# Merge
# ---------------------------------------------

merged = go.merge(
    seq,
    on="uniprot_id",
    how="inner"
)


print("\nAfter sequence intersection:")

print(
    "Proteins:",
    merged["uniprot_id"].nunique()
)

print(
    "Annotations:",
    len(merged)
)

print(
    "Unique GO terms:",
    merged["go_id"].nunique()
)


# ---------------------------------------------
# Group annotations per protein
# ---------------------------------------------

protein_df = (
    merged.groupby(
        [
            "uniprot_id",
            "sequence",
            "length"
        ]
    )
    .agg(
        go_terms=(
            "go_id",
            lambda x: sorted(set(x))
        ),
        evidence_codes=(
            "evidence_code",
            lambda x: sorted(set(x))
        )
    )
    .reset_index()
)


print("\n==============================")
print("FULL MF PROTEIN DATASET")
print("==============================")

print(
    f"Proteins: "
    f"{len(protein_df):,}"
)

print("\nLength statistics:")

print(
    protein_df["length"].describe()
)

print("\nGO labels per protein:")

print(
    protein_df["go_terms"]
    .apply(len)
    .describe()
)


protein_df.to_pickle(
    "data/processed/full_mf_proteins.pkl"
)

protein_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved:"
    "\ndata/processed/full_mf_proteins.pkl"
)

print(
    "data/processed/full_mf_proteins.csv"
)