from pathlib import Path
import pandas as pd


INPUT = Path(
    "data/processed/structure_ready_proteins.pkl"
)

OUTPUT_PKL = Path(
    "data/processed/primary_mf_cohort.pkl"
)

OUTPUT_CSV = Path(
    "data/processed/primary_mf_cohort.csv"
)

OUTPUT_FASTA = Path(
    "data/processed/primary_mf_cohort.fasta"
)

OUTPUT_IDS = Path(
    "data/processed/primary_mf_cohort_accessions.txt"
)


MIN_LENGTH = 50
MAX_LENGTH = 2000


print("Loading structure-ready dataset...")

df = pd.read_pickle(INPUT)

print(f"Input proteins: {len(df):,}")


# ---------------------------------------------------------
# Apply primary computational length policy
# ---------------------------------------------------------

cohort = df[
    (df["length"] >= MIN_LENGTH)
    &
    (df["length"] <= MAX_LENGTH)
].copy()


# ---------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------

assert cohort["uniprot_id"].is_unique, \
    "Duplicate UniProt IDs detected."

assert cohort["sequence"].notna().all(), \
    "Missing sequences detected."

assert cohort["alphafold_available"].eq(True).all(), \
    "Protein without confirmed AlphaFold model detected."

assert (
    cohort["length"]
    ==
    cohort["sequence"].str.len()
).all(), \
    "Stored length does not match sequence length."


print("\n" + "=" * 60)
print("PRIMARY PRE-SPLIT MF COHORT")
print("=" * 60)

print(f"Length policy: {MIN_LENGTH}-{MAX_LENGTH}")

print(
    f"Proteins retained: "
    f"{len(cohort):,}"
)

print(
    f"Proteins removed: "
    f"{len(df) - len(cohort):,}"
)

print(
    f"Percent retained: "
    f"{100 * len(cohort) / len(df):.2f}%"
)


print("\nLength statistics:")

print(
    cohort["length"].describe()
)


print("\nAlphaFold confidence statistics:")

print(
    pd.to_numeric(
        cohort["global_metric"],
        errors="coerce"
    ).describe()
)


print("\nPropagated labels per protein:")

print(
    cohort[
        "n_propagated_labels"
    ].describe()
)


# ---------------------------------------------------------
# Save tabular datasets
# ---------------------------------------------------------

cohort.to_pickle(
    OUTPUT_PKL
)

cohort.to_csv(
    OUTPUT_CSV,
    index=False
)


# ---------------------------------------------------------
# Save accession list
# ---------------------------------------------------------

cohort[
    ["uniprot_id"]
].to_csv(
    OUTPUT_IDS,
    index=False,
    header=False
)


# ---------------------------------------------------------
# Create FASTA for sequence clustering
# ---------------------------------------------------------

with open(
    OUTPUT_FASTA,
    "w",
    encoding="utf-8"
) as f:

    for row in cohort.itertuples():

        f.write(
            f">{row.uniprot_id}\n"
        )

        sequence = str(
            row.sequence
        ).strip()

        # 80 residues per FASTA line
        for i in range(
            0,
            len(sequence),
            80
        ):

            f.write(
                sequence[i:i+80]
                + "\n"
            )


# ---------------------------------------------------------
# Verify FASTA
# ---------------------------------------------------------

with open(
    OUTPUT_FASTA,
    "r",
    encoding="utf-8"
) as f:

    fasta_headers = sum(
        1
        for line in f
        if line.startswith(">")
    )

assert fasta_headers == len(cohort), \
    "FASTA protein count mismatch."


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(OUTPUT_PKL)
print(OUTPUT_CSV)
print(OUTPUT_FASTA)
print(OUTPUT_IDS)

print(
    f"\nFASTA sequences: "
    f"{fasta_headers:,}"
)

print(
    "\nPrimary pre-split cohort successfully created."
)