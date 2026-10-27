import gzip
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# --------------------------------------------------
# File locations
# --------------------------------------------------

INPUT_FILE = Path("data/raw/uniprot_sprot.fasta.gz")
OUTPUT_FILE = Path("data/processed/swissprot_sequences.csv")


# --------------------------------------------------
# Check that the input exists
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}"
    )


# Make sure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Read Swiss-Prot FASTA
# --------------------------------------------------

records = []

with gzip.open(INPUT_FILE, "rt") as handle:

    for record in SeqIO.parse(handle, "fasta"):

        # Example FASTA identifier:
        #
        # sp|P12345|PROTEIN_NAME
        #
        # We want P12345.

        parts = record.id.split("|")

        if len(parts) >= 2:
            accession = parts[1]
        else:
            accession = record.id

        sequence = str(record.seq)

        records.append(
            {
                "uniprot_id": accession,
                "sequence": sequence,
                "length": len(sequence),
            }
        )


# --------------------------------------------------
# Convert to DataFrame
# --------------------------------------------------

df = pd.DataFrame(records)


# --------------------------------------------------
# Basic quality checks
# --------------------------------------------------

print("\n========== SWISS-PROT SUMMARY ==========")

print(f"Number of proteins: {len(df):,}")

print(
    f"Unique UniProt IDs: "
    f"{df['uniprot_id'].nunique():,}"
)

print(
    f"Duplicate IDs: "
    f"{df['uniprot_id'].duplicated().sum():,}"
)

print("\nProtein length statistics:")

print(df["length"].describe())


# --------------------------------------------------
# Show examples
# --------------------------------------------------

print("\nFirst five proteins:")

print(
    df[
        ["uniprot_id", "length"]
    ].head()
)


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved dataset to:\n{OUTPUT_FILE}"
)

print("\nDone.")