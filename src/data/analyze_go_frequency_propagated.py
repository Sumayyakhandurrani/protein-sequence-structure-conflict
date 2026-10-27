import pandas as pd

INPUT_FILE = (
    "data/processed/go_mf_propagated.csv"
)

df = pd.read_csv(INPUT_FILE)

term_counts = (
    df.groupby("go_id")["uniprot_id"]
    .nunique()
    .sort_values(ascending=False)
)

print(
    "Proteins:",
    df["uniprot_id"].nunique()
)

print(
    "Unique GO terms:",
    df["go_id"].nunique()
)

print("\nFrequency statistics:")

print(
    term_counts.describe()
)

thresholds = [
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
]

print(
    "\nTerms with at least N proteins:"
)

for threshold in thresholds:

    n = (
        term_counts >= threshold
    ).sum()

    print(
        f">= {threshold:3d}: "
        f"{n:,} GO terms"
    )


print("\nTop 30 GO terms:")

print(
    term_counts.head(30)
)

term_counts.to_csv(
    "data/processed/"
    "go_mf_propagated_term_frequencies.csv",
    header=["protein_count"]
)

print("\nDone.")