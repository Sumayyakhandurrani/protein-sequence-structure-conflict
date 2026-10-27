import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/go_mf_experimental.csv")

df = pd.read_csv(INPUT_FILE)

term_counts = (
    df.groupby("go_id")["uniprot_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("Total experimental MF proteins:",
      df["uniprot_id"].nunique())

print("Total experimental MF GO terms:",
      df["go_id"].nunique())

print("\nGO term frequency statistics:")
print(term_counts.describe())

thresholds = [2, 5, 10, 20, 50, 100]

print("\nTerms with at least N proteins:")

for threshold in thresholds:
    count = (term_counts >= threshold).sum()
    print(f">= {threshold:3d}: {count:,} GO terms")

print("\nTop 30 GO terms:")
print(term_counts.head(30))

term_counts.to_csv(
    "data/processed/go_mf_term_frequencies.csv",
    header=["protein_count"]
)

print(
    "\nSaved to:"
    "\ndata/processed/go_mf_term_frequencies.csv"
)