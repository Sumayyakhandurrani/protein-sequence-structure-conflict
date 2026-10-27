import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/swissprot_sequences.csv")
OUTPUT_FILE = Path("data/processed/swissprot_candidates.csv")

df = pd.read_csv(INPUT_FILE)

print("Original proteins:", len(df))

# Temporary development-length filter
filtered = df[
    (df["length"] >= 50) &
    (df["length"] <= 1500)
].copy()

print("After length filtering:", len(filtered))

# Reproducible sample
candidate_df = filtered.sample(
    n=min(20000, len(filtered)),
    random_state=42
)

candidate_df = candidate_df.sort_values("uniprot_id")

candidate_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Candidate proteins:", len(candidate_df))
print("Saved to:", OUTPUT_FILE)

print("\nLength statistics:")
print(candidate_df["length"].describe())

print("\nFirst 10 proteins:")
print(candidate_df[["uniprot_id", "length"]].head(10))