import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "full_mf_propagated.csv"
)


df = pd.read_csv(INPUT_FILE)


counts = (
    df.groupby("go_id")[
        "uniprot_id"
    ]
    .nunique()
    .sort_values(
        ascending=False
    )
)


print(
    "Proteins:",
    df["uniprot_id"].nunique()
)

print(
    "GO terms:",
    df["go_id"].nunique()
)


print("\nFrequency statistics:")

print(
    counts.describe()
)


thresholds = [
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
]


print(
    "\nTerms with at least N proteins:"
)


for t in thresholds:

    n = (
        counts >= t
    ).sum()

    print(
        f">= {t:5d}: "
        f"{n:,}"
    )


print(
    "\nTop 40 GO terms:"
)

print(
    counts.head(40)
)


counts.to_csv(
    "data/processed/"
    "full_mf_term_frequencies.csv",
    header=["protein_count"]
)