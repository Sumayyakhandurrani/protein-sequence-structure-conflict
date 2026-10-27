from pathlib import Path
import pandas as pd


TRAIN_FILE = Path(
    "data/splits/mmseqs30_graph/train.pkl"
)

GO_FILE = Path(
    "data/processed/full_mf_propagated.csv"
)

OUTPUT_FREQ = Path(
    "data/splits/mmseqs30_graph/train_go_term_frequencies.csv"
)


print("Loading training proteins...")

train = pd.read_pickle(
    TRAIN_FILE
)

print(
    f"Training proteins: "
    f"{len(train):,}"
)


print("Loading propagated GO annotations...")

go = pd.read_csv(
    GO_FILE
)


train_ids = set(
    train["uniprot_id"]
)


train_go = go[
    go["uniprot_id"].isin(train_ids)
].copy()


print("\n" + "=" * 60)
print("TRAINING GO SUMMARY")
print("=" * 60)

print(
    f"Proteins with propagated GO: "
    f"{train_go['uniprot_id'].nunique():,}"
)

print(
    f"Annotations: "
    f"{len(train_go):,}"
)

print(
    f"Unique GO terms: "
    f"{train_go['go_id'].nunique():,}"
)


# ---------------------------------------------------------
# Frequency = unique training proteins per GO term
# ---------------------------------------------------------

freq = (
    train_go.groupby("go_id")["uniprot_id"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="train_protein_count")
)


print("\n" + "=" * 60)
print("TRAIN GO TERM FREQUENCIES")
print("=" * 60)

print(
    freq["train_protein_count"].describe()
)


print("\nTerms with at least N TRAIN proteins:")

thresholds = [
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    200,
    500,
    1000,
]

for t in thresholds:

    n = (
        freq["train_protein_count"] >= t
    ).sum()

    print(
        f">= {t:4d}: "
        f"{n:5,d} GO terms"
    )


print("\nTop 40 training GO terms:")

print(
    freq.head(40).to_string(
        index=False
    )
)


freq.to_csv(
    OUTPUT_FREQ,
    index=False
)


print("\nSaved:")

print(
    OUTPUT_FREQ
)