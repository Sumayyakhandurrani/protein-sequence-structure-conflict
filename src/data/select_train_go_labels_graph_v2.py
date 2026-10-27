from pathlib import Path
import pandas as pd


BASE = Path("data/splits/mmseqs30_graph_v2")

TRAIN_FILE = BASE / "train.pkl"
PROPAGATED_FILE = Path("data/processed/full_mf_propagated.csv")

OUTPUT_FILE = BASE / "train_go_term_frequencies.csv"


print("=" * 70)
print("SELECT TRAIN GO LABELS - GRAPH V2")
print("=" * 70)

print("\nLoading training split...")

train = pd.read_pickle(TRAIN_FILE)

train["uniprot_id"] = train["uniprot_id"].astype(str)

print(f"Training proteins: {len(train):,}")

if len(train) != 38031:
    raise ValueError(
        f"Expected 38,031 training proteins, found {len(train):,}"
    )

if train["uniprot_id"].nunique() != len(train):
    raise ValueError("Duplicate training UniProt IDs detected.")


# ============================================================
# LOAD PROPAGATED GO-MF ANNOTATIONS
# ============================================================

print("\nLoading propagated GO-MF annotations...")

go = pd.read_csv(PROPAGATED_FILE)

required_cols = {"uniprot_id", "go_id"}

if not required_cols.issubset(go.columns):
    raise ValueError(
        f"Missing required columns: "
        f"{required_cols - set(go.columns)}"
    )

go["uniprot_id"] = go["uniprot_id"].astype(str)
go["go_id"] = go["go_id"].astype(str)

train_ids = set(train["uniprot_id"])


# ============================================================
# TRAINING ANNOTATIONS ONLY
# ============================================================

train_go = go[
    go["uniprot_id"].isin(train_ids)
].copy()

print(
    f"Training propagated annotations: "
    f"{len(train_go):,}"
)

print(
    f"Proteins represented: "
    f"{train_go['uniprot_id'].nunique():,}"
)


# ============================================================
# COUNT UNIQUE PROTEINS PER GO TERM
#
# We count proteins, not raw annotation rows.
# ============================================================

term_frequency = (
    train_go
    .drop_duplicates(
        subset=["uniprot_id", "go_id"]
    )
    .groupby("go_id")["uniprot_id"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="train_protein_count")
)


print(
    f"Unique GO terms: "
    f"{len(term_frequency):,}"
)


# ============================================================
# SUMMARY
# ============================================================

counts = term_frequency["train_protein_count"]

print("\nFrequency statistics:")

print(counts.describe().to_string())


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

print("\nGO terms by minimum training support:")

for threshold in thresholds:

    n = int(
        (counts >= threshold).sum()
    )

    print(
        f">= {threshold:4d}: "
        f"{n:,}"
    )


print("\nTop 20 GO-MF terms:")

print(
    term_frequency
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

term_frequency.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDONE")