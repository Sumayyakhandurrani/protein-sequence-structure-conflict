from pathlib import Path
import pandas as pd


BASE = Path("data/splits/mmseqs30")

FREQ_FILE = BASE / "train_go_term_frequencies.csv"

PRIMARY_VOCAB = BASE / "primary_go_mf_labels.csv"
PRIMARY_IDS = BASE / "primary_go_mf_label_ids.txt"

RARE_VOCAB = BASE / "secondary_rare_go_mf_labels.csv"

PRIMARY_THRESHOLD = 75
RARE_MIN = 50
RARE_MAX = 74


print("Loading training GO frequencies...")

freq = pd.read_csv(FREQ_FILE)


# =========================================================
# PRIMARY VOCABULARY
# =========================================================

primary = freq[
    freq["train_protein_count"] >= PRIMARY_THRESHOLD
].copy()

primary = primary.sort_values(
    ["train_protein_count", "go_id"],
    ascending=[False, True]
).reset_index(drop=True)

primary["label_index"] = range(len(primary))


# =========================================================
# SECONDARY RARE-LABEL SET
# =========================================================

rare = freq[
    (freq["train_protein_count"] >= RARE_MIN)
    &
    (freq["train_protein_count"] <= RARE_MAX)
].copy()

rare = rare.sort_values(
    ["train_protein_count", "go_id"],
    ascending=[False, True]
).reset_index(drop=True)


# =========================================================
# SAVE
# =========================================================

primary.to_csv(
    PRIMARY_VOCAB,
    index=False
)

primary[
    ["go_id"]
].to_csv(
    PRIMARY_IDS,
    index=False,
    header=False
)

rare.to_csv(
    RARE_VOCAB,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("PRIMARY GO-MF VOCABULARY")
print("=" * 60)

print(
    f"Training support threshold: "
    f">= {PRIMARY_THRESHOLD}"
)

print(
    f"Primary labels: "
    f"{len(primary):,}"
)

print(
    f"Minimum training support: "
    f"{primary['train_protein_count'].min():,}"
)

print(
    f"Maximum training support: "
    f"{primary['train_protein_count'].max():,}"
)


print("\n" + "=" * 60)
print("SECONDARY RARE-LABEL SET")
print("=" * 60)

print(
    f"Training support range: "
    f"{RARE_MIN}-{RARE_MAX}"
)

print(
    f"Rare labels: "
    f"{len(rare):,}"
)


print("\nTop 20 primary labels:")

print(
    primary.head(20).to_string(
        index=False
    )
)


print("\nSaved:")

print(PRIMARY_VOCAB)
print(PRIMARY_IDS)
print(RARE_VOCAB)