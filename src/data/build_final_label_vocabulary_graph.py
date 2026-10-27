from pathlib import Path
import pandas as pd


BASE = Path("data/splits/mmseqs30_graph")

FREQ_FILE = BASE / "train_go_term_frequencies.csv"

PRIMARY_CSV = BASE / "primary_go_mf_labels.csv"
PRIMARY_TXT = BASE / "primary_go_mf_label_ids.txt"

SECONDARY_CSV = BASE / "secondary_rare_go_mf_labels.csv"
SECONDARY_TXT = BASE / "secondary_rare_go_mf_label_ids.txt"


PRIMARY_THRESHOLD = 75
SECONDARY_MIN = 50
SECONDARY_MAX = 74


print("Loading training GO term frequencies...")

freq = pd.read_csv(FREQ_FILE)

print(f"Training GO terms: {len(freq):,}")


# ---------------------------------------------------------
# Primary vocabulary
# ---------------------------------------------------------

primary = freq[
    freq["train_protein_count"] >= PRIMARY_THRESHOLD
].copy()

primary = primary.sort_values(
    ["train_protein_count", "go_id"],
    ascending=[False, True]
).reset_index(drop=True)


# ---------------------------------------------------------
# Secondary rare-label vocabulary
# ---------------------------------------------------------

secondary = freq[
    (freq["train_protein_count"] >= SECONDARY_MIN)
    &
    (freq["train_protein_count"] <= SECONDARY_MAX)
].copy()

secondary = secondary.sort_values(
    ["train_protein_count", "go_id"],
    ascending=[False, True]
).reset_index(drop=True)


# ---------------------------------------------------------
# Add stable label indices
# ---------------------------------------------------------

primary.insert(
    0,
    "label_index",
    range(len(primary))
)

secondary.insert(
    0,
    "label_index",
    range(len(secondary))
)


# ---------------------------------------------------------
# Save CSV files
# ---------------------------------------------------------

primary.to_csv(
    PRIMARY_CSV,
    index=False
)

secondary.to_csv(
    SECONDARY_CSV,
    index=False
)


# ---------------------------------------------------------
# Save plain GO-ID lists
# ---------------------------------------------------------

with open(PRIMARY_TXT, "w") as f:

    for go_id in primary["go_id"]:
        f.write(go_id + "\n")


with open(SECONDARY_TXT, "w") as f:

    for go_id in secondary["go_id"]:
        f.write(go_id + "\n")


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL TRAIN-DERIVED GO-MF VOCABULARY")
print("=" * 70)

print(
    f"Primary threshold: "
    f">= {PRIMARY_THRESHOLD} training proteins"
)

print(
    f"Primary labels: "
    f"{len(primary):,}"
)

print()

print(
    f"Secondary range: "
    f"{SECONDARY_MIN}-{SECONDARY_MAX} training proteins"
)

print(
    f"Secondary labels: "
    f"{len(secondary):,}"
)


print("\nPrimary training support:")

print(
    primary["train_protein_count"]
    .describe()
)


print("\nSecondary training support:")

print(
    secondary["train_protein_count"]
    .describe()
)


print("\nSaved:")

print(PRIMARY_CSV)
print(PRIMARY_TXT)
print(SECONDARY_CSV)
print(SECONDARY_TXT)