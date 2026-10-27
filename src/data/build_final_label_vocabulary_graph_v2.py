from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path("data/splits/mmseqs30_graph_v2")

FREQ_FILE = BASE / "train_go_term_frequencies.csv"

PRIMARY_FILE = BASE / "primary_go_mf_labels.csv"
PRIMARY_IDS_FILE = BASE / "primary_go_mf_label_ids.txt"

RARE_FILE = BASE / "secondary_rare_go_mf_labels.csv"


# ============================================================
# FROZEN LABEL POLICY
# ============================================================

PRIMARY_MIN_SUPPORT = 75

RARE_MIN_SUPPORT = 50
RARE_MAX_SUPPORT = 74

EXPECTED_PRIMARY = 494
EXPECTED_RARE = 151


print("=" * 70)
print("BUILD FINAL GO-MF LABEL VOCABULARY - GRAPH V2")
print("=" * 70)

print("\nLoading TRAIN-only GO frequencies...")

freq = pd.read_csv(FREQ_FILE)

required = {
    "go_id",
    "train_protein_count",
}

if not required.issubset(freq.columns):
    raise ValueError(
        f"Missing required columns: "
        f"{required - set(freq.columns)}"
    )

freq["go_id"] = freq["go_id"].astype(str)

freq["train_protein_count"] = pd.to_numeric(
    freq["train_protein_count"],
    errors="raise",
).astype(int)


if freq["go_id"].duplicated().any():
    raise ValueError(
        "Duplicate GO IDs found in frequency table."
    )


print(f"Total TRAIN GO terms: {len(freq):,}")


# ============================================================
# PRIMARY VOCABULARY
#
# Frozen policy:
# training protein support >= 75
# ============================================================

primary = freq[
    freq["train_protein_count"] >= PRIMARY_MIN_SUPPORT
].copy()

primary = primary.sort_values(
    ["train_protein_count", "go_id"],
    ascending=[False, True],
).reset_index(drop=True)


# Keep the historical final schema:
#
# go_id | train_protein_count | label_index
#
primary["label_index"] = range(len(primary))


# ============================================================
# SECONDARY RARE-LABEL SET
#
# Frozen policy:
# training support 50-74
# ============================================================

rare = freq[
    (freq["train_protein_count"] >= RARE_MIN_SUPPORT)
    & (freq["train_protein_count"] <= RARE_MAX_SUPPORT)
].copy()

rare = rare.sort_values(
    ["train_protein_count", "go_id"],
    ascending=[False, True],
).reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

print("\nPrimary vocabulary:")
print(
    f"Training support >= {PRIMARY_MIN_SUPPORT}"
)
print(f"Labels: {len(primary):,}")

print("\nSecondary rare-label vocabulary:")
print(
    f"Training support "
    f"{RARE_MIN_SUPPORT}-{RARE_MAX_SUPPORT}"
)
print(f"Labels: {len(rare):,}")


if len(primary) != EXPECTED_PRIMARY:
    raise ValueError(
        f"Expected {EXPECTED_PRIMARY} primary labels, "
        f"found {len(primary)}."
    )

if len(rare) != EXPECTED_RARE:
    raise ValueError(
        f"Expected {EXPECTED_RARE} rare labels, "
        f"found {len(rare)}."
    )


if primary["train_protein_count"].min() < 75:
    raise ValueError(
        "Primary vocabulary contains term below support 75."
    )

if rare["train_protein_count"].min() < 50:
    raise ValueError(
        "Rare vocabulary contains term below support 50."
    )

if rare["train_protein_count"].max() > 74:
    raise ValueError(
        "Rare vocabulary contains term above support 74."
    )


primary_ids = set(primary["go_id"])
rare_ids = set(rare["go_id"])

if primary_ids & rare_ids:
    raise ValueError(
        "Primary and rare vocabularies overlap."
    )


expected_indices = list(
    range(EXPECTED_PRIMARY)
)

if primary["label_index"].tolist() != expected_indices:
    raise ValueError(
        "Primary label indices are not contiguous 0..493."
    )


# ============================================================
# SAVE
# ============================================================

primary.to_csv(
    PRIMARY_FILE,
    index=False,
)

rare.to_csv(
    RARE_FILE,
    index=False,
)

with open(
    PRIMARY_IDS_FILE,
    "w",
    encoding="utf-8",
) as f:

    for go_id in primary["go_id"]:
        f.write(f"{go_id}\n")


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL LABEL VOCABULARY SUMMARY")
print("=" * 70)

print(
    f"Primary labels:       {len(primary):,}"
)
print(
    f"Secondary rare:       {len(rare):,}"
)

print(
    "Primary support range: "
    f"{primary['train_protein_count'].min():,}"
    " - "
    f"{primary['train_protein_count'].max():,}"
)

print(
    "Rare support range:    "
    f"{rare['train_protein_count'].min():,}"
    " - "
    f"{rare['train_protein_count'].max():,}"
)


print("\nTop 20 primary labels:")

print(
    primary.head(20).to_string(
        index=False
    )
)


print("\nSaved:")
print(PRIMARY_FILE)
print(PRIMARY_IDS_FILE)
print(RARE_FILE)

print("\nDONE")