from pathlib import Path
import pandas as pd


BASE = Path("data/splits/mmseqs30_graph_v2")

SPLITS = {
    "train": BASE / "train.pkl",
    "validation": BASE / "val.pkl",
    "test": BASE / "test.pkl",
}

OUTPUTS = {
    "train": BASE / "train.fasta",
    "validation": BASE / "val.fasta",
    "test": BASE / "test.fasta",
}

EXPECTED = {
    "train": 38031,
    "validation": 4754,
    "test": 4754,
}


def write_fasta(df, path):
    with open(path, "w", encoding="utf-8") as f:

        for row in df.itertuples(index=False):

            protein_id = str(row.uniprot_id)
            sequence = str(row.sequence).strip()

            if not protein_id:
                raise ValueError("Empty UniProt ID detected")

            if not sequence:
                raise ValueError(
                    f"Empty sequence detected for {protein_id}"
                )

            f.write(f">{protein_id}\n")
            f.write(f"{sequence}\n")


print("=" * 70)
print("EXPORT GRAPH-V2 SPLIT FASTAS")
print("=" * 70)

all_ids = {}

for split in ["train", "validation", "test"]:

    print(f"\nProcessing {split}...")

    df = pd.read_pickle(SPLITS[split])

    required = {"uniprot_id", "sequence"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{split} is missing required columns: "
            f"{required - set(df.columns)}"
        )

    if len(df) != EXPECTED[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED[split]:,} proteins, "
            f"found {len(df):,}"
        )

    if df["uniprot_id"].nunique() != len(df):
        raise ValueError(
            f"{split}: duplicate UniProt IDs detected"
        )

    all_ids[split] = set(
        df["uniprot_id"].astype(str)
    )

    write_fasta(
        df,
        OUTPUTS[split],
    )

    fasta_count = 0

    with open(
        OUTPUTS[split],
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if line.startswith(">"):
                fasta_count += 1

    if fasta_count != EXPECTED[split]:
        raise ValueError(
            f"{split}: FASTA has {fasta_count:,} records; "
            f"expected {EXPECTED[split]:,}"
        )

    print(
        f"{split}: {fasta_count:,} sequences -> "
        f"{OUTPUTS[split]}"
    )


# ============================================================
# DISJOINTNESS CHECK
# ============================================================

if all_ids["train"] & all_ids["validation"]:
    raise ValueError(
        "Train/validation protein overlap detected"
    )

if all_ids["train"] & all_ids["test"]:
    raise ValueError(
        "Train/test protein overlap detected"
    )

if all_ids["validation"] & all_ids["test"]:
    raise ValueError(
        "Validation/test protein overlap detected"
    )

total_ids = (
    all_ids["train"]
    | all_ids["validation"]
    | all_ids["test"]
)

if len(total_ids) != 47539:
    raise ValueError(
        f"Expected 47,539 unique proteins, "
        f"found {len(total_ids):,}"
    )


print("\n" + "=" * 70)
print("FASTA EXPORT SUMMARY")
print("=" * 70)

print("Train:      38,031")
print("Validation:  4,754")
print("Test:        4,754")
print("Total:      47,539")

print("\nProtein split disjointness: PASS")
print("DONE")