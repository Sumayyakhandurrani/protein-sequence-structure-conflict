from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/go_annotations_raw.csv"
)

OUTPUT_FILE = Path(
    "data/processed/go_mf_experimental.csv"
)


# Experimental GO evidence codes
EXPERIMENTAL_CODES = {
    "EXP",
    "IDA",
    "IPI",
    "IMP",
    "IGI",
    "IEP",
}


df = pd.read_csv(INPUT_FILE)


print("All GO annotations:", len(df))

print(
    "Proteins with any GO:",
    df["uniprot_id"].nunique()
)


# ---------------------------------------
# Molecular Function only
# ---------------------------------------

mf = df[
    df["aspect"] == "F"
].copy()

print("\nMF annotations:", len(mf))

print(
    "Proteins with MF:",
    mf["uniprot_id"].nunique()
)


# ---------------------------------------
# Experimental evidence only
# ---------------------------------------

mf_exp = mf[
    mf["evidence_code"].isin(
        EXPERIMENTAL_CODES
    )
].copy()


mf_exp = mf_exp.drop_duplicates(
    subset=[
        "uniprot_id",
        "go_id",
        "evidence_code",
    ]
)


print("\n==============================")
print("EXPERIMENTAL MF DATASET")
print("==============================")

print(
    f"Annotations: {len(mf_exp):,}"
)

print(
    f"Proteins: "
    f"{mf_exp['uniprot_id'].nunique():,}"
)

print(
    f"Unique GO-MF terms: "
    f"{mf_exp['go_id'].nunique():,}"
)

print("\nEvidence distribution:")

print(
    mf_exp["evidence_code"]
    .value_counts()
)


mf_exp.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)