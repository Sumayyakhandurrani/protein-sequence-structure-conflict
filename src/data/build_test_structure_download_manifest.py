from pathlib import Path
import hashlib
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

COHORT_FILE = Path(
    "data/processed/primary_mf_cohort.csv"
)

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2"
)

LABEL_DIR = SPLIT_DIR / "labels"

OUTPUT_DIR = Path(
    "data/structures/manifests"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_IDS_PATH = (
    LABEL_DIR / "test_uniprot_ids.txt"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "test_structure_manifest.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "test_structure_manifest_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# LOAD PRIMARY COHORT
# ============================================================

print("=" * 78)
print("STRUCTURE DOWNLOAD MANIFEST — TEST ONLY")
print("=" * 78)

cohort = pd.read_csv(
    COHORT_FILE
)

required_columns = [
    "uniprot_id",
    "sequence",
    "length",
    "alphafold_available",
    "model_id",
    "cif_url",
    "pdb_url",
    "global_metric",
]

missing_columns = [
    c
    for c in required_columns
    if c not in cohort.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing cohort columns: {missing_columns}"
    )

if not cohort["uniprot_id"].is_unique:
    raise RuntimeError(
        "Primary cohort contains duplicate UniProt IDs."
    )

print("Primary cohort:", len(cohort))


# ============================================================
# LOAD FROZEN TEST IDS
# ============================================================

test_ids = read_ids(
    TEST_IDS_PATH
)

if len(test_ids) != 4754:
    raise RuntimeError(
        f"Expected 4,754 TEST IDs, got {len(test_ids):,}"
    )

if len(set(test_ids)) != len(test_ids):
    raise RuntimeError(
        "Duplicate TEST IDs detected."
    )

print("TEST IDs:", len(test_ids))


# ============================================================
# INDEX COHORT
# ============================================================

cohort_indexed = cohort.set_index(
    "uniprot_id",
    drop=False
)

missing_ids = [
    uid
    for uid in test_ids
    if uid not in cohort_indexed.index
]

if missing_ids:
    raise RuntimeError(
        f"{len(missing_ids)} TEST IDs missing from primary cohort."
    )


# ============================================================
# BUILD TEST MANIFEST
# ============================================================

# Preserve exact frozen TEST order.
df = cohort_indexed.loc[
    test_ids
].copy()

if list(df["uniprot_id"]) != test_ids:
    raise RuntimeError(
        "TEST manifest order differs from frozen TEST order."
    )

unavailable = (
    ~df["alphafold_available"].astype(bool)
)

missing_pdb = (
    df["pdb_url"].isna()
    |
    (
        df["pdb_url"]
        .astype(str)
        .str.strip()
        == ""
    )
)

missing_model = (
    df["model_id"].isna()
)

print()
print("TEST")
print("  Proteins:", len(df))
print(
    "  AlphaFold available:",
    int((~unavailable).sum())
)
print(
    "  AlphaFold unavailable:",
    int(unavailable.sum())
)
print(
    "  Missing PDB URL:",
    int(missing_pdb.sum())
)
print(
    "  Missing model ID:",
    int(missing_model.sum())
)


# ============================================================
# SAME STRICT POLICY AS TRAIN / VAL
# ============================================================

if unavailable.any():

    bad = df.loc[
        unavailable,
        [
            "uniprot_id",
            "alphafold_available",
            "model_id",
            "pdb_url",
        ],
    ]

    print()
    print("Unexpected unavailable TEST proteins:")
    print(
        bad.head(20).to_string(
            index=False
        )
    )

    raise RuntimeError(
        "TEST contains proteins without AlphaFold availability."
    )

if missing_pdb.any():
    raise RuntimeError(
        "TEST contains missing PDB URLs."
    )

if missing_model.any():
    raise RuntimeError(
        "TEST contains missing model IDs."
    )


# ============================================================
# EXACT SAME MANIFEST COLUMNS
# ============================================================

output = df[
    [
        "uniprot_id",
        "length",
        "model_id",
        "pdb_url",
        "cif_url",
        "global_metric",
    ]
].copy()

output.insert(
    0,
    "split",
    "test"
)

if list(output["uniprot_id"]) != test_ids:
    raise RuntimeError(
        "Final TEST manifest row order mismatch."
    )


# ============================================================
# SAVE
# ============================================================

output.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# VERIFY SAVED FILE
# ============================================================

saved = pd.read_csv(
    OUTPUT_PATH
)

if saved["uniprot_id"].astype(str).tolist() != test_ids:
    raise RuntimeError(
        "Saved TEST manifest order mismatch."
    )

if len(saved) != 4754:
    raise RuntimeError(
        "Saved TEST manifest row count mismatch."
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "protocol": "structure_v1",
    "alphafold_model_version": "v6",
    "file_format_primary": "PDB",
    "split": "test",
    "test_proteins": len(output),
    "alphafold_available": int(
        (~unavailable).sum()
    ),
    "alphafold_unavailable": int(
        unavailable.sum()
    ),
    "missing_pdb_url": int(
        missing_pdb.sum()
    ),
    "missing_model_id": int(
        missing_model.sum()
    ),
    "frozen_test_order_preserved": True,
    "test_labels_loaded": False,
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 78)
print("TEST STRUCTURE MANIFEST COMPLETE")
print("=" * 78)

print("Rows:", len(output))
print("Frozen TEST order preserved: YES")
print("TEST labels loaded: NO")

print()
print("Saved:")
print(OUTPUT_PATH)
print("SHA256:", sha256(OUTPUT_PATH))

print()
print(SUMMARY_PATH)
print("SHA256:", sha256(SUMMARY_PATH))

print("=" * 78)
