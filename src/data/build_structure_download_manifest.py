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
print("STRUCTURE DOWNLOAD MANIFEST — TRAIN / VAL ONLY")
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

missing = [
    c
    for c in required_columns
    if c not in cohort.columns
]

if missing:

    raise RuntimeError(
        f"Missing cohort columns: {missing}"
    )


assert cohort[
    "uniprot_id"
].is_unique


print(
    "\nPrimary cohort:",
    len(cohort)
)


# ============================================================
# LOAD FROZEN IDS
# ============================================================

train_ids = read_ids(
    LABEL_DIR
    / "train_uniprot_ids.txt"
)

val_ids = read_ids(
    LABEL_DIR
    / "val_uniprot_ids.txt"
)

# We intentionally do NOT load test IDs.

assert len(train_ids) == 38031
assert len(val_ids) == 4754

assert len(set(train_ids)) == len(train_ids)
assert len(set(val_ids)) == len(val_ids)

assert set(train_ids).isdisjoint(
    set(val_ids)
)


print(
    "TRAIN IDs:",
    len(train_ids)
)

print(
    "VAL IDs:",
    len(val_ids)
)

print(
    "TEST IDs loaded: NO"
)


# ============================================================
# INDEX COHORT
# ============================================================

cohort_indexed = cohort.set_index(
    "uniprot_id",
    drop=False
)


def build_manifest(ids, split):

    missing_ids = [
        uid
        for uid in ids
        if uid not in cohort_indexed.index
    ]

    if missing_ids:

        raise RuntimeError(
            f"{split}: "
            f"{len(missing_ids)} IDs missing "
            "from primary cohort."
        )

    # Preserve exact frozen split order.
    df = cohort_indexed.loc[
        ids
    ].copy()

    assert list(
        df["uniprot_id"]
    ) == ids

    unavailable = (
        ~df[
            "alphafold_available"
        ].astype(bool)
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

    print(
        f"\n{split}"
    )

    print(
        "  Proteins:",
        len(df)
    )

    print(
        "  AlphaFold available:",
        int(
            (
                ~unavailable
            ).sum()
        )
    )

    print(
        "  AlphaFold unavailable:",
        int(
            unavailable.sum()
        )
    )

    print(
        "  Missing PDB URL:",
        int(
            missing_pdb.sum()
        )
    )

    print(
        "  Missing model ID:",
        int(
            missing_model.sum()
        )
    )

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

        print(
            "\nUnexpected unavailable proteins:"
        )

        print(
            bad.head(20).to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{split} contains proteins without "
            "AlphaFold availability."
        )

    if missing_pdb.any():

        raise RuntimeError(
            f"{split} contains missing PDB URLs."
        )

    if missing_model.any():

        raise RuntimeError(
            f"{split} contains missing model IDs."
        )

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
        split.lower()
    )

    return output


# ============================================================
# BUILD
# ============================================================

train_manifest = build_manifest(
    train_ids,
    "TRAIN"
)

val_manifest = build_manifest(
    val_ids,
    "VAL"
)


# ============================================================
# SAVE
# ============================================================

train_path = (
    OUTPUT_DIR
    / "train_structure_manifest.csv"
)

val_path = (
    OUTPUT_DIR
    / "val_structure_manifest.csv"
)

train_manifest.to_csv(
    train_path,
    index=False
)

val_manifest.to_csv(
    val_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "protocol":
        "structure_v1",

    "alphafold_model_version":
        "v6",

    "file_format_primary":
        "PDB",

    "train_proteins":
        len(train_manifest),

    "val_proteins":
        len(val_manifest),

    "development_proteins":
        (
            len(train_manifest)
            +
            len(val_manifest)
        ),

    "test_ids_loaded":
        False,

    "test_structures_requested":
        False,

    "structure_policy": {

        "graph_level":
            "residue",

        "coordinate":
            "CA",

        "node_features":
            [
                "amino_acid_identity",
                "residue_plddt",
            ],

        "edges":
            [
                "sequential_neighbors",
                "CA_spatial_contacts",
            ],

        "spatial_contact_cutoff_angstrom":
            10.0,

        "esm_embeddings_used":
            False,

        "low_plddt_proteins_removed":
            False,
    },
}


summary_path = (
    OUTPUT_DIR
    / "structure_v1_manifest_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


checksums = {

    train_path.name:
        sha256(train_path),

    val_path.name:
        sha256(val_path),

    summary_path.name:
        sha256(summary_path),
}


checksum_path = (
    OUTPUT_DIR
    / "SHA256SUMS.json"
)

with open(
    checksum_path,
    "w"
) as f:

    json.dump(
        checksums,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 78)
print("STRUCTURE MANIFEST COMPLETE")
print("=" * 78)

print(
    "\nTRAIN structures:",
    len(train_manifest)
)

print(
    "VAL structures:",
    len(val_manifest)
)

print(
    "Development total:",
    len(train_manifest)
    +
    len(val_manifest)
)

print("\nTEST IDs loaded: NO")
print("TEST structures requested: NO")

print("\nSaved:")
print(train_path)
print(val_path)
print(summary_path)
print(checksum_path)

print("=" * 78)
