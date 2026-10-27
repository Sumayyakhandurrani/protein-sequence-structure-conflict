from pathlib import Path
import hashlib
import json

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

VALIDATION_PATH = Path(
    "data/structures/alphafold_v6/test/"
    "structure_validation.csv"
)

FROZEN_TEST_IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/"
    "test_uniprot_ids.txt"
)

OUTPUT_ROOT = Path(
    "data/structures/structure_protocol_v1"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

EXPECTED_COUNT = 4754


# ============================================================
# HELPERS
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def to_bool_series(series):

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    unknown = sorted(
        set(normalized.unique())
        - set(mapping.keys())
    )

    if unknown:
        raise RuntimeError(
            "Unexpected boolean values in "
            f"sequence_exact_match: {unknown}"
        )

    return normalized.map(mapping).astype(bool)


# ============================================================
# LOAD FROZEN TEST IDS
# ============================================================

test_ids = [
    line.strip()
    for line in
    FROZEN_TEST_IDS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

if len(test_ids) != EXPECTED_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_COUNT} frozen TEST IDs, "
        f"found {len(test_ids)}."
    )

if len(test_ids) != len(set(test_ids)):
    raise RuntimeError(
        "Duplicate frozen TEST IDs detected."
    )


# ============================================================
# LOAD VALIDATION RESULTS
# ============================================================

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Missing validation file: {VALIDATION_PATH}"
    )

df = pd.read_csv(
    VALIDATION_PATH
)

required = [
    "uniprot_id",
    "sequence_exact_match",
]

for col in required:
    if col not in df.columns:
        raise RuntimeError(
            f"Missing required column: {col}"
        )

if len(df) != EXPECTED_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_COUNT} validation rows, "
        f"found {len(df)}."
    )

df["uniprot_id"] = (
    df["uniprot_id"]
    .astype(str)
    .str.strip()
)

if df["uniprot_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate UniProt IDs in TEST validation."
    )

if df["uniprot_id"].eq("").any():
    raise RuntimeError(
        "Empty UniProt ID in TEST validation."
    )


# ============================================================
# VERIFY EXACT FROZEN TEST MEMBERSHIP + ORDER
# ============================================================

validation_ids = df[
    "uniprot_id"
].tolist()

if validation_ids != test_ids:
    raise RuntimeError(
        "TEST validation IDs/order do not exactly match "
        "the frozen TEST split."
    )


# ============================================================
# FROZEN ELIGIBILITY RULE
#
# EXACTLY the same rule used for TRAIN / VAL:
#
# structure eligible iff sequence_exact_match == True
#
# No alignment rescue.
# No sequence remapping.
# No approximate matching.
# ============================================================

eligible = to_bool_series(
    df["sequence_exact_match"]
)

out = pd.DataFrame(
    {
        "uniprot_id":
            df["uniprot_id"],

        "structure_eligible":
            eligible,
    }
)


# ============================================================
# OUTPUT PATHS
# ============================================================

output_csv = (
    OUTPUT_ROOT
    / "test_structure_eligibility.csv"
)

eligible_path = (
    OUTPUT_ROOT
    / "test_structure_eligible_ids.txt"
)

ineligible_path = (
    OUTPUT_ROOT
    / "test_structure_ineligible_ids.txt"
)

summary_path = (
    OUTPUT_ROOT
    / "test_structure_eligibility_summary.json"
)


# ============================================================
# SAVE
# ============================================================

out.to_csv(
    output_csv,
    index=False,
)

eligible_ids = (
    out.loc[
        out["structure_eligible"],
        "uniprot_id",
    ]
    .tolist()
)

ineligible_ids = (
    out.loc[
        ~out["structure_eligible"],
        "uniprot_id",
    ]
    .tolist()
)

eligible_path.write_text(
    "\n".join(eligible_ids) + "\n",
    encoding="utf-8",
)

ineligible_path.write_text(
    "\n".join(ineligible_ids) + "\n",
    encoding="utf-8",
)


# ============================================================
# STRICT COUNT CHECKS
# ============================================================

eligible_count = len(
    eligible_ids
)

ineligible_count = len(
    ineligible_ids
)

if (
    eligible_count
    + ineligible_count
    != EXPECTED_COUNT
):
    raise RuntimeError(
        "Eligible + ineligible counts do not equal "
        "the frozen TEST size."
    )

# Ensure membership has not changed.
if set(eligible_ids) | set(ineligible_ids) != set(test_ids):
    raise RuntimeError(
        "Eligibility output changed frozen TEST membership."
    )

if set(eligible_ids) & set(ineligible_ids):
    raise RuntimeError(
        "Protein appears in both eligible and "
        "ineligible TEST sets."
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "protocol":
        "structure_protocol_v1",

    "split":
        "test",

    "eligibility_rule":
        "Exact residue-level sequence correspondence "
        "between the frozen benchmark sequence and "
        "the AlphaFold structure-derived sequence.",

    "alignment_rescue":
        False,

    "sequence_remapping":
        False,

    "approximate_sequence_matching":
        False,

    "frozen_graph_v2_membership_changed":
        False,

    "test_used":
        True,

    "total":
        EXPECTED_COUNT,

    "eligible":
        eligible_count,

    "ineligible":
        ineligible_count,

    "eligible_fraction":
        eligible_count / EXPECTED_COUNT,

    "validation_sha256":
        sha256(VALIDATION_PATH),

    "eligibility_csv_sha256":
        sha256(output_csv),

    "eligible_ids_sha256":
        sha256(eligible_path),

    "ineligible_ids_sha256":
        sha256(ineligible_path),
}

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("STRUCTURE PROTOCOL V1 — TEST ELIGIBILITY")
print("=" * 80)

print(
    f"TEST: total={EXPECTED_COUNT:,}  "
    f"eligible={eligible_count:,}  "
    f"ineligible={ineligible_count:,}  "
    f"eligible%="
    f"{100 * eligible_count / EXPECTED_COUNT:.3f}"
)

print()
print("Rule: EXACT sequence match")
print("Alignment rescue: NO")
print("Sequence remapping: NO")
print("Approximate sequence matching: NO")
print("Frozen Graph-V2 membership changed: NO")
print("TEST USED: YES")

print()
print("Saved:")
print(output_csv)
print("SHA256:", sha256(output_csv))

print()
print(eligible_path)
print("SHA256:", sha256(eligible_path))

print()
print(ineligible_path)
print("SHA256:", sha256(ineligible_path))

print()
print(summary_path)
print("SHA256:", sha256(summary_path))

print("=" * 80)
