from pathlib import Path
import json
import hashlib

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

STRUCTURE_ROOT = Path(
    "data/structures/alphafold_v6"
)

OUTPUT_ROOT = Path(
    "data/structures/structure_protocol_v1"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

EXPECTED_COUNTS = {
    "train": 38031,
    "val": 4754,
}


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    """
    Calculate SHA256 checksum of a file.
    """

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def to_bool_series(series):
    """
    Convert a pandas Series safely to boolean.

    Handles:
        True / False
        "True" / "False"
        "true" / "false"
        1 / 0

    Raises an error for unexpected values.
    """

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
# BUILD ELIGIBILITY MASKS
# ============================================================

summary = {
    "protocol": "structure_protocol_v1",

    "eligibility_rule":
        "Exact residue-level sequence correspondence "
        "between the frozen benchmark sequence and "
        "the AlphaFold structure-derived sequence.",

    "alignment_rescue": False,

    "sequence_remapping": False,

    "frozen_graph_v2_membership_changed": False,

    "test_used": False,

    "splits": {},
}


for split in ["train", "val"]:

    validation_path = (
        STRUCTURE_ROOT
        / split
        / "structure_validation.csv"
    )

    # --------------------------------------------------------
    # Check validation file exists
    # --------------------------------------------------------

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Missing validation file: "
            f"{validation_path}"
        )

    # --------------------------------------------------------
    # Load validation results
    # --------------------------------------------------------

    df = pd.read_csv(
        validation_path
    )

    print(
        f"\n{split.upper()} columns:"
    )

    print(
        list(df.columns)
    )

    # --------------------------------------------------------
    # Check expected number of proteins
    # --------------------------------------------------------

    expected_count = (
        EXPECTED_COUNTS[split]
    )

    if len(df) != expected_count:
        raise RuntimeError(
            f"{split}: expected "
            f"{expected_count} rows, "
            f"but found {len(df)}"
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "uniprot_id",
        "sequence_exact_match",
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"{split}: missing required "
                f"column '{col}'."
            )

    # --------------------------------------------------------
    # Check UniProt IDs
    # --------------------------------------------------------

    df["uniprot_id"] = (
        df["uniprot_id"]
        .astype(str)
        .str.strip()
    )

    if df["uniprot_id"].duplicated().any():

        duplicates = (
            df.loc[
                df["uniprot_id"].duplicated(
                    keep=False
                ),
                "uniprot_id",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"{split}: duplicate UniProt IDs "
            f"found: {duplicates[:20]}"
        )

    if (
        df["uniprot_id"]
        .eq("")
        .any()
    ):
        raise RuntimeError(
            f"{split}: empty UniProt ID found."
        )

    # --------------------------------------------------------
    # Strict structure eligibility
    #
    # A protein is structure-eligible ONLY when the sequence
    # extracted from its AlphaFold PDB exactly matches the
    # frozen benchmark sequence.
    #
    # No alignment rescue.
    # No sequence remapping.
    # No approximate matching.
    # --------------------------------------------------------

    eligible = to_bool_series(
        df["sequence_exact_match"]
    )

    # --------------------------------------------------------
    # Create eligibility table
    # --------------------------------------------------------

    out = pd.DataFrame(
        {
            "uniprot_id":
                df["uniprot_id"],

            "structure_eligible":
                eligible,
        }
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if len(out) != expected_count:
        raise RuntimeError(
            f"{split}: eligibility output "
            "row count changed unexpectedly."
        )

    if not out["uniprot_id"].is_unique:
        raise RuntimeError(
            f"{split}: eligibility output "
            "contains duplicate IDs."
        )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_ROOT
        / f"{split}_structure_eligibility.csv"
    )

    eligible_path = (
        OUTPUT_ROOT
        / f"{split}_structure_eligible_ids.txt"
    )

    ineligible_path = (
        OUTPUT_ROOT
        / f"{split}_structure_ineligible_ids.txt"
    )

    # --------------------------------------------------------
    # Save full eligibility table
    # --------------------------------------------------------

    out.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Extract eligible/ineligible IDs
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save ID lists
    # --------------------------------------------------------

    eligible_path.write_text(
        (
            "\n".join(
                eligible_ids
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    ineligible_path.write_text(
        (
            "\n".join(
                ineligible_ids
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Verify counts
    # --------------------------------------------------------

    eligible_count = len(
        eligible_ids
    )

    ineligible_count = len(
        ineligible_ids
    )

    if (
        eligible_count
        + ineligible_count
        != expected_count
    ):
        raise RuntimeError(
            f"{split}: eligible + ineligible "
            "counts do not equal total."
        )

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    summary["splits"][split] = {

        "total":
            int(expected_count),

        "eligible":
            int(eligible_count),

        "ineligible":
            int(ineligible_count),

        "eligible_fraction":
            float(
                eligible_count
                / expected_count
            ),

        "validation_sha256":
            sha256(
                validation_path
            ),

        "eligibility_csv_sha256":
            sha256(
                output_csv
            ),

        "eligible_ids_sha256":
            sha256(
                eligible_path
            ),

        "ineligible_ids_sha256":
            sha256(
                ineligible_path
            ),
    }


# ============================================================
# CROSS-SPLIT CHECK
# ============================================================

train_eligible = set(
    (
        OUTPUT_ROOT
        / "train_structure_eligible_ids.txt"
    )
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)

val_eligible = set(
    (
        OUTPUT_ROOT
        / "val_structure_eligible_ids.txt"
    )
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)

train_ineligible = set(
    (
        OUTPUT_ROOT
        / "train_structure_ineligible_ids.txt"
    )
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)

val_ineligible = set(
    (
        OUTPUT_ROOT
        / "val_structure_ineligible_ids.txt"
    )
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)


# There should be no TRAIN/VAL overlap.

train_all = (
    train_eligible
    | train_ineligible
)

val_all = (
    val_eligible
    | val_ineligible
)

cross_split_overlap = (
    train_all
    & val_all
)

if cross_split_overlap:
    raise RuntimeError(
        "TRAIN/VAL overlap detected in "
        "structure eligibility files: "
        f"{sorted(cross_split_overlap)[:20]}"
    )


# ============================================================
# DEVELOPMENT TOTAL
# ============================================================

total = sum(
    x["total"]
    for x
    in summary["splits"].values()
)

eligible_total = sum(
    x["eligible"]
    for x
    in summary["splits"].values()
)

ineligible_total = sum(
    x["ineligible"]
    for x
    in summary["splits"].values()
)


summary["development_total"] = {

    "total":
        int(total),

    "eligible":
        int(eligible_total),

    "ineligible":
        int(ineligible_total),

    "eligible_fraction":
        float(
            eligible_total
            / total
        ),
}


summary[
    "train_val_overlap_count"
] = int(
    len(cross_split_overlap)
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    OUTPUT_ROOT
    / "structure_eligibility_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STRUCTURE PROTOCOL V1 — ELIGIBILITY"
)

print(
    "=" * 80
)


for split in [
    "train",
    "val",
]:

    x = (
        summary["splits"][split]
    )

    print(
        f"{split.upper():5s}: "
        f"total={x['total']:,}  "
        f"eligible={x['eligible']:,}  "
        f"ineligible={x['ineligible']:,}  "
        f"eligible%="
        f"{100 * x['eligible_fraction']:.3f}"
    )


print(
    "\nDevelopment total:"
)

print(
    f"total={total:,}"
)

print(
    f"eligible={eligible_total:,}"
)

print(
    f"ineligible={ineligible_total:,}"
)

print(
    "eligible%="
    f"{100 * eligible_total / total:.3f}"
)

print(
    "\nTRAIN/VAL overlap: "
    f"{len(cross_split_overlap)}"
)

print(
    "\nRule: EXACT sequence match"
)

print(
    "Alignment rescue: NO"
)

print(
    "Sequence remapping: NO"
)

print(
    "Approximate sequence matching: NO"
)

print(
    "Frozen Graph-V2 membership changed: NO"
)

print(
    "TEST USED: NO"
)

print(
    "\nSaved:"
)

print(
    OUTPUT_ROOT
    / "train_structure_eligibility.csv"
)

print(
    OUTPUT_ROOT
    / "train_structure_eligible_ids.txt"
)

print(
    OUTPUT_ROOT
    / "train_structure_ineligible_ids.txt"
)

print(
    OUTPUT_ROOT
    / "val_structure_eligibility.csv"
)

print(
    OUTPUT_ROOT
    / "val_structure_eligible_ids.txt"
)

print(
    OUTPUT_ROOT
    / "val_structure_ineligible_ids.txt"
)

print(
    summary_path
)

print(
    "=" * 80
)