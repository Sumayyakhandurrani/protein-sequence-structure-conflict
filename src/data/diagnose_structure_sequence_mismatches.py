from pathlib import Path
import argparse
import difflib

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

STRUCTURE_ROOT = Path(
    "data/structures/alphafold_v6"
)

PRIMARY_COHORT = Path(
    "data/processed/primary_mf_cohort.csv"
)


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
}


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    required=True,
    choices=[
        "train",
        "val",
    ],
)

args = parser.parse_args()

split = args.split


# ============================================================
# PATHS
# ============================================================

split_dir = (
    STRUCTURE_ROOT
    / split
)

pdb_dir = (
    split_dir
    / "pdb"
)

failure_path = (
    split_dir
    / "structure_validation_failures.csv"
)

output_path = (
    split_dir
    / "sequence_mismatch_diagnostic.csv"
)


# ============================================================
# PDB SEQUENCE PARSER
# ============================================================

def extract_ca_sequence(pdb_path):

    residues = []
    seen = set()

    with open(
        pdb_path,
        "r",
        errors="replace",
    ) as f:

        for line in f:

            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()

            if atom_name != "CA":
                continue

            altloc = line[16].strip()

            if altloc not in ("", "A"):
                continue

            resname = line[17:20].strip()

            chain = line[21].strip()

            resseq = line[22:26].strip()

            icode = line[26].strip()

            key = (
                chain,
                resseq,
                icode,
            )

            if key in seen:
                continue

            seen.add(key)

            aa = AA3_TO_AA1.get(
                resname,
                "X",
            )

            residues.append(aa)

    return "".join(residues)


# ============================================================
# LOAD INPUTS
# ============================================================

if not failure_path.exists():

    raise FileNotFoundError(
        f"Missing failure file: {failure_path}"
    )


failures = pd.read_csv(
    failure_path
)

cohort = pd.read_csv(
    PRIMARY_COHORT,
    usecols=[
        "uniprot_id",
        "sequence",
        "length",
        "model_id",
        "pdb_url",
    ],
)


failure_ids = set(
    failures["uniprot_id"]
    .astype(str)
)


cohort = cohort[
    cohort["uniprot_id"]
    .astype(str)
    .isin(failure_ids)
].copy()


if len(cohort) != len(failure_ids):

    found = set(
        cohort["uniprot_id"]
        .astype(str)
    )

    missing = sorted(
        failure_ids - found
    )

    raise RuntimeError(
        "Some failure IDs were not found "
        f"in primary cohort: {missing}"
    )


# ============================================================
# CHARACTERIZE MISMATCHES
# ============================================================

records = []


for _, row in cohort.iterrows():

    uid = str(
        row["uniprot_id"]
    )

    expected_sequence = str(
        row["sequence"]
    ).strip()

    pdb_path = (
        pdb_dir
        / f"{uid}.pdb"
    )

    if not pdb_path.exists():

        raise FileNotFoundError(
            f"Missing PDB: {pdb_path}"
        )


    structure_sequence = (
        extract_ca_sequence(
            pdb_path
        )
    )


    matcher = difflib.SequenceMatcher(
        None,
        expected_sequence,
        structure_sequence,
        autojunk=False,
    )


    matching_residues = sum(
        block.size
        for block
        in matcher.get_matching_blocks()
    )


    expected_len = len(
        expected_sequence
    )

    structure_len = len(
        structure_sequence
    )


    # --------------------------------------------------------
    # First naive position where strings differ.
    #
    # This is diagnostic only. It is NOT a biological
    # sequence alignment.
    # --------------------------------------------------------

    first_difference = None

    min_len = min(
        expected_len,
        structure_len,
    )

    for i in range(min_len):

        if (
            expected_sequence[i]
            !=
            structure_sequence[i]
        ):

            first_difference = i + 1
            break


    if (
        first_difference is None
        and
        expected_len != structure_len
    ):

        first_difference = (
            min_len + 1
        )


    records.append(
        {
            "uniprot_id":
                uid,

            "model_id":
                row["model_id"],

            "expected_len":
                expected_len,

            "structure_len":
                structure_len,

            "length_delta":
                structure_len
                - expected_len,

            "matching_residues":
                matching_residues,

            "expected_match_fraction":
                (
                    matching_residues
                    / expected_len
                    if expected_len
                    else np.nan
                ),

            "structure_match_fraction":
                (
                    matching_residues
                    / structure_len
                    if structure_len
                    else np.nan
                ),

            "sequence_matcher_ratio":
                matcher.ratio(),

            "first_naive_difference_position":
                first_difference,
        }
    )


result = pd.DataFrame(
    records
).sort_values(
    "uniprot_id"
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    output_path,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 110)

print(
    f"{split.upper()} SEQUENCE ↔ "
    "ALPHAFOLD MISMATCH CHARACTERIZATION"
)

print("=" * 110)


print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 110)

print(
    "Mismatch proteins:",
    len(result)
)

print(
    "Same-length mismatches:",
    int(
        (
            result["length_delta"]
            == 0
        ).sum()
    )
)

print(
    "Different-length mismatches:",
    int(
        (
            result["length_delta"]
            != 0
        ).sum()
    )
)

print(
    "Benchmark sequence completely "
    "contained according to diagnostic:",
    int(
        (
            result[
                "expected_match_fraction"
            ]
            == 1.0
        ).sum()
    )
)

print(
    "Structure sequence completely "
    "contained according to diagnostic:",
    int(
        (
            result[
                "structure_match_fraction"
            ]
            == 1.0
        ).sum()
    )
)

print(
    "Median SequenceMatcher ratio:",
    f"{result['sequence_matcher_ratio'].median():.4f}"
)

print(
    "Minimum SequenceMatcher ratio:",
    f"{result['sequence_matcher_ratio'].min():.4f}"
)

print("\nIMPORTANT:")
print(
    "difflib.SequenceMatcher is used only "
    "for mismatch characterization."
)

print(
    "It is NOT treated as a biological "
    "sequence alignment or residue-mapping method."
)

print("\nTEST USED: NO")

print("\nSaved:")
print(output_path)

print("=" * 110)