from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    required=True,
    choices=["train", "val", "test"],
)

args = parser.parse_args()

SPLIT = args.split


# ============================================================
# PATHS
# ============================================================

COHORT_FILE = Path(
    "data/processed/primary_mf_cohort.csv"
)

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2"
)

STRUCTURE_DIR = (
    Path("data/structures/alphafold_v6")
    / SPLIT
)

PDB_DIR = (
    STRUCTURE_DIR / "pdb"
)

OUTPUT_CSV = (
    STRUCTURE_DIR
    / "structure_validation.csv"
)

OUTPUT_JSON = (
    STRUCTURE_DIR
    / "structure_validation_summary.json"
)


# ============================================================
# EXPECTED SPLIT SIZE
# ============================================================

EXPECTED_COUNTS = {
    "train": 38031,
    "val": 4754,
    "test": 4754,
}

EXPECTED_COUNT = EXPECTED_COUNTS[
    SPLIT
]


# ============================================================
# AMINO-ACID MAPPING
# ============================================================

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

    # Selenocysteine
    "SEC": "U",
}


# ============================================================
# CA PARSER
#
# Keep behavior consistent with the already-tested
# smoke-test validator.
# ============================================================

def parse_ca_atoms(pdb_path):

    residues = []

    seen = set()

    with open(
        pdb_path,
        "r",
        errors="replace"
    ) as f:

        for line in f:

            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()

            if atom_name != "CA":
                continue

            altloc = line[16].strip()

            # Accept blank or primary alternate location.
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

            try:

                x = float(
                    line[30:38]
                )

                y = float(
                    line[38:46]
                )

                z = float(
                    line[46:54]
                )

                bfactor = float(
                    line[60:66]
                )

            except ValueError as e:

                raise RuntimeError(
                    f"Could not parse ATOM line "
                    f"in {pdb_path}:\n{line}"
                ) from e

            aa = AA3_TO_AA1.get(
                resname,
                "X"
            )

            residues.append(
                {
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "aa": aa,
                    "x": x,
                    "y": y,
                    "z": z,
                    "plddt": bfactor,
                }
            )

    return residues


# ============================================================
# LOAD FROZEN SPLIT IDS
# ============================================================
id_file = (
    SPLIT_DIR
    / "labels"
    / f"{SPLIT}_uniprot_ids.txt"
)

if not id_file.exists():

    raise FileNotFoundError(
        f"Frozen split ID file not found: {id_file}"
    )


split_ids = [
    line.strip()
    for line in id_file.read_text().splitlines()
    if line.strip()
]


if len(split_ids) != EXPECTED_COUNT:

    raise RuntimeError(
        f"{SPLIT}: expected "
        f"{EXPECTED_COUNT} frozen IDs, "
        f"found {len(split_ids)}."
    )


if len(split_ids) != len(set(split_ids)):

    raise RuntimeError(
        f"{SPLIT}: duplicate IDs detected "
        f"in frozen split."
    )


# ============================================================
# LOAD PRIMARY COHORT
# ============================================================

cohort = pd.read_csv(
    COHORT_FILE,
    usecols=[
        "uniprot_id",
        "sequence",
        "length",
        "global_metric",
    ],
)


if not cohort[
    "uniprot_id"
].is_unique:

    raise RuntimeError(
        "primary_mf_cohort.csv contains "
        "duplicate UniProt IDs."
    )


cohort = cohort.set_index(
    "uniprot_id"
)


missing_from_cohort = [
    uid
    for uid in split_ids
    if uid not in cohort.index
]


if missing_from_cohort:

    raise RuntimeError(
        f"{len(missing_from_cohort)} "
        f"{SPLIT} proteins are missing "
        f"from primary_mf_cohort.csv."
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)

print(
    f"ALPHAFOLD {SPLIT.upper()} — "
    f"FULL STRUCTURE / SEQUENCE VALIDATION"
)

print("=" * 80)

print(
    "\nFrozen split proteins:",
    len(split_ids)
)

print(
    "PDB directory:",
    PDB_DIR
)

print(
    "TEST SPLIT PROCESSING:",
    "YES" if SPLIT == "test" else "NO"
)


# ============================================================
# VALIDATE
# ============================================================

results = []

for i, uid in enumerate(
    split_ids,
    start=1
):

    expected_sequence = str(
        cohort.loc[
            uid,
            "sequence"
        ]
    ).strip()

    expected_length = int(
        cohort.loc[
            uid,
            "length"
        ]
    )

    global_metric = float(
        cohort.loc[
            uid,
            "global_metric"
        ]
    )

    pdb_path = (
        PDB_DIR
        / f"{uid}.pdb"
    )


    # --------------------------------------------------------
    # Missing file
    # --------------------------------------------------------

    if not pdb_path.exists():

        results.append(
            {
                "uniprot_id":
                    uid,

                "pdb_exists":
                    False,

                "expected_length":
                    expected_length,

                "ca_residues":
                    0,

                "length_match":
                    False,

                "sequence_exact_match":
                    False,

                "unknown_residues":
                    None,

                "finite_ca_coordinates":
                    False,

                "plddt_valid_0_100":
                    False,

                "mean_residue_plddt":
                    np.nan,

                "manifest_global_metric":
                    global_metric,

                "mean_global_abs_difference":
                    np.nan,

                "n_sequence_mismatches":
                    None,

                "parse_error":
                    "PDB file missing",
            }
        )

        continue


    # --------------------------------------------------------
    # Parse structure
    # --------------------------------------------------------

    try:

        residues = parse_ca_atoms(
            pdb_path
        )

    except Exception as e:

        results.append(
            {
                "uniprot_id":
                    uid,

                "pdb_exists":
                    True,

                "expected_length":
                    expected_length,

                "ca_residues":
                    0,

                "length_match":
                    False,

                "sequence_exact_match":
                    False,

                "unknown_residues":
                    None,

                "finite_ca_coordinates":
                    False,

                "plddt_valid_0_100":
                    False,

                "mean_residue_plddt":
                    np.nan,

                "manifest_global_metric":
                    global_metric,

                "mean_global_abs_difference":
                    np.nan,

                "n_sequence_mismatches":
                    None,

                "parse_error":
                    repr(e),
            }
        )

        continue


    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    observed_length = len(
        residues
    )

    observed_sequence = "".join(
        r["aa"]
        for r in residues
    )

    unknown_residues = int(
        observed_sequence.count("X")
    )


    length_match = (
        observed_length
        ==
        expected_length
    )

    sequence_exact = (
        observed_sequence
        ==
        expected_sequence
    )


    if (
        observed_length
        ==
        len(expected_sequence)
    ):

        mismatch_positions = [
            j + 1
            for j, (expected, observed)
            in enumerate(
                zip(
                    expected_sequence,
                    observed_sequence
                )
            )
            if expected != observed
        ]

    else:

        mismatch_positions = []


    # --------------------------------------------------------
    # pLDDT
    # --------------------------------------------------------

    plddt = np.array(
        [
            r["plddt"]
            for r in residues
        ],
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    coordinates = np.array(
        [
            [
                r["x"],
                r["y"],
                r["z"],
            ]
            for r in residues
        ],
        dtype=np.float64,
    )


    ca_finite = bool(
        len(residues) > 0
        and
        np.isfinite(
            coordinates
        ).all()
    )


    plddt_finite = bool(
        len(plddt) > 0
        and
        np.isfinite(
            plddt
        ).all()
    )


    plddt_range_valid = bool(
        plddt_finite
        and
        plddt.min() >= 0
        and
        plddt.max() <= 100
    )


    mean_plddt = (
        float(
            plddt.mean()
        )
        if len(plddt)
        else np.nan
    )


    min_plddt = (
        float(
            plddt.min()
        )
        if len(plddt)
        else np.nan
    )


    max_plddt = (
        float(
            plddt.max()
        )
        if len(plddt)
        else np.nan
    )


    global_difference = (
        abs(
            mean_plddt
            - global_metric
        )
        if np.isfinite(mean_plddt)
        else np.nan
    )


    # --------------------------------------------------------
    # Save protein-level result
    # --------------------------------------------------------

    results.append(
        {
            "uniprot_id":
                uid,

            "pdb_exists":
                True,

            "expected_length":
                expected_length,

            "ca_residues":
                observed_length,

            "length_match":
                length_match,

            "sequence_exact_match":
                sequence_exact,

            "unknown_residues":
                unknown_residues,

            "finite_ca_coordinates":
                ca_finite,

            "plddt_valid_0_100":
                plddt_range_valid,

            "min_residue_plddt":
                min_plddt,

            "max_residue_plddt":
                max_plddt,

            "mean_residue_plddt":
                mean_plddt,

            "manifest_global_metric":
                global_metric,

            "mean_global_abs_difference":
                global_difference,

            "n_sequence_mismatches":
                (
                    len(mismatch_positions)
                    if length_match
                    else None
                ),

            "parse_error":
                None,
        }
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        i % 250 == 0
        or
        i == len(split_ids)
    ):

        print(
            f"[{i:>6}/{len(split_ids)}] "
            f"validated"
        )


# ============================================================
# DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# CRITICAL QC COUNTS
# ============================================================

n_missing = int(
    (~result_df["pdb_exists"]).sum()
)

n_length_fail = int(
    (~result_df["length_match"]).sum()
)

n_sequence_fail = int(
    (~result_df[
        "sequence_exact_match"
    ]).sum()
)

n_coordinate_fail = int(
    (~result_df[
        "finite_ca_coordinates"
    ]).sum()
)

n_plddt_fail = int(
    (~result_df[
        "plddt_valid_0_100"
    ]).sum()
)

n_unknown = int(
    (
        result_df[
            "unknown_residues"
        ].fillna(0) > 0
    ).sum()
)

n_parse_errors = int(
    result_df[
        "parse_error"
    ].notna().sum()
)


# ============================================================
# GLOBAL-METRIC DIAGNOSTIC
#
# Diagnostic only.
# NOT a critical pass/fail criterion.
# ============================================================

valid_global_diff = (
    result_df[
        "mean_global_abs_difference"
    ].dropna()
)

max_global_diff = (
    float(
        valid_global_diff.max()
    )
    if len(valid_global_diff)
    else None
)


# ============================================================
# STRICT OVERALL PASS
# ============================================================

overall_pass = bool(
    n_missing == 0
    and
    n_length_fail == 0
    and
    n_sequence_fail == 0
    and
    n_coordinate_fail == 0
    and
    n_plddt_fail == 0
)


# ============================================================
# SAVE CSV
# ============================================================

result_df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "protocol":
        "structure_v1_pre_graph_qc",

    "split":
        SPLIT,

    "expected_proteins":
        EXPECTED_COUNT,

    "proteins_checked":
        len(result_df),

    "missing_pdb_files":
        n_missing,

    "parse_errors":
        n_parse_errors,

    "length_failures":
        n_length_fail,

    "sequence_failures":
        n_sequence_fail,

    "coordinate_failures":
        n_coordinate_fail,

    "plddt_failures":
        n_plddt_fail,

    "proteins_with_unknown_residues":
        n_unknown,

    "max_mean_global_abs_difference":
        max_global_diff,

    "global_metric_comparison_is_diagnostic_only":
        True,

    "overall_pass":
        overall_pass,

    "test_used":
        bool(SPLIT == "test"),
}


with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)

print(
    f"{SPLIT.upper()} STRUCTURE QC SUMMARY"
)

print("=" * 80)


print(
    "Expected proteins:",
    EXPECTED_COUNT
)

print(
    "Proteins checked:",
    len(result_df)
)

print(
    "Missing PDB files:",
    n_missing
)

print(
    "Parse errors:",
    n_parse_errors
)

print(
    "Residue-count failures:",
    n_length_fail
)

print(
    "Exact-sequence failures:",
    n_sequence_fail
)

print(
    "Coordinate failures:",
    n_coordinate_fail
)

print(
    "pLDDT failures:",
    n_plddt_fail
)

print(
    "Proteins with unknown residues:",
    n_unknown
)

print(
    "Maximum |mean residue pLDDT "
    "- manifest global metric|:",
    (
        f"{max_global_diff:.4f}"
        if max_global_diff is not None
        else "NA"
    )
)

print(
    "\nOverall:",
    "PASS"
    if overall_pass
    else "FAIL"
)

print("\nTEST USED: NO")

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_JSON)

print("=" * 80)


# ============================================================
# STRICT FAILURE
# ============================================================

if not overall_pass:

    failed = result_df[
        (~result_df["pdb_exists"])
        |
        (~result_df["length_match"])
        |
        (~result_df["sequence_exact_match"])
        |
        (~result_df["finite_ca_coordinates"])
        |
        (~result_df["plddt_valid_0_100"])
    ]

    failure_file = (
        STRUCTURE_DIR
        / "structure_validation_failures.csv"
    )

    failed.to_csv(
        failure_file,
        index=False,
    )

    print(
        "\nFailure details saved:",
        failure_file
    )

    raise RuntimeError(
        f"{SPLIT.upper()} structural QC failed "
        f"for {len(failed)} proteins."
    )