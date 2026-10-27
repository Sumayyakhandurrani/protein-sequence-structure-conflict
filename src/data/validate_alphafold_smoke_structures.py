from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

COHORT_FILE = Path(
    "data/processed/primary_mf_cohort.csv"
)

SMOKE_DIR = Path(
    "data/structures/smoke_test"
)

SELECTION_FILE = (
    SMOKE_DIR / "smoke_test_selection.csv"
)

PDB_DIR = (
    SMOKE_DIR / "pdb"
)


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

    # Common special residue
    "SEC": "U",
}


# ============================================================
# PDB PARSER
#
# We intentionally parse only CA atoms.
# This avoids needing Biopython at this stage.
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
                icode
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
# LOAD DATA
# ============================================================

print("=" * 80)
print("ALPHAFOLD SMOKE TEST — STRUCTURE / SEQUENCE VALIDATION")
print("=" * 80)

selection = pd.read_csv(
    SELECTION_FILE
)

cohort = pd.read_csv(
    COHORT_FILE,
    usecols=[
        "uniprot_id",
        "sequence",
        "length",
        "global_metric",
    ],
)

assert cohort[
    "uniprot_id"
].is_unique

cohort = cohort.set_index(
    "uniprot_id"
)

assert len(selection) == 12

print(
    "\nSmoke-test proteins:",
    len(selection)
)

print("TEST USED: NO")


# ============================================================
# VALIDATE EACH STRUCTURE
# ============================================================

results = []

all_exact = True
all_lengths = True
all_ca = True
all_plddt_valid = True


for _, row in selection.iterrows():

    uid = row["uniprot_id"]

    if uid not in cohort.index:

        raise RuntimeError(
            f"{uid} missing from primary cohort"
        )

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
        PDB_DIR / f"{uid}.pdb"
    )

    if not pdb_path.exists():

        raise FileNotFoundError(
            pdb_path
        )

    residues = parse_ca_atoms(
        pdb_path
    )

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
            i + 1
            for i, (a, b)
            in enumerate(
                zip(
                    expected_sequence,
                    observed_sequence
                )
            )
            if a != b
        ]

    else:

        mismatch_positions = []


    plddt = np.array(
        [
            r["plddt"]
            for r in residues
        ],
        dtype=np.float64
    )


    coordinates = np.array(
        [
            [
                r["x"],
                r["y"],
                r["z"],
            ]
            for r in residues
        ],
        dtype=np.float64
    )


    ca_finite = bool(
        np.isfinite(
            coordinates
        ).all()
    )

    plddt_finite = bool(
        np.isfinite(
            plddt
        ).all()
    )

    plddt_range_valid = bool(
        plddt_finite
        and
        len(plddt) > 0
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


    print("\n" + "-" * 80)

    print(
        f"{uid}"
        f" | expected={expected_length}"
        f" | CA={observed_length}"
    )

    print(
        "Length match:",
        "PASS"
        if length_match
        else "FAIL"
    )

    print(
        "Exact sequence:",
        "PASS"
        if sequence_exact
        else "FAIL"
    )

    print(
        "Unknown residues:",
        unknown_residues
    )

    print(
        "Finite CA coordinates:",
        "PASS"
        if ca_finite
        else "FAIL"
    )

    print(
        "pLDDT range:",
        (
            f"{min_plddt:.2f}"
            f" – {max_plddt:.2f}"
        )
    )

    print(
        "Mean residue pLDDT:",
        f"{mean_plddt:.4f}"
    )

    print(
        "Manifest global metric:",
        f"{global_metric:.4f}"
    )

    print(
        "Absolute mean/global difference:",
        f"{global_difference:.4f}"
    )

    if not sequence_exact:

        if mismatch_positions:

            print(
                "First mismatch positions:",
                mismatch_positions[:20]
            )

        else:

            print(
                "Sequence lengths differ; "
                "position-wise comparison skipped."
            )


    all_lengths &= length_match
    all_exact &= sequence_exact
    all_ca &= ca_finite
    all_plddt_valid &= plddt_range_valid


    results.append(
        {
            "uniprot_id":
                uid,

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
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(
    results
)

result_path = (
    SMOKE_DIR
    / "structure_validation.csv"
)

result_df.to_csv(
    result_path,
    index=False
)


summary = {

    "protocol":
        "structure_v1",

    "proteins_checked":
        len(result_df),

    "all_length_match":
        bool(all_lengths),

    "all_exact_sequence_match":
        bool(all_exact),

    "all_ca_coordinates_finite":
        bool(all_ca),

    "all_plddt_valid":
        bool(all_plddt_valid),

    "max_mean_global_abs_difference":
        float(
            result_df[
                "mean_global_abs_difference"
            ].max()
        ),

    "test_used":
        False,
}


summary_path = (
    SMOKE_DIR
    / "structure_validation_summary.json"
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


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(
    "Proteins checked:",
    len(result_df)
)

print(
    "All lengths exact:",
    "PASS"
    if all_lengths
    else "FAIL"
)

print(
    "All sequences exact:",
    "PASS"
    if all_exact
    else "FAIL"
)

print(
    "All CA coordinates finite:",
    "PASS"
    if all_ca
    else "FAIL"
)

print(
    "All pLDDT values in [0,100]:",
    "PASS"
    if all_plddt_valid
    else "FAIL"
)

print(
    "Maximum |mean residue pLDDT "
    "- manifest global metric|:",
    f"{summary['max_mean_global_abs_difference']:.4f}"
)

print("\nTEST USED: NO")

print("\nSaved:")
print(result_path)
print(summary_path)

print("=" * 80)


# ============================================================
# STRICT FAILURE
# ============================================================

if not all_lengths:
    raise RuntimeError(
        "At least one structure has a residue-count mismatch."
    )

if not all_exact:
    raise RuntimeError(
        "At least one structure does not exactly match "
        "the benchmark sequence."
    )

if not all_ca:
    raise RuntimeError(
        "Non-finite CA coordinates detected."
    )

if not all_plddt_valid:
    raise RuntimeError(
        "Invalid pLDDT values detected."
    )