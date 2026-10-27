from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    required=True,
    choices=["train", "val"],
)

parser.add_argument(
    "--progress-every",
    type=int,
    default=500,
)

args = parser.parse_args()

SPLIT = args.split

EXPECTED = {
    "train": 37919,
    "val": 4742,
}

EXPECTED_COUNT = EXPECTED[SPLIT]


# ============================================================
# FROZEN PROTOCOL
# ============================================================

SPATIAL_CUTOFF = 10.0

# Saved coordinates/edge distances are float32.
DISTANCE_TOLERANCE = 1e-4


# ============================================================
# PATHS
# ============================================================

PROTOCOL_ROOT = Path(
    "data/structures/structure_protocol_v1"
)

GRAPH_DIR = (
    PROTOCOL_ROOT
    / "graphs"
    / SPLIT
)

ELIGIBLE_IDS_PATH = (
    PROTOCOL_ROOT
    / f"{SPLIT}_structure_eligible_ids.txt"
)

PRIMARY_COHORT_PATH = Path(
    "data/processed/primary_mf_cohort.csv"
)

AUDIT_CSV = (
    GRAPH_DIR
    / "graph_validation.csv"
)

FAILURES_CSV = (
    GRAPH_DIR
    / "graph_validation_failures.csv"
)

SUMMARY_JSON = (
    GRAPH_DIR
    / "graph_validation_summary.json"
)


# ============================================================
# LOAD EXPECTED DATA
# ============================================================

eligible_ids = [
    x.strip()
    for x in
    ELIGIBLE_IDS_PATH
    .read_text()
    .splitlines()
    if x.strip()
]

if len(eligible_ids) != EXPECTED_COUNT:

    raise RuntimeError(
        f"{SPLIT}: expected "
        f"{EXPECTED_COUNT:,} eligible IDs, "
        f"found {len(eligible_ids):,}"
    )


cohort = pd.read_csv(
    PRIMARY_COHORT_PATH,
    usecols=[
        "uniprot_id",
        "sequence",
        "length",
    ],
)

cohort["uniprot_id"] = (
    cohort["uniprot_id"]
    .astype(str)
)

cohort = cohort.set_index(
    "uniprot_id",
    drop=False,
)


missing_metadata = (
    set(eligible_ids)
    -
    set(cohort.index)
)

if missing_metadata:

    raise RuntimeError(
        f"{len(missing_metadata)} IDs "
        "missing from primary cohort."
    )


# ============================================================
# VALIDATE ONE GRAPH
# ============================================================

def validate_one(
    uid,
    expected_sequence,
    expected_length,
):

    graph_path = (
        GRAPH_DIR
        / f"{uid}.npz"
    )

    result = {
        "uniprot_id":
            uid,

        "expected_length":
            expected_length,

        "graph_exists":
            False,

        "load_ok":
            False,

        "shapes_ok":
            False,

        "dtypes_ok":
            False,

        "finite_ok":
            False,

        "sequence_ok":
            False,

        "onehot_ok":
            False,

        "edge_indices_ok":
            False,

        "no_self_edges":
            False,

        "no_duplicate_directed_edges":
            False,

        "edge_symmetry_ok":
            False,

        "sequential_coverage_ok":
            False,

        "edge_features_ok":
            False,

        "spatial_cutoff_ok":
            False,

        "plddt_ok":
            False,

        "directed_edges":
            0,

        "max_distance_error":
            np.nan,

        "file_size_bytes":
            0,

        "pass":
            False,

        "error":
            "",
    }


    if not graph_path.exists():

        result["error"] = (
            "Graph file missing"
        )

        return result


    result["graph_exists"] = True

    result["file_size_bytes"] = (
        graph_path.stat().st_size
    )


    try:

        with np.load(
            graph_path,
            allow_pickle=False,
        ) as data:

            required = {
                "x",
                "coords",
                "edge_index",
                "edge_attr",
                "plddt",
                "uniprot_id",
                "sequence",
            }

            if not required.issubset(
                set(data.files)
            ):

                missing = (
                    required
                    -
                    set(data.files)
                )

                raise RuntimeError(
                    "Missing arrays: "
                    + ",".join(
                        sorted(missing)
                    )
                )


            x = data["x"]
            coords = data["coords"]
            edge_index = data["edge_index"]
            edge_attr = data["edge_attr"]
            plddt = data["plddt"]

            saved_uid = str(
                data["uniprot_id"]
            )

            saved_sequence = str(
                data["sequence"]
            )


        result["load_ok"] = True


        # ----------------------------------------------------
        # Identity / sequence
        # ----------------------------------------------------

        if saved_uid != uid:

            raise RuntimeError(
                "Stored UniProt ID mismatch"
            )


        result["sequence_ok"] = (
            saved_sequence
            == expected_sequence
        )


        # ----------------------------------------------------
        # Shapes
        # ----------------------------------------------------

        n = expected_length

        shapes_ok = (
            x.shape == (n, 21)
            and
            coords.shape == (n, 3)
            and
            plddt.shape == (n,)
            and
            edge_index.ndim == 2
            and
            edge_index.shape[0] == 2
            and
            edge_attr.shape
            ==
            (
                edge_index.shape[1],
                3,
            )
        )

        result["shapes_ok"] = bool(
            shapes_ok
        )


        if not shapes_ok:

            raise RuntimeError(
                "Array shape validation failed"
            )


        # ----------------------------------------------------
        # Dtypes
        # ----------------------------------------------------

        dtypes_ok = (
            x.dtype == np.float32
            and
            coords.dtype == np.float32
            and
            edge_index.dtype == np.int64
            and
            edge_attr.dtype == np.float32
            and
            plddt.dtype == np.float32
        )

        result["dtypes_ok"] = bool(
            dtypes_ok
        )


        # ----------------------------------------------------
        # Finite values
        # ----------------------------------------------------

        finite_ok = (
            np.isfinite(x).all()
            and
            np.isfinite(coords).all()
            and
            np.isfinite(edge_attr).all()
            and
            np.isfinite(plddt).all()
        )

        result["finite_ok"] = bool(
            finite_ok
        )


        # ----------------------------------------------------
        # One-hot
        # ----------------------------------------------------

        onehot_ok = (
            np.all(
                (x == 0.0)
                |
                (x == 1.0)
            )
            and
            np.allclose(
                x.sum(axis=1),
                1.0,
            )
        )

        result["onehot_ok"] = bool(
            onehot_ok
        )


        # ----------------------------------------------------
        # Edge indices
        # ----------------------------------------------------

        src = edge_index[0]
        dst = edge_index[1]

        e = edge_index.shape[1]

        result["directed_edges"] = int(
            e
        )


        edge_indices_ok = (
            (src >= 0).all()
            and
            (dst >= 0).all()
            and
            (src < n).all()
            and
            (dst < n).all()
        )

        result["edge_indices_ok"] = bool(
            edge_indices_ok
        )


        no_self = bool(
            np.all(
                src != dst
            )
        )

        result["no_self_edges"] = (
            no_self
        )


        # ----------------------------------------------------
        # Directed edge set
        # ----------------------------------------------------

        edge_set = set(
            zip(
                src.tolist(),
                dst.tolist(),
            )
        )


        no_duplicates = (
            len(edge_set)
            == e
        )

        result[
            "no_duplicate_directed_edges"
        ] = bool(
            no_duplicates
        )


        # ----------------------------------------------------
        # Symmetry
        # ----------------------------------------------------

        symmetry_ok = all(
            (j, i) in edge_set
            for i, j in edge_set
        )

        result[
            "edge_symmetry_ok"
        ] = bool(
            symmetry_ok
        )


        # ----------------------------------------------------
        # Sequential coverage
        # ----------------------------------------------------

        sequential_ok = True

        for i in range(n - 1):

            if (
                (i, i + 1)
                not in edge_set
                or
                (i + 1, i)
                not in edge_set
            ):

                sequential_ok = False
                break


        result[
            "sequential_coverage_ok"
        ] = bool(
            sequential_ok
        )


        # ----------------------------------------------------
        # Edge feature reconstruction
        # ----------------------------------------------------

        stored_distance = (
            edge_attr[:, 0]
        )

        stored_sequential = (
            edge_attr[:, 1]
        )

        stored_separation = (
            edge_attr[:, 2]
        )


        calculated_distance = (
            np.linalg.norm(
                coords[src]
                -
                coords[dst],
                axis=1,
            )
        )


        distance_error = np.abs(
            calculated_distance
            -
            stored_distance
        )


        max_distance_error = float(
            distance_error.max()
            if len(distance_error)
            else 0.0
        )

        result[
            "max_distance_error"
        ] = max_distance_error


        expected_separation = (
            np.abs(
                src - dst
            )
            .astype(
                np.float32
            )
        )


        expected_sequential = (
            expected_separation
            == 1
        ).astype(
            np.float32
        )


        edge_features_ok = (
            max_distance_error
            <= DISTANCE_TOLERANCE
            and
            np.array_equal(
                stored_separation,
                expected_separation,
            )
            and
            np.array_equal(
                stored_sequential,
                expected_sequential,
            )
        )


        result[
            "edge_features_ok"
        ] = bool(
            edge_features_ok
        )


        # ----------------------------------------------------
        # Spatial cutoff
        #
        # Sequential edges are retained even if
        # their geometry is unusual.
        # ----------------------------------------------------

        non_sequential = (
            stored_sequential
            == 0
        )


        if np.any(
            non_sequential
        ):

            cutoff_ok = bool(
                np.all(
                    stored_distance[
                        non_sequential
                    ]
                    <=
                    (
                        SPATIAL_CUTOFF
                        +
                        DISTANCE_TOLERANCE
                    )
                )
            )

        else:

            cutoff_ok = True


        result[
            "spatial_cutoff_ok"
        ] = cutoff_ok


        # ----------------------------------------------------
        # pLDDT metadata
        # ----------------------------------------------------

        plddt_ok = bool(
            np.isfinite(
                plddt
            ).all()
            and
            (plddt >= 0).all()
            and
            (plddt <= 100).all()
        )


        result[
            "plddt_ok"
        ] = plddt_ok


        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        checks = [
            result["graph_exists"],
            result["load_ok"],
            result["shapes_ok"],
            result["dtypes_ok"],
            result["finite_ok"],
            result["sequence_ok"],
            result["onehot_ok"],
            result["edge_indices_ok"],
            result["no_self_edges"],
            result[
                "no_duplicate_directed_edges"
            ],
            result["edge_symmetry_ok"],
            result[
                "sequential_coverage_ok"
            ],
            result["edge_features_ok"],
            result["spatial_cutoff_ok"],
            result["plddt_ok"],
        ]


        result["pass"] = bool(
            all(checks)
        )


        if not result["pass"]:

            result["error"] = (
                "One or more integrity "
                "checks failed"
            )


    except Exception as exc:

        result["error"] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )


    return result


# ============================================================
# FULL AUDIT
# ============================================================

print("\n" + "=" * 90)

print(
    "STRUCTURE PROTOCOL V1 — "
    f"{SPLIT.upper()} GRAPH AUDIT"
)

print("=" * 90)

print(
    f"Expected graphs: "
    f"{EXPECTED_COUNT:,}"
)

print(
    "TEST USED: NO\n"
)


records = []

passed = 0
failed = 0

start = time.time()


for number, uid in enumerate(
    eligible_ids,
    start=1,
):

    row = cohort.loc[uid]

    result = validate_one(
        uid=uid,
        expected_sequence=str(
            row["sequence"]
        ),
        expected_length=int(
            row["length"]
        ),
    )

    records.append(
        result
    )


    if result["pass"]:
        passed += 1
    else:
        failed += 1


    if (
        number
        % args.progress_every
        == 0
        or
        number
        == EXPECTED_COUNT
    ):

        elapsed = (
            time.time()
            -
            start
        )

        rate = (
            number / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"[{number:,}/{EXPECTED_COUNT:,}] "
            f"passed={passed:,}  "
            f"failed={failed:,}  "
            f"rate={rate:.2f}/s"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    records
)

results.to_csv(
    AUDIT_CSV,
    index=False,
)


failures = results[
    ~results["pass"]
].copy()

failures.to_csv(
    FAILURES_CSV,
    index=False,
)


max_distance_error = float(
    results[
        "max_distance_error"
    ]
    .dropna()
    .max()
)


total_bytes = int(
    results[
        "file_size_bytes"
    ].sum()
)


elapsed = (
    time.time()
    -
    start
)


summary = {
    "protocol":
        "structure_protocol_v1",

    "split":
        SPLIT,

    "expected_graphs":
        EXPECTED_COUNT,

    "checked":
        len(results),

    "passed":
        passed,

    "failed":
        failed,

    "all_passed":
        failed == 0,

    "max_edge_distance_error":
        max_distance_error,

    "total_size_bytes":
        total_bytes,

    "total_size_gib":
        total_bytes
        / (1024 ** 3),

    "runtime_seconds":
        elapsed,

    "test_used":
        False,
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print("\n" + "=" * 90)

print(
    f"{SPLIT.upper()} GRAPH AUDIT SUMMARY"
)

print("=" * 90)

print(
    f"Expected: {EXPECTED_COUNT:,}"
)

print(
    f"Checked:  {len(results):,}"
)

print(
    f"Passed:   {passed:,}"
)

print(
    f"Failed:   {failed:,}"
)

print(
    f"Maximum edge-distance error: "
    f"{max_distance_error:.8f} Å"
)

print(
    f"Total graph size: "
    f"{summary['total_size_gib']:.2f} GiB"
)

print(
    f"Runtime: "
    f"{elapsed / 60:.1f} minutes"
)

print(
    "TEST USED: NO"
)

print("\nSaved:")

print(AUDIT_CSV)
print(FAILURES_CSV)
print(SUMMARY_JSON)

print("=" * 90)


if failed != 0:

    raise RuntimeError(
        f"{failed} graph(s) failed "
        "the full integrity audit."
    )