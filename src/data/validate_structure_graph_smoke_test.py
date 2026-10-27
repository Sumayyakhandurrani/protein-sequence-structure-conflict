from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

GRAPH_DIR = Path(
    "data/structures/structure_protocol_v1/graph_smoke_test"
)

SUMMARY_CSV = (
    GRAPH_DIR
    / "graph_smoke_test_summary.csv"
)

OUTPUT_JSON = (
    GRAPH_DIR
    / "graph_smoke_test_validation.json"
)

SPATIAL_CUTOFF = 10.0
DISTANCE_TOLERANCE = 1e-4


# ============================================================
# LOAD SMOKE-TEST SUMMARY
# ============================================================

if not SUMMARY_CSV.exists():
    raise FileNotFoundError(SUMMARY_CSV)

summary = pd.read_csv(SUMMARY_CSV)

if len(summary) != 6:
    raise RuntimeError(
        f"Expected 6 smoke-test graphs, found {len(summary)}"
    )


# ============================================================
# VALIDATE ONE GRAPH
# ============================================================

def validate_graph(uid, expected_length):

    path = GRAPH_DIR / f"{uid}.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:

        required = {
            "x",
            "coords",
            "edge_index",
            "edge_attr",
            "plddt",
            "uniprot_id",
            "sequence",
        }

        missing = required - set(data.files)

        if missing:
            raise RuntimeError(
                f"{uid}: missing arrays: {sorted(missing)}"
            )

        x = data["x"]
        coords = data["coords"]
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        plddt = data["plddt"]
        saved_uid = str(data["uniprot_id"])
        sequence = str(data["sequence"])


    # --------------------------------------------------------
    # Basic shapes
    # --------------------------------------------------------

    n = expected_length

    if saved_uid != uid:
        raise RuntimeError(
            f"{uid}: stored UniProt ID is {saved_uid}"
        )

    if len(sequence) != n:
        raise RuntimeError(
            f"{uid}: sequence length mismatch"
        )

    if x.shape != (n, 21):
        raise RuntimeError(
            f"{uid}: invalid x shape {x.shape}"
        )

    if coords.shape != (n, 3):
        raise RuntimeError(
            f"{uid}: invalid coords shape {coords.shape}"
        )

    if plddt.shape != (n,):
        raise RuntimeError(
            f"{uid}: invalid pLDDT shape {plddt.shape}"
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError(
            f"{uid}: invalid edge_index shape {edge_index.shape}"
        )

    e = edge_index.shape[1]

    if edge_attr.shape != (e, 3):
        raise RuntimeError(
            f"{uid}: invalid edge_attr shape {edge_attr.shape}"
        )


    # --------------------------------------------------------
    # Dtypes
    # --------------------------------------------------------

    if x.dtype != np.float32:
        raise RuntimeError(
            f"{uid}: x dtype is {x.dtype}"
        )

    if coords.dtype != np.float32:
        raise RuntimeError(
            f"{uid}: coords dtype is {coords.dtype}"
        )

    if edge_index.dtype != np.int64:
        raise RuntimeError(
            f"{uid}: edge_index dtype is {edge_index.dtype}"
        )

    if edge_attr.dtype != np.float32:
        raise RuntimeError(
            f"{uid}: edge_attr dtype is {edge_attr.dtype}"
        )

    if plddt.dtype != np.float32:
        raise RuntimeError(
            f"{uid}: pLDDT dtype is {plddt.dtype}"
        )


    # --------------------------------------------------------
    # Finite values
    # --------------------------------------------------------

    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{uid}: non-finite x"
        )

    if not np.isfinite(coords).all():
        raise RuntimeError(
            f"{uid}: non-finite coordinates"
        )

    if not np.isfinite(edge_attr).all():
        raise RuntimeError(
            f"{uid}: non-finite edge attributes"
        )

    if not np.isfinite(plddt).all():
        raise RuntimeError(
            f"{uid}: non-finite pLDDT"
        )


    # --------------------------------------------------------
    # One-hot node features
    # --------------------------------------------------------

    if not np.all(
        (x == 0.0) | (x == 1.0)
    ):
        raise RuntimeError(
            f"{uid}: x is not binary one-hot"
        )

    if not np.allclose(
        x.sum(axis=1),
        1.0,
    ):
        raise RuntimeError(
            f"{uid}: one-hot rows do not sum to 1"
        )


    # --------------------------------------------------------
    # Edge index validity
    # --------------------------------------------------------

    src = edge_index[0]
    dst = edge_index[1]

    if (
        (src < 0).any()
        or (dst < 0).any()
        or (src >= n).any()
        or (dst >= n).any()
    ):
        raise RuntimeError(
            f"{uid}: edge index outside residue range"
        )

    if np.any(src == dst):
        raise RuntimeError(
            f"{uid}: self-edge detected"
        )


    # --------------------------------------------------------
    # Duplicate directed edges
    # --------------------------------------------------------

    directed_pairs = list(
        zip(
            src.tolist(),
            dst.tolist(),
        )
    )

    if len(directed_pairs) != len(set(directed_pairs)):
        raise RuntimeError(
            f"{uid}: duplicate directed edges detected"
        )


    # --------------------------------------------------------
    # Symmetry
    # --------------------------------------------------------

    edge_set = set(directed_pairs)

    missing_reverse = 0

    for i, j in edge_set:

        if (j, i) not in edge_set:
            missing_reverse += 1

    if missing_reverse != 0:
        raise RuntimeError(
            f"{uid}: {missing_reverse} edges "
            "lack reverse counterparts"
        )


    # --------------------------------------------------------
    # Validate edge features
    #
    # edge_attr:
    # 0 = C-alpha distance
    # 1 = sequential indicator
    # 2 = absolute sequence separation
    # --------------------------------------------------------

    stored_distance = edge_attr[:, 0]
    sequential_indicator = edge_attr[:, 1]
    sequence_separation = edge_attr[:, 2]

    calculated_distance = np.linalg.norm(
        coords[src] - coords[dst],
        axis=1,
    )

    max_distance_error = float(
        np.max(
            np.abs(
                calculated_distance
                - stored_distance
            )
        )
    )

    if max_distance_error > DISTANCE_TOLERANCE:
        raise RuntimeError(
            f"{uid}: edge distance mismatch; "
            f"max error={max_distance_error}"
        )


    expected_separation = np.abs(
        src - dst
    ).astype(np.float32)

    if not np.array_equal(
        sequence_separation,
        expected_separation,
    ):
        raise RuntimeError(
            f"{uid}: sequence-separation feature mismatch"
        )


    expected_sequential = (
        expected_separation == 1
    ).astype(np.float32)

    if not np.array_equal(
        sequential_indicator,
        expected_sequential,
    ):
        raise RuntimeError(
            f"{uid}: sequential indicator mismatch"
        )


    # --------------------------------------------------------
    # Every sequential residue pair must exist
    # --------------------------------------------------------

    missing_sequential = 0

    for i in range(n - 1):

        if (
            (i, i + 1) not in edge_set
            or
            (i + 1, i) not in edge_set
        ):
            missing_sequential += 1

    if missing_sequential != 0:
        raise RuntimeError(
            f"{uid}: {missing_sequential} "
            "sequential pairs missing"
        )


    # --------------------------------------------------------
    # Non-sequential edges MUST obey 10 Å cutoff.
    #
    # Sequential edges are explicitly retained even if,
    # due to unusual geometry, their distance exceeds 10 Å.
    # --------------------------------------------------------

    nonseq = (
        sequential_indicator == 0
    )

    nonseq_over_cutoff = int(
        np.sum(
            stored_distance[nonseq]
            >
            (
                SPATIAL_CUTOFF
                + DISTANCE_TOLERANCE
            )
        )
    )

    if nonseq_over_cutoff != 0:
        raise RuntimeError(
            f"{uid}: {nonseq_over_cutoff} "
            "non-sequential edges exceed 10 Å"
        )


    # --------------------------------------------------------
    # pLDDT metadata validation
    # --------------------------------------------------------

    if (
        (plddt < 0).any()
        or
        (plddt > 100).any()
    ):
        raise RuntimeError(
            f"{uid}: pLDDT outside [0,100]"
        )


    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    size_bytes = path.stat().st_size


    return {
        "uniprot_id":
            uid,

        "length":
            n,

        "directed_edges":
            e,

        "unique_undirected_pairs":
            e // 2,

        "missing_reverse_edges":
            missing_reverse,

        "missing_sequential_pairs":
            missing_sequential,

        "nonsequential_edges_over_10A":
            nonseq_over_cutoff,

        "max_edge_distance_error":
            max_distance_error,

        "file_size_bytes":
            size_bytes,

        "file_size_mb":
            size_bytes / (1024 ** 2),

        "pass":
            True,
    }


# ============================================================
# VALIDATE ALL SIX
# ============================================================

records = []

print("\n" + "=" * 90)
print(
    "STRUCTURE GRAPH SMOKE-TEST VALIDATION"
)
print("=" * 90)


for _, row in summary.iterrows():

    uid = str(
        row["uniprot_id"]
    )

    length = int(
        row["length"]
    )

    result = validate_graph(
        uid,
        length,
    )

    records.append(result)

    print(
        f"{uid:12s} "
        f"length={length:4d}  "
        f"edges={result['directed_edges']:6d}  "
        f"size={result['file_size_mb']:.3f} MB  "
        "PASS"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_size = sum(
    r["file_size_bytes"]
    for r in records
)

max_error = max(
    r["max_edge_distance_error"]
    for r in records
)


output = {
    "protocol":
        "structure_protocol_v1",

    "graphs_checked":
        len(records),

    "graphs_passed":
        sum(
            int(r["pass"])
            for r in records
        ),

    "all_passed":
        all(
            r["pass"]
            for r in records
        ),

    "max_edge_distance_error":
        max_error,

    "total_smoke_graph_size_bytes":
        total_size,

    "test_used":
        False,

    "records":
        records,
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


print("\n" + "=" * 90)
print("VALIDATION SUMMARY")
print("=" * 90)

print(
    f"Graphs checked: {len(records)}"
)

print(
    f"Graphs passed: "
    f"{sum(int(r['pass']) for r in records)}"
)

print(
    "Edge symmetry: PASS"
)

print(
    "Sequential-edge coverage: PASS"
)

print(
    "10 Å non-sequential cutoff: PASS"
)

print(
    "One-hot node features: PASS"
)

print(
    "Edge features: PASS"
)

print(
    "pLDDT metadata: PASS"
)

print(
    f"Maximum distance error: "
    f"{max_error:.8f} Å"
)

print(
    "TEST USED: NO"
)

print(
    f"\nSaved:\n{OUTPUT_JSON}"
)

print("=" * 90)