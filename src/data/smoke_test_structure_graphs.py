from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# FROZEN STRUCTURE PROTOCOL V1
# ============================================================

PDB_DIR = Path(
    "data/structures/alphafold_v6/train/pdb"
)

ELIGIBLE_IDS_PATH = Path(
    "data/structures/structure_protocol_v1/"
    "train_structure_eligible_ids.txt"
)

PRIMARY_COHORT_PATH = Path(
    "data/processed/primary_mf_cohort.csv"
)

OUTPUT_DIR = Path(
    "data/structures/structure_protocol_v1/"
    "graph_smoke_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SPATIAL_CUTOFF = 10.0


# Frozen 21-AA ordering
AA_VOCAB = [
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
    "U",
]

AA_TO_INDEX = {
    aa: i
    for i, aa in enumerate(AA_VOCAB)
}


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
# PDB PARSER
# ============================================================

def parse_pdb(pdb_path):

    residues = []
    coords = []
    plddt = []

    seen = set()

    with open(
        pdb_path,
        "r",
        errors="replace",
    ) as f:

        for line in f:

            if not line.startswith("ATOM"):
                continue

            atom_name = (
                line[12:16].strip()
            )

            if atom_name != "CA":
                continue

            altloc = (
                line[16].strip()
            )

            if altloc not in ("", "A"):
                continue

            resname = (
                line[17:20].strip()
            )

            chain = (
                line[21].strip()
            )

            resseq = (
                line[22:26].strip()
            )

            icode = (
                line[26].strip()
            )

            key = (
                chain,
                resseq,
                icode,
            )

            if key in seen:
                continue

            seen.add(key)

            aa = AA3_TO_AA1.get(
                resname
            )

            if aa is None:
                raise RuntimeError(
                    f"{pdb_path.name}: "
                    f"unsupported residue "
                    f"{resname}"
                )

            x = float(
                line[30:38]
            )

            y = float(
                line[38:46]
            )

            z = float(
                line[46:54]
            )

            confidence = float(
                line[60:66]
            )

            residues.append(aa)

            coords.append(
                [x, y, z]
            )

            plddt.append(
                confidence
            )

    return (
        "".join(residues),

        np.asarray(
            coords,
            dtype=np.float32,
        ),

        np.asarray(
            plddt,
            dtype=np.float32,
        ),
    )


# ============================================================
# ONE-HOT NODE FEATURES
# ============================================================

def make_node_features(sequence):

    n = len(sequence)

    x = np.zeros(
        (n, 21),
        dtype=np.float32,
    )

    for i, aa in enumerate(sequence):

        if aa not in AA_TO_INDEX:
            raise RuntimeError(
                f"Unsupported amino acid: {aa}"
            )

        x[
            i,
            AA_TO_INDEX[aa],
        ] = 1.0

    return x


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph(
    sequence,
    coords,
):

    n = len(sequence)

    if len(coords) != n:
        raise RuntimeError(
            "Sequence/coordinate length mismatch."
        )

    # --------------------------------------------------------
    # Find all unordered spatial pairs <= 10 Å
    # --------------------------------------------------------

    tree = cKDTree(
        coords
    )

    spatial_pairs = tree.query_pairs(
        r=SPATIAL_CUTOFF,
        output_type="set",
    )

    # --------------------------------------------------------
    # Union spatial edges with sequential edges.
    #
    # Each unordered residue pair occurs once.
    # --------------------------------------------------------

    pairs = set(
        spatial_pairs
    )

    for i in range(n - 1):
        pairs.add(
            (i, i + 1)
        )

    pairs = sorted(
        pairs
    )

    # --------------------------------------------------------
    # Convert each undirected relation into TWO directed edges.
    #
    # This is convenient for message passing later.
    # --------------------------------------------------------

    edge_src = []
    edge_dst = []
    edge_features = []

    spatial_pair_count = 0
    sequential_pair_count = 0
    both_pair_count = 0

    for i, j in pairs:

        distance = float(
            np.linalg.norm(
                coords[i]
                -
                coords[j]
            )
        )

        sequential = (
            abs(i - j) == 1
        )

        spatial = (
            distance
            <= SPATIAL_CUTOFF
        )

        if spatial:
            spatial_pair_count += 1

        if sequential:
            sequential_pair_count += 1

        if spatial and sequential:
            both_pair_count += 1

        seq_separation = float(
            abs(i - j)
        )

        feature = [
            distance,
            float(sequential),
            seq_separation,
        ]

        # i -> j
        edge_src.append(i)
        edge_dst.append(j)
        edge_features.append(
            feature
        )

        # j -> i
        edge_src.append(j)
        edge_dst.append(i)
        edge_features.append(
            feature
        )

    edge_index = np.asarray(
        [
            edge_src,
            edge_dst,
        ],
        dtype=np.int64,
    )

    edge_attr = np.asarray(
        edge_features,
        dtype=np.float32,
    )

    return (
        edge_index,
        edge_attr,
        {
            "unique_pairs":
                len(pairs),

            "directed_edges":
                edge_index.shape[1],

            "spatial_pairs":
                spatial_pair_count,

            "sequential_pairs":
                sequential_pair_count,

            "spatial_and_sequential_pairs":
                both_pair_count,
        },
    )


# ============================================================
# LOAD ELIGIBLE TRAIN PROTEINS
# ============================================================

eligible_ids = [
    x.strip()
    for x
    in ELIGIBLE_IDS_PATH
    .read_text()
    .splitlines()
    if x.strip()
]

if len(eligible_ids) != 37919:
    raise RuntimeError(
        "Expected 37,919 eligible "
        f"TRAIN proteins, found "
        f"{len(eligible_ids)}"
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

cohort = cohort[
    cohort["uniprot_id"].isin(
        set(eligible_ids)
    )
].copy()


# ============================================================
# SELECT DETERMINISTIC SMOKE-TEST PROTEINS
# ============================================================

# Select proteins near representative lengths:
# ~100, ~300, ~600, ~1000, ~1500, ~1900 residues.

targets = [
    100,
    300,
    600,
    1000,
    1500,
    1900,
]

selected_rows = []

used_ids = set()

for target in targets:

    candidates = (
        cohort.assign(
            distance_from_target=(
                cohort["length"]
                - target
            ).abs()
        )
        .sort_values(
            [
                "distance_from_target",
                "uniprot_id",
            ]
        )
    )

    candidates = candidates[
        ~candidates["uniprot_id"]
        .isin(used_ids)
    ]

    row = candidates.iloc[0]

    selected_rows.append(
        row
    )

    used_ids.add(
        row["uniprot_id"]
    )


# ============================================================
# BUILD SMOKE-TEST GRAPHS
# ============================================================

records = []


print(
    "\n"
    + "=" * 90
)

print(
    "STRUCTURE PROTOCOL V1 — "
    "GRAPH GENERATION SMOKE TEST"
)

print(
    "=" * 90
)


for row in selected_rows:

    uid = str(
        row["uniprot_id"]
    )

    expected_sequence = str(
        row["sequence"]
    )

    expected_length = int(
        row["length"]
    )

    pdb_path = (
        PDB_DIR
        / f"{uid}.pdb"
    )

    if not pdb_path.exists():
        raise FileNotFoundError(
            pdb_path
        )

    (
        structure_sequence,
        coords,
        plddt,
    ) = parse_pdb(
        pdb_path
    )

    # --------------------------------------------------------
    # Strict frozen eligibility check
    # --------------------------------------------------------

    if (
        structure_sequence
        != expected_sequence
    ):
        raise RuntimeError(
            f"{uid}: exact sequence "
            "match failed during graph "
            "generation."
        )

    if len(coords) != expected_length:
        raise RuntimeError(
            f"{uid}: expected "
            f"{expected_length} residues, "
            f"found {len(coords)}"
        )

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    x = make_node_features(
        structure_sequence
    )

    # --------------------------------------------------------
    # Graph edges
    # --------------------------------------------------------

    (
        edge_index,
        edge_attr,
        edge_stats,
    ) = build_graph(
        structure_sequence,
        coords,
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    if x.shape != (
        expected_length,
        21,
    ):
        raise RuntimeError(
            f"{uid}: invalid node "
            f"feature shape {x.shape}"
        )

    if not np.allclose(
        x.sum(axis=1),
        1.0,
    ):
        raise RuntimeError(
            f"{uid}: one-hot validation "
            "failed."
        )

    if edge_index.shape[0] != 2:
        raise RuntimeError(
            f"{uid}: invalid edge_index."
        )

    if (
        edge_index.shape[1]
        != edge_attr.shape[0]
    ):
        raise RuntimeError(
            f"{uid}: edge count mismatch."
        )

    if edge_attr.shape[1] != 3:
        raise RuntimeError(
            f"{uid}: expected three "
            "edge features."
        )

    if not np.isfinite(
        coords
    ).all():
        raise RuntimeError(
            f"{uid}: non-finite coordinates."
        )

    if not np.isfinite(
        edge_attr
    ).all():
        raise RuntimeError(
            f"{uid}: non-finite edge features."
        )

    if not (
        (
            plddt >= 0
        )
        &
        (
            plddt <= 100
        )
    ).all():
        raise RuntimeError(
            f"{uid}: invalid pLDDT."
        )

    # --------------------------------------------------------
    # Save graph as compressed NumPy archive
    #
    # pLDDT is stored but is NOT part of x.
    # --------------------------------------------------------

    graph_path = (
        OUTPUT_DIR
        / f"{uid}.npz"
    )

    np.savez_compressed(
        graph_path,

        x=x,

        coords=coords,

        edge_index=edge_index,

        edge_attr=edge_attr,

        plddt=plddt,

        uniprot_id=np.asarray(
            uid
        ),

        sequence=np.asarray(
            structure_sequence
        ),
    )

    records.append(
        {
            "uniprot_id":
                uid,

            "length":
                expected_length,

            "node_feature_shape":
                str(x.shape),

            "unique_pairs":
                edge_stats[
                    "unique_pairs"
                ],

            "directed_edges":
                edge_stats[
                    "directed_edges"
                ],

            "spatial_pairs":
                edge_stats[
                    "spatial_pairs"
                ],

            "sequential_pairs":
                edge_stats[
                    "sequential_pairs"
                ],

            "spatial_and_sequential_pairs":
                edge_stats[
                    "spatial_and_sequential_pairs"
                ],

            "mean_plddt":
                float(
                    plddt.mean()
                ),

            "graph_file":
                str(
                    graph_path
                ),
        }
    )

    print(
        f"{uid:12s} "
        f"length={expected_length:4d}  "
        f"x={str(x.shape):12s}  "
        f"pairs="
        f"{edge_stats['unique_pairs']:6d}  "
        f"directed_edges="
        f"{edge_stats['directed_edges']:6d}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

result = pd.DataFrame(
    records
)

summary_csv = (
    OUTPUT_DIR
    / "graph_smoke_test_summary.csv"
)

result.to_csv(
    summary_csv,
    index=False,
)


summary = {
    "protocol":
        "structure_protocol_v1",

    "split":
        "train",

    "proteins":
        len(result),

    "spatial_cutoff_angstrom":
        SPATIAL_CUTOFF,

    "node_feature_dimension":
        21,

    "edge_feature_dimension":
        3,

    "plddt_stored":
        True,

    "plddt_in_node_features":
        False,

    "esm_embeddings":
        False,

    "test_used":
        False,
}


summary_json = (
    OUTPUT_DIR
    / "graph_smoke_test_summary.json"
)

with open(
    summary_json,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\n"
    + "=" * 90
)

print(
    "SMOKE TEST SUMMARY"
)

print(
    "=" * 90
)

print(
    f"Graphs generated: "
    f"{len(result)}"
)

print(
    "Node features: 21-D one-hot"
)

print(
    "Edge features: "
    "[distance, sequential indicator, "
    "sequence separation]"
)

print(
    "Spatial cutoff: 10 Å"
)

print(
    "Neighbor cap: NONE"
)

print(
    "pLDDT stored: YES"
)

print(
    "pLDDT used as node feature: NO"
)

print(
    "ESM-2 features: NO"
)

print(
    "TEST USED: NO"
)

print(
    "\nSaved:"
)

print(
    summary_csv
)

print(
    summary_json
)

print(
    OUTPUT_DIR
)

print(
    "=" * 90
)