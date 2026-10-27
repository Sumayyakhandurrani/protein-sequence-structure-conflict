from pathlib import Path
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# FROZEN STRUCTURE PROTOCOL V1
# ============================================================

SPATIAL_CUTOFF = 10.0

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
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    required=True,
    choices=["train", "val"],
)

parser.add_argument(
    "--checkpoint-every",
    type=int,
    default=250,
)

args = parser.parse_args()

SPLIT = args.split

EXPECTED_COUNTS = {
    "train": 37919,
    "val": 4742,
}

EXPECTED_COUNT = EXPECTED_COUNTS[SPLIT]


# ============================================================
# PATHS
# ============================================================

STRUCTURE_ROOT = Path(
    "data/structures/alphafold_v6"
)

PROTOCOL_ROOT = Path(
    "data/structures/structure_protocol_v1"
)

PDB_DIR = (
    STRUCTURE_ROOT
    / SPLIT
    / "pdb"
)

ELIGIBLE_IDS_PATH = (
    PROTOCOL_ROOT
    / f"{SPLIT}_structure_eligible_ids.txt"
)

PRIMARY_COHORT_PATH = Path(
    "data/processed/primary_mf_cohort.csv"
)

OUTPUT_DIR = (
    PROTOCOL_ROOT
    / "graphs"
    / SPLIT
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "generation_checkpoint.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "generation_summary.json"
)


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

            if line[12:16].strip() != "CA":
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
                resname
            )

            if aa is None:
                raise RuntimeError(
                    f"Unsupported residue {resname}"
                )

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

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
# NODE FEATURES
# ============================================================

def make_node_features(sequence):

    x = np.zeros(
        (len(sequence), 21),
        dtype=np.float32,
    )

    for i, aa in enumerate(sequence):

        if aa not in AA_TO_INDEX:
            raise RuntimeError(
                f"Unsupported amino acid {aa}"
            )

        x[
            i,
            AA_TO_INDEX[aa],
        ] = 1.0

    return x


# ============================================================
# GRAPH
# ============================================================

def build_graph(sequence, coords):

    n = len(sequence)

    tree = cKDTree(coords)

    spatial_pairs = tree.query_pairs(
        r=SPATIAL_CUTOFF,
        output_type="set",
    )

    pairs = set(spatial_pairs)

    # Frozen sequential adjacency.
    for i in range(n - 1):
        pairs.add(
            (i, i + 1)
        )

    pairs = sorted(pairs)

    num_pairs = len(pairs)

    # Two directed edges per undirected pair.
    edge_index = np.empty(
        (2, num_pairs * 2),
        dtype=np.int64,
    )

    edge_attr = np.empty(
        (num_pairs * 2, 3),
        dtype=np.float32,
    )

    k = 0

    for i, j in pairs:

        distance = float(
            np.linalg.norm(
                coords[i] - coords[j]
            )
        )

        sequential = float(
            abs(i - j) == 1
        )

        separation = float(
            abs(i - j)
        )

        feature = (
            distance,
            sequential,
            separation,
        )

        edge_index[:, k] = (
            i,
            j,
        )

        edge_attr[k] = feature

        k += 1

        edge_index[:, k] = (
            j,
            i,
        )

        edge_attr[k] = feature

        k += 1

    return (
        edge_index,
        edge_attr,
        num_pairs,
    )


# ============================================================
# FAST EXISTING-GRAPH VALIDATION
# ============================================================

def existing_graph_valid(
    path,
    uid,
    expected_length,
):

    if not path.exists():
        return False

    try:

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

            if not required.issubset(
                set(data.files)
            ):
                return False

            if data["x"].shape != (
                expected_length,
                21,
            ):
                return False

            if data["coords"].shape != (
                expected_length,
                3,
            ):
                return False

            if data["plddt"].shape != (
                expected_length,
            ):
                return False

            edge_index = data[
                "edge_index"
            ]

            edge_attr = data[
                "edge_attr"
            ]

            if (
                edge_index.ndim != 2
                or
                edge_index.shape[0] != 2
            ):
                return False

            if edge_attr.shape != (
                edge_index.shape[1],
                3,
            ):
                return False

            saved_uid = str(
                data["uniprot_id"]
            )

            if saved_uid != uid:
                return False

            sequence = str(
                data["sequence"]
            )

            if len(sequence) != expected_length:
                return False

        return True

    except Exception:
        return False


# ============================================================
# LOAD IDS + COHORT
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

missing_ids = (
    set(eligible_ids)
    -
    set(cohort.index)
)

if missing_ids:

    raise RuntimeError(
        f"{len(missing_ids)} eligible IDs "
        "missing from primary cohort."
    )


# ============================================================
# GENERATE
# ============================================================

records = []

generated_now = 0
existing_valid = 0
failed = 0

start_time = time.time()


print("\n" + "=" * 90)

print(
    "STRUCTURE PROTOCOL V1 — "
    f"{SPLIT.upper()} GRAPH GENERATION"
)

print("=" * 90)

print(
    f"Eligible proteins: "
    f"{len(eligible_ids):,}"
)

print(
    f"Spatial cutoff: "
    f"{SPATIAL_CUTOFF:.1f} Å"
)

print(
    "Neighbor cap: NONE"
)

print(
    "pLDDT as GNN input: NO"
)

print(
    "TEST USED: NO\n"
)


for number, uid in enumerate(
    eligible_ids,
    start=1,
):

    row = cohort.loc[uid]

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

    graph_path = (
        OUTPUT_DIR
        / f"{uid}.npz"
    )

    status = "failed"
    error = ""

    try:

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if existing_graph_valid(
            graph_path,
            uid,
            expected_length,
        ):

            status = "existing_valid"
            existing_valid += 1

        else:

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

            # Frozen exact-match requirement.
            if (
                structure_sequence
                != expected_sequence
            ):
                raise RuntimeError(
                    "Exact sequence match failed."
                )

            if len(coords) != expected_length:
                raise RuntimeError(
                    "Residue count mismatch."
                )

            if not np.isfinite(
                coords
            ).all():
                raise RuntimeError(
                    "Non-finite coordinates."
                )

            if not np.isfinite(
                plddt
            ).all():
                raise RuntimeError(
                    "Non-finite pLDDT."
                )

            if (
                (plddt < 0).any()
                or
                (plddt > 100).any()
            ):
                raise RuntimeError(
                    "pLDDT outside [0,100]."
                )

            x = make_node_features(
                structure_sequence
            )

            (
                edge_index,
                edge_attr,
                unique_pairs,
            ) = build_graph(
                structure_sequence,
                coords,
            )

            # -----------------------------------------------
            # Basic generation-time assertions
            # -----------------------------------------------

            if x.shape != (
                expected_length,
                21,
            ):
                raise RuntimeError(
                    "Node-feature shape failure."
                )

            if not np.allclose(
                x.sum(axis=1),
                1.0,
            ):
                raise RuntimeError(
                    "One-hot feature failure."
                )

            if edge_index.shape[1] != (
                unique_pairs * 2
            ):
                raise RuntimeError(
                    "Directed-edge count failure."
                )

            if edge_attr.shape != (
                edge_index.shape[1],
                3,
            ):
                raise RuntimeError(
                    "Edge-feature shape failure."
                )

            # -----------------------------------------------
            # Atomic-ish write:
            # write temporary NPZ, then rename.
            # -----------------------------------------------

            temp_path = graph_path.with_suffix(
                ".tmp.npz"
            )

            np.savez_compressed(
                temp_path,

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

            temp_path.replace(
                graph_path
            )

            status = "generated"
            generated_now += 1

    except Exception as exc:

        failed += 1

        error = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        # Remove incomplete temporary file.
        temp_path = graph_path.with_suffix(
            ".tmp.npz"
        )

        if temp_path.exists():
            temp_path.unlink()


    records.append(
        {
            "uniprot_id":
                uid,

            "expected_length":
                expected_length,

            "status":
                status,

            "error":
                error,

            "graph_path":
                str(graph_path),
        }
    )


    # --------------------------------------------------------
    # Progress + checkpoint
    # --------------------------------------------------------

    if (
        number % args.checkpoint_every == 0
        or
        number == len(eligible_ids)
    ):

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            number / elapsed
            if elapsed > 0
            else 0
        )

        pd.DataFrame(
            records
        ).to_csv(
            CHECKPOINT_PATH,
            index=False,
        )

        print(
            f"[{number:,}/{len(eligible_ids):,}] "
            f"generated_now={generated_now:,}  "
            f"existing={existing_valid:,}  "
            f"failed={failed:,}  "
            f"rate={rate:.2f}/s"
        )


# ============================================================
# FINAL RECONCILIATION
# ============================================================

valid_graphs = 0
invalid_graphs = 0

total_bytes = 0


print(
    "\nFinal graph reconciliation..."
)


for uid in eligible_ids:

    row = cohort.loc[uid]

    expected_length = int(
        row["length"]
    )

    path = (
        OUTPUT_DIR
        / f"{uid}.npz"
    )

    if existing_graph_valid(
        path,
        uid,
        expected_length,
    ):

        valid_graphs += 1

        total_bytes += (
            path.stat().st_size
        )

    else:

        invalid_graphs += 1


elapsed = (
    time.time()
    - start_time
)


summary = {
    "protocol":
        "structure_protocol_v1",

    "split":
        SPLIT,

    "eligible_proteins":
        len(eligible_ids),

    "valid_graphs":
        valid_graphs,

    "invalid_or_missing_graphs":
        invalid_graphs,

    "generated_this_run":
        generated_now,

    "already_valid":
        existing_valid,

    "failed_this_run":
        failed,

    "spatial_cutoff_angstrom":
        SPATIAL_CUTOFF,

    "spatial_neighbor_cap":
        None,

    "node_feature_dimension":
        21,

    "edge_feature_dimension":
        3,

    "plddt_stored":
        True,

    "plddt_as_primary_gnn_input":
        False,

    "esm_embeddings":
        False,

    "total_graph_size_bytes":
        total_bytes,

    "total_graph_size_gib":
        total_bytes / (
            1024 ** 3
        ),

    "runtime_seconds":
        elapsed,

    "test_used":
        False,
}


with open(
    SUMMARY_PATH,
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
    f"{SPLIT.upper()} GRAPH GENERATION SUMMARY"
)

print("=" * 90)

print(
    f"Eligible proteins: "
    f"{len(eligible_ids):,}"
)

print(
    f"Valid graphs: "
    f"{valid_graphs:,}"
)

print(
    f"Invalid/missing graphs: "
    f"{invalid_graphs:,}"
)

print(
    f"Generated this run: "
    f"{generated_now:,}"
)

print(
    f"Already valid: "
    f"{existing_valid:,}"
)

print(
    f"Failed this run: "
    f"{failed:,}"
)

print(
    f"Total graph size: "
    f"{summary['total_graph_size_gib']:.2f} GiB"
)

print(
    f"Runtime: "
    f"{elapsed / 60:.1f} minutes"
)

print(
    "TEST USED: NO"
)

print(
    "\nSaved:"
)

print(
    CHECKPOINT_PATH
)

print(
    SUMMARY_PATH
)

print("=" * 90)


if (
    invalid_graphs != 0
    or failed != 0
):

    raise RuntimeError(
        "Graph generation did not "
        "finish cleanly. Inspect checkpoint."
    )