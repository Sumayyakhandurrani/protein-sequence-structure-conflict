from pathlib import Path
import json
import hashlib
import platform
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(".")

PROTOCOL_DIR = Path(
    "data/structures/structure_protocol_v1"
)

PROTOCOL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ELIGIBILITY_SUMMARY = (
    PROTOCOL_DIR
    / "structure_eligibility_summary.json"
)

GRAPH_STATS_SUMMARY = (
    PROTOCOL_DIR
    / "graph_statistics"
    / "train_500_graph_statistics_summary.json"
)

TRAIN_ELIGIBLE = (
    PROTOCOL_DIR
    / "train_structure_eligible_ids.txt"
)

TRAIN_INELIGIBLE = (
    PROTOCOL_DIR
    / "train_structure_ineligible_ids.txt"
)

VAL_ELIGIBLE = (
    PROTOCOL_DIR
    / "val_structure_eligible_ids.txt"
)

VAL_INELIGIBLE = (
    PROTOCOL_DIR
    / "val_structure_ineligible_ids.txt"
)

PRIMARY_LABELS = Path(
    "data/splits/mmseqs30_graph_v2/"
    "primary_go_mf_labels.csv"
)


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


def require(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


# ============================================================
# VERIFY REQUIRED FILES
# ============================================================

required_files = [
    ELIGIBILITY_SUMMARY,
    GRAPH_STATS_SUMMARY,
    TRAIN_ELIGIBLE,
    TRAIN_INELIGIBLE,
    VAL_ELIGIBLE,
    VAL_INELIGIBLE,
]

for path in required_files:
    require(path)


# ============================================================
# LOAD QC RESULTS
# ============================================================

with open(
    ELIGIBILITY_SUMMARY,
    "r",
    encoding="utf-8",
) as f:

    eligibility = json.load(f)


with open(
    GRAPH_STATS_SUMMARY,
    "r",
    encoding="utf-8",
) as f:

    graph_stats = json.load(f)


# ============================================================
# VERIFY ELIGIBILITY COUNTS
# ============================================================

train = eligibility["splits"]["train"]
val = eligibility["splits"]["val"]

expected = {
    "train_total": 38031,
    "train_eligible": 37919,
    "train_ineligible": 112,

    "val_total": 4754,
    "val_eligible": 4742,
    "val_ineligible": 12,

    "development_total": 42785,
    "development_eligible": 42661,
    "development_ineligible": 124,
}


checks = {
    "train_total":
        train["total"],

    "train_eligible":
        train["eligible"],

    "train_ineligible":
        train["ineligible"],

    "val_total":
        val["total"],

    "val_eligible":
        val["eligible"],

    "val_ineligible":
        val["ineligible"],

    "development_total":
        eligibility[
            "development_total"
        ]["total"],

    "development_eligible":
        eligibility[
            "development_total"
        ]["eligible"],

    "development_ineligible":
        eligibility[
            "development_total"
        ]["ineligible"],
}


for key, expected_value in expected.items():

    observed = checks[key]

    if observed != expected_value:

        raise RuntimeError(
            f"{key}: expected "
            f"{expected_value}, "
            f"observed {observed}"
        )


# ============================================================
# VERIFY GRAPH STATISTICS EXPERIMENT
# ============================================================

if graph_stats["split"] != "train":

    raise RuntimeError(
        "Graph statistics were not "
        "computed on TRAIN."
    )


if graph_stats["test_used"] is not False:

    raise RuntimeError(
        "Graph statistics indicate "
        "TEST usage."
    )


if graph_stats["sample_size"] != 500:

    raise RuntimeError(
        "Expected graph-statistics "
        "sample size of 500."
    )


if (
    float(
        graph_stats[
            "spatial_cutoff_angstrom"
        ]
    )
    != 10.0
):

    raise RuntimeError(
        "Expected 10 Å graph "
        "statistics."
    )


# ============================================================
# AMINO-ACID VOCABULARY
# ============================================================

# Fixed ordering. This ordering must be used when graph
# node features are generated.

AA_VOCAB = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "U",
]

AA_TO_INDEX = {
    aa: i
    for i, aa in enumerate(
        AA_VOCAB
    )
}


if len(AA_VOCAB) != 21:
    raise RuntimeError(
        "Expected 21 amino-acid tokens."
    )


if len(set(AA_VOCAB)) != 21:
    raise RuntimeError(
        "Duplicate amino-acid token."
    )


# ============================================================
# LABEL INFORMATION
# ============================================================

label_info = {
    "label_count": 494,
    "label_source":
        "Frozen Graph-V2 TRAIN-only "
        "GO-MF vocabulary",
    "label_ordering":
        "Frozen; must not be changed",
}


if PRIMARY_LABELS.exists():

    labels = pd.read_csv(
        PRIMARY_LABELS
    )

    if len(labels) != 494:

        raise RuntimeError(
            "Primary GO-MF label file "
            f"contains {len(labels)} labels; "
            "expected 494."
        )

    required_label_columns = {
        "go_id",
        "train_protein_count",
        "label_index",
    }

    if not required_label_columns.issubset(
        labels.columns
    ):

        raise RuntimeError(
            "Primary label file schema "
            "does not match expected schema."
        )

    expected_indices = list(
        range(494)
    )

    observed_indices = (
        labels["label_index"]
        .astype(int)
        .tolist()
    )

    if (
        observed_indices
        != expected_indices
    ):

        raise RuntimeError(
            "Frozen label ordering "
            "is not 0..493."
        )

    label_info[
        "label_file"
    ] = str(
        PRIMARY_LABELS
    )

    label_info[
        "label_file_sha256"
    ] = sha256(
        PRIMARY_LABELS
    )

else:

    label_info[
        "label_file"
    ] = (
        "Not found at expected path; "
        "494-label frozen benchmark "
        "requirement retained."
    )


# ============================================================
# PROTOCOL
# ============================================================

protocol = {

    "protocol_name":
        "structure_protocol_v1",

    "status":
        "FROZEN",

    "purpose":
        "Primary structure-derived/geometric "
        "expert for sequence-structure "
        "conflict analysis.",


    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    "benchmark": {

        "split":
            "Graph-V2",

        "graph_v2_membership_modified":
            False,

        "train_total":
            38031,

        "val_total":
            4754,

        "test_status":
            "LOCKED",

        "test_used":
            False,

        "labels":
            label_info,
    },


    # --------------------------------------------------------
    # STRUCTURE SOURCE
    # --------------------------------------------------------

    "structure_source": {

        "source":
            "AlphaFold",

        "local_format":
            "PDB",

        "downloaded_response_normalization":
            (
                "Responses detected as gzip "
                "by magic bytes are decompressed "
                "and stored as plain PDB."
            ),

        "primary_structure_eligibility":
            (
                "Exact residue-level sequence "
                "correspondence between frozen "
                "benchmark sequence and "
                "structure-derived sequence."
            ),

        "alignment_rescue":
            False,

        "approximate_sequence_matching":
            False,

        "sequence_remapping":
            False,

        "train_structure_eligible":
            37919,

        "train_structure_ineligible":
            112,

        "val_structure_eligible":
            4742,

        "val_structure_ineligible":
            12,
    },


    # --------------------------------------------------------
    # GRAPH REPRESENTATION
    # --------------------------------------------------------

    "graph_representation": {

        "graph_level":
            "protein",

        "node_level":
            "residue",

        "coordinate_atom":
            "CA",

        "coordinate_units":
            "angstrom",

        "self_edges":
            False,

        "primary_spatial_cutoff_angstrom":
            10.0,

        "spatial_neighbor_cap":
            None,

        "spatial_neighbor_cap_policy":
            (
                "No neighbor cap in primary "
                "protocol. TRAIN-only diagnostic "
                "showed sparse 10 Å graphs."
            ),

        "sequential_adjacency":
            True,

        "spatial_adjacency":
            True,
    },


    # --------------------------------------------------------
    # NODE FEATURES
    # --------------------------------------------------------

    "node_features": {

        "amino_acid_identity":
            True,

        "encoding":
            "21-dimensional one-hot",

        "vocabulary":
            AA_VOCAB,

        "vocabulary_index":
            AA_TO_INDEX,

        "esm_embeddings":
            False,

        "pretrained_sequence_embeddings":
            False,

        "plddt_as_primary_gnn_input":
            False,

        "plddt_retained_as_metadata":
            True,

        "biochemical_descriptors":
            False,
    },


    # --------------------------------------------------------
    # EDGE DEFINITION
    # --------------------------------------------------------

    "edges": {

        "spatial_rule":
            (
                "Undirected residue pair is "
                "spatially connected when "
                "CA distance <= 10 Å."
            ),

        "sequential_rule":
            (
                "Consecutive residues are "
                "explicitly represented as "
                "sequential neighbors."
            ),

        "duplicate_pair_policy":
            (
                "One residue-pair edge relation; "
                "sequential adjacency is encoded "
                "as an edge feature when the "
                "same pair is also within the "
                "spatial cutoff."
            ),

        "edge_features": [
            "CA_distance_angstrom",
            "sequential_neighbor_indicator",
            "absolute_sequence_separation",
        ],

        "neighbor_cap":
            None,
    },


    # --------------------------------------------------------
    # CONFIDENCE POLICY
    # --------------------------------------------------------

    "confidence_policy": {

        "plddt_definition":
            (
                "AlphaFold per-residue confidence; "
                "not treated as true structural "
                "accuracy."
            ),

        "filter_low_plddt_residues":
            False,

        "filter_low_plddt_proteins":
            False,

        "plddt_primary_node_feature":
            False,

        "plddt_analysis_role":
            (
                "Stored separately for confidence "
                "stratification/control in "
                "conflict analysis."
            ),

        "planned_plddt_input_ablation":
            True,
    },


    # --------------------------------------------------------
    # MODALITY SEPARATION
    # --------------------------------------------------------

    "modality_policy": {

        "esm2_used":
            False,

        "protein_language_model_used":
            False,

        "residue_identity_used":
            True,

        "terminology":
            "structure-derived/geometric expert",

        "independence_claim":
            (
                "Not claimed to be statistically "
                "sequence-independent because "
                "residue identity is supplied "
                "as a node feature."
            ),
    },


    # --------------------------------------------------------
    # DEVELOPMENT POLICY
    # --------------------------------------------------------

    "development_policy": {

        "train_available_for_development":
            True,

        "validation_available_for_development":
            True,

        "test_locked":
            True,

        "test_graphs_generated":
            False,

        "test_structures_processed":
            False,

        "structure_calibration":
            (
                "Must be performed independently "
                "from sequence-expert calibration."
            ),

        "sequence_temperature_reused":
            False,

        "sequence_expert_modified":
            False,
    },


    # --------------------------------------------------------
    # PRE-FREEZE GRAPH STATISTICS
    # --------------------------------------------------------

    "pre_freeze_graph_statistics": {

        "split":
            "TRAIN",

        "sample_size":
            500,

        "seed":
            42,

        "cutoff_angstrom":
            10.0,

        "mean_of_protein_mean_degree":
            graph_stats[
                "spatial_degree"
            ][
                "mean_of_protein_means"
            ],

        "p95_of_protein_p95_degree":
            graph_stats[
                "spatial_degree"
            ][
                "p95_of_protein_p95"
            ],

        "p99_of_protein_p99_degree":
            graph_stats[
                "spatial_degree"
            ][
                "p99_of_protein_p99"
            ],

        "largest_observed_degree":
            graph_stats[
                "spatial_degree"
            ][
                "largest_observed_degree"
            ],

        "maximum_spatial_edges":
            graph_stats[
                "spatial_edges"
            ][
                "max_per_protein"
            ],

        "test_used":
            False,
    },


    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    "environment": {

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "numpy":
            np.__version__,

        "pandas":
            pd.__version__,
    },


    # --------------------------------------------------------
    # PROVENANCE CHECKSUMS
    # --------------------------------------------------------

    "provenance": {

        "eligibility_summary_sha256":
            sha256(
                ELIGIBILITY_SUMMARY
            ),

        "graph_statistics_summary_sha256":
            sha256(
                GRAPH_STATS_SUMMARY
            ),

        "train_eligible_ids_sha256":
            sha256(
                TRAIN_ELIGIBLE
            ),

        "train_ineligible_ids_sha256":
            sha256(
                TRAIN_INELIGIBLE
            ),

        "val_eligible_ids_sha256":
            sha256(
                VAL_ELIGIBLE
            ),

        "val_ineligible_ids_sha256":
            sha256(
                VAL_INELIGIBLE
            ),
    },
}


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = (
    PROTOCOL_DIR
    / "structure_protocol_v1_manifest.json"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        protocol,
        f,
        indent=2,
    )


# ============================================================
# SAVE HUMAN-READABLE FREEZE FILE
# ============================================================

freeze_text = """\
STRUCTURE PROTOCOL V1
=====================

STATUS: FROZEN

PURPOSE
-------
Primary structure-derived/geometric expert for
sequence-structure conflict analysis.

BENCHMARK
---------
Graph-V2 membership remains unchanged.
TRAIN total: 38,031
VAL total: 4,754
Frozen GO-MF labels: 494
TEST: LOCKED AND UNUSED

STRUCTURE ELIGIBILITY
---------------------
Exact benchmark-sequence <-> AlphaFold-structure
sequence correspondence is required.

TRAIN eligible: 37,919
TRAIN ineligible: 112

VAL eligible: 4,742
VAL ineligible: 12

Alignment rescue: NO
Approximate matching: NO
Sequence remapping: NO

GRAPH
-----
One protein = one graph.
One residue = one node.
Coordinates = C-alpha coordinates.

NODE FEATURES
-------------
21-dimensional amino-acid one-hot:
A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,U

ESM-2 embeddings: NO
Pretrained sequence embeddings: NO
pLDDT as primary GNN input: NO
pLDDT retained separately: YES
Additional biochemical descriptors: NO

EDGES
-----
Spatial edge:
C-alpha distance <= 10 Angstrom.

Sequential adjacency:
YES.

Spatial neighbor cap:
NONE.

If a sequential pair is also within 10 Angstrom,
the residue pair is represented once and the
sequential relationship is recorded as an
edge feature.

EDGE FEATURES
-------------
1. C-alpha distance in Angstrom
2. Sequential-neighbor indicator
3. Absolute sequence separation

PLDDT POLICY
------------
No low-pLDDT residue filtering.
No low-pLDDT protein filtering.
pLDDT is confidence, not true structural accuracy.
pLDDT is retained for later stratification/control.
pLDDT-as-input may be evaluated as an ablation.

MODALITY POLICY
---------------
No ESM-2 or protein-language-model features.

Residue identity IS supplied to the graph model.
Therefore this branch is described as a
structure-derived/geometric expert and is NOT
claimed to be statistically sequence-independent.

DEVELOPMENT POLICY
------------------
TRAIN and VAL only during development.
TEST remains locked.
TEST structures are not processed.
TEST graphs are not generated.

Structure calibration must be performed
independently.

Sequence temperature must NOT be reused.

sequence_expert_v1 remains frozen.

VERSIONING
----------
Any change to the frozen primary graph definition,
node features, edge definition, eligibility rule,
or pLDDT-input policy requires a new protocol
version rather than silently modifying v1.
"""


freeze_path = (
    PROTOCOL_DIR
    / "STRUCTURE_PROTOCOL_V1_FREEZE.txt"
)

freeze_path.write_text(
    freeze_text,
    encoding="utf-8",
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("STRUCTURE PROTOCOL V1 FREEZE")
print("=" * 80)

print("Status: FROZEN")

print("\nStructure eligibility:")
print("TRAIN: 37,919 / 38,031")
print("VAL:   4,742 / 4,754")

print("\nNode:")
print("Residue-level")
print("21-D amino-acid one-hot")
print("C-alpha geometry")
print("pLDDT input: NO")
print("pLDDT retained: YES")
print("ESM-2 input: NO")

print("\nEdges:")
print("Spatial cutoff: 10 Å")
print("Sequential adjacency: YES")
print("Spatial neighbor cap: NONE")

print("\nLabels:")
print("494 frozen GO-MF labels")

print("\nBenchmark:")
print("Graph-V2 membership modified: NO")
print("Sequence expert modified: NO")
print("TEST USED: NO")

print("\nSaved:")
print(manifest_path)
print(freeze_path)

print("=" * 80)