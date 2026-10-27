from pathlib import Path
import hashlib
import json

import pandas as pd


BASE = Path("data/splits/mmseqs30_graph_v2")

MANIFEST_FILE = BASE / "benchmark_manifest.json"
FREEZE_FILE = BASE / "BENCHMARK_FREEZE.txt"


EXPECTED_TOTAL = 47539
EXPECTED_TRAIN = 38031
EXPECTED_VAL = 4754
EXPECTED_TEST = 4754

EXPECTED_COMPONENTS = 19749
EXPECTED_LARGEST_COMPONENT = 446

EXPECTED_PRIMARY_LABELS = 494
EXPECTED_RARE_LABELS = 151

MMSEQS_VERSION = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def count_fasta(path):
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                count += 1

    return count


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    return path


print("=" * 70)
print("CREATE GRAPH-V2 BENCHMARK MANIFEST")
print("=" * 70)


# ============================================================
# REQUIRED FILES
# ============================================================

train_pkl = require(BASE / "train.pkl")
val_pkl = require(BASE / "val.pkl")
test_pkl = require(BASE / "test.pkl")

train_fasta = require(BASE / "train.fasta")
val_fasta = require(BASE / "val.fasta")
test_fasta = require(BASE / "test.fasta")

split_assignments = require(
    BASE / "split_assignments.csv"
)

component_stats_file = require(
    BASE / "component_stats.csv"
)

freq_file = require(
    BASE / "train_go_term_frequencies.csv"
)

primary_vocab_file = require(
    BASE / "primary_go_mf_labels.csv"
)

rare_vocab_file = require(
    BASE / "secondary_rare_go_mf_labels.csv"
)

audit50_summary_file = require(
    BASE / "audit50" / "graph_v2_audit_summary.csv"
)

audit30_summary_file = require(
    BASE / "audit30" / "graph_v2_audit_summary.csv"
)

train_labels_file = require(
    BASE / "labels" / "train_labels.npy"
)

val_labels_file = require(
    BASE / "labels" / "val_labels.npy"
)

test_labels_file = require(
    BASE / "labels" / "test_labels.npy"
)

label_map_file = require(
    BASE / "labels" / "primary_label_map.csv"
)

label_metadata_file = require(
    BASE / "labels" / "label_metadata.json"
)


# ============================================================
# LOAD SPLITS
# ============================================================

print("\nLoading splits...")

train = pd.read_pickle(train_pkl)
val = pd.read_pickle(val_pkl)
test = pd.read_pickle(test_pkl)

if len(train) != EXPECTED_TRAIN:
    raise ValueError(
        f"Train size mismatch: {len(train)}"
    )

if len(val) != EXPECTED_VAL:
    raise ValueError(
        f"Validation size mismatch: {len(val)}"
    )

if len(test) != EXPECTED_TEST:
    raise ValueError(
        f"Test size mismatch: {len(test)}"
    )

total = len(train) + len(val) + len(test)

if total != EXPECTED_TOTAL:
    raise ValueError(
        f"Total size mismatch: {total}"
    )


# ============================================================
# SPLIT DISJOINTNESS
# ============================================================

train_ids = set(
    train["uniprot_id"].astype(str)
)

val_ids = set(
    val["uniprot_id"].astype(str)
)

test_ids = set(
    test["uniprot_id"].astype(str)
)

if train_ids & val_ids:
    raise ValueError(
        "Train/validation overlap detected."
    )

if train_ids & test_ids:
    raise ValueError(
        "Train/test overlap detected."
    )

if val_ids & test_ids:
    raise ValueError(
        "Validation/test overlap detected."
    )


# ============================================================
# COMPONENT CHECKS
# ============================================================

print("Checking connected components...")

component_stats = pd.read_csv(
    component_stats_file
)

required_component_cols = {
    "cluster_id",
    "n_proteins",
}

if not required_component_cols.issubset(
    component_stats.columns
):
    raise ValueError(
        "Unexpected component_stats schema."
    )

n_components = len(component_stats)

largest_component = int(
    component_stats["n_proteins"].max()
)

if n_components != EXPECTED_COMPONENTS:
    raise ValueError(
        f"Expected {EXPECTED_COMPONENTS} components, "
        f"found {n_components}"
    )

if largest_component != EXPECTED_LARGEST_COMPONENT:
    raise ValueError(
        f"Largest component mismatch: "
        f"{largest_component}"
    )


# ============================================================
# LABEL VOCABULARY
# ============================================================

print("Checking GO vocabulary...")

freq = pd.read_csv(freq_file)
primary = pd.read_csv(primary_vocab_file)
rare = pd.read_csv(rare_vocab_file)

if len(primary) != EXPECTED_PRIMARY_LABELS:
    raise ValueError(
        f"Primary vocabulary mismatch: "
        f"{len(primary)}"
    )

if len(rare) != EXPECTED_RARE_LABELS:
    raise ValueError(
        f"Rare vocabulary mismatch: "
        f"{len(rare)}"
    )

if primary["train_protein_count"].min() < 75:
    raise ValueError(
        "Primary vocabulary violates support >=75."
    )

if (
    rare["train_protein_count"].min() < 50
    or rare["train_protein_count"].max() > 74
):
    raise ValueError(
        "Rare vocabulary violates support 50-74."
    )


# ============================================================
# AUDIT CHECKS
# ============================================================

print("Checking audit results...")

audit50 = pd.read_csv(
    audit50_summary_file
)

audit30 = pd.read_csv(
    audit30_summary_file
)

audit50_violations = int(
    audit50["primary_violations"].sum()
)

audit30_violations = int(
    audit30["primary_violations"].sum()
)

if audit50_violations != 0:
    raise ValueError(
        "AUDIT50 contains primary violations."
    )

if audit30_violations != 0:
    raise ValueError(
        "AUDIT30 contains primary violations."
    )


# ============================================================
# FASTA COUNTS
# ============================================================

print("Checking FASTA counts...")

fasta_counts = {
    "train": count_fasta(train_fasta),
    "validation": count_fasta(val_fasta),
    "test": count_fasta(test_fasta),
}

if fasta_counts["train"] != EXPECTED_TRAIN:
    raise ValueError(
        "Train FASTA count mismatch."
    )

if fasta_counts["validation"] != EXPECTED_VAL:
    raise ValueError(
        "Validation FASTA count mismatch."
    )

if fasta_counts["test"] != EXPECTED_TEST:
    raise ValueError(
        "Test FASTA count mismatch."
    )


# ============================================================
# CHECKSUMS
# ============================================================

checksum_files = {
    "split_assignments.csv": split_assignments,
    "component_stats.csv": component_stats_file,
    "train_go_term_frequencies.csv": freq_file,
    "primary_go_mf_labels.csv": primary_vocab_file,
    "secondary_rare_go_mf_labels.csv": rare_vocab_file,

    "audit50_summary.csv": audit50_summary_file,
    "audit30_summary.csv": audit30_summary_file,

    "train_labels.npy": train_labels_file,
    "val_labels.npy": val_labels_file,
    "test_labels.npy": test_labels_file,

    "primary_label_map.csv": label_map_file,
    "label_metadata.json": label_metadata_file,

    "train.fasta": train_fasta,
    "val.fasta": val_fasta,
    "test.fasta": test_fasta,

    "train.pkl": train_pkl,
    "val.pkl": val_pkl,
    "test.pkl": test_pkl,
}

checksums = {}

for name, path in checksum_files.items():
    checksums[name] = sha256_file(path)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "benchmark_name": (
        "GO-MF Sequence-Structure Conflict Benchmark Graph-V2"
    ),

    "benchmark_status": "FROZEN",

    "cohort": {
        "total_proteins": EXPECTED_TOTAL,
        "length_filter": {
            "minimum_aa": 50,
            "maximum_aa": 2000,
        },
        "structure_requirement": (
            "AlphaFold structure available"
        ),
    },

    "annotation_policy": {
        "ontology": "Gene Ontology Molecular Function",
        "source_sequences": "UniProt Swiss-Prot reviewed proteins",
        "experimental_evidence_codes": [
            "EXP",
            "IDA",
            "IPI",
            "IMP",
            "IGI",
            "IEP",
        ],
        "go_propagation": (
            "Direct experimental MF annotations propagated "
            "to MF ancestors using go-basic.obo; "
            "MF root GO:0003674 excluded."
        ),
    },

    "homology_control": {
        "software": "MMseqs2",
        "version_hash": MMSEQS_VERSION,

        "graph_candidate_search": {
            "min_seq_identity": 0.30,
            "candidate_min_coverage": 0.50,
            "coverage_mode": 0,
            "sensitivity": 7.5,
            "max_seqs": 10000,
        },

        "primary_edge_definition": {
            "min_seq_identity": 0.30,
            "min_query_coverage": 0.80,
            "min_target_coverage": 0.80,
        },

        "component_method": (
            "Connected components over qualifying "
            "pairwise homology edges"
        ),
    },

    "split_policy": {
        "method": (
            "Connected-component-level greedy assignment "
            "with seed 42 and relative remaining-capacity "
            "balancing"
        ),

        "target_fraction": {
            "train": 0.80,
            "validation": 0.10,
            "test": 0.10,
        },

        "protein_counts": {
            "train": EXPECTED_TRAIN,
            "validation": EXPECTED_VAL,
            "test": EXPECTED_TEST,
        },

        "connected_components": EXPECTED_COMPONENTS,

        "largest_component": EXPECTED_LARGEST_COMPONENT,

        "crossing_components": 0,
    },

    "label_policy": {
        "selection_source": "TRAIN ONLY",

        "primary": {
            "minimum_training_protein_support": 75,
            "n_labels": EXPECTED_PRIMARY_LABELS,
        },

        "secondary_rare_analysis": {
            "minimum_training_protein_support": 50,
            "maximum_training_protein_support": 74,
            "n_labels": EXPECTED_RARE_LABELS,
        },
    },

    "homology_audits": {
        "audit50": {
            "candidate_min_coverage": 0.50,
            "candidate_pairs": {
                row["comparison"]: int(
                    row["candidate_pairs"]
                )
                for _, row in audit50.iterrows()
            },
            "primary_cross_split_violations": (
                audit50_violations
            ),
        },

        "audit30": {
            "candidate_min_coverage": 0.30,
            "candidate_pairs": {
                row["comparison"]: int(
                    row["candidate_pairs"]
                )
                for _, row in audit30.iterrows()
            },
            "primary_cross_split_violations": (
                audit30_violations
            ),
        },
    },

    "test_lock_policy": {
        "status": "LOCKED",
        "allowed_before_final_evaluation": (
            "Mechanical artifact generation and integrity/QC only"
        ),
        "forbidden_before_final_evaluation": [
            "model selection",
            "hyperparameter tuning",
            "calibration fitting",
            "decision threshold tuning",
            "architecture selection",
            "performance-based iteration",
        ],
        "note": (
            "Test label prevalence was accidentally inspected "
            "during earlier data-development steps. No test-set "
            "model performance was used for development decisions."
        ),
    },

    "fasta_counts": fasta_counts,

    "checksums_sha256": checksums,

    "important_notes": [
        (
            "Homology audits establish no detected cross-split "
            "pairs meeting the primary criterion under the "
            "specified MMseqs2 search configuration; they are "
            "not a mathematical exhaustive guarantee."
        ),
        (
            "AlphaFold confidence (including pLDDT/global metric) "
            "is a confidence signal, not ground-truth structural "
            "accuracy."
        ),
        (
            "Low-confidence structures are retained rather than "
            "removed so structural confidence can be analyzed "
            "as part of the conflict framework."
        ),
        (
            "Exact UniProt, GO, and AlphaFold database release "
            "metadata should be recorded from DATA_VERSIONS.txt "
            "and finalized before publication."
        ),
    ],
}


with open(
    MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


# ============================================================
# FREEZE RECORD
# ============================================================

freeze_text = f"""GRAPH-V2 BENCHMARK FREEZE

Status: FROZEN

Total proteins: {EXPECTED_TOTAL:,}
Train: {EXPECTED_TRAIN:,}
Validation: {EXPECTED_VAL:,}
Test: {EXPECTED_TEST:,}

Connected components: {EXPECTED_COMPONENTS:,}
Largest component: {EXPECTED_LARGEST_COMPONENT:,}

Primary GO-MF labels: {EXPECTED_PRIMARY_LABELS}
Primary threshold: train support >= 75

Secondary rare GO-MF labels: {EXPECTED_RARE_LABELS}
Rare support range: 50-74

MMseqs2 version:
{MMSEQS_VERSION}

Primary homology criterion:
identity >= 30%
query coverage >= 80%
target coverage >= 80%

AUDIT50 primary violations:
{audit50_violations}

AUDIT30 primary violations:
{audit30_violations}

Test set:
LOCKED for final evaluation only.

Do not change:
- protein split membership
- connected-component assignment
- primary GO vocabulary
- label index ordering
- homology thresholds

Any future modification requires creation of a new
benchmark version rather than overwriting Graph-V2.
"""

with open(
    FREEZE_FILE,
    "w",
    encoding="utf-8",
) as f:
    f.write(freeze_text)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK FREEZE COMPLETE")
print("=" * 70)

print(f"Total proteins:       {EXPECTED_TOTAL:,}")
print(f"Train:                {EXPECTED_TRAIN:,}")
print(f"Validation:           {EXPECTED_VAL:,}")
print(f"Test:                 {EXPECTED_TEST:,}")

print(
    f"Connected components: {EXPECTED_COMPONENTS:,}"
)

print(
    f"Largest component:    "
    f"{EXPECTED_LARGEST_COMPONENT:,}"
)

print(
    f"Primary labels:        "
    f"{EXPECTED_PRIMARY_LABELS}"
)

print(
    f"Secondary rare:        "
    f"{EXPECTED_RARE_LABELS}"
)

print(
    f"Audit50 violations:    "
    f"{audit50_violations}"
)

print(
    f"Audit30 violations:    "
    f"{audit30_violations}"
)

print(
    f"FASTA counts:          "
    f"{fasta_counts}"
)

print("\nSaved:")
print(MANIFEST_FILE)
print(FREEZE_FILE)

print("\nDONE")