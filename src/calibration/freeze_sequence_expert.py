from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path(
    "data/models/sequence_mlp_v1"
)

CAL_DIR = Path(
    "data/calibration/sequence_mlp_v1_global_temperature"
)

EMBED_DIR = Path(
    "data/embeddings/esm2_t12_35M"
)

SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2"
)

OUTPUT_DIR = Path(
    "data/models/sequence_expert_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def sha256(path):

    path = Path(path)

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def find_existing(directory, names):

    for name in names:

        p = directory / name

        if p.exists():
            return p

    return None


# ============================================================
# REQUIRED FILES
# ============================================================

checkpoint = find_existing(
    MODEL_DIR,
    [
        "best_model.pt",
        "best_checkpoint.pt",
        "model.pt",
        "checkpoint.pt",
    ],
)

if checkpoint is None:

    candidates = sorted(
        MODEL_DIR.glob("*.pt")
    )

    if len(candidates) == 1:
        checkpoint = candidates[0]

    else:
        raise FileNotFoundError(
            "Could not uniquely identify MLP v1 checkpoint. "
            f"Found: {[str(x) for x in candidates]}"
        )


temperature_file = (
    CAL_DIR / "temperature_metadata.json"
)

cal_eval_file = (
    CAL_DIR / "calibration_eval_summary.json"
)

embedding_environment = (
    EMBED_DIR
    / "esm2_sequence_v1_environment.json"
)

benchmark_manifest = (
    SPLIT_DIR
    / "benchmark_manifest.json"
)

primary_labels = (
    SPLIT_DIR
    / "primary_go_mf_labels.csv"
)


required = [
    checkpoint,
    temperature_file,
    cal_eval_file,
    embedding_environment,
    benchmark_manifest,
    primary_labels,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    temperature_file
) as f:

    temperature_meta = json.load(f)


with open(
    cal_eval_file
) as f:

    cal_eval = json.load(f)


with open(
    embedding_environment
) as f:

    embedding_meta = json.load(f)


temperature = float(
    temperature_meta["temperature"]
)


# ============================================================
# VERIFY CALIBRATION RESULT
# ============================================================

assert temperature > 0

assert (
    cal_eval["nll"]["after"]
    <
    cal_eval["nll"]["before"]
)

assert (
    cal_eval["brier"]["after"]
    <=
    cal_eval["brier"]["before"]
)

assert (
    cal_eval["ece_15"]["after"]
    <
    cal_eval["ece_15"]["before"]
)

assert (
    cal_eval["test_used"]
    is False
)


# ============================================================
# VERIFY DEVELOPMENT ARRAYS
# ============================================================

train_embeddings = np.load(
    EMBED_DIR / "train" / "train_embeddings.npy",
    mmap_mode="r"
)

val_embeddings = np.load(
    EMBED_DIR / "val" / "val_embeddings.npy",
    mmap_mode="r"
)

assert train_embeddings.shape == (
    38031,
    480,
)

assert val_embeddings.shape == (
    4754,
    480,
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "expert_name":
        "sequence_expert_v1",

    "status":
        "FROZEN_FOR_CONFLICT_DEVELOPMENT",

    "freeze_timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "benchmark": {

        "name":
            "mmseqs30_graph_v2",

        "train_proteins":
            38031,

        "validation_proteins":
            4754,

        "test_proteins":
            4754,

        "primary_go_mf_labels":
            494,

        "test_status":
            "LOCKED_AND_UNUSED_FOR_MODEL_DEVELOPMENT",
    },

    "sequence_representation": {

        "model":
            "facebook/esm2_t12_35M_UR50D",

        "revision":
            "6fbf070e65b0b7291e7bbcd451118c216cff79d8",

        "embedding_dimension":
            480,

        "fine_tuned":
            False,

        "representation":
            "final_hidden_state_residue_mean",

        "max_chunk_residues":
            1000,

        "chunking":
            (
                "consecutive non-overlapping chunks; "
                "length-weighted mean"
            ),

        "precision":
            "float32",
    },

    "predictor": {

        "name":
            "sequence_mlp_v1",

        "architecture":
            (
                "Linear(480,512) -> "
                "LayerNorm(512) -> "
                "GELU -> "
                "Dropout(0.20) -> "
                "Linear(512,494)"
            ),

        "parameters":
            500718,

        "loss":
            "unweighted BCEWithLogitsLoss",

        "optimizer":
            "AdamW",

        "learning_rate":
            1e-3,

        "weight_decay":
            1e-4,

        "batch_size":
            256,

        "seed":
            42,

        "selected_epoch":
            20,

        "checkpoint":
            str(checkpoint),
    },

    "validation_model_selection": {

        "full_validation_micro_auprc":
            0.629344,

        "full_validation_macro_auprc":
            0.395429,

        "macro_supported_labels":
            482,

        "linear_probe_micro_auprc":
            0.596223,

        "linear_probe_macro_auprc":
            0.361316,
    },

    "imbalance_ablation": {

        "method":
            (
                "sqrt positive-class weighting "
                "capped at 10"
            ),

        "selected":
            False,

        "weighted_micro_auprc":
            0.567866,

        "weighted_macro_auprc":
            0.381444,

        "reason_not_selected":
            (
                "Lower validation AUPRC overall "
                "and lower mean AP in every "
                "TRAIN-support stratum."
            ),
    },

    "calibration": {

        "method":
            "global_temperature_scaling",

        "temperature":
            temperature,

        "fit_subset_proteins":
            2377,

        "evaluation_subset_proteins":
            2377,

        "component_aware_subdivision":
            True,

        "heldout_nll_before":
            cal_eval["nll"]["before"],

        "heldout_nll_after":
            cal_eval["nll"]["after"],

        "heldout_brier_before":
            cal_eval["brier"]["before"],

        "heldout_brier_after":
            cal_eval["brier"]["after"],

        "heldout_ece15_before":
            cal_eval["ece_15"]["before"],

        "heldout_ece15_after":
            cal_eval["ece_15"]["after"],
    },

    "development_policy": {

        "sequence_expert_frozen_before_structure_conflict_analysis":
            True,

        "further_sequence_model_selection":
            False,

        "threshold_optimization_performed":
            False,

        "test_used":
            False,
    },

    "important_interpretation": (
        "Temperature-scaled sigmoid outputs are used "
        "as the calibrated sequence-expert confidence "
        "scores for subsequent conflict development. "
        "The locked benchmark test partition remains "
        "unused until final evaluation."
    ),
}


# ============================================================
# FILE CHECKSUMS
# ============================================================

manifest["checksums"] = {

    "predictor_checkpoint_sha256":
        sha256(checkpoint),

    "temperature_metadata_sha256":
        sha256(temperature_file),

    "calibration_eval_summary_sha256":
        sha256(cal_eval_file),

    "embedding_environment_sha256":
        sha256(embedding_environment),

    "benchmark_manifest_sha256":
        sha256(benchmark_manifest),

    "primary_label_vocabulary_sha256":
        sha256(primary_labels),
}


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = (
    OUTPUT_DIR
    / "sequence_expert_v1_manifest.json"
)

with open(
    manifest_path,
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# HUMAN-READABLE FREEZE FILE
# ============================================================

freeze_text = f"""SEQUENCE EXPERT V1 — FROZEN

Status:
FROZEN_FOR_CONFLICT_DEVELOPMENT

Benchmark:
mmseqs30_graph_v2

Representation:
facebook/esm2_t12_35M_UR50D
revision:
6fbf070e65b0b7291e7bbcd451118c216cff79d8

Predictor:
sequence_mlp_v1

Architecture:
480 -> 512 -> 494
LayerNorm + GELU + Dropout(0.20)

Selected epoch:
20

Loss:
Unweighted BCEWithLogitsLoss

Full validation discrimination:
Micro AUPRC = 0.629344
Macro AUPRC = 0.395429

Imbalance-weighted ablation:
NOT SELECTED

Calibration:
Global temperature scaling

Temperature:
{temperature:.8f}

Held-out calibration evaluation:

NLL:
{cal_eval["nll"]["before"]:.8f}
->
{cal_eval["nll"]["after"]:.8f}

Brier:
{cal_eval["brier"]["before"]:.8f}
->
{cal_eval["brier"]["after"]:.8f}

ECE-15:
{cal_eval["ece_15"]["before"]:.8f}
->
{cal_eval["ece_15"]["after"]:.8f}

Threshold optimization:
NO

TEST USED:
NO

POLICY:
Do not modify this sequence expert during
structure/conflict development.

Any later sequence-model change must be treated
as a new expert version rather than overwriting v1.
"""


freeze_path = (
    OUTPUT_DIR
    / "SEQUENCE_EXPERT_FREEZE.txt"
)

with open(
    freeze_path,
    "w"
) as f:

    f.write(
        freeze_text
    )


# ============================================================
# FINAL
# ============================================================

print("=" * 76)
print("SEQUENCE EXPERT V1 — FREEZE")
print("=" * 76)

print(
    "\nCheckpoint:",
    checkpoint
)

print(
    "Checkpoint SHA256:",
    manifest["checksums"][
        "predictor_checkpoint_sha256"
    ]
)

print(
    "\nESM revision:",
    manifest[
        "sequence_representation"
    ]["revision"]
)

print(
    "\nTemperature:",
    f"{temperature:.8f}"
)

print(
    "\nHeld-out calibration:"
)

print(
    "NLL:",
    f'{cal_eval["nll"]["before"]:.8f}',
    "->",
    f'{cal_eval["nll"]["after"]:.8f}'
)

print(
    "Brier:",
    f'{cal_eval["brier"]["before"]:.8f}',
    "->",
    f'{cal_eval["brier"]["after"]:.8f}'
)

print(
    "ECE-15:",
    f'{cal_eval["ece_15"]["before"]:.8f}',
    "->",
    f'{cal_eval["ece_15"]["after"]:.8f}'
)

print(
    "\nSequence expert status: FROZEN"
)

print(
    "Threshold optimization performed: NO"
)

print(
    "TEST SET USED: NO"
)

print("\nSaved:")
print(manifest_path)
print(freeze_path)

print("=" * 76)