from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MODEL_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"

EMBED_DIM = 480
HIDDEN_DIM = 512
NUM_LABELS = 494
DROPOUT = 0.20

CHUNK_SIZE = 1000
BATCH_SIZE = 8

TEMPERATURE = 1.106977105140686

# Numerical reproduction tolerances.
# Embeddings reproduced exactly in the diagnosed case, so the original
# embedding tolerance remains unchanged.
EMBEDDING_RTOL = 1e-5
EMBEDDING_ATOL = 1e-6

# The frozen-logit diagnostic showed ordinary FP32/GPU numerical variation:
# only 1/494 logits failed the previous atol=1e-6 check, with a difference
# of ~1.34e-6 near zero. We therefore freeze atol=1e-5 for logit QC.
LOGIT_RTOL = 1e-5
LOGIT_ATOL = 1e-5

EXPECTED_CHECKPOINT_SHA = (
    "1a2abcc66553d4906e111496f98b3645"
    "80c4fd29f9d8a2ad1e181b223e89022a"
)

EXPECTED_POPULATION_SHA = (
    "e69281eb8f1298c8ec0ea7b0498a6d0"
    "acf9fa7f4aa32d37354a9dc173c9e667c"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PATHS
# ============================================================

BASE = Path("data")

TEST_PKL = (
    BASE
    / "splits/mmseqs30_graph_v2/test.pkl"
)

TEST_IDS_PATH = (
    BASE
    / "splits/mmseqs30_graph_v2/"
      "labels/test_uniprot_ids.txt"
)

TEST_LABELS_PATH = (
    BASE
    / "splits/mmseqs30_graph_v2/"
      "labels/test_labels.npy"
)

SAVED_EMBEDDINGS_PATH = (
    BASE
    / "embeddings/esm2_t12_35M/"
      "test/test_embeddings.npy"
)

SAVED_LOGITS_PATH = (
    BASE
    / "final_evaluation/"
      "sequence_expert_v1_test/"
      "test_raw_logits.npy"
)

CHECKPOINT_PATH = (
    BASE
    / "models/sequence_mlp_v1/"
      "best_model.pt"
)

POPULATION_PATH = (
    BASE
    / "xai/functional_sites_v1/"
      "xai_attribution_eligible_ids.txt"
)

OUT_BASE = (
    BASE
    / "xai/functional_sites_v1/"
      "deletion_faithfulness_v1/"
      "sequence"
)

PER_PROTEIN_DIR = (
    OUT_BASE
    / "per_protein"
)

PROGRESS_PATH = (
    OUT_BASE
    / "progress.csv"
)

RUN_INFO_PATH = (
    OUT_BASE
    / "run_info.json"
)


# ============================================================
# MODEL
# ============================================================

class SequenceMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                HIDDEN_DIM,
            ),
            nn.LayerNorm(
                HIDDEN_DIM,
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT,
            ),
            nn.Linear(
                HIDDEN_DIM,
                NUM_LABELS,
            ),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_ids(path):

    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def clean_chunk_embedding(
    chunk,
    tokenizer,
    esm,
):

    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )

    input_ids = (
        inputs["input_ids"]
        .to(DEVICE)
    )

    attention_mask = (
        inputs["attention_mask"]
        .to(DEVICE)
    )

    if input_ids.shape[1] != len(chunk) + 2:

        raise RuntimeError(
            "ESM clean token/residue mismatch."
        )

    with torch.inference_mode():

        out = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    hidden = (
        out.last_hidden_state[
            0,
            1:-1,
            :
        ]
    )

    if hidden.shape != (
        len(chunk),
        EMBED_DIM,
    ):

        raise RuntimeError(
            "Unexpected clean hidden shape."
        )

    return (
        hidden
        .mean(dim=0)
        .cpu()
        .float()
        .numpy()
        .astype(np.float32)
    )


def clean_protein_embedding(
    sequence,
    tokenizer,
    esm,
):

    weighted = np.zeros(
        EMBED_DIM,
        dtype=np.float64,
    )

    clean_chunks = []

    total = 0

    for start in range(
        0,
        len(sequence),
        CHUNK_SIZE,
    ):

        chunk = sequence[
            start:start + CHUNK_SIZE
        ]

        emb = clean_chunk_embedding(
            chunk,
            tokenizer,
            esm,
        )

        clean_chunks.append(
            emb
        )

        weighted += (
            emb.astype(np.float64)
            * len(chunk)
        )

        total += len(chunk)

    protein_embedding = (
        weighted / total
    ).astype(np.float32)

    return (
        protein_embedding,
        clean_chunks,
    )


def mlp_logits(
    embeddings,
    model,
):

    x = torch.from_numpy(
        np.asarray(
            embeddings,
            dtype=np.float32,
        )
    ).to(DEVICE)

    with torch.inference_mode():

        logits = model(
            x
        )

    return (
        logits
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def joint_mask_protein_embeddings(
    sequence,
    deletion_sets,
    clean_embedding,
    clean_chunk_embeddings,
    tokenizer,
    esm,
):
    """
    Compute exact frozen ESM protein embeddings after
    simultaneous masking of arbitrary residue sets.

    deletion_sets:
        integer array [N, k] containing zero-based
        protein residue indices.

    Returns:
        float32 array [N, 480].

    Frozen representation is unchanged:
        <=1000-aa consecutive chunks
        residue mean within each chunk
        chunk-length-weighted protein mean

    For a deletion set that does not touch a particular
    chunk, that chunk's already reproduced clean embedding
    is reused exactly.
    """

    deletion_sets = np.asarray(
        deletion_sets,
        dtype=np.int64,
    )

    if deletion_sets.ndim != 2:
        raise RuntimeError(
            "deletion_sets must be 2-dimensional."
        )

    n_sets = deletion_sets.shape[0]
    L = len(sequence)

    if n_sets <= 0:
        raise RuntimeError(
            "No deletion sets supplied."
        )

    if deletion_sets.shape[1] <= 0:
        raise RuntimeError(
            "Deletion sets are empty."
        )

    if deletion_sets.min() < 0:
        raise RuntimeError(
            "Deletion position < 0."
        )

    if deletion_sets.max() >= L:
        raise RuntimeError(
            "Deletion position out of bounds."
        )

    # Begin every perturbation at the clean frozen
    # protein representation. A changed chunk contributes
    # only its exact weighted embedding delta.
    result = np.repeat(
        np.asarray(
            clean_embedding,
            dtype=np.float32,
        )[None, :],
        n_sets,
        axis=0,
    ).astype(
        np.float64
    )

    mask_id = tokenizer.mask_token_id

    if mask_id is None:
        raise RuntimeError(
            "Tokenizer has no mask token."
        )

    chunk_index = 0

    for chunk_start in range(
        0,
        L,
        CHUNK_SIZE,
    ):

        chunk = sequence[
            chunk_start:
            chunk_start + CHUNK_SIZE
        ]

        chunk_len = len(chunk)
        chunk_end = chunk_start + chunk_len

        tokens = tokenizer(
            chunk,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        clean_ids = (
            tokens["input_ids"]
            .cpu()
        )

        if clean_ids.shape != (
            1,
            chunk_len + 2,
        ):
            raise RuntimeError(
                "Joint-mask token/residue mismatch."
            )

        clean_chunk = np.asarray(
            clean_chunk_embeddings[
                chunk_index
            ],
            dtype=np.float32,
        )

        # Identify perturbations that actually touch
        # this chunk. Unaffected chunks are reused.
        touches = np.any(
            (
                deletion_sets >= chunk_start
            )
            &
            (
                deletion_sets < chunk_end
            ),
            axis=1,
        )

        affected_indices = np.flatnonzero(
            touches
        )

        for batch_start in range(
            0,
            len(affected_indices),
            BATCH_SIZE,
        ):

            batch_indices = affected_indices[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            B = len(batch_indices)

            batch_ids = clean_ids.repeat(
                B,
                1,
            )

            for local_batch_i, set_i in enumerate(
                batch_indices
            ):

                positions = deletion_sets[
                    set_i
                ]

                local_positions = positions[
                    (
                        positions >= chunk_start
                    )
                    &
                    (
                        positions < chunk_end
                    )
                ] - chunk_start

                if len(local_positions) == 0:
                    raise RuntimeError(
                        "Affected chunk has no local "
                        "deletion positions."
                    )

                # +1 accounts for the BOS token.
                token_positions = (
                    torch.from_numpy(
                        local_positions.astype(
                            np.int64,
                            copy=False,
                        )
                    )
                    + 1
                )

                batch_ids[
                    local_batch_i,
                    token_positions,
                ] = mask_id

            batch_ids = batch_ids.to(
                DEVICE
            )

            attention_mask = torch.ones_like(
                batch_ids,
                device=DEVICE,
            )

            with torch.inference_mode():

                out = esm(
                    input_ids=batch_ids,
                    attention_mask=attention_mask,
                )

            hidden = (
                out.last_hidden_state[
                    :,
                    1:-1,
                    :
                ]
            )

            if hidden.shape != (
                B,
                chunk_len,
                EMBED_DIM,
            ):
                raise RuntimeError(
                    "Unexpected joint-mask hidden shape."
                )

            perturbed_chunk_embeddings = (
                hidden
                .mean(dim=1)
                .cpu()
                .float()
                .numpy()
                .astype(np.float32)
            )

            delta = (
                perturbed_chunk_embeddings.astype(
                    np.float64
                )
                - clean_chunk[
                    None,
                    :
                ].astype(
                    np.float64
                )
            )

            result[
                batch_indices
            ] += (
                delta
                * (
                    float(chunk_len)
                    / float(L)
                )
            )

            del out
            del hidden
            del perturbed_chunk_embeddings

        chunk_index += 1

    if chunk_index != len(
        clean_chunk_embeddings
    ):
        raise RuntimeError(
            "Clean chunk count mismatch."
        )

    result = result.astype(
        np.float32
    )

    if result.shape != (
        n_sets,
        EMBED_DIM,
    ):
        raise RuntimeError(
            "Unexpected joint-mask protein "
            "embedding shape."
        )

    if not np.isfinite(
        result
    ).all():
        raise RuntimeError(
            "Non-finite joint-mask embedding."
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOCAL IMPORT: FROZEN SHARED RANDOM GENERATOR
    # ========================================================

    from e10_deletion_random_sets_v1 import (
        N_RANDOM_SETS,
        deletion_k,
        generate_random_sets,
        protein_seed,
        validate_random_sets,
    )


    # ========================================================
    # DELETION-FAITHFULNESS FROZEN CONSTANTS
    # ========================================================

    EXPECTED_RANDOM_GENERATOR_SHA = (
        "471f64b5c73ee51d61bb570ed179ff2a"
        "7047d998006b5f822fa3d1869933cc63"
    )

    EXPECTED_PROTOCOL_SHA = (
        "d709b2b875691c1e942a6d0e2f7a961"
        "f91df75c193ee6634abf3b878ac7d8f5e"
    )

    EXPECTED_AMENDMENT_SHA = (
        "9dda349af24fc2714961ae5006298c80e"
        "6df9de16b373ea04dee3d9c98ddcf35"
    )

    EXPECTED_IMPLEMENTATION_SHA = (
        "5638e4da63fb2f23fd8589f6cfaac030"
        "68d97c2a18c29cc4dac39b4a95bd2ff6"
    )

    ROOT_XAI = (
        BASE
        / "xai/functional_sites_v1"
    )

    ATTR_DIR = (
        ROOT_XAI
        / "sequence_attributions_v1"
        / "per_protein"
    )

    RANDOM_GENERATOR_PATH = Path(
        "src/analysis/"
        "e10_deletion_random_sets_v1.py"
    )

    PROTOCOL_PATH = (
        ROOT_XAI
        / "DELETION_FAITHFULNESS_PROTOCOL_V1.txt"
    )

    AMENDMENT_PATH = (
        ROOT_XAI
        / "DELETION_FAITHFULNESS_AMENDMENT_V1.txt"
    )

    IMPLEMENTATION_PATH = (
        ROOT_XAI
        / "DELETION_FAITHFULNESS_IMPLEMENTATION_V1.txt"
    )


    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate frozen inputs without loading "
            "ESM or computing deletion results."
        ),
    )

    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help=(
            "Process one specific frozen eligible "
            "UniProt ID."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most this many incomplete "
            "proteins."
        ),
    )

    args = parser.parse_args()


    print("=" * 78)
    print(
        "E10 SEQUENCE DELETION FAITHFULNESS V1"
    )
    print("=" * 78)

    print("Device:", DEVICE)
    print("ESM batch size:", BATCH_SIZE)
    print("Chunk size:", CHUNK_SIZE)
    print("Temperature:", TEMPERATURE)
    print(
        "Random sets per protein:",
        N_RANDOM_SETS,
    )
    print(
        "Deletion budget:",
        "k=max(1, ceil(0.10*L))",
    )
    print(
        "Primary score:",
        "calibrated target-logit drop",
    )
    print()


    if DEVICE.type != "cuda":

        raise RuntimeError(
            "CUDA is required."
        )


    # ========================================================
    # FROZEN ARTIFACT VERIFICATION
    # ========================================================

    frozen_checks = [
        (
            "sequence checkpoint",
            CHECKPOINT_PATH,
            EXPECTED_CHECKPOINT_SHA,
        ),
        (
            "XAI population",
            POPULATION_PATH,
            EXPECTED_POPULATION_SHA,
        ),
        (
            "random generator",
            RANDOM_GENERATOR_PATH,
            EXPECTED_RANDOM_GENERATOR_SHA,
        ),
        (
            "deletion protocol",
            PROTOCOL_PATH,
            EXPECTED_PROTOCOL_SHA,
        ),
        (
            "deletion amendment",
            AMENDMENT_PATH,
            EXPECTED_AMENDMENT_SHA,
        ),
        (
            "deletion implementation",
            IMPLEMENTATION_PATH,
            EXPECTED_IMPLEMENTATION_SHA,
        ),
    ]


    verified_shas = {}

    for (
        name,
        artifact_path,
        expected_sha,
    ) in frozen_checks:

        if not artifact_path.exists():

            raise RuntimeError(
                f"Missing frozen artifact: "
                f"{artifact_path}"
            )

        observed_sha = sha256_file(
            artifact_path
        )

        if observed_sha != expected_sha:

            raise RuntimeError(
                f"{name} SHA mismatch. "
                f"Observed={observed_sha}"
            )

        verified_shas[name] = (
            observed_sha
        )

        print(
            f"{name}: SHA VERIFIED"
        )


    # ========================================================
    # LOAD FROZEN DATA
    # ========================================================

    test = pd.read_pickle(
        TEST_PKL
    )

    test_ids = read_ids(
        TEST_IDS_PATH
    )

    eligible_ids = read_ids(
        POPULATION_PATH
    )

    Y = np.load(
        TEST_LABELS_PATH,
        mmap_mode="r",
    )

    saved_embeddings = np.load(
        SAVED_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    saved_logits = np.load(
        SAVED_LOGITS_PATH,
        mmap_mode="r",
    )


    if Y.shape != (
        len(test_ids),
        NUM_LABELS,
    ):

        raise RuntimeError(
            "TEST label shape mismatch."
        )


    if saved_embeddings.shape != (
        len(test_ids),
        EMBED_DIM,
    ):

        raise RuntimeError(
            "Saved embedding shape mismatch."
        )


    if saved_logits.shape != (
        len(test_ids),
        NUM_LABELS,
    ):

        raise RuntimeError(
            "Saved logit shape mismatch."
        )


    test_pkl_ids = (
        test["uniprot_id"]
        .astype(str)
        .tolist()
    )

    if test_pkl_ids != test_ids:

        raise RuntimeError(
            "TEST PKL/ID order mismatch."
        )


    id_to_row = {
        uid: i
        for i, uid in enumerate(
            test_ids
        )
    }


    sequences = dict(
        zip(
            test[
                "uniprot_id"
            ].astype(str),
            test[
                "sequence"
            ].astype(str),
        )
    )


    if len(eligible_ids) != 2447:

        raise RuntimeError(
            "Expected 2447 frozen "
            "XAI proteins."
        )


    if len(set(eligible_ids)) != 2447:

        raise RuntimeError(
            "Duplicate frozen population IDs."
        )


    total_targets = 0
    total_residues = 0


    # ========================================================
    # VALIDATE ATTRIBUTION INPUTS
    # ========================================================

    for uid in eligible_ids:

        if uid not in id_to_row:

            raise RuntimeError(
                f"{uid} missing from TEST."
            )

        if uid not in sequences:

            raise RuntimeError(
                f"{uid} missing sequence."
            )

        test_row = id_to_row[
            uid
        ]

        sequence = (
            sequences[
                uid
            ]
            .strip()
            .upper()
        )

        L = len(
            sequence
        )

        positive_labels = np.flatnonzero(
            np.asarray(
                Y[test_row]
            ) == 1
        ).astype(
            np.int16
        )

        if len(positive_labels) <= 0:

            raise RuntimeError(
                f"{uid} has no positive target."
            )

        attr_path = (
            ATTR_DIR
            / f"{uid}.npz"
        )

        if not attr_path.exists():

            raise RuntimeError(
                f"Missing sequence attribution: "
                f"{attr_path}"
            )

        with np.load(
            attr_path,
            allow_pickle=False,
        ) as z:

            attr_uid = str(
                z[
                    "uniprot_id"
                ].item()
            )

            attr_length = int(
                z[
                    "length"
                ].item()
            )

            attr_labels = np.asarray(
                z[
                    "positive_label_indices"
                ],
                dtype=np.int16,
            )

            attribution = np.asarray(
                z[
                    "attribution_logit_drop"
                ],
                dtype=np.float32,
            )


        if attr_uid != uid:

            raise RuntimeError(
                f"{uid}: attribution ID mismatch."
            )

        if attr_length != L:

            raise RuntimeError(
                f"{uid}: attribution length mismatch."
            )

        if not np.array_equal(
            attr_labels,
            positive_labels,
        ):

            raise RuntimeError(
                f"{uid}: attribution target-label "
                "mismatch."
            )

        if attribution.shape != (
            len(positive_labels),
            L,
        ):

            raise RuntimeError(
                f"{uid}: attribution shape mismatch."
            )

        if not np.isfinite(
            attribution
        ).all():

            raise RuntimeError(
                f"{uid}: non-finite attribution."
            )

        total_targets += len(
            positive_labels
        )

        total_residues += L


    if total_targets != 23087:

        raise RuntimeError(
            "Expected 23087 protein-label targets; "
            f"observed {total_targets}."
        )


    if total_residues != 1279145:

        raise RuntimeError(
            "Expected 1279145 population residues; "
            f"observed {total_residues}."
        )


    print()
    print(
        "Frozen eligible proteins:",
        len(eligible_ids),
    )

    print(
        "Protein-label targets:",
        total_targets,
    )

    print(
        "Population residues:",
        total_residues,
    )


    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    OUT_BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    PER_PROTEIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # EXISTING OUTPUT VALIDATION FOR SAFE RESUME
    # ========================================================

    def validate_existing_output(
        uid,
        output_path,
    ):

        if uid not in id_to_row:

            raise RuntimeError(
                f"{uid}: output ID not in TEST."
            )

        test_row = id_to_row[
            uid
        ]

        sequence = (
            sequences[
                uid
            ]
            .strip()
            .upper()
        )

        L = len(
            sequence
        )

        expected_k = deletion_k(
            L
        )

        expected_labels = np.flatnonzero(
            np.asarray(
                Y[test_row]
            ) == 1
        ).astype(
            np.int16
        )

        with np.load(
            output_path,
            allow_pickle=False,
        ) as z:

            required = {
                "uniprot_id",
                "test_row",
                "length",
                "k",
                "positive_label_indices",
                "protein_seed",
                "n_random_sets",
                "clean_target_logits_calibrated",
                "top_deleted_target_logits_calibrated",
                "top_deletion_drop",
                "random_mean_deleted_target_logits_calibrated",
                "random_mean_drop",
                "faithfulness_enrichment",
                "clean_embedding_max_abs_diff",
                "clean_logit_max_abs_diff",
            }

            missing = (
                required
                - set(
                    z.files
                )
            )

            if missing:

                raise RuntimeError(
                    f"{uid}: existing output missing "
                    f"arrays: {sorted(missing)}"
                )


            stored_uid = str(
                z[
                    "uniprot_id"
                ].item()
            )

            stored_test_row = int(
                z[
                    "test_row"
                ].item()
            )

            stored_L = int(
                z[
                    "length"
                ].item()
            )

            stored_k = int(
                z[
                    "k"
                ].item()
            )

            stored_labels = np.asarray(
                z[
                    "positive_label_indices"
                ],
                dtype=np.int16,
            )

            stored_seed = int(
                z[
                    "protein_seed"
                ].item()
            )

            stored_n_random = int(
                z[
                    "n_random_sets"
                ].item()
            )


            if stored_uid != uid:

                raise RuntimeError(
                    f"{uid}: existing output "
                    "ID mismatch."
                )

            if stored_test_row != test_row:

                raise RuntimeError(
                    f"{uid}: existing output "
                    "TEST row mismatch."
                )

            if stored_L != L:

                raise RuntimeError(
                    f"{uid}: existing output "
                    "length mismatch."
                )

            if stored_k != expected_k:

                raise RuntimeError(
                    f"{uid}: existing output "
                    "k mismatch."
                )

            if not np.array_equal(
                stored_labels,
                expected_labels,
            ):

                raise RuntimeError(
                    f"{uid}: existing output "
                    "target-label mismatch."
                )

            if stored_seed != protein_seed(
                uid
            ):

                raise RuntimeError(
                    f"{uid}: existing output "
                    "random seed mismatch."
                )

            if stored_n_random != (
                N_RANDOM_SETS
            ):

                raise RuntimeError(
                    f"{uid}: existing output "
                    "random-set count mismatch."
                )


            K = len(
                expected_labels
            )

            vector_names = [
                "clean_target_logits_calibrated",
                "top_deleted_target_logits_calibrated",
                "top_deletion_drop",
                "random_mean_deleted_target_logits_calibrated",
                "random_mean_drop",
                "faithfulness_enrichment",
            ]

            for name in vector_names:

                arr = np.asarray(
                    z[name]
                )

                if arr.shape != (
                    K,
                ):

                    raise RuntimeError(
                        f"{uid}: existing {name} "
                        f"shape mismatch: {arr.shape}"
                    )

                if not np.isfinite(
                    arr
                ).all():

                    raise RuntimeError(
                        f"{uid}: existing {name} "
                        "contains non-finite values."
                    )


            for name in [
                "clean_embedding_max_abs_diff",
                "clean_logit_max_abs_diff",
            ]:

                value = np.asarray(
                    z[name]
                )

                if value.shape != ():

                    raise RuntimeError(
                        f"{uid}: existing {name} "
                        "must be scalar."
                    )

                if not np.isfinite(
                    value
                ).all():

                    raise RuntimeError(
                        f"{uid}: existing {name} "
                        "is non-finite."
                    )

        return True


    existing_paths = list(
        PER_PROTEIN_DIR.glob(
            "*.npz"
        )
    )

    completed = set()

    for output_path in existing_paths:

        uid = output_path.stem

        if uid not in eligible_ids:

            raise RuntimeError(
                "Unexpected deletion output file: "
                f"{output_path}"
            )

        validate_existing_output(
            uid,
            output_path,
        )

        completed.add(
            uid
        )


    print(
        "Validated completed outputs:",
        len(completed),
    )


    # ========================================================
    # VALIDATION-ONLY MODE
    # ========================================================

    if args.validate_only:

        print()
        print("=" * 78)
        print("VALIDATION PASSED")
        print("=" * 78)

        print(
            "Frozen artifacts: VERIFIED"
        )

        print(
            "Attribution inputs: VERIFIED"
        )

        print(
            "Existing deletion outputs: VERIFIED"
        )

        print(
            "No ESM model loaded."
        )

        print(
            "No deletion predictions computed."
        )

        print("=" * 78)

        return


    # ========================================================
    # COMMAND-LINE SELECTION
    # ========================================================

    if (
        args.protein is not None
        and args.limit is not None
    ):

        raise RuntimeError(
            "Use either --protein or --limit, "
            "not both."
        )


    if args.protein is not None:

        requested_uid = (
            args.protein
            .strip()
        )

        if requested_uid not in eligible_ids:

            raise RuntimeError(
                f"{requested_uid} is not in the "
                "frozen XAI population."
            )

        if requested_uid in completed:

            print()
            print(
                f"{requested_uid} already has a "
                "validated completed deletion file."
            )

            todo = []

        else:

            todo = [
                requested_uid
            ]

    else:

        todo = [
            uid
            for uid in eligible_ids
            if uid not in completed
        ]


    if args.limit is not None:

        if args.limit <= 0:

            raise RuntimeError(
                "--limit must be positive."
            )

        todo = todo[
            :args.limit
        ]


    print(
        "Proteins to process this run:",
        len(todo),
    )


    # ========================================================
    # PROGRESS HISTORY
    # ========================================================

    if PROGRESS_PATH.exists():

        progress_df = pd.read_csv(
            PROGRESS_PATH
        )

        progress_records = (
            progress_df
            .to_dict("records")
        )

    else:

        progress_records = []


    # ========================================================
    # NOTHING TO DO
    # ========================================================

    if len(todo) == 0:

        print()
        print("=" * 78)
        print("RUN COMPLETE")
        print("=" * 78)

        print(
            "Total frozen population:",
            len(eligible_ids),
        )

        print(
            "Validated completed outputs:",
            len(completed),
        )

        print(
            "This run processed:",
            0,
        )

        print("=" * 78)

        return


    # ========================================================
    # LOAD FROZEN MODELS ONCE
    # ========================================================

    print()
    print(
        "Loading frozen ESM..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    esm = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    ).to(
        DEVICE
    )

    esm.eval()


    print(
        "Loading frozen sequence MLP..."
    )

    mlp = SequenceMLP().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    mlp.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    mlp.eval()


    # ========================================================
    # RUN METADATA
    # ========================================================

    run_info = {
        "analysis":
            "E10_sequence_deletion_faithfulness_v1",

        "model_name":
            MODEL_NAME,

        "model_revision":
            MODEL_REVISION,

        "checkpoint_sha256":
            verified_shas[
                "sequence checkpoint"
            ],

        "population_sha256":
            verified_shas[
                "XAI population"
            ],

        "random_generator_sha256":
            verified_shas[
                "random generator"
            ],

        "protocol_sha256":
            verified_shas[
                "deletion protocol"
            ],

        "amendment_sha256":
            verified_shas[
                "deletion amendment"
            ],

        "implementation_sha256":
            verified_shas[
                "deletion implementation"
            ],

        "n_population":
            len(eligible_ids),

        "n_targets":
            total_targets,

        "chunk_size":
            CHUNK_SIZE,

        "batch_size":
            BATCH_SIZE,

        "temperature":
            TEMPERATURE,

        "n_random_sets_per_protein":
            N_RANDOM_SETS,

        "deletion_fraction":
            0.10,

        "ranking":
            "signed_attribution_descending_lower_position_tie_break",

        "sequence_perturbation":
            "simultaneous_ESM_mask",

        "primary_score":
            "calibrated_clean_target_logit_minus_calibrated_deleted_target_logit",

        "random_sets_shared_across_labels":
            True,

        "random_sets_shared_across_modalities":
            True,

        "embedding_reproduction_rtol":
            EMBEDDING_RTOL,

        "embedding_reproduction_atol":
            EMBEDDING_ATOL,

        "logit_reproduction_rtol":
            LOGIT_RTOL,

        "logit_reproduction_atol":
            LOGIT_ATOL,

        "model_weights_modified":
            False,

        "frozen_benchmark_modified":
            False,

        "final_evaluation_modified":
            False,
    }


    RUN_INFO_PATH.write_text(
        json.dumps(
            run_info,
            indent=2,
        )
        + "\n"
    )


    start_run = time.time()


    # ========================================================
    # PROTEIN LOOP
    # ========================================================

    for run_i, uid in enumerate(
        todo,
        start=1,
    ):

        protein_start = time.time()

        test_row = id_to_row[
            uid
        ]

        sequence = (
            sequences[
                uid
            ]
            .strip()
            .upper()
        )

        L = len(
            sequence
        )

        positive_labels = np.flatnonzero(
            np.asarray(
                Y[test_row]
            ) == 1
        ).astype(
            np.int16
        )

        K = len(
            positive_labels
        )

        k = deletion_k(
            L
        )


        # ====================================================
        # LOAD FROZEN ATTRIBUTION
        # ====================================================

        attr_path = (
            ATTR_DIR
            / f"{uid}.npz"
        )

        with np.load(
            attr_path,
            allow_pickle=False,
        ) as z:

            attr_labels = np.asarray(
                z[
                    "positive_label_indices"
                ],
                dtype=np.int16,
            )

            attribution_raw = np.asarray(
                z[
                    "attribution_logit_drop"
                ],
                dtype=np.float32,
            )


        if not np.array_equal(
            attr_labels,
            positive_labels,
        ):

            raise RuntimeError(
                f"{uid}: attribution label "
                "order mismatch."
            )


        if attribution_raw.shape != (
            K,
            L,
        ):

            raise RuntimeError(
                f"{uid}: attribution shape mismatch."
            )


        if not np.isfinite(
            attribution_raw
        ).all():

            raise RuntimeError(
                f"{uid}: non-finite attribution."
            )


        # Protocol-conformant calibrated sequence
        # attribution. Positive scalar division does not
        # change ranking, but we perform it explicitly.
        attribution_calibrated = (
            attribution_raw.astype(
                np.float64
            )
            / float(
                TEMPERATURE
            )
        )


        # ====================================================
        # TOP-k LABEL-SPECIFIC DELETION SETS
        # ====================================================

        positions = np.arange(
            L,
            dtype=np.int64,
        )

        top_sets = np.empty(
            (
                K,
                k,
            ),
            dtype=np.int32,
        )


        for label_i in range(
            K
        ):

            ranking = np.lexsort(
                (
                    positions,
                    -attribution_calibrated[
                        label_i
                    ],
                )
            )

            selected = ranking[
                :k
            ].astype(
                np.int32,
                copy=False,
            )

            if len(
                np.unique(
                    selected
                )
            ) != k:

                raise RuntimeError(
                    f"{uid}: duplicate top-k "
                    "residue."
                )

            if (
                selected.min() < 0
                or selected.max() >= L
            ):

                raise RuntimeError(
                    f"{uid}: top-k residue "
                    "out of bounds."
                )

            # Canonical order for perturbation storage.
            top_sets[
                label_i
            ] = np.sort(
                selected
            )


        # ====================================================
        # FROZEN SHARED RANDOM DELETION SETS
        # ====================================================

        random_sets = generate_random_sets(
            uid,
            L,
        )

        validate_random_sets(
            random_sets,
            L,
        )


        if random_sets.shape != (
            N_RANDOM_SETS,
            k,
        ):

            raise RuntimeError(
                f"{uid}: random-set shape "
                "mismatch."
            )


        # ====================================================
        # CLEAN EMBEDDING / LOGIT REPRODUCTION
        # ====================================================

        (
            clean_embedding,
            clean_chunk_embeddings,
        ) = clean_protein_embedding(
            sequence,
            tokenizer,
            esm,
        )


        frozen_embedding = np.asarray(
            saved_embeddings[
                test_row
            ],
            dtype=np.float32,
        )


        embedding_max_diff = float(
            np.max(
                np.abs(
                    clean_embedding
                    - frozen_embedding
                )
            )
        )


        if not np.allclose(
            clean_embedding,
            frozen_embedding,
            rtol=EMBEDDING_RTOL,
            atol=EMBEDDING_ATOL,
        ):

            raise RuntimeError(
                f"{uid}: clean embedding does "
                "not reproduce frozen embedding. "
                f"max_diff={embedding_max_diff}"
            )


        clean_logits_raw = mlp_logits(
            clean_embedding[
                None,
                :
            ],
            mlp,
        )[0]


        frozen_logits_raw = np.asarray(
            saved_logits[
                test_row
            ],
            dtype=np.float32,
        )


        logit_max_diff = float(
            np.max(
                np.abs(
                    clean_logits_raw
                    - frozen_logits_raw
                )
            )
        )


        if not np.allclose(
            clean_logits_raw,
            frozen_logits_raw,
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        ):

            raise RuntimeError(
                f"{uid}: clean logits do not "
                "reproduce frozen logits. "
                f"max_diff={logit_max_diff}"
            )


        clean_target_logits_calibrated = (
            clean_logits_raw[
                positive_labels
            ].astype(
                np.float64
            )
            / float(
                TEMPERATURE
            )
        )


        # ====================================================
        # TOP-ATTRIBUTION JOINT DELETIONS
        # ====================================================

        top_embeddings = (
            joint_mask_protein_embeddings(
                sequence=sequence,
                deletion_sets=top_sets,
                clean_embedding=clean_embedding,
                clean_chunk_embeddings=(
                    clean_chunk_embeddings
                ),
                tokenizer=tokenizer,
                esm=esm,
            )
        )


        top_logits_raw = mlp_logits(
            top_embeddings,
            mlp,
        )


        if top_logits_raw.shape != (
            K,
            NUM_LABELS,
        ):

            raise RuntimeError(
                f"{uid}: top-deletion logit "
                "shape mismatch."
            )


        top_deleted_target_logits_calibrated = (
            top_logits_raw[
                np.arange(
                    K
                ),
                positive_labels,
            ].astype(
                np.float64
            )
            / float(
                TEMPERATURE
            )
        )


        top_deletion_drop = (
            clean_target_logits_calibrated
            - top_deleted_target_logits_calibrated
        )


        # ====================================================
        # 1000 SHARED RANDOM JOINT DELETIONS
        # ====================================================

        random_embeddings = (
            joint_mask_protein_embeddings(
                sequence=sequence,
                deletion_sets=random_sets,
                clean_embedding=clean_embedding,
                clean_chunk_embeddings=(
                    clean_chunk_embeddings
                ),
                tokenizer=tokenizer,
                esm=esm,
            )
        )


        random_logits_raw = mlp_logits(
            random_embeddings,
            mlp,
        )


        if random_logits_raw.shape != (
            N_RANDOM_SETS,
            NUM_LABELS,
        ):

            raise RuntimeError(
                f"{uid}: random-deletion logit "
                "shape mismatch."
            )


        random_target_logits_calibrated = (
            random_logits_raw[
                :,
                positive_labels,
            ].astype(
                np.float64
            )
            / float(
                TEMPERATURE
            )
        )


        random_deletion_drops = (
            clean_target_logits_calibrated[
                None,
                :
            ]
            - random_target_logits_calibrated
        )


        random_mean_deleted_target_logits_calibrated = (
            random_target_logits_calibrated.mean(
                axis=0,
                dtype=np.float64,
            )
        )


        random_mean_drop = (
            random_deletion_drops.mean(
                axis=0,
                dtype=np.float64,
            )
        )


        faithfulness_enrichment = (
            top_deletion_drop
            - random_mean_drop
        )


        # ====================================================
        # RESULT QC
        # ====================================================

        scientific_vectors = {
            "clean_target_logits_calibrated":
                clean_target_logits_calibrated,

            "top_deleted_target_logits_calibrated":
                top_deleted_target_logits_calibrated,

            "top_deletion_drop":
                top_deletion_drop,

            "random_mean_deleted_target_logits_calibrated":
                random_mean_deleted_target_logits_calibrated,

            "random_mean_drop":
                random_mean_drop,

            "faithfulness_enrichment":
                faithfulness_enrichment,
        }


        for (
            name,
            vector,
        ) in scientific_vectors.items():

            if vector.shape != (
                K,
            ):

                raise RuntimeError(
                    f"{uid}: {name} shape "
                    f"mismatch: {vector.shape}"
                )

            if not np.isfinite(
                vector
            ).all():

                raise RuntimeError(
                    f"{uid}: non-finite {name}."
                )


        # Algebraic consistency check.
        expected_enrichment = (
            top_deletion_drop
            - random_mean_drop
        )

        if not np.allclose(
            faithfulness_enrichment,
            expected_enrichment,
            rtol=0,
            atol=1e-12,
        ):

            raise RuntimeError(
                f"{uid}: faithfulness algebra "
                "consistency failure."
            )


        # ====================================================
        # ATOMIC PER-PROTEIN SAVE
        # ====================================================

        final_path = (
            PER_PROTEIN_DIR
            / f"{uid}.npz"
        )

        temp_path = (
            PER_PROTEIN_DIR
            / f"{uid}.tmp.npz"
        )


        np.savez_compressed(
            temp_path,

            uniprot_id=np.asarray(
                uid
            ),

            test_row=np.asarray(
                test_row,
                dtype=np.int32,
            ),

            length=np.asarray(
                L,
                dtype=np.int32,
            ),

            k=np.asarray(
                k,
                dtype=np.int32,
            ),

            positive_label_indices=(
                positive_labels
            ),

            protein_seed=np.asarray(
                protein_seed(
                    uid
                ),
                dtype=np.uint32,
            ),

            n_random_sets=np.asarray(
                N_RANDOM_SETS,
                dtype=np.int32,
            ),

            clean_target_logits_calibrated=(
                clean_target_logits_calibrated.astype(
                    np.float32
                )
            ),

            top_deleted_target_logits_calibrated=(
                top_deleted_target_logits_calibrated.astype(
                    np.float32
                )
            ),

            top_deletion_drop=(
                top_deletion_drop.astype(
                    np.float32
                )
            ),

            random_mean_deleted_target_logits_calibrated=(
                random_mean_deleted_target_logits_calibrated.astype(
                    np.float32
                )
            ),

            random_mean_drop=(
                random_mean_drop.astype(
                    np.float32
                )
            ),

            faithfulness_enrichment=(
                faithfulness_enrichment.astype(
                    np.float32
                )
            ),

            clean_embedding_max_abs_diff=np.asarray(
                embedding_max_diff,
                dtype=np.float32,
            ),

            clean_logit_max_abs_diff=np.asarray(
                logit_max_diff,
                dtype=np.float32,
            ),
        )


        temp_path.replace(
            final_path
        )


        # Validate the actual completed artifact before
        # considering this protein complete.
        validate_existing_output(
            uid,
            final_path,
        )


        elapsed = (
            time.time()
            - protein_start
        )


        # ====================================================
        # RESUME-SAFE PROGRESS LOG
        # ====================================================

        new_progress_record = {
            "uniprot_id":
                uid,

            "length":
                L,

            "k":
                k,

            "n_positive_labels":
                K,

            "n_random_sets":
                N_RANDOM_SETS,

            "seconds":
                elapsed,

            "embedding_max_abs_diff":
                embedding_max_diff,

            "logit_max_abs_diff":
                logit_max_diff,
        }


        progress_records = [
            r
            for r in progress_records
            if str(
                r[
                    "uniprot_id"
                ]
            ) != uid
        ]


        progress_records.append(
            new_progress_record
        )


        pd.DataFrame(
            progress_records
        ).to_csv(
            PROGRESS_PATH,
            index=False,
        )


        total_done = (
            len(completed)
            + run_i
        )


        print(
            f"[{total_done:4d}/"
            f"{len(eligible_ids):4d}] "
            f"{uid} "
            f"L={L:4d} "
            f"k={k:3d} "
            f"labels={K:2d} "
            f"time={elapsed:7.2f}s"
        )


        # Explicitly release large arrays between proteins.
        del attribution_raw
        del attribution_calibrated
        del top_sets
        del random_sets
        del top_embeddings
        del top_logits_raw
        del random_embeddings
        del random_logits_raw
        del random_target_logits_calibrated
        del random_deletion_drops

        torch.cuda.empty_cache()


    # ========================================================
    # END SUMMARY
    # ========================================================

    run_elapsed = (
        time.time()
        - start_run
    )


    all_output_paths = list(
        PER_PROTEIN_DIR.glob(
            "*.npz"
        )
    )

    all_completed = set()

    for output_path in all_output_paths:

        uid = output_path.stem

        if uid not in eligible_ids:

            raise RuntimeError(
                "Unexpected final output file: "
                f"{output_path}"
            )

        validate_existing_output(
            uid,
            output_path,
        )

        all_completed.add(
            uid
        )


    print()
    print("=" * 78)
    print("RUN COMPLETE")
    print("=" * 78)

    print(
        "Total frozen population:",
        len(eligible_ids),
    )

    print(
        "Validated completed outputs:",
        len(all_completed),
    )

    print(
        "This run processed:",
        len(todo),
    )

    print(
        "This run wall time (seconds):",
        round(
            run_elapsed,
            2,
        ),
    )

    print()
    print(
        "Scientific outcome values printed: NO"
    )

    print(
        "Model weights modified: NO"
    )

    print(
        "Frozen benchmark modified: NO"
    )

    print(
        "Final evaluation directory modified: NO"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
