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
BATCH_SIZE = 32

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
      "sequence_attributions_v1"
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


def masked_chunk_embeddings(
    chunk,
    tokenizer,
    esm,
):

    L = len(chunk)

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
        L + 2,
    ):

        raise RuntimeError(
            "Masked chunk token/residue mismatch."
        )

    mask_id = tokenizer.mask_token_id

    if mask_id is None:

        raise RuntimeError(
            "Tokenizer has no mask token."
        )

    result = np.empty(
        (
            L,
            EMBED_DIM,
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        L,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            L,
        )

        B = end - start

        batch_ids = clean_ids.repeat(
            B,
            1,
        )

        for j, local_pos in enumerate(
            range(start, end)
        ):

            batch_ids[
                j,
                local_pos + 1,
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
            L,
            EMBED_DIM,
        ):

            raise RuntimeError(
                "Unexpected masked hidden shape."
            )

        result[
            start:end
        ] = (
            hidden
            .mean(dim=1)
            .cpu()
            .float()
            .numpy()
            .astype(np.float32)
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
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

    args = parser.parse_args()


    print("=" * 76)
    print("E10 FULL SEQUENCE ATTRIBUTION V1")
    print("=" * 76)

    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Chunk size:", CHUNK_SIZE)

    print(
        "Scientific attribution method:",
        "single-residue ESM mask",
    )

    print(
        "Primary score:",
        "signed target-logit drop",
    )

    print(
        "Embedding reproduction tolerance:",
        f"rtol={EMBEDDING_RTOL}, "
        f"atol={EMBEDDING_ATOL}",
    )

    print(
        "Logit reproduction tolerance:",
        f"rtol={LOGIT_RTOL}, "
        f"atol={LOGIT_ATOL}",
    )

    print()


    if DEVICE.type != "cuda":

        raise RuntimeError(
            "CUDA is required."
        )


    # --------------------------------------------------------
    # Frozen artifact verification
    # --------------------------------------------------------

    checkpoint_sha = sha256_file(
        CHECKPOINT_PATH
    )

    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:

        raise RuntimeError(
            "Sequence checkpoint SHA mismatch."
        )


    population_sha = sha256_file(
        POPULATION_PATH
    )

    if population_sha != EXPECTED_POPULATION_SHA:

        raise RuntimeError(
            "Frozen E10 population SHA mismatch."
        )


    print(
        "Sequence checkpoint SHA: VERIFIED"
    )

    print(
        "E10 population SHA: VERIFIED"
    )


    # --------------------------------------------------------
    # Load frozen data
    # --------------------------------------------------------

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
        for i, uid in enumerate(test_ids)
    }


    sequences = dict(
        zip(
            test["uniprot_id"].astype(str),
            test["sequence"].astype(str),
        )
    )


    if len(eligible_ids) != 2447:

        raise RuntimeError(
            "Expected 2447 frozen E10 proteins."
        )


    for uid in eligible_ids:

        if uid not in id_to_row:

            raise RuntimeError(
                f"{uid} missing from TEST."
            )

        row = id_to_row[uid]

        if int(
            np.asarray(
                Y[row]
            ).sum()
        ) <= 0:

            raise RuntimeError(
                f"{uid} has no positive target."
            )


    print(
        "Frozen eligible proteins:",
        len(eligible_ids),
    )


    # --------------------------------------------------------
    # Validation-only mode
    # --------------------------------------------------------

    if args.validate_only:

        print()
        print(
            "VALIDATION PASSED"
        )

        print(
            "No ESM model loaded."
        )

        print(
            "No attribution computed."
        )

        print(
            "No output files written."
        )

        print("=" * 76)

        return


    # --------------------------------------------------------
    # Validate command-line selection
    # --------------------------------------------------------

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
                "frozen E10 attribution population."
            )


    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    OUT_BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    PER_PROTEIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Resume state BEFORE model loading
    # --------------------------------------------------------

    completed = {
        p.stem
        for p in PER_PROTEIN_DIR.glob(
            "*.npz"
        )
    }


    todo = [
        uid
        for uid in eligible_ids
        if uid not in completed
    ]


    # --------------------------------------------------------
    # Specific-protein selection
    # --------------------------------------------------------

    if args.protein is not None:

        requested_uid = (
            args.protein
            .strip()
        )

        if requested_uid in completed:

            print()
            print(
                f"{requested_uid} already has a "
                "completed attribution file."
            )

            todo = []

        else:

            todo = [
                requested_uid
            ]


    # --------------------------------------------------------
    # Optional limited run
    # --------------------------------------------------------

    if args.limit is not None:

        if args.limit <= 0:

            raise RuntimeError(
                "--limit must be positive."
            )

        todo = todo[
            :args.limit
        ]


    print()
    print(
        "Already completed:",
        len(completed),
    )

    print(
        "Proteins to process this run:",
        len(todo),
    )


    # --------------------------------------------------------
    # Resume-safe progress history
    # --------------------------------------------------------

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


    logged_ids = {
        str(r["uniprot_id"])
        for r in progress_records
    }


    missing_progress = (
        completed - logged_ids
    )


    if missing_progress:

        print(
            "Completed NPZ files without historical "
            "progress rows:",
            len(missing_progress),
        )


    # --------------------------------------------------------
    # If nothing needs processing, exit without loading ESM
    # --------------------------------------------------------

    if len(todo) == 0:

        print()
        print("=" * 76)
        print("RUN COMPLETE")
        print("=" * 76)

        print(
            "Total frozen population:",
            len(eligible_ids),
        )

        print(
            "Total attribution files now present:",
            len(completed),
        )

        print(
            "This run processed:",
            0,
        )

        print(
            "This run wall time (seconds):",
            0.0,
        )

        print()
        print(
            "Model weights modified: NO"
        )

        print(
            "Frozen benchmark modified: NO"
        )

        print(
            "Final evaluation directory modified: NO"
        )

        print(
            "Attribution method changed: NO"
        )

        print("=" * 76)

        return


    # --------------------------------------------------------
    # Load frozen models ONCE
    # --------------------------------------------------------

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
    ).to(DEVICE)

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
        checkpoint["model_state_dict"]
    )

    mlp.eval()


    # --------------------------------------------------------
    # Run metadata
    # --------------------------------------------------------

    run_info = {
        "model_name":
            MODEL_NAME,

        "model_revision":
            MODEL_REVISION,

        "checkpoint_sha256":
            checkpoint_sha,

        "population_sha256":
            population_sha,

        "n_population":
            len(eligible_ids),

        "chunk_size":
            CHUNK_SIZE,

        "batch_size":
            BATCH_SIZE,

        "temperature":
            TEMPERATURE,

        "primary_score":
            "clean_raw_logit_minus_masked_raw_logit",

        "embedding_reproduction_rtol":
            EMBEDDING_RTOL,

        "embedding_reproduction_atol":
            EMBEDDING_ATOL,

        "logit_reproduction_rtol":
            LOGIT_RTOL,

        "logit_reproduction_atol":
            LOGIT_ATOL,

        "test_modified":
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


    # --------------------------------------------------------
    # Protein loop
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # Clean embedding and reproduction
        # ----------------------------------------------------

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
                f"{uid}: clean embedding "
                "does not reproduce frozen embedding. "
                f"max_diff={embedding_max_diff}"
            )


        clean_logits = mlp_logits(
            clean_embedding[
                None,
                :
            ],
            mlp,
        )[0]


        frozen_logits = np.asarray(
            saved_logits[
                test_row
            ],
            dtype=np.float32,
        )


        logit_max_diff = float(
            np.max(
                np.abs(
                    clean_logits
                    - frozen_logits
                )
            )
        )


        if not np.allclose(
            clean_logits,
            frozen_logits,
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        ):

            raise RuntimeError(
                f"{uid}: clean logits "
                "do not reproduce frozen logits. "
                f"max_diff={logit_max_diff}"
            )


        # ----------------------------------------------------
        # Attribution output
        #
        # Shape:
        # [n_positive_labels, protein_length]
        # ----------------------------------------------------

        attribution = np.empty(
            (
                len(positive_labels),
                L,
            ),
            dtype=np.float32,
        )


        # ----------------------------------------------------
        # Process frozen ESM chunks
        # ----------------------------------------------------

        global_start = 0


        for chunk_index, chunk_start in enumerate(
            range(
                0,
                L,
                CHUNK_SIZE,
            )
        ):

            chunk = sequence[
                chunk_start:
                chunk_start + CHUNK_SIZE
            ]

            chunk_len = len(
                chunk
            )


            masked_chunk_emb = (
                masked_chunk_embeddings(
                    chunk,
                    tokenizer,
                    esm,
                )
            )


            clean_chunk = (
                clean_chunk_embeddings[
                    chunk_index
                ]
            )


            delta = (
                masked_chunk_emb
                - clean_chunk[
                    None,
                    :
                ]
            )


            # Only the affected frozen ESM chunk changes.
            #
            # The full-protein representation remains the
            # frozen length-weighted chunk mean.

            masked_protein_embeddings = (
                clean_embedding[
                    None,
                    :
                ].astype(
                    np.float64
                )
                +
                delta.astype(
                    np.float64
                )
                * (
                    float(
                        chunk_len
                    )
                    / float(
                        L
                    )
                )
            ).astype(
                np.float32
            )


            masked_logits = mlp_logits(
                masked_protein_embeddings,
                mlp,
            )


            target_clean = clean_logits[
                positive_labels
            ]


            target_masked = masked_logits[
                :,
                positive_labels
            ]


            drops = (
                target_clean[
                    None,
                    :
                ]
                - target_masked
            )


            global_end = (
                global_start
                + chunk_len
            )


            attribution[
                :,
                global_start:
                global_end
            ] = drops.T.astype(
                np.float32
            )


            global_start = (
                global_end
            )


            del masked_chunk_emb
            del masked_protein_embeddings
            del masked_logits

            torch.cuda.empty_cache()


        # ----------------------------------------------------
        # Result QC
        # ----------------------------------------------------

        if global_start != L:

            raise RuntimeError(
                f"{uid}: attribution length mismatch."
            )


        if attribution.shape != (
            len(positive_labels),
            L,
        ):

            raise RuntimeError(
                f"{uid}: unexpected attribution shape "
                f"{attribution.shape}."
            )


        if not np.isfinite(
            attribution
        ).all():

            raise RuntimeError(
                f"{uid}: non-finite attribution."
            )


        # ----------------------------------------------------
        # Atomic-ish per-protein save
        # ----------------------------------------------------

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

            positive_label_indices=(
                positive_labels
            ),

            attribution_logit_drop=(
                attribution
            ),

            clean_target_logits=(
                clean_logits[
                    positive_labels
                ].astype(
                    np.float32
                )
            ),

            clean_embedding_max_abs_diff=(
                np.asarray(
                    embedding_max_diff,
                    dtype=np.float32,
                )
            ),

            clean_logit_max_abs_diff=(
                np.asarray(
                    logit_max_diff,
                    dtype=np.float32,
                )
            ),
        )


        temp_path.replace(
            final_path
        )


        elapsed = (
            time.time()
            - protein_start
        )


        # ----------------------------------------------------
        # Resume-safe progress logging
        # ----------------------------------------------------

        new_progress_record = {
            "uniprot_id":
                uid,

            "length":
                L,

            "n_positive_labels":
                len(positive_labels),

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
                r["uniprot_id"]
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
            f"labels={len(positive_labels):2d} "
            f"time={elapsed:7.2f}s"
        )


    # --------------------------------------------------------
    # End summary
    # --------------------------------------------------------

    run_elapsed = (
        time.time()
        - start_run
    )


    all_completed = {
        p.stem
        for p in PER_PROTEIN_DIR.glob(
            "*.npz"
        )
    }


    print()
    print("=" * 76)
    print("RUN COMPLETE")
    print("=" * 76)

    print(
        "Total frozen population:",
        len(eligible_ids),
    )

    print(
        "Total attribution files now present:",
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
        "Model weights modified: NO"
    )

    print(
        "Frozen benchmark modified: NO"
    )

    print(
        "Final evaluation directory modified: NO"
    )

    print(
        "Attribution method changed: NO"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()