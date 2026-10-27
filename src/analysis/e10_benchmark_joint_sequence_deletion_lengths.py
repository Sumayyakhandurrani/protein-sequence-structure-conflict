from pathlib import Path
import hashlib
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ============================================================
# E10 TIMING-ONLY MULTI-LENGTH BENCHMARK
#
# No clean-minus-deleted scores are calculated.
# No faithfulness outcomes are calculated.
# ============================================================

REPRESENTATIVES = [
    ("SHORT",     "P17609"),
    ("MEDIAN",    "P40160"),
    ("NEAR_1000", "O14983"),
    ("LONG",      "O60312"),
    ("VERY_LONG", "Q8CC88"),
]

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MODEL_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"

EMBED_DIM = 480
HIDDEN_DIM = 512
NUM_LABELS = 494
DROPOUT = 0.20

CHUNK_SIZE = 1000

# Start conservatively because 1000-aa ESM batches
# require much more memory than the previous 425-aa test.
BATCH_SIZES = [2, 4, 8]

N_BENCH = 32

EXPECTED_CHECKPOINT_SHA = (
    "1a2abcc66553d4906e111496f98b3645"
    "80c4fd29f9d8a2ad1e181b223e89022a"
)

CHECKPOINT_PATH = Path(
    "data/models/sequence_mlp_v1/best_model.pt"
)

TEST_PKL = Path(
    "data/splits/mmseqs30_graph_v2/test.pkl"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class SequenceMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_LABELS),
        )

    def forward(self, x):
        return self.network(x)


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def joint_mask_protein_embeddings(
    sequence,
    deletion_sets,
    tokenizer,
    esm,
    batch_size,
):

    """
    Reproduce the frozen sequence-expert representation:

      1. consecutive non-overlapping <=1000-aa chunks
      2. mask all selected residues simultaneously
      3. mean pool residues within each chunk
      4. length-weighted mean across chunks

    deletion_sets contains 0-based protein residue positions.
    """

    protein_length = len(sequence)

    chunks = []

    for chunk_start in range(
        0,
        protein_length,
        CHUNK_SIZE,
    ):

        chunk = sequence[
            chunk_start:
            chunk_start + CHUNK_SIZE
        ]

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
            len(chunk) + 2,
        ):
            raise RuntimeError(
                "ESM token/residue mismatch."
            )

        chunks.append(
            (
                chunk_start,
                len(chunk),
                clean_ids,
            )
        )

    mask_id = tokenizer.mask_token_id

    if mask_id is None:
        raise RuntimeError(
            "Tokenizer has no mask token."
        )

    result = np.zeros(
        (
            len(deletion_sets),
            EMBED_DIM,
        ),
        dtype=np.float64,
    )

    for (
        chunk_start,
        chunk_length,
        clean_ids,
    ) in chunks:

        chunk_end = (
            chunk_start
            + chunk_length
        )

        for batch_start in range(
            0,
            len(deletion_sets),
            batch_size,
        ):

            batch_end = min(
                batch_start + batch_size,
                len(deletion_sets),
            )

            current_sets = deletion_sets[
                batch_start:batch_end
            ]

            B = len(current_sets)

            batch_ids = clean_ids.repeat(
                B,
                1,
            )

            for j, protein_positions in enumerate(
                current_sets
            ):

                protein_positions = np.asarray(
                    protein_positions,
                    dtype=np.int64,
                )

                local = protein_positions[
                    (protein_positions >= chunk_start)
                    & (protein_positions < chunk_end)
                ] - chunk_start

                if len(local) > 0:

                    local_tensor = torch.from_numpy(
                        local + 1
                    ).long()

                    batch_ids[
                        j,
                        local_tensor,
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
                chunk_length,
                EMBED_DIM,
            ):
                raise RuntimeError(
                    "Unexpected hidden shape."
                )

            chunk_embeddings = (
                hidden
                .mean(dim=1)
                .cpu()
                .float()
                .numpy()
                .astype(np.float32)
            )

            result[
                batch_start:batch_end
            ] += (
                chunk_embeddings.astype(
                    np.float64
                )
                * chunk_length
            )

            del batch_ids
            del attention_mask
            del out
            del hidden
            del chunk_embeddings

    result /= protein_length

    return result.astype(
        np.float32
    )


# ============================================================
# START
# ============================================================

print("=" * 78)
print("E10 MULTI-LENGTH SEQUENCE JOINT-DELETION TIMING BENCHMARK")
print("=" * 78)

print("Device:", DEVICE)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA GPU required."
    )


checkpoint_sha = sha256_file(
    CHECKPOINT_PATH
)

print(
    "Checkpoint SHA verified:",
    checkpoint_sha
    == EXPECTED_CHECKPOINT_SHA,
)

if (
    checkpoint_sha
    != EXPECTED_CHECKPOINT_SHA
):
    raise RuntimeError(
        "Frozen sequence checkpoint SHA mismatch."
    )


df = pd.read_pickle(
    TEST_PKL
)

if "uniprot_id" not in df.columns:
    raise RuntimeError(
        "uniprot_id missing from test.pkl"
    )

if "sequence" not in df.columns:
    raise RuntimeError(
        "sequence missing from test.pkl"
    )


# ============================================================
# LOAD FROZEN MODELS ONCE
# ============================================================

print()
print("Loading frozen ESM...")

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


# ============================================================
# BENCHMARK
# ============================================================

print()
print("=" * 78)
print("TIMING RESULTS")
print("=" * 78)


for group, uid in REPRESENTATIVES:

    rows = df.loc[
        df["uniprot_id"].astype(str)
        == uid
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"{uid}: expected one TEST row, "
            f"got {len(rows)}"
        )

    sequence = str(
        rows.iloc[0]["sequence"]
    )

    L = len(sequence)

    k = max(
        1,
        math.ceil(
            0.10 * L
        ),
    )

    n_chunks = math.ceil(
        L / CHUNK_SIZE
    )

    # Timing-only deterministic masks.
    # These are not scientific random controls.
    seed_material = (
        f"42|{uid}|TIMING_ONLY"
    ).encode()

    seed = int.from_bytes(
        hashlib.sha256(
            seed_material
        ).digest()[:8],
        "little",
    ) % (2**32)

    rng = np.random.default_rng(
        seed
    )

    deletion_sets = [
        np.sort(
            rng.choice(
                L,
                size=k,
                replace=False,
            )
        )
        for _ in range(
            N_BENCH
        )
    ]

    print()
    print("-" * 78)
    print(
        group,
        uid,
        f"length={L}",
        f"chunks={n_chunks}",
        f"k={k}",
    )
    print("-" * 78)

    for batch_size in BATCH_SIZES:

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warm-up using two perturbations.
        _ = joint_mask_protein_embeddings(
            sequence,
            deletion_sets[:2],
            tokenizer,
            esm,
            batch_size=min(
                2,
                batch_size,
            ),
        )

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        embeddings = (
            joint_mask_protein_embeddings(
                sequence,
                deletion_sets,
                tokenizer,
                esm,
                batch_size=batch_size,
            )
        )

        x = torch.from_numpy(
            embeddings
        ).float().to(
            DEVICE
        )

        with torch.inference_mode():
            logits = mlp(x)

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        peak_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        rate = (
            N_BENCH
            / elapsed
        )

        print(
            f"batch={batch_size:2d}  "
            f"time={elapsed:8.3f}s  "
            f"rate={rate:8.3f}/s  "
            f"peak={peak_gb:6.3f} GB  "
            f"emb={embeddings.shape}  "
            f"logits={tuple(logits.shape)}"
        )

        # Do not inspect prediction values.
        del embeddings
        del x
        del logits


print()
print("=" * 78)
print("BENCHMARK COMPLETE")
print(
    "No deletion drops or faithfulness outcomes "
    "were calculated."
)
print("=" * 78)
