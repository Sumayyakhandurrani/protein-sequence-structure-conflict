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
# TIMING-ONLY BENCHMARK
# No deletion/faithfulness scores are calculated.
# ============================================================

UID = "P40160"

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MODEL_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"

EMBED_DIM = 480
HIDDEN_DIM = 512
NUM_LABELS = 494
DROPOUT = 0.20

CHUNK_SIZE = 1000

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


def joint_mask_embeddings(
    sequence,
    deletion_sets,
    tokenizer,
    esm,
    batch_size,
):

    # P40160 is <1000 aa, therefore exactly one frozen ESM chunk.
    if len(sequence) > CHUNK_SIZE:
        raise RuntimeError(
            "Timing benchmark expects a <=1000-aa protein."
        )

    tokens = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )

    clean_ids = tokens["input_ids"].cpu()

    if clean_ids.shape != (
        1,
        len(sequence) + 2,
    ):
        raise RuntimeError(
            "ESM token/residue mismatch."
        )

    mask_id = tokenizer.mask_token_id

    if mask_id is None:
        raise RuntimeError(
            "Tokenizer has no mask token."
        )

    outputs = np.empty(
        (
            len(deletion_sets),
            EMBED_DIM,
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(deletion_sets),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(deletion_sets),
        )

        current_sets = deletion_sets[start:end]

        batch_ids = clean_ids.repeat(
            len(current_sets),
            1,
        )

        for j, positions in enumerate(current_sets):

            positions = np.asarray(
                positions,
                dtype=np.int64,
            )

            # positions are 0-based residues;
            # +1 accounts for BOS token.
            batch_ids[
                j,
                torch.from_numpy(positions + 1),
            ] = mask_id

        batch_ids = batch_ids.to(DEVICE)

        attention_mask = torch.ones_like(
            batch_ids,
            device=DEVICE,
        )

        with torch.inference_mode():

            out = esm(
                input_ids=batch_ids,
                attention_mask=attention_mask,
            )

        hidden = out.last_hidden_state[
            :,
            1:-1,
            :
        ]

        outputs[start:end] = (
            hidden
            .mean(dim=1)
            .cpu()
            .float()
            .numpy()
            .astype(np.float32)
        )

        del batch_ids
        del attention_mask
        del out
        del hidden

    return outputs


# ============================================================
# INPUT
# ============================================================

print("=" * 72)
print("E10 SEQUENCE JOINT-DELETION TIMING BENCHMARK")
print("=" * 72)

print("Device:", DEVICE)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA GPU required for this benchmark."
    )

checkpoint_sha = sha256_file(
    CHECKPOINT_PATH
)

print(
    "Checkpoint SHA verified:",
    checkpoint_sha == EXPECTED_CHECKPOINT_SHA,
)

if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:
    raise RuntimeError(
        "Frozen sequence checkpoint SHA mismatch."
    )


df = pd.read_pickle(TEST_PKL)

print("test.pkl columns:", list(df.columns))


# Locate ID column without changing scientific data.
id_candidates = [
    "uniprot_id",
    "accession",
    "Entry",
]

id_col = None

for col in id_candidates:
    if col in df.columns:
        id_col = col
        break

if id_col is None:
    raise RuntimeError(
        "Could not identify UniProt ID column."
    )


row = df.loc[
    df[id_col].astype(str) == UID
]

if len(row) != 1:
    raise RuntimeError(
        f"Expected exactly one row for {UID}, got {len(row)}."
    )


row = row.iloc[0]

sequence = str(
    row["sequence"]
)

L = len(sequence)
k = max(
    1,
    math.ceil(0.10 * L),
)

print("Protein:", UID)
print("Length:", L)
print("Deletion cardinality:", k)

if L != 425:
    raise RuntimeError(
        f"Unexpected P40160 length: {L}"
    )


# ============================================================
# DETERMINISTIC TIMING MASKS
# ============================================================

# Timing masks only.
# They are NOT used as scientific random-control outcomes.

seed_material = (
    f"42|{UID}|TIMING_ONLY"
).encode()

seed = int.from_bytes(
    hashlib.sha256(seed_material).digest()[:8],
    "little",
) % (2**32)

rng = np.random.default_rng(seed)

N_BENCH = 64

deletion_sets = [
    np.sort(
        rng.choice(
            L,
            size=k,
            replace=False,
        )
    )
    for _ in range(N_BENCH)
]

print("Timing perturbations:", N_BENCH)


# ============================================================
# LOAD FROZEN MODELS
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


print("Loading frozen sequence MLP...")

mlp = SequenceMLP().to(DEVICE)

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

# Test conservative batch sizes for the 3.7-GB GPU.
batch_sizes = [
    4,
    8,
    16,
]

print()
print("=" * 72)
print("TIMING")
print("=" * 72)


for batch_size in batch_sizes:

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Small warm-up.
    _ = joint_mask_embeddings(
        sequence,
        deletion_sets[:4],
        tokenizer,
        esm,
        batch_size=min(
            batch_size,
            4,
        ),
    )

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    embeddings = joint_mask_embeddings(
        sequence,
        deletion_sets,
        tokenizer,
        esm,
        batch_size=batch_size,
    )

    # Include frozen MLP inference in timing.
    x = torch.from_numpy(
        embeddings
    ).float().to(DEVICE)

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
        N_BENCH / elapsed
    )

    print()
    print("Batch size:", batch_size)
    print("Embedding shape:", embeddings.shape)
    print("Logit shape:", tuple(logits.shape))
    print("Elapsed seconds:", round(elapsed, 3))
    print(
        "Perturbations/second:",
        round(rate, 3),
    )
    print(
        "Peak allocated GPU GB:",
        round(peak_gb, 3),
    )

    # Do NOT inspect or print prediction values.
    del embeddings
    del x
    del logits


print()
print("=" * 72)
print("BENCHMARK COMPLETE")
print("No deletion drops or faithfulness outcomes were calculated.")
print("=" * 72)
