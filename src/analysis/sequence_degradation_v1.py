from pathlib import Path
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
CHUNK_SIZE = 1000
NUM_LABELS = 494

HIDDEN_DIM = 512
DROPOUT = 0.20
TEMPERATURE = 1.106977105140686

GLOBAL_SEED = 42

MASK_LEVELS = [
    0.0,
    0.10,
    0.20,
    0.40,
]

DEVICE = "cuda"

BASE = Path("data")

VAL_PATH = (
    BASE
    / "splits"
    / "mmseqs30_graph_v2"
    / "val.pkl"
)

INPUT_DIR = (
    BASE
    / "conflict"
    / "go_frequency_v1_inputs"
)

EVAL_IDS_PATH = (
    INPUT_DIR
    / "eval_ids.txt"
)

MODEL_DIR = (
    BASE
    / "models"
    / "sequence_mlp_v1"
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

OUT_BASE = (
    BASE
    / "robustness"
    / "sequence_degradation_v1"
)

PROTOCOL_PATH = (
    OUT_BASE
    / "SEQUENCE_DEGRADATION_PROTOCOL_V1.txt"
)

EXPECTED_PROTOCOL_SHA = (
    "4954b628d10bbaf276ca42be2b224bfc4d8cbdaa11f972f03df04ad2fe007d6b"
)

EXPECTED_CHECKPOINT_SHA = (
    "1a2abcc66553d4906e111496f98b364580c4fd29f9d8a2ad1e181b223e89022a"
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
                HIDDEN_DIM
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
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

def sha256(path):

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


def condition_name(fraction):

    if fraction == 0:
        return "clean"

    return (
        "mask_"
        + str(int(round(fraction * 100)))
        + "pct"
    )


def protein_seed(uid, fraction):

    text = (
        f"{GLOBAL_SEED}|"
        f"{uid}|"
        f"{fraction:.6f}"
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def make_position_tokens(
    sequence,
    uid,
    fraction,
):

    tokens = list(sequence)

    n = len(tokens)

    if fraction == 0:
        return tokens, 0

    n_mask = int(
        round(
            fraction * n
        )
    )

    if n_mask <= 0:
        return tokens, 0

    n_mask = min(
        n_mask,
        n,
    )

    rng = np.random.default_rng(
        protein_seed(
            uid,
            fraction,
        )
    )

    positions = rng.choice(
        n,
        size=n_mask,
        replace=False,
    )

    for pos in positions:
        tokens[int(pos)] = "<mask>"

    return tokens, n_mask


def embed_token_chunk(
    position_tokens,
    tokenizer,
    esm_model,
):

    text = "".join(
        position_tokens
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )

    input_ids = inputs[
        "input_ids"
    ][0]

    expected_token_count = (
        len(position_tokens) + 2
    )

    if len(input_ids) != expected_token_count:
        raise RuntimeError(
            "Tokenizer position mismatch: "
            f"expected {expected_token_count} "
            f"tokens including BOS/EOS, "
            f"observed {len(input_ids)}."
        )

    inputs = {
        k: v.to(DEVICE)
        for k, v
        in inputs.items()
    }

    with torch.inference_mode():
        output = esm_model(
            **inputs
        )

    residues = (
        output
        .last_hidden_state[
            0,
            1:-1,
            :
        ]
    )

    if (
        residues.shape[0]
        != len(position_tokens)
    ):
        raise RuntimeError(
            "Residue-position/token mismatch: "
            f"positions={len(position_tokens)}, "
            f"embeddings={residues.shape[0]}"
        )

    embedding = residues.mean(
        dim=0
    )

    return (
        embedding
        .cpu()
        .float()
        .numpy()
    )


def embed_protein(
    position_tokens,
    tokenizer,
    esm_model,
):

    chunks = [
        position_tokens[
            i:i + CHUNK_SIZE
        ]
        for i in range(
            0,
            len(position_tokens),
            CHUNK_SIZE,
        )
    ]

    weighted_sum = np.zeros(
        EMBED_DIM,
        dtype=np.float64,
    )

    total_positions = 0

    for chunk in chunks:

        chunk_embedding = (
            embed_token_chunk(
                chunk,
                tokenizer,
                esm_model,
            )
        )

        weighted_sum += (
            chunk_embedding.astype(
                np.float64
            )
            * len(chunk)
        )

        total_positions += len(
            chunk
        )

    if total_positions != len(
        position_tokens
    ):
        raise RuntimeError(
            "Protein pooling position count mismatch."
        )

    embedding = (
        weighted_sum
        / total_positions
    )

    return (
        embedding.astype(
            np.float32
        ),
        len(chunks),
    )


# ============================================================
# VERIFY FROZEN FILES
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available."
    )

if sha256(
    PROTOCOL_PATH
) != EXPECTED_PROTOCOL_SHA:
    raise RuntimeError(
        "E7 frozen protocol SHA256 mismatch."
    )

if sha256(
    CHECKPOINT_PATH
) != EXPECTED_CHECKPOINT_SHA:
    raise RuntimeError(
        "Frozen sequence checkpoint SHA256 mismatch."
    )


# ============================================================
# LOAD CAL-EVAL SEQUENCES
# ============================================================

eval_ids = read_ids(
    EVAL_IDS_PATH
)

if len(eval_ids) != 2373:
    raise RuntimeError(
        f"Expected 2373 CAL-EVAL IDs, "
        f"found {len(eval_ids)}."
    )

if len(set(eval_ids)) != len(
    eval_ids
):
    raise RuntimeError(
        "CAL-EVAL IDs are not unique."
    )

df = pd.read_pickle(
    VAL_PATH
)

if (
    df["uniprot_id"]
    .astype(str)
    .duplicated()
    .any()
):
    raise RuntimeError(
        "Duplicate UniProt IDs in VAL."
    )

seq_map = dict(
    zip(
        df["uniprot_id"].astype(
            str
        ),
        df["sequence"].astype(
            str
        ),
    )
)

missing = [
    uid
    for uid in eval_ids
    if uid not in seq_map
]

if missing:
    raise RuntimeError(
        f"{len(missing)} CAL-EVAL IDs "
        "missing from VAL."
    )

sequences = [
    seq_map[uid]
    .strip()
    .upper()
    for uid in eval_ids
]

if any(
    len(seq) == 0
    for seq in sequences
):
    raise RuntimeError(
        "Empty sequence found."
    )


# ============================================================
# LOAD FROZEN ESM-2
# ============================================================

print("=" * 80)
print("E7 SEQUENCE DEGRADATION V1")
print("=" * 80)

print(
    "CAL-EVAL proteins:",
    len(eval_ids),
)

print(
    "Mask levels:",
    MASK_LEVELS,
)

print(
    "ESM model:",
    MODEL_NAME,
)

print(
    "ESM revision:",
    MODEL_REVISION,
)

print(
    "Protocol SHA256: VERIFIED"
)

print(
    "Sequence checkpoint SHA256: VERIFIED"
)

print(
    "GPU:",
    torch.cuda.get_device_name(0),
)

print()

print("Loading tokenizer...")

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
)

if tokenizer.mask_token != "<mask>":
    raise RuntimeError(
        "Unexpected ESM mask token."
    )

print("Loading frozen ESM-2...")

esm_model = (
    AutoModel
    .from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
    .to(DEVICE)
)

esm_model.eval()


# ============================================================
# LOAD FROZEN SEQUENCE EXPERT
# ============================================================

print(
    "Loading frozen sequence expert..."
)

sequence_model = (
    SequenceMLP()
    .to(DEVICE)
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

sequence_model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

sequence_model.eval()

print(
    "Sequence checkpoint epoch:",
    checkpoint["epoch"],
)

print(
    "Temperature:",
    TEMPERATURE,
)

print()


# ============================================================
# RUN CONDITIONS
# ============================================================

for fraction in MASK_LEVELS:

    name = condition_name(
        fraction
    )

    out_dir = (
        OUT_BASE / name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        out_dir
        / "embeddings.npy"
    )

    logits_path = (
        out_dir
        / "raw_logits.npy"
    )

    probabilities_path = (
        out_dir
        / "calibrated_probabilities.npy"
    )

    ids_path = (
        out_dir
        / "uniprot_ids.txt"
    )

    mask_counts_path = (
        out_dir
        / "mask_counts.npy"
    )

    progress_path = (
        out_dir
        / "progress.json"
    )

    metadata_path = (
        out_dir
        / "metadata.json"
    )

    n = len(eval_ids)

    print("=" * 80)
    print(
        "Condition:",
        name,
    )
    print(
        "Mask fraction:",
        fraction,
    )
    print("=" * 80)

    # --------------------------------------------------------
    # CREATE / RESUME EMBEDDINGS
    # --------------------------------------------------------

    if embeddings_path.exists():

        embeddings = (
            np.lib.format.open_memmap(
                embeddings_path,
                mode="r+",
            )
        )

        if embeddings.shape != (
            n,
            EMBED_DIM,
        ):
            raise RuntimeError(
                f"{name}: existing embedding "
                f"shape {embeddings.shape} invalid."
            )

    else:

        embeddings = (
            np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    n,
                    EMBED_DIM,
                ),
            )
        )

        embeddings[:] = np.nan
        embeddings.flush()

    if mask_counts_path.exists():

        mask_counts = (
            np.lib.format.open_memmap(
                mask_counts_path,
                mode="r+",
            )
        )

        if mask_counts.shape != (
            n,
        ):
            raise RuntimeError(
                f"{name}: mask-count shape invalid."
            )

    else:

        mask_counts = (
            np.lib.format.open_memmap(
                mask_counts_path,
                mode="w+",
                dtype=np.int32,
                shape=(n,),
            )
        )

        mask_counts[:] = -1
        mask_counts.flush()

    if progress_path.exists():

        progress = json.loads(
            progress_path.read_text()
        )

        completed = int(
            progress["completed"]
        )

    else:
        completed = 0

    if (
        completed < 0
        or completed > n
    ):
        raise RuntimeError(
            f"{name}: invalid progress."
        )

    if completed > 0:

        if not np.isfinite(
            np.asarray(
                embeddings[:completed]
            )
        ).all():
            raise RuntimeError(
                f"{name}: completed embeddings invalid."
            )

        if (
            np.asarray(
                mask_counts[:completed]
            ) < 0
        ).any():
            raise RuntimeError(
                f"{name}: completed mask counts invalid."
            )

    if completed < n:

        if not np.isnan(
            np.asarray(
                embeddings[completed:]
            )
        ).all():
            raise RuntimeError(
                f"{name}: unexpected future embeddings."
            )

        if not (
            np.asarray(
                mask_counts[completed:]
            ) == -1
        ).all():
            raise RuntimeError(
                f"{name}: unexpected future mask counts."
            )

    print(
        "Resume position:",
        completed,
        "/",
        n,
    )

    start = time.time()

    for i in range(
        completed,
        n,
    ):

        uid = eval_ids[i]
        sequence = sequences[i]

        position_tokens, n_mask = (
            make_position_tokens(
                sequence,
                uid,
                fraction,
            )
        )

        if len(
            position_tokens
        ) != len(sequence):
            raise RuntimeError(
                f"{uid}: position count changed."
            )

        embedding, n_chunks = (
            embed_protein(
                position_tokens,
                tokenizer,
                esm_model,
            )
        )

        if embedding.shape != (
            EMBED_DIM,
        ):
            raise RuntimeError(
                f"{uid}: invalid embedding shape."
            )

        if not np.isfinite(
            embedding
        ).all():
            raise RuntimeError(
                f"{uid}: embedding NaN/Inf."
            )

        embeddings[i] = embedding
        mask_counts[i] = n_mask

        embeddings.flush()
        mask_counts.flush()

        completed_now = i + 1

        progress = {
            "protocol":
                "sequence_degradation_v1",
            "condition":
                name,
            "mask_fraction":
                fraction,
            "completed":
                completed_now,
            "total":
                n,
            "last_uniprot_id":
                uid,
        }

        progress_path.write_text(
            json.dumps(
                progress,
                indent=2,
            )
            + "\n"
        )

        elapsed = (
            time.time() - start
        )

        processed = (
            completed_now
            - completed
        )

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            n - completed_now
        )

        eta = (
            remaining
            / rate
            / 60
            if rate > 0
            else float("inf")
        )

        if (
            completed_now == n
            or completed_now % 50 == 0
        ):
            print(
                f"{completed_now:5d}/{n} | "
                f"{uid} | "
                f"L={len(sequence):4d} | "
                f"masked={n_mask:4d} | "
                f"chunks={n_chunks} | "
                f"{rate:.2f} proteins/s | "
                f"ETA={eta:.1f} min"
            )

    # --------------------------------------------------------
    # VALIDATE COMPLETE EMBEDDINGS
    # --------------------------------------------------------

    X = np.asarray(
        embeddings
    )

    if X.shape != (
        n,
        EMBED_DIM,
    ):
        raise RuntimeError(
            f"{name}: final embedding shape invalid."
        )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            f"{name}: final embeddings contain NaN/Inf."
        )

    counts = np.asarray(
        mask_counts
    )

    if (
        counts < 0
    ).any():
        raise RuntimeError(
            f"{name}: invalid final mask counts."
        )

    # --------------------------------------------------------
    # FROZEN SEQUENCE EXPERT INFERENCE
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = sequence_model(
            torch.from_numpy(
                X
            )
            .float()
            .to(DEVICE)
        )

        probabilities = torch.sigmoid(
            logits / TEMPERATURE
        )

    raw_logits = (
        logits
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    calibrated_probabilities = (
        probabilities
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    expected_shape = (
        n,
        NUM_LABELS,
    )

    if raw_logits.shape != expected_shape:
        raise RuntimeError(
            f"{name}: invalid logits shape."
        )

    if (
        calibrated_probabilities.shape
        != expected_shape
    ):
        raise RuntimeError(
            f"{name}: invalid probability shape."
        )

    if not np.isfinite(
        calibrated_probabilities
    ).all():
        raise RuntimeError(
            f"{name}: probability NaN/Inf."
        )

    np.save(
        logits_path,
        raw_logits,
    )

    np.save(
        probabilities_path,
        calibrated_probabilities,
    )

    ids_path.write_text(
        "\n".join(
            eval_ids
        )
        + "\n"
    )

    metadata = {
        "protocol":
            "sequence_degradation_v1",
        "condition":
            name,
        "mask_fraction":
            fraction,
        "global_seed":
            GLOBAL_SEED,
        "proteins":
            n,
        "labels":
            NUM_LABELS,
        "esm_model":
            MODEL_NAME,
        "esm_revision":
            MODEL_REVISION,
        "chunk_size_original_positions":
            CHUNK_SIZE,
        "chunk_overlap":
            0,
        "embedding_dimension":
            EMBED_DIM,
        "embedding_dtype":
            "float32",
        "mask_token":
            tokenizer.mask_token,
        "mask_token_id":
            tokenizer.mask_token_id,
        "sequence_checkpoint_sha256":
            EXPECTED_CHECKPOINT_SHA,
        "sequence_checkpoint_epoch":
            int(
                checkpoint["epoch"]
            ),
        "temperature":
            TEMPERATURE,
        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA,
        "test_data_used":
            False,
        "status":
            "COMPLETE",
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Embeddings:",
        X.shape,
        X.dtype,
    )

    print(
        "Probabilities:",
        calibrated_probabilities.shape,
        calibrated_probabilities.dtype,
    )

    print(
        "Mean masked residues:",
        float(
            counts.mean()
        ),
    )

    print(
        "Embedding SHA256:",
        sha256(
            embeddings_path
        ),
    )

    print(
        "Probability SHA256:",
        sha256(
            probabilities_path
        ),
    )

    print()


print("=" * 80)
print("E7 EXTRACTION / INFERENCE COMPLETE")
print("TEST data used: NO")
print("=" * 80)
