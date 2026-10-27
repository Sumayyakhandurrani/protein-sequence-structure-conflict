from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


UID = "Q47149"

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MODEL_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"

EMBED_DIM = 480
HIDDEN_DIM = 512
NUM_LABELS = 494
DROPOUT = 0.20
CHUNK_SIZE = 1000

TEMPERATURE = 1.106977105140686

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLIT = Path("data/splits/mmseqs30_graph_v2/test.pkl")

IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/test_uniprot_ids.txt"
)

SAVED_EMBEDDINGS = Path(
    "data/embeddings/esm2_t12_35M/test/test_embeddings.npy"
)

SAVED_LOGITS = Path(
    "data/final_evaluation/"
    "sequence_expert_v1_test/test_raw_logits.npy"
)

SAVED_PROBS = Path(
    "data/final_evaluation/"
    "sequence_expert_v1_test/test_calibrated_probabilities.npy"
)

CHECKPOINT = Path(
    "data/models/sequence_mlp_v1/best_model.pt"
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


def embed_chunk(sequence, tokenizer, esm):

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output = esm(**inputs)

    residues = output.last_hidden_state[
        0, 1:-1, :
    ]

    if residues.shape[0] != len(sequence):
        raise RuntimeError(
            f"Residue/token mismatch: "
            f"{len(sequence)} vs {residues.shape[0]}"
        )

    return (
        residues.mean(dim=0)
        .cpu()
        .float()
        .numpy()
    )


def embed_protein(sequence, tokenizer, esm):

    weighted_sum = np.zeros(
        EMBED_DIM,
        dtype=np.float64,
    )

    total = 0

    for start in range(
        0,
        len(sequence),
        CHUNK_SIZE,
    ):

        chunk = sequence[
            start:start + CHUNK_SIZE
        ]

        emb = embed_chunk(
            chunk,
            tokenizer,
            esm,
        )

        weighted_sum += (
            emb.astype(np.float64)
            * len(chunk)
        )

        total += len(chunk)

    return (
        weighted_sum / total
    ).astype(np.float32)


print("=" * 72)
print("E10 — Q47149 CLEAN SEQUENCE REPRODUCTION")
print("=" * 72)

if DEVICE != "cuda":
    raise RuntimeError(
        "CUDA is required for this reproduction check."
    )

df = pd.read_pickle(SPLIT)

ids = [
    x.strip()
    for x in IDS_PATH.read_text().splitlines()
    if x.strip()
]

if df["uniprot_id"].astype(str).tolist() != ids:
    raise RuntimeError(
        "TEST PKL/ID ordering mismatch."
    )

row_index = ids.index(UID)

sequence = str(
    df.iloc[row_index]["sequence"]
).strip().upper()

print("UniProt:", UID)
print("TEST row:", row_index)
print("Sequence length:", len(sequence))
print("Device:", DEVICE)

stored_embeddings = np.load(
    SAVED_EMBEDDINGS
)

stored_logits = np.load(
    SAVED_LOGITS
)

stored_probs = np.load(
    SAVED_PROBS
)

print()
print("Loading exact frozen ESM revision...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
)

esm = AutoModel.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
).to(DEVICE)

esm.eval()

print("Computing clean embedding...")

new_embedding = embed_protein(
    sequence,
    tokenizer,
    esm,
)

stored_embedding = stored_embeddings[
    row_index
].astype(np.float32)

embedding_abs_diff = np.abs(
    new_embedding - stored_embedding
)

print()
print("Embedding comparison:")
print(
    "  max abs difference:",
    float(embedding_abs_diff.max()),
)
print(
    "  mean abs difference:",
    float(embedding_abs_diff.mean()),
)
print(
    "  allclose rtol=1e-5 atol=1e-6:",
    bool(
        np.allclose(
            new_embedding,
            stored_embedding,
            rtol=1e-5,
            atol=1e-6,
        )
    ),
)


print()
print("Loading frozen sequence MLP...")

model = SequenceMLP().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

with torch.inference_mode():

    x = torch.from_numpy(
        new_embedding[None, :]
    ).float().to(DEVICE)

    new_logits = model(x)[0]

    new_probs = torch.sigmoid(
        new_logits / TEMPERATURE
    )

new_logits = (
    new_logits.cpu().numpy()
    .astype(np.float32)
)

new_probs = (
    new_probs.cpu().numpy()
    .astype(np.float32)
)

stored_logit = stored_logits[
    row_index
].astype(np.float32)

stored_prob = stored_probs[
    row_index
].astype(np.float32)

logit_diff = np.abs(
    new_logits - stored_logit
)

prob_diff = np.abs(
    new_probs - stored_prob
)

print()
print("Logit comparison:")
print(
    "  max abs difference:",
    float(logit_diff.max()),
)
print(
    "  mean abs difference:",
    float(logit_diff.mean()),
)
print(
    "  allclose rtol=1e-5 atol=1e-6:",
    bool(
        np.allclose(
            new_logits,
            stored_logit,
            rtol=1e-5,
            atol=1e-6,
        )
    ),
)

print()
print("Probability comparison:")
print(
    "  max abs difference:",
    float(prob_diff.max()),
)
print(
    "  mean abs difference:",
    float(prob_diff.mean()),
)
print(
    "  allclose rtol=1e-5 atol=1e-6:",
    bool(
        np.allclose(
            new_probs,
            stored_prob,
            rtol=1e-5,
            atol=1e-6,
        )
    ),
)

print()
print("=" * 72)

if not np.allclose(
    new_embedding,
    stored_embedding,
    rtol=1e-5,
    atol=1e-6,
):
    print("STATUS: FAIL — embedding reproduction mismatch")

elif not np.allclose(
    new_logits,
    stored_logit,
    rtol=1e-5,
    atol=1e-6,
):
    print("STATUS: FAIL — logit reproduction mismatch")

elif not np.allclose(
    new_probs,
    stored_prob,
    rtol=1e-5,
    atol=1e-6,
):
    print("STATUS: FAIL — probability reproduction mismatch")

else:
    print("STATUS: PASS")
    print(
        "Clean Q47149 reproduces the frozen "
        "sequence expert."
    )

print("=" * 72)

print()
print("Attribution computed: NO")
print("Model modified: NO")
print("Benchmark modified: NO")