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

BATCH_SIZE = 32

DEVICE = "cuda"


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


print("=" * 72)
print("E10 — BATCHED SEQUENCE OCCLUSION EQUIVALENCE")
print("=" * 72)

# ------------------------------------------------------------
# Load Q47149 sequence
# ------------------------------------------------------------

test = pd.read_pickle(
    "data/splits/mmseqs30_graph_v2/test.pkl"
)

row = test[
    test["uniprot_id"].astype(str) == UID
]

if len(row) != 1:
    raise RuntimeError(
        f"Expected one {UID} row."
    )

sequence = str(
    row.iloc[0]["sequence"]
).strip().upper()

L = len(sequence)

if L != 92:
    raise RuntimeError(
        f"Expected length 92, got {L}."
    )

print("Protein:", UID)
print("Length:", L)
print("Batch size:", BATCH_SIZE)


# ------------------------------------------------------------
# Load previous one-at-a-time result
# ------------------------------------------------------------

reference = pd.read_csv(
    "data/xai/functional_sites_v1/"
    "sanity_q47149/"
    "sequence_occlusion_q47149.csv"
)

positive_labels = np.sort(
    reference["label_index"]
    .unique()
    .astype(int)
)

if len(positive_labels) != 13:
    raise RuntimeError(
        "Expected 13 reference labels."
    )


# ------------------------------------------------------------
# Load frozen ESM
# ------------------------------------------------------------

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

mask_id = tokenizer.mask_token_id

if mask_id is None:
    raise RuntimeError(
        "No mask token ID."
    )


# ------------------------------------------------------------
# Tokenize clean sequence
# ------------------------------------------------------------

tokens = tokenizer(
    sequence,
    return_tensors="pt",
    add_special_tokens=True,
    padding=False,
    truncation=False,
)

clean_ids = tokens[
    "input_ids"
]

if clean_ids.shape != (
    1,
    L + 2,
):
    raise RuntimeError(
        f"Unexpected token shape: {clean_ids.shape}"
    )


# ------------------------------------------------------------
# Load frozen MLP
# ------------------------------------------------------------

model = SequenceMLP().to(
    DEVICE
)

checkpoint = torch.load(
    "data/models/sequence_mlp_v1/best_model.pt",
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ------------------------------------------------------------
# Batched residue masking
# ------------------------------------------------------------

print()
print("Running batched masking...")

all_masked_logits = np.empty(
    (
        L,
        NUM_LABELS,
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

    positions = list(
        range(
            start,
            end,
        )
    )

    B = len(
        positions
    )

    batch_ids = (
        clean_ids
        .repeat(
            B,
            1,
        )
    )

    for j, residue_index0 in enumerate(
        positions
    ):

        batch_ids[
            j,
            residue_index0 + 1,
        ] = mask_id


    batch_ids = batch_ids.to(
        DEVICE
    )

    attention_mask = torch.ones_like(
        batch_ids,
        device=DEVICE,
    )


    with torch.inference_mode():

        output = esm(
            input_ids=batch_ids,
            attention_mask=attention_mask,
        )

        residue_hidden = (
            output.last_hidden_state[
                :,
                1:-1,
                :
            ]
        )

        if residue_hidden.shape != (
            B,
            L,
            EMBED_DIM,
        ):
            raise RuntimeError(
                "Unexpected hidden-state shape: "
                f"{tuple(residue_hidden.shape)}"
            )

        embeddings = (
            residue_hidden
            .mean(dim=1)
        )

        logits = model(
            embeddings
        )


    all_masked_logits[
        start:end
    ] = (
        logits
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    print(
        f"  residues {start + 1:3d}"
        f"-{end:3d}"
    )


# ------------------------------------------------------------
# Compare against frozen one-at-a-time attribution output
# ------------------------------------------------------------

max_masked_logit_diff = 0.0
max_logit_drop_diff = 0.0

sum_masked_diff = 0.0
sum_drop_diff = 0.0

n = 0


for _, r in reference.iterrows():

    residue_position = int(
        r["residue_position"]
    )

    label_index = int(
        r["label_index"]
    )

    new_masked_logit = float(
        all_masked_logits[
            residue_position - 1,
            label_index,
        ]
    )

    old_masked_logit = float(
        r["masked_logit"]
    )

    clean_logit = float(
        r["clean_logit"]
    )

    new_drop = (
        clean_logit
        - new_masked_logit
    )

    old_drop = float(
        r["logit_drop"]
    )


    d1 = abs(
        new_masked_logit
        - old_masked_logit
    )

    d2 = abs(
        new_drop
        - old_drop
    )


    max_masked_logit_diff = max(
        max_masked_logit_diff,
        d1,
    )

    max_logit_drop_diff = max(
        max_logit_drop_diff,
        d2,
    )

    sum_masked_diff += d1
    sum_drop_diff += d2
    n += 1


mean_masked_diff = (
    sum_masked_diff / n
)

mean_drop_diff = (
    sum_drop_diff / n
)


print()
print("=" * 72)
print("EQUIVALENCE RESULTS")
print("=" * 72)

print(
    "Compared attribution rows:",
    n,
)

print()
print(
    "Masked-logit max abs difference:",
    max_masked_logit_diff,
)

print(
    "Masked-logit mean abs difference:",
    mean_masked_diff,
)

print()
print(
    "Logit-drop max abs difference:",
    max_logit_drop_diff,
)

print(
    "Logit-drop mean abs difference:",
    mean_drop_diff,
)


# Small floating-point differences from GPU batching are expected.
PASS_TOL = 1e-4

passed = (
    max_masked_logit_diff
    <= PASS_TOL
    and
    max_logit_drop_diff
    <= PASS_TOL
)


print()
print(
    "Required maximum difference <=",
    PASS_TOL,
)

print()
print(
    "STATUS:",
    "PASS" if passed else "FAIL",
)


if not passed:

    raise RuntimeError(
        "Batched implementation is not "
        "numerically equivalent to the "
        "one-at-a-time frozen implementation."
    )


print()
print(
    "Scientific XAI method changed: NO"
)

print(
    "Only computational batching tested."
)

print(
    "No population XAI results generated."
)

print("=" * 72)
