from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

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

BASE = Path("data")

TEST_PKL = (
    BASE
    / "splits/mmseqs30_graph_v2/test.pkl"
)

TEST_IDS_PATH = (
    BASE
    / "splits/mmseqs30_graph_v2/labels/test_uniprot_ids.txt"
)

TEST_LABELS_PATH = (
    BASE
    / "splits/mmseqs30_graph_v2/labels/test_labels.npy"
)

LABEL_MAP_PATH = (
    BASE
    / "splits/mmseqs30_graph_v2/primary_go_mf_labels.csv"
)

CHECKPOINT_PATH = (
    BASE
    / "models/sequence_mlp_v1/best_model.pt"
)

SAVED_LOGITS_PATH = (
    BASE
    / "final_evaluation/"
    "sequence_expert_v1_test/test_raw_logits.npy"
)

SITES_PATH = (
    BASE
    / "xai/functional_sites_v1/"
    "functional_sites_primary.csv"
)

OUT_DIR = (
    BASE
    / "xai/functional_sites_v1/"
    "sanity_q47149"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = (
    OUT_DIR
    / "sequence_occlusion_q47149.csv"
)


EXPECTED_CHECKPOINT_SHA = (
    "1a2abcc66553d4906e111496f98b3645"
    "80c4fd29f9d8a2ad1e181b223e89022a"
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


def embed_chunk(
    sequence,
    tokenizer,
    esm,
):

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

        output = esm(
            **inputs
        )

    residues = (
        output.last_hidden_state[
            0,
            1:-1,
            :
        ]
    )

    if residues.shape[0] != len(sequence):

        raise RuntimeError(
            "Residue/token mismatch: "
            f"{len(sequence)} vs "
            f"{residues.shape[0]}"
        )

    return (
        residues
        .mean(dim=0)
        .cpu()
        .float()
        .numpy()
    )


def embed_protein(
    sequence,
    tokenizer,
    esm,
):

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


def predict_logits(
    embedding,
    model,
):

    x = torch.from_numpy(
        embedding[None, :]
    ).float().to(DEVICE)

    with torch.inference_mode():

        logits = model(x)[0]

    return (
        logits
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("E10 — Q47149 SEQUENCE RESIDUE OCCLUSION")
print("=" * 78)

print("UniProt:", UID)
print("Device:", DEVICE)
print("Primary attribution: signed raw-logit drop")
print("Perturbation: ESM <mask> at one residue")
print("Frozen sequence temperature:", TEMPERATURE)
print()


if DEVICE != "cuda":

    raise RuntimeError(
        "CUDA GPU is required."
    )


# ============================================================
# VERIFY CHECKPOINT
# ============================================================

actual_sha = sha256_file(
    CHECKPOINT_PATH
)

if actual_sha != EXPECTED_CHECKPOINT_SHA:

    raise RuntimeError(
        "Frozen sequence checkpoint SHA mismatch.\n"
        f"Expected: {EXPECTED_CHECKPOINT_SHA}\n"
        f"Actual:   {actual_sha}"
    )

print(
    "Frozen sequence checkpoint SHA256: VERIFIED"
)


# ============================================================
# LOAD TEST PROTEIN + LABELS
# ============================================================

df = pd.read_pickle(
    TEST_PKL
)

test_ids = [
    x.strip()
    for x in (
        TEST_IDS_PATH
        .read_text()
        .splitlines()
    )
    if x.strip()
]

if (
    df["uniprot_id"]
    .astype(str)
    .tolist()
    != test_ids
):

    raise RuntimeError(
        "TEST PKL/ID ordering mismatch."
    )

row_index = test_ids.index(
    UID
)

sequence = str(
    df.iloc[row_index]["sequence"]
).strip().upper()

length = len(sequence)

if length != 92:

    raise RuntimeError(
        f"Expected Q47149 length 92, got {length}."
    )


Y = np.load(
    TEST_LABELS_PATH
)

if Y.shape != (
    len(test_ids),
    NUM_LABELS,
):

    raise RuntimeError(
        f"Unexpected TEST label shape: {Y.shape}"
    )


positive_labels = np.flatnonzero(
    Y[row_index] == 1
)

if len(positive_labels) != 13:

    raise RuntimeError(
        "Expected 13 positive frozen labels "
        f"for Q47149, got {len(positive_labels)}."
    )


label_map = pd.read_csv(
    LABEL_MAP_PATH
).sort_values(
    "label_index"
).reset_index(
    drop=True
)

if not np.array_equal(
    label_map["label_index"].to_numpy(),
    np.arange(NUM_LABELS),
):

    raise RuntimeError(
        "Frozen GO label indexing mismatch."
    )


print("TEST row:", row_index)
print("Sequence length:", length)
print(
    "Positive frozen GO labels:",
    len(positive_labels),
)


# ============================================================
# CURATED SITE
# ============================================================

sites = pd.read_csv(
    SITES_PATH
)

qsites = sites[
    sites["uniprot_id"] == UID
].copy()

curated_positions = set()

for _, row in qsites.iterrows():

    start = int(
        row["start"]
    )

    end = int(
        row["end"]
    )

    for pos in range(
        start,
        end + 1,
    ):
        curated_positions.add(pos)


print(
    "Curated functional positions:",
    sorted(curated_positions),
)


# ============================================================
# LOAD FROZEN ESM
# ============================================================

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


mask_token = tokenizer.mask_token

if mask_token is None:

    raise RuntimeError(
        "Tokenizer has no mask token."
    )

print(
    "Tokenizer mask token:",
    repr(mask_token),
)


# ============================================================
# LOAD FROZEN MLP
# ============================================================

model = SequenceMLP().to(
    DEVICE
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Frozen sequence MLP loaded."
)


# ============================================================
# CLEAN PREDICTION
# ============================================================

print()
print("Computing clean prediction...")

clean_embedding = embed_protein(
    sequence,
    tokenizer,
    esm,
)

clean_logits = predict_logits(
    clean_embedding,
    model,
)


# ============================================================
# VERIFY AGAINST FROZEN TEST LOGITS
# ============================================================

saved_logits = np.load(
    SAVED_LOGITS_PATH
)

frozen_logits = (
    saved_logits[
        row_index
    ]
    .astype(np.float32)
)

max_logit_diff = float(
    np.max(
        np.abs(
            clean_logits
            - frozen_logits
        )
    )
)

logits_match = np.allclose(
    clean_logits,
    frozen_logits,
    rtol=1e-5,
    atol=1e-6,
)

print(
    "Clean/frozen max logit difference:",
    max_logit_diff,
)

print(
    "Clean/frozen logits allclose:",
    bool(logits_match),
)

if not logits_match:

    raise RuntimeError(
        "Clean sequence inference does not "
        "reproduce frozen TEST logits."
    )


# ============================================================
# OCCLUSION
#
# IMPORTANT:
# ESM's <mask> is a TOKEN, not a one-character amino acid.
#
# Therefore we cannot insert the literal string "<mask>"
# into the amino-acid sequence and then use len(sequence)
# as if it were one residue.
#
# We tokenize the clean protein first and replace exactly
# one residue token ID with mask_token_id.
# ============================================================

print()
print(
    "Running one-residue ESM mask occlusion..."
)


clean_inputs = tokenizer(
    sequence,
    return_tensors="pt",
    add_special_tokens=True,
    padding=False,
    truncation=False,
)

clean_input_ids = clean_inputs[
    "input_ids"
]

# Expected:
# BOS + 92 residues + EOS
expected_tokens = (
    length + 2
)

if clean_input_ids.shape != (
    1,
    expected_tokens,
):

    raise RuntimeError(
        "Unexpected clean token shape: "
        f"{tuple(clean_input_ids.shape)}"
    )


mask_token_id = (
    tokenizer.mask_token_id
)

if mask_token_id is None:

    raise RuntimeError(
        "Tokenizer mask_token_id is None."
    )


records = []

for residue_index0 in range(
    length
):

    residue_position = (
        residue_index0 + 1
    )

    input_ids = (
        clean_input_ids
        .clone()
        .to(DEVICE)
    )

    # +1 because token 0 is BOS.
    token_position = (
        residue_index0 + 1
    )

    original_token_id = int(
        input_ids[
            0,
            token_position
        ].item()
    )

    input_ids[
        0,
        token_position
    ] = mask_token_id


    attention_mask = torch.ones_like(
        input_ids,
        device=DEVICE,
    )


    with torch.inference_mode():

        output = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


    residue_hidden = (
        output.last_hidden_state[
            0,
            1:-1,
            :
        ]
    )

    if residue_hidden.shape != (
        length,
        EMBED_DIM,
    ):

        raise RuntimeError(
            f"Residue hidden shape mismatch "
            f"at position {residue_position}: "
            f"{tuple(residue_hidden.shape)}"
        )


    masked_embedding = (
        residue_hidden
        .mean(dim=0)
        .cpu()
        .float()
        .numpy()
        .astype(np.float32)
    )


    masked_logits = predict_logits(
        masked_embedding,
        model,
    )


    for label_index in positive_labels:

        label_index = int(
            label_index
        )

        clean_logit = float(
            clean_logits[
                label_index
            ]
        )

        masked_logit = float(
            masked_logits[
                label_index
            ]
        )

        logit_drop = (
            clean_logit
            - masked_logit
        )


        clean_prob = float(
            torch.sigmoid(
                torch.tensor(
                    clean_logit
                    / TEMPERATURE
                )
            ).item()
        )

        masked_prob = float(
            torch.sigmoid(
                torch.tensor(
                    masked_logit
                    / TEMPERATURE
                )
            ).item()
        )


        label_row = (
            label_map.iloc[
                label_index
            ]
        )


        records.append(
            {
                "uniprot_id":
                    UID,

                "residue_position":
                    residue_position,

                "residue":
                    sequence[
                        residue_index0
                    ],

                "original_token_id":
                    original_token_id,

                "is_curated_functional_site":
                    residue_position
                    in curated_positions,

                "label_index":
                    label_index,

                "go_id":
                    str(
                        label_row[
                            "go_id"
                        ]
                    ),

                "train_protein_count":
                    int(
                        label_row[
                            "train_protein_count"
                        ]
                    ),

                "clean_logit":
                    clean_logit,

                "masked_logit":
                    masked_logit,

                "logit_drop":
                    logit_drop,

                "clean_probability":
                    clean_prob,

                "masked_probability":
                    masked_prob,

                "probability_drop":
                    (
                        clean_prob
                        - masked_prob
                    ),
            }
        )


    if (
        residue_position % 10 == 0
        or residue_position == length
    ):

        print(
            f"  {residue_position:3d}/"
            f"{length} residues"
        )


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(
    records
)

expected_rows = (
    length
    * len(positive_labels)
)

if len(result) != expected_rows:

    raise RuntimeError(
        f"Expected {expected_rows} rows, "
        f"got {len(result)}."
    )

if not np.isfinite(
    result[
        [
            "clean_logit",
            "masked_logit",
            "logit_drop",
            "clean_probability",
            "masked_probability",
            "probability_drop",
        ]
    ].to_numpy()
).all():

    raise RuntimeError(
        "Non-finite attribution output."
    )


result.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# SANITY SUMMARY
# ============================================================

print()
print("=" * 78)
print("SEQUENCE OCCLUSION COMPLETE")
print("=" * 78)

print(
    "Residues evaluated:",
    result[
        "residue_position"
    ].nunique(),
)

print(
    "GO labels evaluated:",
    result[
        "label_index"
    ].nunique(),
)

print(
    "Protein-label-residue rows:",
    len(result),
)

print(
    "Curated position(s):",
    sorted(curated_positions),
)


print()
print(
    "Top residue by logit drop "
    "for each GO label:"
)

top_rows = (
    result
    .sort_values(
        [
            "label_index",
            "logit_drop",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .groupby(
        "label_index",
        as_index=False,
    )
    .first()
)

print(
    top_rows[
        [
            "label_index",
            "go_id",
            "residue_position",
            "residue",
            "logit_drop",
            "is_curated_functional_site",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Curated residue 87 ranks:"
)

ranked = result.copy()

ranked["rank"] = (
    ranked
    .groupby(
        "label_index"
    )["logit_drop"]
    .rank(
        method="min",
        ascending=False,
    )
)

site87 = ranked[
    ranked[
        "residue_position"
    ] == 87
]

print(
    site87[
        [
            "label_index",
            "go_id",
            "residue",
            "logit_drop",
            "rank",
        ]
    ].to_string(
        index=False
    )
)


print()
print("Saved:")
print(OUT_CSV)

print(
    "SHA256:",
    sha256_file(
        OUT_CSV
    ),
)

print()
print(
    "Model weights modified: NO"
)

print(
    "Benchmark modified: NO"
)

print(
    "Method selection performed: NO"
)

print(
    "This is an implementation "
    "sanity-check protein."
)

print("=" * 78)