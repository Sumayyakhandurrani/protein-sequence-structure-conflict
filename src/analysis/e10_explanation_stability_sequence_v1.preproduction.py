from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr
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

GLOBAL_SEED = 42
TOLERANCE = 0.01

EXPECTED_CHECKPOINT_SHA = (
    "1a2abcc66553d4906e111496f98b3645"
    "80c4fd29f9d8a2ad1e181b223e89022a"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PATHS
# ============================================================

BASE = Path("data")

POPULATION_PATH = (
    BASE
    / "xai/functional_sites_v1/"
      "explanation_stability_v1/"
      "stability_population_ids.txt"
)

TEST_PKL = (
    BASE
    / "splits/mmseqs30_graph_v2/test.pkl"
)

CHECKPOINT_PATH = (
    BASE
    / "models/sequence_mlp_v1/best_model.pt"
)

ATTR_DIR = (
    BASE
    / "xai/functional_sites_v1/"
      "sequence_attributions_v1/per_protein"
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


def sigmoid(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return (
        1.0
        / (1.0 + np.exp(-x))
    )


def candidate_order(
    uid,
    length,
):

    key = (
        f"explanation_stability_v1|"
        f"{GLOBAL_SEED}|{uid}|sequence"
    ).encode("utf-8")

    seed = int.from_bytes(
        hashlib.sha256(
            key
        ).digest()[:8],
        byteorder="little",
        signed=False,
    )

    return np.random.default_rng(
        seed
    ).permutation(length)


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
        logits = model(x)

    return (
        logits
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def chunk_token_ids(
    chunk,
    tokenizer,
):

    tokens = tokenizer(
        chunk,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )

    ids = (
        tokens["input_ids"]
        .cpu()
    )

    if ids.shape != (
        1,
        len(chunk) + 2,
    ):
        raise RuntimeError(
            "ESM token/residue mismatch."
        )

    return ids


def embedding_from_ids(
    ids,
    residue_length,
    esm,
):

    ids = ids.to(DEVICE)

    attention_mask = torch.ones_like(
        ids,
        device=DEVICE,
    )

    with torch.inference_mode():

        out = esm(
            input_ids=ids,
            attention_mask=attention_mask,
        )

    hidden = (
        out.last_hidden_state[
            :,
            1:-1,
            :
        ]
    )

    if hidden.shape[1:] != (
        residue_length,
        EMBED_DIM,
    ):
        raise RuntimeError(
            "Unexpected ESM hidden shape."
        )

    return (
        hidden
        .mean(dim=1)
        .cpu()
        .float()
        .numpy()
        .astype(np.float32)
    )


def clean_chunks_and_embedding(
    sequence,
    tokenizer,
    esm,
):

    chunks = []
    chunk_embeddings = []

    weighted = np.zeros(
        EMBED_DIM,
        dtype=np.float64,
    )

    for start in range(
        0,
        len(sequence),
        CHUNK_SIZE,
    ):

        chunk = sequence[
            start:start + CHUNK_SIZE
        ]

        ids = chunk_token_ids(
            chunk,
            tokenizer,
        )

        emb = embedding_from_ids(
            ids,
            len(chunk),
            esm,
        )[0]

        chunks.append(
            (
                start,
                chunk,
                ids,
            )
        )

        chunk_embeddings.append(
            emb
        )

        weighted += (
            emb.astype(np.float64)
            * len(chunk)
        )

    protein_embedding = (
        weighted / len(sequence)
    ).astype(np.float32)

    return (
        protein_embedding,
        chunks,
        chunk_embeddings,
    )


def base_protein_state(
    sequence,
    base_position0,
    tokenizer,
    esm,
    clean_chunks,
    clean_chunk_embeddings,
):

    mask_id = tokenizer.mask_token_id

    if mask_id is None:
        raise RuntimeError(
            "Tokenizer has no mask token."
        )

    base_chunks = [
        x.copy()
        for x in clean_chunk_embeddings
    ]

    base_chunk_index = (
        base_position0 // CHUNK_SIZE
    )

    base_chunk_start, base_chunk, clean_ids = (
        clean_chunks[
            base_chunk_index
        ]
    )

    local_base = (
        base_position0
        - base_chunk_start
    )

    base_ids = clean_ids.clone()

    base_ids[
        0,
        local_base + 1,
    ] = mask_id

    base_chunk_embedding = (
        embedding_from_ids(
            base_ids,
            len(base_chunk),
            esm,
        )[0]
    )

    base_chunks[
        base_chunk_index
    ] = base_chunk_embedding

    weighted = np.zeros(
        EMBED_DIM,
        dtype=np.float64,
    )

    for (
        (_, chunk, _),
        emb,
    ) in zip(
        clean_chunks,
        base_chunks,
    ):

        weighted += (
            emb.astype(np.float64)
            * len(chunk)
        )

    base_embedding = (
        weighted / len(sequence)
    ).astype(np.float32)

    return (
        base_embedding,
        base_chunks,
        base_chunk_index,
        local_base,
    )


def perturbed_explanation_for_group(
    sequence,
    base_position0,
    target_labels,
    tokenizer,
    esm,
    mlp,
    clean_chunks,
    clean_chunk_embeddings,
):

    (
        base_embedding,
        base_chunks,
        base_chunk_index,
        local_base,
    ) = base_protein_state(
        sequence,
        base_position0,
        tokenizer,
        esm,
        clean_chunks,
        clean_chunk_embeddings,
    )

    base_logits_all = (
        mlp_logits(
            base_embedding[None, :],
            mlp,
        )[0]
    )

    base_target_logits = (
        base_logits_all[
            target_labels
        ]
    )

    K = len(target_labels)
    L = len(sequence)

    attribution = np.full(
        (K, L),
        np.nan,
        dtype=np.float32,
    )

    mask_id = tokenizer.mask_token_id

    for chunk_index, (
        chunk_start,
        chunk,
        clean_ids,
    ) in enumerate(
        clean_chunks
    ):

        chunk_length = len(chunk)

        candidate_local_positions = [
            p
            for p in range(
                chunk_length
            )
            if (
                chunk_start + p
                != base_position0
            )
        ]

        for batch_start in range(
            0,
            len(candidate_local_positions),
            BATCH_SIZE,
        ):

            local_batch = (
                candidate_local_positions[
                    batch_start:
                    batch_start + BATCH_SIZE
                ]
            )

            B = len(
                local_batch
            )

            if B == 0:
                continue

            batch_ids = clean_ids.repeat(
                B,
                1,
            )

            # Apply frozen base perturbation whenever
            # this is the base-containing chunk.
            if chunk_index == base_chunk_index:

                batch_ids[
                    :,
                    local_base + 1,
                ] = mask_id

            # Apply second perturbation.
            for j, local_pos in enumerate(
                local_batch
            ):

                batch_ids[
                    j,
                    local_pos + 1,
                ] = mask_id

            second_chunk_embeddings = (
                embedding_from_ids(
                    batch_ids,
                    chunk_length,
                    esm,
                )
            )

            # Convert each changed chunk embedding into
            # a whole-protein embedding relative to the
            # already base-perturbed protein.
            protein_embeddings = np.repeat(
                base_embedding[None, :],
                B,
                axis=0,
            )

            delta = (
                second_chunk_embeddings
                - base_chunks[
                    chunk_index
                ][None, :]
            )

            protein_embeddings += (
                delta
                * (
                    chunk_length
                    / len(sequence)
                )
            ).astype(np.float32)

            double_logits = mlp_logits(
                protein_embeddings,
                mlp,
            )

            double_target_logits = (
                double_logits[
                    :,
                    target_labels
                ]
            )

            drops = (
                base_target_logits[
                    None,
                    :
                ]
                - double_target_logits
            )

            for j, local_pos in enumerate(
                local_batch
            ):

                global_pos = (
                    chunk_start
                    + local_pos
                )

                attribution[
                    :,
                    global_pos
                ] = drops[
                    j,
                    :
                ]

    return (
        attribution,
        base_target_logits,
    )


def top_fraction_indices(
    values,
    original_positions,
):

    n = len(values)

    k = max(
        1,
        int(np.ceil(0.10 * n)),
    )

    # Primary rank:
    # descending signed attribution.
    #
    # Tie-break:
    # lower original residue position.
    order = np.lexsort(
        (
            original_positions,
            -values,
        )
    )

    return set(
        original_positions[
            order[:k]
        ].tolist()
    )


# ============================================================
# SMOKE TEST
# ============================================================

if DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA is required."
    )


if sha256_file(
    CHECKPOINT_PATH
) != EXPECTED_CHECKPOINT_SHA:

    raise RuntimeError(
        "Sequence checkpoint SHA mismatch."
    )


ids = [
    x.strip()
    for x in POPULATION_PATH
    .read_text()
    .splitlines()
    if x.strip()
]

uid = ids[0]

test = pd.read_pickle(
    TEST_PKL
)

row = test.loc[
    test["uniprot_id"].astype(str)
    == uid
]

if len(row) != 1:
    raise RuntimeError(
        "Could not uniquely locate smoke protein."
    )

sequence = str(
    row.iloc[0]["sequence"]
).strip().upper()


attr_path = (
    ATTR_DIR / f"{uid}.npz"
)

with np.load(
    attr_path,
    allow_pickle=False,
) as z:

    positive_labels = np.asarray(
        z["positive_label_indices"],
        dtype=np.int32,
    )

    clean_target_logits = np.asarray(
        z["clean_target_logits"],
        dtype=np.float64,
    )

    clean_attribution = np.asarray(
        z["attribution_logit_drop"],
        dtype=np.float64,
    )


L = len(sequence)

if clean_attribution.shape != (
    len(positive_labels),
    L,
):
    raise RuntimeError(
        "Reference attribution shape mismatch."
    )


# ------------------------------------------------------------
# Frozen candidate selection from exact existing
# one-residue counterfactuals
# ------------------------------------------------------------

order = candidate_order(
    uid,
    L,
)

clean_p = sigmoid(
    clean_target_logits
    / TEMPERATURE
)

selected = np.full(
    len(positive_labels),
    -1,
    dtype=np.int32,
)

selected_p = np.full(
    len(positive_labels),
    np.nan,
    dtype=np.float64,
)


for pos in order:

    perturbed_logits = (
        clean_target_logits
        - clean_attribution[:, pos]
    )

    perturbed_p = sigmoid(
        perturbed_logits
        / TEMPERATURE
    )

    shift = np.abs(
        perturbed_p
        - clean_p
    )

    newly = (
        (selected < 0)
        & (shift <= TOLERANCE)
    )

    selected[newly] = int(pos)
    selected_p[newly] = (
        perturbed_p[newly]
    )

    if np.all(
        selected >= 0
    ):
        break


if not np.all(
    selected >= 0
):
    raise RuntimeError(
        "Smoke protein has ineligible target."
    )


print("=" * 72)
print("SEQUENCE STABILITY V1 — ONE-PROTEIN SMOKE TEST")
print("=" * 72)

print("Device:", DEVICE)
print("Protein:", uid)
print("Length:", L)
print("Targets:", len(positive_labels))

print(
    "Selected base positions (1-based):",
    (selected + 1).tolist(),
)

print(
    "Unique bases:",
    len(np.unique(selected)),
)


# ------------------------------------------------------------
# Load exact frozen models
# ------------------------------------------------------------

print()
print("Loading frozen ESM...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    local_files_only=True,
)

esm = AutoModel.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    local_files_only=True,
).to(DEVICE)

esm.eval()


print("Loading frozen sequence MLP...")

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


# ------------------------------------------------------------
# Reproduce clean embedding once
# ------------------------------------------------------------

print()
print("Computing clean chunk state...")

(
    reproduced_clean_embedding,
    clean_chunks,
    clean_chunk_embeddings,
) = clean_chunks_and_embedding(
    sequence,
    tokenizer,
    esm,
)


reproduced_clean_logits = (
    mlp_logits(
        reproduced_clean_embedding[
            None,
            :
        ],
        mlp,
    )[0]
)


clean_target_reproduced = (
    reproduced_clean_logits[
        positive_labels
    ]
)


clean_logit_max_diff = float(
    np.max(
        np.abs(
            clean_target_reproduced
            - clean_target_logits
        )
    )
)


print(
    "Clean target-logit max abs diff:",
    f"{clean_logit_max_diff:.3e}",
)


if not np.allclose(
    clean_target_reproduced,
    clean_target_logits,
    rtol=1e-5,
    atol=1e-5,
):
    raise RuntimeError(
        "Clean target logits failed reproduction."
    )


# ------------------------------------------------------------
# Group targets by selected base residue
# ------------------------------------------------------------

groups = {}

for target_row, base_pos in enumerate(
    selected
):

    groups.setdefault(
        int(base_pos),
        [],
    ).append(
        target_row
    )


spearman_values = np.full(
    len(positive_labels),
    np.nan,
    dtype=np.float64,
)

jaccard_values = np.full(
    len(positive_labels),
    np.nan,
    dtype=np.float64,
)

base_reproduction_diffs = []


for group_number, (
    base_pos,
    target_rows,
) in enumerate(
    sorted(groups.items()),
    start=1,
):

    group_labels = (
        positive_labels[
            target_rows
        ]
    )

    print()
    print(
        f"Base group {group_number}/{len(groups)}:"
    )

    print(
        "  residue:",
        base_pos + 1,
    )

    print(
        "  assigned targets:",
        len(target_rows),
    )


    (
        perturbed_attr,
        base_target_logits,
    ) = perturbed_explanation_for_group(
        sequence,
        base_pos,
        group_labels,
        tokenizer,
        esm,
        mlp,
        clean_chunks,
        clean_chunk_embeddings,
    )


    # Existing frozen single-residue attribution tells
    # us what the base-perturbed target logits must be.
    expected_base_logits = (
        clean_target_logits[
            target_rows
        ]
        - clean_attribution[
            target_rows,
            base_pos
        ]
    )


    base_diff = float(
        np.max(
            np.abs(
                base_target_logits
                - expected_base_logits
            )
        )
    )

    base_reproduction_diffs.append(
        base_diff
    )


    print(
        "  base-logit reproduction max diff:",
        f"{base_diff:.3e}",
    )


    if not np.allclose(
        base_target_logits,
        expected_base_logits,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Base perturbation failed reproduction."
        )


    comparison_mask = np.ones(
        L,
        dtype=bool,
    )

    comparison_mask[
        base_pos
    ] = False

    original_positions = np.arange(
        L,
        dtype=np.int32,
    )[
        comparison_mask
    ]


    for local_row, target_row in enumerate(
        target_rows
    ):

        clean_vec = (
            clean_attribution[
                target_row,
                comparison_mask
            ]
        )

        perturbed_vec = (
            perturbed_attr[
                local_row,
                comparison_mask
            ]
        )


        if not np.isfinite(
            perturbed_vec
        ).all():

            raise RuntimeError(
                "Non-finite perturbed explanation."
            )


        rho = spearmanr(
            clean_vec,
            perturbed_vec,
        ).statistic


        clean_top = top_fraction_indices(
            clean_vec,
            original_positions,
        )

        perturbed_top = top_fraction_indices(
            perturbed_vec,
            original_positions,
        )

        union = (
            clean_top
            | perturbed_top
        )

        intersection = (
            clean_top
            & perturbed_top
        )

        jac = (
            len(intersection)
            / len(union)
        )


        spearman_values[
            target_row
        ] = rho

        jaccard_values[
            target_row
        ] = jac


print()
print("=" * 72)
print("SMOKE TEST RESULT")
print("=" * 72)

print(
    "Targets evaluated:",
    len(positive_labels),
)

print(
    "Finite Spearman:",
    int(
        np.isfinite(
            spearman_values
        ).sum()
    ),
)

print(
    "Mean Spearman:",
    f"{np.nanmean(spearman_values):.6f}",
)

print(
    "Median Spearman:",
    f"{np.nanmedian(spearman_values):.6f}",
)

print(
    "Mean top-10% Jaccard:",
    f"{np.nanmean(jaccard_values):.6f}",
)

print(
    "Median top-10% Jaccard:",
    f"{np.nanmedian(jaccard_values):.6f}",
)

print(
    "Maximum base-logit reproduction diff:",
    f"{max(base_reproduction_diffs):.3e}",
)

print()
print(
    "Smoke test only: no stability files written."
)
print(
    "Frozen models modified: NO"
)
print(
    "Frozen attribution files modified: NO"
)
