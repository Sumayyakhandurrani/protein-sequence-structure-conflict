from pathlib import Path
import argparse
import hashlib
import os
import time

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
# PRODUCTION DRIVER
# ============================================================

EXPECTED_POPULATION_SIZE = 200
EXPECTED_TARGET_COUNT = 1770

OUT_DIR = (
    BASE
    / "xai/functional_sites_v1/"
      "explanation_stability_v1/"
      "per_protein/sequence"
)


def atomic_save_npz(
    path,
    **arrays,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp.npz"
    )

    np.savez_compressed(
        tmp,
        **arrays,
    )

    os.replace(
        tmp,
        path,
    )


def validate_existing_output(
    path,
    uid,
    expected_length,
    expected_labels,
):

    try:

        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            required = {
                "uniprot_id",
                "length",
                "positive_label_indices",
                "selected_base_position0",
                "clean_target_probability",
                "perturbed_target_probability",
                "absolute_probability_shift",
                "eligible",
                "spearman",
                "top10_jaccard",
            }

            if not required.issubset(
                set(z.files)
            ):
                return False

            if str(
                z["uniprot_id"].item()
            ) != uid:
                return False

            if int(
                z["length"].item()
            ) != expected_length:
                return False

            labels = np.asarray(
                z["positive_label_indices"],
                dtype=np.int32,
            )

            if not np.array_equal(
                labels,
                expected_labels,
            ):
                return False

            n = len(
                expected_labels
            )

            for key in [
                "selected_base_position0",
                "clean_target_probability",
                "perturbed_target_probability",
                "absolute_probability_shift",
                "eligible",
                "spearman",
                "top10_jaccard",
            ]:

                if np.asarray(
                    z[key]
                ).shape != (n,):
                    return False

            eligible = np.asarray(
                z["eligible"],
                dtype=bool,
            )

            selected = np.asarray(
                z["selected_base_position0"],
                dtype=np.int32,
            )

            shifts = np.asarray(
                z["absolute_probability_shift"],
                dtype=np.float64,
            )

            if not np.all(
                eligible
            ):
                return False

            if np.any(
                selected < 0
            ):
                return False

            if not np.isfinite(
                shifts
            ).all():
                return False

            if np.any(
                shifts
                > TOLERANCE + 1e-12
            ):
                return False

            jaccard = np.asarray(
                z["top10_jaccard"],
                dtype=np.float64,
            )

            if not np.isfinite(
                jaccard
            ).all():
                return False

            if np.any(
                (jaccard < 0.0)
                | (jaccard > 1.0)
            ):
                return False

            # Spearman may legitimately be NaN for a
            # degenerate explanation vector under the
            # frozen protocol. Do not reject NaN here.

    except Exception:
        return False

    return True


def select_base_positions(
    uid,
    clean_target_logits,
    clean_attribution,
    length,
):

    order = candidate_order(
        uid,
        length,
    )

    clean_p = sigmoid(
        clean_target_logits
        / TEMPERATURE
    )

    selected = np.full(
        len(clean_target_logits),
        -1,
        dtype=np.int32,
    )

    selected_p = np.full(
        len(clean_target_logits),
        np.nan,
        dtype=np.float64,
    )

    selected_shift = np.full(
        len(clean_target_logits),
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

        selected[newly] = int(
            pos
        )

        selected_p[newly] = (
            perturbed_p[newly]
        )

        selected_shift[newly] = (
            shift[newly]
        )

        if np.all(
            selected >= 0
        ):
            break

    return (
        selected,
        clean_p,
        selected_p,
        selected_shift,
    )


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

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise RuntimeError(
            "--limit must be positive."
        )


    print("=" * 78)
    print(
        "E10 — EXPLANATION STABILITY V1 "
        "— SEQUENCE PRODUCTION"
    )
    print("=" * 78)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Tolerance:",
        TOLERANCE,
    )

    print(
        "Temperature:",
        TEMPERATURE,
    )

    print()


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


    print(
        "Sequence checkpoint SHA: VERIFIED"
    )


    ids = [
        x.strip()
        for x in POPULATION_PATH
        .read_text()
        .splitlines()
        if x.strip()
    ]


    if len(ids) != EXPECTED_POPULATION_SIZE:
        raise RuntimeError(
            "Unexpected stability population size."
        )


    if len(set(ids)) != len(ids):
        raise RuntimeError(
            "Duplicate stability population IDs."
        )


    test = pd.read_pickle(
        TEST_PKL
    )

    test_ids = (
        test["uniprot_id"]
        .astype(str)
        .tolist()
    )

    row_by_id = {
        uid: i
        for i, uid in enumerate(
            test_ids
        )
    }


    total_targets = 0

    metadata = {}


    for uid in ids:

        if uid not in row_by_id:
            raise RuntimeError(
                f"{uid}: missing from TEST."
            )


        row = row_by_id[
            uid
        ]

        sequence = str(
            test.iloc[
                row
            ]["sequence"]
        ).strip().upper()


        attr_path = (
            ATTR_DIR
            / f"{uid}.npz"
        )


        if not attr_path.exists():
            raise RuntimeError(
                f"{uid}: missing frozen "
                "sequence attribution."
            )


        with np.load(
            attr_path,
            allow_pickle=False,
        ) as z:

            if str(
                z["uniprot_id"].item()
            ) != uid:
                raise RuntimeError(
                    f"{uid}: attribution ID mismatch."
                )

            length = int(
                z["length"].item()
            )

            labels = np.asarray(
                z["positive_label_indices"],
                dtype=np.int32,
            )


        if length != len(
            sequence
        ):
            raise RuntimeError(
                f"{uid}: attribution/sequence "
                "length mismatch."
            )


        if len(labels) <= 0:
            raise RuntimeError(
                f"{uid}: zero target labels."
            )


        total_targets += len(
            labels
        )


        metadata[
            uid
        ] = (
            sequence,
            labels,
        )


    if total_targets != EXPECTED_TARGET_COUNT:
        raise RuntimeError(
            "Unexpected stability target count: "
            f"{total_targets}"
        )


    print(
        "Population proteins:",
        len(ids),
    )

    print(
        "Population targets:",
        total_targets,
    )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    completed = []

    invalid_existing = []


    for uid in ids:

        sequence, labels = metadata[
            uid
        ]

        out_path = (
            OUT_DIR
            / f"{uid}.npz"
        )

        if not out_path.exists():
            continue


        if validate_existing_output(
            out_path,
            uid,
            len(sequence),
            labels,
        ):

            completed.append(
                uid
            )

        else:

            invalid_existing.append(
                uid
            )


    if invalid_existing:

        raise RuntimeError(
            "Invalid existing sequence stability "
            "outputs detected: "
            + ", ".join(
                invalid_existing[:10]
            )
        )


    completed_set = set(
        completed
    )


    todo = [
        uid
        for uid in ids
        if uid not in completed_set
    ]


    if args.limit is not None:

        todo = todo[
            :args.limit
        ]


    print(
        "Validated completed outputs:",
        len(completed),
    )

    print(
        "Remaining frozen population:",
        len(ids)
        - len(completed),
    )

    print(
        "Scheduled this invocation:",
        len(todo),
    )


    if args.validate_only:

        print()
        print("=" * 78)
        print("VALIDATION ONLY: PASS")
        print("=" * 78)

        print(
            "No ESM model loaded."
        )

        print(
            "No stability outcome generated."
        )

        return


    if not todo:

        print()
        print(
            "Nothing to process."
        )

        return


    # --------------------------------------------------------
    # Load frozen models once
    # --------------------------------------------------------

    print()
    print(
        "Loading frozen ESM..."
    )


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=True,
    )


    esm = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=True,
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


    if (
        checkpoint.get(
            "input_dim"
        ) != EMBED_DIM
        or checkpoint.get(
            "hidden_dim"
        ) != HIDDEN_DIM
        or checkpoint.get(
            "num_labels"
        ) != NUM_LABELS
    ):

        raise RuntimeError(
            "Unexpected sequence checkpoint "
            "architecture metadata."
        )


    mlp.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    mlp.eval()


    run_start = time.time()

    processed_this_run = 0


    # --------------------------------------------------------
    # Production loop
    # --------------------------------------------------------

    for number, uid in enumerate(
        todo,
        start=1,
    ):

        protein_start = time.time()


        sequence, expected_labels = (
            metadata[
                uid
            ]
        )

        L = len(
            sequence
        )


        attr_path = (
            ATTR_DIR
            / f"{uid}.npz"
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


        if not np.array_equal(
            positive_labels,
            expected_labels,
        ):

            raise RuntimeError(
                f"{uid}: target-label mismatch."
            )


        if clean_attribution.shape != (
            len(positive_labels),
            L,
        ):

            raise RuntimeError(
                f"{uid}: attribution shape mismatch."
            )


        # ----------------------------------------------------
        # Frozen first-qualifying base selection
        # ----------------------------------------------------

        (
            selected,
            clean_p,
            selected_p,
            selected_shift,
        ) = select_base_positions(
            uid,
            clean_target_logits,
            clean_attribution,
            L,
        )


        eligible = (
            selected >= 0
        )


        if not np.all(
            eligible
        ):

            raise RuntimeError(
                f"{uid}: unexpected ineligible "
                "sequence target."
            )


        # ----------------------------------------------------
        # Reproduce clean protein state
        # ----------------------------------------------------

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


        reproduced_targets = (
            reproduced_clean_logits[
                positive_labels
            ]
        )


        clean_diff = float(
            np.max(
                np.abs(
                    reproduced_targets
                    - clean_target_logits
                )
            )
        )


        if not np.allclose(
            reproduced_targets,
            clean_target_logits,
            rtol=1e-5,
            atol=1e-5,
        ):

            raise RuntimeError(
                f"{uid}: clean sequence logits "
                f"failed reproduction; "
                f"max_diff={clean_diff}"
            )


        # ----------------------------------------------------
        # Group labels by selected base
        # ----------------------------------------------------

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

        base_reproduction_diff = np.full(
            len(positive_labels),
            np.nan,
            dtype=np.float64,
        )


        for base_pos, target_rows in sorted(
            groups.items()
        ):

            group_labels = (
                positive_labels[
                    target_rows
                ]
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


            expected_base_logits = (
                clean_target_logits[
                    target_rows
                ]
                - clean_attribution[
                    target_rows,
                    base_pos
                ]
            )


            diffs = np.abs(
                base_target_logits
                - expected_base_logits
            )


            if not np.allclose(
                base_target_logits,
                expected_base_logits,
                rtol=1e-5,
                atol=1e-5,
            ):

                raise RuntimeError(
                    f"{uid}: base sequence "
                    "perturbation failed reproduction; "
                    f"base={base_pos + 1}, "
                    f"max_diff={float(diffs.max())}"
                )


            base_reproduction_diff[
                target_rows
            ] = diffs


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
                        f"{uid}: non-finite "
                        "perturbed explanation."
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


                intersection = (
                    clean_top
                    & perturbed_top
                )

                union = (
                    clean_top
                    | perturbed_top
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


        if not np.isfinite(
            jaccard_values
        ).all():

            raise RuntimeError(
                f"{uid}: non-finite Jaccard."
            )


        if not np.isfinite(
            base_reproduction_diff
        ).all():

            raise RuntimeError(
                f"{uid}: missing base "
                "reproduction QC."
            )


        out_path = (
            OUT_DIR
            / f"{uid}.npz"
        )


        atomic_save_npz(
            out_path,

            uniprot_id=np.asarray(
                uid
            ),

            length=np.asarray(
                L,
                dtype=np.int32,
            ),

            positive_label_indices=(
                positive_labels.astype(
                    np.int32,
                    copy=False,
                )
            ),

            selected_base_position0=(
                selected.astype(
                    np.int32,
                    copy=False,
                )
            ),

            clean_target_probability=(
                clean_p.astype(
                    np.float32
                )
            ),

            perturbed_target_probability=(
                selected_p.astype(
                    np.float32
                )
            ),

            absolute_probability_shift=(
                selected_shift.astype(
                    np.float32
                )
            ),

            eligible=(
                eligible.astype(
                    np.bool_
                )
            ),

            spearman=(
                spearman_values.astype(
                    np.float32
                )
            ),

            top10_jaccard=(
                jaccard_values.astype(
                    np.float32
                )
            ),

            clean_logit_max_abs_diff=(
                np.asarray(
                    clean_diff,
                    dtype=np.float32,
                )
            ),

            base_logit_abs_diff=(
                base_reproduction_diff.astype(
                    np.float32
                )
            ),
        )


        if not validate_existing_output(
            out_path,
            uid,
            L,
            positive_labels,
        ):

            raise RuntimeError(
                f"{uid}: saved output failed "
                "post-save validation."
            )


        processed_this_run += 1


        elapsed = (
            time.time()
            - protein_start
        )


        total_present = len(
            list(
                OUT_DIR.glob(
                    "*.npz"
                )
            )
        )


        print(
            f"[{number}/{len(todo)}] "
            f"{uid} "
            f"L={L} "
            f"K={len(positive_labels)} "
            f"bases={len(groups)} "
            f"finite_rho="
            f"{int(np.isfinite(spearman_values).sum())}/"
            f"{len(spearman_values)} "
            f"time={elapsed:.2f}s "
            f"present={total_present}",
            flush=True,
        )


    wall_time = (
        time.time()
        - run_start
    )


    print()
    print("=" * 78)
    print("SEQUENCE STABILITY INVOCATION COMPLETE")
    print("=" * 78)

    print(
        "Processed this invocation:",
        processed_this_run,
    )

    print(
        "Wall time seconds:",
        f"{wall_time:.2f}",
    )

    print(
        "Output directory:",
        OUT_DIR,
    )

    print(
        "Frozen model modified: NO"
    )

    print(
        "Frozen attribution files modified: NO"
    )


if __name__ == "__main__":
    main()
