from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# ============================================================
# IMPORT FROZEN COMPONENTS
# ============================================================

ANALYSIS_DIR = Path(__file__).resolve().parent
MODEL_CODE_DIR = Path(__file__).resolve().parents[1] / "models"

for p in (ANALYSIS_DIR, MODEL_CODE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e10_deletion_random_sets_v1 import (
    N_RANDOM_SETS,
    deletion_k,
    generate_random_sets,
    protein_seed,
    validate_random_sets,
)
from structure_gnn import StructureGNN


# ============================================================
# FROZEN CONSTANTS
# ============================================================

NUM_LABELS = 494
EPS = 1e-7

SEQ_TEMPERATURE = 1.106977105140686
STRUCTURE_TEMPERATURE = 1.0

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MODEL_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
EMBED_DIM = 480
SEQ_HIDDEN_DIM = 512
SEQ_DROPOUT = 0.20
CHUNK_SIZE = 1000
BATCH_SIZE = 8

NODE_DIM = 21
STRUCT_HIDDEN_DIM = 128
STRUCT_NUM_LAYERS = 2
STRUCT_DROPOUT = 0.10
EXPECTED_STRUCT_PARAMETER_COUNT = 298990

EMBEDDING_RTOL = 1e-5
EMBEDDING_ATOL = 1e-6
LOGIT_RTOL = 1e-5
LOGIT_ATOL = 1e-5

EXPECTED_TEST = 4754
EXPECTED_PAIRED = 4731
EXPECTED_POPULATION = 2447
EXPECTED_TARGETS = 23087
EXPECTED_RESIDUES = 1279145

EXPECTED_PROTOCOL_SHA256 = (
    "c62bb7952afeab57870718dff013068c11709b15262f094174b838b4a7329bf9"
)
EXPECTED_IMPLEMENTATION_SHA256 = (
    "ce6576a5662dcea187d0a450caf31948f4b988cdcb40594b81a40a6c0fe485a5"
)
EXPECTED_RANDOM_SHA256 = (
    "471f64b5c73ee51d61bb570ed179ff2a7047d998006b5f822fa3d1869933cc63"
)
EXPECTED_POPULATION_SHA256 = (
    "e69281eb8f1298c8ec6416d26849423ad43b507eb9ca7a6bb783f3c2e2adc76d"
)
# NOTE: The line above is deliberately NOT used for verification.
# The canonical XAI population hash is defined separately below.
EXPECTED_XAI_POPULATION_SHA256 = (
    "e69281eb8f1298c8ec0ea7b0498a6d0acf9fa7f4aa32d37354a9dc173c9e667c"
)
EXPECTED_SEQ_CHECKPOINT_SHA256 = (
    "1a2abcc66553d4906e111496f98b364580c4fd29f9d8a2ad1e181b223e89022a"
)
EXPECTED_STRUCT_CHECKPOINT_SHA256 = (
    "f1434b60a67e7dd695f289f7094d528bb91a28f249b28efd0d26af28d698991e"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PATHS
# ============================================================

TEST_PKL = Path("data/splits/mmseqs30_graph_v2/test.pkl")
TEST_IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/test_uniprot_ids.txt"
)
TEST_LABELS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/test_labels.npy"
)

POPULATION_PATH = Path(
    "data/xai/functional_sites_v1/xai_attribution_eligible_ids.txt"
)

SEQ_ATTR_DIR = Path(
    "data/xai/functional_sites_v1/sequence_attributions_v1/per_protein"
)
STRUCT_ATTR_DIR = Path(
    "data/xai/functional_sites_v1/structure_attributions_v1/per_protein"
)

SEQ_EMBEDDINGS_PATH = Path(
    "data/embeddings/esm2_t12_35M/test/test_embeddings.npy"
)
SEQ_RAW_LOGITS_PATH = Path(
    "data/final_evaluation/sequence_expert_v1_test/test_raw_logits.npy"
)
SEQ_PROBS_PATH = Path(
    "data/final_evaluation/sequence_expert_v1_test/test_calibrated_probabilities.npy"
)
SEQ_IDS_PATH = Path(
    "data/final_evaluation/sequence_expert_v1_test/test_uniprot_ids.txt"
)
SEQ_CHECKPOINT_PATH = Path(
    "data/models/sequence_mlp_v1/best_model.pt"
)

STRUCT_IDS_PATH = Path(
    "data/structures/structure_protocol_v1/test_structure_eligible_ids.txt"
)
STRUCT_RAW_LOGITS_PATH = Path(
    "data/final_evaluation/structure_expert_v1_test/test_raw_logits.npy"
)
STRUCT_PROBS_PATH = Path(
    "data/final_evaluation/structure_expert_v1_test/test_probabilities.npy"
)
STRUCT_CHECKPOINT_PATH = Path(
    "data/models/structure_gnn_v1_full/final_model.pt"
)
GRAPH_DIR = Path(
    "data/structures/structure_protocol_v1/graphs/test"
)

FROZEN_JS_PATH = Path(
    "data/final_evaluation/final_test_primary_v1/bernoulli_js.npy"
)
PAIRED_IDS_PATH = Path(
    "data/final_evaluation/final_test_primary_v1/paired_test_uniprot_ids.txt"
)

PROTOCOL_PATH = Path(
    "data/xai/functional_sites_v1/CONFLICT_DELETION_PROTOCOL_V1.txt"
)
IMPLEMENTATION_PATH = Path(
    "data/xai/functional_sites_v1/CONFLICT_DELETION_IMPLEMENTATION_V1.txt"
)
RANDOM_GENERATOR_PATH = Path(
    "src/analysis/e10_deletion_random_sets_v1.py"
)

OUT_ROOT = Path(
    "data/xai/functional_sites_v1/conflict_deletion_v1"
)
OUT_DIR = OUT_ROOT / "per_protein"
RUN_INFO_PATH = OUT_ROOT / "run_info.json"


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha(path, expected, name):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{name} SHA mismatch.\nExpected: {expected}\nActual:   {actual}"
        )
    print(f"{name}: SHA VERIFIED")
    return actual


def bernoulli_js(ps, pt):
    ps = np.clip(ps, EPS, 1.0 - EPS)
    pt = np.clip(pt, EPS, 1.0 - EPS)
    m = 0.5 * (ps + pt)

    return 0.5 * (
        ps * np.log(ps / m)
        + (1.0 - ps) * np.log((1.0 - ps) / (1.0 - m))
        + pt * np.log(pt / m)
        + (1.0 - pt) * np.log((1.0 - pt) / (1.0 - m))
    )


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def atomic_save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ============================================================
# SEQUENCE EXPERT
# ============================================================

class SequenceMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(EMBED_DIM, SEQ_HIDDEN_DIM),
            torch.nn.LayerNorm(SEQ_HIDDEN_DIM),
            torch.nn.GELU(),
            torch.nn.Dropout(SEQ_DROPOUT),
            torch.nn.Linear(SEQ_HIDDEN_DIM, NUM_LABELS),
        )

    def forward(self, x):
        return self.network(x)


def clean_chunk_embedding(chunk, tokenizer, esm):
    enc = tokenizer(
        chunk,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    if input_ids.shape[1] != len(chunk) + 2:
        raise RuntimeError("Unexpected ESM tokenized chunk length.")

    with torch.inference_mode():
        hidden = esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[0, 1:-1, :]

    return hidden.mean(dim=0).detach().cpu().numpy().astype(np.float32)


def clean_protein_embedding(sequence, tokenizer, esm):
    weighted = np.zeros(EMBED_DIM, dtype=np.float64)
    clean_chunks = []

    for start in range(0, len(sequence), CHUNK_SIZE):
        chunk = sequence[start:start + CHUNK_SIZE]
        emb = clean_chunk_embedding(chunk, tokenizer, esm)
        clean_chunks.append(emb)
        weighted += emb.astype(np.float64) * len(chunk)

    return (
        (weighted / float(len(sequence))).astype(np.float32),
        clean_chunks,
    )


def joint_mask_protein_embeddings(
    sequence,
    deletion_sets,
    clean_embedding,
    clean_chunk_embeddings,
    tokenizer,
    esm,
):
    deletion_sets = np.asarray(deletion_sets, dtype=np.int64)

    if deletion_sets.ndim != 2:
        raise RuntimeError("deletion_sets must be 2-D.")

    n_sets = deletion_sets.shape[0]
    L = len(sequence)

    if deletion_sets.size:
        if deletion_sets.min() < 0 or deletion_sets.max() >= L:
            raise RuntimeError("Sequence deletion position out of bounds.")

    result = np.repeat(
        clean_embedding[None, :].astype(np.float64),
        n_sets,
        axis=0,
    )

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise RuntimeError("ESM tokenizer has no mask token.")

    chunk_i = 0

    for start in range(0, L, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, L)
        chunk = sequence[start:end]
        chunk_len = end - start
        clean_chunk = clean_chunk_embeddings[chunk_i].astype(np.float64)

        affected = np.any(
            (deletion_sets >= start) & (deletion_sets < end),
            axis=1,
        )
        affected_indices = np.flatnonzero(affected)

        if len(affected_indices):
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                add_special_tokens=True,
            )
            clean_ids = enc["input_ids"][0]
            clean_attn = enc["attention_mask"][0]

            if clean_ids.shape[0] != chunk_len + 2:
                raise RuntimeError("Unexpected ESM tokenized chunk length.")

            for b0 in range(0, len(affected_indices), BATCH_SIZE):
                batch_indices = affected_indices[b0:b0 + BATCH_SIZE]

                ids = clean_ids.unsqueeze(0).repeat(
                    len(batch_indices), 1
                )
                attn = clean_attn.unsqueeze(0).repeat(
                    len(batch_indices), 1
                )

                for local_b, set_i in enumerate(batch_indices):
                    positions = deletion_sets[set_i]
                    local_positions = positions[
                        (positions >= start) & (positions < end)
                    ] - start
                    ids[local_b, local_positions + 1] = mask_token_id

                ids = ids.to(DEVICE)
                attn = attn.to(DEVICE)

                with torch.inference_mode():
                    hidden = esm(
                        input_ids=ids,
                        attention_mask=attn,
                    ).last_hidden_state[:, 1:-1, :]

                perturbed_chunks = (
                    hidden.mean(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                delta = (
                    perturbed_chunks.astype(np.float64)
                    - clean_chunk[None, :]
                )

                result[batch_indices] += (
                    delta * (float(chunk_len) / float(L))
                )

        chunk_i += 1

    return result.astype(np.float32)


def seq_mlp_logits(embeddings, model):
    x = torch.from_numpy(
        np.asarray(embeddings, dtype=np.float32)
    ).to(DEVICE)

    with torch.inference_mode():
        logits = model(x)

    return logits.detach().cpu().numpy().astype(np.float32)


# ============================================================
# STRUCTURE EXPERT
# ============================================================

def struct_predict(model, x, edge_index, edge_attr):
    batch_index = torch.zeros(
        x.shape[0],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():
        logits = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch_index=batch_index,
            num_graphs=1,
        )[0]

    return logits.detach().cpu().numpy().astype(np.float32)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_existing_output(
    path,
    uid,
    test_row,
    structure_row,
    length,
    positive_labels,
):
    with np.load(path, allow_pickle=False) as z:
        required = {
            "uniprot_id",
            "test_row",
            "structure_row",
            "length",
            "k",
            "positive_label_indices",
            "protein_seed",
            "n_random_sets",
            "clean_js",
            "top_deleted_js",
            "top_conflict_reduction",
            "random_mean_deleted_js",
            "random_mean_conflict_reduction",
            "conflict_deletion_enrichment",
            "sequence_clean_embedding_max_abs_diff",
            "sequence_clean_logit_max_abs_diff",
            "structure_clean_logit_max_abs_diff",
            "canonical_clean_js_max_abs_diff",
        }

        missing = required.difference(z.files)
        if missing:
            raise RuntimeError(
                f"{uid}: existing output missing keys {sorted(missing)}"
            )

        if str(z["uniprot_id"].item()) != uid:
            raise RuntimeError(f"{uid}: output ID mismatch.")
        if int(z["test_row"].item()) != test_row:
            raise RuntimeError(f"{uid}: output test_row mismatch.")
        if int(z["structure_row"].item()) != structure_row:
            raise RuntimeError(f"{uid}: output structure_row mismatch.")
        if int(z["length"].item()) != length:
            raise RuntimeError(f"{uid}: output length mismatch.")
        if int(z["k"].item()) != deletion_k(length):
            raise RuntimeError(f"{uid}: output k mismatch.")
        if int(z["protein_seed"].item()) != protein_seed(uid):
            raise RuntimeError(f"{uid}: output protein seed mismatch.")
        if int(z["n_random_sets"].item()) != N_RANDOM_SETS:
            raise RuntimeError(f"{uid}: output random-set count mismatch.")

        labels = np.asarray(
            z["positive_label_indices"],
            dtype=np.int32,
        )
        if not np.array_equal(labels, positive_labels):
            raise RuntimeError(f"{uid}: output labels mismatch.")

        K = len(positive_labels)

        for name in [
            "clean_js",
            "top_deleted_js",
            "top_conflict_reduction",
            "random_mean_deleted_js",
            "random_mean_conflict_reduction",
            "conflict_deletion_enrichment",
        ]:
            arr = np.asarray(z[name])
            if arr.shape != (K,):
                raise RuntimeError(
                    f"{uid}: {name} shape {arr.shape}; expected {(K,)}."
                )
            if not np.isfinite(arr).all():
                raise RuntimeError(f"{uid}: non-finite {name}.")

        for name in [
            "sequence_clean_embedding_max_abs_diff",
            "sequence_clean_logit_max_abs_diff",
            "structure_clean_logit_max_abs_diff",
            "canonical_clean_js_max_abs_diff",
        ]:
            if not np.isfinite(float(z[name].item())):
                raise RuntimeError(f"{uid}: non-finite {name}.")

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.protein is not None and args.limit is not None:
        raise RuntimeError("--protein and --limit are mutually exclusive.")
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be positive.")

    print("=" * 78)
    print("E10 CONFLICT-DELETION V1 — PRODUCTION RUNNER")
    print("=" * 78)
    print("Device:", DEVICE)
    print("Frozen GO-MF labels:", NUM_LABELS)
    print("Sequence temperature:", SEQ_TEMPERATURE)
    print("Structure temperature:", STRUCTURE_TEMPERATURE)
    print("Random sets per protein:", N_RANDOM_SETS)
    print("Primary conflict: Bernoulli JS, natural log, EPS=1e-7")
    print("Deletion budget: k=max(1, ceil(0.10*L))")
    print("Same selected residues perturbed in BOTH experts")
    print("Random JS paired by random-set index BEFORE averaging")
    print()

    required_files = [
        TEST_PKL,
        TEST_IDS_PATH,
        TEST_LABELS_PATH,
        POPULATION_PATH,
        SEQ_EMBEDDINGS_PATH,
        SEQ_RAW_LOGITS_PATH,
        SEQ_PROBS_PATH,
        SEQ_IDS_PATH,
        SEQ_CHECKPOINT_PATH,
        STRUCT_IDS_PATH,
        STRUCT_RAW_LOGITS_PATH,
        STRUCT_PROBS_PATH,
        STRUCT_CHECKPOINT_PATH,
        FROZEN_JS_PATH,
        PAIRED_IDS_PATH,
        PROTOCOL_PATH,
        IMPLEMENTATION_PATH,
        RANDOM_GENERATOR_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise RuntimeError(f"Missing required file: {path}")

    actual_protocol_sha = verify_sha(
        PROTOCOL_PATH,
        EXPECTED_PROTOCOL_SHA256,
        "conflict-deletion protocol",
    )
    actual_implementation_sha = verify_sha(
        IMPLEMENTATION_PATH,
        EXPECTED_IMPLEMENTATION_SHA256,
        "conflict-deletion implementation",
    )
    actual_random_sha = verify_sha(
        RANDOM_GENERATOR_PATH,
        EXPECTED_RANDOM_SHA256,
        "random generator",
    )
    actual_population_sha = verify_sha(
        POPULATION_PATH,
        EXPECTED_XAI_POPULATION_SHA256,
        "XAI population",
    )
    actual_seq_checkpoint_sha = verify_sha(
        SEQ_CHECKPOINT_PATH,
        EXPECTED_SEQ_CHECKPOINT_SHA256,
        "sequence checkpoint",
    )
    actual_struct_checkpoint_sha = verify_sha(
        STRUCT_CHECKPOINT_PATH,
        EXPECTED_STRUCT_CHECKPOINT_SHA256,
        "structure checkpoint",
    )

    test_ids = read_ids(TEST_IDS_PATH)
    seq_ids = read_ids(SEQ_IDS_PATH)
    struct_ids = read_ids(STRUCT_IDS_PATH)
    paired_ids = read_ids(PAIRED_IDS_PATH)
    population_ids = read_ids(POPULATION_PATH)

    if len(test_ids) != EXPECTED_TEST:
        raise RuntimeError(f"Unexpected TEST count: {len(test_ids)}")
    if seq_ids != test_ids:
        raise RuntimeError(
            "Sequence TEST IDs do not exactly match frozen TEST order."
        )
    if len(struct_ids) != EXPECTED_PAIRED:
        raise RuntimeError(
            f"Unexpected structure TEST count: {len(struct_ids)}"
        )
    if paired_ids != struct_ids:
        raise RuntimeError(
            "Saved paired TEST IDs do not match structure eligible order."
        )
    if len(population_ids) != EXPECTED_POPULATION:
        raise RuntimeError(
            f"Unexpected XAI population count: {len(population_ids)}"
        )
    if len(set(population_ids)) != len(population_ids):
        raise RuntimeError("Duplicate IDs in frozen XAI population.")

    test_index = {
        uid: i
        for i, uid in enumerate(test_ids)
    }
    structure_index = {
        uid: i
        for i, uid in enumerate(struct_ids)
    }

    if any(uid not in test_index for uid in struct_ids):
        raise RuntimeError(
            "Structure TEST ID missing from frozen TEST."
        )
    if any(uid not in structure_index for uid in population_ids):
        raise RuntimeError(
            "XAI population contains non-structure-eligible ID."
        )

    paired_rows = np.asarray(
        [test_index[uid] for uid in struct_ids],
        dtype=np.int64,
    )

    Y = np.load(
        TEST_LABELS_PATH,
        mmap_mode="r",
    )
    if Y.shape != (EXPECTED_TEST, NUM_LABELS):
        raise RuntimeError(
            f"Unexpected TEST label shape: {Y.shape}"
        )

    seq_embeddings = np.load(
        SEQ_EMBEDDINGS_PATH,
        mmap_mode="r",
    )
    seq_raw_logits = np.load(
        SEQ_RAW_LOGITS_PATH,
        mmap_mode="r",
    )
    struct_raw_logits = np.load(
        STRUCT_RAW_LOGITS_PATH,
        mmap_mode="r",
    )

    if seq_embeddings.shape != (EXPECTED_TEST, EMBED_DIM):
        raise RuntimeError(
            f"Unexpected sequence embedding shape: {seq_embeddings.shape}"
        )
    if seq_raw_logits.shape != (EXPECTED_TEST, NUM_LABELS):
        raise RuntimeError(
            f"Unexpected sequence raw-logit shape: {seq_raw_logits.shape}"
        )
    if struct_raw_logits.shape != (EXPECTED_PAIRED, NUM_LABELS):
        raise RuntimeError(
            f"Unexpected structure raw-logit shape: {struct_raw_logits.shape}"
        )

    # Exact original final-test numerical pathway.
    ps_all = np.load(
        SEQ_PROBS_PATH
    ).astype(np.float64)

    pt = np.load(
        STRUCT_PROBS_PATH
    ).astype(np.float64)

    frozen_js = np.load(
        FROZEN_JS_PATH
    )

    if ps_all.shape != (EXPECTED_TEST, NUM_LABELS):
        raise RuntimeError(
            f"Unexpected sequence probability shape: {ps_all.shape}"
        )
    if pt.shape != (EXPECTED_PAIRED, NUM_LABELS):
        raise RuntimeError(
            f"Unexpected structure probability shape: {pt.shape}"
        )
    if frozen_js.shape != (EXPECTED_PAIRED, NUM_LABELS):
        raise RuntimeError(
            f"Unexpected frozen JS shape: {frozen_js.shape}"
        )

    ps = ps_all[paired_rows]

    reproduced_js = bernoulli_js(
        ps,
        pt,
    ).astype(np.float32)

    canonical_js_max_diff = float(
        np.max(
            np.abs(
                reproduced_js.astype(np.float64)
                - frozen_js.astype(np.float64)
            )
        )
    )

    if not np.array_equal(
        reproduced_js,
        frozen_js,
    ):
        raise RuntimeError(
            "Frozen Bernoulli-JS did not reproduce bit-for-bit. "
            f"max_diff={canonical_js_max_diff}"
        )

    # --------------------------------------------------------
    # PREVALIDATE COMPLETE XAI POPULATION
    # --------------------------------------------------------

    population_meta = {}
    total_targets = 0
    total_residues = 0

    for uid in population_ids:
        test_row = test_index[uid]
        structure_row = structure_index[uid]

        positive_labels = (
            np.flatnonzero(
                Y[test_row] == 1
            )
            .astype(np.int32)
        )

        if len(positive_labels) <= 0:
            raise RuntimeError(
                f"{uid}: zero positive labels."
            )

        seq_attr_path = (
            SEQ_ATTR_DIR / f"{uid}.npz"
        )
        struct_attr_path = (
            STRUCT_ATTR_DIR / f"{uid}.npz"
        )
        graph_path = (
            GRAPH_DIR / f"{uid}.npz"
        )

        for path in (
            seq_attr_path,
            struct_attr_path,
            graph_path,
        ):
            if not path.exists():
                raise RuntimeError(
                    f"{uid}: missing required artifact {path}"
                )

        with np.load(
            seq_attr_path,
            allow_pickle=False,
        ) as a:
            if str(a["uniprot_id"].item()) != uid:
                raise RuntimeError(
                    f"{uid}: sequence attribution ID mismatch."
                )
            if int(a["test_row"].item()) != test_row:
                raise RuntimeError(
                    f"{uid}: sequence attribution test_row mismatch."
                )

            seq_labels = np.asarray(
                a["positive_label_indices"],
                dtype=np.int32,
            )
            seq_attr = np.asarray(
                a["attribution_logit_drop"],
                dtype=np.float32,
            )
            seq_clean_attr_logits = np.asarray(
                a["clean_target_logits"],
                dtype=np.float32,
            )
            seq_length = int(
                a["length"].item()
            )

        with np.load(
            struct_attr_path,
            allow_pickle=False,
        ) as a:
            if str(a["uniprot_id"].item()) != uid:
                raise RuntimeError(
                    f"{uid}: structure attribution ID mismatch."
                )
            if int(a["test_row"].item()) != test_row:
                raise RuntimeError(
                    f"{uid}: structure attribution test_row mismatch."
                )
            if int(a["structure_row"].item()) != structure_row:
                raise RuntimeError(
                    f"{uid}: structure attribution structure_row mismatch."
                )

            struct_labels = np.asarray(
                a["positive_label_indices"],
                dtype=np.int32,
            )
            struct_attr = np.asarray(
                a["attribution_logit_drop"],
                dtype=np.float32,
            )
            struct_clean_attr_logits = np.asarray(
                a["clean_target_logits"],
                dtype=np.float32,
            )
            struct_length = int(
                a["length"].item()
            )

        if not np.array_equal(
            seq_labels,
            positive_labels,
        ):
            raise RuntimeError(
                f"{uid}: sequence attribution labels mismatch."
            )

        if not np.array_equal(
            struct_labels,
            positive_labels,
        ):
            raise RuntimeError(
                f"{uid}: structure attribution labels mismatch."
            )

        if seq_length != struct_length:
            raise RuntimeError(
                f"{uid}: attribution length mismatch."
            )

        K = len(positive_labels)

        if seq_attr.shape != (
            K,
            seq_length,
        ):
            raise RuntimeError(
                f"{uid}: sequence attribution shape mismatch."
            )

        if struct_attr.shape != (
            K,
            seq_length,
        ):
            raise RuntimeError(
                f"{uid}: structure attribution shape mismatch."
            )

        if not np.isfinite(seq_attr).all():
            raise RuntimeError(
                f"{uid}: non-finite sequence attribution."
            )
        if not np.isfinite(struct_attr).all():
            raise RuntimeError(
                f"{uid}: non-finite structure attribution."
            )

        frozen_seq_targets = np.asarray(
            seq_raw_logits[
                test_row,
                positive_labels,
            ],
            dtype=np.float32,
        )

        frozen_struct_targets = np.asarray(
            struct_raw_logits[
                structure_row,
                positive_labels,
            ],
            dtype=np.float32,
        )

        if not np.allclose(
            seq_clean_attr_logits,
            frozen_seq_targets,
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        ):
            raise RuntimeError(
                f"{uid}: sequence attribution clean-logit mismatch."
            )

        if not np.allclose(
            struct_clean_attr_logits,
            frozen_struct_targets,
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        ):
            raise RuntimeError(
                f"{uid}: structure attribution clean-logit mismatch."
            )

        with np.load(
            graph_path,
            allow_pickle=False,
        ) as g:
            if str(g["uniprot_id"].item()) != uid:
                raise RuntimeError(
                    f"{uid}: graph ID mismatch."
                )
            graph_length = int(
                g["x"].shape[0]
            )

        if graph_length != seq_length:
            raise RuntimeError(
                f"{uid}: graph/attribution length mismatch."
            )

        population_meta[uid] = (
            test_row,
            structure_row,
            seq_length,
            positive_labels,
        )

        total_targets += K
        total_residues += seq_length

    if total_targets != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Unexpected target count: {total_targets}"
        )

    if total_residues != EXPECTED_RESIDUES:
        raise RuntimeError(
            f"Unexpected residue count: {total_residues}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = set()

    for path in sorted(
        OUT_DIR.glob("*.npz")
    ):
        uid = path.stem

        if uid not in population_meta:
            raise RuntimeError(
                f"Unexpected output file: {path}"
            )

        (
            test_row,
            structure_row,
            length,
            positive_labels,
        ) = population_meta[uid]

        validate_existing_output(
            path,
            uid,
            test_row,
            structure_row,
            length,
            positive_labels,
        )

        completed.add(uid)

    print("Frozen XAI proteins:", len(population_ids))
    print("Positive protein-label targets:", total_targets)
    print("Population residues:", total_residues)
    print("Paired TEST proteins:", len(struct_ids))
    print("Canonical frozen JS: BIT-FOR-BIT VERIFIED")
    print(
        "Validated existing conflict-deletion outputs:",
        len(completed),
    )

    if args.validate_only:
        print()
        print("VALIDATION PASSED")
        print("No model loaded.")
        print("No perturbation performed.")
        print("No conflict-deletion outcome computed.")
        print("=" * 78)
        return

    # --------------------------------------------------------
    # SELECT REQUESTED PROTEINS
    # --------------------------------------------------------

    if args.protein is not None:
        uid = args.protein.strip()

        if uid not in population_meta:
            raise RuntimeError(
                f"{uid} is not in frozen XAI population."
            )

        requested_ids = [uid]

    elif args.limit is not None:
        requested_ids = population_ids[
            :args.limit
        ]

    else:
        requested_ids = list(
            population_ids
        )

    todo = [
        uid
        for uid in requested_ids
        if uid not in completed
    ]

    print()
    print("Requested proteins:", len(requested_ids))
    print(
        "Already completed:",
        len(requested_ids) - len(todo),
    )
    print("To process:", len(todo))

    if len(todo) == 0:
        print("Nothing to do.")
        print("=" * 78)
        return

    # --------------------------------------------------------
    # LOAD TEST SEQUENCES
    # --------------------------------------------------------

    test_df = pd.read_pickle(
        TEST_PKL
    )

    id_col = None
    for candidate in (
        "uniprot_id",
        "accession",
        "Entry",
    ):
        if candidate in test_df.columns:
            id_col = candidate
            break

    seq_col = None
    for candidate in (
        "sequence",
        "Sequence",
    ):
        if candidate in test_df.columns:
            seq_col = candidate
            break

    if id_col is None or seq_col is None:
        raise RuntimeError(
            "Could not identify ID/sequence columns in "
            f"{TEST_PKL}. Columns={list(test_df.columns)}"
        )

    sequence_by_id = dict(
        zip(
            test_df[id_col].astype(str),
            test_df[seq_col].astype(str),
        )
    )

    for uid in todo:
        if uid not in sequence_by_id:
            raise RuntimeError(
                f"{uid}: sequence missing from TEST pickle."
            )

        if (
            len(sequence_by_id[uid])
            != population_meta[uid][2]
        ):
            raise RuntimeError(
                f"{uid}: TEST sequence length mismatch."
            )

    # --------------------------------------------------------
    # LOAD FROZEN SEQUENCE EXPERT
    # --------------------------------------------------------

    print()
    print("Loading frozen sequence expert...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    esm = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    ).to(DEVICE)

    esm.eval()

    seq_checkpoint = torch.load(
        SEQ_CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    seq_model = SequenceMLP().to(
        DEVICE
    )

    if (
        isinstance(seq_checkpoint, dict)
        and "model_state_dict" in seq_checkpoint
    ):
        seq_state = seq_checkpoint[
            "model_state_dict"
        ]
    else:
        seq_state = seq_checkpoint

    seq_model.load_state_dict(
        seq_state
    )
    seq_model.eval()

    # --------------------------------------------------------
    # LOAD FROZEN STRUCTURE EXPERT
    # --------------------------------------------------------

    print("Loading frozen structure expert...")

    struct_checkpoint = torch.load(
        STRUCT_CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if struct_checkpoint["epoch"] != 21:
        raise RuntimeError(
            "Unexpected structure checkpoint epoch."
        )

    struct_model = StructureGNN(
        node_dim=NODE_DIM,
        hidden_dim=STRUCT_HIDDEN_DIM,
        num_labels=NUM_LABELS,
        num_layers=STRUCT_NUM_LAYERS,
        dropout=STRUCT_DROPOUT,
    ).to(DEVICE)

    struct_model.load_state_dict(
        struct_checkpoint[
            "model_state_dict"
        ]
    )

    struct_model.eval()

    n_struct_params = sum(
        p.numel()
        for p in struct_model.parameters()
    )

    if (
        n_struct_params
        != EXPECTED_STRUCT_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Unexpected structure parameter count: "
            f"{n_struct_params}"
        )

    print("Frozen models loaded.")

    # --------------------------------------------------------
    # RUN INFO
    # --------------------------------------------------------

    run_info = {
        "method":
            "conflict_deletion_v1",

        "protocol_sha256":
            actual_protocol_sha,

        "implementation_sha256":
            actual_implementation_sha,

        "random_generator_sha256":
            actual_random_sha,

        "population_sha256":
            actual_population_sha,

        "sequence_checkpoint_sha256":
            actual_seq_checkpoint_sha,

        "structure_checkpoint_sha256":
            actual_struct_checkpoint_sha,

        "population_size":
            len(population_ids),

        "total_positive_targets":
            total_targets,

        "total_residues":
            total_residues,

        "num_labels":
            NUM_LABELS,

        "sequence_temperature":
            SEQ_TEMPERATURE,

        "structure_temperature":
            STRUCTURE_TEMPERATURE,

        "eps":
            EPS,

        "deletion_fraction":
            0.10,

        "n_random_sets":
            N_RANDOM_SETS,

        "primary_conflict":
            "per-label Bernoulli JS with natural logarithms",

        "driver":
            "delta_JS_sequence + delta_JS_structure",

        "top_ranking":
            "signed driver descending; lower residue position breaks ties",

        "joint_top_perturbation":
            "same selected positions masked in sequence expert "
            "and zeroed in structure expert",

        "joint_random_perturbation":
            "same frozen random set used in both experts; "
            "JS calculated per paired random set before averaging",

        "negative_values_retained":
            True,

        "categorical_normalization":
            False,

        "plddt_structure_input":
            False,
    }

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RUN_INFO_PATH,
        "w",
    ) as f:
        json.dump(
            run_info,
            f,
            indent=2,
            sort_keys=True,
        )

    # --------------------------------------------------------
    # PRODUCTION
    # --------------------------------------------------------

    run_start = time.time()
    processed_this_run = 0

    for uid in requested_ids:

        if uid in completed:
            continue

        protein_start = time.time()

        (
            test_row,
            structure_row,
            L,
            positive_labels,
        ) = population_meta[uid]

        K = len(
            positive_labels
        )

        k = deletion_k(
            L
        )

        sequence = sequence_by_id[
            uid
        ]

        # ====================================================
        # LOAD FROZEN SINGLE-RESIDUE ATTRIBUTIONS
        # ====================================================

        with np.load(
            SEQ_ATTR_DIR / f"{uid}.npz",
            allow_pickle=False,
        ) as a:
            seq_attr_raw = np.asarray(
                a["attribution_logit_drop"],
                dtype=np.float32,
            )

            seq_clean_raw = np.asarray(
                a["clean_target_logits"],
                dtype=np.float32,
            )

        with np.load(
            STRUCT_ATTR_DIR / f"{uid}.npz",
            allow_pickle=False,
        ) as a:
            struct_attr_raw = np.asarray(
                a["attribution_logit_drop"],
                dtype=np.float32,
            )

            struct_clean_raw = np.asarray(
                a["clean_target_logits"],
                dtype=np.float32,
            )

        # ====================================================
        # CANONICAL CLEAN CONFLICT
        # ====================================================

        clean_ps = ps[
            structure_row,
            positive_labels,
        ].astype(
            np.float64
        )

        clean_pt = pt[
            structure_row,
            positive_labels,
        ].astype(
            np.float64
        )

        clean_js = frozen_js[
            structure_row,
            positive_labels,
        ].astype(
            np.float64
        )

        local_clean_js = bernoulli_js(
            clean_ps,
            clean_pt,
        )

        clean_js_max_diff = float(
            np.max(
                np.abs(
                    local_clean_js
                    - clean_js
                )
            )
        )

        # ====================================================
        # SINGLE-RESIDUE CONFLICT DRIVER
        # ====================================================

        seq_perturbed_z = (
            seq_clean_raw[
                :,
                None,
            ].astype(
                np.float64
            )
            - seq_attr_raw.astype(
                np.float64
            )
        ) / float(
            SEQ_TEMPERATURE
        )

        struct_perturbed_z = (
            struct_clean_raw[
                :,
                None,
            ].astype(
                np.float64
            )
            - struct_attr_raw.astype(
                np.float64
            )
        ) / float(
            STRUCTURE_TEMPERATURE
        )

        seq_perturbed_p = sigmoid(
            seq_perturbed_z
        )

        struct_perturbed_p = sigmoid(
            struct_perturbed_z
        )

        delta_js_seq = (
            clean_js[
                :,
                None,
            ]
            - bernoulli_js(
                seq_perturbed_p,
                clean_pt[
                    :,
                    None,
                ],
            )
        )

        delta_js_structure = (
            clean_js[
                :,
                None,
            ]
            - bernoulli_js(
                clean_ps[
                    :,
                    None,
                ],
                struct_perturbed_p,
            )
        )

        conflict_driver = (
            delta_js_seq
            + delta_js_structure
        )

        if conflict_driver.shape != (
            K,
            L,
        ):
            raise RuntimeError(
                f"{uid}: conflict-driver shape mismatch."
            )

        if not np.isfinite(
            conflict_driver
        ).all():
            raise RuntimeError(
                f"{uid}: non-finite conflict driver."
            )

        # ====================================================
        # TOP-k RESIDUES PER TARGET LABEL
        # ====================================================

        residue_positions = np.arange(
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
                    residue_positions,
                    -conflict_driver[
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
                    f"{uid}: duplicate top-k residue."
                )

            if (
                selected.min() < 0
                or selected.max() >= L
            ):
                raise RuntimeError(
                    f"{uid}: top-k residue out of bounds."
                )

            top_sets[
                label_i
            ] = np.sort(
                selected
            )

        # ====================================================
        # EXACT SHARED RANDOM SETS
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
                f"{uid}: random-set shape mismatch."
            )

        # ====================================================
        # CLEAN SEQUENCE REPRODUCTION
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
            seq_embeddings[
                test_row
            ],
            dtype=np.float32,
        )

        seq_embedding_max_diff = float(
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
                f"{uid}: clean sequence embedding mismatch. "
                f"max_diff={seq_embedding_max_diff}"
            )

        seq_clean_logits_reproduced = (
            seq_mlp_logits(
                clean_embedding[
                    None,
                    :
                ],
                seq_model,
            )[0]
        )

        frozen_seq_logits = np.asarray(
            seq_raw_logits[
                test_row
            ],
            dtype=np.float32,
        )

        seq_logit_max_diff = float(
            np.max(
                np.abs(
                    seq_clean_logits_reproduced
                    - frozen_seq_logits
                )
            )
        )

        if not np.allclose(
            seq_clean_logits_reproduced,
            frozen_seq_logits,
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        ):
            raise RuntimeError(
                f"{uid}: clean sequence logits mismatch. "
                f"max_diff={seq_logit_max_diff}"
            )

        # ====================================================
        # CLEAN STRUCTURE REPRODUCTION
        # ====================================================

        with np.load(
            GRAPH_DIR / f"{uid}.npz",
            allow_pickle=False,
        ) as g:
            x_np = (
                g["x"]
                .copy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            edge_index_np = (
                g["edge_index"]
                .copy()
            )

            edge_attr_np = (
                g["edge_attr"]
                .copy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        if x_np.shape != (
            L,
            NODE_DIM,
        ):
            raise RuntimeError(
                f"{uid}: structure x shape mismatch."
            )

        if not np.allclose(
            x_np.sum(
                axis=1
            ),
            1.0,
            rtol=0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{uid}: clean node features are not one-hot."
            )

        if (
            edge_index_np.ndim != 2
            or edge_index_np.shape[0] != 2
        ):
            raise RuntimeError(
                f"{uid}: invalid edge_index shape."
            )

        if (
            edge_attr_np.ndim != 2
            or edge_attr_np.shape[1] != 3
        ):
            raise RuntimeError(
                f"{uid}: invalid edge_attr shape."
            )

        if (
            edge_index_np.shape[1]
            != edge_attr_np.shape[0]
        ):
            raise RuntimeError(
                f"{uid}: edge count mismatch."
            )

        x_clean = (
            torch.from_numpy(
                x_np
            )
            .float()
            .to(
                DEVICE
            )
        )

        edge_index = (
            torch.from_numpy(
                edge_index_np
            )
            .long()
            .to(
                DEVICE
            )
        )

        edge_attr = (
            torch.from_numpy(
                edge_attr_np
            )
            .float()
            .to(
                DEVICE
            )
        )

        struct_clean_logits_reproduced = (
            struct_predict(
                struct_model,
                x_clean,
                edge_index,
                edge_attr,
            )
        )

        frozen_struct_logits = np.asarray(
            struct_raw_logits[
                structure_row
            ],
            dtype=np.float32,
        )

        struct_logit_max_diff = float(
            np.max(
                np.abs(
                    struct_clean_logits_reproduced
                    - frozen_struct_logits
                )
            )
        )

        if not np.allclose(
            struct_clean_logits_reproduced,
            frozen_struct_logits,
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        ):
            raise RuntimeError(
                f"{uid}: clean structure logits mismatch. "
                f"max_diff={struct_logit_max_diff}"
            )

        # ====================================================
        # TOP JOINT DELETION
        # ====================================================

        top_seq_embeddings = (
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

        top_seq_logits = seq_mlp_logits(
            top_seq_embeddings,
            seq_model,
        )

        top_seq_target_z = (
            top_seq_logits[
                np.arange(
                    K
                ),
                positive_labels,
            ].astype(
                np.float64
            )
            / float(
                SEQ_TEMPERATURE
            )
        )

        top_struct_target_z = np.empty(
            K,
            dtype=np.float64,
        )

        for label_i in range(
            K
        ):
            x_deleted = x_clean.clone()

            pos = torch.as_tensor(
                top_sets[
                    label_i
                ],
                dtype=torch.long,
                device=DEVICE,
            )

            x_deleted[
                pos
            ] = 0.0

            deleted_logits = struct_predict(
                struct_model,
                x_deleted,
                edge_index,
                edge_attr,
            )

            top_struct_target_z[
                label_i
            ] = (
                float(
                    deleted_logits[
                        positive_labels[
                            label_i
                        ]
                    ]
                )
                / float(
                    STRUCTURE_TEMPERATURE
                )
            )

        top_deleted_js = bernoulli_js(
            sigmoid(
                top_seq_target_z
            ),
            sigmoid(
                top_struct_target_z
            ),
        )

        top_conflict_reduction = (
            clean_js
            - top_deleted_js
        )

        # ====================================================
        # 1000 PAIRED RANDOM JOINT DELETIONS
        #
        # IMPORTANT:
        # random set j is applied to BOTH experts.
        # JS_j is computed before any averaging.
        # ====================================================

        random_seq_embeddings = (
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

        random_seq_logits = seq_mlp_logits(
            random_seq_embeddings,
            seq_model,
        )

        random_seq_target_z = (
            random_seq_logits[
                :,
                positive_labels,
            ].astype(
                np.float64
            )
            / float(
                SEQ_TEMPERATURE
            )
        )

        random_struct_target_z = np.empty(
            (
                N_RANDOM_SETS,
                K,
            ),
            dtype=np.float64,
        )

        for (
            random_i,
            random_positions,
        ) in enumerate(
            random_sets
        ):
            x_deleted = x_clean.clone()

            pos = torch.as_tensor(
                random_positions,
                dtype=torch.long,
                device=DEVICE,
            )

            x_deleted[
                pos
            ] = 0.0

            deleted_logits = struct_predict(
                struct_model,
                x_deleted,
                edge_index,
                edge_attr,
            )

            random_struct_target_z[
                random_i
            ] = (
                deleted_logits[
                    positive_labels
                ].astype(
                    np.float64
                )
                / float(
                    STRUCTURE_TEMPERATURE
                )
            )

        random_deleted_js = bernoulli_js(
            sigmoid(
                random_seq_target_z
            ),
            sigmoid(
                random_struct_target_z
            ),
        )

        if random_deleted_js.shape != (
            N_RANDOM_SETS,
            K,
        ):
            raise RuntimeError(
                f"{uid}: random JS shape mismatch."
            )

        random_conflict_reduction = (
            clean_js[
                None,
                :
            ]
            - random_deleted_js
        )

        random_mean_deleted_js = (
            random_deleted_js.mean(
                axis=0,
                dtype=np.float64,
            )
        )

        random_mean_conflict_reduction = (
            random_conflict_reduction.mean(
                axis=0,
                dtype=np.float64,
            )
        )

        conflict_deletion_enrichment = (
            top_conflict_reduction
            - random_mean_conflict_reduction
        )

        # ====================================================
        # QC
        # ====================================================

        vectors = {
            "clean_js":
                clean_js,

            "top_deleted_js":
                top_deleted_js,

            "top_conflict_reduction":
                top_conflict_reduction,

            "random_mean_deleted_js":
                random_mean_deleted_js,

            "random_mean_conflict_reduction":
                random_mean_conflict_reduction,

            "conflict_deletion_enrichment":
                conflict_deletion_enrichment,
        }

        for (
            name,
            arr,
        ) in vectors.items():

            if np.asarray(
                arr
            ).shape != (
                K,
            ):
                raise RuntimeError(
                    f"{uid}: {name} shape mismatch."
                )

            if not np.isfinite(
                arr
            ).all():
                raise RuntimeError(
                    f"{uid}: non-finite {name}."
                )

        expected_enrichment = (
            top_conflict_reduction
            - random_mean_conflict_reduction
        )

        if not np.allclose(
            conflict_deletion_enrichment,
            expected_enrichment,
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{uid}: enrichment algebra failure."
            )

        # ====================================================
        # ATOMIC SAVE
        # ====================================================

        out_path = (
            OUT_DIR / f"{uid}.npz"
        )

        atomic_save_npz(
            out_path,

            uniprot_id=np.asarray(
                uid
            ),

            test_row=np.asarray(
                test_row,
                dtype=np.int32,
            ),

            structure_row=np.asarray(
                structure_row,
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

            clean_js=(
                clean_js.astype(
                    np.float32
                )
            ),

            top_deleted_js=(
                top_deleted_js.astype(
                    np.float32
                )
            ),

            top_conflict_reduction=(
                top_conflict_reduction.astype(
                    np.float32
                )
            ),

            random_mean_deleted_js=(
                random_mean_deleted_js.astype(
                    np.float32
                )
            ),

            random_mean_conflict_reduction=(
                random_mean_conflict_reduction.astype(
                    np.float32
                )
            ),

            conflict_deletion_enrichment=(
                conflict_deletion_enrichment.astype(
                    np.float32
                )
            ),

            sequence_clean_embedding_max_abs_diff=np.asarray(
                seq_embedding_max_diff,
                dtype=np.float32,
            ),

            sequence_clean_logit_max_abs_diff=np.asarray(
                seq_logit_max_diff,
                dtype=np.float32,
            ),

            structure_clean_logit_max_abs_diff=np.asarray(
                struct_logit_max_diff,
                dtype=np.float32,
            ),

            canonical_clean_js_max_abs_diff=np.asarray(
                clean_js_max_diff,
                dtype=np.float32,
            ),
        )

        validate_existing_output(
            out_path,
            uid,
            test_row,
            structure_row,
            L,
            positive_labels,
        )

        processed_this_run += 1

        elapsed = (
            time.time()
            - protein_start
        )

        print(
            f"[{len(completed) + processed_this_run:4d}/"
            f"{len(population_ids):4d}] "
            f"{uid} "
            f"L={L:4d} "
            f"k={k:3d} "
            f"labels={K:2d} "
            f"time={elapsed:7.2f}s",
            flush=True,
        )

        del seq_attr_raw
        del struct_attr_raw
        del seq_perturbed_z
        del struct_perturbed_z
        del seq_perturbed_p
        del struct_perturbed_p
        del delta_js_seq
        del delta_js_structure
        del conflict_driver
        del top_sets
        del random_sets
        del top_seq_embeddings
        del top_seq_logits
        del random_seq_embeddings
        del random_seq_logits
        del random_seq_target_z
        del random_struct_target_z
        del random_deleted_js
        del random_conflict_reduction

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # FINAL REVALIDATION
    # --------------------------------------------------------

    final_completed = set()

    for path in sorted(
        OUT_DIR.glob("*.npz")
    ):
        uid = path.stem

        if uid not in population_meta:
            raise RuntimeError(
                f"Unexpected final output file: {path}"
            )

        (
            test_row,
            structure_row,
            L,
            positive_labels,
        ) = population_meta[uid]

        validate_existing_output(
            path,
            uid,
            test_row,
            structure_row,
            L,
            positive_labels,
        )

        final_completed.add(
            uid
        )

    print()
    print("=" * 78)
    print("RUN COMPLETE")
    print("=" * 78)
    print(
        "Total frozen population:",
        len(population_ids),
    )
    print(
        "Validated completed outputs:",
        len(final_completed),
    )
    print(
        "This run processed:",
        processed_this_run,
    )
    print(
        "This run wall time (seconds):",
        round(
            time.time() - run_start,
            2,
        ),
    )
    print()
    print("Scientific outcome values printed: NO")
    print("Model weights modified: NO")
    print("Frozen benchmark modified: NO")
    print("Final evaluation directory modified: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
