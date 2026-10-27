from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch


# ============================================================
# IMPORT FROZEN STRUCTURE MODEL
# ============================================================

MODEL_CODE_DIR = Path(__file__).resolve().parents[1] / "models"
if str(MODEL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_CODE_DIR))

from structure_gnn import StructureGNN


# ============================================================
# IMPORT FROZEN RANDOM-SET GENERATOR
# ============================================================

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from e10_deletion_random_sets_v1 import (
    N_RANDOM_SETS,
    deletion_k,
    generate_random_sets,
    protein_seed,
    validate_random_sets,
)


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

NUM_LABELS = 494
NODE_DIM = 21
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10

STRUCTURE_TEMPERATURE = 1.0

LOGIT_RTOL = 1e-5
LOGIT_ATOL = 1e-5

EXPECTED_PARAMETER_COUNT = 298990

EXPECTED_MODEL_SHA256 = (
    "f1434b60a67e7dd695f289f7094d528b"
    "b91a28f249b28efd0d26af28d698991e"
)

EXPECTED_POPULATION_SHA256 = (
    "e69281eb8f1298c8ec0ea7b0498a6d0"
    "acf9fa7f4aa32d37354a9dc173c9e667c"
)

EXPECTED_RANDOM_GENERATOR_SHA256 = (
    "471f64b5c73ee51d61bb570ed179ff2a"
    "7047d998006b5f822fa3d1869933cc63"
)

EXPECTED_PROTOCOL_SHA256 = (
    "d709b2b875691c1e942a6d0e2f7a961f"
    "91df75c193ee6634abf3b878ac7d8f5e"
)

EXPECTED_AMENDMENT_SHA256 = (
    "9dda349af24fc2714961ae5006298c80e"
    "6df9de16b373ea04dee3d9c98ddcf35"
)

EXPECTED_IMPLEMENTATION_SHA256 = (
    "5638e4da63fb2f23fd8589f6cfaac030"
    "68d97c2a18c29cc4dac39b4a95bd2ff6"
)


MODEL_PATH = Path(
    "data/models/structure_gnn_v1_full/final_model.pt"
)

GRAPH_DIR = Path(
    "data/structures/structure_protocol_v1/graphs/test"
)

STRUCTURE_IDS_PATH = Path(
    "data/structures/structure_protocol_v1/"
    "test_structure_eligible_ids.txt"
)

TEST_IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "labels/test_uniprot_ids.txt"
)

TEST_LABELS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "labels/test_labels.npy"
)

SAVED_LOGITS_PATH = Path(
    "data/final_evaluation/"
    "structure_expert_v1_test/test_raw_logits.npy"
)

POPULATION_PATH = Path(
    "data/xai/functional_sites_v1/"
    "xai_attribution_eligible_ids.txt"
)

ATTRIBUTION_DIR = Path(
    "data/xai/functional_sites_v1/"
    "structure_attributions_v1/per_protein"
)

RANDOM_GENERATOR_PATH = Path(
    "src/analysis/e10_deletion_random_sets_v1.py"
)

PROTOCOL_PATH = Path(
    "data/xai/functional_sites_v1/"
    "DELETION_FAITHFULNESS_PROTOCOL_V1.txt"
)

AMENDMENT_PATH = Path(
    "data/xai/functional_sites_v1/"
    "DELETION_FAITHFULNESS_AMENDMENT_V1.txt"
)

IMPLEMENTATION_PATH = Path(
    "data/xai/functional_sites_v1/"
    "DELETION_FAITHFULNESS_IMPLEMENTATION_V1.txt"
)

OUT_ROOT = Path(
    "data/xai/functional_sites_v1/"
    "deletion_faithfulness_v1/structure"
)

OUT_DIR = OUT_ROOT / "per_protein"
RUN_INFO_PATH = OUT_ROOT / "run_info.json"


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def verify_sha(path, expected, name):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{name} SHA mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )
    print(f"{name}: SHA VERIFIED")
    return actual


def atomic_save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp.npz")

    np.savez_compressed(
        tmp,
        **arrays,
    )

    os.replace(tmp, path)


def predict(
    model,
    x,
    edge_index,
    edge_attr,
    device,
):
    batch_index = torch.zeros(
        x.shape[0],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch_index=batch_index,
            num_graphs=1,
        )[0]

    return (
        logits.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


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
            "clean_target_logits_calibrated",
            "top_deleted_target_logits_calibrated",
            "top_deletion_drop",
            "random_mean_deleted_target_logits_calibrated",
            "random_mean_drop",
            "faithfulness_enrichment",
            "clean_logit_max_abs_diff",
            "clean_logit_mean_abs_diff",
        }

        missing = required.difference(z.files)
        if missing:
            raise RuntimeError(
                f"{uid}: existing output missing keys: "
                f"{sorted(missing)}"
            )

        if str(z["uniprot_id"].item()) != uid:
            raise RuntimeError(
                f"{uid}: existing output ID mismatch."
            )

        if int(z["test_row"].item()) != test_row:
            raise RuntimeError(
                f"{uid}: existing test_row mismatch."
            )

        if int(z["structure_row"].item()) != structure_row:
            raise RuntimeError(
                f"{uid}: existing structure_row mismatch."
            )

        if int(z["length"].item()) != length:
            raise RuntimeError(
                f"{uid}: existing length mismatch."
            )

        expected_k = deletion_k(length)

        if int(z["k"].item()) != expected_k:
            raise RuntimeError(
                f"{uid}: existing k mismatch."
            )

        labels = np.asarray(
            z["positive_label_indices"],
            dtype=np.int32,
        )

        if not np.array_equal(
            labels,
            positive_labels,
        ):
            raise RuntimeError(
                f"{uid}: existing positive labels mismatch."
            )

        if int(z["protein_seed"].item()) != protein_seed(uid):
            raise RuntimeError(
                f"{uid}: existing protein seed mismatch."
            )

        if int(z["n_random_sets"].item()) != N_RANDOM_SETS:
            raise RuntimeError(
                f"{uid}: existing random-set count mismatch."
            )

        K = len(positive_labels)

        vector_names = [
            "clean_target_logits_calibrated",
            "top_deleted_target_logits_calibrated",
            "top_deletion_drop",
            "random_mean_deleted_target_logits_calibrated",
            "random_mean_drop",
            "faithfulness_enrichment",
        ]

        for name in vector_names:
            arr = np.asarray(z[name])

            if arr.shape != (K,):
                raise RuntimeError(
                    f"{uid}: existing {name} shape "
                    f"{arr.shape}; expected {(K,)}."
                )

            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"{uid}: existing {name} non-finite."
                )

        for name in [
            "clean_logit_max_abs_diff",
            "clean_logit_mean_abs_diff",
        ]:
            value = float(z[name].item())

            if not np.isfinite(value):
                raise RuntimeError(
                    f"{uid}: existing {name} non-finite."
                )

    return True


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--protein",
    type=str,
    default=None,
)

parser.add_argument(
    "--limit",
    type=int,
    default=None,
)

parser.add_argument(
    "--validate-only",
    action="store_true",
)

args = parser.parse_args()

if args.protein is not None and args.limit is not None:
    raise RuntimeError(
        "--protein and --limit are mutually exclusive."
    )

if args.limit is not None and args.limit <= 0:
    raise RuntimeError(
        "--limit must be positive."
    )


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("E10 STRUCTURE DELETION FAITHFULNESS V1")
print("=" * 78)

print("Device:", DEVICE)
print("Frozen GO-MF labels:", NUM_LABELS)
print("Structure temperature:", STRUCTURE_TEMPERATURE)
print("Random sets per protein:", N_RANDOM_SETS)
print("Deletion budget: k=max(1, ceil(0.10*L))")
print("Perturbation: zero selected 21-D node identity vectors")
print("Graph topology changed: NO")
print("Edge attributes changed: NO")
print("pLDDT model input: NO")
print("ESM model input: NO")
print("Primary score: calibrated target-logit drop")
print()


# ============================================================
# VERIFY REQUIRED FILES + FROZEN SHAS
# ============================================================

required_files = [
    MODEL_PATH,
    STRUCTURE_IDS_PATH,
    TEST_IDS_PATH,
    TEST_LABELS_PATH,
    SAVED_LOGITS_PATH,
    POPULATION_PATH,
    RANDOM_GENERATOR_PATH,
    PROTOCOL_PATH,
    AMENDMENT_PATH,
    IMPLEMENTATION_PATH,
]

for path in required_files:
    if not path.exists():
        raise RuntimeError(
            f"Missing required file: {path}"
        )

actual_model_sha = verify_sha(
    MODEL_PATH,
    EXPECTED_MODEL_SHA256,
    "structure checkpoint",
)

actual_population_sha = verify_sha(
    POPULATION_PATH,
    EXPECTED_POPULATION_SHA256,
    "XAI population",
)

actual_random_sha = verify_sha(
    RANDOM_GENERATOR_PATH,
    EXPECTED_RANDOM_GENERATOR_SHA256,
    "random generator",
)

actual_protocol_sha = verify_sha(
    PROTOCOL_PATH,
    EXPECTED_PROTOCOL_SHA256,
    "deletion protocol",
)

actual_amendment_sha = verify_sha(
    AMENDMENT_PATH,
    EXPECTED_AMENDMENT_SHA256,
    "deletion amendment",
)

actual_implementation_sha = verify_sha(
    IMPLEMENTATION_PATH,
    EXPECTED_IMPLEMENTATION_SHA256,
    "deletion implementation",
)


# ============================================================
# LOAD FROZEN POPULATIONS
# ============================================================

test_ids = read_ids(TEST_IDS_PATH)
structure_ids = read_ids(STRUCTURE_IDS_PATH)
population_ids = read_ids(POPULATION_PATH)

if len(population_ids) != 2447:
    raise RuntimeError(
        "Unexpected frozen XAI population size: "
        f"{len(population_ids)}"
    )

if len(set(population_ids)) != len(population_ids):
    raise RuntimeError(
        "Duplicate IDs in frozen XAI population."
    )

if len(set(structure_ids)) != len(structure_ids):
    raise RuntimeError(
        "Duplicate structure eligible IDs."
    )

test_row_by_id = {
    uid: i for i, uid in enumerate(test_ids)
}

structure_row_by_id = {
    uid: i for i, uid in enumerate(structure_ids)
}

for uid in population_ids:

    if uid not in test_row_by_id:
        raise RuntimeError(
            f"{uid} missing from frozen TEST."
        )

    if uid not in structure_row_by_id:
        raise RuntimeError(
            f"{uid} is not structure eligible."
        )

    graph_path = GRAPH_DIR / f"{uid}.npz"
    attribution_path = ATTRIBUTION_DIR / f"{uid}.npz"

    if not graph_path.exists():
        raise RuntimeError(
            f"Missing graph for {uid}: {graph_path}"
        )

    if not attribution_path.exists():
        raise RuntimeError(
            f"Missing frozen structure attribution "
            f"for {uid}: {attribution_path}"
        )


# ============================================================
# LOAD LABELS / FROZEN LOGITS
# ============================================================

Y = np.load(
    TEST_LABELS_PATH,
    mmap_mode="r",
)

if Y.shape != (len(test_ids), NUM_LABELS):
    raise RuntimeError(
        f"Unexpected TEST label shape: {Y.shape}"
    )

saved_logits = np.load(
    SAVED_LOGITS_PATH,
    mmap_mode="r",
)

if saved_logits.shape != (
    len(structure_ids),
    NUM_LABELS,
):
    raise RuntimeError(
        "Unexpected structure logits shape: "
        f"{saved_logits.shape}"
    )


# ============================================================
# PREVALIDATE COMPLETE FROZEN POPULATION
# ============================================================

total_targets = 0
total_residues = 0
population_meta = {}

for uid in population_ids:

    test_row = test_row_by_id[uid]
    structure_row = structure_row_by_id[uid]

    positive_labels = (
        np.flatnonzero(Y[test_row] == 1)
        .astype(np.int32)
    )

    if len(positive_labels) <= 0:
        raise RuntimeError(
            f"{uid} has zero positive labels."
        )

    graph_path = GRAPH_DIR / f"{uid}.npz"

    with np.load(
        graph_path,
        allow_pickle=False,
    ) as g:

        if str(g["uniprot_id"].item()) != uid:
            raise RuntimeError(
                f"{uid}: graph ID mismatch."
            )

        x_shape = g["x"].shape

        if len(x_shape) != 2 or x_shape[1] != NODE_DIM:
            raise RuntimeError(
                f"{uid}: invalid graph x shape {x_shape}."
            )

        length = int(x_shape[0])

    attr_path = ATTRIBUTION_DIR / f"{uid}.npz"

    with np.load(
        attr_path,
        allow_pickle=False,
    ) as a:

        if str(a["uniprot_id"].item()) != uid:
            raise RuntimeError(
                f"{uid}: attribution ID mismatch."
            )

        if int(a["test_row"].item()) != test_row:
            raise RuntimeError(
                f"{uid}: attribution test_row mismatch."
            )

        if int(a["structure_row"].item()) != structure_row:
            raise RuntimeError(
                f"{uid}: attribution structure_row mismatch."
            )

        if int(a["length"].item()) != length:
            raise RuntimeError(
                f"{uid}: attribution length mismatch."
            )

        attr_labels = np.asarray(
            a["positive_label_indices"],
            dtype=np.int32,
        )

        if not np.array_equal(
            attr_labels,
            positive_labels,
        ):
            raise RuntimeError(
                f"{uid}: attribution labels mismatch."
            )

        attribution = np.asarray(
            a["attribution_logit_drop"],
            dtype=np.float32,
        )

        if attribution.shape != (
            len(positive_labels),
            length,
        ):
            raise RuntimeError(
                f"{uid}: attribution shape mismatch "
                f"{attribution.shape}."
            )

        if not np.isfinite(attribution).all():
            raise RuntimeError(
                f"{uid}: non-finite attribution."
            )

    population_meta[uid] = (
        test_row,
        structure_row,
        length,
        positive_labels,
    )

    total_targets += len(positive_labels)
    total_residues += length


if total_targets != 23087:
    raise RuntimeError(
        "Unexpected total positive target count: "
        f"{total_targets}"
    )

if total_residues != 1279145:
    raise RuntimeError(
        "Unexpected total residue count: "
        f"{total_residues}"
    )


# ============================================================
# VALIDATE EXISTING COMPLETED OUTPUTS
# ============================================================

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


print("Frozen eligible proteins:", len(population_ids))
print("Protein-label targets:", total_targets)
print("Population residues:", total_residues)
print("Validated completed outputs:", len(completed))


# ============================================================
# VALIDATE-ONLY MODE
# ============================================================

if args.validate_only:

    print()
    print("VALIDATION PASSED")
    print("No deletion predictions computed.")
    print("No scientific outcome values printed.")
    print("=" * 78)
    raise SystemExit(0)


# ============================================================
# CHOOSE REQUESTED POPULATION
# ============================================================

if args.protein is not None:

    uid = args.protein.strip()

    if uid not in population_meta:
        raise RuntimeError(
            f"{uid} is not in frozen XAI population."
        )

    requested_ids = [uid]

elif args.limit is not None:

    requested_ids = population_ids[:args.limit]

else:

    requested_ids = list(population_ids)


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

    print()
    print("Nothing to do.")
    print("=" * 78)
    raise SystemExit(0)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

if checkpoint["epoch"] != 21:
    raise RuntimeError(
        "Unexpected structure checkpoint epoch."
    )

model = StructureGNN(
    node_dim=NODE_DIM,
    hidden_dim=HIDDEN_DIM,
    num_labels=NUM_LABELS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

if num_parameters != EXPECTED_PARAMETER_COUNT:
    raise RuntimeError(
        f"Unexpected parameter count: {num_parameters}"
    )

print("Frozen structure model loaded.")
print("Parameters:", num_parameters)


# ============================================================
# RUN INFO
# ============================================================

run_info = {
    "method":
        "structure_deletion_faithfulness_v1",

    "population_file":
        str(POPULATION_PATH),

    "population_sha256":
        actual_population_sha,

    "population_size":
        len(population_ids),

    "total_positive_targets":
        total_targets,

    "total_residues":
        total_residues,

    "model_path":
        str(MODEL_PATH),

    "model_sha256":
        actual_model_sha,

    "checkpoint_epoch":
        int(checkpoint["epoch"]),

    "num_parameters":
        int(num_parameters),

    "num_labels":
        NUM_LABELS,

    "temperature":
        STRUCTURE_TEMPERATURE,

    "deletion_fraction":
        0.10,

    "n_random_sets":
        N_RANDOM_SETS,

    "random_generator_sha256":
        actual_random_sha,

    "protocol_sha256":
        actual_protocol_sha,

    "amendment_sha256":
        actual_amendment_sha,

    "implementation_sha256":
        actual_implementation_sha,

    "top_ranking":
        "signed attribution descending; "
        "lower residue position breaks ties",

    "perturbation":
        "jointly zero selected residue "
        "21-D node identity vectors",

    "graph_topology_changed":
        False,

    "edge_attributes_changed":
        False,

    "plddt_model_input":
        False,

    "esm_model_input":
        False,

    "primary_score":
        "calibrated clean target logit minus "
        "deleted target logit",

    "negative_values_retained":
        True,

    "absolute_attribution_primary":
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


# ============================================================
# MAIN COMPUTATION
# ============================================================

run_start = time.time()
processed_this_run = 0

for global_number, uid in enumerate(
    requested_ids,
    start=1,
):

    if uid in completed:
        continue

    protein_start = time.time()

    (
        test_row,
        structure_row,
        length,
        positive_labels,
    ) = population_meta[uid]

    K = len(positive_labels)
    k = deletion_k(length)

    # --------------------------------------------------------
    # LOAD GRAPH — EXACT FROZEN REPRESENTATION
    # --------------------------------------------------------

    graph_path = GRAPH_DIR / f"{uid}.npz"

    with np.load(
        graph_path,
        allow_pickle=False,
    ) as g:

        graph_uid = str(
            g["uniprot_id"].item()
        )

        if graph_uid != uid:
            raise RuntimeError(
                f"{uid}: graph ID mismatch."
            )

        x_np = (
            g["x"]
            .copy()
            .astype(np.float32, copy=False)
        )

        edge_index_np = (
            g["edge_index"]
            .copy()
        )

        edge_attr_np = (
            g["edge_attr"]
            .copy()
            .astype(np.float32, copy=False)
        )

    if x_np.shape != (length, NODE_DIM):
        raise RuntimeError(
            f"{uid}: unexpected x shape {x_np.shape}."
        )

    row_sums = x_np.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"{uid}: clean node features "
            "are not one-hot identity rows."
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
        torch.from_numpy(x_np)
        .float()
        .to(DEVICE)
    )

    edge_index = (
        torch.from_numpy(edge_index_np)
        .long()
        .to(DEVICE)
    )

    edge_attr = (
        torch.from_numpy(edge_attr_np)
        .float()
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # CLEAN REPRODUCTION
    # --------------------------------------------------------

    clean_logits = predict(
        model,
        x_clean,
        edge_index,
        edge_attr,
        DEVICE,
    )

    frozen_logits = np.asarray(
        saved_logits[structure_row],
        dtype=np.float32,
    )

    diff = np.abs(
        clean_logits - frozen_logits
    )

    max_logit_diff = float(
        diff.max()
    )

    mean_logit_diff = float(
        diff.mean()
    )

    if not np.allclose(
        clean_logits,
        frozen_logits,
        rtol=LOGIT_RTOL,
        atol=LOGIT_ATOL,
    ):

        allowed = (
            LOGIT_ATOL
            + LOGIT_RTOL
            * np.abs(frozen_logits)
        )

        ratio = diff / allowed

        worst_label = int(
            np.argmax(ratio)
        )

        raise RuntimeError(
            f"{uid}: clean structure logits "
            "do not reproduce frozen logits. "
            f"max_diff={max_logit_diff}, "
            f"worst_label={worst_label}, "
            f"worst_ratio="
            f"{float(ratio[worst_label])}"
        )

    clean_target_logits = (
        clean_logits[positive_labels]
        .astype(np.float32, copy=True)
        / STRUCTURE_TEMPERATURE
    )

    # --------------------------------------------------------
    # LOAD FROZEN ATTRIBUTIONS
    # --------------------------------------------------------

    attr_path = (
        ATTRIBUTION_DIR / f"{uid}.npz"
    )

    with np.load(
        attr_path,
        allow_pickle=False,
    ) as a:

        attr_labels = np.asarray(
            a["positive_label_indices"],
            dtype=np.int32,
        )

        attribution = np.asarray(
            a["attribution_logit_drop"],
            dtype=np.float32,
        )

    if not np.array_equal(
        attr_labels,
        positive_labels,
    ):
        raise RuntimeError(
            f"{uid}: attribution label mismatch."
        )

    if attribution.shape != (K, length):
        raise RuntimeError(
            f"{uid}: attribution shape mismatch."
        )

    if not np.isfinite(attribution).all():
        raise RuntimeError(
            f"{uid}: attribution non-finite."
        )

    # --------------------------------------------------------
    # TOP-ATTRIBUTION JOINT DELETION
    #
    # Signed descending.
    # np.lexsort: last key is primary.
    # position ascending breaks exact ties.
    # --------------------------------------------------------

    residue_positions = np.arange(
        length,
        dtype=np.int32,
    )

    top_deleted_logits = np.empty(
        K,
        dtype=np.float32,
    )

    for label_row in range(K):

        order = np.lexsort(
            (
                residue_positions,
                -attribution[label_row],
            )
        )

        top_positions = (
            order[:k]
            .astype(np.int64, copy=False)
        )

        x_deleted = x_clean.clone()

        x_deleted[top_positions] = 0.0

        deleted_logits = predict(
            model,
            x_deleted,
            edge_index,
            edge_attr,
            DEVICE,
        )

        top_deleted_logits[label_row] = (
            deleted_logits[
                positive_labels[label_row]
            ]
            / STRUCTURE_TEMPERATURE
        )

    top_deletion_drop = (
        clean_target_logits
        - top_deleted_logits
    ).astype(np.float32)

    # --------------------------------------------------------
    # EXACT FROZEN RANDOM SETS
    # --------------------------------------------------------

    random_sets = generate_random_sets(
        uid,
        length,
    )

    validate_random_sets(
        random_sets,
        length,
    )

    if random_sets.shape != (
        N_RANDOM_SETS,
        k,
    ):
        raise RuntimeError(
            f"{uid}: random-set shape mismatch."
        )

    # FP64 accumulator, as frozen protocol permits.
    random_deleted_sum = np.zeros(
        K,
        dtype=np.float64,
    )

    for random_positions in random_sets:

        x_deleted = x_clean.clone()

        pos = torch.as_tensor(
            random_positions,
            dtype=torch.long,
            device=DEVICE,
        )

        x_deleted[pos] = 0.0

        deleted_logits = predict(
            model,
            x_deleted,
            edge_index,
            edge_attr,
            DEVICE,
        )

        random_deleted_sum += (
            deleted_logits[positive_labels]
            / STRUCTURE_TEMPERATURE
        ).astype(np.float64)

    random_mean_deleted_logits = (
        random_deleted_sum
        / float(N_RANDOM_SETS)
    ).astype(np.float32)

    random_mean_drop = (
        clean_target_logits
        - random_mean_deleted_logits
    ).astype(np.float32)

    faithfulness_enrichment = (
        top_deletion_drop
        - random_mean_drop
    ).astype(np.float32)

    # --------------------------------------------------------
    # FINAL QC
    # --------------------------------------------------------

    vectors = {
        "clean_target_logits_calibrated":
            clean_target_logits,

        "top_deleted_target_logits_calibrated":
            top_deleted_logits,

        "top_deletion_drop":
            top_deletion_drop,

        "random_mean_deleted_target_logits_calibrated":
            random_mean_deleted_logits,

        "random_mean_drop":
            random_mean_drop,

        "faithfulness_enrichment":
            faithfulness_enrichment,
    }

    for name, arr in vectors.items():

        if arr.shape != (K,):
            raise RuntimeError(
                f"{uid}: {name} shape mismatch."
            )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"{uid}: {name} non-finite."
            )

    # --------------------------------------------------------
    # ATOMIC SAVE
    # --------------------------------------------------------

    out_path = (
        OUT_DIR / f"{uid}.npz"
    )

    atomic_save_npz(
        out_path,

        uniprot_id=np.asarray(uid),

        test_row=np.asarray(
            test_row,
            dtype=np.int32,
        ),

        structure_row=np.asarray(
            structure_row,
            dtype=np.int32,
        ),

        length=np.asarray(
            length,
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
            protein_seed(uid),
            dtype=np.uint32,
        ),

        n_random_sets=np.asarray(
            N_RANDOM_SETS,
            dtype=np.int32,
        ),

        clean_target_logits_calibrated=(
            clean_target_logits
        ),

        top_deleted_target_logits_calibrated=(
            top_deleted_logits
        ),

        top_deletion_drop=(
            top_deletion_drop
        ),

        random_mean_deleted_target_logits_calibrated=(
            random_mean_deleted_logits
        ),

        random_mean_drop=(
            random_mean_drop
        ),

        faithfulness_enrichment=(
            faithfulness_enrichment
        ),

        clean_logit_max_abs_diff=np.asarray(
            max_logit_diff,
            dtype=np.float32,
        ),

        clean_logit_mean_abs_diff=np.asarray(
            mean_logit_diff,
            dtype=np.float32,
        ),
    )

    processed_this_run += 1

    elapsed = time.time() - protein_start

    total_present = len(
        list(OUT_DIR.glob("*.npz"))
    )

    print(
        f"[{total_present:4d}/{len(population_ids)}] "
        f"{uid} "
        f"L={length:4d} "
        f"k={k:3d} "
        f"labels={K:2d} "
        f"time={elapsed:7.2f}s",
        flush=True,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

wall_time = time.time() - run_start

# Revalidate every output after production.
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

    final_completed.add(uid)


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
    round(wall_time, 2),
)

print()
print("Scientific outcome values printed: NO")
print("Model weights modified: NO")
print("Graph topology modified: NO")
print("Edge attributes modified: NO")
print("pLDDT used as model input: NO")
print("ESM used as structure input: NO")
print("Frozen benchmark modified: NO")
print("Final evaluation directory modified: NO")
print("=" * 78)
