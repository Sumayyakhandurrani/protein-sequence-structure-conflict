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

MODEL_CODE_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
)

if str(MODEL_CODE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODEL_CODE_DIR),
    )

from structure_gnn import StructureGNN


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

MODEL_PATH = Path(
    "data/models/structure_gnn_v1_full/"
    "final_model.pt"
)

GRAPH_DIR = Path(
    "data/structures/structure_protocol_v1/"
    "graphs/test"
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
    "structure_expert_v1_test/"
    "test_raw_logits.npy"
)

POPULATION_PATH = Path(
    "data/xai/functional_sites_v1/"
    "xai_attribution_eligible_ids.txt"
)

OUT_ROOT = Path(
    "data/xai/functional_sites_v1/"
    "structure_attributions_v1"
)

OUT_DIR = (
    OUT_ROOT / "per_protein"
)

RUN_INFO_PATH = (
    OUT_ROOT / "run_info.json"
)


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
        logits
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


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

if (
    args.protein is not None
    and args.limit is not None
):
    raise RuntimeError(
        "--protein and --limit are mutually exclusive."
    )

if (
    args.limit is not None
    and args.limit <= 0
):
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
print("E10 — STRUCTURE ATTRIBUTION POPULATION V1")
print("=" * 78)

print("Device:", DEVICE)
print("Frozen GO-MF labels:", NUM_LABELS)

print(
    "Perturbation:",
    "zero one residue's 21-D node identity",
)

print(
    "Graph topology changed: NO"
)

print(
    "Edge attributes changed: NO"
)

print(
    "Structure temperature:",
    STRUCTURE_TEMPERATURE,
)

print(
    "Primary attribution:",
    "signed clean raw logit - masked raw logit",
)

print()


# ============================================================
# VERIFY REQUIRED FILES
# ============================================================

required_files = [
    MODEL_PATH,
    STRUCTURE_IDS_PATH,
    TEST_IDS_PATH,
    TEST_LABELS_PATH,
    SAVED_LOGITS_PATH,
    POPULATION_PATH,
]

for path in required_files:

    if not path.exists():

        raise RuntimeError(
            f"Missing required file: {path}"
        )


# ============================================================
# VERIFY FROZEN CHECKPOINT
# ============================================================

actual_model_sha = sha256_file(
    MODEL_PATH
)

if (
    actual_model_sha
    != EXPECTED_MODEL_SHA256
):

    raise RuntimeError(
        "Frozen structure checkpoint SHA mismatch.\n"
        f"Expected: {EXPECTED_MODEL_SHA256}\n"
        f"Actual:   {actual_model_sha}"
    )

print(
    "Frozen structure checkpoint SHA256: VERIFIED"
)


# ============================================================
# LOAD FROZEN POPULATIONS
# ============================================================

test_ids = read_ids(
    TEST_IDS_PATH
)

structure_ids = read_ids(
    STRUCTURE_IDS_PATH
)

population_ids = read_ids(
    POPULATION_PATH
)


if len(population_ids) != 2447:

    raise RuntimeError(
        "Unexpected frozen XAI population size: "
        f"{len(population_ids)}"
    )


if len(set(population_ids)) != len(
    population_ids
):

    raise RuntimeError(
        "Duplicate IDs in frozen XAI population."
    )


if len(set(structure_ids)) != len(
    structure_ids
):

    raise RuntimeError(
        "Duplicate structure eligible IDs."
    )


test_row_by_id = {
    uid: i
    for i, uid in enumerate(test_ids)
}

structure_row_by_id = {
    uid: i
    for i, uid in enumerate(structure_ids)
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

    graph_path = (
        GRAPH_DIR / f"{uid}.npz"
    )

    if not graph_path.exists():

        raise RuntimeError(
            f"Missing graph for {uid}: "
            f"{graph_path}"
        )


print(
    "Frozen XAI population:",
    len(population_ids),
)

print(
    "Structure TEST population:",
    len(structure_ids),
)


# ============================================================
# LOAD FROZEN LABELS / LOGITS
# ============================================================

Y = np.load(
    TEST_LABELS_PATH,
    mmap_mode="r",
)

if Y.shape != (
    len(test_ids),
    NUM_LABELS,
):

    raise RuntimeError(
        "Unexpected TEST label shape: "
        f"{Y.shape}"
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
# VERIFY TARGET COUNTS
# ============================================================

total_targets = 0

for uid in population_ids:

    test_row = test_row_by_id[
        uid
    ]

    k = int(
        np.count_nonzero(
            Y[test_row] == 1
        )
    )

    if k <= 0:

        raise RuntimeError(
            f"{uid} has zero positive labels."
        )

    total_targets += k


if total_targets != 23087:

    raise RuntimeError(
        "Unexpected total positive target count: "
        f"{total_targets}"
    )


print(
    "Positive protein-label targets:",
    total_targets,
)


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

if (
    num_parameters
    != EXPECTED_PARAMETER_COUNT
):

    raise RuntimeError(
        "Unexpected parameter count: "
        f"{num_parameters}"
    )


print(
    "Frozen structure model loaded."
)

print(
    "Parameters:",
    num_parameters,
)


# ============================================================
# VALIDATE-ONLY MODE
# ============================================================

if args.validate_only:

    print()
    print(
        "VALIDATION ONLY: PASS"
    )

    print(
        "No attribution files written."
    )

    print("=" * 78)

    raise SystemExit(0)


# ============================================================
# CHOOSE POPULATION
# ============================================================

if args.protein is not None:

    uid = args.protein.strip()

    if uid not in population_ids:

        raise RuntimeError(
            f"{uid} is not in the frozen "
            "XAI attribution population."
        )

    requested_ids = [
        uid
    ]

elif args.limit is not None:

    requested_ids = (
        population_ids[
            :args.limit
        ]
    )

else:

    requested_ids = list(
        population_ids
    )


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


completed_before = {
    p.stem
    for p in OUT_DIR.glob(
        "*.npz"
    )
}


todo = [
    uid
    for uid in requested_ids
    if uid not in completed_before
]


print()
print(
    "Requested proteins:",
    len(requested_ids),
)

print(
    "Already completed:",
    len(requested_ids)
    - len(todo),
)

print(
    "To process:",
    len(todo),
)


if len(todo) == 0:

    print()
    print(
        "Nothing to do."
    )

    print("=" * 78)

    raise SystemExit(0)


# ============================================================
# RUN INFO — WRITTEN BEFORE COMPUTATION
# ============================================================

run_info = {
    "method":
        "structure_attribution_v1",

    "population_file":
        str(POPULATION_PATH),

    "population_sha256":
        sha256_file(
            POPULATION_PATH
        ),

    "population_size":
        len(population_ids),

    "total_positive_targets":
        total_targets,

    "model_path":
        str(MODEL_PATH),

    "model_sha256":
        actual_model_sha,

    "checkpoint_epoch":
        int(checkpoint["epoch"]),

    "num_parameters":
        int(num_parameters),

    "node_dim":
        NODE_DIM,

    "hidden_dim":
        HIDDEN_DIM,

    "num_layers":
        NUM_LAYERS,

    "dropout":
        DROPOUT,

    "num_labels":
        NUM_LABELS,

    "temperature":
        STRUCTURE_TEMPERATURE,

    "logit_rtol":
        LOGIT_RTOL,

    "logit_atol":
        LOGIT_ATOL,

    "perturbation":
        "zero selected residue 21-D node identity",

    "graph_topology_changed":
        False,

    "edge_attributes_changed":
        False,

    "plddt_model_input":
        False,

    "esm_model_input":
        False,

    "primary_score":
        "clean_raw_logit_minus_masked_raw_logit",

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


for number, uid in enumerate(
    todo,
    start=1,
):

    protein_start = time.time()

    test_row = (
        test_row_by_id[
            uid
        ]
    )

    structure_row = (
        structure_row_by_id[
            uid
        ]
    )

    positive_labels = (
        np.flatnonzero(
            Y[test_row] == 1
        )
        .astype(np.int32)
    )


    graph_path = (
        GRAPH_DIR / f"{uid}.npz"
    )


    with np.load(
        graph_path,
        allow_pickle=False,
    ) as g:

        graph_uid = str(
            g["uniprot_id"].item()
        )

        if graph_uid != uid:

            raise RuntimeError(
                f"{uid}: graph ID mismatch "
                f"({graph_uid})"
            )

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


    # --------------------------------------------------------
    # GRAPH QC
    # --------------------------------------------------------

    if (
        x_np.ndim != 2
        or x_np.shape[1] != NODE_DIM
    ):

        raise RuntimeError(
            f"{uid}: unexpected x shape "
            f"{x_np.shape}"
        )


    n_nodes = int(
        x_np.shape[0]
    )


    if n_nodes <= 0:

        raise RuntimeError(
            f"{uid}: zero graph nodes."
        )


    if (
        edge_index_np.ndim != 2
        or edge_index_np.shape[0] != 2
    ):

        raise RuntimeError(
            f"{uid}: invalid edge_index shape "
            f"{edge_index_np.shape}"
        )


    if (
        edge_attr_np.ndim != 2
        or edge_attr_np.shape[1] != 3
    ):

        raise RuntimeError(
            f"{uid}: invalid edge_attr shape "
            f"{edge_attr_np.shape}"
        )


    if (
        edge_index_np.shape[1]
        != edge_attr_np.shape[0]
    ):

        raise RuntimeError(
            f"{uid}: edge count mismatch."
        )


    row_sums = x_np.sum(
        axis=1
    )


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


    # --------------------------------------------------------
    # DEVICE TENSORS
    # --------------------------------------------------------

    x_clean = torch.from_numpy(
        x_np
    ).float().to(
        DEVICE
    )

    edge_index = torch.from_numpy(
        edge_index_np
    ).long().to(
        DEVICE
    )

    edge_attr = torch.from_numpy(
        edge_attr_np
    ).float().to(
        DEVICE
    )


    # --------------------------------------------------------
    # CLEAN INFERENCE
    # --------------------------------------------------------

    clean_logits = predict(
        model,
        x_clean,
        edge_index,
        edge_attr,
        DEVICE,
    )


    frozen_logits = np.asarray(
        saved_logits[
            structure_row
        ],
        dtype=np.float32,
    )


    diff = np.abs(
        clean_logits
        - frozen_logits
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
            * np.abs(
                frozen_logits
            )
        )

        ratio = (
            diff / allowed
        )

        worst_label = int(
            np.argmax(
                ratio
            )
        )

        raise RuntimeError(
            f"{uid}: clean structure logits "
            "do not reproduce frozen logits. "
            f"max_diff={max_logit_diff}, "
            f"worst_label={worst_label}, "
            f"worst_ratio="
            f"{float(ratio[worst_label])}"
        )


    # --------------------------------------------------------
    # ATTRIBUTION ARRAY
    #
    # Shape:
    #     K positive labels x L residues
    # --------------------------------------------------------

    k_labels = len(
        positive_labels
    )

    attribution = np.empty(
        (
            k_labels,
            n_nodes,
        ),
        dtype=np.float32,
    )


    clean_target_logits = (
        clean_logits[
            positive_labels
        ]
        .astype(
            np.float32,
            copy=True,
        )
    )


    # --------------------------------------------------------
    # ONE-RESIDUE IDENTITY OCCLUSION
    # --------------------------------------------------------

    for residue_index0 in range(
        n_nodes
    ):

        x_masked = (
            x_clean.clone()
        )

        original_identity = (
            x_masked[
                residue_index0
            ]
            .clone()
        )


        if not torch.allclose(
            original_identity.sum(),
            torch.tensor(
                1.0,
                device=DEVICE,
            ),
            rtol=0,
            atol=1e-6,
        ):

            raise RuntimeError(
                f"{uid}: residue "
                f"{residue_index0 + 1} "
                "was not one-hot before masking."
            )


        x_masked[
            residue_index0
        ] = 0.0


        if float(
            x_masked[
                residue_index0
            ]
            .abs()
            .sum()
            .item()
        ) != 0.0:

            raise RuntimeError(
                f"{uid}: node masking failed "
                f"at residue "
                f"{residue_index0 + 1}."
            )


        masked_logits = predict(
            model,
            x_masked,
            edge_index,
            edge_attr,
            DEVICE,
        )


        attribution[
            :,
            residue_index0
        ] = (
            clean_target_logits
            - masked_logits[
                positive_labels
            ]
        )


    # --------------------------------------------------------
    # FINAL PER-PROTEIN QC
    # --------------------------------------------------------

    if attribution.shape != (
        k_labels,
        n_nodes,
    ):

        raise RuntimeError(
            f"{uid}: attribution shape error."
        )


    if not np.isfinite(
        attribution
    ).all():

        raise RuntimeError(
            f"{uid}: non-finite attribution."
        )


    if not np.isfinite(
        clean_target_logits
    ).all():

        raise RuntimeError(
            f"{uid}: non-finite clean logits."
        )


    # --------------------------------------------------------
    # ATOMIC SAVE
    # --------------------------------------------------------

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
            n_nodes,
            dtype=np.int32,
        ),

        positive_label_indices=(
            positive_labels
        ),

        attribution_logit_drop=(
            attribution
        ),

        clean_target_logits=(
            clean_target_logits
        ),

        clean_logit_max_abs_diff=(
            np.asarray(
                max_logit_diff,
                dtype=np.float32,
            )
        ),

        clean_logit_mean_abs_diff=(
            np.asarray(
                mean_logit_diff,
                dtype=np.float32,
            )
        ),
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
        f"L={n_nodes} "
        f"K={k_labels} "
        f"maxdiff={max_logit_diff:.3e} "
        f"time={elapsed:.2f}s "
        f"present={total_present}",
        flush=True,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

wall_time = (
    time.time()
    - run_start
)

final_files = list(
    OUT_DIR.glob(
        "*.npz"
    )
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
    "Total attribution files now present:",
    len(final_files),
)

print(
    "This run processed:",
    processed_this_run,
)

print(
    "This run wall time (seconds):",
    round(
        wall_time,
        2,
    ),
)

print(
    "Model weights modified: NO"
)

print(
    "Graph topology modified: NO"
)

print(
    "Edge attributes modified: NO"
)

print(
    "pLDDT used as model input: NO"
)

print(
    "ESM used as structure input: NO"
)

print(
    "Frozen benchmark modified: NO"
)

print(
    "Final evaluation directory modified: NO"
)

print(
    "Attribution method changed: NO"
)

print("=" * 78)
