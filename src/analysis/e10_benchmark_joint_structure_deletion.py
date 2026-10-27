from pathlib import Path
import hashlib
import math
import sys
import time

import numpy as np
import torch


# ============================================================
# IMPORT EXACT FROZEN STRUCTURE MODEL
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
# TIMING-ONLY CONFIGURATION
# ============================================================

REPRESENTATIVES = [
    ("SHORT",     "P17609"),
    ("MEDIAN",    "P40160"),
    ("NEAR_1000", "O14983"),
    ("LONG",      "O60312"),
    ("VERY_LONG", "Q8CC88"),
]

NUM_LABELS = 494
NODE_DIM = 21
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_BENCH = 32


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


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


print("=" * 78)
print("E10 STRUCTURE JOINT-DELETION TIMING BENCHMARK")
print("=" * 78)

print("Device:", DEVICE)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA GPU required."
    )


model_sha = sha256_file(
    MODEL_PATH
)

print(
    "Checkpoint SHA verified:",
    model_sha
    == EXPECTED_MODEL_SHA256,
)

if model_sha != EXPECTED_MODEL_SHA256:
    raise RuntimeError(
        "Frozen structure checkpoint SHA mismatch."
    )


# ============================================================
# LOAD EXACT FROZEN MODEL
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

print(
    "Parameter count:",
    num_parameters,
)

if num_parameters != EXPECTED_PARAMETER_COUNT:
    raise RuntimeError(
        "Unexpected structure parameter count."
    )


# ============================================================
# BENCHMARK
# ============================================================

print()
print("=" * 78)
print("TIMING RESULTS")
print("=" * 78)


for group, uid in REPRESENTATIVES:

    graph_path = (
        GRAPH_DIR / f"{uid}.npz"
    )

    if not graph_path.exists():
        raise RuntimeError(
            f"Missing graph: {graph_path}"
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
                f"{uid}: graph ID mismatch."
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


    if (
        x_np.ndim != 2
        or x_np.shape[1] != NODE_DIM
    ):
        raise RuntimeError(
            f"{uid}: invalid node matrix."
        )

    L = int(
        x_np.shape[0]
    )

    if not np.allclose(
        x_np.sum(axis=1),
        1.0,
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"{uid}: node rows are not one-hot."
        )

    if (
        edge_index_np.ndim != 2
        or edge_index_np.shape[0] != 2
    ):
        raise RuntimeError(
            f"{uid}: invalid edge_index."
        )

    if (
        edge_attr_np.ndim != 2
        or edge_attr_np.shape[1] != 3
    ):
        raise RuntimeError(
            f"{uid}: invalid edge_attr."
        )

    if (
        edge_index_np.shape[1]
        != edge_attr_np.shape[0]
    ):
        raise RuntimeError(
            f"{uid}: edge count mismatch."
        )


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


    k = max(
        1,
        math.ceil(
            0.10 * L
        ),
    )


    # Timing-only deterministic residue sets.
    seed_material = (
        f"42|{uid}|TIMING_ONLY"
    ).encode()

    seed = int.from_bytes(
        hashlib.sha256(
            seed_material
        ).digest()[:8],
        "little",
    ) % (2**32)

    rng = np.random.default_rng(
        seed
    )

    deletion_sets = [
        np.sort(
            rng.choice(
                L,
                size=k,
                replace=False,
            )
        )
        for _ in range(
            N_BENCH
        )
    ]


    # Warm-up.
    warm = x_clean.clone()

    warm[
        torch.from_numpy(
            deletion_sets[0]
        ).long().to(DEVICE)
    ] = 0.0

    _ = predict(
        model,
        warm,
        edge_index,
        edge_attr,
        DEVICE,
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    start = time.perf_counter()

    for positions in deletion_sets:

        x_masked = (
            x_clean.clone()
        )

        position_tensor = (
            torch.from_numpy(
                np.asarray(
                    positions,
                    dtype=np.int64,
                )
            )
            .long()
            .to(DEVICE)
        )

        # Exact frozen structure perturbation,
        # now applied jointly to the selected 10%.
        x_masked[
            position_tensor
        ] = 0.0

        # Perturbation QC.
        if float(
            x_masked[
                position_tensor
            ]
            .abs()
            .sum()
            .item()
        ) != 0.0:
            raise RuntimeError(
                f"{uid}: joint node masking failed."
            )

        _ = predict(
            model,
            x_masked,
            edge_index,
            edge_attr,
            DEVICE,
        )


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
    print(
        f"{group:10s} {uid:12s} "
        f"nodes={L:4d} "
        f"edges={edge_index_np.shape[1]:7d} "
        f"k={k:3d}"
    )

    print(
        f"  time={elapsed:.3f}s  "
        f"rate={rate:.3f}/s  "
        f"peak={peak_gb:.3f} GB"
    )


print()
print("=" * 78)
print("BENCHMARK COMPLETE")
print(
    "No deletion drops or faithfulness outcomes "
    "were calculated."
)
print("=" * 78)
