from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree


# ============================================================
# IMPORT FROZEN STRUCTURE MODEL
# ============================================================

MODEL_DIR = Path("src/models").resolve()

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from structure_gnn import StructureGNN


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_LABELS = 494

MODEL_PATH = Path(
    "data/models/structure_gnn_v1_full/final_model.pt"
)

EXPECTED_MODEL_SHA256 = (
    "f1434b60a67e7dd695f289f7094d528b"
    "b91a28f249b28efd0d26af28d698991e"
)

# IMPORTANT:
# E6 is evaluated on the development VAL population,
# not used as a new confirmatory TEST experiment.

GRAPH_DIR = Path(
    "data/structures/structure_protocol_v1/graphs/val"
)

OUTPUT_DIR = Path(
    "data/robustness/structural_degradation_v1"
)

PROTOCOL_PATH = (
    OUTPUT_DIR / "STRUCTURAL_DEGRADATION_PROTOCOL_V1.txt"
)

EXPECTED_PROTOCOL_SHA256 = (
    "1b8b8a32a199df7138ac3f3d8be0ce59"
    "c90ebc429574f321c671be0ec7b18fef"
)

SIGMAS = [
    0.5,
    1.0,
    2.0,
]

GLOBAL_SEED = 42

SPATIAL_CUTOFF = 10.0


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def deterministic_seed(
    uniprot_id,
    sigma,
):

    text = (
        f"{GLOBAL_SEED}|"
        f"{uniprot_id}|"
        f"{sigma:.6f}"
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def rebuild_graph(coords):

    n = coords.shape[0]

    tree = cKDTree(
        coords
    )

    spatial_pairs = tree.query_pairs(
        r=SPATIAL_CUTOFF,
        output_type="set",
    )

    pairs = set(
        spatial_pairs
    )

    # Sequential edges are always retained.
    for i in range(n - 1):
        pairs.add(
            (i, i + 1)
        )

    pairs = sorted(
        pairs
    )

    num_pairs = len(
        pairs
    )

    edge_index = np.empty(
        (
            2,
            num_pairs * 2,
        ),
        dtype=np.int64,
    )

    edge_attr = np.empty(
        (
            num_pairs * 2,
            3,
        ),
        dtype=np.float32,
    )

    k = 0

    for i, j in pairs:

        distance = float(
            np.linalg.norm(
                coords[i] - coords[j]
            )
        )

        sequential = float(
            abs(i - j) == 1
        )

        separation = float(
            abs(i - j)
        )

        feature = (
            distance,
            sequential,
            separation,
        )

        edge_index[:, k] = (
            i,
            j,
        )

        edge_attr[k] = feature

        k += 1

        edge_index[:, k] = (
            j,
            i,
        )

        edge_attr[k] = feature

        k += 1

    return (
        edge_index,
        edge_attr,
        num_pairs,
    )


def sigmoid(logits):

    return (
        1.0
        /
        (
            1.0
            + np.exp(
                -np.clip(
                    logits,
                    -80.0,
                    80.0,
                )
            )
        )
    ).astype(
        np.float32
    )


# ============================================================
# VERIFY FROZEN PROTOCOL
# ============================================================

print("=" * 88)
print("STRUCTURAL DEGRADATION V1")
print("=" * 88)

print("Device:", DEVICE)
print()

if not PROTOCOL_PATH.exists():
    raise RuntimeError(
        "Frozen E6 protocol file is missing."
    )

actual_protocol_sha = sha256_file(
    PROTOCOL_PATH
)

if (
    actual_protocol_sha
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "E6 protocol SHA256 mismatch.\n"
        f"Expected: {EXPECTED_PROTOCOL_SHA256}\n"
        f"Actual:   {actual_protocol_sha}"
    )

print(
    "E6 protocol SHA256 verified:",
    actual_protocol_sha,
)


# ============================================================
# VERIFY FROZEN MODEL
# ============================================================

actual_model_sha = sha256_file(
    MODEL_PATH
)

if (
    actual_model_sha
    != EXPECTED_MODEL_SHA256
):
    raise RuntimeError(
        "Frozen structure checkpoint SHA256 mismatch.\n"
        f"Expected: {EXPECTED_MODEL_SHA256}\n"
        f"Actual:   {actual_model_sha}"
    )

print(
    "Structure checkpoint SHA256 verified:",
    actual_model_sha,
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
    node_dim=21,
    hidden_dim=128,
    num_labels=NUM_LABELS,
    num_layers=2,
    dropout=0.10,
).to(
    DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

if num_parameters != 298990:
    raise RuntimeError(
        f"Unexpected parameter count: "
        f"{num_parameters:,}"
    )

print(
    "Frozen structure model loaded."
)

print(
    "Parameters:",
    f"{num_parameters:,}",
)


# ============================================================
# DISCOVER DEVELOPMENT GRAPHS
# ============================================================

graph_paths = sorted(
    GRAPH_DIR.glob("*.npz")
)

if not graph_paths:
    raise RuntimeError(
        f"No graphs found in {GRAPH_DIR}"
    )

print()
print(
    "Development structure graphs:",
    f"{len(graph_paths):,}",
)

print(
    "Noise levels:",
    SIGMAS,
)

print(
    "Spatial cutoff:",
    SPATIAL_CUTOFF,
    "Angstrom",
)


# ============================================================
# RUN EACH FROZEN PERTURBATION LEVEL
#
# One graph at a time is intentional:
# simple, deterministic, and safe for the laptop GPU.
# ============================================================

for sigma in SIGMAS:

    print()
    print("=" * 88)
    print(
        f"SIGMA = {sigma:.1f} Angstrom"
    )
    print("=" * 88)

    sigma_dir = (
        OUTPUT_DIR
        / f"sigma_{sigma:.1f}"
    )

    sigma_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_logits = []
    all_ids = []

    original_pair_counts = []
    perturbed_pair_counts = []

    start = time.time()

    with torch.inference_mode():

        for number, graph_path in enumerate(
            graph_paths,
            start=1,
        ):

            with np.load(
                graph_path,
                allow_pickle=False,
            ) as graph:

                x = graph[
                    "x"
                ].astype(
                    np.float32,
                    copy=False,
                )

                coords = graph[
                    "coords"
                ].astype(
                    np.float32,
                    copy=True,
                )

                original_edge_index = graph[
                    "edge_index"
                ]

                uid = str(
                    graph["uniprot_id"].item()
                )

            if x.shape[0] != coords.shape[0]:
                raise RuntimeError(
                    f"Node/coordinate mismatch: {uid}"
                )

            if x.shape[1] != 21:
                raise RuntimeError(
                    f"Unexpected node dimension: {uid}"
                )

            seed = deterministic_seed(
                uid,
                sigma,
            )

            rng = np.random.default_rng(
                seed
            )

            noise = rng.normal(
                loc=0.0,
                scale=sigma,
                size=coords.shape,
            ).astype(
                np.float32
            )

            perturbed_coords = (
                coords + noise
            ).astype(
                np.float32
            )

            (
                edge_index,
                edge_attr,
                num_pairs,
            ) = rebuild_graph(
                perturbed_coords
            )

            original_pairs = (
                original_edge_index.shape[1]
                // 2
            )

            original_pair_counts.append(
                original_pairs
            )

            perturbed_pair_counts.append(
                num_pairs
            )

            x_t = torch.from_numpy(
                x
            ).to(
                DEVICE
            )

            edge_index_t = torch.from_numpy(
                edge_index
            ).to(
                DEVICE
            )

            edge_attr_t = torch.from_numpy(
                edge_attr
            ).to(
                DEVICE
            )

            batch_index = torch.zeros(
                x.shape[0],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = model(
                x=x_t,
                edge_index=edge_index_t,
                edge_attr=edge_attr_t,
                batch_index=batch_index,
                num_graphs=1,
            )

            all_logits.append(
                logits.cpu().numpy()[0]
            )

            all_ids.append(
                uid
            )

            if (
                number % 500 == 0
                or number == len(graph_paths)
            ):
                print(
                    f"Processed "
                    f"{number:,}/"
                    f"{len(graph_paths):,}"
                )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    runtime = (
        time.time()
        - start
    )

    logits = np.stack(
        all_logits,
        axis=0,
    ).astype(
        np.float32
    )

    probs = sigmoid(
        logits
    )

    ids_path = (
        sigma_dir
        / "uniprot_ids.txt"
    )

    logits_path = (
        sigma_dir
        / "structure_logits.npy"
    )

    probs_path = (
        sigma_dir
        / "structure_probabilities.npy"
    )

    metadata_path = (
        sigma_dir
        / "metadata.json"
    )

    ids_path.write_text(
        "\n".join(all_ids) + "\n",
        encoding="utf-8",
    )

    np.save(
        logits_path,
        logits,
    )

    np.save(
        probs_path,
        probs,
    )

    original_pair_counts = np.asarray(
        original_pair_counts,
        dtype=np.int64,
    )

    perturbed_pair_counts = np.asarray(
        perturbed_pair_counts,
        dtype=np.int64,
    )

    metadata = {
        "experiment":
            "structural_degradation_v1",

        "status":
            "post-confirmatory development robustness experiment",

        "population":
            "structure_protocol_v1 VAL graphs",

        "sigma_angstrom":
            sigma,

        "global_seed":
            GLOBAL_SEED,

        "spatial_cutoff_angstrom":
            SPATIAL_CUTOFF,

        "proteins":
            len(all_ids),

        "labels":
            NUM_LABELS,

        "checkpoint_sha256":
            actual_model_sha,

        "protocol_sha256":
            actual_protocol_sha,

        "calibration_temperature":
            1.0,

        "residue_identity_changed":
            False,

        "plddt_model_input":
            False,

        "raw_xyz_model_input":
            False,

        "graph_rebuilt_after_noise":
            True,

        "mean_original_undirected_pairs":
            float(
                original_pair_counts.mean()
            ),

        "mean_perturbed_undirected_pairs":
            float(
                perturbed_pair_counts.mean()
            ),

        "mean_pair_count_change":
            float(
                (
                    perturbed_pair_counts
                    - original_pair_counts
                ).mean()
            ),

        "runtime_seconds":
            runtime,

        "ids_sha256":
            sha256_file(ids_path),

        "logits_sha256":
            sha256_file(logits_path),

        "probabilities_sha256":
            sha256_file(probs_path),

        "logits_shape":
            list(logits.shape),

        "probabilities_shape":
            list(probs.shape),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "Proteins:",
        len(all_ids)
    )

    print(
        "Logits:",
        logits.shape
    )

    print(
        "Probabilities:",
        probs.shape
    )

    print(
        "Mean original pairs:",
        float(
            original_pair_counts.mean()
        )
    )

    print(
        "Mean perturbed pairs:",
        float(
            perturbed_pair_counts.mean()
        )
    )

    print(
        f"Runtime: "
        f"{runtime / 60:.2f} min"
    )

    print(
        "Saved:",
        sigma_dir
    )


print()
print("=" * 88)
print("STRUCTURAL DEGRADATION INFERENCE COMPLETE")
print("=" * 88)

print(
    "No model retraining performed."
)

print(
    "No perturbation outcomes used for tuning."
)
