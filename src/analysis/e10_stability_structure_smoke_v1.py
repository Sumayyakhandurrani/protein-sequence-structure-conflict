from pathlib import Path
import hashlib
import sys

import numpy as np
import torch
from scipy.stats import spearmanr


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

TEMPERATURE = 1.0

GLOBAL_SEED = 42
TOLERANCE = 0.01

EXPECTED_PARAMETER_COUNT = 298990

EXPECTED_MODEL_SHA = (
    "f1434b60a67e7dd695f289f7094d528b"
    "b91a28f249b28efd0d26af28d698991e"
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

MODEL_PATH = (
    BASE
    / "models/structure_gnn_v1_full/"
      "final_model.pt"
)

GRAPH_DIR = (
    BASE
    / "structures/structure_protocol_v1/"
      "graphs/test"
)

ATTR_DIR = (
    BASE
    / "xai/functional_sites_v1/"
      "structure_attributions_v1/"
      "per_protein"
)


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
        f"{GLOBAL_SEED}|{uid}|structure"
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


def predict(
    model,
    x,
    edge_index,
    edge_attr,
):

    batch = torch.zeros(
        x.shape[0],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():

        logits = model(
            x,
            edge_index,
            edge_attr,
            batch,
            1,
        )

    return (
        logits[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
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
# START
# ============================================================

if DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA is required."
    )


if sha256_file(
    MODEL_PATH
) != EXPECTED_MODEL_SHA:

    raise RuntimeError(
        "Structure checkpoint SHA mismatch."
    )


ids = [
    x.strip()
    for x in POPULATION_PATH
    .read_text()
    .splitlines()
    if x.strip()
]

uid = ids[0]


graph_path = (
    GRAPH_DIR / f"{uid}.npz"
)

attr_path = (
    ATTR_DIR / f"{uid}.npz"
)


# ============================================================
# LOAD GRAPH
# ============================================================

with np.load(
    graph_path,
    allow_pickle=False,
) as g:

    x_np = np.asarray(
        g["x"],
        dtype=np.float32,
    )

    edge_index_np = np.asarray(
        g["edge_index"],
        dtype=np.int64,
    )

    edge_attr_np = np.asarray(
        g["edge_attr"],
        dtype=np.float32,
    )

    graph_uid = str(
        g["uniprot_id"].item()
    )


if graph_uid != uid:
    raise RuntimeError(
        "Graph ID mismatch."
    )


L = x_np.shape[0]


if x_np.shape != (
    L,
    NODE_DIM,
):
    raise RuntimeError(
        "Unexpected node feature shape."
    )


if not np.allclose(
    x_np.sum(axis=1),
    1.0,
    rtol=0,
    atol=1e-6,
):
    raise RuntimeError(
        "Clean node features are not one-hot."
    )


# ============================================================
# LOAD FROZEN REFERENCE ATTRIBUTION
# ============================================================

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


if clean_attribution.shape != (
    len(positive_labels),
    L,
):
    raise RuntimeError(
        "Reference attribution shape mismatch."
    )


# ============================================================
# FROZEN CANDIDATE SELECTION
# ============================================================

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
print("STRUCTURE STABILITY V1 — ONE-PROTEIN SMOKE TEST")
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
        "Unexpected checkpoint epoch."
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
        "Unexpected structure parameter count."
    )


# ============================================================
# DEVICE GRAPH
# ============================================================

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


# ============================================================
# CLEAN REPRODUCTION
# ============================================================

clean_logits = predict(
    model,
    x_clean,
    edge_index,
    edge_attr,
)


clean_target_reproduced = (
    clean_logits[
        positive_labels
    ]
)


clean_diff = float(
    np.max(
        np.abs(
            clean_target_reproduced
            - clean_target_logits
        )
    )
)


print()
print(
    "Clean target-logit max abs diff:",
    f"{clean_diff:.3e}",
)


if not np.allclose(
    clean_target_reproduced,
    clean_target_logits,
    rtol=1e-5,
    atol=1e-5,
):
    raise RuntimeError(
        "Clean structure logits failed reproduction."
    )


# ============================================================
# GROUP TARGETS BY BASE POSITION
# ============================================================

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


# ============================================================
# PERTURBED EXPLANATIONS
# ============================================================

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


    # --------------------------------------------------------
    # Base perturbation
    # --------------------------------------------------------

    x_base = x_clean.clone()

    x_base[
        base_pos
    ] = 0.0


    base_logits_all = predict(
        model,
        x_base,
        edge_index,
        edge_attr,
    )


    base_target_logits = (
        base_logits_all[
            group_labels
        ]
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
            "Base structure perturbation "
            "failed reproduction."
        )


    # --------------------------------------------------------
    # Recompute explanation on base-perturbed input
    # --------------------------------------------------------

    K = len(
        group_labels
    )

    perturbed_attr = np.full(
        (
            K,
            L,
        ),
        np.nan,
        dtype=np.float32,
    )


    for second_pos in range(
        L
    ):

        if second_pos == base_pos:
            continue


        x_double = (
            x_base.clone()
        )

        x_double[
            second_pos
        ] = 0.0


        double_logits = predict(
            model,
            x_double,
            edge_index,
            edge_attr,
        )


        perturbed_attr[
            :,
            second_pos
        ] = (
            base_target_logits
            - double_logits[
                group_labels
            ]
        )


    # --------------------------------------------------------
    # Stability metrics
    # --------------------------------------------------------

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


# ============================================================
# SUMMARY
# ============================================================

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
    "Frozen model modified: NO"
)

print(
    "Frozen attribution files modified: NO"
)
