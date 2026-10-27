from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parents[1] / "models"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from structure_gnn import StructureGNN


# ============================================================
# CONFIGURATION
# ============================================================

UID = "Q47149"

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

GRAPH_PATH = Path(
    "data/structures/structure_protocol_v1/"
    f"graphs/test/{UID}.npz"
)

ELIGIBLE_IDS_PATH = Path(
    "data/structures/structure_protocol_v1/"
    "test_structure_eligible_ids.txt"
)

SAVED_LOGITS_PATH = Path(
    "data/final_evaluation/"
    "structure_expert_v1_test/test_raw_logits.npy"
)

TEST_IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "labels/test_uniprot_ids.txt"
)

TEST_LABELS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "labels/test_labels.npy"
)

LABEL_MAP_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "primary_go_mf_labels.csv"
)

SITES_PATH = Path(
    "data/xai/functional_sites_v1/"
    "functional_sites_primary.csv"
)

OUT_DIR = Path(
    "data/xai/functional_sites_v1/"
    "sanity_q47149"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = (
    OUT_DIR
    / "structure_occlusion_q47149.csv"
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


def predict(
    model,
    x,
    edge_index,
    edge_attr,
):

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
print("E10 — Q47149 STRUCTURE RESIDUE OCCLUSION")
print("=" * 78)

print("UniProt:", UID)
print("Device:", DEVICE)
print("Primary attribution: signed raw-logit drop")
print("Perturbation: zero one residue's 21-D node identity")
print("Graph topology changed: NO")
print("Edge geometry changed: NO")
print("Structure temperature: 1.0")
print()


# ============================================================
# VERIFY FROZEN CHECKPOINT
# ============================================================

actual_sha = sha256_file(
    MODEL_PATH
)

if actual_sha != EXPECTED_MODEL_SHA256:

    raise RuntimeError(
        "Frozen structure checkpoint SHA mismatch.\n"
        f"Expected: {EXPECTED_MODEL_SHA256}\n"
        f"Actual:   {actual_sha}"
    )

print(
    "Frozen structure checkpoint SHA256: VERIFIED"
)


# ============================================================
# LOAD IDS / TRUE LABEL TARGETS
# ============================================================

test_ids = read_ids(
    TEST_IDS_PATH
)

eligible_ids = read_ids(
    ELIGIBLE_IDS_PATH
)

if UID not in test_ids:
    raise RuntimeError(
        f"{UID} is not in frozen TEST."
    )

if UID not in eligible_ids:
    raise RuntimeError(
        f"{UID} is not structure eligible."
    )

test_row = test_ids.index(
    UID
)

structure_row = eligible_ids.index(
    UID
)

Y = np.load(
    TEST_LABELS_PATH
)

if Y.shape != (
    len(test_ids),
    NUM_LABELS,
):
    raise RuntimeError(
        f"Unexpected TEST labels shape: {Y.shape}"
    )

positive_labels = np.flatnonzero(
    Y[test_row] == 1
)

if len(positive_labels) != 13:
    raise RuntimeError(
        "Expected 13 positive frozen labels, "
        f"got {len(positive_labels)}."
    )


label_map = (
    pd.read_csv(
        LABEL_MAP_PATH
    )
    .sort_values(
        "label_index"
    )
    .reset_index(
        drop=True
    )
)

if not np.array_equal(
    label_map["label_index"].to_numpy(),
    np.arange(NUM_LABELS),
):
    raise RuntimeError(
        "Frozen GO label indexing mismatch."
    )


print("Frozen TEST row:", test_row)
print("Structure prediction row:", structure_row)
print(
    "Positive frozen GO labels:",
    len(positive_labels),
)


# ============================================================
# LOAD CURATED SITES
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
        curated_positions.add(
            pos
        )

print(
    "Curated functional positions:",
    sorted(curated_positions),
)


# ============================================================
# LOAD EXACT FROZEN GRAPH
# ============================================================

if not GRAPH_PATH.exists():
    raise RuntimeError(
        f"Missing graph: {GRAPH_PATH}"
    )

with np.load(
    GRAPH_PATH,
    allow_pickle=False,
) as g:

    graph_uid = str(
        g["uniprot_id"].item()
    )

    if graph_uid != UID:
        raise RuntimeError(
            f"Graph ID mismatch: {graph_uid}"
        )

    x_np = g["x"].copy()
    edge_index_np = (
        g["edge_index"].copy()
    )
    edge_attr_np = (
        g["edge_attr"].copy()
    )


if x_np.ndim != 2:
    raise RuntimeError(
        f"Unexpected x shape: {x_np.shape}"
    )

if x_np.shape[1] != 21:
    raise RuntimeError(
        f"Expected node_dim=21, got {x_np.shape}"
    )

n_nodes = int(
    x_np.shape[0]
)

if n_nodes != 92:
    raise RuntimeError(
        f"Expected 92 graph nodes, got {n_nodes}."
    )

if edge_index_np.shape[0] != 2:
    raise RuntimeError(
        "edge_index first dimension must be 2."
    )

if edge_attr_np.shape[1] != 3:
    raise RuntimeError(
        "Expected 3 edge attributes."
    )

if edge_index_np.shape[1] != edge_attr_np.shape[0]:
    raise RuntimeError(
        "Edge count mismatch."
    )


# Critical perturbation assumption:
# clean node rows should encode one residue identity.
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
        "Clean node features are not one-hot "
        "residue identity rows."
    )


print("Graph nodes:", n_nodes)
print(
    "Directed edges:",
    edge_index_np.shape[1],
)
print(
    "Node feature shape:",
    x_np.shape,
)
print(
    "Edge attribute shape:",
    edge_attr_np.shape,
)
print(
    "Clean x row sums all 1:",
    True,
)


# ============================================================
# LOAD FROZEN STRUCTURE MODEL
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
    num_labels=494,
    num_layers=2,
    dropout=0.10,
).to(DEVICE)

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
        f"Unexpected parameter count: {num_parameters}"
    )

print()
print(
    "Frozen structure model loaded."
)
print(
    "Parameters:",
    num_parameters,
)


# ============================================================
# MOVE CLEAN GRAPH TO DEVICE
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
# CLEAN INFERENCE
# ============================================================

clean_logits = predict(
    model,
    x_clean,
    edge_index,
    edge_attr,
)


# ============================================================
# REPRODUCE SAVED FROZEN STRUCTURE LOGITS
# ============================================================

saved_logits = np.load(
    SAVED_LOGITS_PATH
)

expected_shape = (
    len(eligible_ids),
    NUM_LABELS,
)

if saved_logits.shape != expected_shape:
    raise RuntimeError(
        "Unexpected saved structure logits shape: "
        f"{saved_logits.shape}; "
        f"expected {expected_shape}"
    )

frozen_logits = (
    saved_logits[
        structure_row
    ]
    .astype(np.float32)
)

logit_diff = np.abs(
    clean_logits
    - frozen_logits
)

max_logit_diff = float(
    logit_diff.max()
)

mean_logit_diff = float(
    logit_diff.mean()
)

logits_match = np.allclose(
    clean_logits,
    frozen_logits,
    rtol=1e-5,
    atol=1e-6,
)


print()
print("Clean structure reproduction:")
print(
    "  max abs logit difference:",
    max_logit_diff,
)
print(
    "  mean abs logit difference:",
    mean_logit_diff,
)
print(
    "  allclose rtol=1e-5 atol=1e-6:",
    bool(logits_match),
)


if not logits_match:
    raise RuntimeError(
        "Clean structure inference does not "
        "reproduce frozen TEST logits."
    )


# ============================================================
# ONE-NODE IDENTITY OCCLUSION
# ============================================================

print()
print(
    "Running one-residue structure occlusion..."
)

records = []

for residue_index0 in range(
    n_nodes
):

    residue_position = (
        residue_index0 + 1
    )

    # Copy the CLEAN graph every time.
    # Only this node's 21-D identity is removed.
    x_masked = (
        x_clean
        .clone()
    )

    original_identity = (
        x_masked[
            residue_index0
        ]
        .clone()
    )

    x_masked[
        residue_index0
    ] = 0.0


    # Guard against accidental graph changes.
    if float(
        x_masked[
            residue_index0
        ].abs().sum().item()
    ) != 0.0:
        raise RuntimeError(
            "Node masking failed."
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
            "Original node was not one-hot."
        )


    masked_logits = predict(
        model,
        x_masked,
        edge_index,
        edge_attr,
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


        # Structure expert uses T = 1.
        clean_prob = float(
            torch.sigmoid(
                torch.tensor(
                    clean_logit
                )
            ).item()
        )

        masked_prob = float(
            torch.sigmoid(
                torch.tensor(
                    masked_logit
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
        or residue_position == n_nodes
    ):

        print(
            f"  {residue_position:3d}/"
            f"{n_nodes} residues"
        )


# ============================================================
# VALIDATE + SAVE
# ============================================================

result = pd.DataFrame(
    records
)

expected_rows = (
    n_nodes
    * len(positive_labels)
)

if len(result) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} rows, "
        f"got {len(result)}."
    )

numeric_cols = [
    "clean_logit",
    "masked_logit",
    "logit_drop",
    "clean_probability",
    "masked_probability",
    "probability_drop",
]

if not np.isfinite(
    result[
        numeric_cols
    ].to_numpy()
).all():
    raise RuntimeError(
        "Non-finite structure attribution output."
    )


result.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 78)
print("STRUCTURE OCCLUSION COMPLETE")
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
            "logit_drop",
            "is_curated_functional_site",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# CURATED RESIDUE RANK
# ============================================================

ranked = (
    result.copy()
)

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

print()
print("Curated residue 87 ranks:")

print(
    site87[
        [
            "label_index",
            "go_id",
            "logit_drop",
            "rank",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FINAL
# ============================================================

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