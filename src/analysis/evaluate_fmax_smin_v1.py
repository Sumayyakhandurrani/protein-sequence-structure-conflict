from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

PROTOCOL = Path(
    "data/evaluation/fmax_smin_v1/FMAX_SMIN_PROTOCOL_V2.txt"
)

EXPECTED_PROTOCOL_SHA256 = (
    "6f875ece2913eab837b0181a6fc4d08d6bf53cbb513d202f2522fe749c2fc751"
)

LABEL_FILE = Path(
    "data/splits/mmseqs30_graph_v2/primary_go_mf_labels.csv"
)

TRAIN_LABELS_FILE = Path(
    "data/splits/mmseqs30_graph_v2/labels/train_labels.npy"
)

TEST_ROOT = Path(
    "data/final_evaluation/final_test_primary_v1"
)

OUT = Path(
    "data/evaluation/fmax_smin_v1/results"
)

N_LABELS = 494
N_TRAIN = 38031
N_TEST = 4731

THRESHOLDS = np.arange(
    0,
    101,
    dtype=np.float64,
) / 100.0


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def load_probability_array(path):
    x = np.load(path)

    if x.shape != (N_TEST, N_LABELS):
        raise RuntimeError(
            f"{path}: unexpected shape {x.shape}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{path}: non-finite probabilities"
        )

    if x.min() < 0 or x.max() > 1:
        raise RuntimeError(
            f"{path}: probabilities outside [0,1]"
        )

    return x.astype(
        np.float64,
        copy=False,
    )


# ============================================================
# METRIC IMPLEMENTATION
# ============================================================

def evaluate_model(
    name,
    probs,
    y_true,
    information_content,
):

    rows = []

    true_count = y_true.sum(
        axis=1
    )

    if np.any(true_count == 0):
        raise RuntimeError(
            "At least one TEST protein has zero "
            "positive labels in frozen label space."
        )

    for threshold in THRESHOLDS:

        pred = probs >= threshold

        pred_count = pred.sum(
            axis=1
        )

        tp = np.logical_and(
            pred,
            y_true,
        ).sum(axis=1)


        # ----------------------------------------------------
        # CAFA-style protein-centric precision
        # Mean only across proteins with >=1 prediction
        # ----------------------------------------------------

        predicted_proteins = (
            pred_count > 0
        )

        if predicted_proteins.any():

            protein_precision = (
                tp[predicted_proteins]
                /
                pred_count[predicted_proteins]
            )

            precision = float(
                protein_precision.mean()
            )

        else:
            precision = 0.0


        # ----------------------------------------------------
        # Protein-centric recall across all proteins
        # ----------------------------------------------------

        protein_recall = (
            tp / true_count
        )

        recall = float(
            protein_recall.mean()
        )


        # ----------------------------------------------------
        # F
        # ----------------------------------------------------

        if precision + recall > 0:

            f_score = float(
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:
            f_score = 0.0


        # ----------------------------------------------------
        # RU / MI
        # ----------------------------------------------------

        false_negative = np.logical_and(
            y_true,
            np.logical_not(pred),
        )

        false_positive = np.logical_and(
            pred,
            np.logical_not(y_true),
        )


        ru_per_protein = (
            false_negative
            * information_content[None, :]
        ).sum(axis=1)

        mi_per_protein = (
            false_positive
            * information_content[None, :]
        ).sum(axis=1)


        ru = float(
            ru_per_protein.mean()
        )

        mi = float(
            mi_per_protein.mean()
        )

        semantic_distance = float(
            np.sqrt(
                ru ** 2
                + mi ** 2
            )
        )


        rows.append({
            "model":
                name,

            "threshold":
                float(threshold),

            "precision":
                precision,

            "recall":
                recall,

            "f_score":
                f_score,

            "ru":
                ru,

            "mi":
                mi,

            "semantic_distance":
                semantic_distance,

            "n_proteins_with_prediction":
                int(
                    predicted_proteins.sum()
                ),
        })


    table = pd.DataFrame(rows)


    # np.argmax / np.argmin return first occurrence,
    # satisfying the frozen smallest-threshold tie rule.

    f_idx = int(
        np.argmax(
            table["f_score"].to_numpy()
        )
    )

    s_idx = int(
        np.argmin(
            table[
                "semantic_distance"
            ].to_numpy()
        )
    )


    f_row = table.iloc[
        f_idx
    ]

    s_row = table.iloc[
        s_idx
    ]


    summary = {
        "model":
            name,

        "fmax":
            float(
                f_row["f_score"]
            ),

        "fmax_threshold":
            float(
                f_row["threshold"]
            ),

        "precision_at_fmax":
            float(
                f_row["precision"]
            ),

        "recall_at_fmax":
            float(
                f_row["recall"]
            ),

        "smin":
            float(
                s_row[
                    "semantic_distance"
                ]
            ),

        "smin_threshold":
            float(
                s_row["threshold"]
            ),

        "ru_at_smin":
            float(
                s_row["ru"]
            ),

        "mi_at_smin":
            float(
                s_row["mi"]
            ),
    }


    return table, summary


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print(
    "FMAX / SMIN V1 — "
    "POST-CONFIRMATORY FROZEN-PREDICTION EVALUATION"
)
print("=" * 90)


# ------------------------------------------------------------
# Verify protocol
# ------------------------------------------------------------

protocol_sha = sha256_file(
    PROTOCOL
)

print(
    "Protocol SHA256:",
    protocol_sha,
)

if (
    protocol_sha
    != EXPECTED_PROTOCOL_SHA256
):
    raise RuntimeError(
        "Protocol SHA256 mismatch."
    )


# ------------------------------------------------------------
# Verify frozen labels
# ------------------------------------------------------------

label_df = pd.read_csv(
    LABEL_FILE
)

if len(label_df) != N_LABELS:
    raise RuntimeError(
        "Expected exactly 494 labels."
    )

if not np.array_equal(
    label_df[
        "label_index"
    ].to_numpy(),
    np.arange(N_LABELS),
):
    raise RuntimeError(
        "Frozen label_index is not 0..493."
    )


# ------------------------------------------------------------
# TRAIN-derived IC
# ------------------------------------------------------------

train_y = np.load(
    TRAIN_LABELS_FILE
)

if train_y.shape != (
    N_TRAIN,
    N_LABELS,
):
    raise RuntimeError(
        f"Unexpected TRAIN label shape: "
        f"{train_y.shape}"
    )

train_support = train_y.sum(
    axis=0
).astype(np.float64)

if np.any(
    train_support <= 0
):
    raise RuntimeError(
        "Zero TRAIN support encountered."
    )

p_train = (
    train_support
    / float(N_TRAIN)
)

information_content = (
    -np.log(
        p_train
    )
)


print(
    "TRAIN proteins:",
    N_TRAIN,
)

print(
    "Labels:",
    N_LABELS,
)

print(
    "TRAIN support min/median/max:",
    int(train_support.min()),
    float(
        np.median(
            train_support
        )
    ),
    int(train_support.max()),
)

print(
    "IC min/median/max:",
    float(
        information_content.min()
    ),
    float(
        np.median(
            information_content
        )
    ),
    float(
        information_content.max()
    ),
)


# ------------------------------------------------------------
# TEST targets
# ------------------------------------------------------------

y_true = np.load(
    TEST_ROOT
    / "paired_test_labels.npy"
).astype(bool)

if y_true.shape != (
    N_TEST,
    N_LABELS,
):
    raise RuntimeError(
        f"Unexpected TEST target shape: "
        f"{y_true.shape}"
    )

# ------------------------------------------------------------
# V2 metric-specific eligibility
# ------------------------------------------------------------

target_counts = y_true.sum(axis=1)

eligible_mask = (
    target_counts > 0
)

n_eligible = int(
    eligible_mask.sum()
)

n_excluded = int(
    (~eligible_mask).sum()
)

if n_eligible != 4716:
    raise RuntimeError(
        f"Expected 4716 metric-eligible TEST proteins, "
        f"found {n_eligible}."
    )

if n_excluded != 15:
    raise RuntimeError(
        f"Expected 15 zero-positive exclusions, "
        f"found {n_excluded}."
    )

id_path = (
    TEST_ROOT
    / "paired_test_uniprot_ids.txt"
)

test_ids = [
    x.strip()
    for x in id_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

if len(test_ids) != N_TEST:
    raise RuntimeError(
        "TEST ID count does not match target rows."
    )

excluded_ids = [
    test_ids[i]
    for i in np.where(
        ~eligible_mask
    )[0]
]

print(
    "Paired TEST proteins:",
    N_TEST,
)

print(
    "Metric-eligible TEST proteins:",
    n_eligible,
)

print(
    "Zero-positive proteins excluded:",
    n_excluded,
)

# Apply exactly the same frozen eligibility mask
# to targets and every model below.
y_true = y_true[
    eligible_mask
]


# ------------------------------------------------------------
# Frozen predictions
# ------------------------------------------------------------

p_sequence = load_probability_array(
    TEST_ROOT
    / "paired_sequence_probs.npy"
)[eligible_mask]

p_structure = load_probability_array(
    TEST_ROOT
    / "paired_structure_probs.npy"
)[eligible_mask]

p_average = (
    p_sequence
    + p_structure
) / 2.0

p_A1 = load_probability_array(
    TEST_ROOT
    / "A1_predictions.npy"
)[eligible_mask]

p_A2 = load_probability_array(
    TEST_ROOT
    / "A2_predictions.npy"
)[eligible_mask]


models = {
    "sequence":
        p_sequence,

    "structure":
        p_structure,

    "simple_average":
        p_average,

    "A1_uncertainty_arbitration":
        p_A1,

    "A2_conflict_arbitration":
        p_A2,
}


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

summaries = []


for name, probs in models.items():

    print()
    print(
        "Evaluating:",
        name,
    )

    table, summary = evaluate_model(
        name=name,
        probs=probs,
        y_true=y_true,
        information_content=information_content,
    )

    table.to_csv(
        OUT
        / f"{name}_threshold_curve.csv",
        index=False,
    )

    summaries.append(
        summary
    )


# ------------------------------------------------------------
# Save IC
# ------------------------------------------------------------

ic_df = label_df[
    [
        "go_id",
        "train_protein_count",
        "label_index",
    ]
].copy()

ic_df[
    "train_support_from_matrix"
] = train_support.astype(
    np.int64
)

ic_df[
    "train_probability"
] = p_train

ic_df[
    "information_content_natural_log"
] = information_content

ic_df.to_csv(
    OUT
    / "train_information_content.csv",
    index=False,
)


# ------------------------------------------------------------
# Save metric-specific exclusions
# ------------------------------------------------------------

with open(
    OUT / "excluded_zero_positive_test_ids.txt",
    "w",
    encoding="utf-8",
) as f:
    for uid in excluded_ids:
        f.write(uid + "\n")


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    summaries
)

summary_df.to_csv(
    OUT
    / "fmax_smin_summary.csv",
    index=False,
)


manifest = {
    "protocol_sha256":
        protocol_sha,

    "n_paired_test_proteins":
        N_TEST,

    "n_metric_eligible_test_proteins":
        n_eligible,

    "n_zero_positive_excluded":
        n_excluded,

    "excluded_uniprot_ids":
        excluded_ids,

    "n_labels":
        N_LABELS,

    "n_train_proteins_for_ic":
        N_TRAIN,

    "threshold_grid":
        {
            "start": 0.0,
            "stop": 1.0,
            "step": 0.01,
            "count": 101,
        },

    "ic_definition":
        "-ln(TRAIN support / 38031)",

    "models":
        list(models.keys()),

    "test_used_for_training":
        False,

    "post_confirmatory":
        True,
}

with open(
    OUT
    / "fmax_smin_manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# Print final results
# ------------------------------------------------------------

print()
print("=" * 90)
print("RESULTS")
print("=" * 90)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print(
    "Saved:",
    OUT
    / "fmax_smin_summary.csv"
)

print(
    "TEST used for training/tuning: NO"
)

print(
    "Post-confirmatory evaluation: YES"
)

print("=" * 90)
