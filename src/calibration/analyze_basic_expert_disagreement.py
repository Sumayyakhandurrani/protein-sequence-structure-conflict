from pathlib import Path
import json

import numpy as np
import pandas as pd


BASE = Path(
    "data/conflict/paired_experts_v1"
)

OUT = Path(
    "data/conflict/disagreement_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


# ============================================================
# LOAD
# ============================================================

print("=" * 84)
print("SEQUENCE–STRUCTURE BASIC DISAGREEMENT ANALYSIS V1")
print("=" * 84)

print("Primary descriptive disagreement: |Pseq - Pstruct|")
print("Conflict definition finalized: NO")
print("Conflict/error association tested: NO")
print("Threshold optimization: NO")
print("TEST USED: NO")
print()


ids = read_ids(
    BASE / "paired_val_uniprot_ids.txt"
)

y = np.load(
    BASE / "paired_val_labels.npy"
)

p_seq = np.load(
    BASE / "sequence_probs.npy"
)

p_struct = np.load(
    BASE / "structure_probs.npy"
)


expected = (4742, 494)

for name, arr in [
    ("labels", y),
    ("sequence", p_seq),
    ("structure", p_struct),
]:
    if arr.shape != expected:
        raise RuntimeError(
            f"{name} shape mismatch: {arr.shape}"
        )


if len(ids) != 4742:
    raise RuntimeError(
        f"Expected 4742 IDs, got {len(ids)}"
    )


if not np.isfinite(p_seq).all():
    raise RuntimeError(
        "Non-finite sequence probabilities."
    )

if not np.isfinite(p_struct).all():
    raise RuntimeError(
        "Non-finite structure probabilities."
    )


# ============================================================
# ABSOLUTE DISAGREEMENT
# ============================================================

D = np.abs(
    p_seq.astype(np.float64)
    - p_struct.astype(np.float64)
)


signed = (
    p_seq.astype(np.float64)
    - p_struct.astype(np.float64)
)


positive_mask = y == 1
negative_mask = y == 0


# ============================================================
# DISTRIBUTION FUNCTION
# ============================================================

def distribution_stats(values):
    values = np.asarray(values)

    return {
        "count":
            int(values.size),

        "mean":
            float(np.mean(values)),

        "std":
            float(np.std(values)),

        "median":
            float(np.median(values)),

        "q75":
            float(np.quantile(values, 0.75)),

        "q90":
            float(np.quantile(values, 0.90)),

        "q95":
            float(np.quantile(values, 0.95)),

        "q99":
            float(np.quantile(values, 0.99)),

        "max":
            float(np.max(values)),
    }


overall_stats = distribution_stats(
    D.ravel()
)

positive_stats = distribution_stats(
    D[positive_mask]
)

negative_stats = distribution_stats(
    D[negative_mask]
)


# ============================================================
# DESCRIPTIVE DISAGREEMENT BANDS
#
# These are descriptive bins only.
# They are NOT optimized conflict thresholds.
# ============================================================

bands = [
    (0.00, 0.10),
    (0.10, 0.25),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.75, 1.01),
]


band_rows = []

flat_D = D.ravel()

for lo, hi in bands:

    if hi > 1.0:
        mask = (
            (flat_D >= lo)
            & (flat_D <= 1.0)
        )
    else:
        mask = (
            (flat_D >= lo)
            & (flat_D < hi)
        )

    count = int(mask.sum())

    band_rows.append({
        "lower_bound":
            lo,

        "upper_bound":
            min(hi, 1.0),

        "count":
            count,

        "fraction":
            count / flat_D.size,
    })


band_df = pd.DataFrame(
    band_rows
)

band_df.to_csv(
    OUT / "disagreement_bands.csv",
    index=False,
)


# ============================================================
# PROTEIN-LEVEL DISAGREEMENT
# ============================================================

protein_mean_D = D.mean(axis=1)
protein_max_D = D.max(axis=1)
protein_q95_D = np.quantile(
    D,
    0.95,
    axis=1,
)


protein_df = pd.DataFrame({
    "uniprot_id":
        ids,

    "mean_abs_disagreement":
        protein_mean_D,

    "q95_abs_disagreement":
        protein_q95_D,

    "max_abs_disagreement":
        protein_max_D,

    "true_positive_labels":
        y.sum(axis=1),
})


protein_df.to_csv(
    OUT / "protein_disagreement.csv",
    index=False,
)


# ============================================================
# LABEL-LEVEL DISAGREEMENT
# ============================================================

label_mean_D = D.mean(axis=0)
label_max_D = D.max(axis=0)
label_q95_D = np.quantile(
    D,
    0.95,
    axis=0,
)


label_df = pd.DataFrame({
    "label_index":
        np.arange(494),

    "val_positive_support":
        y.sum(axis=0),

    "mean_abs_disagreement":
        label_mean_D,

    "q95_abs_disagreement":
        label_q95_D,

    "max_abs_disagreement":
        label_max_D,
})


label_df.to_csv(
    OUT / "label_disagreement.csv",
    index=False,
)


# ============================================================
# DIRECTION OF DISAGREEMENT
# ============================================================

eps = 1e-12

seq_higher = int(
    (signed > eps).sum()
)

struct_higher = int(
    (signed < -eps).sum()
)

equal = int(
    (
        np.abs(signed) <= eps
    ).sum()
)


# ============================================================
# HIGH-DESCRIPTIVE-DISAGREEMENT COUNTS
#
# Again: descriptive only, NOT selected thresholds.
# ============================================================

threshold_counts = {}

for threshold in [
    0.10,
    0.25,
    0.50,
    0.75,
]:

    mask = D >= threshold

    threshold_counts[
        str(threshold)
    ] = {
        "count":
            int(mask.sum()),

        "fraction":
            float(mask.mean()),

        "positive_pairs":
            int(
                (
                    mask
                    & positive_mask
                ).sum()
            ),

        "negative_pairs":
            int(
                (
                    mask
                    & negative_mask
                ).sum()
            ),
    }


# ============================================================
# SAVE DISAGREEMENT MATRIX
# ============================================================

np.save(
    OUT / "absolute_disagreement.npy",
    D.astype(np.float32),
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "basic_expert_disagreement_v1",

    "proteins":
        4742,

    "labels":
        494,

    "prediction_pairs":
        int(D.size),

    "positive_pairs":
        int(positive_mask.sum()),

    "negative_pairs":
        int(negative_mask.sum()),

    "definition":
        "absolute probability disagreement",

    "formula":
        "|p_sequence - p_structure|",

    "overall":
        overall_stats,

    "ground_truth_positive_pairs":
        positive_stats,

    "ground_truth_negative_pairs":
        negative_stats,

    "direction": {
        "sequence_probability_higher":
            seq_higher,

        "structure_probability_higher":
            struct_higher,

        "equal_within_1e-12":
            equal,
    },

    "descriptive_threshold_counts":
        threshold_counts,

    "conflict_definition_finalized":
        False,

    "conflict_error_association_tested":
        False,

    "threshold_optimization":
        False,

    "plddt_used":
        False,

    "test_used":
        False,
}


with open(
    OUT / "disagreement_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print(
    f"Proteins:                  "
    f"{4742:,}"
)

print(
    f"Labels:                    "
    f"{494}"
)

print(
    f"Prediction pairs:          "
    f"{D.size:,}"
)

print(
    f"Positive pairs:            "
    f"{int(positive_mask.sum()):,}"
)

print(
    f"Negative pairs:            "
    f"{int(negative_mask.sum()):,}"
)

print()

print("ABSOLUTE DISAGREEMENT — ALL PAIRS")
print(
    f"  Mean:                    "
    f"{overall_stats['mean']:.6f}"
)
print(
    f"  Median:                  "
    f"{overall_stats['median']:.6f}"
)
print(
    f"  90th percentile:         "
    f"{overall_stats['q90']:.6f}"
)
print(
    f"  95th percentile:         "
    f"{overall_stats['q95']:.6f}"
)
print(
    f"  99th percentile:         "
    f"{overall_stats['q99']:.6f}"
)
print(
    f"  Maximum:                 "
    f"{overall_stats['max']:.6f}"
)

print()

print("GROUND-TRUTH POSITIVE PAIRS")
print(
    f"  Mean disagreement:       "
    f"{positive_stats['mean']:.6f}"
)
print(
    f"  Median disagreement:     "
    f"{positive_stats['median']:.6f}"
)
print(
    f"  95th percentile:         "
    f"{positive_stats['q95']:.6f}"
)

print()

print("GROUND-TRUTH NEGATIVE PAIRS")
print(
    f"  Mean disagreement:       "
    f"{negative_stats['mean']:.6f}"
)
print(
    f"  Median disagreement:     "
    f"{negative_stats['median']:.6f}"
)
print(
    f"  95th percentile:         "
    f"{negative_stats['q95']:.6f}"
)

print()

print("DIRECTION")
print(
    f"  Sequence probability higher: "
    f"{seq_higher:,} "
    f"({seq_higher / D.size:.2%})"
)
print(
    f"  Structure probability higher:"
    f" {struct_higher:,} "
    f"({struct_higher / D.size:.2%})"
)
print(
    f"  Equal within 1e-12:           "
    f"{equal:,}"
)

print()

print("DESCRIPTIVE DISAGREEMENT COUNTS")

for threshold in [
    0.10,
    0.25,
    0.50,
    0.75,
]:

    info = threshold_counts[
        str(threshold)
    ]

    print(
        f"  D >= {threshold:.2f}: "
        f"{info['count']:,} "
        f"({info['fraction']:.4%})"
    )

print()

print(
    "Conflict definition finalized:       NO"
)

print(
    "Conflict/error association tested:   NO"
)

print(
    "pLDDT used:                          NO"
)

print(
    "Threshold optimization:              NO"
)

print(
    "TEST USED:                           NO"
)

print()

print(
    f"Saved under:\n  {OUT}"
)

print("=" * 84)