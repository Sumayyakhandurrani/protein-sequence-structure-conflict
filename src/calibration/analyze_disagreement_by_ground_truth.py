from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "data/conflict/paired_experts_v1"
)

SPLIT = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

OUT = Path(
    "data/conflict/"
    "ground_truth_stratification_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


def safe_auc(y, score):
    if np.unique(y).size < 2:
        return np.nan

    return float(
        roc_auc_score(y, score)
    )


def safe_ap(y, score):
    if np.sum(y) == 0:
        return np.nan

    return float(
        average_precision_score(
            y,
            score,
        )
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 92)
print("GROUND-TRUTH-STRATIFIED DISAGREEMENT RELIABILITY V1")
print("=" * 92)

print("Population:       structure-eligible calibration-eval")
print("Stratification:   true GO label y=0 versus y=1")
print("Disagreement:     |p_sequence - p_structure|")
print("Threshold optimization: NO")
print("pLDDT used:       NO")
print("TEST USED:        NO")
print()


paired_ids = read_ids(
    BASE / "paired_val_uniprot_ids.txt"
)

eval_ids = read_ids(
    SPLIT / "structure_cal_eval_ids.txt"
)


y_all = np.load(
    BASE / "paired_val_labels.npy"
).astype(np.float64)

p_seq_all = np.load(
    BASE / "sequence_probs.npy"
).astype(np.float64)

p_struct_all = np.load(
    BASE / "structure_probs.npy"
).astype(np.float64)


index = {
    uid: i
    for i, uid in enumerate(paired_ids)
}

rows = np.asarray(
    [index[uid] for uid in eval_ids],
    dtype=np.int64,
)


y = y_all[rows]
p_seq = p_seq_all[rows]
p_struct = p_struct_all[rows]


if y.shape != (2373, 494):
    raise RuntimeError(
        f"Unexpected shape: {y.shape}"
    )


# ============================================================
# QUANTITIES
# ============================================================

D = np.abs(
    p_seq - p_struct
)

seq_error = np.abs(
    y - p_seq
)

struct_error = np.abs(
    y - p_struct
)

mean_error = (
    seq_error + struct_error
) / 2.0


seq_large_error = (
    seq_error >= 0.5
)

struct_large_error = (
    struct_error >= 0.5
)

either_large_error = (
    seq_large_error
    | struct_large_error
)


# ============================================================
# CLASS-SPECIFIC SUMMARY
# ============================================================

summary_rows = []


for class_value, class_name in [
    (0, "negative_y0"),
    (1, "positive_y1"),
]:

    mask = (
        y == class_value
    )

    d = D[mask]
    err = mean_error[mask]

    seq_err = seq_error[mask]
    struct_err = struct_error[mask]

    event = (
        either_large_error[mask]
        .astype(np.uint8)
    )


    corr = float(
        np.corrcoef(
            d,
            err,
        )[0, 1]
    )


    summary_rows.append({
        "ground_truth":
            class_name,

        "pairs":
            int(mask.sum()),

        "fraction_of_all_pairs":
            float(mask.mean()),

        "mean_disagreement":
            float(d.mean()),

        "median_disagreement":
            float(np.median(d)),

        "q90_disagreement":
            float(
                np.quantile(d, 0.90)
            ),

        "q95_disagreement":
            float(
                np.quantile(d, 0.95)
            ),

        "q99_disagreement":
            float(
                np.quantile(d, 0.99)
            ),

        "mean_sequence_error":
            float(seq_err.mean()),

        "mean_structure_error":
            float(struct_err.mean()),

        "mean_expert_error":
            float(err.mean()),

        "large_error_events":
            int(event.sum()),

        "large_error_prevalence":
            float(event.mean()),

        "D_vs_mean_error_correlation":
            corr,

        "D_large_error_ROC_AUC":
            safe_auc(
                event,
                d,
            ),

        "D_large_error_AP":
            safe_ap(
                event,
                d,
            ),
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT /
    "ground_truth_summary.csv",
    index=False,
)


# ============================================================
# WITHIN-CLASS DISAGREEMENT DECILES
#
# Each class receives its OWN disagreement quantiles.
# Therefore prevalence cannot create the decile trend.
# ============================================================

decile_rows = []


for class_value, class_name in [
    (0, "negative_y0"),
    (1, "positive_y1"),
]:

    class_mask = (
        y == class_value
    )

    class_D = D[
        class_mask
    ]

    class_mean_error = mean_error[
        class_mask
    ]

    class_seq_error = seq_error[
        class_mask
    ]

    class_struct_error = struct_error[
        class_mask
    ]

    class_event = either_large_error[
        class_mask
    ]


    edges = np.quantile(
        class_D,
        np.linspace(
            0.0,
            1.0,
            11,
        ),
    )

    edges[-1] = np.nextafter(
        edges[-1],
        np.inf,
    )


    for i in range(10):

        lo = edges[i]
        hi = edges[i + 1]

        m = (
            (class_D >= lo)
            & (class_D < hi)
        )

        if not m.any():
            continue


        decile_rows.append({
            "ground_truth":
                class_name,

            "decile":
                i + 1,

            "lower_D":
                float(lo),

            "upper_D":
                float(hi),

            "pairs":
                int(m.sum()),

            "mean_D":
                float(
                    class_D[m].mean()
                ),

            "mean_sequence_error":
                float(
                    class_seq_error[
                        m
                    ].mean()
                ),

            "mean_structure_error":
                float(
                    class_struct_error[
                        m
                    ].mean()
                ),

            "mean_expert_error":
                float(
                    class_mean_error[
                        m
                    ].mean()
                ),

            "large_error_prevalence":
                float(
                    class_event[
                        m
                    ].mean()
                ),
        })


decile_df = pd.DataFrame(
    decile_rows
)

decile_df.to_csv(
    OUT /
    "within_class_disagreement_deciles.csv",
    index=False,
)


# ============================================================
# HIGH-DISAGREEMENT CLASS-SPECIFIC DIRECTIONALITY
#
# Use top 10% WITHIN EACH CLASS.
# This is descriptive, not an optimized conflict threshold.
# ============================================================

direction_rows = []


advantage = (
    struct_error - seq_error
)


for class_value, class_name in [
    (0, "negative_y0"),
    (1, "positive_y1"),
]:

    class_mask = (
        y == class_value
    )

    threshold = np.quantile(
        D[class_mask],
        0.90,
    )

    mask = (
        class_mask
        & (D >= threshold)
    )

    adv = advantage[mask]

    seq_better = (
        adv > 1e-12
    )

    struct_better = (
        adv < -1e-12
    )


    direction_rows.append({
        "ground_truth":
            class_name,

        "descriptive_top10_D_threshold":
            float(threshold),

        "pairs":
            int(mask.sum()),

        "mean_D":
            float(
                D[mask].mean()
            ),

        "mean_sequence_error":
            float(
                seq_error[
                    mask
                ].mean()
            ),

        "mean_structure_error":
            float(
                struct_error[
                    mask
                ].mean()
            ),

        "sequence_better_fraction":
            float(
                seq_better.mean()
            ),

        "structure_better_fraction":
            float(
                struct_better.mean()
            ),
    })


direction_df = pd.DataFrame(
    direction_rows
)

direction_df.to_csv(
    OUT /
    "within_class_top10_directionality.csv",
    index=False,
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "analysis":
        "ground_truth_stratified_disagreement_v1",

    "population":
        "structure-eligible calibration-eval",

    "proteins":
        len(eval_ids),

    "labels":
        494,

    "pairs":
        int(D.size),

    "negative_pairs":
        int((y == 0).sum()),

    "positive_pairs":
        int((y == 1).sum()),

    "disagreement":
        "|p_sequence-p_structure|",

    "error":
        (
            "mean absolute probabilistic error "
            "of sequence and structure experts"
        ),

    "large_error_event":
        "either expert absolute error >= 0.5",

    "within_class_deciles":
        True,

    "top10_thresholds_are_descriptive":
        True,

    "conflict_threshold_optimized":
        False,

    "plddt_used":
        False,

    "test_used":
        False,
}


with open(
    OUT /
    "ground_truth_stratification_summary.json",
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
    f"Evaluation proteins: {len(eval_ids):,}"
)

print(
    f"Prediction pairs:    {D.size:,}"
)

print(
    f"Negative pairs:      "
    f"{int((y == 0).sum()):,}"
)

print(
    f"Positive pairs:      "
    f"{int((y == 1).sum()):,}"
)

print()


print("=" * 92)
print("CLASS-SPECIFIC ASSOCIATION")
print("=" * 92)


for _, r in summary_df.iterrows():

    print()
    print(
        r["ground_truth"]
    )

    print(
        f"  pairs:                   "
        f"{int(r['pairs']):,}"
    )

    print(
        f"  mean D:                  "
        f"{r['mean_disagreement']:.6f}"
    )

    print(
        f"  mean sequence error:     "
        f"{r['mean_sequence_error']:.6f}"
    )

    print(
        f"  mean structure error:    "
        f"{r['mean_structure_error']:.6f}"
    )

    print(
        f"  large-error prevalence:  "
        f"{r['large_error_prevalence']:.4%}"
    )

    print(
        f"  D vs error correlation:  "
        f"{r['D_vs_mean_error_correlation']:.6f}"
    )

    print(
        f"  D error ROC-AUC:         "
        f"{r['D_large_error_ROC_AUC']:.6f}"
    )

    print(
        f"  D error AP:              "
        f"{r['D_large_error_AP']:.6f}"
    )


print()
print("=" * 92)
print("WITHIN-CLASS DISAGREEMENT DECILES")
print("=" * 92)


for class_name in [
    "negative_y0",
    "positive_y1",
]:

    print()
    print(class_name)

    sub = decile_df[
        decile_df[
            "ground_truth"
        ] == class_name
    ]

    for _, r in sub.iterrows():

        print(
            f"  Q{int(r['decile']):02d}: "
            f"D={r['mean_D']:.6f} | "
            f"error={r['mean_expert_error']:.6f} | "
            f"seq={r['mean_sequence_error']:.6f} | "
            f"struct={r['mean_structure_error']:.6f} | "
            f"large-error={r['large_error_prevalence']:.4%}"
        )


print()
print("=" * 92)
print("TOP 10% DISAGREEMENT WITHIN EACH CLASS")
print("=" * 92)


for _, r in direction_df.iterrows():

    print()
    print(
        r["ground_truth"]
    )

    print(
        f"  descriptive D cutoff:    "
        f"{r['descriptive_top10_D_threshold']:.6f}"
    )

    print(
        f"  pairs:                   "
        f"{int(r['pairs']):,}"
    )

    print(
        f"  sequence better:         "
        f"{r['sequence_better_fraction']:.2%}"
    )

    print(
        f"  structure better:        "
        f"{r['structure_better_fraction']:.2%}"
    )

    print(
        f"  mean sequence error:     "
        f"{r['mean_sequence_error']:.6f}"
    )

    print(
        f"  mean structure error:    "
        f"{r['mean_structure_error']:.6f}"
    )


print()
print("=" * 92)

print(
    "Within-class deciles:             YES"
)

print(
    "Conflict threshold optimized:     NO"
)

print(
    "pLDDT/global confidence used:     NO"
)

print(
    "Final TEST confirmation:          NO"
)

print(
    "TEST USED:                        NO"
)

print()

print(
    f"Saved under:\n  {OUT}"
)

print("=" * 92)