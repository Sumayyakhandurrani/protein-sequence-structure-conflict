from pathlib import Path
import json

import numpy as np
import pandas as pd


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
    "data/conflict/directionality_v1"
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


# ============================================================
# LOAD
# ============================================================

print("=" * 88)
print("SEQUENCE–STRUCTURE DISAGREEMENT DIRECTIONALITY V1")
print("=" * 88)

print("Population:       structure-eligible calibration-eval proteins")
print("Purpose:          determine which expert is closer to truth")
print("Conflict threshold optimized: NO")
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


if y_all.shape != (4742, 494):
    raise RuntimeError(
        f"Unexpected label shape: {y_all.shape}"
    )


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
        f"Unexpected evaluation shape: {y.shape}"
    )


# ============================================================
# DISAGREEMENT AND ERRORS
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


# Positive:
# structure error > sequence error
# therefore sequence is better.
#
# Negative:
# structure error < sequence error
# therefore structure is better.

advantage = (
    struct_error - seq_error
)


eps = 1e-12

seq_better = advantage > eps
struct_better = advantage < -eps
equal = np.abs(advantage) <= eps


# ============================================================
# GLOBAL DIRECTIONALITY
# ============================================================

global_stats = {
    "pairs":
        int(D.size),

    "sequence_better":
        int(seq_better.sum()),

    "structure_better":
        int(struct_better.sum()),

    "equal":
        int(equal.sum()),

    "sequence_better_fraction":
        float(seq_better.mean()),

    "structure_better_fraction":
        float(struct_better.mean()),

    "mean_advantage_structure_minus_sequence":
        float(advantage.mean()),

    "mean_abs_disagreement":
        float(D.mean()),
}


# ============================================================
# DISAGREEMENT DECILES
#
# Descriptive only.
# No optimized threshold.
# ============================================================

flat_D = D.ravel()

edges = np.quantile(
    flat_D,
    np.linspace(0.0, 1.0, 11),
)

edges[-1] = np.nextafter(
    edges[-1],
    np.inf,
)


decile_rows = []


for i in range(10):

    lo = edges[i]
    hi = edges[i + 1]

    mask = (
        (D >= lo)
        & (D < hi)
    )

    n = int(mask.sum())

    if n == 0:
        continue

    seq_n = int(
        seq_better[mask].sum()
    )

    struct_n = int(
        struct_better[mask].sum()
    )

    equal_n = int(
        equal[mask].sum()
    )


    decile_rows.append({
        "decile":
            i + 1,

        "lower_D":
            float(lo),

        "upper_D":
            float(hi),

        "pairs":
            n,

        "mean_D":
            float(D[mask].mean()),

        "positive_prevalence":
            float(y[mask].mean()),

        "mean_sequence_error":
            float(seq_error[mask].mean()),

        "mean_structure_error":
            float(struct_error[mask].mean()),

        "mean_advantage_structure_minus_sequence":
            float(advantage[mask].mean()),

        "sequence_better_count":
            seq_n,

        "structure_better_count":
            struct_n,

        "equal_count":
            equal_n,

        "sequence_better_fraction":
            seq_n / n,

        "structure_better_fraction":
            struct_n / n,

        "equal_fraction":
            equal_n / n,
    })


decile_df = pd.DataFrame(
    decile_rows
)

decile_df.to_csv(
    OUT / "directionality_by_disagreement_decile.csv",
    index=False,
)


# ============================================================
# GROUND-TRUTH STRATIFICATION
# ============================================================

class_rows = []


for class_name, class_mask in [
    ("positive", y == 1),
    ("negative", y == 0),
]:

    n = int(class_mask.sum())

    seq_n = int(
        seq_better[class_mask].sum()
    )

    struct_n = int(
        struct_better[class_mask].sum()
    )

    equal_n = int(
        equal[class_mask].sum()
    )


    class_rows.append({
        "ground_truth":
            class_name,

        "pairs":
            n,

        "mean_D":
            float(
                D[class_mask].mean()
            ),

        "mean_sequence_error":
            float(
                seq_error[
                    class_mask
                ].mean()
            ),

        "mean_structure_error":
            float(
                struct_error[
                    class_mask
                ].mean()
            ),

        "mean_advantage_structure_minus_sequence":
            float(
                advantage[
                    class_mask
                ].mean()
            ),

        "sequence_better_count":
            seq_n,

        "structure_better_count":
            struct_n,

        "equal_count":
            equal_n,

        "sequence_better_fraction":
            seq_n / n,

        "structure_better_fraction":
            struct_n / n,
    })


class_df = pd.DataFrame(
    class_rows
)

class_df.to_csv(
    OUT / "directionality_by_ground_truth.csv",
    index=False,
)


# ============================================================
# TOP DISAGREEMENT DECILE:
# POSITIVE / NEGATIVE SEPARATELY
# ============================================================

top_lo = edges[9]

top_mask = (
    D >= top_lo
)


top_rows = []


for class_name, class_mask in [
    ("all", np.ones_like(y, dtype=bool)),
    ("positive", y == 1),
    ("negative", y == 0),
]:

    mask = (
        top_mask
        & class_mask
    )

    n = int(mask.sum())

    if n == 0:
        continue

    seq_n = int(
        seq_better[mask].sum()
    )

    struct_n = int(
        struct_better[mask].sum()
    )


    top_rows.append({
        "group":
            class_name,

        "pairs":
            n,

        "mean_D":
            float(D[mask].mean()),

        "mean_sequence_error":
            float(seq_error[mask].mean()),

        "mean_structure_error":
            float(struct_error[mask].mean()),

        "mean_advantage_structure_minus_sequence":
            float(advantage[mask].mean()),

        "sequence_better_count":
            seq_n,

        "structure_better_count":
            struct_n,

        "sequence_better_fraction":
            seq_n / n,

        "structure_better_fraction":
            struct_n / n,
    })


top_df = pd.DataFrame(
    top_rows
)

top_df.to_csv(
    OUT / "top_disagreement_decile_directionality.csv",
    index=False,
)


# ============================================================
# SUBSTANTIAL ADVANTAGE
#
# Fixed descriptive margins.
# These are NOT conflict thresholds.
# ============================================================

margin_rows = []


for margin in [
    0.05,
    0.10,
    0.25,
    0.50,
]:

    seq_margin = (
        advantage >= margin
    )

    struct_margin = (
        advantage <= -margin
    )


    margin_rows.append({
        "error_margin":
            margin,

        "sequence_better_by_at_least_margin":
            int(seq_margin.sum()),

        "sequence_fraction":
            float(seq_margin.mean()),

        "structure_better_by_at_least_margin":
            int(struct_margin.sum()),

        "structure_fraction":
            float(struct_margin.mean()),
    })


margin_df = pd.DataFrame(
    margin_rows
)

margin_df.to_csv(
    OUT / "substantial_error_advantage.csv",
    index=False,
)


# ============================================================
# SAVE MATRICES
# ============================================================

np.save(
    OUT / "absolute_disagreement.npy",
    D.astype(np.float32),
)

np.save(
    OUT / "error_advantage_structure_minus_sequence.npy",
    advantage.astype(np.float32),
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "disagreement_directionality_v1",

    "population":
        "structure-eligible calibration-eval",

    "proteins":
        len(eval_ids),

    "labels":
        494,

    "pairs":
        int(D.size),

    "disagreement":
        "|p_sequence - p_structure|",

    "error_advantage":
        (
            "|y-p_structure| - "
            "|y-p_sequence|"
        ),

    "positive_advantage_interpretation":
        "sequence expert closer to truth",

    "negative_advantage_interpretation":
        "structure expert closer to truth",

    "global":
        global_stats,

    "top_disagreement_decile_lower_bound":
        float(top_lo),

    "conflict_threshold_optimized":
        False,

    "plddt_used":
        False,

    "test_used":
        False,
}


with open(
    OUT / "directionality_summary.json",
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
    f"Evaluation proteins:       "
    f"{len(eval_ids):,}"
)

print(
    f"Prediction pairs:          "
    f"{D.size:,}"
)

print()

print("GLOBAL DIRECTIONALITY")

print(
    f"  Sequence closer:         "
    f"{global_stats['sequence_better']:,} "
    f"({global_stats['sequence_better_fraction']:.2%})"
)

print(
    f"  Structure closer:        "
    f"{global_stats['structure_better']:,} "
    f"({global_stats['structure_better_fraction']:.2%})"
)

print(
    f"  Equal:                   "
    f"{global_stats['equal']:,}"
)

print(
    f"  Mean advantage (S-Q):    "
    f"{global_stats['mean_advantage_structure_minus_sequence']:+.6f}"
)

print()

print("DIRECTIONALITY BY DISAGREEMENT DECILE")

for _, r in decile_df.iterrows():

    print(
        f"  Q{int(r['decile']):02d}: "
        f"D={r['mean_D']:.6f} | "
        f"seq better={r['sequence_better_fraction']:.2%} | "
        f"struct better={r['structure_better_fraction']:.2%} | "
        f"adv={r['mean_advantage_structure_minus_sequence']:+.6f} | "
        f"positive prev={r['positive_prevalence']:.2%}"
    )


print()
print(
    "TOP DISAGREEMENT DECILE "
    f"(D >= {top_lo:.6f})"
)

for _, r in top_df.iterrows():

    print(
        f"  {r['group']:<8} "
        f"n={int(r['pairs']):,} | "
        f"seq better={r['sequence_better_fraction']:.2%} | "
        f"struct better={r['structure_better_fraction']:.2%} | "
        f"seq err={r['mean_sequence_error']:.6f} | "
        f"struct err={r['mean_structure_error']:.6f}"
    )


print()
print("SUBSTANTIAL ERROR ADVANTAGE")

for _, r in margin_df.iterrows():

    print(
        f"  Margin >= {r['error_margin']:.2f}: "
        f"sequence={int(r['sequence_better_by_at_least_margin']):,} "
        f"({r['sequence_fraction']:.3%}) | "
        f"structure={int(r['structure_better_by_at_least_margin']):,} "
        f"({r['structure_fraction']:.3%})"
    )


print()
print("Conflict threshold optimized: NO")
print("pLDDT used:                  NO")
print("Final TEST confirmation:     NO")
print("TEST USED:                    NO")

print()
print(
    f"Saved under:\n  {OUT}"
)

print("=" * 88)