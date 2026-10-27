from pathlib import Path
import json

import numpy as np
import pandas as pd


BASE = Path("data/conflict/paired_experts_v1")

SPLIT = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

OUT = Path(
    "data/conflict/signed_disagreement_v1"
)

OUT.mkdir(parents=True, exist_ok=True)


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


print("=" * 92)
print("SIGNED SEQUENCE–STRUCTURE DISAGREEMENT V1")
print("=" * 92)

print("Population:       structure-eligible calibration-eval")
print("Signed score:     S = p_sequence - p_structure")
print("S > 0:            sequence gives stronger GO evidence")
print("S < 0:            structure gives stronger GO evidence")
print("Threshold optimization: NO")
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


# Signed and absolute disagreement
S = p_seq - p_struct
D = np.abs(S)

seq_error = np.abs(y - p_seq)
struct_error = np.abs(y - p_struct)

advantage = (
    struct_error - seq_error
)

# >0 means sequence closer to truth
seq_better = advantage > 1e-12
struct_better = advantage < -1e-12


# ============================================================
# SIGNED-DISAGREEMENT QUANTILES
#
# Quantiles are descriptive, not optimized thresholds.
# ============================================================

flat_S = S.reshape(-1)

edges = np.quantile(
    flat_S,
    np.linspace(0, 1, 11),
)

edges[-1] = np.nextafter(
    edges[-1],
    np.inf,
)


rows_out = []


for i in range(10):

    lo = edges[i]
    hi = edges[i + 1]

    mask = (
        (S >= lo)
        & (S < hi)
    )

    n = int(mask.sum())

    if n == 0:
        continue

    rows_out.append({
        "signed_decile":
            i + 1,

        "lower_S":
            float(lo),

        "upper_S":
            float(hi),

        "pairs":
            n,

        "mean_S":
            float(S[mask].mean()),

        "mean_absolute_D":
            float(D[mask].mean()),

        "positive_prevalence":
            float(y[mask].mean()),

        "mean_p_sequence":
            float(p_seq[mask].mean()),

        "mean_p_structure":
            float(p_struct[mask].mean()),

        "mean_sequence_error":
            float(seq_error[mask].mean()),

        "mean_structure_error":
            float(struct_error[mask].mean()),

        "sequence_better_fraction":
            float(seq_better[mask].mean()),

        "structure_better_fraction":
            float(struct_better[mask].mean()),
    })


signed_df = pd.DataFrame(rows_out)

signed_df.to_csv(
    OUT / "signed_disagreement_deciles.csv",
    index=False,
)


# ============================================================
# DIRECTION × GROUND TRUTH
# ============================================================

direction_rows = []


groups = [
    (
        "sequence_higher",
        S > 0,
    ),
    (
        "structure_higher",
        S < 0,
    ),
]


for direction_name, direction_mask in groups:

    for class_value, class_name in [
        (0, "negative_y0"),
        (1, "positive_y1"),
    ]:

        mask = (
            direction_mask
            & (y == class_value)
        )

        n = int(mask.sum())

        if n == 0:
            continue

        direction_rows.append({
            "direction":
                direction_name,

            "ground_truth":
                class_name,

            "pairs":
                n,

            "mean_signed_disagreement":
                float(S[mask].mean()),

            "mean_absolute_disagreement":
                float(D[mask].mean()),

            "mean_p_sequence":
                float(p_seq[mask].mean()),

            "mean_p_structure":
                float(p_struct[mask].mean()),

            "mean_sequence_error":
                float(seq_error[mask].mean()),

            "mean_structure_error":
                float(struct_error[mask].mean()),

            "sequence_better_fraction":
                float(seq_better[mask].mean()),

            "structure_better_fraction":
                float(struct_better[mask].mean()),
        })


direction_df = pd.DataFrame(
    direction_rows
)

direction_df.to_csv(
    OUT /
    "direction_by_ground_truth.csv",
    index=False,
)


# ============================================================
# LARGE ABSOLUTE DISAGREEMENT BANDS
#
# Fixed descriptive values only.
# Not optimized.
# ============================================================

band_rows = []


for threshold in [
    0.10,
    0.25,
    0.50,
    0.75,
]:

    for direction_name, direction_mask in groups:

        mask = (
            direction_mask
            & (D >= threshold)
        )

        n = int(mask.sum())

        if n == 0:
            continue

        band_rows.append({
            "D_threshold":
                threshold,

            "direction":
                direction_name,

            "pairs":
                n,

            "positive_prevalence":
                float(y[mask].mean()),

            "mean_D":
                float(D[mask].mean()),

            "mean_p_sequence":
                float(p_seq[mask].mean()),

            "mean_p_structure":
                float(p_struct[mask].mean()),

            "mean_sequence_error":
                float(seq_error[mask].mean()),

            "mean_structure_error":
                float(struct_error[mask].mean()),

            "sequence_better_fraction":
                float(seq_better[mask].mean()),

            "structure_better_fraction":
                float(struct_better[mask].mean()),
        })


band_df = pd.DataFrame(
    band_rows
)

band_df.to_csv(
    OUT /
    "fixed_disagreement_direction_bands.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "signed_disagreement_v1",

    "population":
        "structure-eligible calibration-eval",

    "proteins":
        len(eval_ids),

    "labels":
        494,

    "pairs":
        int(S.size),

    "signed_score":
        "p_sequence - p_structure",

    "sequence_higher_pairs":
        int((S > 0).sum()),

    "structure_higher_pairs":
        int((S < 0).sum()),

    "equal_pairs":
        int((S == 0).sum()),

    "conflict_threshold_optimized":
        False,

    "plddt_used":
        False,

    "test_used":
        False,
}


with open(
    OUT / "signed_disagreement_summary.json",
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
    f"Prediction pairs:    {S.size:,}"
)

print(
    f"Sequence higher:     "
    f"{int((S > 0).sum()):,} "
    f"({(S > 0).mean():.2%})"
)

print(
    f"Structure higher:    "
    f"{int((S < 0).sum()):,} "
    f"({(S < 0).mean():.2%})"
)

print()


print("=" * 92)
print("SIGNED-DISAGREEMENT DECILES")
print("=" * 92)

for _, r in signed_df.iterrows():

    print(
        f"Q{int(r['signed_decile']):02d}: "
        f"S={r['mean_S']:+.6f} | "
        f"D={r['mean_absolute_D']:.6f} | "
        f"positive={r['positive_prevalence']:.3%} | "
        f"pseq={r['mean_p_sequence']:.4f} | "
        f"pstruct={r['mean_p_structure']:.4f} | "
        f"seq better={r['sequence_better_fraction']:.2%} | "
        f"struct better={r['structure_better_fraction']:.2%}"
    )


print()
print("=" * 92)
print("DIRECTION × GROUND TRUTH")
print("=" * 92)

for _, r in direction_df.iterrows():

    print()
    print(
        f"{r['direction']} / "
        f"{r['ground_truth']}"
    )

    print(
        f"  pairs:             "
        f"{int(r['pairs']):,}"
    )

    print(
        f"  mean S:            "
        f"{r['mean_signed_disagreement']:+.6f}"
    )

    print(
        f"  mean D:            "
        f"{r['mean_absolute_disagreement']:.6f}"
    )

    print(
        f"  mean p_seq:        "
        f"{r['mean_p_sequence']:.6f}"
    )

    print(
        f"  mean p_struct:     "
        f"{r['mean_p_structure']:.6f}"
    )

    print(
        f"  sequence error:    "
        f"{r['mean_sequence_error']:.6f}"
    )

    print(
        f"  structure error:   "
        f"{r['mean_structure_error']:.6f}"
    )

    print(
        f"  sequence better:   "
        f"{r['sequence_better_fraction']:.2%}"
    )

    print(
        f"  structure better:  "
        f"{r['structure_better_fraction']:.2%}"
    )


print()
print("=" * 92)
print("FIXED LARGE-DISAGREEMENT BANDS")
print("=" * 92)

for _, r in band_df.iterrows():

    print(
        f"D >= {r['D_threshold']:.2f} | "
        f"{r['direction']:<16} | "
        f"n={int(r['pairs']):,} | "
        f"positive={r['positive_prevalence']:.2%} | "
        f"seq better={r['sequence_better_fraction']:.2%} | "
        f"struct better={r['structure_better_fraction']:.2%} | "
        f"seq err={r['mean_sequence_error']:.4f} | "
        f"struct err={r['mean_structure_error']:.4f}"
    )


print()
print("=" * 92)

print(
    "Signed disagreement analyzed:       YES"
)

print(
    "Conflict threshold optimized:       NO"
)

print(
    "pLDDT/global confidence used:       NO"
)

print(
    "Final TEST confirmation:            NO"
)

print(
    "TEST USED:                          NO"
)

print()

print(
    f"Saved under:\n  {OUT}"
)

print("=" * 92)