from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


BASE = Path("data/conflict/paired_experts_v1")
OUT = Path("data/conflict/disagreement_error_v1")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 84)
print("DISAGREEMENT–ERROR RELATIONSHIP V1")
print("=" * 84)
print("Purpose: descriptive association between disagreement and error")
print("Incremental value beyond confidence tested: NO")
print("Conflict threshold optimized: NO")
print("pLDDT used: NO")
print("TEST USED: NO")
print()


y = np.load(
    BASE / "paired_val_labels.npy"
).astype(np.float64)

p_seq = np.load(
    BASE / "sequence_probs.npy"
).astype(np.float64)

p_struct = np.load(
    BASE / "structure_probs.npy"
).astype(np.float64)


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


# ============================================================
# BASIC QUANTITIES
# ============================================================

D = np.abs(p_seq - p_struct)

seq_error = np.abs(y - p_seq)
struct_error = np.abs(y - p_struct)

best_error = np.minimum(
    seq_error,
    struct_error,
)

mean_expert_error = (
    seq_error + struct_error
) / 2.0


# Which expert is closer to truth?
seq_better = seq_error < struct_error
struct_better = struct_error < seq_error
equal_error = seq_error == struct_error


# ============================================================
# DISAGREEMENT DECILES
#
# Quantiles are descriptive bins, not optimized thresholds.
# ============================================================

flat_D = D.ravel()

quantile_edges = np.quantile(
    flat_D,
    np.linspace(0, 1, 11),
)


# Ensure final boundary contains max.
quantile_edges[-1] = np.nextafter(
    quantile_edges[-1],
    np.inf,
)


rows = []

for i in range(10):

    lo = quantile_edges[i]
    hi = quantile_edges[i + 1]

    if i == 0:
        mask = (
            (D >= lo)
            & (D < hi)
        )
    else:
        mask = (
            (D >= lo)
            & (D < hi)
        )

    n = int(mask.sum())

    if n == 0:
        continue

    positives = int(
        y[mask].sum()
    )

    rows.append({
        "decile":
            i + 1,

        "lower_D":
            float(lo),

        "upper_D":
            float(hi),

        "pairs":
            n,

        "positive_pairs":
            positives,

        "positive_prevalence":
            float(y[mask].mean()),

        "mean_D":
            float(D[mask].mean()),

        "mean_sequence_abs_error":
            float(seq_error[mask].mean()),

        "mean_structure_abs_error":
            float(struct_error[mask].mean()),

        "mean_best_expert_abs_error":
            float(best_error[mask].mean()),

        "mean_two_expert_abs_error":
            float(mean_expert_error[mask].mean()),

        "sequence_better_fraction":
            float(seq_better[mask].mean()),

        "structure_better_fraction":
            float(struct_better[mask].mean()),

        "equal_error_fraction":
            float(equal_error[mask].mean()),
    })


decile_df = pd.DataFrame(rows)

decile_df.to_csv(
    OUT / "disagreement_deciles.csv",
    index=False,
)


# ============================================================
# GROUND-TRUTH POSITIVE / NEGATIVE ANALYSIS
# ============================================================

class_rows = []

for class_name, mask in [
    ("positive", y == 1),
    ("negative", y == 0),
]:

    class_rows.append({
        "ground_truth":
            class_name,

        "pairs":
            int(mask.sum()),

        "mean_disagreement":
            float(D[mask].mean()),

        "mean_sequence_abs_error":
            float(seq_error[mask].mean()),

        "mean_structure_abs_error":
            float(struct_error[mask].mean()),

        "mean_best_expert_abs_error":
            float(best_error[mask].mean()),

        "sequence_better_fraction":
            float(seq_better[mask].mean()),

        "structure_better_fraction":
            float(struct_better[mask].mean()),
    })


class_df = pd.DataFrame(
    class_rows
)

class_df.to_csv(
    OUT / "ground_truth_stratified.csv",
    index=False,
)


# ============================================================
# LARGE ERROR EVENTS
#
# These are fixed descriptive definitions only.
# Not optimized conflict thresholds.
#
# Error >= 0.5 means the probability lies on the
# wrong side of the conventional 0.5 boundary.
# ============================================================

seq_large_error = (
    seq_error >= 0.5
).astype(np.uint8)

struct_large_error = (
    struct_error >= 0.5
).astype(np.uint8)

either_large_error = (
    (seq_large_error == 1)
    | (struct_large_error == 1)
).astype(np.uint8)

both_large_error = (
    (seq_large_error == 1)
    & (struct_large_error == 1)
).astype(np.uint8)


def association_metrics(target, score):

    target = target.ravel()
    score = score.ravel()

    positives = int(
        target.sum()
    )

    negatives = int(
        len(target) - positives
    )

    result = {
        "events":
            positives,

        "non_events":
            negatives,

        "prevalence":
            float(target.mean()),
    }

    if (
        positives > 0
        and negatives > 0
    ):
        result["roc_auc"] = float(
            roc_auc_score(
                target,
                score,
            )
        )

        result["average_precision"] = float(
            average_precision_score(
                target,
                score,
            )
        )

    else:
        result["roc_auc"] = None
        result["average_precision"] = None

    return result


association = {
    "sequence_large_error":
        association_metrics(
            seq_large_error,
            D,
        ),

    "structure_large_error":
        association_metrics(
            struct_large_error,
            D,
        ),

    "either_expert_large_error":
        association_metrics(
            either_large_error,
            D,
        ),

    "both_experts_large_error":
        association_metrics(
            both_large_error,
            D,
        ),
}


# ============================================================
# CORRELATIONS
# ============================================================

flat_D = D.ravel()

correlations = {
    "D_vs_sequence_abs_error":
        float(
            np.corrcoef(
                flat_D,
                seq_error.ravel(),
            )[0, 1]
        ),

    "D_vs_structure_abs_error":
        float(
            np.corrcoef(
                flat_D,
                struct_error.ravel(),
            )[0, 1]
        ),

    "D_vs_best_expert_abs_error":
        float(
            np.corrcoef(
                flat_D,
                best_error.ravel(),
            )[0, 1]
        ),

    "D_vs_mean_expert_abs_error":
        float(
            np.corrcoef(
                flat_D,
                mean_expert_error.ravel(),
            )[0, 1]
        ),
}


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "disagreement_error_relationship_v1",

    "proteins":
        4742,

    "labels":
        494,

    "prediction_pairs":
        int(D.size),

    "disagreement":
        "|p_sequence - p_structure|",

    "sequence_error":
        "|y - p_sequence|",

    "structure_error":
        "|y - p_structure|",

    "correlations":
        correlations,

    "large_error_associations":
        association,

    "incremental_value_beyond_confidence_tested":
        False,

    "conflict_definition_finalized":
        False,

    "conflict_threshold_optimized":
        False,

    "plddt_used":
        False,

    "test_used":
        False,
}


with open(
    OUT / "disagreement_error_summary.json",
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

print(f"Proteins:         {4742:,}")
print(f"Labels:           {494}")
print(f"Pairs:            {D.size:,}")

print()
print("CORRELATION WITH ABSOLUTE ERROR")

print(
    "  D vs sequence error:      "
    f"{correlations['D_vs_sequence_abs_error']:.6f}"
)

print(
    "  D vs structure error:     "
    f"{correlations['D_vs_structure_abs_error']:.6f}"
)

print(
    "  D vs best-expert error:   "
    f"{correlations['D_vs_best_expert_abs_error']:.6f}"
)

print(
    "  D vs mean-expert error:   "
    f"{correlations['D_vs_mean_expert_abs_error']:.6f}"
)

print()
print("DISAGREEMENT DECILES")

for _, r in decile_df.iterrows():

    print(
        f"  Q{int(r['decile']):02d}: "
        f"mean D={r['mean_D']:.6f} | "
        f"seq err={r['mean_sequence_abs_error']:.6f} | "
        f"struct err={r['mean_structure_abs_error']:.6f} | "
        f"best err={r['mean_best_expert_abs_error']:.6f} | "
        f"positive prev={r['positive_prevalence']:.4%}"
    )


print()
print("D AS A SCORE FOR LARGE ERROR EVENTS")

for name, result in association.items():

    print(
        f"  {name}:"
    )

    print(
        f"    events={result['events']:,} "
        f"({result['prevalence']:.4%})"
    )

    if result["roc_auc"] is not None:

        print(
            f"    ROC-AUC="
            f"{result['roc_auc']:.6f}"
        )

        print(
            f"    AP="
            f"{result['average_precision']:.6f}"
        )


print()
print(
    "Incremental value beyond confidence tested: NO"
)
print(
    "Conflict definition finalized:             NO"
)
print(
    "Conflict threshold optimized:              NO"
)
print(
    "pLDDT used:                                NO"
)
print(
    "TEST USED:                                 NO"
)

print()
print(
    f"Saved under:\n  {OUT}"
)

print("=" * 84)