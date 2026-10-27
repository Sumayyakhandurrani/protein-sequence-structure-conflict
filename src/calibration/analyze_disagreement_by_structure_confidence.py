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

META = Path(
    "data/processed/primary_mf_cohort.csv"
)

OUT = Path(
    "data/conflict/"
    "structure_confidence_stratification_v1"
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
# LOAD PAIRED PREDICTIONS
# ============================================================

print("=" * 92)
print("DISAGREEMENT BY ALPHAFOLD GLOBAL CONFIDENCE V1")
print("=" * 92)

print("Population:          structure-eligible calibration-eval")
print("Confidence variable: global_metric")
print("Confidence input to GNN: NO")
print("Conflict threshold optimized: NO")
print("TEST USED:           NO")
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
# LOAD CONFIDENCE METADATA IN EXACT EVAL ORDER
# ============================================================

meta = pd.read_csv(META)

if meta["uniprot_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate UniProt IDs in metadata."
    )

meta = meta.set_index(
    "uniprot_id"
)


missing = [
    uid
    for uid in eval_ids
    if uid not in meta.index
]

if missing:
    raise RuntimeError(
        f"{len(missing)} evaluation IDs "
        "missing from metadata."
    )


confidence = np.asarray(
    [
        float(
            meta.loc[
                uid,
                "global_metric"
            ]
        )
        for uid in eval_ids
    ],
    dtype=np.float64,
)


if not np.isfinite(confidence).all():
    raise RuntimeError(
        "Non-finite global confidence found."
    )


# ============================================================
# CORE QUANTITIES
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
# CONFIDENCE BANDS
#
# Prespecified descriptive bands.
# No optimization performed.
# ============================================================

bands = [
    (
        "<50",
        confidence < 50,
    ),
    (
        "50-70",
        (confidence >= 50)
        & (confidence < 70),
    ),
    (
        "70-90",
        (confidence >= 70)
        & (confidence < 90),
    ),
    (
        ">=90",
        confidence >= 90,
    ),
]


records = []


for band_name, protein_mask in bands:

    n_proteins = int(
        protein_mask.sum()
    )

    if n_proteins == 0:
        continue


    band_D = D[
        protein_mask
    ].reshape(-1)

    band_y = y[
        protein_mask
    ].reshape(-1)

    band_seq_error = seq_error[
        protein_mask
    ].reshape(-1)

    band_struct_error = struct_error[
        protein_mask
    ].reshape(-1)

    band_mean_error = mean_error[
        protein_mask
    ].reshape(-1)

    band_event = either_large_error[
        protein_mask
    ].reshape(-1).astype(np.uint8)


    corr_mean_error = float(
        np.corrcoef(
            band_D,
            band_mean_error,
        )[0, 1]
    )


    records.append({
        "confidence_band":
            band_name,

        "proteins":
            n_proteins,

        "pairs":
            int(band_D.size),

        "mean_global_confidence":
            float(
                confidence[
                    protein_mask
                ].mean()
            ),

        "positive_prevalence":
            float(
                band_y.mean()
            ),

        "mean_disagreement":
            float(
                band_D.mean()
            ),

        "median_disagreement":
            float(
                np.median(
                    band_D
                )
            ),

        "q90_disagreement":
            float(
                np.quantile(
                    band_D,
                    0.90,
                )
            ),

        "q95_disagreement":
            float(
                np.quantile(
                    band_D,
                    0.95,
                )
            ),

        "q99_disagreement":
            float(
                np.quantile(
                    band_D,
                    0.99,
                )
            ),

        "mean_sequence_error":
            float(
                band_seq_error.mean()
            ),

        "mean_structure_error":
            float(
                band_struct_error.mean()
            ),

        "mean_expert_error":
            float(
                band_mean_error.mean()
            ),

        "large_error_events":
            int(
                band_event.sum()
            ),

        "large_error_prevalence":
            float(
                band_event.mean()
            ),

        "D_vs_mean_error_correlation":
            corr_mean_error,

        "D_large_error_ROC_AUC":
            safe_auc(
                band_event,
                band_D,
            ),

        "D_large_error_AP":
            safe_ap(
                band_event,
                band_D,
            ),
    })


results = pd.DataFrame(
    records
)

results.to_csv(
    OUT /
    "confidence_band_results.csv",
    index=False,
)


# ============================================================
# HIGH-CONFIDENCE DISAGREEMENT DECILES
#
# Critical analysis:
# does D still track error among proteins >=90 confidence?
# ============================================================

high_mask = (
    confidence >= 90
)

high_D = D[
    high_mask
]

high_mean_error = mean_error[
    high_mask
]

high_event = either_large_error[
    high_mask
]

high_y = y[
    high_mask
]


high_edges = np.quantile(
    high_D.reshape(-1),
    np.linspace(
        0.0,
        1.0,
        11,
    ),
)

high_edges[-1] = np.nextafter(
    high_edges[-1],
    np.inf,
)


decile_records = []


for i in range(10):

    lo = high_edges[i]
    hi = high_edges[i + 1]

    mask = (
        (high_D >= lo)
        & (high_D < hi)
    )

    if not mask.any():
        continue


    decile_records.append({
        "decile":
            i + 1,

        "lower_D":
            float(lo),

        "upper_D":
            float(hi),

        "pairs":
            int(mask.sum()),

        "mean_D":
            float(
                high_D[
                    mask
                ].mean()
            ),

        "positive_prevalence":
            float(
                high_y[
                    mask
                ].mean()
            ),

        "mean_expert_error":
            float(
                high_mean_error[
                    mask
                ].mean()
            ),

        "large_error_prevalence":
            float(
                high_event[
                    mask
                ].mean()
            ),
    })


deciles = pd.DataFrame(
    decile_records
)

deciles.to_csv(
    OUT /
    "high_confidence_disagreement_deciles.csv",
    index=False,
)


# ============================================================
# PROTEIN-LEVEL ASSOCIATION
#
# This avoids relying only on millions of label pairs.
# ============================================================

protein_mean_D = D.mean(
    axis=1
)

protein_mean_error = mean_error.mean(
    axis=1
)

protein_large_error_rate = (
    either_large_error.mean(
        axis=1
    )
)


protein_corr_D_conf = float(
    np.corrcoef(
        protein_mean_D,
        confidence,
    )[0, 1]
)

protein_corr_D_error = float(
    np.corrcoef(
        protein_mean_D,
        protein_mean_error,
    )[0, 1]
)


protein_df = pd.DataFrame({
    "uniprot_id":
        eval_ids,

    "global_confidence":
        confidence,

    "mean_disagreement":
        protein_mean_D,

    "mean_expert_error":
        protein_mean_error,

    "large_error_rate":
        protein_large_error_rate,
})


protein_df.to_csv(
    OUT /
    "protein_level_confidence_disagreement.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "structure_confidence_stratification_v1",

    "population":
        "structure-eligible calibration-eval",

    "proteins":
        len(eval_ids),

    "labels":
        494,

    "pairs":
        int(D.size),

    "confidence_variable":
        "global_metric",

    "confidence_variable_publication_name":
        (
            "AlphaFold global confidence "
            "pending exact metadata-semantic verification"
        ),

    "bands": [
        "<50",
        "50-70",
        "70-90",
        ">=90",
    ],

    "high_confidence_proteins":
        int(
            high_mask.sum()
        ),

    "protein_level_correlation_mean_D_vs_confidence":
        protein_corr_D_conf,

    "protein_level_correlation_mean_D_vs_mean_error":
        protein_corr_D_error,

    "plddt_used_as_model_input":
        False,

    "confidence_used_for_stratification_only":
        True,

    "conflict_threshold_optimized":
        False,

    "test_used":
        False,
}


with open(
    OUT /
    "structure_confidence_summary.json",
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

print()

print("CONFIDENCE BANDS")

for _, r in results.iterrows():

    print()
    print(
        f"  {r['confidence_band']}"
    )

    print(
        f"    proteins:               "
        f"{int(r['proteins']):,}"
    )

    print(
        f"    mean confidence:         "
        f"{r['mean_global_confidence']:.2f}"
    )

    print(
        f"    mean D:                  "
        f"{r['mean_disagreement']:.6f}"
    )

    print(
        f"    mean expert error:       "
        f"{r['mean_expert_error']:.6f}"
    )

    print(
        f"    large-error prevalence:  "
        f"{r['large_error_prevalence']:.4%}"
    )

    print(
        f"    D vs error correlation:  "
        f"{r['D_vs_mean_error_correlation']:.6f}"
    )

    print(
        f"    D error ROC-AUC:         "
        f"{r['D_large_error_ROC_AUC']:.6f}"
    )

    print(
        f"    D error AP:              "
        f"{r['D_large_error_AP']:.6f}"
    )


print()
print("=" * 92)
print(
    "HIGH STRUCTURAL CONFIDENCE "
    "(global_metric >= 90)"
)
print("=" * 92)

print(
    f"Proteins: "
    f"{int(high_mask.sum()):,}"
)

print()

for _, r in deciles.iterrows():

    print(
        f"  Q{int(r['decile']):02d}: "
        f"D={r['mean_D']:.6f} | "
        f"error={r['mean_expert_error']:.6f} | "
        f"large-error={r['large_error_prevalence']:.4%} | "
        f"positive-prev={r['positive_prevalence']:.4%}"
    )


print()
print("PROTEIN-LEVEL ASSOCIATIONS")

print(
    "  mean D vs global confidence: "
    f"{protein_corr_D_conf:+.6f}"
)

print(
    "  mean D vs mean expert error: "
    f"{protein_corr_D_error:+.6f}"
)

print()

print(
    "Confidence used as GNN input:     NO"
)

print(
    "Confidence stratification only:   YES"
)

print(
    "Conflict threshold optimized:     NO"
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