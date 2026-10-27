from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# CONFIGURATION
# ============================================================

PRED_DIR = Path(
    "data/conflict/separate_expert_reliability_ablation_v1"
)

LABEL_FILE = Path(
    "data/splits/mmseqs30_graph_v2/primary_go_mf_labels.csv"
)

OUT = Path(
    "data/conflict/go_frequency_reliability_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

N_PROTEINS = 2373
N_LABELS = 494

# Frozen TRAIN-derived support bands.
BANDS = [
    ("75-99", 75, 99),
    ("100-249", 100, 249),
    ("250-499", 250, 499),
    ("500-999", 500, 999),
    (">=1000", 1000, np.inf),
]


# ============================================================
# HELPERS
# ============================================================

def load_matrix(filename):
    x = np.load(PRED_DIR / filename)

    expected = N_PROTEINS * N_LABELS

    if x.size != expected:
        raise RuntimeError(
            f"{filename}: expected {expected:,} values, "
            f"got {x.size:,}"
        )

    return x.reshape(N_PROTEINS, N_LABELS)


def binary_metrics(y, p):
    y = y.reshape(-1)
    p = p.reshape(-1)

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    out = {
        "n_pairs": int(len(y)),
        "n_positive_events": positives,
        "n_negative_events": negatives,
        "event_prevalence": float(y.mean()),
    }

    # ROC-AUC requires both classes.
    if positives > 0 and negatives > 0:
        out["roc_auc"] = float(
            roc_auc_score(y, p)
        )
    else:
        out["roc_auc"] = np.nan

    # AP is meaningful if positives exist.
    if positives > 0:
        out["average_precision"] = float(
            average_precision_score(y, p)
        )
    else:
        out["average_precision"] = np.nan

    if positives > 0 and negatives > 0:
        out["log_loss"] = float(
            log_loss(y, p, labels=[0, 1])
        )
    else:
        out["log_loss"] = np.nan

    out["brier"] = float(
        brier_score_loss(y, p)
    )

    return out


def continuous_metrics(y, pred):
    y = y.reshape(-1)
    pred = pred.reshape(-1)

    mae = float(
        mean_absolute_error(y, pred)
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(y, pred)
        )
    )

    if np.std(y) > 0 and np.std(pred) > 0:
        corr = float(
            np.corrcoef(y, pred)[0, 1]
        )
    else:
        corr = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
    }


# ============================================================
# LOAD AND VERIFY LABEL SUPPORT
# ============================================================

print("=" * 100)
print("GO-FREQUENCY-STRATIFIED RELIABILITY ANALYSIS V1")
print("=" * 100)

print("Expert analyzed:       STRUCTURE")
print("Primary comparison:    MB - MA")
print("MA:                    U_structure + U_sequence")
print("MB:                    U_structure + U_sequence + D")
print("D:                     |p_sequence - p_structure|")
print("GO support source:     TRAIN ONLY")
print("Models refit:          NO")
print("Threshold optimized:   NO")
print("pLDDT used:            NO")
print("TEST USED:             NO")
print()


labels = pd.read_csv(LABEL_FILE)

required_columns = {
    "go_id",
    "train_protein_count",
    "label_index",
}

if not required_columns.issubset(labels.columns):
    raise RuntimeError(
        "Label file missing required columns."
    )

if len(labels) != N_LABELS:
    raise RuntimeError(
        f"Expected {N_LABELS} labels, got {len(labels)}"
    )

if labels["label_index"].nunique() != N_LABELS:
    raise RuntimeError(
        "label_index is not unique."
    )

if set(labels["label_index"]) != set(range(N_LABELS)):
    raise RuntimeError(
        "label_index is not exactly 0..493."
    )

# Force exact prediction-column order.
labels = (
    labels
    .sort_values("label_index")
    .reset_index(drop=True)
)

print(
    f"Frozen GO labels:      {len(labels):,}"
)
print(
    f"TRAIN support range:   "
    f"{labels['train_protein_count'].min():,} "
    f"to "
    f"{labels['train_protein_count'].max():,}"
)
print()


# ============================================================
# LOAD STRUCTURE RELIABILITY PREDICTIONS
# ============================================================

binary_target = load_matrix(
    "structure_binary_target.npy"
)

binary_MA = load_matrix(
    "structure_binary_MA_predictions.npy"
)

binary_MB = load_matrix(
    "structure_binary_MB_predictions.npy"
)

continuous_target = load_matrix(
    "structure_continuous_target.npy"
)

continuous_MA = load_matrix(
    "structure_continuous_MA_predictions.npy"
)

continuous_MB = load_matrix(
    "structure_continuous_MB_predictions.npy"
)

print(
    f"Evaluation proteins:   {N_PROTEINS:,}"
)
print(
    f"Labels per protein:    {N_LABELS}"
)
print(
    f"Total pairs:           "
    f"{N_PROTEINS * N_LABELS:,}"
)
print()


# ============================================================
# ANALYZE SUPPORT BANDS
# ============================================================

rows = []
json_results = {}


print("=" * 100)
print("SUPPORT-STRATIFIED RESULTS")
print("=" * 100)


for band_name, low, high in BANDS:

    if np.isinf(high):
        mask = (
            labels["train_protein_count"].to_numpy()
            >= low
        )
    else:
        support = (
            labels["train_protein_count"].to_numpy()
        )

        mask = (
            (support >= low)
            & (support <= high)
        )

    label_indices = np.where(mask)[0]

    if len(label_indices) == 0:
        raise RuntimeError(
            f"No labels found for band {band_name}"
        )


    y_binary = binary_target[:, label_indices]
    p_MA = binary_MA[:, label_indices]
    p_MB = binary_MB[:, label_indices]

    y_cont = continuous_target[:, label_indices]
    c_MA = continuous_MA[:, label_indices]
    c_MB = continuous_MB[:, label_indices]


    binary_ma = binary_metrics(
        y_binary,
        p_MA,
    )

    binary_mb = binary_metrics(
        y_binary,
        p_MB,
    )

    cont_ma = continuous_metrics(
        y_cont,
        c_MA,
    )

    cont_mb = continuous_metrics(
        y_cont,
        c_MB,
    )


    deltas = {
        "roc_auc":
            binary_mb["roc_auc"]
            - binary_ma["roc_auc"],

        "average_precision":
            binary_mb["average_precision"]
            - binary_ma["average_precision"],

        "log_loss":
            binary_mb["log_loss"]
            - binary_ma["log_loss"],

        "brier":
            binary_mb["brier"]
            - binary_ma["brier"],

        "mae":
            cont_mb["mae"]
            - cont_ma["mae"],

        "rmse":
            cont_mb["rmse"]
            - cont_ma["rmse"],

        "correlation":
            cont_mb["correlation"]
            - cont_ma["correlation"],
    }


    support_values = labels.loc[
        mask,
        "train_protein_count"
    ]

    print()
    print("-" * 100)
    print(f"TRAIN SUPPORT BAND: {band_name}")
    print("-" * 100)

    print(
        f"Labels:               {len(label_indices):,}"
    )

    print(
        f"Support min/max:      "
        f"{support_values.min():,} / "
        f"{support_values.max():,}"
    )

    print(
        f"Protein-label pairs:  "
        f"{binary_ma['n_pairs']:,}"
    )

    print(
        f"Error events:         "
        f"{binary_ma['n_positive_events']:,} "
        f"({binary_ma['event_prevalence']:.4%})"
    )

    print()
    print("BINARY STRUCTURE-ERROR PREDICTION")

    print(
        f"  ROC-AUC  "
        f"MA={binary_ma['roc_auc']:.6f} | "
        f"MB={binary_mb['roc_auc']:.6f} | "
        f"delta={deltas['roc_auc']:+.6f}"
    )

    print(
        f"  AP       "
        f"MA={binary_ma['average_precision']:.6f} | "
        f"MB={binary_mb['average_precision']:.6f} | "
        f"delta={deltas['average_precision']:+.6f}"
    )

    print(
        f"  Log-loss "
        f"MA={binary_ma['log_loss']:.6f} | "
        f"MB={binary_mb['log_loss']:.6f} | "
        f"delta={deltas['log_loss']:+.6f}"
    )

    print(
        f"  Brier    "
        f"MA={binary_ma['brier']:.6f} | "
        f"MB={binary_mb['brier']:.6f} | "
        f"delta={deltas['brier']:+.6f}"
    )

    print()
    print("CONTINUOUS STRUCTURE ABSOLUTE ERROR")

    print(
        f"  MAE      "
        f"MA={cont_ma['mae']:.6f} | "
        f"MB={cont_mb['mae']:.6f} | "
        f"delta={deltas['mae']:+.6f}"
    )

    print(
        f"  RMSE     "
        f"MA={cont_ma['rmse']:.6f} | "
        f"MB={cont_mb['rmse']:.6f} | "
        f"delta={deltas['rmse']:+.6f}"
    )

    print(
        f"  Corr     "
        f"MA={cont_ma['correlation']:.6f} | "
        f"MB={cont_mb['correlation']:.6f} | "
        f"delta={deltas['correlation']:+.6f}"
    )


    row = {
        "support_band": band_name,
        "n_labels": len(label_indices),
        "support_min": int(support_values.min()),
        "support_max": int(support_values.max()),

        "n_pairs":
            binary_ma["n_pairs"],

        "n_error_events":
            binary_ma["n_positive_events"],

        "error_prevalence":
            binary_ma["event_prevalence"],

        "MA_roc_auc":
            binary_ma["roc_auc"],

        "MB_roc_auc":
            binary_mb["roc_auc"],

        "delta_roc_auc":
            deltas["roc_auc"],

        "MA_average_precision":
            binary_ma["average_precision"],

        "MB_average_precision":
            binary_mb["average_precision"],

        "delta_average_precision":
            deltas["average_precision"],

        "MA_log_loss":
            binary_ma["log_loss"],

        "MB_log_loss":
            binary_mb["log_loss"],

        "delta_log_loss":
            deltas["log_loss"],

        "MA_brier":
            binary_ma["brier"],

        "MB_brier":
            binary_mb["brier"],

        "delta_brier":
            deltas["brier"],

        "MA_mae":
            cont_ma["mae"],

        "MB_mae":
            cont_mb["mae"],

        "delta_mae":
            deltas["mae"],

        "MA_rmse":
            cont_ma["rmse"],

        "MB_rmse":
            cont_mb["rmse"],

        "delta_rmse":
            deltas["rmse"],

        "MA_correlation":
            cont_ma["correlation"],

        "MB_correlation":
            cont_mb["correlation"],

        "delta_correlation":
            deltas["correlation"],
    }

    rows.append(row)

    json_results[band_name] = row


# ============================================================
# VERIFY BAND COVERAGE
# ============================================================

covered_labels = sum(
    row["n_labels"]
    for row in rows
)

print()
print("=" * 100)
print("COVERAGE CHECK")
print("=" * 100)

print(
    f"Labels assigned to bands: "
    f"{covered_labels:,}/{N_LABELS:,}"
)

if covered_labels != N_LABELS:
    raise RuntimeError(
        "Support bands do not cover exactly all 494 labels."
    )

print("All frozen labels covered exactly once: YES")


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(rows)

results_df.to_csv(
    OUT / "go_frequency_reliability.csv",
    index=False,
)

summary = {
    "analysis":
        "go_frequency_stratified_structure_reliability_v1",

    "support_source":
        "frozen TRAIN protein counts only",

    "expert":
        "structure",

    "comparison":
        "MB minus MA",

    "MA_features":
        [
            "structure_uncertainty",
            "sequence_uncertainty",
        ],

    "MB_features":
        [
            "structure_uncertainty",
            "sequence_uncertainty",
            "absolute_disagreement",
        ],

    "models_refit":
        False,

    "threshold_optimized":
        False,

    "plddt_used":
        False,

    "test_used":
        False,

    "bands":
        json_results,
}

with open(
    OUT / "go_frequency_reliability_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 100)
print("FINAL STATUS")
print("=" * 100)

print("GO frequency from TRAIN only:      YES")
print("Frozen 494-label ordering used:    YES")
print("Reliability models refit:          NO")
print("Primary comparison:                MB - MA")
print("Threshold optimization:            NO")
print("pLDDT used:                        NO")
print("TEST USED:                         NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)