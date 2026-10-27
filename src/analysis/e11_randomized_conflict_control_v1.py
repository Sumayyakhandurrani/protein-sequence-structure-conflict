from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE = Path("data/conflict/paired_experts_v1")

SPLIT = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

OUT = Path(
    "data/analysis/randomized_conflict_control_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_LABELS = 494


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def uncertainty(p):
    return 1.0 - 2.0 * np.abs(p - 0.5)


def binary_metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(
            average_precision_score(y, p)
        ),
        "log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
    }


def continuous_metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y, pred))
        ),
        "correlation": float(
            np.corrcoef(y, pred)[0, 1]
        ),
    }


# ============================================================
# LOAD FROZEN PAIRED VAL DATA
# ============================================================

paired_ids = read_ids(
    BASE / "paired_val_uniprot_ids.txt"
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

if not (
    y_all.shape
    == p_seq_all.shape
    == p_struct_all.shape
):
    raise RuntimeError("Paired array shapes disagree.")

if y_all.shape[1] != N_LABELS:
    raise RuntimeError(
        f"Expected {N_LABELS} labels."
    )


# ============================================================
# CAL / CAL-EVAL
# ============================================================

paired_index = {
    uid: i for i, uid in enumerate(paired_ids)
}

cal_ids = read_ids(
    SPLIT / "structure_calibration_ids.txt"
)

eval_ids = read_ids(
    SPLIT / "structure_cal_eval_ids.txt"
)

if set(cal_ids) & set(eval_ids):
    raise RuntimeError("CAL/CAL-EVAL overlap.")

cal_rows = np.asarray(
    [paired_index[x] for x in cal_ids],
    dtype=np.int64,
)

eval_rows = np.asarray(
    [paired_index[x] for x in eval_ids],
    dtype=np.int64,
)

y_cal = y_all[cal_rows]
y_eval = y_all[eval_rows]

seq_cal = p_seq_all[cal_rows]
seq_eval = p_seq_all[eval_rows]

struct_cal = p_struct_all[cal_rows]
struct_eval = p_struct_all[eval_rows]


# ============================================================
# FEATURES
# ============================================================

U_seq_cal = uncertainty(seq_cal)
U_seq_eval = uncertainty(seq_eval)

U_struct_cal = uncertainty(struct_cal)
U_struct_eval = uncertainty(struct_eval)

D_cal = np.abs(seq_cal - struct_cal)
D_eval = np.abs(seq_eval - struct_eval)


# Independent whole-protein permutations
rng_cal = np.random.default_rng(SEED)
rng_eval = np.random.default_rng(SEED)

perm_cal = rng_cal.permutation(len(cal_ids))
perm_eval = rng_eval.permutation(len(eval_ids))

D_random_cal = D_cal[perm_cal]
D_random_eval = D_eval[perm_eval]


# Distribution must be exactly preserved by permutation
if not np.array_equal(
    np.sort(D_cal.ravel()),
    np.sort(D_random_cal.ravel()),
):
    raise RuntimeError(
        "CAL conflict distribution changed."
    )

if not np.array_equal(
    np.sort(D_eval.ravel()),
    np.sort(D_random_eval.ravel()),
):
    raise RuntimeError(
        "CAL-EVAL conflict distribution changed."
    )


# ============================================================
# TARGETS
# ============================================================

seq_abs_cal = np.abs(y_cal - seq_cal)
seq_abs_eval = np.abs(y_eval - seq_eval)

struct_abs_cal = np.abs(y_cal - struct_cal)
struct_abs_eval = np.abs(y_eval - struct_eval)

seq_binary_cal = (
    seq_abs_cal >= 0.5
).astype(np.uint8)

seq_binary_eval = (
    seq_abs_eval >= 0.5
).astype(np.uint8)

struct_binary_cal = (
    struct_abs_cal >= 0.5
).astype(np.uint8)

struct_binary_eval = (
    struct_abs_eval >= 0.5
).astype(np.uint8)


# ============================================================
# RUN EXPERT
# ============================================================

def run_expert(
    name,
    own_cal,
    other_cal,
    own_eval,
    other_eval,
    binary_cal,
    binary_eval,
    continuous_cal,
    continuous_eval,
):

    # Flatten only after protein-level permutation.
    own_cal = own_cal.reshape(-1)
    other_cal = other_cal.reshape(-1)

    own_eval = own_eval.reshape(-1)
    other_eval = other_eval.reshape(-1)

    real_cal = D_cal.reshape(-1)
    real_eval = D_eval.reshape(-1)

    random_cal = D_random_cal.reshape(-1)
    random_eval = D_random_eval.reshape(-1)

    binary_cal = binary_cal.reshape(-1)
    binary_eval = binary_eval.reshape(-1)

    continuous_cal = continuous_cal.reshape(-1)
    continuous_eval = continuous_eval.reshape(-1)

    XA_cal = np.column_stack([
        own_cal,
        other_cal,
    ])

    XA_eval = np.column_stack([
        own_eval,
        other_eval,
    ])

    XREAL_cal = np.column_stack([
        own_cal,
        other_cal,
        real_cal,
    ])

    XREAL_eval = np.column_stack([
        own_eval,
        other_eval,
        real_eval,
    ])

    XRAND_cal = np.column_stack([
        own_cal,
        other_cal,
        random_cal,
    ])

    XRAND_eval = np.column_stack([
        own_eval,
        other_eval,
        random_eval,
    ])

    feature_sets_cal = {
        "MA": XA_cal,
        "MB_real": XREAL_cal,
        "MB_random": XRAND_cal,
    }

    feature_sets_eval = {
        "MA": XA_eval,
        "MB_real": XREAL_eval,
        "MB_random": XRAND_eval,
    }

    binary_results = {}
    binary_predictions = {}
    continuous_results = {}
    binary_coefficients = {}
    continuous_coefficients = {}

    for model_name in [
        "MA",
        "MB_real",
        "MB_random",
    ]:
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

        model.fit(
            feature_sets_cal[model_name],
            binary_cal,
        )

        pred = model.predict_proba(
            feature_sets_eval[model_name]
        )[:, 1]

        binary_results[model_name] = (
            binary_metrics(binary_eval, pred)
        )

        binary_predictions[model_name] = pred

        binary_coefficients[model_name] = [
            float(x)
            for x in model.coef_[0]
        ]

    for model_name in [
        "MA",
        "MB_real",
        "MB_random",
    ]:
        model = Ridge(alpha=1.0)

        model.fit(
            feature_sets_cal[model_name],
            continuous_cal,
        )

        pred = model.predict(
            feature_sets_eval[model_name]
        )

        continuous_results[model_name] = (
            continuous_metrics(
                continuous_eval,
                pred,
            )
        )

        continuous_coefficients[
            model_name
        ] = [
            float(x)
            for x in model.coef_
        ]

    # Save fixed CAL-EVAL binary predictions for
    # protein-level bootstrap without model refitting.
    for model_name, pred in binary_predictions.items():
        np.save(
            OUT / f"{name}_binary_{model_name}_predictions.npy",
            pred.reshape(len(eval_ids), N_LABELS).astype(np.float32),
        )

    np.save(
        OUT / f"{name}_binary_target.npy",
        binary_eval.reshape(len(eval_ids), N_LABELS).astype(np.uint8),
    )

    return {
        "expert": name,
        "binary": binary_results,
        "continuous": continuous_results,
        "binary_coefficients":
            binary_coefficients,
        "continuous_coefficients":
            continuous_coefficients,
    }


sequence = run_expert(
    "sequence",
    U_seq_cal,
    U_struct_cal,
    U_seq_eval,
    U_struct_eval,
    seq_binary_cal,
    seq_binary_eval,
    seq_abs_cal,
    seq_abs_eval,
)

structure = run_expert(
    "structure",
    U_struct_cal,
    U_seq_cal,
    U_struct_eval,
    U_seq_eval,
    struct_binary_cal,
    struct_binary_eval,
    struct_abs_cal,
    struct_abs_eval,
)


# ============================================================
# SUMMARY TABLE
# ============================================================

rows = []

for result in [sequence, structure]:
    name = result["expert"]

    b = result["binary"]
    c = result["continuous"]

    rows.append({
        "expert": name,

        "MA_AP":
            b["MA"]["average_precision"],

        "MB_random_AP":
            b["MB_random"]["average_precision"],

        "MB_real_AP":
            b["MB_real"]["average_precision"],

        "real_minus_random_AP":
            b["MB_real"]["average_precision"]
            - b["MB_random"]["average_precision"],

        "random_minus_MA_AP":
            b["MB_random"]["average_precision"]
            - b["MA"]["average_precision"],

        "real_minus_MA_AP":
            b["MB_real"]["average_precision"]
            - b["MA"]["average_precision"],

        "MA_logloss":
            b["MA"]["log_loss"],

        "MB_random_logloss":
            b["MB_random"]["log_loss"],

        "MB_real_logloss":
            b["MB_real"]["log_loss"],

        "MA_brier":
            b["MA"]["brier"],

        "MB_random_brier":
            b["MB_random"]["brier"],

        "MB_real_brier":
            b["MB_real"]["brier"],

        "MA_continuous_MAE":
            c["MA"]["mae"],

        "MB_random_continuous_MAE":
            c["MB_random"]["mae"],

        "MB_real_continuous_MAE":
            c["MB_real"]["mae"],

        "MA_continuous_corr":
            c["MA"]["correlation"],

        "MB_random_continuous_corr":
            c["MB_random"]["correlation"],

        "MB_real_continuous_corr":
            c["MB_real"]["correlation"],
    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "randomized_conflict_summary.csv",
    index=False,
)


# ============================================================
# SAVE DETAILS
# ============================================================

details = {
    "analysis":
        "randomized_conflict_control_v1",

    "seed": SEED,

    "randomization_unit":
        "whole protein",

    "calibration_proteins":
        len(cal_ids),

    "evaluation_proteins":
        len(eval_ids),

    "labels":
        N_LABELS,

    "conflict":
        "absolute disagreement",

    "test_used":
        False,

    "sequence":
        sequence,

    "structure":
        structure,
}

with open(
    OUT / "randomized_conflict_details.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        details,
        f,
        indent=2,
    )


# ============================================================
# PRINT
# ============================================================

print("=" * 80)
print("E11 RANDOMIZED-CONFLICT CONTROL")
print("=" * 80)

print(f"CAL proteins:      {len(cal_ids):,}")
print(f"CAL-EVAL proteins: {len(eval_ids):,}")
print(f"Labels:            {N_LABELS}")
print(f"Seed:              {SEED}")
print("Randomization:     whole-protein conflict profiles")
print("TEST used:         NO")

print()
print(summary.to_string(index=False))

print()
print("Saved:")
print(OUT / "randomized_conflict_summary.csv")
print(OUT / "randomized_conflict_details.json")
