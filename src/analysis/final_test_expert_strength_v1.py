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
# FROZEN CONFIGURATION
# ============================================================

SEED = 42
N_BOOT = 1000
EPS = 1e-7
N_LABELS = 494

CAL_DIR = Path(
    "data/conflict/go_frequency_v1_inputs"
)

MATCH_FILE = Path(
    "data/conflict/expert_strength_control_v1/"
    "primary_matched_labels.csv"
)

TEST_DIR = Path(
    "data/final_evaluation/final_test_primary_v1"
)

OUT = Path(
    "data/final_evaluation/"
    "final_test_expert_strength_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEFINITIONS
# ============================================================

def uncertainty(p):
    return (
        1.0
        - 2.0 * np.abs(p - 0.5)
    )


def bernoulli_js(ps, pt):

    ps = np.clip(
        ps,
        EPS,
        1-EPS,
    )

    pt = np.clip(
        pt,
        EPS,
        1-EPS,
    )

    m = 0.5 * (
        ps + pt
    )

    return 0.5 * (
        ps * np.log(ps / m)
        +
        (1-ps)
        * np.log(
            (1-ps) / (1-m)
        )
        +
        pt * np.log(pt / m)
        +
        (1-pt)
        * np.log(
            (1-pt) / (1-m)
        )
    )


def binary_metrics(y, p):

    return {
        "roc_auc":
            float(
                roc_auc_score(y, p)
            ),

        "average_precision":
            float(
                average_precision_score(y, p)
            ),

        "log_loss":
            float(
                log_loss(
                    y,
                    p,
                    labels=[0, 1],
                )
            ),

        "brier":
            float(
                brier_score_loss(y, p)
            ),
    }


def continuous_metrics(y, p):

    if (
        np.std(y) > 0
        and np.std(p) > 0
    ):
        corr = float(
            np.corrcoef(y, p)[0, 1]
        )
    else:
        corr = np.nan

    return {
        "mae":
            float(
                mean_absolute_error(y, p)
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(y, p)
                )
            ),

        "correlation":
            corr,
    }


# ============================================================
# LOAD CAL
# ============================================================

ps_cal = np.load(
    CAL_DIR / "p_seq_cal.npy"
).astype(np.float64)

pt_cal = np.load(
    CAL_DIR / "p_struct_cal.npy"
).astype(np.float64)

y_cal = np.load(
    CAL_DIR / "y_cal.npy"
).astype(np.float64)


# ============================================================
# LOAD FROZEN PAIRED TEST
# ============================================================

ps_test = np.load(
    TEST_DIR /
    "paired_sequence_probs.npy"
).astype(np.float64)

pt_test = np.load(
    TEST_DIR /
    "paired_structure_probs.npy"
).astype(np.float64)

y_test = np.load(
    TEST_DIR /
    "paired_test_labels.npy"
).astype(np.float64)


assert (
    ps_cal.shape
    == pt_cal.shape
    == y_cal.shape
    == (2369, 494)
)

assert (
    ps_test.shape
    == pt_test.shape
    == y_test.shape
    == (4731, 494)
)


# ============================================================
# LOAD EXACT PREDEFINED 92 LABELS
# ============================================================

matched = pd.read_csv(
    MATCH_FILE
)

if "label_index" not in matched.columns:
    raise RuntimeError(
        "primary_matched_labels.csv "
        "does not contain label_index."
    )

cols = np.sort(
    matched[
        "label_index"
    ].to_numpy(dtype=int)
)

assert len(cols) == 92
assert len(np.unique(cols)) == 92
assert cols.min() >= 0
assert cols.max() < N_LABELS


# ============================================================
# SELECT FROZEN SUBSET
# ============================================================

ps_c = ps_cal[:, cols]
pt_c = pt_cal[:, cols]
yc = y_cal[:, cols]

ps_t = ps_test[:, cols]
pt_t = pt_test[:, cols]
yt = y_test[:, cols]


# ============================================================
# FEATURES
#
# MA = [U_structure, U_sequence]
# MB = [U_structure, U_sequence, Bernoulli_JS]
# ============================================================

us_c = uncertainty(ps_c)
ut_c = uncertainty(pt_c)
js_c = bernoulli_js(
    ps_c,
    pt_c,
)

us_t = uncertainty(ps_t)
ut_t = uncertainty(pt_t)
js_t = bernoulli_js(
    ps_t,
    pt_t,
)


XA_cal = np.stack(
    [
        ut_c,
        us_c,
    ],
    axis=-1,
).reshape(-1, 2)

XB_cal = np.stack(
    [
        ut_c,
        us_c,
        js_c,
    ],
    axis=-1,
).reshape(-1, 3)


XA_test = np.stack(
    [
        ut_t,
        us_t,
    ],
    axis=-1,
).reshape(-1, 2)

XB_test = np.stack(
    [
        ut_t,
        us_t,
        js_t,
    ],
    axis=-1,
).reshape(-1, 3)


# ============================================================
# STRUCTURE ERROR TARGETS
# ============================================================

err_cal = np.abs(
    yc - pt_c
)

err_test = np.abs(
    yt - pt_t
)

binary_cal_matrix = (
    err_cal >= 0.5
).astype(np.uint8)

binary_test_matrix = (
    err_test >= 0.5
).astype(np.uint8)

binary_cal = (
    binary_cal_matrix.ravel()
)

binary_test = (
    binary_test_matrix.ravel()
)

continuous_cal = (
    err_cal.ravel()
)

continuous_test = (
    err_test.ravel()
)


# ============================================================
# FIT ON CAL ONLY
# ============================================================

model_A_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

model_B_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

model_A_bin.fit(
    XA_cal,
    binary_cal,
)

model_B_bin.fit(
    XB_cal,
    binary_cal,
)


model_A_cont = Ridge(
    alpha=1.0
)

model_B_cont = Ridge(
    alpha=1.0
)

model_A_cont.fit(
    XA_cal,
    continuous_cal,
)

model_B_cont.fit(
    XB_cal,
    continuous_cal,
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

pred_A_bin = (
    model_A_bin
    .predict_proba(
        XA_test
    )[:, 1]
)

pred_B_bin = (
    model_B_bin
    .predict_proba(
        XB_test
    )[:, 1]
)

pred_A_cont = (
    model_A_cont.predict(
        XA_test
    )
)

pred_B_cont = (
    model_B_cont.predict(
        XB_test
    )
)


# ============================================================
# POINT METRICS
# ============================================================

A_bin = binary_metrics(
    binary_test,
    pred_A_bin,
)

B_bin = binary_metrics(
    binary_test,
    pred_B_bin,
)

A_cont = continuous_metrics(
    continuous_test,
    pred_A_cont,
)

B_cont = continuous_metrics(
    continuous_test,
    pred_B_cont,
)


bin_names = [
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
]

cont_names = [
    "mae",
    "rmse",
    "correlation",
]


delta_bin = {
    k:
        B_bin[k] - A_bin[k]
    for k in bin_names
}

delta_cont = {
    k:
        B_cont[k] - A_cont[k]
    for k in cont_names
}


# ============================================================
# REPORT POINT RESULTS
# ============================================================

print("=" * 100)
print(
    "FINAL TEST V1 — "
    "EXPERT-STRENGTH CONTROL"
)
print("=" * 100)

print("Matching population:          CAL only")
print("Matching metric:              per-label Average Precision")
print("Frozen AP-gap cutoff:         <= 0.05")
print("Minimum CAL positives:        >= 5")
print(f"Frozen matched labels:        {len(cols)}")
print("Labels re-matched on TEST:    NO")
print("Primary conflict:             Bernoulli JS")
print("Reliability fitting:          CAL only")
print("TEST tuning:                  NO")
print()

print(
    f"CAL pairs:                    "
    f"{len(binary_cal):,}"
)

print(
    f"TEST pairs:                   "
    f"{len(binary_test):,}"
)

print(
    f"CAL failures:                 "
    f"{int(binary_cal.sum()):,} "
    f"({binary_cal.mean():.6%})"
)

print(
    f"TEST failures:                "
    f"{int(binary_test.sum()):,} "
    f"({binary_test.mean():.6%})"
)

print()

print(
    "MA = [U_structure, U_sequence]"
)

print(
    "MB = [U_structure, U_sequence, "
    "Bernoulli_JS]"
)

print()

print("-" * 100)
print("BINARY STRUCTURE-FAILURE RELIABILITY")
print("-" * 100)

for metric in bin_names:

    print(
        f"{metric:<20}"
        f"MA={A_bin[metric]:.8f} | "
        f"MB={B_bin[metric]:.8f} | "
        f"delta={delta_bin[metric]:+.8f}"
    )


print()
print("-" * 100)
print("CONTINUOUS STRUCTURE ABSOLUTE ERROR")
print("-" * 100)

for metric in cont_names:

    print(
        f"{metric:<20}"
        f"MA={A_cont[metric]:.8f} | "
        f"MB={B_cont[metric]:.8f} | "
        f"delta={delta_cont[metric]:+.8f}"
    )


# ============================================================
# WHOLE-PROTEIN BOOTSTRAP
# ============================================================

shape = (
    len(y_test),
    len(cols),
)

A_bin_matrix = (
    pred_A_bin.reshape(shape)
)

B_bin_matrix = (
    pred_B_bin.reshape(shape)
)

A_cont_matrix = (
    pred_A_cont.reshape(shape)
)

B_cont_matrix = (
    pred_B_cont.reshape(shape)
)


rng = np.random.default_rng(
    SEED
)

boot = np.empty(
    (
        N_BOOT,
        len(bin_names)
        + len(cont_names),
    ),
    dtype=np.float64,
)


print()
print(
    f"Running {N_BOOT} "
    "whole-protein bootstrap replicates..."
)


for b in range(N_BOOT):

    idx = rng.integers(
        0,
        len(y_test),
        size=len(y_test),
    )

    yb = (
        binary_test_matrix[
            idx
        ].ravel()
    )

    ya = (
        err_test[
            idx
        ].ravel()
    )

    pab = (
        A_bin_matrix[
            idx
        ].ravel()
    )

    pbb = (
        B_bin_matrix[
            idx
        ].ravel()
    )

    pac = (
        A_cont_matrix[
            idx
        ].ravel()
    )

    pbc = (
        B_cont_matrix[
            idx
        ].ravel()
    )

    boot[b, 0] = (
        roc_auc_score(
            yb,
            pbb,
        )
        -
        roc_auc_score(
            yb,
            pab,
        )
    )

    boot[b, 1] = (
        average_precision_score(
            yb,
            pbb,
        )
        -
        average_precision_score(
            yb,
            pab,
        )
    )

    boot[b, 2] = (
        log_loss(
            yb,
            pbb,
            labels=[0, 1],
        )
        -
        log_loss(
            yb,
            pab,
            labels=[0, 1],
        )
    )

    boot[b, 3] = (
        brier_score_loss(
            yb,
            pbb,
        )
        -
        brier_score_loss(
            yb,
            pab,
        )
    )

    boot[b, 4] = (
        mean_absolute_error(
            ya,
            pbc,
        )
        -
        mean_absolute_error(
            ya,
            pac,
        )
    )

    boot[b, 5] = (
        np.sqrt(
            mean_squared_error(
                ya,
                pbc,
            )
        )
        -
        np.sqrt(
            mean_squared_error(
                ya,
                pac,
            )
        )
    )

    corr_a = np.corrcoef(
        ya,
        pac,
    )[0, 1]

    corr_b = np.corrcoef(
        ya,
        pbc,
    )[0, 1]

    boot[b, 6] = (
        corr_b - corr_a
    )

    if (
        (b + 1) % 100
        == 0
    ):
        print(
            f"Completed "
            f"{b+1}/{N_BOOT}"
        )


# ============================================================
# BOOTSTRAP SUMMARY
# ============================================================

names = (
    bin_names
    + cont_names
)

observed = np.array(
    [
        delta_bin["roc_auc"],
        delta_bin["average_precision"],
        delta_bin["log_loss"],
        delta_bin["brier"],
        delta_cont["mae"],
        delta_cont["rmse"],
        delta_cont["correlation"],
    ],
    dtype=np.float64,
)


print()
print("=" * 100)
print(
    "BOOTSTRAP — MB MINUS MA "
    "ON FROZEN 92-LABEL SUBSET"
)
print("=" * 100)


rows = []

higher_better = {
    "roc_auc",
    "average_precision",
    "correlation",
}


for j, name in enumerate(
    names
):

    lo, hi = np.percentile(
        boot[:, j],
        [2.5, 97.5],
    )

    if name in higher_better:
        fraction = float(
            np.mean(
                boot[:, j] > 0
            )
        )
    else:
        fraction = float(
            np.mean(
                boot[:, j] < 0
            )
        )

    print(
        f"{name:<20}"
        f"obs={observed[j]:+.8f} | "
        f"95% CI=["
        f"{lo:+.8f}, "
        f"{hi:+.8f}] | "
        f"improve={fraction:.3f}"
    )

    rows.append({
        "metric":
            name,

        "observed_delta":
            float(observed[j]),

        "ci_low":
            float(lo),

        "ci_high":
            float(hi),

        "improvement_fraction":
            fraction,
    })


# ============================================================
# SAVE
# ============================================================

np.save(
    OUT /
    "MB_minus_MA_bootstrap.npy",
    boot,
)

pd.DataFrame(
    rows
).to_csv(
    OUT /
    "bootstrap_summary.csv",
    index=False,
)


np.save(
    OUT /
    "matched_binary_target.npy",
    binary_test_matrix,
)

np.save(
    OUT /
    "matched_binary_MA_predictions.npy",
    A_bin_matrix,
)

np.save(
    OUT /
    "matched_binary_MB_predictions.npy",
    B_bin_matrix,
)

np.save(
    OUT /
    "matched_continuous_target.npy",
    err_test,
)

np.save(
    OUT /
    "matched_continuous_MA_predictions.npy",
    A_cont_matrix,
)

np.save(
    OUT /
    "matched_continuous_MB_predictions.npy",
    B_cont_matrix,
)


summary = {
    "analysis":
        "final_test_expert_strength_v1",

    "status":
        "pre-specified secondary confirmatory control",

    "matching_population":
        "CAL only",

    "matching_rule": {
        "metric":
            "per-label average precision",

        "absolute_gap_max":
            0.05,

        "minimum_cal_positives":
            5,
    },

    "matched_labels":
        int(len(cols)),

    "labels_rematched_on_test":
        False,

    "primary_conflict":
        "Bernoulli Jensen-Shannon divergence",

    "MA_features": [
        "U_structure",
        "U_sequence",
    ],

    "MB_features": [
        "U_structure",
        "U_sequence",
        "Bernoulli_JS",
    ],

    "cal_pairs":
        int(len(binary_cal)),

    "test_pairs":
        int(len(binary_test)),

    "test_failures":
        int(binary_test.sum()),

    "test_failure_prevalence":
        float(binary_test.mean()),

    "binary": {
        "MA": A_bin,
        "MB": B_bin,
        "delta_MB_minus_MA":
            delta_bin,
    },

    "continuous": {
        "MA": A_cont,
        "MB": B_cont,
        "delta_MB_minus_MA":
            delta_cont,
    },

    "bootstrap_replicates":
        N_BOOT,

    "bootstrap_seed":
        SEED,

    "bootstrap_unit":
        "whole protein",

    "models_refitted_in_bootstrap":
        False,

    "test_used_for_tuning":
        False,
}


with open(
    OUT / "summary.json",
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
print("INTEGRITY")
print("=" * 100)

print("92-label subset frozen from CAL:       YES")
print("Labels re-matched on TEST:             NO")
print("Reliability models fitted on CAL:      YES")
print("Primary conflict frozen before TEST:   YES")
print("Threshold optimization:                NO")
print("Whole-protein bootstrap:               YES")
print("Models refitted in bootstrap:          NO")
print("TEST used for tuning:                  NO")
print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)
