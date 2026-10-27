from pathlib import Path
import json

import numpy as np

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

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

INPUT = Path(
    "data/conflict/go_frequency_v1_inputs"
)

ROB = Path(
    "data/robustness/sequence_degradation_v1"
)

OUT = ROB / "reliability_analysis"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EPS = 1e-7

CONDITIONS = [
    "clean",
    "mask_10pct",
    "mask_20pct",
    "mask_40pct",
]


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def uncertainty(p):
    return (
        1.0
        - 2.0 * np.abs(
            p - 0.5
        )
    )


def bernoulli_js(ps, pt):

    ps = np.clip(
        ps,
        EPS,
        1.0 - EPS,
    )

    pt = np.clip(
        pt,
        EPS,
        1.0 - EPS,
    )

    m1 = 0.5 * (
        ps + pt
    )

    m0 = 1.0 - m1

    kl_s = (
        ps * np.log(
            ps / m1
        )
        +
        (1.0 - ps)
        * np.log(
            (1.0 - ps) / m0
        )
    )

    kl_t = (
        pt * np.log(
            pt / m1
        )
        +
        (1.0 - pt)
        * np.log(
            (1.0 - pt) / m0
        )
    )

    return 0.5 * (
        kl_s + kl_t
    )


def binary_metrics(y, p):

    return {
        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    y,
                    p,
                )
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
                brier_score_loss(
                    y,
                    p,
                )
            ),
    }


def continuous_metrics(y, pred):

    return {
        "mae":
            float(
                mean_absolute_error(
                    y,
                    pred,
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y,
                        pred,
                    )
                )
            ),

        "correlation":
            float(
                np.corrcoef(
                    y,
                    pred,
                )[0, 1]
            ),
    }


def align_probs(
    source_ids,
    source_probs,
    target_ids,
):

    index = {
        uid: i
        for i, uid
        in enumerate(source_ids)
    }

    missing = [
        uid
        for uid in target_ids
        if uid not in index
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} IDs missing."
        )

    rows = [
        index[uid]
        for uid in target_ids
    ]

    return source_probs[rows]


# ============================================================
# LOAD CLEAN CAL
# ============================================================

cal_ids = read_ids(
    INPUT / "cal_ids.txt"
)

y_cal = np.load(
    INPUT / "y_cal.npy"
).astype(np.float64)

p_seq_cal = np.load(
    INPUT / "p_seq_cal.npy"
).astype(np.float64)

p_struct_cal = np.load(
    INPUT / "p_struct_cal.npy"
).astype(np.float64)


if not (
    y_cal.shape
    == p_seq_cal.shape
    == p_struct_cal.shape
):
    raise RuntimeError(
        "CAL shapes do not match."
    )


# ============================================================
# LOAD CAL-EVAL TARGET / CLEAN STRUCTURE
# ============================================================

eval_ids = read_ids(
    INPUT / "eval_ids.txt"
)

y_eval = np.load(
    INPUT / "y_eval.npy"
).astype(np.float64)

p_struct_eval = np.load(
    INPUT / "p_struct_eval.npy"
).astype(np.float64)


if y_eval.shape != p_struct_eval.shape:
    raise RuntimeError(
        "CAL-EVAL shapes do not match."
    )


# ============================================================
# BUILD CLEAN CAL FEATURES
#
# MA = [U_sequence, U_structure]
#
# MB = [U_sequence, U_structure, Bernoulli_JS]
#
# Sequence endpoint:
# |y - p_sequence| >= 0.5
#
# Models are fitted ONCE here and NEVER refitted
# for masked CAL-EVAL conditions.
# ============================================================

U_seq_cal = uncertainty(
    p_seq_cal
)

U_struct_cal = uncertainty(
    p_struct_cal
)

JS_cal = bernoulli_js(
    p_seq_cal,
    p_struct_cal,
)


X_MA_cal = np.column_stack([
    U_seq_cal.reshape(-1),
    U_struct_cal.reshape(-1),
])

X_MB_cal = np.column_stack([
    U_seq_cal.reshape(-1),
    U_struct_cal.reshape(-1),
    JS_cal.reshape(-1),
])


seq_abs_cal = np.abs(
    y_cal - p_seq_cal
).reshape(-1)

seq_failure_cal = (
    seq_abs_cal >= 0.5
).astype(np.uint8)


# ============================================================
# FIT FROZEN CLEAN-CAL MODELS
# ============================================================

binary_MA = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)

binary_MB = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)

continuous_MA = Ridge(
    alpha=1.0
)

continuous_MB = Ridge(
    alpha=1.0
)


binary_MA.fit(
    X_MA_cal,
    seq_failure_cal,
)

binary_MB.fit(
    X_MB_cal,
    seq_failure_cal,
)

continuous_MA.fit(
    X_MA_cal,
    seq_abs_cal,
)

continuous_MB.fit(
    X_MB_cal,
    seq_abs_cal,
)


# ============================================================
# HEADER
# ============================================================

print("=" * 104)
print(
    "E7 SEQUENCE DEGRADATION — "
    "FROZEN CLEAN-CAL RELIABILITY"
)
print("=" * 104)

print(
    "CAL proteins:",
    len(cal_ids),
)

print(
    "CAL-EVAL proteins:",
    len(eval_ids),
)

print(
    "Labels:",
    y_eval.shape[1],
)

print(
    "CAL sequence failures:",
    int(seq_failure_cal.sum()),
    f"({seq_failure_cal.mean():.6%})",
)

print(
    "MA: [U_sequence, U_structure]"
)

print(
    "MB: [U_sequence, U_structure, Bernoulli_JS]"
)

print(
    "Fit: clean CAL only"
)

print(
    "Masked-condition refitting: NO"
)

print(
    "TEST data used: NO"
)

print()


# ============================================================
# EVALUATE CONDITIONS
# ============================================================

results = []

for condition in CONDITIONS:

    d = ROB / condition

    condition_ids = read_ids(
        d / "uniprot_ids.txt"
    )

    p_seq = np.load(
        d
        / "calibrated_probabilities.npy"
    ).astype(np.float64)

    p_seq = align_probs(
        condition_ids,
        p_seq,
        eval_ids,
    )


    if p_seq.shape != y_eval.shape:
        raise RuntimeError(
            f"{condition}: shape mismatch."
        )


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    U_seq = uncertainty(
        p_seq
    )

    U_struct = uncertainty(
        p_struct_eval
    )

    JS = bernoulli_js(
        p_seq,
        p_struct_eval,
    )


    X_MA = np.column_stack([
        U_seq.reshape(-1),
        U_struct.reshape(-1),
    ])

    X_MB = np.column_stack([
        U_seq.reshape(-1),
        U_struct.reshape(-1),
        JS.reshape(-1),
    ])


    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    seq_abs = np.abs(
        y_eval - p_seq
    ).reshape(-1)

    seq_failure = (
        seq_abs >= 0.5
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Binary predictions
    # --------------------------------------------------------

    pred_MA = (
        binary_MA
        .predict_proba(
            X_MA
        )[:, 1]
    )

    pred_MB = (
        binary_MB
        .predict_proba(
            X_MB
        )[:, 1]
    )


    bm_MA = binary_metrics(
        seq_failure,
        pred_MA,
    )

    bm_MB = binary_metrics(
        seq_failure,
        pred_MB,
    )


    # --------------------------------------------------------
    # Continuous predictions
    # --------------------------------------------------------

    cont_pred_MA = (
        continuous_MA.predict(
            X_MA
        )
    )

    cont_pred_MB = (
        continuous_MB.predict(
            X_MB
        )
    )


    cm_MA = continuous_metrics(
        seq_abs,
        cont_pred_MA,
    )

    cm_MB = continuous_metrics(
        seq_abs,
        cont_pred_MB,
    )


    # --------------------------------------------------------
    # Save condition predictions
    # --------------------------------------------------------

    np.save(
        OUT
        / f"{condition}_binary_MA.npy",
        pred_MA.astype(np.float32),
    )

    np.save(
        OUT
        / f"{condition}_binary_MB.npy",
        pred_MB.astype(np.float32),
    )

    np.save(
        OUT
        / f"{condition}_binary_target.npy",
        seq_failure,
    )

    np.save(
        OUT
        / f"{condition}_continuous_MA.npy",
        cont_pred_MA.astype(np.float32),
    )

    np.save(
        OUT
        / f"{condition}_continuous_MB.npy",
        cont_pred_MB.astype(np.float32),
    )

    np.save(
        OUT
        / f"{condition}_continuous_target.npy",
        seq_abs.astype(np.float32),
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {
        "condition":
            condition,

        "failure_events":
            int(
                seq_failure.sum()
            ),

        "failure_rate":
            float(
                seq_failure.mean()
            ),

        "mean_bernoulli_js":
            float(
                JS.mean()
            ),

        "binary_MA":
            bm_MA,

        "binary_MB":
            bm_MB,

        "binary_delta_MB_minus_MA": {
            key:
                bm_MB[key]
                - bm_MA[key]
            for key in bm_MA
        },

        "continuous_MA":
            cm_MA,

        "continuous_MB":
            cm_MB,

        "continuous_delta_MB_minus_MA": {
            key:
                cm_MB[key]
                - cm_MA[key]
            for key in cm_MA
        },
    }

    results.append(
        result
    )


# ============================================================
# SAVE JSON
# ============================================================

summary = {
    "endpoint":
        "sequence_expert_error",

    "binary_failure_definition":
        "|y - p_sequence| >= 0.5",

    "conflict":
        "per-label Bernoulli Jensen-Shannon divergence, natural log",

    "MA":
        [
            "U_sequence",
            "U_structure",
        ],

    "MB":
        [
            "U_sequence",
            "U_structure",
            "Bernoulli_JS",
        ],

    "fit_population":
        "clean CAL only",

    "evaluation_population":
        "CAL-EVAL under E7 sequence masking",

    "masked_condition_refitting":
        False,

    "test_used":
        False,

    "results":
        results,
}


with open(
    OUT / "reliability_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# PRINT
# ============================================================

print(
    f"{'Condition':<14}"
    f"{'FailRate':>11}"
    f"{'MeanJS':>11}"
    f"{'MA_AP':>11}"
    f"{'MB_AP':>11}"
    f"{'ΔAP':>11}"
    f"{'ΔROC':>11}"
    f"{'ΔLL':>11}"
    f"{'ΔBrier':>11}"
)

print("-" * 102)

for r in results:

    db = r[
        "binary_delta_MB_minus_MA"
    ]

    print(
        f"{r['condition']:<14}"
        f"{r['failure_rate']:>11.6f}"
        f"{r['mean_bernoulli_js']:>11.6f}"
        f"{r['binary_MA']['average_precision']:>11.6f}"
        f"{r['binary_MB']['average_precision']:>11.6f}"
        f"{db['average_precision']:>+11.6f}"
        f"{db['roc_auc']:>+11.6f}"
        f"{db['log_loss']:>+11.6f}"
        f"{db['brier']:>+11.6f}"
    )


print()
print(
    "Continuous absolute-error prediction"
)
print("-" * 90)

print(
    f"{'Condition':<14}"
    f"{'ΔMAE':>14}"
    f"{'ΔRMSE':>14}"
    f"{'ΔCorrelation':>16}"
)

for r in results:

    dc = r[
        "continuous_delta_MB_minus_MA"
    ]

    print(
        f"{r['condition']:<14}"
        f"{dc['mae']:>+14.8f}"
        f"{dc['rmse']:>+14.8f}"
        f"{dc['correlation']:>+16.8f}"
    )


print()
print(
    "Saved:",
    OUT / "reliability_summary.json",
)

print(
    "Masked-condition model refitting: NO"
)

print(
    "TEST data used: NO"
)

print("=" * 104)
print(
    "E7 RELIABILITY ANALYSIS COMPLETE"
)
print("=" * 104)
