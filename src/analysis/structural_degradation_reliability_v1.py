from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE = Path(
    "data/conflict/go_frequency_v1_inputs"
)

ROB = Path(
    "data/robustness/structural_degradation_v1"
)

OUT = (
    ROB / "analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EPS = 1e-7

CONDITIONS = [
    ("clean", None),
    ("sigma_0.5", "0.5"),
    ("sigma_1.0", "1.0"),
    ("sigma_2.0", "2.0"),
]


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
            f"{len(missing)} target IDs missing."
        )

    rows = [
        index[uid]
        for uid in target_ids
    ]

    return source_probs[
        rows
    ]


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


def continuous_metrics(y, p):

    corr = float(
        np.corrcoef(
            y,
            p,
        )[0, 1]
    )

    return {
        "mae":
            float(
                mean_absolute_error(
                    y,
                    p,
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y,
                        p,
                    )
                )
            ),

        "correlation":
            corr,
    }


# ============================================================
# LOAD CLEAN CAL
# ============================================================

cal_ids = read_ids(
    BASE / "cal_ids.txt"
)

ps_cal = np.load(
    BASE / "p_seq_cal.npy"
).astype(
    np.float64
)

pt_cal = np.load(
    BASE / "p_struct_cal.npy"
).astype(
    np.float64
)

y_cal = np.load(
    BASE / "y_cal.npy"
).astype(
    np.float64
)


# ============================================================
# LOAD CLEAN CAL-EVAL
# ============================================================

eval_ids = read_ids(
    BASE / "eval_ids.txt"
)

ps_eval = np.load(
    BASE / "p_seq_eval.npy"
).astype(
    np.float64
)

pt_eval_clean = np.load(
    BASE / "p_struct_eval.npy"
).astype(
    np.float64
)

y_eval = np.load(
    BASE / "y_eval.npy"
).astype(
    np.float64
)


# ============================================================
# TRAIN FEATURES — CLEAN CAL ONLY
# ============================================================

us_cal = uncertainty(
    ps_cal
)

ut_cal = uncertainty(
    pt_cal
)

js_cal = bernoulli_js(
    ps_cal,
    pt_cal,
)

XA_cal = np.stack(
    [
        ut_cal,
        us_cal,
    ],
    axis=-1,
).reshape(
    -1,
    2,
)

XB_cal = np.column_stack(
    [
        XA_cal,
        js_cal.reshape(-1),
    ]
)


# ============================================================
# TRAIN TARGETS — CLEAN CAL ONLY
# ============================================================

err_cal = np.abs(
    y_cal - pt_cal
)

binary_cal = (
    err_cal >= 0.5
).astype(
    np.uint8
).reshape(-1)

continuous_cal = (
    err_cal.reshape(-1)
)


# ============================================================
# FIT ONCE — CLEAN CAL
# ============================================================

MA_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

MB_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

MA_cont = Ridge(
    alpha=1.0
)

MB_cont = Ridge(
    alpha=1.0
)


MA_bin.fit(
    XA_cal,
    binary_cal,
)

MB_bin.fit(
    XB_cal,
    binary_cal,
)

MA_cont.fit(
    XA_cal,
    continuous_cal,
)

MB_cont.fit(
    XB_cal,
    continuous_cal,
)


print("=" * 96)
print("E6 STRUCTURAL DEGRADATION — FROZEN RELIABILITY ANALYSIS")
print("=" * 96)

print(
    "Training population: clean CAL only"
)

print(
    "CAL proteins:",
    len(cal_ids),
)

print(
    "CAL-EVAL proteins:",
    len(eval_ids),
)

print(
    "CAL structure failures:",
    f"{binary_cal.sum():,}/"
    f"{len(binary_cal):,}",
    f"({binary_cal.mean():.4%})",
)

print()


# ============================================================
# LOAD PERTURBED CAL-EVAL CONDITIONS
# ============================================================

condition_probs = {
    "clean":
        pt_eval_clean
}


for name, sigma in CONDITIONS[1:]:

    d = (
        ROB / f"sigma_{sigma}"
    )

    ids = read_ids(
        d / "uniprot_ids.txt"
    )

    probs = np.load(
        d / "structure_probabilities.npy"
    ).astype(
        np.float64
    )

    condition_probs[
        name
    ] = align_probs(
        ids,
        probs,
        eval_ids,
    )


# ============================================================
# EVALUATE SAME FROZEN MODELS AT EACH CONDITION
# ============================================================

results = []

for name, _ in CONDITIONS:

    pt = condition_probs[
        name
    ]

    us = uncertainty(
        ps_eval
    )

    ut = uncertainty(
        pt
    )

    js = bernoulli_js(
        ps_eval,
        pt,
    )

    XA = np.stack(
        [
            ut,
            us,
        ],
        axis=-1,
    ).reshape(
        -1,
        2,
    )

    XB = np.column_stack(
        [
            XA,
            js.reshape(-1),
        ]
    )

    err = np.abs(
        y_eval - pt
    )

    binary = (
        err >= 0.5
    ).astype(
        np.uint8
    ).reshape(-1)

    continuous = (
        err.reshape(-1)
    )

    pred_MA_bin = (
        MA_bin.predict_proba(
            XA
        )[:, 1]
    )

    pred_MB_bin = (
        MB_bin.predict_proba(
            XB
        )[:, 1]
    )

    pred_MA_cont = (
        MA_cont.predict(
            XA
        )
    )

    pred_MB_cont = (
        MB_cont.predict(
            XB
        )
    )

    ma_bin_metrics = binary_metrics(
        binary,
        pred_MA_bin,
    )

    mb_bin_metrics = binary_metrics(
        binary,
        pred_MB_bin,
    )

    ma_cont_metrics = continuous_metrics(
        continuous,
        pred_MA_cont,
    )

    mb_cont_metrics = continuous_metrics(
        continuous,
        pred_MB_cont,
    )

    result = {
        "condition":
            name,

        "failure_rate":
            float(
                binary.mean()
            ),

        "mean_js":
            float(
                js.mean()
            ),

        "MA_binary":
            ma_bin_metrics,

        "MB_binary":
            mb_bin_metrics,

        "delta_binary": {
            k:
                mb_bin_metrics[k]
                - ma_bin_metrics[k]
            for k in ma_bin_metrics
        },

        "MA_continuous":
            ma_cont_metrics,

        "MB_continuous":
            mb_cont_metrics,

        "delta_continuous": {
            k:
                mb_cont_metrics[k]
                - ma_cont_metrics[k]
            for k in ma_cont_metrics
        },
    }

    results.append(
        result
    )


# ============================================================
# SAVE
# ============================================================

out_path = (
    OUT
    / "structural_degradation_reliability.json"
)

with open(
    out_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print(
    f"{'Condition':<12}"
    f"{'FailRate':>11}"
    f"{'MeanJS':>11}"
    f"{'MA AP':>11}"
    f"{'MB AP':>11}"
    f"{'ΔAP':>11}"
    f"{'ΔLogLoss':>13}"
    f"{'ΔBrier':>11}"
)

print("-" * 92)


for r in results:

    db = r[
        "delta_binary"
    ]

    print(
        f"{r['condition']:<12}"
        f"{r['failure_rate']:>11.6f}"
        f"{r['mean_js']:>11.6f}"
        f"{r['MA_binary']['average_precision']:>11.6f}"
        f"{r['MB_binary']['average_precision']:>11.6f}"
        f"{db['average_precision']:>+11.6f}"
        f"{db['log_loss']:>+13.6f}"
        f"{db['brier']:>+11.6f}"
    )


print()
print("Continuous structure-error reliability")
print("-" * 92)

print(
    f"{'Condition':<12}"
    f"{'ΔMAE':>14}"
    f"{'ΔRMSE':>14}"
    f"{'ΔCorr':>14}"
)

for r in results:

    dc = r[
        "delta_continuous"
    ]

    print(
        f"{r['condition']:<12}"
        f"{dc['mae']:>+14.8f}"
        f"{dc['rmse']:>+14.8f}"
        f"{dc['correlation']:>+14.8f}"
    )


print()
print(
    "Saved:",
    out_path,
)

print()
print(
    "Models fitted once on CLEAN CAL."
)

print(
    "No models refitted on perturbed CAL-EVAL."
)

print(
    "No TEST data used."
)
