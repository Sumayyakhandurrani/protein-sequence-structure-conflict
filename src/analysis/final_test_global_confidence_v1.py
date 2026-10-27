from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

SEED = 42
N_BOOT = 1000
EPS = 1e-7

CAL_DIR = Path(
    "data/conflict/go_frequency_v1_inputs"
)

CAL_SPLIT_DIR = Path(
    "data/splits/mmseqs30_graph_v2/"
    "structure_calibration_v1"
)

TEST_DIR = Path(
    "data/final_evaluation/final_test_primary_v1"
)

AF_PATH = Path(
    "data/processed/alphafold_availability.csv"
)

OUT = Path(
    "data/final_evaluation/"
    "final_test_global_confidence_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# UTILITIES
# ============================================================

def read_ids(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


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


def structure_failure(
    y,
    p_struct,
):
    return (
        np.abs(
            y - p_struct
        ) >= 0.5
    ).astype(np.uint8)


def fit_model(X, y):

    if np.unique(y).size != 2:
        raise RuntimeError(
            "CAL target does not "
            "contain both classes."
        )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )

    model.fit(X, y)

    return model


def metrics(y, p):

    return {
        "ap": float(
            average_precision_score(
                y,
                p,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y,
                p,
            )
        ),

        "log_loss": float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),

        "brier": float(
            brier_score_loss(
                y,
                p,
            )
        ),
    }


# ============================================================
# LOAD CAL — FITTING DATA ONLY
# ============================================================

Y_cal = np.load(
    CAL_DIR / "y_cal.npy"
).astype(np.float64)

P_seq_cal = np.load(
    CAL_DIR / "p_seq_cal.npy"
).astype(np.float64)

P_struct_cal = np.load(
    CAL_DIR / "p_struct_cal.npy"
).astype(np.float64)

cal_ids = read_ids(
    CAL_SPLIT_DIR /
    "structure_calibration_ids.txt"
)


# ============================================================
# LOAD FROZEN PAIRED TEST
# ============================================================

Y_test = np.load(
    TEST_DIR /
    "paired_test_labels.npy"
).astype(np.float64)

P_seq_test = np.load(
    TEST_DIR /
    "paired_sequence_probs.npy"
).astype(np.float64)

P_struct_test = np.load(
    TEST_DIR /
    "paired_structure_probs.npy"
).astype(np.float64)

test_ids = read_ids(
    TEST_DIR /
    "paired_test_uniprot_ids.txt"
)


# ============================================================
# INTEGRITY
# ============================================================

assert (
    Y_cal.shape
    == P_seq_cal.shape
    == P_struct_cal.shape
    == (2369, 494)
)

assert (
    Y_test.shape
    == P_seq_test.shape
    == P_struct_test.shape
    == (4731, 494)
)

assert len(cal_ids) == 2369
assert len(test_ids) == 4731

assert len(set(cal_ids)) == 2369
assert len(set(test_ids)) == 4731

assert not (
    set(cal_ids)
    & set(test_ids)
)

for name, arr in {
    "P_seq_cal": P_seq_cal,
    "P_struct_cal": P_struct_cal,
    "P_seq_test": P_seq_test,
    "P_struct_test": P_struct_test,
}.items():

    assert np.isfinite(arr).all(), name
    assert arr.min() >= 0
    assert arr.max() <= 1


# ============================================================
# LOAD ALPHAFOLD GLOBAL CONFIDENCE
# ============================================================

af = pd.read_csv(
    AF_PATH
)

required = {
    "uniprot_id",
    "global_metric",
}

if not required.issubset(
    af.columns
):
    raise RuntimeError(
        "AlphaFold table missing "
        "required columns."
    )

if af[
    "uniprot_id"
].duplicated().any():
    raise RuntimeError(
        "Duplicate UniProt IDs "
        "in AlphaFold table."
    )

af = af.set_index(
    "uniprot_id"
)


def extract_q(ids, name):

    missing = [
        uid
        for uid in ids
        if uid not in af.index
    ]

    if missing:
        raise RuntimeError(
            f"{name}: {len(missing)} "
            "IDs absent from AlphaFold table."
        )

    q = (
        af.loc[
            ids,
            "global_metric",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(q).all():

        bad = int(
            (~np.isfinite(q)).sum()
        )

        raise RuntimeError(
            f"{name}: {bad} non-finite "
            "global_metric values."
        )

    return q


Q_cal_raw = extract_q(
    cal_ids,
    "CAL",
)

Q_test_raw = extract_q(
    test_ids,
    "TEST",
)


# Same numerical scaling as development.
Q_cal = (
    Q_cal_raw / 100.0
)

Q_test = (
    Q_test_raw / 100.0
)


# ============================================================
# TARGET
# ============================================================

failure_cal = structure_failure(
    Y_cal,
    P_struct_cal,
)

failure_test = structure_failure(
    Y_test,
    P_struct_test,
)


# ============================================================
# FROZEN FEATURES
# ============================================================

U_struct_cal = uncertainty(
    P_struct_cal
)

U_seq_cal = uncertainty(
    P_seq_cal
)

JS_cal = bernoulli_js(
    P_seq_cal,
    P_struct_cal,
)


U_struct_test = uncertainty(
    P_struct_test
)

U_seq_test = uncertainty(
    P_seq_test
)

JS_test = bernoulli_js(
    P_seq_test,
    P_struct_test,
)


Q_cal_matrix = np.repeat(
    Q_cal[:, None],
    494,
    axis=1,
)

Q_test_matrix = np.repeat(
    Q_test[:, None],
    494,
    axis=1,
)


# ============================================================
# NESTED MODELS
#
# M0 = U_struct + U_seq
#
# MA = M0 + AlphaFold global confidence
#
# MB = MA + frozen primary Bernoulli JS conflict
# ============================================================

X0_cal = np.column_stack([
    U_struct_cal.ravel(),
    U_seq_cal.ravel(),
])

XA_cal = np.column_stack([
    U_struct_cal.ravel(),
    U_seq_cal.ravel(),
    Q_cal_matrix.ravel(),
])

XB_cal = np.column_stack([
    U_struct_cal.ravel(),
    U_seq_cal.ravel(),
    Q_cal_matrix.ravel(),
    JS_cal.ravel(),
])


X0_test = np.column_stack([
    U_struct_test.ravel(),
    U_seq_test.ravel(),
])

XA_test = np.column_stack([
    U_struct_test.ravel(),
    U_seq_test.ravel(),
    Q_test_matrix.ravel(),
])

XB_test = np.column_stack([
    U_struct_test.ravel(),
    U_seq_test.ravel(),
    Q_test_matrix.ravel(),
    JS_test.ravel(),
])


y_cal = failure_cal.ravel()
y_test = failure_test.ravel()


# ============================================================
# FIT ON CAL ONLY
# ============================================================

model_M0 = fit_model(
    X0_cal,
    y_cal,
)

model_MA = fit_model(
    XA_cal,
    y_cal,
)

model_MB = fit_model(
    XB_cal,
    y_cal,
)


# ============================================================
# EVALUATE ON LOCKED TEST
# ============================================================

pred_M0 = model_M0.predict_proba(
    X0_test
)[:, 1]

pred_MA = model_MA.predict_proba(
    XA_test
)[:, 1]

pred_MB = model_MB.predict_proba(
    XB_test
)[:, 1]


m0 = metrics(
    y_test,
    pred_M0,
)

ma = metrics(
    y_test,
    pred_MA,
)

mb = metrics(
    y_test,
    pred_MB,
)


def delta(a, b):
    return {
        k: b[k] - a[k]
        for k in [
            "ap",
            "roc_auc",
            "log_loss",
            "brier",
        ]
    }


delta_Q = delta(
    m0,
    ma,
)

delta_JS = delta(
    ma,
    mb,
)


# ============================================================
# REPORT POINT RESULTS
# ============================================================

print("=" * 100)
print("FINAL TEST V1 — GLOBAL ALPHAFOLD-CONFIDENCE CONTROL")
print("=" * 100)

print("TEST tuning:                    NO")
print("Reliability models fitted on:   CAL ONLY")
print("Primary conflict:               Bernoulli JS")
print("Global confidence input:        global_metric / 100")
print()

print(f"CAL proteins:                   {len(cal_ids):,}")
print(f"Paired TEST proteins:           {len(test_ids):,}")
print(f"Labels:                         {Y_test.shape[1]}")
print()

print("AlphaFold global confidence")
print(
    f"CAL:  mean={Q_cal_raw.mean():.4f}, "
    f"median={np.median(Q_cal_raw):.4f}, "
    f"range=[{Q_cal_raw.min():.2f}, "
    f"{Q_cal_raw.max():.2f}]"
)

print(
    f"TEST: mean={Q_test_raw.mean():.4f}, "
    f"median={np.median(Q_test_raw):.4f}, "
    f"range=[{Q_test_raw.min():.2f}, "
    f"{Q_test_raw.max():.2f}]"
)

print()

print(
    f"TEST structure failures: "
    f"{int(failure_test.sum()):,}/"
    f"{failure_test.size:,} "
    f"({failure_test.mean():.6%})"
)

print()

print("=" * 100)
print("TEST MODEL COMPARISON")
print("=" * 100)

for name, result in [
    ("M0 uncertainty", m0),
    ("MA + global confidence", ma),
    ("MB + Bernoulli JS", mb),
]:

    print(
        f"{name:<25}"
        f"AP={result['ap']:.8f}  "
        f"ROC={result['roc_auc']:.8f}  "
        f"LL={result['log_loss']:.8f}  "
        f"Brier={result['brier']:.8f}"
    )


print()
print(
    "Effect of AlphaFold global confidence: "
    "MA - M0"
)

for k, v in delta_Q.items():
    print(
        f"  {k:<12}{v:+.8f}"
    )


print()
print(
    "PRIMARY CONTROL: "
    "effect of JS after global confidence: "
    "MB - MA"
)

for k, v in delta_JS.items():
    print(
        f"  {k:<12}{v:+.8f}"
    )


# ============================================================
# WHOLE-PROTEIN BOOTSTRAP — PRIMARY MB vs MA
# ============================================================

pred_MA_matrix = pred_MA.reshape(
    failure_test.shape
)

pred_MB_matrix = pred_MB.reshape(
    failure_test.shape
)

rng = np.random.default_rng(
    SEED
)

boot = np.empty(
    (N_BOOT, 4),
    dtype=np.float64,
)

print()
print(
    f"Running {N_BOOT} whole-protein "
    "bootstrap replicates..."
)


for b in range(N_BOOT):

    idx = rng.integers(
        0,
        len(test_ids),
        size=len(test_ids),
    )

    yb = failure_test[
        idx
    ].ravel()

    pa = pred_MA_matrix[
        idx
    ].ravel()

    pb = pred_MB_matrix[
        idx
    ].ravel()

    boot[b, 0] = (
        average_precision_score(
            yb,
            pb,
        )
        -
        average_precision_score(
            yb,
            pa,
        )
    )

    boot[b, 1] = (
        roc_auc_score(
            yb,
            pb,
        )
        -
        roc_auc_score(
            yb,
            pa,
        )
    )

    boot[b, 2] = (
        log_loss(
            yb,
            pb,
            labels=[0, 1],
        )
        -
        log_loss(
            yb,
            pa,
            labels=[0, 1],
        )
    )

    boot[b, 3] = (
        brier_score_loss(
            yb,
            pb,
        )
        -
        brier_score_loss(
            yb,
            pa,
        )
    )

    if (
        (b + 1) % 100
        == 0
    ):
        print(
            f"Completed "
            f"{b+1}/{N_BOOT}"
        )


names = [
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier",
]

observed = [
    delta_JS["ap"],
    delta_JS["roc_auc"],
    delta_JS["log_loss"],
    delta_JS["brier"],
]


print()
print("=" * 100)
print(
    "BOOTSTRAP — "
    "JS AFTER GLOBAL CONFIDENCE "
    "(MB - MA)"
)
print("=" * 100)


rows = []

for j, name in enumerate(
    names
):

    lo, hi = np.percentile(
        boot[:, j],
        [2.5, 97.5],
    )

    if name in {
        "average_precision",
        "roc_auc",
    }:
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
        "metric": name,
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

comparison = pd.DataFrame([
    {
        "model": "M0",
        "features":
            "U_struct + U_seq",
        **m0,
    },
    {
        "model": "MA",
        "features":
            "U_struct + U_seq + global_confidence",
        **ma,
    },
    {
        "model": "MB",
        "features":
            "U_struct + U_seq + global_confidence + Bernoulli_JS",
        **mb,
    },
])

comparison.to_csv(
    OUT / "model_comparison.csv",
    index=False,
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
    "MB_minus_MA_bootstrap.npy",
    boot,
)


manifest = {
    "analysis":
        "final_test_global_confidence_v1",

    "status":
        "pre-specified secondary confirmatory control",

    "fit_population":
        "CAL only",

    "evaluation_population":
        "4731 paired frozen TEST proteins",

    "global_confidence":
        "AlphaFold global_metric / 100",

    "global_confidence_interpretation":
        (
            "AlphaFold global confidence; "
            "not treated as true structural accuracy."
        ),

    "M0":
        [
            "U_structure",
            "U_sequence",
        ],

    "MA":
        [
            "U_structure",
            "U_sequence",
            "global_confidence",
        ],

    "MB":
        [
            "U_structure",
            "U_sequence",
            "global_confidence",
            "Bernoulli_JS",
        ],

    "primary_comparison":
        "MB minus MA",

    "primary_metric":
        "average precision",

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
    OUT / "manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


print()
print("=" * 100)
print("INTEGRITY")
print("=" * 100)

print("TEST tuning:                  NO")
print("Models fitted on TEST:        NO")
print("Global confidence selected:   NO — pre-specified control")
print("Conflict re-selected:         NO")
print("Primary conflict:             Bernoulli JS")
print("Bootstrap unit:               whole protein")
print("Models refitted in bootstrap: NO")
print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)
