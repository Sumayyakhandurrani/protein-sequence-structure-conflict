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

SEED = 42
N_BOOT = 1000
EPS = 1e-7

CAL = Path("data/conflict/go_frequency_v1_inputs")
TEST = Path("data/final_evaluation/final_test_primary_v1")
LABEL_FILE = Path("data/splits/mmseqs30_graph_v2/primary_go_mf_labels.csv")

OUT = Path(
    "data/final_evaluation/"
    "final_test_go_frequency_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

BANDS = [
    ("75-99", 75, 99),
    ("100-249", 100, 249),
    ("250-499", 250, 499),
    ("500-999", 500, 999),
    (">=1000", 1000, np.inf),
]


def uncertainty(p):
    return 1.0 - 2.0 * np.abs(p - 0.5)


def bernoulli_js(ps, pt):
    ps = np.clip(ps, EPS, 1-EPS)
    pt = np.clip(pt, EPS, 1-EPS)

    m = 0.5 * (ps + pt)

    return 0.5 * (
        ps*np.log(ps/m)
        + (1-ps)*np.log((1-ps)/(1-m))
        + pt*np.log(pt/m)
        + (1-pt)*np.log((1-pt)/(1-m))
    )


def metrics(y, p):
    return {
        "ap": float(
            average_precision_score(y, p)
        ),
        "roc_auc": float(
            roc_auc_score(y, p)
        ),
        "log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
    }


# ============================================================
# LOAD CAL
# ============================================================

ps_cal = np.load(
    CAL / "p_seq_cal.npy"
).astype(np.float64)

pt_cal = np.load(
    CAL / "p_struct_cal.npy"
).astype(np.float64)

y_cal = np.load(
    CAL / "y_cal.npy"
).astype(np.float64)


# ============================================================
# LOAD FIXED PAIRED TEST ARTIFACTS
# ============================================================

ps_test = np.load(
    TEST / "paired_sequence_probs.npy"
).astype(np.float64)

pt_test = np.load(
    TEST / "paired_structure_probs.npy"
).astype(np.float64)

y_test = np.load(
    TEST / "paired_test_labels.npy"
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
# TRAIN SUPPORT
# ============================================================

df = pd.read_csv(LABEL_FILE)

required = {
    "label_index",
    "train_protein_count",
}

if not required.issubset(df.columns):
    raise RuntimeError(
        f"Expected columns {required}; "
        f"found {df.columns.tolist()}"
    )

df = df.sort_values(
    "label_index"
).reset_index(drop=True)

if len(df) != 494:
    raise RuntimeError(
        f"Expected 494 labels, found {len(df)}"
    )

if not np.array_equal(
    df["label_index"].to_numpy(),
    np.arange(494),
):
    raise RuntimeError(
        "Label indices are not exactly 0..493."
    )

support = df[
    "train_protein_count"
].to_numpy()


# ============================================================
# FEATURES
# ============================================================

us_cal = uncertainty(ps_cal)
ut_cal = uncertainty(pt_cal)
js_cal = bernoulli_js(ps_cal, pt_cal)

us_test = uncertainty(ps_test)
ut_test = uncertainty(pt_test)
js_test = bernoulli_js(ps_test, pt_test)


print("=" * 100)
print("FINAL TEST V1 — TRAIN-FREQUENCY STRATIFIED RELIABILITY")
print("=" * 100)

print("Frequency definition: TRAIN support only")
print("Primary conflict: Bernoulli JS — frozen before TEST")
print("Reliability fitting population: CAL only")
print("Evaluation population: paired frozen TEST")
print("Bootstrap: 1000 whole-protein replicates")
print("Models refitted during bootstrap: NO")
print("TEST tuning: NO")
print()


rows = []

for band, low, high in BANDS:

    if np.isinf(high):
        mask = support >= low
    else:
        mask = (
            (support >= low)
            & (support <= high)
        )

    n_labels = int(mask.sum())

    if n_labels == 0:
        raise RuntimeError(
            f"No labels in band {band}"
        )

    # --------------------------------------------------------
    # Band-specific CAL and TEST
    # --------------------------------------------------------

    yc = y_cal[:, mask]
    yt = y_test[:, mask]

    psc = ps_cal[:, mask]
    ptc = pt_cal[:, mask]

    pst = ps_test[:, mask]
    ptt = pt_test[:, mask]

    usc = us_cal[:, mask]
    utc = ut_cal[:, mask]
    jsc = js_cal[:, mask]

    ust = us_test[:, mask]
    utt = ut_test[:, mask]
    jst = js_test[:, mask]

    failure_cal = (
        np.abs(yc - ptc) >= 0.5
    ).astype(np.uint8)

    failure_test = (
        np.abs(yt - ptt) >= 0.5
    ).astype(np.uint8)

    XA_cal = np.stack(
        [utc, usc],
        axis=-1,
    ).reshape(-1, 2)

    XB_cal = np.column_stack([
        XA_cal,
        jsc.ravel(),
    ])

    XA_test = np.stack(
        [utt, ust],
        axis=-1,
    ).reshape(-1, 2)

    XB_test = np.column_stack([
        XA_test,
        jst.ravel(),
    ])

    y_cal_flat = failure_cal.ravel()
    y_test_flat = failure_test.ravel()

    if np.unique(y_cal_flat).size != 2:
        raise RuntimeError(
            f"CAL band {band} has only one class."
        )

    if np.unique(y_test_flat).size != 2:
        raise RuntimeError(
            f"TEST band {band} has only one class."
        )

    # --------------------------------------------------------
    # Fit frozen-form models on CAL only
    # --------------------------------------------------------

    ma = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )

    mb = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )

    ma.fit(
        XA_cal,
        y_cal_flat,
    )

    mb.fit(
        XB_cal,
        y_cal_flat,
    )

    pa = ma.predict_proba(
        XA_test
    )[:, 1]

    pb = mb.predict_proba(
        XB_test
    )[:, 1]

    ma_m = metrics(
        y_test_flat,
        pa,
    )

    mb_m = metrics(
        y_test_flat,
        pb,
    )

    delta = {
        k: mb_m[k] - ma_m[k]
        for k in ma_m
    }

    pa_matrix = pa.reshape(
        failure_test.shape
    )

    pb_matrix = pb.reshape(
        failure_test.shape
    )

    # --------------------------------------------------------
    # Whole-protein bootstrap
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    boot = np.empty(
        (N_BOOT, 4),
        dtype=np.float64,
    )

    for b in range(N_BOOT):

        idx = rng.integers(
            0,
            len(y_test),
            size=len(y_test),
        )

        yb = failure_test[idx].ravel()
        pab = pa_matrix[idx].ravel()
        pbb = pb_matrix[idx].ravel()

        boot[b, 0] = (
            average_precision_score(yb, pbb)
            - average_precision_score(yb, pab)
        )

        boot[b, 1] = (
            roc_auc_score(yb, pbb)
            - roc_auc_score(yb, pab)
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
            brier_score_loss(yb, pbb)
            -
            brier_score_loss(yb, pab)
        )

    ap_lo, ap_hi = np.percentile(
        boot[:, 0],
        [2.5, 97.5],
    )

    roc_lo, roc_hi = np.percentile(
        boot[:, 1],
        [2.5, 97.5],
    )

    ll_lo, ll_hi = np.percentile(
        boot[:, 2],
        [2.5, 97.5],
    )

    br_lo, br_hi = np.percentile(
        boot[:, 3],
        [2.5, 97.5],
    )

    ap_fraction = float(
        np.mean(boot[:, 0] > 0)
    )

    print("=" * 100)
    print(f"TRAIN support band: {band}")
    print("=" * 100)

    print(f"Labels:            {n_labels}")
    print(
        f"TEST failures:     "
        f"{int(failure_test.sum()):,}/"
        f"{failure_test.size:,} "
        f"({failure_test.mean():.6%})"
    )

    print(
        f"AP:                "
        f"MA={ma_m['ap']:.8f} "
        f"MB={mb_m['ap']:.8f} "
        f"delta={delta['ap']:+.8f}"
    )

    print(
        f"Delta AP 95% CI:   "
        f"[{ap_lo:+.8f}, {ap_hi:+.8f}]"
    )

    print(
        f"AP improve frac:   "
        f"{ap_fraction:.3f}"
    )

    print(
        f"Delta ROC:         "
        f"{delta['roc_auc']:+.8f} "
        f"CI=[{roc_lo:+.8f}, {roc_hi:+.8f}]"
    )

    print(
        f"Delta log-loss:    "
        f"{delta['log_loss']:+.8f} "
        f"CI=[{ll_lo:+.8f}, {ll_hi:+.8f}]"
    )

    print(
        f"Delta Brier:       "
        f"{delta['brier']:+.8f} "
        f"CI=[{br_lo:+.8f}, {br_hi:+.8f}]"
    )

    print()

    safe_band = band.replace(
        ">=", "ge_"
    )

    np.save(
        OUT / f"bootstrap_{safe_band}.npy",
        boot,
    )

    rows.append({
        "band": band,
        "n_labels": n_labels,
        "n_test_proteins":
            int(len(y_test)),
        "n_pairs":
            int(failure_test.size),
        "n_failures":
            int(failure_test.sum()),
        "failure_prevalence":
            float(failure_test.mean()),

        "ap_MA": ma_m["ap"],
        "ap_MB": mb_m["ap"],
        "delta_ap": delta["ap"],
        "delta_ap_ci_low":
            float(ap_lo),
        "delta_ap_ci_high":
            float(ap_hi),
        "ap_improvement_fraction":
            ap_fraction,

        "delta_roc_auc":
            delta["roc_auc"],
        "delta_roc_auc_ci_low":
            float(roc_lo),
        "delta_roc_auc_ci_high":
            float(roc_hi),

        "delta_log_loss":
            delta["log_loss"],
        "delta_log_loss_ci_low":
            float(ll_lo),
        "delta_log_loss_ci_high":
            float(ll_hi),

        "delta_brier":
            delta["brier"],
        "delta_brier_ci_low":
            float(br_lo),
        "delta_brier_ci_high":
            float(br_hi),
    })


summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "go_frequency_test_summary.csv",
    index=False,
)


manifest = {
    "analysis":
        "final_test_go_frequency_v1",

    "status":
        "pre-specified secondary confirmatory control",

    "frequency_definition":
        "TRAIN protein support",

    "bands": [
        "75-99",
        "100-249",
        "250-499",
        "500-999",
        ">=1000",
    ],

    "primary_conflict":
        "Bernoulli Jensen-Shannon divergence",

    "note":
        (
            "The original development frequency analysis "
            "used absolute disagreement. The final TEST "
            "protocol uses Bernoulli JS because JS was "
            "selected and frozen as the primary conflict "
            "formulation before TEST was opened."
        ),

    "MA_features": [
        "U_structure",
        "U_sequence",
    ],

    "MB_features": [
        "U_structure",
        "U_sequence",
        "Bernoulli_JS",
    ],

    "endpoint":
        "|p_structure-y| >= 0.5",

    "fit_population":
        "CAL",

    "evaluation_population":
        "paired frozen TEST",

    "bootstrap_replicates":
        N_BOOT,

    "bootstrap_seed":
        SEED,

    "resampling_unit":
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


print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    summary[
        [
            "band",
            "n_labels",
            "delta_ap",
            "delta_ap_ci_low",
            "delta_ap_ci_high",
            "ap_improvement_fraction",
        ]
    ].to_string(index=False)
)

print()
print("TEST tuning:                 NO")
print("Frequency bands changed:     NO")
print("Conflict re-selected:        NO")
print("Reliability models fit on:   CAL ONLY")
print("Bootstrap unit:              WHOLE PROTEIN")
print("Models refit in bootstrap:   NO")
print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)
