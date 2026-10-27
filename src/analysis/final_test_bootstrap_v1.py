from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

BASE = Path(
    "data/final_evaluation/final_test_primary_v1"
)

OUT = BASE / "bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
SEED = 42
EPS = 1e-7


# ============================================================
# LOAD FIXED TEST TARGETS/PREDICTIONS
# ============================================================

y = np.load(
    BASE / "paired_test_labels.npy"
).astype(np.float64)

y_bin = np.load(
    BASE / "structure_failure_target.npy"
).astype(np.uint8)

y_cont = np.load(
    BASE / "structure_absolute_error.npy"
).astype(np.float64)

ma_bin = np.load(
    BASE / "MA_binary_predictions.npy"
).astype(np.float64)

mb_bin = np.load(
    BASE / "MB_binary_predictions.npy"
).astype(np.float64)

ma_cont = np.load(
    BASE / "MA_continuous_predictions.npy"
).astype(np.float64)

mb_cont = np.load(
    BASE / "MB_continuous_predictions.npy"
).astype(np.float64)

a1 = np.load(
    BASE / "A1_predictions.npy"
).astype(np.float64)

a2 = np.load(
    BASE / "A2_predictions.npy"
).astype(np.float64)


expected = (4731, 494)

for name, arr in {
    "y": y,
    "y_bin": y_bin,
    "y_cont": y_cont,
    "ma_bin": ma_bin,
    "mb_bin": mb_bin,
    "ma_cont": ma_cont,
    "mb_cont": mb_cont,
    "a1": a1,
    "a2": a2,
}.items():

    if arr.shape != expected:
        raise RuntimeError(
            f"{name} shape mismatch: {arr.shape}"
        )


n_proteins, n_labels = y.shape


print("=" * 100)
print("FINAL TEST V1 — PRE-REGISTERED WHOLE-PROTEIN BOOTSTRAP")
print("=" * 100)

print(f"Proteins:             {n_proteins:,}")
print(f"Labels:               {n_labels}")
print(f"Bootstrap replicates: {N_BOOT:,}")
print(f"Seed:                 {SEED}")
print("Resampling unit:      whole protein")
print("Models refitted:      NO")
print("TEST tuning:          NO")
print()


# ============================================================
# RELIABILITY METRICS
# ============================================================

def reliability_delta(idx):

    yb = y_bin[idx].ravel()
    yc = y_cont[idx].ravel()

    pa_bin = ma_bin[idx].ravel()
    pb_bin = mb_bin[idx].ravel()

    pa_cont = ma_cont[idx].ravel()
    pb_cont = mb_cont[idx].ravel()

    return np.array([
        roc_auc_score(yb, pb_bin)
        - roc_auc_score(yb, pa_bin),

        average_precision_score(yb, pb_bin)
        - average_precision_score(yb, pa_bin),

        log_loss(
            yb,
            np.clip(pb_bin, EPS, 1-EPS),
            labels=[0, 1],
        )
        -
        log_loss(
            yb,
            np.clip(pa_bin, EPS, 1-EPS),
            labels=[0, 1],
        ),

        brier_score_loss(yb, pb_bin)
        - brier_score_loss(yb, pa_bin),

        mean_absolute_error(yc, pb_cont)
        - mean_absolute_error(yc, pa_cont),

        np.sqrt(
            mean_squared_error(yc, pb_cont)
        )
        -
        np.sqrt(
            mean_squared_error(yc, pa_cont)
        ),

        np.corrcoef(yc, pb_cont)[0, 1]
        -
        np.corrcoef(yc, pa_cont)[0, 1],
    ])


REL_NAMES = [
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
    "mae",
    "rmse",
    "correlation",
]


# ============================================================
# ARBITRATION METRICS
# ============================================================

def macro_ap_supported(yb, pb):

    support = (
        yb.sum(axis=0) > 0
    )

    vals = []

    for j in np.where(support)[0]:
        vals.append(
            average_precision_score(
                yb[:, j],
                pb[:, j],
            )
        )

    return float(
        np.mean(vals)
    )


def arbitration_metrics(yb, pb):

    yf = yb.ravel()
    pf = pb.ravel()

    return np.array([
        average_precision_score(
            yf,
            pf,
        ),

        macro_ap_supported(
            yb,
            pb,
        ),

        np.mean(
            np.abs(pb - yb)
        ),

        np.mean(
            (pb - yb) ** 2
        ),

        log_loss(
            yf,
            np.clip(
                pf,
                EPS,
                1-EPS,
            ),
            labels=[0, 1],
        ),
    ])


ARB_NAMES = [
    "micro_ap",
    "macro_ap",
    "mae",
    "brier",
    "nll",
]


def arbitration_delta(idx):

    yb = y[idx]

    m1 = arbitration_metrics(
        yb,
        a1[idx],
    )

    m2 = arbitration_metrics(
        yb,
        a2[idx],
    )

    return m2 - m1


# ============================================================
# OBSERVED DELTAS
# ============================================================

all_idx = np.arange(
    n_proteins
)

observed_rel = reliability_delta(
    all_idx
)

observed_arb = arbitration_delta(
    all_idx
)


print("Observed primary reliability MB - MA:")

for name, value in zip(
    REL_NAMES,
    observed_rel,
):
    print(
        f"  {name:<20}"
        f"{value:+.8f}"
    )


print()
print("Observed primary arbitration A2 - A1:")

for name, value in zip(
    ARB_NAMES,
    observed_arb,
):
    print(
        f"  {name:<20}"
        f"{value:+.8f}"
    )

print()


# ============================================================
# WHOLE-PROTEIN BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    SEED
)

rel_boot = np.empty(
    (
        N_BOOT,
        len(REL_NAMES),
    ),
    dtype=np.float64,
)

arb_boot = np.empty(
    (
        N_BOOT,
        len(ARB_NAMES),
    ),
    dtype=np.float64,
)


for b in range(N_BOOT):

    idx = rng.integers(
        0,
        n_proteins,
        size=n_proteins,
    )

    rel_boot[b] = (
        reliability_delta(idx)
    )

    arb_boot[b] = (
        arbitration_delta(idx)
    )

    if (b + 1) % 100 == 0:
        print(
            f"Completed "
            f"{b + 1}/{N_BOOT}"
        )


# ============================================================
# SUMMARIZE
# ============================================================

def summarize(
    values,
    observed,
    names,
    higher_better,
):

    rows = []

    for j, name in enumerate(
        names
    ):

        vals = values[:, j]

        lo, hi = np.percentile(
            vals,
            [2.5, 97.5],
        )

        if name in higher_better:
            fraction = np.mean(
                vals > 0
            )
        else:
            fraction = np.mean(
                vals < 0
            )

        rows.append({
            "metric":
                name,

            "observed_delta":
                float(observed[j]),

            "bootstrap_mean":
                float(vals.mean()),

            "ci_2.5":
                float(lo),

            "ci_97.5":
                float(hi),

            "improvement_fraction":
                float(fraction),
        })

    return pd.DataFrame(
        rows
    )


rel_summary = summarize(
    rel_boot,
    observed_rel,
    REL_NAMES,
    {
        "roc_auc",
        "average_precision",
        "correlation",
    },
)

arb_summary = summarize(
    arb_boot,
    observed_arb,
    ARB_NAMES,
    {
        "micro_ap",
        "macro_ap",
    },
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 100)
print("PRIMARY RELIABILITY — MB MINUS MA")
print("=" * 100)

for _, row in rel_summary.iterrows():

    print(
        f"{row['metric']:<20}"
        f"obs={row['observed_delta']:+.8f} | "
        f"mean={row['bootstrap_mean']:+.8f} | "
        f"95% CI=["
        f"{row['ci_2.5']:+.8f}, "
        f"{row['ci_97.5']:+.8f}] | "
        f"improve="
        f"{row['improvement_fraction']:.3f}"
    )


print()
print("=" * 100)
print("PRIMARY ARBITRATION — A2 MINUS A1")
print("=" * 100)

for _, row in arb_summary.iterrows():

    print(
        f"{row['metric']:<20}"
        f"obs={row['observed_delta']:+.8f} | "
        f"mean={row['bootstrap_mean']:+.8f} | "
        f"95% CI=["
        f"{row['ci_2.5']:+.8f}, "
        f"{row['ci_97.5']:+.8f}] | "
        f"improve="
        f"{row['improvement_fraction']:.3f}"
    )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUT / "reliability_MB_minus_MA_bootstrap.npy",
    rel_boot,
)

np.save(
    OUT / "arbitration_A2_minus_A1_bootstrap.npy",
    arb_boot,
)

rel_summary.to_csv(
    OUT / "reliability_bootstrap_summary.csv",
    index=False,
)

arb_summary.to_csv(
    OUT / "arbitration_bootstrap_summary.csv",
    index=False,
)


manifest = {
    "analysis":
        "final_test_bootstrap_v1",

    "evaluation_population":
        "4,731 paired structure-eligible "
        "frozen Graph-V2 TEST proteins",

    "bootstrap_replicates":
        N_BOOT,

    "seed":
        SEED,

    "resampling_unit":
        "whole protein",

    "all_494_labels_kept_together":
        True,

    "models_refitted_per_bootstrap":
        False,

    "primary_reliability":
        "MB minus MA",

    "primary_reliability_metric":
        "average precision",

    "primary_arbitration":
        "A2 minus A1",

    "interval":
        "percentile 95%",

    "interpretation":
        (
            "Bootstrap intervals quantify TEST sampling "
            "uncertainty conditional on the frozen "
            "CAL-fitted models."
        ),

    "test_used_for_tuning":
        False,
}


with open(
    OUT / "bootstrap_manifest.json",
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

print("Bootstrap replicates:          1000")
print("Seed:                          42")
print("Whole proteins resampled:      YES")
print("All 494 labels kept together:  YES")
print("Models refitted:               NO")
print("New model selection:           NO")
print("New threshold optimization:    NO")
print("TEST used for tuning:          NO")
print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)
