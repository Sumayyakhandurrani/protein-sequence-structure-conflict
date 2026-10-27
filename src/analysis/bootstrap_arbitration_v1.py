from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss


BASE = Path("data/conflict/go_frequency_v1_inputs")
ARB  = Path("data/conflict/arbitration_v1")
OUT  = ARB / "bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
SEED = 42
EPS = 1e-7


# ============================================================
# LOAD
# ============================================================

y = np.load(BASE / "y_eval.npy").astype(np.float64)

p1 = np.load(
    ARB / "A1_predictions.npy"
).astype(np.float64)

p2 = np.load(
    ARB / "A2_predictions.npy"
).astype(np.float64)

assert y.shape == p1.shape == p2.shape

n_proteins, n_labels = y.shape

print("=" * 90)
print("ARBITRATION V1 — PAIRED WHOLE-PROTEIN BOOTSTRAP")
print("=" * 90)
print(f"Proteins: {n_proteins:,}")
print(f"Labels:   {n_labels:,}")
print(f"Replicates: {N_BOOT:,}")
print(f"Seed: {SEED}")
print("Models refit during bootstrap: NO")
print("TEST USED: NO")
print()


# ============================================================
# METRICS
# ============================================================

def basic_metrics(yb, pb):

    yf = yb.ravel()
    pf = pb.ravel()

    micro_ap = average_precision_score(
        yf,
        pf
    )

    mae = np.mean(
        np.abs(pb - yb)
    )

    brier = np.mean(
        (pb - yb) ** 2
    )

    nll = log_loss(
        yf,
        np.clip(pf, EPS, 1-EPS),
        labels=[0, 1]
    )

    return (
        micro_ap,
        mae,
        brier,
        nll,
    )


# Macro AP is calculated separately so labels with
# zero positives in a bootstrap replicate are excluded.
def macro_ap_supported(yb, pb):

    support = yb.sum(axis=0) > 0

    if not np.any(support):
        return np.nan

    vals = []

    for j in np.where(support)[0]:

        vals.append(
            average_precision_score(
                yb[:, j],
                pb[:, j]
            )
        )

    return float(np.mean(vals))


# ============================================================
# OBSERVED
# ============================================================

m1 = basic_metrics(y, p1)
m2 = basic_metrics(y, p2)

macro1 = macro_ap_supported(y, p1)
macro2 = macro_ap_supported(y, p2)

observed = np.array([
    m2[0] - m1[0],       # micro AP: positive better
    macro2 - macro1,     # macro AP: positive better
    m2[1] - m1[1],       # MAE: negative better
    m2[2] - m1[2],       # Brier: negative better
    m2[3] - m1[3],       # NLL: negative better
])

names = [
    "micro_ap",
    "macro_ap",
    "mae",
    "brier",
    "nll",
]


print("Observed A2 - A1:")
for name, value in zip(names, observed):
    print(f"  {name:<10} {value:+.8f}")

print()


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(SEED)

boot = np.empty(
    (N_BOOT, len(names)),
    dtype=np.float64
)

for b in range(N_BOOT):

    idx = rng.integers(
        0,
        n_proteins,
        size=n_proteins
    )

    yb = y[idx]
    p1b = p1[idx]
    p2b = p2[idx]

    a = basic_metrics(yb, p1b)
    c = basic_metrics(yb, p2b)

    macro_a = macro_ap_supported(
        yb,
        p1b
    )

    macro_c = macro_ap_supported(
        yb,
        p2b
    )

    boot[b] = [
        c[0] - a[0],
        macro_c - macro_a,
        c[1] - a[1],
        c[2] - a[2],
        c[3] - a[3],
    ]

    if (b + 1) % 100 == 0:
        print(
            f"Completed {b + 1}/{N_BOOT}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("BOOTSTRAP RESULTS — A2 MINUS A1")
print("=" * 90)

rows = []

for j, name in enumerate(names):

    values = boot[:, j]

    lo, hi = np.percentile(
        values,
        [2.5, 97.5]
    )

    mean = values.mean()

    if name in [
        "micro_ap",
        "macro_ap",
    ]:
        improvement_fraction = np.mean(
            values > 0
        )
        direction = "positive"
    else:
        improvement_fraction = np.mean(
            values < 0
        )
        direction = "negative"

    rows.append({
        "metric": name,
        "observed_delta": observed[j],
        "bootstrap_mean": mean,
        "ci_2.5": lo,
        "ci_97.5": hi,
        "improvement_direction": direction,
        "improvement_fraction":
            improvement_fraction,
    })

    print()
    print(name)
    print(
        f"  observed:             "
        f"{observed[j]:+.8f}"
    )
    print(
        f"  bootstrap mean:       "
        f"{mean:+.8f}"
    )
    print(
        f"  95% CI:               "
        f"[{lo:+.8f}, {hi:+.8f}]"
    )
    print(
        f"  improvement fraction: "
        f"{improvement_fraction:.3f}"
    )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUT / "A2_minus_A1_bootstrap.npy",
    boot
)

pd.DataFrame(rows).to_csv(
    OUT / "bootstrap_summary.csv",
    index=False
)


print()
print("=" * 90)
print("INTERPRETATION")
print("=" * 90)

print(
    "AP: positive delta favors A2."
)
print(
    "MAE/Brier/NLL: negative delta favors A2."
)
print(
    "Whole proteins were resampled; labels were kept together."
)
print(
    "Models were NOT refit during bootstrap."
)
print(
    "Intervals are conditional on the fitted CAL models."
)
print("TEST USED: NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 90)