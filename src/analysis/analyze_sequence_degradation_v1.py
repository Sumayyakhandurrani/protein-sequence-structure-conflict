from pathlib import Path
import json

import numpy as np
from sklearn.metrics import average_precision_score


# ============================================================
# PATHS / CONSTANTS
# ============================================================

INPUT = Path(
    "data/conflict/go_frequency_v1_inputs"
)

ROB = Path(
    "data/robustness/sequence_degradation_v1"
)

OUT = ROB / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

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

    m1 = 0.5 * (ps + pt)
    m0 = 1.0 - m1

    kl_s = (
        ps * np.log(ps / m1)
        +
        (1.0 - ps)
        * np.log(
            (1.0 - ps) / m0
        )
    )

    kl_t = (
        pt * np.log(pt / m1)
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
        for i, uid in enumerate(source_ids)
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

    return source_probs[rows]


# ============================================================
# LOAD FROZEN CAL-EVAL TARGETS / STRUCTURE
# ============================================================

eval_ids = read_ids(
    INPUT / "eval_ids.txt"
)

y = np.load(
    INPUT / "y_eval.npy"
).astype(np.float64)

p_struct = np.load(
    INPUT / "p_struct_eval.npy"
).astype(np.float64)

p_seq_reference = np.load(
    INPUT / "p_seq_eval.npy"
).astype(np.float64)


if y.shape != (2373, 494):
    raise RuntimeError(
        f"Unexpected y shape: {y.shape}"
    )

if p_struct.shape != y.shape:
    raise RuntimeError(
        "Structure probability shape mismatch."
    )

if p_seq_reference.shape != y.shape:
    raise RuntimeError(
        "Reference sequence probability shape mismatch."
    )


# ============================================================
# LOAD ALL E7 CONDITIONS
# ============================================================

condition_probs = {}

for name in CONDITIONS:

    d = ROB / name

    ids = read_ids(
        d / "uniprot_ids.txt"
    )

    p = np.load(
        d / "calibrated_probabilities.npy"
    ).astype(np.float64)

    p = align_probs(
        ids,
        p,
        eval_ids,
    )

    if p.shape != y.shape:
        raise RuntimeError(
            f"{name}: aligned probability shape mismatch."
        )

    if not np.isfinite(p).all():
        raise RuntimeError(
            f"{name}: NaN/Inf probabilities."
        )

    condition_probs[name] = p


# ============================================================
# VERIFY CLEAN REPRODUCTION
# ============================================================

clean = condition_probs[
    "clean"
]

clean_diff = np.abs(
    clean - p_seq_reference
)

print("=" * 96)
print("E7 SEQUENCE DEGRADATION — DESCRIPTIVE ANALYSIS")
print("=" * 96)

print("CAL-EVAL proteins:", len(eval_ids))
print("Labels:", y.shape[1])

print()
print("Clean E7 vs frozen CAL-EVAL sequence probabilities")
print("Exact equality:", np.array_equal(
    clean,
    p_seq_reference,
))
print(
    "Max absolute difference:",
    float(clean_diff.max()),
)
print(
    "Mean absolute difference:",
    float(clean_diff.mean()),
)
print()


# ============================================================
# METRICS
# ============================================================

results = []

for name in CONDITIONS:

    ps = condition_probs[
        name
    ]

    # ---------------- AP ----------------

    micro_ap = float(
        average_precision_score(
            y.reshape(-1),
            ps.reshape(-1),
        )
    )

    label_ap = []

    for j in range(
        y.shape[1]
    ):

        if y[:, j].sum() == 0:
            continue

        label_ap.append(
            average_precision_score(
                y[:, j],
                ps[:, j],
            )
        )

    macro_ap = float(
        np.mean(label_ap)
    )

    # ---------------- Error ----------------

    abs_error = np.abs(
        y - ps
    )

    mae = float(
        abs_error.mean()
    )

    brier = float(
        np.mean(
            (y - ps) ** 2
        )
    )

    failure = (
        abs_error >= 0.5
    )

    failure_rate = float(
        failure.mean()
    )

    # ---------------- Conflict ----------------

    js = bernoulli_js(
        ps,
        p_struct,
    )

    mean_js = float(
        js.mean()
    )

    median_js = float(
        np.median(js)
    )

    # ---------------- Shift from clean ----------------

    probability_shift = np.abs(
        ps - clean
    )

    mean_probability_shift = float(
        probability_shift.mean()
    )

    # ---------------- Save ----------------

    result = {
        "condition":
            name,

        "micro_average_precision":
            micro_ap,

        "macro_average_precision":
            macro_ap,

        "mae":
            mae,

        "brier":
            brier,

        "failure_rate":
            failure_rate,

        "mean_bernoulli_js":
            mean_js,

        "median_bernoulli_js":
            median_js,

        "mean_absolute_probability_shift_from_clean":
            mean_probability_shift,
    }

    results.append(
        result
    )


# ============================================================
# ADD DELTAS RELATIVE TO CLEAN
# ============================================================

baseline = results[0]

for r in results:

    r["delta_vs_clean"] = {
        "micro_average_precision":
            r["micro_average_precision"]
            - baseline["micro_average_precision"],

        "macro_average_precision":
            r["macro_average_precision"]
            - baseline["macro_average_precision"],

        "mae":
            r["mae"]
            - baseline["mae"],

        "brier":
            r["brier"]
            - baseline["brier"],

        "failure_rate":
            r["failure_rate"]
            - baseline["failure_rate"],

        "mean_bernoulli_js":
            r["mean_bernoulli_js"]
            - baseline["mean_bernoulli_js"],
    }


# ============================================================
# SAVE
# ============================================================

out_path = (
    OUT
    / "sequence_degradation_summary.json"
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
# PRINT SUMMARY
# ============================================================

print(
    f"{'Condition':<14}"
    f"{'MicroAP':>11}"
    f"{'MacroAP':>11}"
    f"{'MAE':>11}"
    f"{'Brier':>11}"
    f"{'FailRate':>11}"
    f"{'MeanJS':>11}"
    f"{'|Δp|':>11}"
)

print("-" * 91)

for r in results:

    print(
        f"{r['condition']:<14}"
        f"{r['micro_average_precision']:>11.6f}"
        f"{r['macro_average_precision']:>11.6f}"
        f"{r['mae']:>11.6f}"
        f"{r['brier']:>11.6f}"
        f"{r['failure_rate']:>11.6f}"
        f"{r['mean_bernoulli_js']:>11.6f}"
        f"{r['mean_absolute_probability_shift_from_clean']:>11.6f}"
    )


print()
print("Deltas relative to clean")
print("-" * 91)

print(
    f"{'Condition':<14}"
    f"{'ΔMicroAP':>13}"
    f"{'ΔMacroAP':>13}"
    f"{'ΔMAE':>13}"
    f"{'ΔBrier':>13}"
    f"{'ΔFailRate':>13}"
    f"{'ΔMeanJS':>13}"
)

for r in results:

    d = r[
        "delta_vs_clean"
    ]

    print(
        f"{r['condition']:<14}"
        f"{d['micro_average_precision']:>+13.6f}"
        f"{d['macro_average_precision']:>+13.6f}"
        f"{d['mae']:>+13.6f}"
        f"{d['brier']:>+13.6f}"
        f"{d['failure_rate']:>+13.6f}"
        f"{d['mean_bernoulli_js']:>+13.6f}"
    )


print()
print("Saved:", out_path)
print("TEST data used: NO")
