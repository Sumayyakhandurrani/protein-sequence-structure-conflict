from pathlib import Path
import json

import numpy as np
from sklearn.metrics import average_precision_score


# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE = Path(
    "data/conflict/go_frequency_v1_inputs"
)

ROB = Path(
    "data/robustness/structural_degradation_v1"
)

OUT = ROB / "analysis"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EPS = 1e-7

LEVELS = [
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


def bernoulli_js(ps, pt):
    """
    Frozen conflict_protocol_v1 definition:
    per-label Bernoulli Jensen-Shannon divergence,
    using natural logarithms.
    """

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


def micro_ap(y, p):
    return float(
        average_precision_score(
            y.ravel(),
            p.ravel(),
        )
    )


def macro_ap(y, p):

    values = []

    for j in range(
        y.shape[1]
    ):

        if np.sum(
            y[:, j]
        ) == 0:
            continue

        values.append(
            average_precision_score(
                y[:, j],
                p[:, j],
            )
        )

    return float(
        np.mean(values)
    )


def align_probabilities(
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


# ============================================================
# LOAD FROZEN CAL-EVAL DATA
# ============================================================

eval_ids = read_ids(
    BASE / "eval_ids.txt"
)

p_seq = np.load(
    BASE / "p_seq_eval.npy"
).astype(
    np.float64
)

p_struct_clean = np.load(
    BASE / "p_struct_eval.npy"
).astype(
    np.float64
)

y = np.load(
    BASE / "y_eval.npy"
).astype(
    np.float64
)

expected_shape = (
    len(eval_ids),
    494,
)

for name, arr in [
    ("p_seq", p_seq),
    ("p_struct_clean", p_struct_clean),
    ("y", y),
]:
    if arr.shape != expected_shape:
        raise RuntimeError(
            f"{name} shape mismatch: "
            f"{arr.shape}"
        )

    if not np.isfinite(
        arr
    ).all():
        raise RuntimeError(
            f"Non-finite values in {name}."
        )


print("=" * 88)
print("STRUCTURAL DEGRADATION V1 — CAL-EVAL ANALYSIS")
print("=" * 88)

print(
    "CAL-EVAL proteins:",
    len(eval_ids),
)

print(
    "Labels:",
    y.shape[1],
)

print(
    "Protein-label pairs:",
    f"{y.size:,}",
)

print()


# ============================================================
# LOAD / ALIGN EACH CONDITION
# ============================================================

conditions = {
    "clean":
        p_struct_clean
}


for name, sigma in LEVELS[1:]:

    sigma_dir = (
        ROB / f"sigma_{sigma}"
    )

    ids = read_ids(
        sigma_dir
        / "uniprot_ids.txt"
    )

    probs = np.load(
        sigma_dir
        / "structure_probabilities.npy"
    ).astype(
        np.float64
    )

    if probs.shape != (
        len(ids),
        494,
    ):
        raise RuntimeError(
            f"{name} probability shape mismatch."
        )

    aligned = align_probabilities(
        ids,
        probs,
        eval_ids,
    )

    if aligned.shape != expected_shape:
        raise RuntimeError(
            f"{name} aligned shape mismatch."
        )

    conditions[
        name
    ] = aligned


# ============================================================
# CLEAN RECONCILIATION
#
# Perturbation script did not regenerate sigma=0.
# The existing frozen CAL-EVAL structure probabilities
# are the clean reference.
# ============================================================

clean = conditions[
    "clean"
]

clean_micro_ap = micro_ap(
    y,
    clean,
)

clean_macro_ap = macro_ap(
    y,
    clean,
)

clean_mae = float(
    np.mean(
        np.abs(
            y - clean
        )
    )
)

clean_brier = float(
    np.mean(
        (
            y - clean
        ) ** 2
    )
)

clean_conflict = bernoulli_js(
    p_seq,
    clean,
)

clean_mean_js = float(
    clean_conflict.mean()
)


# ============================================================
# ANALYZE CONDITIONS
# ============================================================

rows = []

for name, _ in LEVELS:

    pt = conditions[
        name
    ]

    abs_error = np.abs(
        y - pt
    )

    conflict = bernoulli_js(
        p_seq,
        pt,
    )

    prediction_shift = np.abs(
        pt - clean
    )

    structure_failure = (
        abs_error >= 0.5
    )

    row = {
        "condition":
            name,

        "proteins":
            len(eval_ids),

        "pairs":
            int(y.size),

        "structure_micro_ap":
            micro_ap(
                y,
                pt,
            ),

        "structure_macro_ap":
            macro_ap(
                y,
                pt,
            ),

        "structure_mae":
            float(
                abs_error.mean()
            ),

        "structure_brier":
            float(
                np.mean(
                    (
                        y - pt
                    ) ** 2
                )
            ),

        "structure_failure_rate":
            float(
                structure_failure.mean()
            ),

        "mean_bernoulli_js":
            float(
                conflict.mean()
            ),

        "median_bernoulli_js":
            float(
                np.median(
                    conflict
                )
            ),

        "mean_abs_prediction_shift_vs_clean":
            float(
                prediction_shift.mean()
            ),

        "delta_micro_ap_vs_clean":
            micro_ap(
                y,
                pt,
            )
            - clean_micro_ap,

        "delta_macro_ap_vs_clean":
            macro_ap(
                y,
                pt,
            )
            - clean_macro_ap,

        "delta_mae_vs_clean":
            float(
                abs_error.mean()
            )
            - clean_mae,

        "delta_brier_vs_clean":
            float(
                np.mean(
                    (
                        y - pt
                    ) ** 2
                )
            )
            - clean_brier,

        "delta_mean_js_vs_clean":
            float(
                conflict.mean()
            )
            - clean_mean_js,
    }

    rows.append(
        row
    )


# ============================================================
# SAVE
# ============================================================

json_path = (
    OUT / "structural_degradation_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        rows,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print(
    f"{'Condition':<12}"
    f"{'microAP':>12}"
    f"{'macroAP':>12}"
    f"{'MAE':>12}"
    f"{'Brier':>12}"
    f"{'FailRate':>12}"
    f"{'Mean JS':>12}"
    f"{'|Δp|':>12}"
)

print("-" * 96)

for r in rows:

    print(
        f"{r['condition']:<12}"
        f"{r['structure_micro_ap']:>12.6f}"
        f"{r['structure_macro_ap']:>12.6f}"
        f"{r['structure_mae']:>12.6f}"
        f"{r['structure_brier']:>12.6f}"
        f"{r['structure_failure_rate']:>12.6f}"
        f"{r['mean_bernoulli_js']:>12.6f}"
        f"{r['mean_abs_prediction_shift_vs_clean']:>12.6f}"
    )


print()
print("Changes relative to CLEAN")
print("-" * 88)

for r in rows[1:]:

    print()
    print(
        r["condition"]
    )

    print(
        "  Δ microAP:",
        f"{r['delta_micro_ap_vs_clean']:+.8f}",
    )

    print(
        "  Δ macroAP:",
        f"{r['delta_macro_ap_vs_clean']:+.8f}",
    )

    print(
        "  Δ MAE:",
        f"{r['delta_mae_vs_clean']:+.8f}",
    )

    print(
        "  Δ Brier:",
        f"{r['delta_brier_vs_clean']:+.8f}",
    )

    print(
        "  Δ mean JS:",
        f"{r['delta_mean_js_vs_clean']:+.8f}",
    )


print()
print(
    "Saved:",
    json_path,
)

print()
print(
    "NOTE: This is a post-confirmatory "
    "CAL-EVAL robustness analysis."
)

print(
    "No TEST data are used here."
)
