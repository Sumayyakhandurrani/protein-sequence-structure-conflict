from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# ============================================================
# CONFIG
# ============================================================

BASE = Path("data/conflict/go_frequency_v1_inputs")

OUT = Path(
    "data/conflict/selective_prediction_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-7
RIDGE_ALPHA = 1.0

# Predefined reporting points.
COVERAGES = [
    1.00,
    0.95,
    0.90,
    0.80,
    0.70,
    0.50,
]


# ============================================================
# LOAD
# ============================================================

ps_cal = np.load(
    BASE / "p_seq_cal.npy"
).astype(np.float64)

pt_cal = np.load(
    BASE / "p_struct_cal.npy"
).astype(np.float64)

y_cal = np.load(
    BASE / "y_cal.npy"
).astype(np.float64)

ps_eval = np.load(
    BASE / "p_seq_eval.npy"
).astype(np.float64)

pt_eval = np.load(
    BASE / "p_struct_eval.npy"
).astype(np.float64)

y_eval = np.load(
    BASE / "y_eval.npy"
).astype(np.float64)


assert ps_cal.shape == pt_cal.shape == y_cal.shape
assert ps_eval.shape == pt_eval.shape == y_eval.shape

assert np.isfinite(ps_cal).all()
assert np.isfinite(pt_cal).all()
assert np.isfinite(y_cal).all()

assert np.isfinite(ps_eval).all()
assert np.isfinite(pt_eval).all()
assert np.isfinite(y_eval).all()


print("=" * 100)
print("SELECTIVE PREDICTION V1")
print("=" * 100)

print(f"CAL:       {y_cal.shape}")
print(f"CAL-EVAL:  {y_eval.shape}")
print("Target:    continuous sequence absolute error")
print("TEST USED: NO")
print()


# ============================================================
# FEATURES
# ============================================================

def uncertainty(p):
    return 1.0 - 2.0 * np.abs(p - 0.5)


def bernoulli_js(ps, pt):

    ps = np.clip(
        ps,
        EPS,
        1.0 - EPS
    )

    pt = np.clip(
        pt,
        EPS,
        1.0 - EPS
    )

    m = 0.5 * (ps + pt)

    return 0.5 * (
        ps * np.log(ps / m)
        +
        (1.0 - ps)
        * np.log(
            (1.0 - ps) / (1.0 - m)
        )
        +
        pt * np.log(pt / m)
        +
        (1.0 - pt)
        * np.log(
            (1.0 - pt) / (1.0 - m)
        )
    )


us_cal = uncertainty(ps_cal)
ut_cal = uncertainty(pt_cal)
js_cal = bernoulli_js(ps_cal, pt_cal)

us_eval = uncertainty(ps_eval)
ut_eval = uncertainty(pt_eval)
js_eval = bernoulli_js(ps_eval, pt_eval)


# ============================================================
# TARGET
# ============================================================

err_cal = np.abs(
    ps_cal - y_cal
)

err_eval = np.abs(
    ps_eval - y_eval
)

target_cal = err_cal.reshape(-1)
target_eval = err_eval.reshape(-1)

print(
    f"CAL mean sequence error:      "
    f"{target_cal.mean():.8f}"
)

print(
    f"CAL-EVAL mean sequence error: "
    f"{target_eval.mean():.8f}"
)

print()


# ============================================================
# MODEL DEFINITIONS
# ============================================================

features_cal = {
    "R1_seq_uncertainty":
        us_cal.reshape(-1, 1),

    "R2_both_uncertainties":
        np.column_stack([
            us_cal.ravel(),
            ut_cal.ravel(),
        ]),

    "R3_uncertainties_plus_JS":
        np.column_stack([
            us_cal.ravel(),
            ut_cal.ravel(),
            js_cal.ravel(),
        ]),
}


features_eval = {
    "R1_seq_uncertainty":
        us_eval.reshape(-1, 1),

    "R2_both_uncertainties":
        np.column_stack([
            us_eval.ravel(),
            ut_eval.ravel(),
        ]),

    "R3_uncertainties_plus_JS":
        np.column_stack([
            us_eval.ravel(),
            ut_eval.ravel(),
            js_eval.ravel(),
        ]),
}


# ============================================================
# FIT ON CAL ONLY
# ============================================================

predictions = {}
models = {}

for name in features_cal:

    print(f"Fitting {name}...")

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        features_cal[name],
        target_cal
    )

    pred = model.predict(
        features_eval[name]
    )

    models[name] = model
    predictions[name] = pred

    print(
        "  coefficients:",
        model.coef_.tolist()
    )

    print(
        f"  intercept: "
        f"{model.intercept_:.8f}"
    )

print()


# ============================================================
# RISK-COVERAGE
# ============================================================

def risk_at_coverage(
    true_error,
    predicted_risk,
    coverage,
):
    """
    Retain lowest predicted-risk observations.
    """

    order = np.argsort(
        predicted_risk
    )

    n_keep = int(
        np.floor(
            coverage
            * len(true_error)
        )
    )

    n_keep = max(
        1,
        min(n_keep, len(true_error))
    )

    keep = order[:n_keep]

    return float(
        np.mean(
            true_error[keep]
        )
    )


# Dense grid for AURC.
#
# Start above zero to avoid an unstable
# single-observation endpoint.
AURC_COVERAGES = np.linspace(
    0.01,
    1.0,
    100
)


def calculate_aurc(
    true_error,
    predicted_risk,
):

    risks = np.array([
        risk_at_coverage(
            true_error,
            predicted_risk,
            c
        )
        for c in AURC_COVERAGES
    ])

    # Normalize by integration interval so AURC
    # remains on approximately the same scale
    # as the risk.
    aurc = np.trapezoid(
        risks,
        AURC_COVERAGES
    ) / (
        AURC_COVERAGES[-1]
        - AURC_COVERAGES[0]
    )

    return float(aurc), risks


rows = []

print("=" * 100)
print("FIXED-COVERAGE RESULTS")
print("=" * 100)

for name, pred in predictions.items():

    print()
    print(name)
    print("-" * 100)

    for coverage in COVERAGES:

        risk = risk_at_coverage(
            target_eval,
            pred,
            coverage
        )

        rows.append({
            "model": name,
            "coverage": coverage,
            "risk_mae": risk,
        })

        print(
            f"Coverage={coverage:>5.0%} | "
            f"retained MAE={risk:.8f}"
        )


coverage_df = pd.DataFrame(rows)

coverage_df.to_csv(
    OUT / "fixed_coverage_results.csv",
    index=False
)


# ============================================================
# AURC
# ============================================================

aurc_results = {}
curve_rows = []

print()
print("=" * 100)
print("AURC")
print("=" * 100)

for name, pred in predictions.items():

    aurc, risks = calculate_aurc(
        target_eval,
        pred
    )

    aurc_results[name] = aurc

    print(
        f"{name:<30} "
        f"{aurc:.8f}"
    )

    for c, r in zip(
        AURC_COVERAGES,
        risks
    ):
        curve_rows.append({
            "model": name,
            "coverage": float(c),
            "risk_mae": float(r),
        })


pd.DataFrame(
    curve_rows
).to_csv(
    OUT / "risk_coverage_curves.csv",
    index=False
)


# ============================================================
# DIRECT R3 vs R2
# ============================================================

print()
print("=" * 100)
print("PRIMARY COMPARISON: R3 - R2")
print("=" * 100)

r2_name = "R2_both_uncertainties"
r3_name = "R3_uncertainties_plus_JS"

r2_rows = coverage_df[
    coverage_df["model"] == r2_name
].set_index("coverage")

r3_rows = coverage_df[
    coverage_df["model"] == r3_name
].set_index("coverage")

comparison_rows = []

for coverage in COVERAGES:

    r2 = float(
        r2_rows.loc[
            coverage,
            "risk_mae"
        ]
    )

    r3 = float(
        r3_rows.loc[
            coverage,
            "risk_mae"
        ]
    )

    delta = r3 - r2

    comparison_rows.append({
        "coverage": coverage,
        "R2_risk": r2,
        "R3_risk": r3,
        "R3_minus_R2": delta,
    })

    print(
        f"Coverage={coverage:>5.0%} | "
        f"R2={r2:.8f} | "
        f"R3={r3:.8f} | "
        f"delta={delta:+.8f}"
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    OUT / "R3_vs_R2_fixed_coverage.csv",
    index=False
)


aurc_delta = (
    aurc_results[r3_name]
    - aurc_results[r2_name]
)

print()
print(
    f"R2 AURC:      "
    f"{aurc_results[r2_name]:.8f}"
)

print(
    f"R3 AURC:      "
    f"{aurc_results[r3_name]:.8f}"
)

print(
    f"R3 - R2 AURC: "
    f"{aurc_delta:+.8f}"
)

print()
print(
    "Negative R3-R2 risk/AURC means "
    "adding JS improved selective prediction."
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

shape = y_eval.shape

for name, pred in predictions.items():

    np.save(
        OUT / f"{name}_predicted_risk.npy",
        pred.reshape(shape)
    )

np.save(
    OUT / "sequence_absolute_error.npy",
    err_eval
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "analysis":
        "selective_prediction_v1",

    "target":
        "absolute calibrated sequence prediction error",

    "target_formula":
        "|p_sequence - y|",

    "models": {
        "R1":
            ["U_sequence"],

        "R2":
            [
                "U_sequence",
                "U_structure",
            ],

        "R3":
            [
                "U_sequence",
                "U_structure",
                "Bernoulli_JS",
            ],
    },

    "reliability_model":
        "Ridge",

    "ridge_alpha":
        RIDGE_ALPHA,

    "fit_population":
        "CAL",

    "evaluation_population":
        "CAL-EVAL",

    "fixed_coverages":
        COVERAGES,

    "primary_comparison":
        "R3 versus R2",

    "primary_global_metric":
        "AURC",

    "aurc": {
        k: float(v)
        for k, v
        in aurc_results.items()
    },

    "R3_minus_R2_AURC":
        float(aurc_delta),

    "interpretation":
        (
            "Lower risk and lower AURC are better. "
            "R3 versus R2 isolates the incremental "
            "selective-prediction utility of the "
            "frozen JS conflict signal beyond both "
            "experts' uncertainty summaries."
        ),

    "threshold_selected":
        False,

    "test_used":
        False,
}


with open(
    OUT / "selective_prediction_manifest.json",
    "w"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# INTEGRITY
# ============================================================

print()
print("=" * 100)
print("INTEGRITY")
print("=" * 100)

print("Reliability models fitted on CAL:       YES")
print("Evaluation performed on CAL-EVAL:       YES")
print("Frozen JS formulation used:             YES")
print("Coverage points predefined:             YES")
print("Threshold optimized on CAL-EVAL:        NO")
print("TEST USED:                              NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 100)