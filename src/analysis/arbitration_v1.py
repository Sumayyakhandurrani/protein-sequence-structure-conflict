from pathlib import Path
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE = Path("data/conflict/go_frequency_v1_inputs")

OUT = Path("data/conflict/arbitration_v1")
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-7


# ============================================================
# LOAD
# ============================================================

ps_cal = np.load(BASE / "p_seq_cal.npy").astype(np.float64)
pt_cal = np.load(BASE / "p_struct_cal.npy").astype(np.float64)
y_cal  = np.load(BASE / "y_cal.npy").astype(np.float64)

ps_eval = np.load(BASE / "p_seq_eval.npy").astype(np.float64)
pt_eval = np.load(BASE / "p_struct_eval.npy").astype(np.float64)
y_eval  = np.load(BASE / "y_eval.npy").astype(np.float64)

assert ps_cal.shape == pt_cal.shape == y_cal.shape
assert ps_eval.shape == pt_eval.shape == y_eval.shape


# ============================================================
# FEATURES
# ============================================================

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


us_cal = uncertainty(ps_cal)
ut_cal = uncertainty(pt_cal)
js_cal = bernoulli_js(ps_cal, pt_cal)

us_eval = uncertainty(ps_eval)
ut_eval = uncertainty(pt_eval)
js_eval = bernoulli_js(ps_eval, pt_eval)


# ============================================================
# GATE TARGET
# ============================================================

es_cal = np.abs(ps_cal - y_cal)
et_cal = np.abs(pt_cal - y_cal)

es_eval = np.abs(ps_eval - y_eval)
et_eval = np.abs(pt_eval - y_eval)

# 1 = structure is closer to truth
gate_y_cal = (
    et_cal < es_cal
).astype(np.uint8).ravel()

gate_y_eval = (
    et_eval < es_eval
).astype(np.uint8).ravel()


# ============================================================
# GATE FEATURES
# ============================================================

X1_cal = np.column_stack([
    us_cal.ravel(),
    ut_cal.ravel(),
])

X1_eval = np.column_stack([
    us_eval.ravel(),
    ut_eval.ravel(),
])

X2_cal = np.column_stack([
    us_cal.ravel(),
    ut_cal.ravel(),
    js_cal.ravel(),
])

X2_eval = np.column_stack([
    us_eval.ravel(),
    ut_eval.ravel(),
    js_eval.ravel(),
])


# ============================================================
# TRAIN GATES — CAL ONLY
# ============================================================

gate1 = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

gate2 = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

print("=" * 90)
print("ARBITRATION V1")
print("=" * 90)

print(f"CAL:      {y_cal.shape}")
print(f"CAL-EVAL: {y_eval.shape}")
print()

print(
    f"CAL structure-better target prevalence: "
    f"{gate_y_cal.mean():.4%}"
)

print(
    f"EVAL structure-better prevalence:       "
    f"{gate_y_eval.mean():.4%}"
)

print()

print("Fitting A1 uncertainty-only gate...")
gate1.fit(X1_cal, gate_y_cal)

print("Fitting A2 conflict-aware gate...")
gate2.fit(X2_cal, gate_y_cal)


g1 = gate1.predict_proba(X1_eval)[:, 1]
g2 = gate2.predict_proba(X2_eval)[:, 1]


print()
print("Gate diagnostics:")
print(
    f"A1 gate ROC-AUC: "
    f"{roc_auc_score(gate_y_eval, g1):.8f}"
)
print(
    f"A1 gate AP:      "
    f"{average_precision_score(gate_y_eval, g1):.8f}"
)

print(
    f"A2 gate ROC-AUC: "
    f"{roc_auc_score(gate_y_eval, g2):.8f}"
)
print(
    f"A2 gate AP:      "
    f"{average_precision_score(gate_y_eval, g2):.8f}"
)

print()
print("A1 coefficients:", gate1.coef_[0].tolist())
print("A2 coefficients:", gate2.coef_[0].tolist())


# ============================================================
# SOFT ARBITRATION
# ============================================================

shape = y_eval.shape

g1_matrix = g1.reshape(shape)
g2_matrix = g2.reshape(shape)

p_a0 = ps_eval

p_a1 = (
    (1.0 - g1_matrix) * ps_eval
    + g1_matrix * pt_eval
)

p_a2 = (
    (1.0 - g2_matrix) * ps_eval
    + g2_matrix * pt_eval
)

p_avg = 0.5 * (
    ps_eval + pt_eval
)


# ============================================================
# METRICS
# ============================================================

supported = (
    y_eval.sum(axis=0) > 0
)


def metrics(p):

    micro_ap = average_precision_score(
        y_eval.ravel(),
        p.ravel()
    )

    macro_ap = average_precision_score(
        y_eval[:, supported],
        p[:, supported],
        average="macro"
    )

    mae = np.mean(
        np.abs(p - y_eval)
    )

    brier = np.mean(
        (p - y_eval) ** 2
    )

    nll = log_loss(
        y_eval.ravel(),
        np.clip(
            p.ravel(),
            EPS,
            1-EPS
        ),
        labels=[0, 1]
    )

    return {
        "micro_ap": float(micro_ap),
        "macro_ap": float(macro_ap),
        "mae": float(mae),
        "brier": float(brier),
        "nll": float(nll),
    }


models = {
    "A0_sequence": p_a0,
    "simple_average": p_avg,
    "A1_uncertainty_gate": p_a1,
    "A2_conflict_gate": p_a2,
}


results = {
    name: metrics(p)
    for name, p in models.items()
}


print()
print("=" * 90)
print("FINAL PREDICTION PERFORMANCE — CAL-EVAL")
print("=" * 90)

print(
    f"{'Model':<24}"
    f"{'Micro AP':>12}"
    f"{'Macro AP':>12}"
    f"{'MAE':>12}"
    f"{'Brier':>12}"
    f"{'NLL':>12}"
)

for name, r in results.items():

    print(
        f"{name:<24}"
        f"{r['micro_ap']:>12.8f}"
        f"{r['macro_ap']:>12.8f}"
        f"{r['mae']:>12.8f}"
        f"{r['brier']:>12.8f}"
        f"{r['nll']:>12.8f}"
    )


# ============================================================
# PRIMARY COMPARISONS
# ============================================================

print()
print("=" * 90)
print("PRIMARY: A2 CONFLICT GATE vs A1 UNCERTAINTY GATE")
print("=" * 90)

for metric in [
    "micro_ap",
    "macro_ap",
    "mae",
    "brier",
    "nll",
]:

    delta = (
        results["A2_conflict_gate"][metric]
        - results["A1_uncertainty_gate"][metric]
    )

    print(
        f"{metric:<12} "
        f"A2-A1 = {delta:+.8f}"
    )


print()
print("=" * 90)
print("A2 vs SEQUENCE")
print("=" * 90)

for metric in [
    "micro_ap",
    "macro_ap",
    "mae",
    "brier",
    "nll",
]:

    delta = (
        results["A2_conflict_gate"][metric]
        - results["A0_sequence"][metric]
    )

    print(
        f"{metric:<12} "
        f"A2-sequence = {delta:+.8f}"
    )


# ============================================================
# ORACLE REFERENCE — DIAGNOSTIC ONLY
# ============================================================

oracle_mae = np.minimum(
    es_eval,
    et_eval
).mean()

print()
print("=" * 90)
print("ORACLE REFERENCE")
print("=" * 90)

print(
    f"Sequence MAE:       "
    f"{results['A0_sequence']['mae']:.8f}"
)

print(
    f"A2 MAE:             "
    f"{results['A2_conflict_gate']['mae']:.8f}"
)

print(
    f"Oracle-choice MAE:  "
    f"{oracle_mae:.8f}"
)

possible_gain = (
    results["A0_sequence"]["mae"]
    - oracle_mae
)

realized_gain = (
    results["A0_sequence"]["mae"]
    - results["A2_conflict_gate"]["mae"]
)

if possible_gain > 0:
    fraction = realized_gain / possible_gain
else:
    fraction = np.nan

print(
    f"Oracle headroom realized by A2: "
    f"{fraction:.2%}"
)


# ============================================================
# SAVE
# ============================================================

np.save(
    OUT / "A1_gate_probability.npy",
    g1_matrix
)

np.save(
    OUT / "A2_gate_probability.npy",
    g2_matrix
)

np.save(
    OUT / "A1_predictions.npy",
    p_a1
)

np.save(
    OUT / "A2_predictions.npy",
    p_a2
)


manifest = {
    "analysis": "arbitration_v1",

    "gate_target":
        "structure expert has smaller absolute error than sequence expert",

    "A0":
        "sequence expert",

    "A1_features": [
        "U_sequence",
        "U_structure",
    ],

    "A2_features": [
        "U_sequence",
        "U_structure",
        "frozen Bernoulli JS conflict",
    ],

    "gate_model":
        "logistic regression",

    "arbitration":
        "soft probability mixture",

    "formula":
        "(1-g)*p_sequence + g*p_structure",

    "fit_population":
        "CAL",

    "evaluation_population":
        "CAL-EVAL",

    "primary_comparison":
        "A2 conflict-aware gate versus A1 uncertainty-only gate",

    "results":
        results,

    "oracle_mae":
        float(oracle_mae),

    "test_used":
        False,
}


with open(
    OUT / "arbitration_manifest.json",
    "w"
) as f:
    json.dump(
        manifest,
        f,
        indent=2
    )


print()
print("=" * 90)
print("INTEGRITY")
print("=" * 90)

print("Gate training population: CAL")
print("Evaluation population:    CAL-EVAL")
print("Frozen JS used:           YES")
print("Threshold optimization:   NONE")
print("TEST USED:                NO")

print()
print(f"Saved under:\n  {OUT}")
print("=" * 90)