from pathlib import Path
import hashlib
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
# FROZEN CONFIGURATION
# ============================================================

EPS = 1e-7

CAL_BASE = Path(
    "data/conflict/go_frequency_v1_inputs"
)

SEQ_DIR = Path(
    "data/final_evaluation/sequence_expert_v1_test"
)

STRUCT_DIR = Path(
    "data/final_evaluation/structure_expert_v1_test"
)

TEST_LABELS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/test_labels.npy"
)

TEST_IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/test_uniprot_ids.txt"
)

OUT = Path(
    "data/final_evaluation/final_test_primary_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_TEST = 4754
EXPECTED_PAIRED = 4731
NUM_LABELS = 494


# ============================================================
# HELPERS
# ============================================================

def read_ids(path):
    return [
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def uncertainty(p):
    return (
        1.0
        - 2.0 * np.abs(p - 0.5)
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

    m = 0.5 * (
        ps + pt
    )

    return 0.5 * (
        ps * np.log(ps / m)
        + (1.0 - ps)
        * np.log(
            (1.0 - ps)
            / (1.0 - m)
        )
        + pt * np.log(pt / m)
        + (1.0 - pt)
        * np.log(
            (1.0 - pt)
            / (1.0 - m)
        )
    )


def prediction_metrics(y, p):

    supported = (
        y.sum(axis=0) > 0
    )

    micro_ap = (
        average_precision_score(
            y.ravel(),
            p.ravel(),
        )
    )

    macro_ap = (
        average_precision_score(
            y[:, supported],
            p[:, supported],
            average="macro",
        )
    )

    mae = np.mean(
        np.abs(p - y)
    )

    brier = np.mean(
        (p - y) ** 2
    )

    nll = log_loss(
        y.ravel(),
        np.clip(
            p.ravel(),
            EPS,
            1.0 - EPS,
        ),
        labels=[0, 1],
    )

    return {
        "micro_ap":
            float(micro_ap),

        "macro_ap_supported_labels":
            float(macro_ap),

        "mae":
            float(mae),

        "brier":
            float(brier),

        "nll":
            float(nll),

        "supported_labels":
            int(supported.sum()),

        "zero_positive_labels":
            int((~supported).sum()),
    }


def binary_metrics(y, p):

    return {
        "roc_auc":
            float(
                roc_auc_score(y, p)
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
            float(
                np.corrcoef(
                    y,
                    p,
                )[0, 1]
            ),
    }


# ============================================================
# START
# ============================================================

print("=" * 100)
print("FINAL TEST V1 — PRIMARY CONFIRMATORY EVALUATION")
print("=" * 100)

print(
    "TEST ground truth opening: YES — "
    "pre-registered confirmatory evaluation"
)

print("Model tuning on TEST: NO")
print("Calibration fitting on TEST: NO")
print("Conflict selection on TEST: NO")
print("Threshold optimization on TEST: NO")
print("Arbitration fitting on TEST: NO")
print()


# ============================================================
# LOAD CAL — FITTING POPULATION ONLY
# ============================================================

ps_cal = np.load(
    CAL_BASE / "p_seq_cal.npy"
).astype(np.float64)

pt_cal = np.load(
    CAL_BASE / "p_struct_cal.npy"
).astype(np.float64)

y_cal = np.load(
    CAL_BASE / "y_cal.npy"
).astype(np.float64)


if not (
    ps_cal.shape
    == pt_cal.shape
    == y_cal.shape
    == (2369, 494)
):
    raise RuntimeError(
        "Unexpected CAL shapes."
    )


# ============================================================
# LOAD FROZEN TEST PREDICTIONS
# ============================================================

test_ids = read_ids(
    TEST_IDS_PATH
)

seq_ids = read_ids(
    SEQ_DIR / "test_uniprot_ids.txt"
)

struct_ids = read_ids(
    STRUCT_DIR
    / "test_structure_eligible_ids.txt"
)


ps_all = np.load(
    SEQ_DIR
    / "test_calibrated_probabilities.npy"
).astype(np.float64)

pt = np.load(
    STRUCT_DIR
    / "test_probabilities.npy"
).astype(np.float64)


if len(test_ids) != EXPECTED_TEST:
    raise RuntimeError(
        "Frozen TEST ID count mismatch."
    )

if seq_ids != test_ids:
    raise RuntimeError(
        "Sequence TEST IDs do not exactly match "
        "frozen TEST order."
    )

if len(struct_ids) != EXPECTED_PAIRED:
    raise RuntimeError(
        "Structure eligible TEST count mismatch."
    )

if ps_all.shape != (
    EXPECTED_TEST,
    NUM_LABELS,
):
    raise RuntimeError(
        "Sequence TEST probability shape mismatch."
    )

if pt.shape != (
    EXPECTED_PAIRED,
    NUM_LABELS,
):
    raise RuntimeError(
        "Structure TEST probability shape mismatch."
    )


# ============================================================
# PAIRED TEST ORDER
# ============================================================

test_index = {
    uid: i
    for i, uid in enumerate(test_ids)
}

if any(
    uid not in test_index
    for uid in struct_ids
):
    raise RuntimeError(
        "Structure TEST ID missing from frozen TEST."
    )

paired_rows = np.asarray(
    [
        test_index[uid]
        for uid in struct_ids
    ],
    dtype=np.int64,
)

ps = ps_all[
    paired_rows
]


# ============================================================
# FIRST DELIBERATE TEST GROUND-TRUTH LOAD
# ============================================================

y_all = np.load(
    TEST_LABELS_PATH
).astype(np.uint8)

if y_all.shape != (
    EXPECTED_TEST,
    NUM_LABELS,
):
    raise RuntimeError(
        "Frozen TEST label shape mismatch."
    )

y = y_all[
    paired_rows
].astype(np.float64)

if y.shape != (
    EXPECTED_PAIRED,
    NUM_LABELS,
):
    raise RuntimeError(
        "Paired TEST label shape mismatch."
    )

if not np.all(
    (y == 0)
    | (y == 1)
):
    raise RuntimeError(
        "TEST labels are not binary."
    )


print(
    f"Frozen TEST proteins:       "
    f"{EXPECTED_TEST:,}"
)

print(
    f"Paired TEST proteins:       "
    f"{EXPECTED_PAIRED:,}"
)

print(
    f"Structure-ineligible:       "
    f"{EXPECTED_TEST - EXPECTED_PAIRED:,}"
)

print(
    f"Labels:                     "
    f"{NUM_LABELS}"
)

print(
    f"Paired positive annotations:"
    f" {int(y.sum()):,}"
)

print()


# ============================================================
# EXPERT PERFORMANCE
# ============================================================

p_avg = 0.5 * (
    ps + pt
)

expert_results = {
    "sequence":
        prediction_metrics(y, ps),

    "structure":
        prediction_metrics(y, pt),

    "simple_average":
        prediction_metrics(y, p_avg),
}


print("=" * 100)
print("FROZEN EXPERT PERFORMANCE — PAIRED TEST")
print("=" * 100)

print(
    f"{'Model':<20}"
    f"{'Micro AP':>12}"
    f"{'Macro AP':>12}"
    f"{'MAE':>12}"
    f"{'Brier':>12}"
    f"{'NLL':>12}"
)

for name in [
    "sequence",
    "structure",
    "simple_average",
]:

    r = expert_results[name]

    print(
        f"{name:<20}"
        f"{r['micro_ap']:>12.8f}"
        f"{r['macro_ap_supported_labels']:>12.8f}"
        f"{r['mae']:>12.8f}"
        f"{r['brier']:>12.8f}"
        f"{r['nll']:>12.8f}"
    )

print()


# ============================================================
# FROZEN CONFLICT + UNCERTAINTY FEATURES
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


us_test = uncertainty(
    ps
)

ut_test = uncertainty(
    pt
)

js_test = bernoulli_js(
    ps,
    pt,
)


# ============================================================
# PRIMARY RELIABILITY ENDPOINT
#
# Structure failure:
# |p_structure - y| >= 0.5
#
# MA = [U_structure, U_sequence]
# MB = [U_structure, U_sequence, JS]
# ============================================================

err_cal = np.abs(
    y_cal - pt_cal
)

err_test = np.abs(
    y - pt
)


binary_cal = (
    err_cal >= 0.5
).astype(np.uint8).ravel()

binary_test = (
    err_test >= 0.5
).astype(np.uint8).ravel()


continuous_cal = (
    err_cal.ravel()
)

continuous_test = (
    err_test.ravel()
)


XA_cal = np.stack(
    [
        ut_cal,
        us_cal,
    ],
    axis=-1,
).reshape(-1, 2)

XA_test = np.stack(
    [
        ut_test,
        us_test,
    ],
    axis=-1,
).reshape(-1, 2)


XB_cal = np.column_stack(
    [
        XA_cal,
        js_cal.ravel(),
    ]
)

XB_test = np.column_stack(
    [
        XA_test,
        js_test.ravel(),
    ]
)


# ---------------- MA ----------------

ma_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

ma_cont = Ridge(
    alpha=1.0
)

ma_bin.fit(
    XA_cal,
    binary_cal,
)

ma_cont.fit(
    XA_cal,
    continuous_cal,
)


# ---------------- MB ----------------

mb_bin = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

mb_cont = Ridge(
    alpha=1.0
)

mb_bin.fit(
    XB_cal,
    binary_cal,
)

mb_cont.fit(
    XB_cal,
    continuous_cal,
)


ma_bin_pred = (
    ma_bin.predict_proba(
        XA_test
    )[:, 1]
)

mb_bin_pred = (
    mb_bin.predict_proba(
        XB_test
    )[:, 1]
)

ma_cont_pred = (
    ma_cont.predict(
        XA_test
    )
)

mb_cont_pred = (
    mb_cont.predict(
        XB_test
    )
)


ma_bin_metrics = binary_metrics(
    binary_test,
    ma_bin_pred,
)

mb_bin_metrics = binary_metrics(
    binary_test,
    mb_bin_pred,
)

ma_cont_metrics = continuous_metrics(
    continuous_test,
    ma_cont_pred,
)

mb_cont_metrics = continuous_metrics(
    continuous_test,
    mb_cont_pred,
)


reliability_delta_binary = {
    k:
        mb_bin_metrics[k]
        - ma_bin_metrics[k]
    for k in ma_bin_metrics
}

reliability_delta_continuous = {
    k:
        mb_cont_metrics[k]
        - ma_cont_metrics[k]
    for k in ma_cont_metrics
}


print("=" * 100)
print("PRIMARY CONFIRMATORY RELIABILITY — STRUCTURE EXPERT")
print("=" * 100)

print(
    "Structure failure prevalence: "
    f"{binary_test.mean():.6%}"
)

print()

for metric in [
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
]:

    print(
        f"{metric:<20}"
        f"MA={ma_bin_metrics[metric]:.8f}  "
        f"MB={mb_bin_metrics[metric]:.8f}  "
        f"MB-MA="
        f"{reliability_delta_binary[metric]:+.8f}"
    )

print()

for metric in [
    "mae",
    "rmse",
    "correlation",
]:

    print(
        f"{metric:<20}"
        f"MA={ma_cont_metrics[metric]:.8f}  "
        f"MB={mb_cont_metrics[metric]:.8f}  "
        f"MB-MA="
        f"{reliability_delta_continuous[metric]:+.8f}"
    )

print()


# ============================================================
# FROZEN ARBITRATION V1
#
# A0 = sequence
# A1 = uncertainty-only gate
# A2 = uncertainty + JS gate
#
# Gate fitting: CAL ONLY.
# ============================================================

es_cal = np.abs(
    ps_cal - y_cal
)

et_cal = np.abs(
    pt_cal - y_cal
)

gate_y_cal = (
    et_cal < es_cal
).astype(np.uint8).ravel()


X1_cal = np.column_stack(
    [
        us_cal.ravel(),
        ut_cal.ravel(),
    ]
)

X1_test = np.column_stack(
    [
        us_test.ravel(),
        ut_test.ravel(),
    ]
)


X2_cal = np.column_stack(
    [
        us_cal.ravel(),
        ut_cal.ravel(),
        js_cal.ravel(),
    ]
)

X2_test = np.column_stack(
    [
        us_test.ravel(),
        ut_test.ravel(),
        js_test.ravel(),
    ]
)


gate1 = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)

gate2 = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
)


gate1.fit(
    X1_cal,
    gate_y_cal,
)

gate2.fit(
    X2_cal,
    gate_y_cal,
)


g1 = (
    gate1.predict_proba(
        X1_test
    )[:, 1]
    .reshape(y.shape)
)

g2 = (
    gate2.predict_proba(
        X2_test
    )[:, 1]
    .reshape(y.shape)
)


p_a0 = ps

p_a1 = (
    (1.0 - g1) * ps
    + g1 * pt
)

p_a2 = (
    (1.0 - g2) * ps
    + g2 * pt
)


arbitration_results = {
    "A0_sequence":
        prediction_metrics(
            y,
            p_a0,
        ),

    "A1_uncertainty_gate":
        prediction_metrics(
            y,
            p_a1,
        ),

    "A2_conflict_gate":
        prediction_metrics(
            y,
            p_a2,
        ),
}


print("=" * 100)
print("FROZEN ARBITRATION V1 — PAIRED TEST")
print("=" * 100)

print(
    f"{'Model':<24}"
    f"{'Micro AP':>12}"
    f"{'Macro AP':>12}"
    f"{'MAE':>12}"
    f"{'Brier':>12}"
    f"{'NLL':>12}"
)

for name in [
    "A0_sequence",
    "A1_uncertainty_gate",
    "A2_conflict_gate",
]:

    r = arbitration_results[name]

    print(
        f"{name:<24}"
        f"{r['micro_ap']:>12.8f}"
        f"{r['macro_ap_supported_labels']:>12.8f}"
        f"{r['mae']:>12.8f}"
        f"{r['brier']:>12.8f}"
        f"{r['nll']:>12.8f}"
    )


arb_a2_minus_a1 = {}

arb_a2_minus_a0 = {}

for metric in [
    "micro_ap",
    "macro_ap_supported_labels",
    "mae",
    "brier",
    "nll",
]:

    arb_a2_minus_a1[metric] = (
        arbitration_results[
            "A2_conflict_gate"
        ][metric]
        -
        arbitration_results[
            "A1_uncertainty_gate"
        ][metric]
    )

    arb_a2_minus_a0[metric] = (
        arbitration_results[
            "A2_conflict_gate"
        ][metric]
        -
        arbitration_results[
            "A0_sequence"
        ][metric]
    )


print()
print("PRIMARY: A2 - A1")

for k, v in (
    arb_a2_minus_a1.items()
):
    print(
        f"  {k:<28}"
        f"{v:+.8f}"
    )


print()
print("REFERENCE: A2 - A0 sequence")

for k, v in (
    arb_a2_minus_a0.items()
):
    print(
        f"  {k:<28}"
        f"{v:+.8f}"
    )


# ============================================================
# ORACLE — DIAGNOSTIC ONLY
# ============================================================

es_test = np.abs(
    ps - y
)

et_test = np.abs(
    pt - y
)

oracle_mae = np.minimum(
    es_test,
    et_test,
).mean()

sequence_mae = (
    arbitration_results[
        "A0_sequence"
    ]["mae"]
)

a2_mae = (
    arbitration_results[
        "A2_conflict_gate"
    ]["mae"]
)

possible_gain = (
    sequence_mae
    - oracle_mae
)

realized_gain = (
    sequence_mae
    - a2_mae
)

if possible_gain > 0:
    oracle_fraction = (
        realized_gain
        / possible_gain
    )
else:
    oracle_fraction = np.nan


# ============================================================
# SAVE FIXED TEST ARRAYS
#
# These allow conditional bootstrap without refitting models.
# ============================================================

np.save(
    OUT / "paired_test_labels.npy",
    y.astype(np.uint8),
)

np.save(
    OUT / "paired_sequence_probs.npy",
    ps.astype(np.float32),
)

np.save(
    OUT / "paired_structure_probs.npy",
    pt.astype(np.float32),
)

np.save(
    OUT / "bernoulli_js.npy",
    js_test.astype(np.float32),
)

np.save(
    OUT / "structure_failure_target.npy",
    binary_test.reshape(
        y.shape
    ),
)

np.save(
    OUT / "structure_absolute_error.npy",
    err_test.astype(np.float32),
)

np.save(
    OUT / "MA_binary_predictions.npy",
    ma_bin_pred.reshape(
        y.shape
    ).astype(np.float32),
)

np.save(
    OUT / "MB_binary_predictions.npy",
    mb_bin_pred.reshape(
        y.shape
    ).astype(np.float32),
)

np.save(
    OUT / "MA_continuous_predictions.npy",
    ma_cont_pred.reshape(
        y.shape
    ).astype(np.float32),
)

np.save(
    OUT / "MB_continuous_predictions.npy",
    mb_cont_pred.reshape(
        y.shape
    ).astype(np.float32),
)

np.save(
    OUT / "A1_gate_probability.npy",
    g1.astype(np.float32),
)

np.save(
    OUT / "A2_gate_probability.npy",
    g2.astype(np.float32),
)

np.save(
    OUT / "A1_predictions.npy",
    p_a1.astype(np.float32),
)

np.save(
    OUT / "A2_predictions.npy",
    p_a2.astype(np.float32),
)

(
    OUT / "paired_test_uniprot_ids.txt"
).write_text(
    "\n".join(struct_ids) + "\n",
    encoding="utf-8",
)


# ============================================================
# RESULTS + MANIFEST
# ============================================================

results = {
    "population": {
        "frozen_test_total":
            EXPECTED_TEST,

        "paired_structure_eligible":
            EXPECTED_PAIRED,

        "structure_ineligible":
            EXPECTED_TEST
            - EXPECTED_PAIRED,

        "labels":
            NUM_LABELS,

        "positive_annotations":
            int(y.sum()),
    },

    "expert_performance":
        expert_results,

    "primary_reliability": {
        "endpoint":
            "structure failure: "
            "|p_structure-y| >= 0.5",

        "conflict":
            "Bernoulli Jensen-Shannon "
            "divergence, natural log",

        "failure_prevalence":
            float(
                binary_test.mean()
            ),

        "MA_binary":
            ma_bin_metrics,

        "MB_binary":
            mb_bin_metrics,

        "MB_minus_MA_binary":
            reliability_delta_binary,

        "MA_continuous":
            ma_cont_metrics,

        "MB_continuous":
            mb_cont_metrics,

        "MB_minus_MA_continuous":
            reliability_delta_continuous,

        "MA_binary_coefficients":
            ma_bin.coef_[0].tolist(),

        "MB_binary_coefficients":
            mb_bin.coef_[0].tolist(),

        "MA_continuous_coefficients":
            ma_cont.coef_.tolist(),

        "MB_continuous_coefficients":
            mb_cont.coef_.tolist(),
    },

    "arbitration_v1": {
        "results":
            arbitration_results,

        "A2_minus_A1":
            arb_a2_minus_a1,

        "A2_minus_A0":
            arb_a2_minus_a0,

        "A1_coefficients":
            gate1.coef_[0].tolist(),

        "A2_coefficients":
            gate2.coef_[0].tolist(),

        "oracle_mae_diagnostic_only":
            float(oracle_mae),

        "oracle_headroom_realized_by_A2":
            float(oracle_fraction),
    },
}


with open(
    OUT / "primary_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=2,
    )


manifest = {
    "analysis":
        "final_test_primary_v1",

    "status":
        "confirmatory locked TEST evaluation",

    "test_ground_truth_opened":
        True,

    "test_used_for_training":
        False,

    "test_used_for_model_selection":
        False,

    "test_used_for_calibration":
        False,

    "test_used_for_conflict_selection":
        False,

    "test_used_for_threshold_optimization":
        False,

    "test_used_for_arbitration_fitting":
        False,

    "fit_population":
        "CAL only",

    "evaluation_population":
        "4,731 structure-eligible "
        "frozen Graph-V2 TEST proteins",

    "primary_conflict":
        "Bernoulli Jensen-Shannon "
        "divergence with natural logarithm",

    "primary_reliability_comparison":
        "MB versus MA for structure failure",

    "primary_reliability_metric":
        "average precision",

    "primary_arbitration_comparison":
        "A2 conflict gate versus "
        "A1 uncertainty-only gate",

    "bootstrap":
        "not performed in this script; "
        "fixed TEST predictions saved "
        "for pre-registered conditional "
        "whole-protein bootstrap",

    "test_labels_sha256":
        sha256_file(
            TEST_LABELS_PATH
        ),

    "paired_ids_sha256":
        sha256_file(
            OUT
            / "paired_test_uniprot_ids.txt"
        ),

    "sequence_probabilities_sha256":
        sha256_file(
            SEQ_DIR
            / "test_calibrated_probabilities.npy"
        ),

    "structure_probabilities_sha256":
        sha256_file(
            STRUCT_DIR
            / "test_probabilities.npy"
        ),

    "results_sha256":
        sha256_file(
            OUT
            / "primary_results.json"
        ),
}


with open(
    OUT / "primary_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


# ============================================================
# FINAL INTEGRITY REPORT
# ============================================================

print()
print("=" * 100)
print("FINAL TEST PRIMARY V1 — INTEGRITY")
print("=" * 100)

print("TEST ground truth opened:           YES")
print("Models trained on TEST:             NO")
print("Calibration fitted on TEST:         NO")
print("Conflict selected on TEST:          NO")
print("Threshold optimized on TEST:        NO")
print("Arbitration gates fitted on TEST:   NO")
print("Reliability models fitted on:       CAL ONLY")
print("Arbitration gates fitted on:        CAL ONLY")
print("Primary conflict:                   Bernoulli JS")
print("Bootstrap performed here:           NO")
print()

print(
    "Results SHA256:",
    sha256_file(
        OUT / "primary_results.json"
    )
)

print(
    "Manifest SHA256:",
    sha256_file(
        OUT / "primary_manifest.json"
    )
)

print("=" * 100)
