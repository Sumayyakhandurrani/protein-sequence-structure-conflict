from pathlib import Path
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# Post-hoc supporting analysis.
# Does not modify the frozen confirmatory TEST protocol.

CAL = Path("data/conflict/go_frequency_v1_inputs")
SEQ = Path("data/final_evaluation/sequence_expert_v1_test")
STRUCT = Path("data/final_evaluation/structure_expert_v1_test")
SPLIT = Path("data/splits/mmseqs30_graph_v2/labels")
OUT = Path("data/final_evaluation/posthoc_raw_probability_baseline_v1")

EPS = 1e-7
N_LABELS = 494


def read_ids(path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def uncertainty(p):
    return 1.0 - 2.0 * np.abs(p - 0.5)


def bernoulli_js(p, q):
    p = np.clip(p, EPS, 1.0 - EPS)
    q = np.clip(q, EPS, 1.0 - EPS)
    m = 0.5 * (p + q)

    def kl(a, b):
        return a * np.log(a / b) + (1.0 - a) * np.log(
            (1.0 - a) / (1.0 - b)
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# Load CAL.
ps_cal = np.load(CAL / "p_seq_cal.npy").astype(np.float64)
pt_cal = np.load(CAL / "p_struct_cal.npy").astype(np.float64)
y_cal = np.load(CAL / "y_cal.npy").astype(np.float64)

assert ps_cal.shape == pt_cal.shape == y_cal.shape == (2369, N_LABELS)

# Load and align TEST.
test_ids = read_ids(SPLIT / "test_uniprot_ids.txt")
seq_ids = read_ids(SEQ / "test_uniprot_ids.txt")
struct_ids = read_ids(STRUCT / "test_structure_eligible_ids.txt")

assert len(test_ids) == 4754
assert seq_ids == test_ids
assert len(struct_ids) == 4731
assert len(set(struct_ids)) == 4731

test_index = {uid: i for i, uid in enumerate(test_ids)}
paired_rows = np.array([test_index[uid] for uid in struct_ids])

ps_all = np.load(
    SEQ / "test_calibrated_probabilities.npy"
).astype(np.float64)

pt = np.load(
    STRUCT / "test_probabilities.npy"
).astype(np.float64)

y_all = np.load(SPLIT / "test_labels.npy").astype(np.float64)

assert ps_all.shape == y_all.shape == (4754, N_LABELS)
assert pt.shape == (4731, N_LABELS)

ps = ps_all[paired_rows]
y = y_all[paired_rows]

assert np.isfinite(ps_cal).all()
assert np.isfinite(pt_cal).all()
assert np.isfinite(ps).all()
assert np.isfinite(pt).all()
assert np.all((y_cal == 0) | (y_cal == 1))
assert np.all((y == 0) | (y == 1))

# Same structure-failure endpoint as the frozen primary analysis.
target_cal = (np.abs(pt_cal - y_cal) >= 0.5).astype(np.uint8).ravel()
target_test = (np.abs(pt - y) >= 0.5).astype(np.uint8).ravel()

features_cal = {
    "MA": np.stack([
        uncertainty(pt_cal),
        uncertainty(ps_cal)
    ], axis=-1).reshape(-1, 2),

    "MC": np.stack([
        pt_cal, ps_cal
    ], axis=-1).reshape(-1, 2),
}

features_test = {
    "MA": np.stack([
        uncertainty(pt),
        uncertainty(ps)
    ], axis=-1).reshape(-1, 2),

    "MC": np.stack([
        pt, ps
    ], axis=-1).reshape(-1, 2),
}

features_cal["MB"] = np.column_stack([
    features_cal["MA"],
    bernoulli_js(ps_cal, pt_cal).ravel()
])

features_test["MB"] = np.column_stack([
    features_test["MA"],
    bernoulli_js(ps, pt).ravel()
])

features_cal["MD"] = np.column_stack([
    features_cal["MC"],
    bernoulli_js(ps_cal, pt_cal).ravel()
])

features_test["MD"] = np.column_stack([
    features_test["MC"],
    bernoulli_js(ps, pt).ravel()
])

results = {}
predictions = {}

for name in ["MA", "MB", "MC", "MD"]:
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    model.fit(features_cal[name], target_cal)
    pred = model.predict_proba(features_test[name])[:, 1]

    predictions[name] = pred.reshape(4731, N_LABELS)

    results[name] = {
        "average_precision": float(
            average_precision_score(target_test, pred)
        ),
        "roc_auc": float(
            roc_auc_score(target_test, pred)
        ),
        "n_features": int(features_cal[name].shape[1]),
        "n_iter": int(model.n_iter_[0]),
        "converged_before_limit": bool(model.n_iter_[0] < 2000)
    }

print("\nPOST-HOC RAW-PROBABILITY RELIABILITY BASELINE")
print("CAL proteins: 2369")
print("Paired TEST proteins: 4731")
print("Labels: 494")
print("Endpoint: structure |p-y| >= 0.5\n")

for name, r in results.items():
    print(
        f"{name}: AP={r['average_precision']:.9f} "
        f"ROC-AUC={r['roc_auc']:.9f} "
        f"iterations={r['n_iter']}"
    )

print(
    "\nMB - MA AP:",
    results["MB"]["average_precision"] -
    results["MA"]["average_precision"]
)

print(
    "MD - MC AP:",
    results["MD"]["average_precision"] -
    results["MC"]["average_precision"]
)

OUT.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUT / "test_reliability_predictions.npz",
    **predictions
)

payload = {
    "analysis_classification": "post_hoc_supporting",
    "test_used_for_model_fitting": False,
    "test_used_for_feature_selection": False,
    "cal_proteins": 2369,
    "paired_test_proteins": 4731,
    "labels": N_LABELS,
    "endpoint": "abs(p_structure-y)>=0.5",
    "models": results,
    "delta_ap_MB_minus_MA": (
        results["MB"]["average_precision"] -
        results["MA"]["average_precision"]
    ),
    "delta_ap_MD_minus_MC": (
        results["MD"]["average_precision"] -
        results["MC"]["average_precision"]
    ),
    "source_script_sha256": sha256_file(__file__),
}

(OUT / "results.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)

print("\nSaved:", OUT / "results.json")
print("Saved:", OUT / "test_reliability_predictions.npz")
