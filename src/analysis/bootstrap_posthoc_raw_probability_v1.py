from pathlib import Path
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path("data/final_evaluation/posthoc_raw_probability_baseline_v1")
PRIMARY = Path("data/final_evaluation/final_test_primary_v1")

N_BOOT = 1000
SEED = 42
SHAPE = (4731, 494)

predictions = np.load(BASE / "test_reliability_predictions.npz")

for name in ["MA", "MB", "MC", "MD"]:
    if name not in predictions.files:
        raise RuntimeError(f"Missing predictions: {name}")
    if predictions[name].shape != SHAPE:
        raise RuntimeError(f"Unexpected shape for {name}")

target = np.load(
    PRIMARY / "structure_failure_target.npy"
).astype(np.uint8)

if target.shape != SHAPE:
    raise RuntimeError("Unexpected structure-failure target shape")

if not np.all((target == 0) | (target == 1)):
    raise RuntimeError("Target must be binary")

mc = predictions["MC"].astype(np.float64)
md = predictions["MD"].astype(np.float64)

if not np.isfinite(mc).all() or not np.isfinite(md).all():
    raise RuntimeError("Non-finite predictions")

def delta_metrics(idx):
    y = target[idx].ravel()
    p_mc = mc[idx].ravel()
    p_md = md[idx].ravel()

    return (
        average_precision_score(y, p_md)
        - average_precision_score(y, p_mc),
        roc_auc_score(y, p_md)
        - roc_auc_score(y, p_mc),
    )

all_idx = np.arange(SHAPE[0])
observed_ap, observed_auc = delta_metrics(all_idx)

rng = np.random.default_rng(SEED)
boot = np.empty((N_BOOT, 2), dtype=np.float64)

print("POST-HOC RAW-PROBABILITY BASELINE BOOTSTRAP")
print("Resampling: whole proteins")
print(f"Replicates: {N_BOOT}")
print(f"Seed: {SEED}")
print("Models refitted: NO")
print()

for b in range(N_BOOT):
    idx = rng.integers(
        0,
        SHAPE[0],
        size=SHAPE[0],
    )
    boot[b] = delta_metrics(idx)

    if (b + 1) % 100 == 0:
        print(f"Completed {b + 1}/{N_BOOT}")

ap_ci = np.percentile(boot[:, 0], [2.5, 97.5])
auc_ci = np.percentile(boot[:, 1], [2.5, 97.5])

result = {
    "classification": "post_hoc_supporting",
    "comparison": "MD_minus_MC",
    "n_bootstrap": N_BOOT,
    "seed": SEED,
    "resampling_unit": "whole_protein",
    "models_refitted": False,
    "average_precision": {
        "observed_delta": float(observed_ap),
        "ci_95_percentile": ap_ci.tolist(),
        "fraction_bootstrap_positive": float(
            np.mean(boot[:, 0] > 0)
        ),
    },
    "roc_auc": {
        "observed_delta": float(observed_auc),
        "ci_95_percentile": auc_ci.tolist(),
        "fraction_bootstrap_positive": float(
            np.mean(boot[:, 1] > 0)
        ),
    },
}

BASE.mkdir(parents=True, exist_ok=True)

( BASE / "bootstrap_results.json" ).write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)

np.save(BASE / "bootstrap_deltas.npy", boot)

print()
print(
    f"MD - MC AP: {observed_ap:+.9f} "
    f"95% CI [{ap_ci[0]:+.9f}, {ap_ci[1]:+.9f}]"
)
print(
    f"MD - MC ROC-AUC: {observed_auc:+.9f} "
    f"95% CI [{auc_ci[0]:+.9f}, {auc_ci[1]:+.9f}]"
)
print()
print("Saved:", BASE / "bootstrap_results.json")
print("Saved:", BASE / "bootstrap_deltas.npy")
