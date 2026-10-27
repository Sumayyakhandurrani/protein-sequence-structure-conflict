"""
GO-frequency-stratified sequence-structure disagreement analysis.

Primary question
----------------
Does explicit sequence-structure disagreement improve prediction of
STRUCTURE-EXPERT failure across GO terms with different TRAIN frequencies?

Models
------
M_A = [U_struct, U_seq]
M_B = [U_struct, U_seq, D]

where
    U(p) = 1 - 2 * |p - 0.5|
    D    = |p_seq - p_struct|

Failure event
-------------
structure_failure = 1 if |y - p_struct| >= 0.5 else 0

IMPORTANT
---------
* GO-frequency bands MUST be defined using TRAIN support only.
* Reliability models are fitted on CALIBRATION proteins.
* Metrics are evaluated on CALIBRATION-EVALUATION proteins.
* Bootstrap resamples whole evaluation proteins, NOT individual
  protein-label pairs.
* Reliability models are NOT refitted inside bootstrap replicates.
* TEST data must never be loaded by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42
N_BOOTSTRAP = 1000

# Fixed in advance from TRAIN support.
FREQUENCY_BANDS = [
    ("75-99", 75, 99),
    ("100-249", 100, 249),
    ("250-499", 250, 499),
    ("500-999", 500, 999),
    (">=1000", 1000, np.inf),
]


# ============================================================
# Core definitions
# ============================================================

def uncertainty(p: np.ndarray) -> np.ndarray:
    """
    Probability uncertainty:
        U(p) = 1 - 2 * |p - 0.5|

    p=0.5 -> U=1
    p=0/1 -> U=0
    """
    return 1.0 - 2.0 * np.abs(p - 0.5)


def structure_failure(
    y: np.ndarray,
    p_struct: np.ndarray
) -> np.ndarray:
    """
    Diagnostic binary structure-error event used in the previous analysis.

        failure = 1 if |y - p_struct| >= 0.5

    Equivalent to incorrect binary classification at threshold 0.5.
    """
    return (np.abs(y - p_struct) >= 0.5).astype(np.uint8)


def build_features(
    p_seq: np.ndarray,
    p_struct: np.ndarray,
):
    """
    M_A = [U_struct, U_seq]
    M_B = [U_struct, U_seq, D]
    """
    u_seq = uncertainty(p_seq)
    u_struct = uncertainty(p_struct)
    disagreement = np.abs(p_seq - p_struct)

    x_a = np.column_stack([
        u_struct.ravel(),
        u_seq.ravel(),
    ])

    x_b = np.column_stack([
        u_struct.ravel(),
        u_seq.ravel(),
        disagreement.ravel(),
    ])

    return x_a, x_b


# ============================================================
# Metrics
# ============================================================

def safe_metrics(
    y_true: np.ndarray,
    pred_prob: np.ndarray,
) -> dict:
    """
    Calculate metrics safely.

    ROC-AUC and AP require both classes to be present.
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    pred_prob = np.asarray(pred_prob).astype(np.float64)

    result = {
        "n_pairs": int(len(y_true)),
        "n_failures": int(y_true.sum()),
        "failure_prevalence": float(y_true.mean()),
        "ap": np.nan,
        "roc_auc": np.nan,
        "log_loss": np.nan,
        "brier": np.nan,
    }

    unique = np.unique(y_true)

    if len(unique) == 2:
        result["ap"] = float(
            average_precision_score(y_true, pred_prob)
        )

        result["roc_auc"] = float(
            roc_auc_score(y_true, pred_prob)
        )

        result["log_loss"] = float(
            log_loss(
                y_true,
                pred_prob,
                labels=[0, 1],
            )
        )

        result["brier"] = float(
            brier_score_loss(y_true, pred_prob)
        )

    return result


# ============================================================
# Reliability model
# ============================================================

def fit_reliability_model(
    X: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    """
    Low-capacity logistic reliability model.

    Keep this identical across frequency bands.
    """
    if np.unique(y).size < 2:
        raise ValueError(
            "Reliability fitting data contains only one class."
        )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )

    model.fit(X, y)

    return model


# ============================================================
# Protein bootstrap
# ============================================================

def protein_bootstrap(
    y_eval_matrix: np.ndarray,
    pred_a_matrix: np.ndarray,
    pred_b_matrix: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Bootstrap whole proteins.

    Matrices have shape:
        n_eval_proteins x n_labels_in_band

    A sampled protein contributes ALL labels in the current
    frequency band.

    Models are NOT refitted.
    """
    rng = np.random.default_rng(seed)

    n_proteins = y_eval_matrix.shape[0]

    records = []

    for b in range(n_bootstrap):

        sampled_idx = rng.integers(
            0,
            n_proteins,
            size=n_proteins,
        )

        y_b = y_eval_matrix[sampled_idx].ravel()
        pa_b = pred_a_matrix[sampled_idx].ravel()
        pb_b = pred_b_matrix[sampled_idx].ravel()

        # AP / ROC undefined if only one class is sampled.
        if np.unique(y_b).size < 2:
            continue

        ap_a = average_precision_score(y_b, pa_b)
        ap_b = average_precision_score(y_b, pb_b)

        roc_a = roc_auc_score(y_b, pa_b)
        roc_b = roc_auc_score(y_b, pb_b)

        ll_a = log_loss(y_b, pa_b, labels=[0, 1])
        ll_b = log_loss(y_b, pb_b, labels=[0, 1])

        br_a = brier_score_loss(y_b, pa_b)
        br_b = brier_score_loss(y_b, pb_b)

        records.append({
            "bootstrap": b,
            "delta_ap": ap_b - ap_a,
            "delta_roc_auc": roc_b - roc_a,

            # Negative = improvement
            "delta_log_loss": ll_b - ll_a,
            "delta_brier": br_b - br_a,
        })

    return pd.DataFrame(records)


def summarize_bootstrap(
    bootstrap_df: pd.DataFrame,
) -> dict:

    output = {}

    for metric in [
        "delta_ap",
        "delta_roc_auc",
        "delta_log_loss",
        "delta_brier",
    ]:

        values = bootstrap_df[metric].dropna().values

        if len(values) == 0:
            output[f"{metric}_ci_low"] = np.nan
            output[f"{metric}_ci_high"] = np.nan
            continue

        low, high = np.percentile(
            values,
            [2.5, 97.5],
        )

        output[f"{metric}_ci_low"] = float(low)
        output[f"{metric}_ci_high"] = float(high)

    # Fraction of bootstrap replicates showing AP improvement.
    if len(bootstrap_df) > 0:
        output["ap_improvement_fraction"] = float(
            (bootstrap_df["delta_ap"] > 0).mean()
        )
    else:
        output["ap_improvement_fraction"] = np.nan

    output["n_valid_bootstrap"] = int(
        len(bootstrap_df)
    )

    return output


# ============================================================
# Band analysis
# ============================================================

def analyse_band(
    band_name: str,
    label_mask: np.ndarray,

    y_cal: np.ndarray,
    p_seq_cal: np.ndarray,
    p_struct_cal: np.ndarray,

    y_eval: np.ndarray,
    p_seq_eval: np.ndarray,
    p_struct_eval: np.ndarray,

    n_bootstrap: int,
    output_dir: Path,
) -> dict:

    print()
    print("=" * 70)
    print(f"Frequency band: {band_name}")
    print("=" * 70)

    n_labels = int(label_mask.sum())

    print(f"Labels in band: {n_labels}")

    if n_labels == 0:
        return {
            "band": band_name,
            "n_labels": 0,
        }

    # --------------------------------------------------------
    # Select labels belonging to this TRAIN-frequency band
    # --------------------------------------------------------

    y_cal_band = y_cal[:, label_mask]
    seq_cal_band = p_seq_cal[:, label_mask]
    struct_cal_band = p_struct_cal[:, label_mask]

    y_eval_band = y_eval[:, label_mask]
    seq_eval_band = p_seq_eval[:, label_mask]
    struct_eval_band = p_struct_eval[:, label_mask]

    # --------------------------------------------------------
    # Failure labels
    # --------------------------------------------------------

    failure_cal = structure_failure(
        y_cal_band,
        struct_cal_band,
    )

    failure_eval = structure_failure(
        y_eval_band,
        struct_eval_band,
    )

    print(
        f"CAL failures: "
        f"{failure_cal.sum():,} / {failure_cal.size:,} "
        f"({failure_cal.mean():.6%})"
    )

    print(
        f"EVAL failures: "
        f"{failure_eval.sum():,} / {failure_eval.size:,} "
        f"({failure_eval.mean():.6%})"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    x_a_cal, x_b_cal = build_features(
        seq_cal_band,
        struct_cal_band,
    )

    x_a_eval, x_b_eval = build_features(
        seq_eval_band,
        struct_eval_band,
    )

    y_cal_flat = failure_cal.ravel()
    y_eval_flat = failure_eval.ravel()

    # --------------------------------------------------------
    # Fit M_A and M_B on CAL only
    # --------------------------------------------------------

    model_a = fit_reliability_model(
        x_a_cal,
        y_cal_flat,
    )

    model_b = fit_reliability_model(
        x_b_cal,
        y_cal_flat,
    )

    # --------------------------------------------------------
    # Evaluate on separate calibration-evaluation proteins
    # --------------------------------------------------------

    pred_a_flat = model_a.predict_proba(
        x_a_eval
    )[:, 1]

    pred_b_flat = model_b.predict_proba(
        x_b_eval
    )[:, 1]

    metrics_a = safe_metrics(
        y_eval_flat,
        pred_a_flat,
    )

    metrics_b = safe_metrics(
        y_eval_flat,
        pred_b_flat,
    )

    delta_ap = metrics_b["ap"] - metrics_a["ap"]
    delta_roc = (
        metrics_b["roc_auc"]
        - metrics_a["roc_auc"]
    )

    delta_ll = (
        metrics_b["log_loss"]
        - metrics_a["log_loss"]
    )

    delta_brier = (
        metrics_b["brier"]
        - metrics_a["brier"]
    )

    print()
    print("Evaluation results")
    print("------------------")
    print(f"AP M_A:       {metrics_a['ap']:.6f}")
    print(f"AP M_B:       {metrics_b['ap']:.6f}")
    print(f"Delta AP:     {delta_ap:+.6f}")

    print(
        f"ROC M_A:      "
        f"{metrics_a['roc_auc']:.6f}"
    )

    print(
        f"ROC M_B:      "
        f"{metrics_b['roc_auc']:.6f}"
    )

    print(
        f"Delta ROC:    "
        f"{delta_roc:+.6f}"
    )

    print(
        f"Delta logloss:"
        f" {delta_ll:+.6f}"
    )

    print(
        f"Delta Brier: "
        f"{delta_brier:+.6f}"
    )

    # --------------------------------------------------------
    # Reshape reliability predictions back to
    # protein x labels for whole-protein bootstrap
    # --------------------------------------------------------

    pred_a_matrix = pred_a_flat.reshape(
        failure_eval.shape
    )

    pred_b_matrix = pred_b_flat.reshape(
        failure_eval.shape
    )

    # --------------------------------------------------------
    # Protein bootstrap
    # --------------------------------------------------------

    print()
    print(
        f"Running {n_bootstrap} "
        f"whole-protein bootstrap replicates..."
    )

    boot = protein_bootstrap(
        y_eval_matrix=failure_eval,
        pred_a_matrix=pred_a_matrix,
        pred_b_matrix=pred_b_matrix,
        n_bootstrap=n_bootstrap,
        seed=SEED,
    )

    boot_summary = summarize_bootstrap(boot)

    print(
        "Delta AP 95% CI: "
        f"[{boot_summary['delta_ap_ci_low']:+.6f}, "
        f"{boot_summary['delta_ap_ci_high']:+.6f}]"
    )

    print(
        "AP improvement fraction: "
        f"{boot_summary['ap_improvement_fraction']:.3f}"
    )

    # Save bootstrap replicates separately.
    boot.to_csv(
        output_dir /
        f"bootstrap_{band_name.replace('>=', 'ge_')}.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final band record
    # --------------------------------------------------------

    record = {
        "band": band_name,
        "n_labels": n_labels,

        "n_eval_proteins": int(
            failure_eval.shape[0]
        ),

        "n_pairs": int(
            failure_eval.size
        ),

        "n_failures": int(
            failure_eval.sum()
        ),

        "failure_prevalence": float(
            failure_eval.mean()
        ),

        "ap_MA": metrics_a["ap"],
        "ap_MB": metrics_b["ap"],
        "delta_ap": delta_ap,

        "roc_auc_MA": metrics_a["roc_auc"],
        "roc_auc_MB": metrics_b["roc_auc"],
        "delta_roc_auc": delta_roc,

        "log_loss_MA": metrics_a["log_loss"],
        "log_loss_MB": metrics_b["log_loss"],
        "delta_log_loss": delta_ll,

        "brier_MA": metrics_a["brier"],
        "brier_MB": metrics_b["brier"],
        "delta_brier": delta_brier,
    }

    record.update(boot_summary)

    return record


# ============================================================
# Main
# ============================================================

def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading frozen development artifacts...")

    # --------------------------------------------------------
    # Required arrays
    #
    # All probability / target arrays must have shape:
    #     n_proteins x 494
    #
    # IMPORTANT:
    # CAL and EVAL protein ordering must match across:
    #     y
    #     sequence probabilities
    #     structure probabilities
    # --------------------------------------------------------

    y_cal = np.load(args.y_cal)
    y_eval = np.load(args.y_eval)

    p_seq_cal = np.load(args.p_seq_cal)
    p_seq_eval = np.load(args.p_seq_eval)

    p_struct_cal = np.load(args.p_struct_cal)
    p_struct_eval = np.load(args.p_struct_eval)

    # CSV must contain one row per frozen GO label.
    labels = pd.read_csv(args.label_csv)

    if "train_protein_count" not in labels.columns:
        raise ValueError(
            "Label CSV must contain column "
            "'train_protein_count'."
        )

    train_support = (
        labels["train_protein_count"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    arrays = {
        "y_cal": y_cal,
        "y_eval": y_eval,
        "p_seq_cal": p_seq_cal,
        "p_seq_eval": p_seq_eval,
        "p_struct_cal": p_struct_cal,
        "p_struct_eval": p_struct_eval,
    }

    for name, arr in arrays.items():
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be 2-D, "
                f"got {arr.shape}"
            )

        if arr.shape[1] != len(labels):
            raise ValueError(
                f"{name}: label dimension "
                f"{arr.shape[1]} != "
                f"{len(labels)}"
            )

        if not np.isfinite(arr).all():
            raise ValueError(
                f"{name} contains NaN/Inf."
            )

    if y_cal.shape != p_seq_cal.shape:
        raise ValueError(
            "CAL y/sequence shapes differ."
        )

    if y_cal.shape != p_struct_cal.shape:
        raise ValueError(
            "CAL y/structure shapes differ."
        )

    if y_eval.shape != p_seq_eval.shape:
        raise ValueError(
            "EVAL y/sequence shapes differ."
        )

    if y_eval.shape != p_struct_eval.shape:
        raise ValueError(
            "EVAL y/structure shapes differ."
        )

    if not np.isin(
        y_cal,
        [0, 1],
    ).all():
        raise ValueError(
            "y_cal is not binary."
        )

    if not np.isin(
        y_eval,
        [0, 1],
    ).all():
        raise ValueError(
            "y_eval is not binary."
        )

    for name in [
        "p_seq_cal",
        "p_seq_eval",
        "p_struct_cal",
        "p_struct_eval",
    ]:
        arr = arrays[name]

        if arr.min() < 0 or arr.max() > 1:
            raise ValueError(
                f"{name} contains values "
                f"outside [0,1]."
            )

    print(f"CAL shape:  {y_cal.shape}")
    print(f"EVAL shape: {y_eval.shape}")
    print(f"Labels:     {len(labels)}")

    # --------------------------------------------------------
    # Analyse frequency bands
    # --------------------------------------------------------

    results = []

    for (
        band_name,
        lower,
        upper,
    ) in FREQUENCY_BANDS:

        if np.isinf(upper):
            mask = train_support >= lower
        else:
            mask = (
                (train_support >= lower)
                & (train_support <= upper)
            )

        record = analyse_band(
            band_name=band_name,
            label_mask=mask,

            y_cal=y_cal,
            p_seq_cal=p_seq_cal,
            p_struct_cal=p_struct_cal,

            y_eval=y_eval,
            p_seq_eval=p_seq_eval,
            p_struct_eval=p_struct_eval,

            n_bootstrap=args.n_bootstrap,
            output_dir=output_dir,
        )

        results.append(record)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_path = (
        output_dir /
        "go_frequency_conflict_summary.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    columns = [
        "band",
        "n_labels",
        "n_failures",
        "failure_prevalence",
        "ap_MA",
        "ap_MB",
        "delta_ap",
        "delta_ap_ci_low",
        "delta_ap_ci_high",
        "ap_improvement_fraction",
    ]

    available = [
        c for c in columns
        if c in results_df.columns
    ]

    print(
        results_df[available]
        .to_string(index=False)
    )

    # Save run configuration.
    config = {
        "seed": SEED,
        "n_bootstrap": args.n_bootstrap,

        "frequency_bands": [
            {
                "name": name,
                "lower": int(low),
                "upper": (
                    None
                    if np.isinf(high)
                    else int(high)
                ),
            }
            for name, low, high
            in FREQUENCY_BANDS
        ],

        "failure_definition":
            "abs(y - p_struct) >= 0.5",

        "M_A":
            "[U_struct, U_seq]",

        "M_B":
            "[U_struct, U_seq, "
            "abs(p_seq-p_struct)]",

        "bootstrap_unit":
            "protein",

        "models_refit_in_bootstrap":
            False,
    }

    with open(
        output_dir / "analysis_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print()
    print(f"Saved results to:")
    print(results_path)

    print()
    print(
        "TEST was not required or accessed "
        "by this analysis."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--y-cal",
        required=True,
        help="Calibration target matrix (.npy)",
    )

    parser.add_argument(
        "--y-eval",
        required=True,
        help="Calibration-evaluation target matrix (.npy)",
    )

    parser.add_argument(
        "--p-seq-cal",
        required=True,
        help="Calibrated sequence probabilities, CAL",
    )

    parser.add_argument(
        "--p-seq-eval",
        required=True,
        help="Calibrated sequence probabilities, EVAL",
    )

    parser.add_argument(
        "--p-struct-cal",
        required=True,
        help="Structure probabilities, CAL",
    )

    parser.add_argument(
        "--p-struct-eval",
        required=True,
        help="Structure probabilities, EVAL",
    )

    parser.add_argument(
        "--label-csv",
        required=True,
        help=(
            "Frozen 494-label CSV containing "
            "train_protein_count"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "experiments/conflict/"
            "go_frequency_v1"
        ),
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    main(args)