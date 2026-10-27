from pathlib import Path
import json

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# CONFIG
# ============================================================

ROB = Path(
    "data/robustness/sequence_degradation_v1"
)

INPUT = (
    ROB
    / "reliability_analysis"
)

OUT = (
    ROB
    / "reliability_analysis"
    / "bootstrap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CONDITIONS = [
    "clean",
    "mask_10pct",
    "mask_20pct",
    "mask_40pct",
]

N_PROTEINS = 2373
N_LABELS = 494

N_BOOT = 1000
SEED = 42


# ============================================================
# METRICS
# ============================================================

def binary_metrics(y, p):

    return {
        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
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


def percentile_ci(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return [
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    ]


# ============================================================
# HEADER
# ============================================================

print("=" * 104)
print(
    "E7 SEQUENCE DEGRADATION — "
    "WHOLE-PROTEIN BOOTSTRAP"
)
print("=" * 104)

print(
    "Proteins:",
    N_PROTEINS,
)

print(
    "Labels per protein:",
    N_LABELS,
)

print(
    "Bootstrap replicates:",
    N_BOOT,
)

print(
    "Seed:",
    SEED,
)

print(
    "Models refitted during bootstrap: NO"
)

print(
    "Bootstrap unit: whole protein"
)

print(
    "TEST data used: NO"
)

print()


# ============================================================
# ONE COMMON BOOTSTRAP INDEX SET
#
# Same protein resamples are used for all four conditions.
# Models/predictions remain fixed.
# ============================================================

rng = np.random.default_rng(
    SEED
)

bootstrap_indices = rng.integers(
    low=0,
    high=N_PROTEINS,
    size=(
        N_BOOT,
        N_PROTEINS,
    ),
    dtype=np.int32,
)


# ============================================================
# CONDITIONS
# ============================================================

all_results = {}

for condition in CONDITIONS:

    print("=" * 104)
    print(
        "Condition:",
        condition,
    )
    print("=" * 104)

    # --------------------------------------------------------
    # Load fixed predictions / targets
    # --------------------------------------------------------

    y_binary = np.load(
        INPUT
        / f"{condition}_binary_target.npy"
    )

    p_MA = np.load(
        INPUT
        / f"{condition}_binary_MA.npy"
    ).astype(np.float64)

    p_MB = np.load(
        INPUT
        / f"{condition}_binary_MB.npy"
    ).astype(np.float64)


    y_cont = np.load(
        INPUT
        / f"{condition}_continuous_target.npy"
    ).astype(np.float64)

    c_MA = np.load(
        INPUT
        / f"{condition}_continuous_MA.npy"
    ).astype(np.float64)

    c_MB = np.load(
        INPUT
        / f"{condition}_continuous_MB.npy"
    ).astype(np.float64)


    expected = (
        N_PROTEINS
        * N_LABELS
    )

    arrays = [
        y_binary,
        p_MA,
        p_MB,
        y_cont,
        c_MA,
        c_MB,
    ]

    if any(
        x.size != expected
        for x in arrays
    ):
        raise RuntimeError(
            f"{condition}: unexpected array size."
        )


    # --------------------------------------------------------
    # Reshape so bootstrap is explicitly by protein
    # --------------------------------------------------------

    y_binary = y_binary.reshape(
        N_PROTEINS,
        N_LABELS,
    )

    p_MA = p_MA.reshape(
        N_PROTEINS,
        N_LABELS,
    )

    p_MB = p_MB.reshape(
        N_PROTEINS,
        N_LABELS,
    )

    y_cont = y_cont.reshape(
        N_PROTEINS,
        N_LABELS,
    )

    c_MA = c_MA.reshape(
        N_PROTEINS,
        N_LABELS,
    )

    c_MB = c_MB.reshape(
        N_PROTEINS,
        N_LABELS,
    )


    # --------------------------------------------------------
    # Observed deltas
    # --------------------------------------------------------

    obs_bin_MA = binary_metrics(
        y_binary.reshape(-1),
        p_MA.reshape(-1),
    )

    obs_bin_MB = binary_metrics(
        y_binary.reshape(-1),
        p_MB.reshape(-1),
    )

    obs_cont_MA = continuous_metrics(
        y_cont.reshape(-1),
        c_MA.reshape(-1),
    )

    obs_cont_MB = continuous_metrics(
        y_cont.reshape(-1),
        c_MB.reshape(-1),
    )


    observed_binary_delta = {
        key:
            obs_bin_MB[key]
            - obs_bin_MA[key]
        for key in obs_bin_MA
    }

    observed_cont_delta = {
        key:
            obs_cont_MB[key]
            - obs_cont_MA[key]
        for key in obs_cont_MA
    }


    # --------------------------------------------------------
    # Bootstrap storage
    # --------------------------------------------------------

    boot_binary = {
        "roc_auc": [],
        "average_precision": [],
        "log_loss": [],
        "brier": [],
    }

    boot_cont = {
        "mae": [],
        "rmse": [],
        "correlation": [],
    }


    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    for b in range(
        N_BOOT
    ):

        rows = bootstrap_indices[
            b
        ]


        yb = y_binary[
            rows
        ].reshape(-1)

        ma = p_MA[
            rows
        ].reshape(-1)

        mb = p_MB[
            rows
        ].reshape(-1)


        # Extremely unlikely, but guard ROC if a bootstrap
        # replicate somehow contains only one binary class.
        if np.unique(yb).size < 2:
            raise RuntimeError(
                f"{condition}: bootstrap "
                f"{b} has one binary class."
            )


        m_ma = binary_metrics(
            yb,
            ma,
        )

        m_mb = binary_metrics(
            yb,
            mb,
        )


        for key in boot_binary:

            boot_binary[key].append(
                m_mb[key]
                - m_ma[key]
            )


        yc = y_cont[
            rows
        ].reshape(-1)

        ca = c_MA[
            rows
        ].reshape(-1)

        cb = c_MB[
            rows
        ].reshape(-1)


        cm_ma = continuous_metrics(
            yc,
            ca,
        )

        cm_mb = continuous_metrics(
            yc,
            cb,
        )


        for key in boot_cont:

            boot_cont[key].append(
                cm_mb[key]
                - cm_ma[key]
            )


        if (
            (b + 1) % 100
            == 0
        ):
            print(
                f"  bootstrap "
                f"{b + 1}/{N_BOOT}"
            )


    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    binary_summary = {}

    for key, values in boot_binary.items():

        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        # Favorable direction:
        # AP / ROC > 0
        # log-loss / Brier < 0

        if key in [
            "average_precision",
            "roc_auc",
        ]:

            favorable_fraction = float(
                np.mean(
                    arr > 0
                )
            )

        else:

            favorable_fraction = float(
                np.mean(
                    arr < 0
                )
            )


        binary_summary[key] = {
            "observed_delta":
                observed_binary_delta[
                    key
                ],

            "bootstrap_mean_delta":
                float(
                    arr.mean()
                ),

            "ci95":
                percentile_ci(
                    arr
                ),

            "favorable_fraction":
                favorable_fraction,
        }


    continuous_summary = {}

    for key, values in boot_cont.items():

        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        # Favorable:
        # MAE/RMSE < 0
        # correlation > 0

        if key == "correlation":

            favorable_fraction = float(
                np.mean(
                    arr > 0
                )
            )

        else:

            favorable_fraction = float(
                np.mean(
                    arr < 0
                )
            )


        continuous_summary[key] = {
            "observed_delta":
                observed_cont_delta[
                    key
                ],

            "bootstrap_mean_delta":
                float(
                    arr.mean()
                ),

            "ci95":
                percentile_ci(
                    arr
                ),

            "favorable_fraction":
                favorable_fraction,
        }


    result = {
        "condition":
            condition,

        "n_proteins":
            N_PROTEINS,

        "n_labels":
            N_LABELS,

        "n_bootstrap":
            N_BOOT,

        "seed":
            SEED,

        "bootstrap_unit":
            "whole protein",

        "model_refitting":
            False,

        "binary":
            binary_summary,

        "continuous":
            continuous_summary,
    }


    all_results[
        condition
    ] = result


    # Save raw bootstrap deltas too
    np.savez_compressed(
        OUT
        / f"{condition}_bootstrap_deltas.npz",

        roc_auc=np.asarray(
            boot_binary["roc_auc"]
        ),

        average_precision=np.asarray(
            boot_binary[
                "average_precision"
            ]
        ),

        log_loss=np.asarray(
            boot_binary["log_loss"]
        ),

        brier=np.asarray(
            boot_binary["brier"]
        ),

        mae=np.asarray(
            boot_cont["mae"]
        ),

        rmse=np.asarray(
            boot_cont["rmse"]
        ),

        correlation=np.asarray(
            boot_cont["correlation"]
        ),
    )


    print()
    print(
        "Observed / 95% bootstrap CI"
    )

    print(
        "ΔAP:",
        f"{observed_binary_delta['average_precision']:+.8f}",
        percentile_ci(
            boot_binary[
                "average_precision"
            ]
        ),
    )

    print(
        "ΔROC:",
        f"{observed_binary_delta['roc_auc']:+.8f}",
        percentile_ci(
            boot_binary[
                "roc_auc"
            ]
        ),
    )

    print(
        "ΔLogLoss:",
        f"{observed_binary_delta['log_loss']:+.8f}",
        percentile_ci(
            boot_binary[
                "log_loss"
            ]
        ),
    )

    print(
        "ΔBrier:",
        f"{observed_binary_delta['brier']:+.8f}",
        percentile_ci(
            boot_binary[
                "brier"
            ]
        ),
    )

    print(
        "ΔMAE:",
        f"{observed_cont_delta['mae']:+.8f}",
        percentile_ci(
            boot_cont["mae"]
        ),
    )

    print(
        "ΔRMSE:",
        f"{observed_cont_delta['rmse']:+.8f}",
        percentile_ci(
            boot_cont["rmse"]
        ),
    )

    print(
        "ΔCorrelation:",
        f"{observed_cont_delta['correlation']:+.8f}",
        percentile_ci(
            boot_cont[
                "correlation"
            ]
        ),
    )

    print()


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary = {
    "experiment":
        "E7 sequence degradation reliability bootstrap",

    "conditions":
        CONDITIONS,

    "n_bootstrap":
        N_BOOT,

    "seed":
        SEED,

    "bootstrap_unit":
        "whole protein",

    "all_494_labels_kept_together":
        True,

    "models_refitted":
        False,

    "test_used":
        False,

    "results":
        all_results,
}


summary_path = (
    OUT
    / "bootstrap_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print("=" * 104)
print(
    "E7 BOOTSTRAP COMPLETE"
)
print("=" * 104)

print(
    "Saved:",
    summary_path,
)

print(
    "Models refitted: NO"
)

print(
    "TEST data used: NO"
)
