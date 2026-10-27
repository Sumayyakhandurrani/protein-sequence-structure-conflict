from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

BASE_SEED = 42
N_BOOTSTRAP = 1000

EXPECTED_POPULATION_SIZE = 2447
EXPECTED_TARGETS = 23087
EXPECTED_RESIDUES = 1279145

POPULATION_PATH = Path(
    "data/xai/functional_sites_v1/"
    "xai_attribution_eligible_ids.txt"
)

ROOT = Path(
    "data/xai/functional_sites_v1/"
    "deletion_faithfulness_v1"
)

SEQUENCE_DIR = ROOT / "sequence/per_protein"
STRUCTURE_DIR = ROOT / "structure/per_protein"

OUT_DIR = ROOT / "aggregate_v1"

LABEL_OUT = OUT_DIR / "label_level_faithfulness.csv"
PROTEIN_OUT = OUT_DIR / "protein_level_faithfulness.csv"
SUMMARY_OUT = OUT_DIR / "faithfulness_summary.csv"
RUN_INFO_OUT = OUT_DIR / "run_info.json"


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_ids(path):
    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def bootstrap_seed(modality):
    material = (
        f"{BASE_SEED}|{modality}|"
        "DELETION_FAITHFULNESS_BOOTSTRAP_V1"
    ).encode("utf-8")

    digest = hashlib.sha256(
        material
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )
        % (2**32)
    )


def bootstrap_mean_ci(values, modality):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n != EXPECTED_POPULATION_SIZE:
        raise RuntimeError(
            f"{modality}: unexpected protein count "
            f"{n}"
        )

    rng = np.random.default_rng(
        bootstrap_seed(modality)
    )

    means = np.empty(
        N_BOOTSTRAP,
        dtype=np.float64,
    )

    for b in range(N_BOOTSTRAP):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        means[b] = values[
            indices
        ].mean()

    lower, upper = np.quantile(
        means,
        [0.025, 0.975],
    )

    return (
        float(lower),
        float(upper),
    )


# ============================================================
# LOAD FROZEN POPULATION
# ============================================================

population = read_ids(
    POPULATION_PATH
)

if len(population) != EXPECTED_POPULATION_SIZE:
    raise RuntimeError(
        "Unexpected frozen population size."
    )

if len(set(population)) != len(population):
    raise RuntimeError(
        "Duplicate IDs in frozen population."
    )


# ============================================================
# AGGREGATE BOTH MODALITIES
# ============================================================

label_rows = []
protein_rows = []

total_targets_by_modality = {}
total_residues_by_modality = {}


for modality, directory in [
    ("sequence", SEQUENCE_DIR),
    ("structure", STRUCTURE_DIR),
]:

    files = sorted(
        directory.glob("*.npz")
    )

    if len(files) != EXPECTED_POPULATION_SIZE:
        raise RuntimeError(
            f"{modality}: expected "
            f"{EXPECTED_POPULATION_SIZE} files, "
            f"found {len(files)}."
        )

    file_ids = {
        p.stem
        for p in files
    }

    if file_ids != set(population):
        raise RuntimeError(
            f"{modality}: output ID set mismatch."
        )

    modality_targets = 0
    modality_residues = 0

    for uid in population:

        path = directory / f"{uid}.npz"

        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            stored_uid = str(
                z["uniprot_id"].item()
            )

            if stored_uid != uid:
                raise RuntimeError(
                    f"{modality}/{uid}: ID mismatch."
                )

            length = int(
                z["length"].item()
            )

            k = int(
                z["k"].item()
            )

            labels = np.asarray(
                z["positive_label_indices"],
                dtype=np.int32,
            )

            top_drop = np.asarray(
                z["top_deletion_drop"],
                dtype=np.float64,
            )

            random_drop = np.asarray(
                z["random_mean_drop"],
                dtype=np.float64,
            )

            stored_enrichment = np.asarray(
                z["faithfulness_enrichment"],
                dtype=np.float64,
            )

        K = len(labels)

        if K <= 0:
            raise RuntimeError(
                f"{modality}/{uid}: zero targets."
            )

        expected_shape = (K,)

        for name, arr in [
            ("top_drop", top_drop),
            ("random_drop", random_drop),
            (
                "stored_enrichment",
                stored_enrichment,
            ),
        ]:

            if arr.shape != expected_shape:
                raise RuntimeError(
                    f"{modality}/{uid}: "
                    f"{name} shape mismatch."
                )

            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"{modality}/{uid}: "
                    f"{name} contains non-finite values."
                )

        recomputed_enrichment = (
            top_drop - random_drop
        )

        if not np.allclose(
            stored_enrichment,
            recomputed_enrichment,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{modality}/{uid}: stored "
                "faithfulness enrichment does not "
                "equal top_drop - random_drop."
            )

        for j in range(K):

            label_rows.append({
                "modality":
                    modality,

                "uniprot_id":
                    uid,

                "go_label_index":
                    int(labels[j]),

                "length":
                    length,

                "deletion_k":
                    k,

                "top_deletion_drop":
                    float(top_drop[j]),

                "random_mean_drop":
                    float(random_drop[j]),

                "faithfulness_enrichment":
                    float(
                        recomputed_enrichment[j]
                    ),
            })

        protein_enrichment = float(
            recomputed_enrichment.mean()
        )

        protein_top_drop = float(
            top_drop.mean()
        )

        protein_random_drop = float(
            random_drop.mean()
        )

        protein_rows.append({
            "modality":
                modality,

            "uniprot_id":
                uid,

            "length":
                length,

            "deletion_k":
                k,

            "n_positive_labels":
                K,

            "mean_top_deletion_drop":
                protein_top_drop,

            "mean_random_deletion_drop":
                protein_random_drop,

            "mean_faithfulness_enrichment":
                protein_enrichment,
        })

        modality_targets += K
        modality_residues += length

    if modality_targets != EXPECTED_TARGETS:
        raise RuntimeError(
            f"{modality}: target count "
            f"{modality_targets}; expected "
            f"{EXPECTED_TARGETS}."
        )

    if modality_residues != EXPECTED_RESIDUES:
        raise RuntimeError(
            f"{modality}: residue count "
            f"{modality_residues}; expected "
            f"{EXPECTED_RESIDUES}."
        )

    total_targets_by_modality[
        modality
    ] = modality_targets

    total_residues_by_modality[
        modality
    ] = modality_residues


# ============================================================
# DATA FRAMES
# ============================================================

label_df = pd.DataFrame(
    label_rows
)

protein_df = pd.DataFrame(
    protein_rows
)

expected_label_rows = (
    2 * EXPECTED_TARGETS
)

expected_protein_rows = (
    2 * EXPECTED_POPULATION_SIZE
)

if len(label_df) != expected_label_rows:
    raise RuntimeError(
        "Unexpected label-level row count."
    )

if len(protein_df) != expected_protein_rows:
    raise RuntimeError(
        "Unexpected protein-level row count."
    )


# ============================================================
# PRIMARY SUMMARY
#
# Primary statistical unit = protein.
# Labels are averaged within protein first.
# Every protein then receives equal weight.
# ============================================================

summary_rows = []

for modality in [
    "sequence",
    "structure",
]:

    subset = (
        protein_df.loc[
            protein_df["modality"]
            == modality
        ]
        .copy()
    )

    values = subset[
        "mean_faithfulness_enrichment"
    ].to_numpy(
        dtype=np.float64
    )

    if len(values) != EXPECTED_POPULATION_SIZE:
        raise RuntimeError(
            f"{modality}: unexpected "
            "protein-level count."
        )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            values,
            modality,
        )
    )

    summary_rows.append({
        "modality":
            modality,

        "n_proteins":
            int(len(values)),

        "n_protein_label_targets":
            int(
                total_targets_by_modality[
                    modality
                ]
            ),

        "mean_protein_faithfulness_enrichment":
            float(values.mean()),

        "median_protein_faithfulness_enrichment":
            float(np.median(values)),

        "fraction_proteins_positive":
            float(
                np.mean(values > 0.0)
            ),

        "bootstrap_mean_ci95_low":
            ci_low,

        "bootstrap_mean_ci95_high":
            ci_high,

        "bootstrap_replicates":
            N_BOOTSTRAP,

        "bootstrap_base_seed":
            BASE_SEED,

        "bootstrap_stream_seed":
            int(
                bootstrap_seed(
                    modality
                )
            ),
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# WRITE OUTPUTS
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

label_df.to_csv(
    LABEL_OUT,
    index=False,
)

protein_df.to_csv(
    PROTEIN_OUT,
    index=False,
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)


run_info = {
    "analysis":
        "deletion_faithfulness_aggregate_v1",

    "status":
        "pre-specified aggregation",

    "population_path":
        str(POPULATION_PATH),

    "population_sha256":
        sha256_file(
            POPULATION_PATH
        ),

    "population_size":
        EXPECTED_POPULATION_SIZE,

    "targets_per_modality":
        EXPECTED_TARGETS,

    "residues_per_modality":
        EXPECTED_RESIDUES,

    "modalities": [
        "sequence",
        "structure",
    ],

    "primary_statistical_unit":
        "protein",

    "within_protein_aggregation":
        "mean across all true-positive "
        "frozen GO-MF labels",

    "protein_weighting":
        "equal",

    "primary_quantity":
        "top_deletion_drop minus "
        "random_mean_drop",

    "negative_values_retained":
        True,

    "bootstrap":
        {
            "replicates":
                N_BOOTSTRAP,

            "base_seed":
                BASE_SEED,

            "resampling_unit":
                "whole protein",

            "confidence_interval":
                "2.5th and 97.5th percentiles "
                "of bootstrap mean",
        },
}


with open(
    RUN_INFO_OUT,
    "w",
) as f:

    json.dump(
        run_info,
        f,
        indent=2,
        sort_keys=True,
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 78)
print("E10 DELETION FAITHFULNESS — PRIMARY SUMMARY")
print("=" * 78)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Label-level rows:", len(label_df))
print("Protein-level rows:", len(protein_df))
print("Aggregation complete.")
print("=" * 78)
