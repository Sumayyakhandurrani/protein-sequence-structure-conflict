from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

ROOT = Path("data/xai/functional_sites_v1")

SITES_PATH = ROOT / "functional_sites_primary.csv"
POPULATION_PATH = ROOT / "xai_attribution_eligible_ids.txt"

SEQ_DIR = ROOT / "sequence_attributions_v1/per_protein"
STRUCT_DIR = ROOT / "structure_attributions_v1/per_protein"

OUT_DIR = ROOT / "biological_enrichment_v1"

N_RANDOM = 1000
BASE_SEED = 42
N_BOOTSTRAP = 1000

SEQ_TEMPERATURE = 1.106977105140686
STRUCT_TEMPERATURE = 1.0

CATEGORIES = [
    "ACT_SITE",
    "BINDING",
    "UNION",
]

MODALITIES = [
    "sequence",
    "structure",
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


def deterministic_seed(uid, category):
    """
    Deterministic protein/category-specific seed derived from
    frozen base seed + UniProt ID + category.

    Does not depend on iteration order or Python hash().
    """
    key = f"{BASE_SEED}|{uid}|{category}".encode("utf-8")

    digest = hashlib.sha256(key).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# LOAD FROZEN POPULATION + SITES
# ============================================================

population = read_ids(POPULATION_PATH)
population_set = set(population)

if len(population) != 2447:
    raise RuntimeError(
        f"Expected 2447 proteins, found {len(population)}"
    )

sites = pd.read_csv(SITES_PATH)

sites = sites[
    sites["uniprot_id"].isin(population_set)
].copy()


# ============================================================
# BUILD FROZEN CURATED POSITION SETS
#
# UniProt coordinates = 1-based.
# Attribution arrays = 0-based.
# ============================================================

curated = {
    uid: {
        "ACT_SITE": set(),
        "BINDING": set(),
        "UNION": set(),
    }
    for uid in population
}


for row in sites.itertuples(index=False):

    uid = row.uniprot_id
    start = int(row.start)
    end = int(row.end)

    if start > end:
        raise RuntimeError(
            f"{uid}: invalid range {start}..{end}"
        )

    positions_zero_based = range(
        start - 1,
        end,
    )

    curated[
        uid
    ][
        row.feature_type
    ].update(
        positions_zero_based
    )


for uid in population:

    curated[
        uid
    ][
        "UNION"
    ] = (
        curated[
            uid
        ][
            "ACT_SITE"
        ]
        |
        curated[
            uid
        ][
            "BINDING"
        ]
    )


# ============================================================
# OUTPUT STORAGE
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

label_rows = []
protein_rows = []

random_exclusions = []


# ============================================================
# PRIMARY MATCHED-RANDOM ANALYSIS
# ============================================================

for protein_number, uid in enumerate(
    population,
    start=1,
):

    seq_path = SEQ_DIR / f"{uid}.npz"
    struct_path = STRUCT_DIR / f"{uid}.npz"

    if not seq_path.exists():
        raise RuntimeError(
            f"Missing sequence attribution: {uid}"
        )

    if not struct_path.exists():
        raise RuntimeError(
            f"Missing structure attribution: {uid}"
        )


    # --------------------------------------------------------
    # Sequence attribution
    # --------------------------------------------------------

    with np.load(
        seq_path,
        allow_pickle=False,
    ) as z:

        seq_labels = (
            z["positive_label_indices"]
            .astype(np.int64)
        )

        seq_attr_raw = (
            z["attribution_logit_drop"]
            .astype(np.float64)
        )

        if "length" in z.files:
            seq_length = int(
                z["length"].item()
            )
        elif "sequence_length" in z.files:
            seq_length = int(
                z["sequence_length"].item()
            )
        else:
            seq_length = int(
                seq_attr_raw.shape[1]
            )


    # Frozen calibrated sequence attribution.
    seq_attr = (
        seq_attr_raw
        / SEQ_TEMPERATURE
    )


    # --------------------------------------------------------
    # Structure attribution
    # --------------------------------------------------------

    with np.load(
        struct_path,
        allow_pickle=False,
    ) as z:

        struct_labels = (
            z["positive_label_indices"]
            .astype(np.int64)
        )

        struct_attr = (
            z["attribution_logit_drop"]
            .astype(np.float64)
            / STRUCT_TEMPERATURE
        )

        struct_length = int(
            z["length"].item()
        )


    # --------------------------------------------------------
    # Cross-modal QC
    # --------------------------------------------------------

    if seq_length != struct_length:
        raise RuntimeError(
            f"{uid}: sequence/structure length mismatch"
        )

    if not np.array_equal(
        seq_labels,
        struct_labels,
    ):
        raise RuntimeError(
            f"{uid}: target labels differ"
        )

    if seq_attr.shape != struct_attr.shape:
        raise RuntimeError(
            f"{uid}: attribution shape mismatch "
            f"{seq_attr.shape} vs {struct_attr.shape}"
        )

    if seq_attr.shape[1] != seq_length:
        raise RuntimeError(
            f"{uid}: attribution/residue mismatch"
        )


    L = seq_length
    labels = seq_labels


    modality_arrays = {
        "sequence": seq_attr,
        "structure": struct_attr,
    }


    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    for category in CATEGORIES:

        curated_positions = np.array(
            sorted(curated[uid][category]),
            dtype=np.int64,
        )

        k = len(curated_positions)

        if k == 0:
            continue


        if (
            curated_positions.min() < 0
            or curated_positions.max() >= L
        ):
            raise RuntimeError(
                f"{uid}: {category} "
                f"curated coordinate out of bounds"
            )


        all_positions = np.arange(
            L,
            dtype=np.int64,
        )

        candidate_mask = np.ones(
            L,
            dtype=bool,
        )

        candidate_mask[
            curated_positions
        ] = False

        candidate_positions = all_positions[
            candidate_mask
        ]


        if len(candidate_positions) < k:

            random_exclusions.append({
                "uniprot_id": uid,
                "category": category,
                "length": L,
                "k": k,
                "candidate_count":
                    len(candidate_positions),
            })

            continue


        # ----------------------------------------------------
        # Frozen deterministic matched random sets.
        # Same sets reused across labels and modalities.
        # ----------------------------------------------------

        rng = np.random.default_rng(
            deterministic_seed(
                uid,
                category,
            )
        )


        random_sets = np.empty(
            (N_RANDOM, k),
            dtype=np.int64,
        )


        for r in range(N_RANDOM):

            random_sets[r] = rng.choice(
                candidate_positions,
                size=k,
                replace=False,
            )


        # ----------------------------------------------------
        # Analyze modalities separately.
        # ----------------------------------------------------

        for modality in MODALITIES:

            attr = modality_arrays[
                modality
            ]


            # Shape:
            # labels x curated residues
            curated_values = attr[
                :,
                curated_positions,
            ]

            curated_means = (
                curated_values.mean(
                    axis=1
                )
            )


            # Compute random baseline label by label.
            #
            # We intentionally avoid constructing a huge
            # labels x 1000 x k tensor.
            random_mean_sum = np.zeros(
                len(labels),
                dtype=np.float64,
            )


            for r in range(N_RANDOM):

                random_mean_sum += (
                    attr[
                        :,
                        random_sets[r],
                    ]
                    .mean(axis=1)
                )


            random_means = (
                random_mean_sum
                / N_RANDOM
            )


            enrichment = (
                curated_means
                - random_means
            )


            # ------------------------------------------------
            # Label-level records
            # ------------------------------------------------

            for j, label_index in enumerate(
                labels
            ):

                label_rows.append({
                    "uniprot_id":
                        uid,

                    "modality":
                        modality,

                    "category":
                        category,

                    "label_index":
                        int(label_index),

                    "protein_length":
                        L,

                    "n_curated":
                        k,

                    "curated_mean":
                        float(
                            curated_means[j]
                        ),

                    "matched_random_mean":
                        float(
                            random_means[j]
                        ),

                    "enrichment":
                        float(
                            enrichment[j]
                        ),
                })


            # ------------------------------------------------
            # Equal-protein aggregation:
            # average label enrichments inside protein.
            # ------------------------------------------------

            protein_enrichment = float(
                enrichment.mean()
            )


            protein_rows.append({
                "uniprot_id":
                    uid,

                "modality":
                    modality,

                "category":
                    category,

                "protein_length":
                    L,

                "n_positive_labels":
                    len(labels),

                "n_curated":
                    k,

                "protein_enrichment":
                    protein_enrichment,
            })


    if (
        protein_number % 100 == 0
        or protein_number == len(population)
    ):

        print(
            f"Processed "
            f"{protein_number}/"
            f"{len(population)} proteins"
        )


# ============================================================
# SAVE LABEL + PROTEIN LEVEL RESULTS
# ============================================================

label_df = pd.DataFrame(
    label_rows
)

protein_df = pd.DataFrame(
    protein_rows
)

exclusion_df = pd.DataFrame(
    random_exclusions
)


label_path = (
    OUT_DIR
    / "label_level_enrichment.csv"
)

protein_path = (
    OUT_DIR
    / "protein_level_enrichment.csv"
)

exclusion_path = (
    OUT_DIR
    / "matched_random_exclusions.csv"
)


label_df.to_csv(
    label_path,
    index=False,
)

protein_df.to_csv(
    protein_path,
    index=False,
)

exclusion_df.to_csv(
    exclusion_path,
    index=False,
)


# ============================================================
# PRIMARY POPULATION SUMMARY
#
# Equal protein weighting.
# Whole-protein bootstrap.
# ============================================================

summary_rows = []


for modality in MODALITIES:

    for category in CATEGORIES:

        x = protein_df[
            (
                protein_df[
                    "modality"
                ] == modality
            )
            &
            (
                protein_df[
                    "category"
                ] == category
            )
        ][
            "protein_enrichment"
        ].to_numpy(
            dtype=np.float64
        )


        if len(x) == 0:
            continue


        mean_value = float(
            np.mean(x)
        )

        median_value = float(
            np.median(x)
        )

        fraction_positive = float(
            np.mean(x > 0)
        )


        # Deterministic bootstrap stream.
        bootstrap_key = (
            f"{BASE_SEED}|bootstrap|"
            f"{modality}|{category}"
        ).encode("utf-8")

        bootstrap_seed = int.from_bytes(
            hashlib.sha256(
                bootstrap_key
            ).digest()[:8],
            byteorder="little",
            signed=False,
        )

        rng = np.random.default_rng(
            bootstrap_seed
        )


        bootstrap_means = np.empty(
            N_BOOTSTRAP,
            dtype=np.float64,
        )


        n = len(x)


        for b in range(
            N_BOOTSTRAP
        ):

            sampled_indices = rng.integers(
                0,
                n,
                size=n,
            )

            bootstrap_means[b] = (
                x[
                    sampled_indices
                ].mean()
            )


        ci_low = float(
            np.quantile(
                bootstrap_means,
                0.025,
            )
        )

        ci_high = float(
            np.quantile(
                bootstrap_means,
                0.975,
            )
        )


        summary_rows.append({
            "modality":
                modality,

            "category":
                category,

            "n_proteins":
                n,

            "mean_protein_enrichment":
                mean_value,

            "median_protein_enrichment":
                median_value,

            "fraction_positive":
                fraction_positive,

            "bootstrap_ci95_low":
                ci_low,

            "bootstrap_ci95_high":
                ci_high,
        })


summary_df = pd.DataFrame(
    summary_rows
)


summary_path = (
    OUT_DIR
    / "primary_enrichment_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# RUN METADATA
# ============================================================

run_info = {
    "population_size":
        len(population),

    "n_random_sets":
        N_RANDOM,

    "random_seed":
        BASE_SEED,

    "n_bootstrap":
        N_BOOTSTRAP,

    "sequence_temperature":
        SEQ_TEMPERATURE,

    "structure_temperature":
        STRUCT_TEMPERATURE,

    "categories":
        CATEGORIES,

    "modalities":
        MODALITIES,

    "coordinate_mapping":
        "inclusive_range_expansion",

    "primary_attribution":
        "signed_calibrated_target_logit_drop",

    "aggregation":
        "label_mean_within_protein_then_equal_protein_mean",
}


with open(
    OUT_DIR / "run_info.json",
    "w",
) as f:

    json.dump(
        run_info,
        f,
        indent=2,
    )


# ============================================================
# PRINT PRIMARY RESULTS
# ============================================================

print()
print("=" * 90)
print("E10 PRIMARY BIOLOGICAL ENRICHMENT")
print("=" * 90)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print(
    "Matched-random exclusions:",
    len(exclusion_df),
)

print()
print("Saved:")
print(" ", label_path)
print(" ", protein_path)
print(" ", summary_path)
print(" ", exclusion_path)

print("=" * 90)