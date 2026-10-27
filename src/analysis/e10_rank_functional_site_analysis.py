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

OUT_DIR = ROOT / "rank_analysis_v1"

SEQ_TEMPERATURE = 1.106977105140686

BASE_SEED = 42
N_BOOTSTRAP = 1000

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


def bootstrap_seed(modality, category):
    key = (
        f"{BASE_SEED}|rank_bootstrap|"
        f"{modality}|{category}"
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.sha256(key).digest()[:8],
        byteorder="little",
        signed=False,
    )


# ============================================================
# LOAD POPULATION + FUNCTIONAL SITES
# ============================================================

population = read_ids(
    POPULATION_PATH
)

if len(population) != 2447:
    raise RuntimeError(
        f"Expected 2447 proteins, found {len(population)}"
    )

population_set = set(
    population
)

sites = pd.read_csv(
    SITES_PATH
)

sites = sites[
    sites["uniprot_id"].isin(
        population_set
    )
].copy()


# ============================================================
# BUILD CURATED POSITION SETS
#
# UniProt = 1-based
# attribution arrays = 0-based
# ranges expanded inclusively
# ============================================================

curated = {
    uid: {
        "ACT_SITE": set(),
        "BINDING": set(),
        "UNION": set(),
    }
    for uid in population
}


for row in sites.itertuples(
    index=False
):

    uid = row.uniprot_id
    start = int(row.start)
    end = int(row.end)

    if start > end:
        raise RuntimeError(
            f"{uid}: invalid range {start}..{end}"
        )

    positions = range(
        start - 1,
        end,
    )

    curated[
        uid
    ][
        row.feature_type
    ].update(
        positions
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
# ANALYSIS
# ============================================================

label_rows = []
protein_rows = []


for protein_number, uid in enumerate(
    population,
    start=1,
):

    seq_path = (
        SEQ_DIR / f"{uid}.npz"
    )

    struct_path = (
        STRUCT_DIR / f"{uid}.npz"
    )


    if not seq_path.exists():
        raise RuntimeError(
            f"Missing sequence attribution: {uid}"
        )

    if not struct_path.exists():
        raise RuntimeError(
            f"Missing structure attribution: {uid}"
        )


    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    with np.load(
        seq_path,
        allow_pickle=False,
    ) as z:

        seq_labels = (
            z["positive_label_indices"]
            .astype(np.int64)
        )

        seq_attr = (
            z["attribution_logit_drop"]
            .astype(np.float64)
            / SEQ_TEMPERATURE
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
                seq_attr.shape[1]
            )


    # --------------------------------------------------------
    # Structure
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
        )

        struct_length = int(
            z["length"].item()
        )


    # --------------------------------------------------------
    # QC
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
            f"{uid}: positive labels differ"
        )

    if seq_attr.shape != struct_attr.shape:
        raise RuntimeError(
            f"{uid}: attribution shapes differ"
        )

    if seq_attr.shape[1] != seq_length:
        raise RuntimeError(
            f"{uid}: attribution length mismatch"
        )


    L = seq_length
    labels = seq_labels

    residue_indices = np.arange(
        L,
        dtype=np.int64,
    )


    modality_arrays = {
        "sequence": seq_attr,
        "structure": struct_attr,
    }


    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    for category in CATEGORIES:

        curated_positions = np.array(
            sorted(
                curated[
                    uid
                ][
                    category
                ]
            ),
            dtype=np.int64,
        )

        k = len(
            curated_positions
        )

        if k == 0:
            continue


        if (
            curated_positions.min() < 0
            or curated_positions.max() >= L
        ):
            raise RuntimeError(
                f"{uid}: {category} coordinate out of bounds"
            )


        curated_set = set(
            curated_positions.tolist()
        )


        # Expected recall of random top-k ranking.
        chance_recall = (
            k / L
        )


        for modality in MODALITIES:

            attr = modality_arrays[
                modality
            ]

            label_recalls = []
            label_adjusted = []


            for j, label_index in enumerate(
                labels
            ):

                scores = attr[
                    j
                ]


                # np.lexsort:
                # last key is primary.
                #
                # Primary:
                #   -scores -> descending score
                #
                # Secondary:
                #   residue index ascending
                #
                # Thus ties favor lower UniProt position.
                ranking = np.lexsort(
                    (
                        residue_indices,
                        -scores,
                    )
                )


                top_k = ranking[
                    :k
                ]


                overlap_count = sum(
                    int(pos) in curated_set
                    for pos in top_k
                )


                top_k_recall = (
                    overlap_count / k
                )

                chance_adjusted = (
                    top_k_recall
                    - chance_recall
                )


                label_recalls.append(
                    top_k_recall
                )

                label_adjusted.append(
                    chance_adjusted
                )


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

                    "overlap_count":
                        int(overlap_count),

                    "top_k_recall":
                        float(top_k_recall),

                    "chance_recall":
                        float(chance_recall),

                    "chance_adjusted_recall":
                        float(chance_adjusted),
                })


            # Equal-label averaging inside protein.
            protein_recall = float(
                np.mean(
                    label_recalls
                )
            )

            protein_adjusted = float(
                np.mean(
                    label_adjusted
                )
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

                "chance_recall":
                    float(chance_recall),

                "protein_top_k_recall":
                    protein_recall,

                "protein_chance_adjusted_recall":
                    protein_adjusted,
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
# SAVE LABEL + PROTEIN LEVEL
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

label_df = pd.DataFrame(
    label_rows
)

protein_df = pd.DataFrame(
    protein_rows
)


label_path = (
    OUT_DIR
    / "label_level_rank_analysis.csv"
)

protein_path = (
    OUT_DIR
    / "protein_level_rank_analysis.csv"
)


label_df.to_csv(
    label_path,
    index=False,
)

protein_df.to_csv(
    protein_path,
    index=False,
)


# ============================================================
# POPULATION SUMMARY + WHOLE-PROTEIN BOOTSTRAP
# ============================================================

summary_rows = []


for modality in MODALITIES:

    for category in CATEGORIES:

        sub = protein_df[
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
        ].copy()


        if len(sub) == 0:
            continue


        recall = sub[
            "protein_top_k_recall"
        ].to_numpy(
            dtype=np.float64
        )

        chance = sub[
            "chance_recall"
        ].to_numpy(
            dtype=np.float64
        )

        adjusted = sub[
            "protein_chance_adjusted_recall"
        ].to_numpy(
            dtype=np.float64
        )


        rng = np.random.default_rng(
            bootstrap_seed(
                modality,
                category,
            )
        )


        n = len(
            adjusted
        )

        bootstrap_means = np.empty(
            N_BOOTSTRAP,
            dtype=np.float64,
        )


        for b in range(
            N_BOOTSTRAP
        ):

            idx = rng.integers(
                0,
                n,
                size=n,
            )

            bootstrap_means[b] = (
                adjusted[
                    idx
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

            "mean_top_k_recall":
                float(
                    np.mean(recall)
                ),

            "median_top_k_recall":
                float(
                    np.median(recall)
                ),

            "mean_chance_recall":
                float(
                    np.mean(chance)
                ),

            "mean_chance_adjusted_recall":
                float(
                    np.mean(adjusted)
                ),

            "median_chance_adjusted_recall":
                float(
                    np.median(adjusted)
                ),

            "fraction_positive_adjusted":
                float(
                    np.mean(
                        adjusted > 0
                    )
                ),

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
    / "rank_analysis_summary.csv"
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

    "sequence_temperature":
        SEQ_TEMPERATURE,

    "structure_temperature":
        1.0,

    "ranking":
        "signed attribution descending",

    "tie_break":
        "lower residue position first",

    "top_k":
        "number of unique curated residues",

    "chance_recall":
        "k / protein_length",

    "aggregation":
        "label mean within protein, then equal protein",

    "bootstrap":
        {
            "n": N_BOOTSTRAP,
            "base_seed": BASE_SEED,
        },

    "categories":
        CATEGORIES,

    "modalities":
        MODALITIES,
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
# PRINT RESULTS
# ============================================================

print()
print("=" * 110)
print("E10 RANK-BASED FUNCTIONAL-SITE VALIDATION")
print("=" * 110)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(" ", label_path)
print(" ", protein_path)
print(" ", summary_path)

print("=" * 110)