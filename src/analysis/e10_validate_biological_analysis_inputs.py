from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "data/xai/functional_sites_v1"
)

SITES_PATH = (
    ROOT / "functional_sites_primary.csv"
)

POPULATION_PATH = (
    ROOT / "xai_attribution_eligible_ids.txt"
)

SEQ_DIR = (
    ROOT / "sequence_attributions_v1/per_protein"
)

STRUCT_DIR = (
    ROOT / "structure_attributions_v1/per_protein"
)


def read_ids(path):
    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


population = read_ids(
    POPULATION_PATH
)

population_set = set(
    population
)

sites = pd.read_csv(
    SITES_PATH
)


print("=" * 78)
print("E10 BIOLOGICAL ANALYSIS — INPUT VALIDATION")
print("=" * 78)

print("Frozen attribution population :", len(population))
print("Functional-site rows          :", len(sites))


# ------------------------------------------------------------
# BASIC SITE TABLE QC
# ------------------------------------------------------------

required_columns = {
    "uniprot_id",
    "feature_type",
    "start",
    "end",
    "coordinate_valid",
    "benchmark_length",
    "coordinate_in_bounds",
    "exact_sequence_match",
    "retain_primary",
}

missing_columns = (
    required_columns
    - set(sites.columns)
)

if missing_columns:
    raise RuntimeError(
        f"Missing site columns: {missing_columns}"
    )


bad_types = (
    set(
        sites["feature_type"].unique()
    )
    - {"ACT_SITE", "BINDING"}
)

if bad_types:
    raise RuntimeError(
        f"Unexpected feature types: {bad_types}"
    )


if not sites[
    "coordinate_valid"
].all():
    raise RuntimeError(
        "Non-valid coordinates present."
    )


if not sites[
    "coordinate_in_bounds"
].all():
    raise RuntimeError(
        "Out-of-bounds coordinates present."
    )


if not sites[
    "exact_sequence_match"
].all():
    raise RuntimeError(
        "Non-exact sequence matches present."
    )


if not sites[
    "retain_primary"
].all():
    raise RuntimeError(
        "Non-primary site rows present."
    )



# ------------------------------------------------------------
# RESTRICT TO FROZEN ATTRIBUTION POPULATION
# ------------------------------------------------------------

eligible_sites = sites[
    sites["uniprot_id"].isin(
        population_set
    )
].copy()


site_proteins = set(
    eligible_sites[
        "uniprot_id"
    ].unique()
)


print(
    "Site rows in XAI population  :",
    len(eligible_sites),
)

print(
    "Proteins with >=1 site       :",
    len(site_proteins),
)


# ------------------------------------------------------------
# BUILD UNIQUE POSITION SETS
# ------------------------------------------------------------

categories = {
    uid: {
        "ACT_SITE": set(),
        "BINDING": set(),
        "UNION": set(),
    }
    for uid in population
}


for row in eligible_sites.itertuples(
    index=False
):

    uid = row.uniprot_id
    start = int(row.start)
    end = int(row.end)
    feature = row.feature_type

    if start > end:
        raise RuntimeError(
            f"{uid}: invalid coordinate "
            f"{start}..{end}"
        )

    # Frozen FUNCTIONAL_SITE_POSITION_MAPPING_V1:
    # expand UniProt range inclusively.
    positions = range(
        start,
        end + 1,
    )

    categories[
        uid
    ][
        feature
    ].update(
        positions
    )


for uid in population:

    categories[
        uid
    ][
        "UNION"
    ] = (
        categories[
            uid
        ][
            "ACT_SITE"
        ]
        |
        categories[
            uid
        ][
            "BINDING"
        ]
    )


# ------------------------------------------------------------
# ATTRIBUTION FILE + SHAPE + POSITION VALIDATION
# ------------------------------------------------------------

errors = []

category_protein_counts = {
    "ACT_SITE": 0,
    "BINDING": 0,
    "UNION": 0,
}

category_position_counts = {
    "ACT_SITE": 0,
    "BINDING": 0,
    "UNION": 0,
}

insufficient_random_pool = {
    "ACT_SITE": 0,
    "BINDING": 0,
    "UNION": 0,
}


for uid in population:

    seq_path = (
        SEQ_DIR / f"{uid}.npz"
    )

    struct_path = (
        STRUCT_DIR / f"{uid}.npz"
    )


    if not seq_path.exists():
        errors.append(
            f"{uid}: missing sequence attribution"
        )
        continue


    if not struct_path.exists():
        errors.append(
            f"{uid}: missing structure attribution"
        )
        continue


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
        )

        # Sequence production may use either key.
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
        )

        struct_length = int(
            z["length"].item()
        )


    if seq_length != struct_length:
        errors.append(
            f"{uid}: sequence/structure length mismatch "
            f"{seq_length} vs {struct_length}"
        )


    if seq_attr.shape[1] != seq_length:
        errors.append(
            f"{uid}: sequence attribution length mismatch"
        )


    if struct_attr.shape[1] != struct_length:
        errors.append(
            f"{uid}: structure attribution length mismatch"
        )


    if not np.array_equal(
        seq_labels,
        struct_labels,
    ):
        errors.append(
            f"{uid}: sequence/structure target labels differ"
        )


    if seq_attr.shape[0] != len(
        seq_labels
    ):
        errors.append(
            f"{uid}: sequence target dimension mismatch"
        )


    if struct_attr.shape[0] != len(
        struct_labels
    ):
        errors.append(
            f"{uid}: structure target dimension mismatch"
        )


    for category in [
        "ACT_SITE",
        "BINDING",
        "UNION",
    ]:

        positions = categories[
            uid
        ][
            category
        ]


        if not positions:
            continue


        category_protein_counts[
            category
        ] += 1

        category_position_counts[
            category
        ] += len(
            positions
        )


        bad_positions = [
            p
            for p in positions
            if p < 1
            or p > seq_length
        ]


        if bad_positions:
            errors.append(
                f"{uid}: {category} "
                f"out-of-range positions "
                f"{bad_positions[:5]}"
            )


        k = len(
            positions
        )

        non_curated = (
            seq_length - k
        )


        if non_curated < k:
            insufficient_random_pool[
                category
            ] += 1


print()
print("UNIQUE CURATED POSITION SUMMARY")
print("-" * 78)

for category in [
    "ACT_SITE",
    "BINDING",
    "UNION",
]:

    print(
        f"{category:10s} "
        f"proteins={category_protein_counts[category]:4d} "
        f"positions={category_position_counts[category]:6d} "
        f"insufficient_random_pool="
        f"{insufficient_random_pool[category]}"
    )


print()
print("Input validation errors       :", len(errors))

if errors:

    print()
    print("First errors:")

    for error in errors[:20]:
        print("  -", error)


passed = (
    len(population) == 2447
    and len(errors) == 0
)


print()
print("=" * 78)

print(
    "BIOLOGICAL ANALYSIS INPUT VALIDATION:",
    "PASS"
    if passed
    else "FAIL",
)

print("=" * 78)
