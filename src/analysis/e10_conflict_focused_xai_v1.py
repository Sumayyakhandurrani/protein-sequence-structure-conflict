from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# FROZEN CONSTANTS
# ============================================================

BASE_SEED = 42
N_BOOTSTRAP = 1000

EXPECTED_XAI_PROTEINS = 2447
EXPECTED_PAIRED_TEST = 4731
EXPECTED_LABELS = 494

EXPECTED_PAIRED_SHA = (
    "f30654de5cee02f547db48e679f4711436edfbccc3f4ad4e22f6fc4153350704"
)

EXPECTED_PROTOCOL_SHA = (
    "e9b00e7f93d27662a7dd227d86912394e0e0cacdd062143d8878db4c8fdd4d1a"
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path("data/xai/functional_sites_v1")

PROTOCOL = ROOT / "CONFLICT_FOCUSED_XAI_PROTOCOL_V1.txt"

PAIRED_IDS = Path(
    "data/final_evaluation/final_test_primary_v1/"
    "paired_test_uniprot_ids.txt"
)

JS_PATH = Path(
    "data/final_evaluation/final_test_primary_v1/"
    "bernoulli_js.npy"
)

LABELS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/labels/test_labels.npy"
)

XAI_IDS_PATH = ROOT / "xai_attribution_eligible_ids.txt"

ENRICHMENT_PATH = (
    ROOT
    / "biological_enrichment_v1"
    / "protein_level_enrichment.csv"
)

RANK_PATH = (
    ROOT
    / "rank_analysis_v1"
    / "protein_level_rank_analysis.csv"
)

FAITH_PATH = (
    ROOT
    / "deletion_faithfulness_v1"
    / "aggregate_v1"
    / "protein_level_faithfulness.csv"
)

OUT_DIR = ROOT / "conflict_focused_xai_v1"

PROTEIN_CONFLICT_OUT = OUT_DIR / "protein_level_conflict.csv"
ASSOCIATIONS_OUT = OUT_DIR / "primary_associations.csv"
RUN_INFO_OUT = OUT_DIR / "run_info.json"


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_ids(path):
    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def bootstrap_seed(modality, outcome, category):
    material = (
        f"{BASE_SEED}|{modality}|{outcome}|{category}|"
        "CONFLICT_FOCUSED_XAI_V1"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()

    return int.from_bytes(
        digest[:8],
        "little",
    ) % (2**32)


def bootstrap_spearman_ci(x, y, seed):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)
    rng = np.random.default_rng(seed)

    values = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        xb = x[idx]
        yb = y[idx]

        if np.ptp(xb) == 0 or np.ptp(yb) == 0:
            continue

        rho = spearmanr(
            xb,
            yb,
        ).statistic

        if np.isfinite(rho):
            values.append(float(rho))

    if not values:
        raise RuntimeError(
            "No valid bootstrap Spearman replicates."
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    low, high = np.quantile(
        values,
        [0.025, 0.975],
    )

    return (
        float(low),
        float(high),
        int(len(values)),
    )


def bh_fdr(pvalues):
    p = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    if np.any(~np.isfinite(p)):
        raise ValueError("Non-finite p-value.")

    m = len(p)

    order = np.argsort(
        p,
        kind="mergesort",
    )

    ranked = p[order]

    adjusted = (
        ranked
        * m
        / np.arange(1, m + 1)
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(
        adjusted,
        1.0,
    )

    q = np.empty(
        m,
        dtype=np.float64,
    )

    q[order] = adjusted

    return q


# ============================================================
# VERIFY FROZEN INPUTS
# ============================================================

if sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
    raise RuntimeError(
        "Conflict-focused XAI protocol SHA mismatch."
    )

if sha256_file(PAIRED_IDS) != EXPECTED_PAIRED_SHA:
    raise RuntimeError(
        "Paired TEST ID SHA mismatch."
    )

paired_ids = read_ids(PAIRED_IDS)
xai_ids = read_ids(XAI_IDS_PATH)

if len(paired_ids) != EXPECTED_PAIRED_TEST:
    raise RuntimeError(
        f"Expected {EXPECTED_PAIRED_TEST} paired TEST IDs, "
        f"found {len(paired_ids)}."
    )

if len(set(paired_ids)) != EXPECTED_PAIRED_TEST:
    raise RuntimeError(
        "Duplicate paired TEST IDs."
    )

if len(xai_ids) != EXPECTED_XAI_PROTEINS:
    raise RuntimeError(
        f"Expected {EXPECTED_XAI_PROTEINS} XAI IDs, "
        f"found {len(xai_ids)}."
    )

if len(set(xai_ids)) != EXPECTED_XAI_PROTEINS:
    raise RuntimeError(
        "Duplicate XAI IDs."
    )

paired_index = {
    uid: i
    for i, uid in enumerate(paired_ids)
}

missing_xai = [
    uid
    for uid in xai_ids
    if uid not in paired_index
]

if missing_xai:
    raise RuntimeError(
        f"{len(missing_xai)} XAI proteins missing from paired TEST."
    )


# ============================================================
# LOAD CONFLICT + FROZEN TEST LABELS
# ============================================================

js = np.load(
    JS_PATH,
    mmap_mode="r",
)

labels = np.load(
    LABELS_PATH,
    mmap_mode="r",
)

if js.shape != (
    EXPECTED_PAIRED_TEST,
    EXPECTED_LABELS,
):
    raise RuntimeError(
        f"Unexpected JS shape: {js.shape}"
    )

if labels.ndim != 2:
    raise RuntimeError(
        f"Unexpected TEST label dimensions: {labels.shape}"
    )

if labels.shape[1] != EXPECTED_LABELS:
    raise RuntimeError(
        f"Expected {EXPECTED_LABELS} TEST labels, "
        f"found {labels.shape[1]}."
    )


# ============================================================
# MAP FROZEN TEST LABEL ROWS
# ============================================================

# The benchmark TEST label matrix follows the full frozen Graph-V2
# TEST ID order. Recover that order from the standard split files.

candidate_test_id_paths = [
    Path(
        "data/splits/mmseqs30_graph_v2/"
        "labels/test_uniprot_ids.txt"
    ),
]

test_id_path = None

for candidate in candidate_test_id_paths:
    if candidate.exists():
        test_id_path = candidate
        break

if test_id_path is None:
    raise RuntimeError(
        "Could not locate frozen Graph-V2 TEST ID ordering."
    )

test_ids = read_ids(test_id_path)

if len(test_ids) != labels.shape[0]:
    raise RuntimeError(
        "TEST ID count does not match TEST label rows."
    )

if len(set(test_ids)) != len(test_ids):
    raise RuntimeError(
        "Duplicate IDs in frozen TEST ID ordering."
    )

test_label_index = {
    uid: i
    for i, uid in enumerate(test_ids)
}


# ============================================================
# COMPUTE PROTEIN-LEVEL CONFLICT
# ============================================================

protein_conflict_rows = []

for uid in xai_ids:
    paired_row = paired_index[uid]

    if uid not in test_label_index:
        raise RuntimeError(
            f"{uid} missing from frozen TEST label ordering."
        )

    label_row = test_label_index[uid]

    y = np.asarray(
        labels[label_row],
    )

    positive = np.flatnonzero(
        y > 0
    )

    if len(positive) == 0:
        raise RuntimeError(
            f"{uid} has zero positive frozen labels."
        )

    values = np.asarray(
        js[paired_row, positive],
        dtype=np.float64,
    )

    if len(values) != len(positive):
        raise RuntimeError(
            f"Conflict extraction mismatch for {uid}."
        )

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"Non-finite conflict values for {uid}."
        )

    protein_conflict_rows.append(
        {
            "uniprot_id": uid,
            "n_positive_labels": int(len(positive)),
            "protein_conflict_mean_bernoulli_js": float(
                np.mean(values)
            ),
        }
    )


conflict_df = pd.DataFrame(
    protein_conflict_rows
)

if len(conflict_df) != EXPECTED_XAI_PROTEINS:
    raise RuntimeError(
        "Unexpected protein-conflict row count."
    )

if conflict_df["uniprot_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate protein-conflict IDs."
    )


# ============================================================
# LOAD EXISTING FROZEN XAI OUTCOMES
# ============================================================

enrichment = pd.read_csv(
    ENRICHMENT_PATH
)

rank = pd.read_csv(
    RANK_PATH
)

faith = pd.read_csv(
    FAITH_PATH
)


# ============================================================
# BUILD PRE-SPECIFIED ASSOCIATIONS
# ============================================================

analysis_specs = []

for modality in [
    "sequence",
    "structure",
]:
    for category in [
        "ACT_SITE",
        "BINDING",
        "UNION",
    ]:
        analysis_specs.append(
            {
                "modality": modality,
                "outcome": "matched_random_enrichment",
                "category": category,
                "source": "enrichment",
                "column": "protein_enrichment",
            }
        )

        analysis_specs.append(
            {
                "modality": modality,
                "outcome": "chance_adjusted_top_k_recall",
                "category": category,
                "source": "rank",
                "column": "protein_chance_adjusted_recall",
            }
        )

for modality in [
    "sequence",
    "structure",
]:
    analysis_specs.append(
        {
            "modality": modality,
            "outcome": "deletion_faithfulness_enrichment",
            "category": "NA",
            "source": "faith",
            "column": "mean_faithfulness_enrichment",
        }
    )

if len(analysis_specs) != 14:
    raise RuntimeError(
        f"Expected 14 primary tests, found {len(analysis_specs)}."
    )


# ============================================================
# RUN ASSOCIATIONS
# ============================================================

association_rows = []

for spec in analysis_specs:
    modality = spec["modality"]
    outcome = spec["outcome"]
    category = spec["category"]

    if spec["source"] == "enrichment":
        source = enrichment[
            (enrichment["modality"] == modality)
            & (enrichment["category"] == category)
        ][
            [
                "uniprot_id",
                spec["column"],
            ]
        ].copy()

    elif spec["source"] == "rank":
        source = rank[
            (rank["modality"] == modality)
            & (rank["category"] == category)
        ][
            [
                "uniprot_id",
                spec["column"],
            ]
        ].copy()

    elif spec["source"] == "faith":
        source = faith[
            faith["modality"] == modality
        ][
            [
                "uniprot_id",
                spec["column"],
            ]
        ].copy()

    else:
        raise RuntimeError(
            f"Unknown source: {spec['source']}"
        )

    if source["uniprot_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate IDs for {modality} {outcome} {category}."
        )

    merged = source.merge(
        conflict_df[
            [
                "uniprot_id",
                "protein_conflict_mean_bernoulli_js",
            ]
        ],
        on="uniprot_id",
        how="inner",
        validate="one_to_one",
    )

    x = merged[
        "protein_conflict_mean_bernoulli_js"
    ].to_numpy(
        dtype=np.float64
    )

    y = merged[
        spec["column"]
    ].to_numpy(
        dtype=np.float64
    )

    if len(x) < 2:
        raise RuntimeError(
            f"Too few proteins for {modality} {outcome} {category}."
        )

    if not np.all(np.isfinite(x)):
        raise RuntimeError(
            "Non-finite conflict values."
        )

    if not np.all(np.isfinite(y)):
        raise RuntimeError(
            "Non-finite XAI outcome values."
        )

    result = spearmanr(
        x,
        y,
    )

    rho = float(
        result.statistic
    )

    pvalue = float(
        result.pvalue
    )

    seed = bootstrap_seed(
        modality,
        outcome,
        category,
    )

    ci_low, ci_high, valid_bootstrap = (
        bootstrap_spearman_ci(
            x,
            y,
            seed,
        )
    )

    association_rows.append(
        {
            "modality": modality,
            "outcome": outcome,
            "category": category,
            "n_proteins": int(len(merged)),
            "spearman_rho": rho,
            "spearman_pvalue_two_sided": pvalue,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "bootstrap_valid_replicates": valid_bootstrap,
            "bootstrap_requested_replicates": N_BOOTSTRAP,
            "bootstrap_base_seed": BASE_SEED,
            "bootstrap_stream_seed": seed,
        }
    )


associations = pd.DataFrame(
    association_rows
)

if len(associations) != 14:
    raise RuntimeError(
        "Primary association row count is not 14."
    )


# ============================================================
# BH-FDR ACROSS ALL 14 PRIMARY TESTS
# ============================================================

associations[
    "spearman_qvalue_bh_fdr"
] = bh_fdr(
    associations[
        "spearman_pvalue_two_sided"
    ].to_numpy()
)


# ============================================================
# SAVE
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

conflict_df.to_csv(
    PROTEIN_CONFLICT_OUT,
    index=False,
)

associations.to_csv(
    ASSOCIATIONS_OUT,
    index=False,
)

run_info = {
    "analysis": "conflict_focused_xai_v1",
    "status": "frozen-protocol descriptive TEST XAI analysis",
    "n_xai_proteins": int(len(conflict_df)),
    "n_primary_association_tests": int(len(associations)),
    "primary_conflict": (
        "mean true-positive-label Bernoulli JS per protein"
    ),
    "conflict_logarithm": "natural",
    "categorical_normalization_of_494_labels": False,
    "correlation": "Spearman",
    "pvalue": "two-sided",
    "multiple_testing": (
        "Benjamini-Hochberg FDR across all 14 primary tests"
    ),
    "bootstrap_unit": "whole protein",
    "bootstrap_replicates": N_BOOTSTRAP,
    "bootstrap_base_seed": BASE_SEED,
    "protocol_sha256": sha256_file(PROTOCOL),
    "paired_ids_sha256": sha256_file(PAIRED_IDS),
    "bernoulli_js_sha256": sha256_file(JS_PATH),
    "test_labels_sha256": sha256_file(LABELS_PATH),
    "xai_ids_sha256": sha256_file(XAI_IDS_PATH),
    "biological_enrichment_sha256": sha256_file(
        ENRICHMENT_PATH
    ),
    "rank_analysis_sha256": sha256_file(
        RANK_PATH
    ),
    "deletion_faithfulness_sha256": sha256_file(
        FAITH_PATH
    ),
    "test_id_order_path": str(test_id_path),
    "test_id_order_sha256": sha256_file(
        test_id_path
    ),
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
# SCIENTIFIC OUTPUT
# ============================================================

print("=" * 110)
print("E10 CONFLICT-FOCUSED XAI — PRIMARY ASSOCIATIONS")
print("=" * 110)

display_cols = [
    "modality",
    "outcome",
    "category",
    "n_proteins",
    "spearman_rho",
    "spearman_pvalue_two_sided",
    "spearman_qvalue_bh_fdr",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
]

print(
    associations[
        display_cols
    ].to_string(
        index=False
    )
)

print()
print(
    "Protein-level conflict rows:",
    len(conflict_df),
)

print(
    "Primary association tests:",
    len(associations),
)

print("Analysis complete.")
print("=" * 110)
