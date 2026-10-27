from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(".")

OUT = ROOT / "data/manuscript/figures_v1"
OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# Authoritative frozen E10 artifacts
# =========================================================

enrichment_path = (
    ROOT
    / "data/xai/functional_sites_v1/biological_enrichment_v1/"
      "primary_enrichment_summary.csv"
)

faithfulness_path = (
    ROOT
    / "data/xai/functional_sites_v1/deletion_faithfulness_v1/"
      "aggregate_v1/faithfulness_summary.csv"
)

conflict_deletion_path = (
    ROOT
    / "data/xai/functional_sites_v1/conflict_deletion_v1/"
      "conflict_deletion_summary_v1.csv"
)

stability_path = (
    ROOT
    / "data/xai/functional_sites_v1/explanation_stability_v1/"
      "aggregation_v1/stability_summary.csv"
)


# =========================================================
# Helpers
# =========================================================

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def require_unique(rows, **conditions):
    matches = [
        row
        for row in rows
        if all(
            row.get(key) == value
            for key, value in conditions.items()
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one row for "
            f"{conditions}, found {len(matches)}."
        )

    return matches[0]


# =========================================================
# Load data
# =========================================================

enrichment_rows = read_csv(enrichment_path)
faithfulness_rows = read_csv(faithfulness_path)
conflict_rows = read_csv(conflict_deletion_path)
stability_rows = read_csv(stability_path)


# =========================================================
# Panel A — functional-site enrichment
# =========================================================

modalities = [
    "sequence",
    "structure",
]

categories = [
    "ACT_SITE",
    "BINDING",
    "UNION",
]

panel_a = {}

for modality in modalities:
    panel_a[modality] = {}

    for category in categories:
        row = require_unique(
            enrichment_rows,
            modality=modality,
            category=category,
        )

        panel_a[modality][category] = {
            "n_proteins": int(row["n_proteins"]),
            "mean": float(
                row["mean_protein_enrichment"]
            ),
            "low": float(
                row["bootstrap_ci95_low"]
            ),
            "high": float(
                row["bootstrap_ci95_high"]
            ),
        }


# Frozen population consistency checks
expected_n = {
    "ACT_SITE": 1152,
    "BINDING": 2146,
    "UNION": 2447,
}

for modality in modalities:
    for category in categories:
        observed_n = (
            panel_a[modality][category]["n_proteins"]
        )

        if observed_n != expected_n[category]:
            raise ValueError(
                "Unexpected Panel A population for "
                f"{modality}/{category}: {observed_n}"
            )


# =========================================================
# Panel B — deletion faithfulness
# =========================================================

panel_b = {}

for modality in modalities:
    row = require_unique(
        faithfulness_rows,
        modality=modality,
    )

    panel_b[modality] = {
        "n_proteins": int(
            row["n_proteins"]
        ),
        "n_targets": int(
            row["n_protein_label_targets"]
        ),
        "mean": float(
            row[
                "mean_protein_faithfulness_enrichment"
            ]
        ),
        "low": float(
            row["bootstrap_mean_ci95_low"]
        ),
        "high": float(
            row["bootstrap_mean_ci95_high"]
        ),
    }


for modality in modalities:
    if panel_b[modality]["n_proteins"] != 2447:
        raise ValueError(
            f"Unexpected faithfulness protein count "
            f"for {modality}."
        )

    if panel_b[modality]["n_targets"] != 23087:
        raise ValueError(
            f"Unexpected faithfulness target count "
            f"for {modality}."
        )


# =========================================================
# Panel C — conflict-driver perturbation enrichment
# =========================================================

if len(conflict_rows) != 1:
    raise ValueError(
        "Expected exactly one conflict-deletion summary row."
    )

conflict_row = conflict_rows[0]

panel_c = {
    "n_proteins": int(
        conflict_row["n_proteins"]
    ),
    "n_targets": int(
        conflict_row["n_protein_label_targets"]
    ),
    "mean": float(
        conflict_row["mean_protein_enrichment"]
    ),
    "low": float(
        conflict_row["mean_enrichment_ci95_low"]
    ),
    "high": float(
        conflict_row["mean_enrichment_ci95_high"]
    ),
    "fraction_positive": float(
        conflict_row["fraction_proteins_positive"]
    ),
}

if panel_c["n_proteins"] != 2447:
    raise ValueError(
        "Unexpected conflict-deletion protein count."
    )

if panel_c["n_targets"] != 23087:
    raise ValueError(
        "Unexpected conflict-deletion target count."
    )


# =========================================================
# Panel D — explanation stability
# =========================================================

panel_d = {}

for modality in modalities:
    row = require_unique(
        stability_rows,
        modality=modality,
    )

    panel_d[modality] = {
        "n_proteins": int(
            row["frozen_proteins_attempted"]
        ),
        "n_targets": int(
            row["protein_label_targets_attempted"]
        ),
        "eligible_targets": int(
            row["eligible_protein_label_targets"]
        ),
        "spearman": float(
            row["spearman_mean_protein_level"]
        ),
        "spearman_low": float(
            row["spearman_bootstrap_ci95_low"]
        ),
        "spearman_high": float(
            row["spearman_bootstrap_ci95_high"]
        ),
        "jaccard": float(
            row["jaccard_mean_protein_level"]
        ),
        "jaccard_low": float(
            row["jaccard_bootstrap_ci95_low"]
        ),
        "jaccard_high": float(
            row["jaccard_bootstrap_ci95_high"]
        ),
    }


for modality in modalities:
    if panel_d[modality]["n_proteins"] != 200:
        raise ValueError(
            f"Unexpected stability protein count "
            f"for {modality}."
        )

    if panel_d[modality]["n_targets"] != 1770:
        raise ValueError(
            f"Unexpected stability target count "
            f"for {modality}."
        )

    if panel_d[modality]["eligible_targets"] != 1770:
        raise ValueError(
            f"Unexpected stability eligible target count "
            f"for {modality}."
        )


# =========================================================
# Verification against frozen E10 targets
# =========================================================

verification_targets = {
    "seq_union": 0.010291080895498756,
    "struct_union": 0.018188307449326105,
    "seq_faithfulness": 2.4033468562321896,
    "struct_faithfulness": 2.864877653352747,
    "conflict_enrichment": 0.0437242691401686,
    "seq_spearman": 0.9948918356489657,
    "seq_jaccard": 0.9157356701641245,
    "struct_spearman": 0.9852980830446234,
    "struct_jaccard": 0.911514941430058,
}

observed_targets = {
    "seq_union":
        panel_a["sequence"]["UNION"]["mean"],

    "struct_union":
        panel_a["structure"]["UNION"]["mean"],

    "seq_faithfulness":
        panel_b["sequence"]["mean"],

    "struct_faithfulness":
        panel_b["structure"]["mean"],

    "conflict_enrichment":
        panel_c["mean"],

    "seq_spearman":
        panel_d["sequence"]["spearman"],

    "seq_jaccard":
        panel_d["sequence"]["jaccard"],

    "struct_spearman":
        panel_d["structure"]["spearman"],

    "struct_jaccard":
        panel_d["structure"]["jaccard"],
}

for key, expected in verification_targets.items():
    observed = observed_targets[key]

    if not np.isclose(
        observed,
        expected,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(
            f"Frozen E10 verification failed for {key}: "
            f"observed={observed}, expected={expected}"
        )


# =========================================================
# Figure style
# =========================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


fig, axes = plt.subplots(
    2,
    2,
    figsize=(11.5, 7.5),
)

axA = axes[0, 0]
axB = axes[0, 1]
axC = axes[1, 0]
axD = axes[1, 1]


# =========================================================
# Panel A — functional-site enrichment
# =========================================================

x = np.arange(len(categories))

offset = 0.11

sequence_x = x - offset
structure_x = x + offset


seq_mean = np.array([
    panel_a["sequence"][category]["mean"]
    for category in categories
])

seq_low = np.array([
    panel_a["sequence"][category]["low"]
    for category in categories
])

seq_high = np.array([
    panel_a["sequence"][category]["high"]
    for category in categories
])


struct_mean = np.array([
    panel_a["structure"][category]["mean"]
    for category in categories
])

struct_low = np.array([
    panel_a["structure"][category]["low"]
    for category in categories
])

struct_high = np.array([
    panel_a["structure"][category]["high"]
    for category in categories
])


axA.errorbar(
    sequence_x,
    seq_mean,
    yerr=np.vstack([
        seq_mean - seq_low,
        seq_high - seq_mean,
    ]),
    fmt="o",
    capsize=4,
    linewidth=1.5,
    markersize=6,
    label="Sequence",
)

axA.errorbar(
    structure_x,
    struct_mean,
    yerr=np.vstack([
        struct_mean - struct_low,
        struct_high - struct_mean,
    ]),
    fmt="s",
    capsize=4,
    linewidth=1.5,
    markersize=6,
    label="Structure",
)

axA.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.65,
)

axA.set_xticks(x)

axA.set_xticklabels([
    "Active site",
    "Binding",
    "Union",
])

axA.set_ylabel(
    "Mean protein-level enrichment"
)

axA.set_title(
    "A  Functional-site enrichment",
    loc="left",
    fontweight="bold",
)

axA.legend(
    frameon=False,
    fontsize=8,
)


# =========================================================
# Panel B — deletion faithfulness
# =========================================================

b_labels = [
    "Sequence",
    "Structure",
]

# Keep both categories comfortably inside the panel.
b_x = np.array([
    0.25,
    0.75,
])

b_mean = np.array([
    panel_b["sequence"]["mean"],
    panel_b["structure"]["mean"],
])

b_low = np.array([
    panel_b["sequence"]["low"],
    panel_b["structure"]["low"],
])

b_high = np.array([
    panel_b["sequence"]["high"],
    panel_b["structure"]["high"],
])


axB.errorbar(
    b_x,
    b_mean,
    yerr=np.vstack([
        b_mean - b_low,
        b_high - b_mean,
    ]),
    fmt="o",
    capsize=4,
    linewidth=1.5,
    markersize=6,
)

axB.set_xticks(b_x)
axB.set_xticklabels(b_labels)

axB.set_xlim(
    0.0,
    1.0,
)

axB.set_ylabel(
    "Mean protein-level faithfulness enrichment"
)

axB.set_title(
    "B  Deletion faithfulness",
    loc="left",
    fontweight="bold",
)


# Give annotation labels clear space above CI caps.
faithfulness_range = (
    max(b_high)
    - min(b_low)
)

faithfulness_pad = max(
    0.04,
    faithfulness_range * 0.15,
)

for i, value in enumerate(b_mean):
    axB.text(
        b_x[i],
        b_high[i] + faithfulness_pad,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

axB.set_ylim(
    min(b_low) - 0.15,
    max(b_high) + 0.22,
)


# =========================================================
# Panel C — conflict-driver enrichment
# =========================================================

c_x = np.array([0])

c_mean = np.array([
    panel_c["mean"]
])

c_low = np.array([
    panel_c["low"]
])

c_high = np.array([
    panel_c["high"]
])


# Main frozen point estimate and 95% bootstrap CI.
axC.errorbar(
    c_x,
    c_mean,
    yerr=np.vstack([
        c_mean - c_low,
        c_high - c_mean,
    ]),
    fmt="o",
    capsize=5,
    linewidth=1.5,
    markersize=7,
)

# Zero-enrichment reference.
axC.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.65,
)

# Compact horizontal range for the single estimate.
axC.set_xlim(
    -0.40,
    0.40,
)

axC.set_ylim(
    -0.005,
    0.060,
)

axC.set_xticks([0])

axC.set_xticklabels([
    "Conflict-focused\nresidue evidence"
])

axC.set_ylabel(
    "Excess disagreement reduction vs. random"
)

axC.set_title(
    "C  Conflict-driver perturbation enrichment",
    loc="left",
    fontweight="bold",
)

# Keep the numerical estimate as the visual focus.
axC.text(
    0,
    panel_c["high"] + 0.003,
    f"{panel_c['mean']:+.3f}",
    ha="center",
    va="bottom",
    fontsize=8,
    fontweight="bold",
)


# =========================================================
# Panel D — explanation stability
# =========================================================

d_metrics = [
    "Spearman",
    "Jaccard",
]

d_x = np.arange(2)

d_offset = 0.11

d_seq_x = d_x - d_offset
d_struct_x = d_x + d_offset


d_seq_mean = np.array([
    panel_d["sequence"]["spearman"],
    panel_d["sequence"]["jaccard"],
])

d_seq_low = np.array([
    panel_d["sequence"]["spearman_low"],
    panel_d["sequence"]["jaccard_low"],
])

d_seq_high = np.array([
    panel_d["sequence"]["spearman_high"],
    panel_d["sequence"]["jaccard_high"],
])


d_struct_mean = np.array([
    panel_d["structure"]["spearman"],
    panel_d["structure"]["jaccard"],
])

d_struct_low = np.array([
    panel_d["structure"]["spearman_low"],
    panel_d["structure"]["jaccard_low"],
])

d_struct_high = np.array([
    panel_d["structure"]["spearman_high"],
    panel_d["structure"]["jaccard_high"],
])


axD.errorbar(
    d_seq_x,
    d_seq_mean,
    yerr=np.vstack([
        d_seq_mean - d_seq_low,
        d_seq_high - d_seq_mean,
    ]),
    fmt="o",
    capsize=4,
    linewidth=1.5,
    markersize=6,
    label="Sequence",
)

axD.errorbar(
    d_struct_x,
    d_struct_mean,
    yerr=np.vstack([
        d_struct_mean - d_struct_low,
        d_struct_high - d_struct_mean,
    ]),
    fmt="s",
    capsize=4,
    linewidth=1.5,
    markersize=6,
    label="Structure",
)

axD.set_xticks(d_x)
axD.set_xticklabels(d_metrics)

axD.set_ylabel(
    "Stability"
)

axD.set_ylim(
    0.89,
    1.005,
)

axD.set_title(
    "D  Explanation stability",
    loc="left",
    fontweight="bold",
)

axD.legend(
    frameon=False,
    fontsize=8,
)


# =========================================================
# Shared formatting
# =========================================================

for ax in axes.flat:
    ax.tick_params(
        axis="x",
        pad=6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.18,
    )


# =========================================================
# Overall title
# =========================================================

fig.suptitle(
    "Residue-level explanations show functional-site enrichment, "
    "faithfulness, conflict relevance, and high stability",
    fontsize=14,
    fontweight="bold",
    y=0.985,
)


# =========================================================
# Interpretation notes
# =========================================================

fig.text(
    0.5,
    0.043,
    "Points show frozen E10 aggregate estimates; whiskers show "
    "95% whole-protein bootstrap CIs where available.",
    ha="center",
    fontsize=8,
)

fig.text(
    0.5,
    0.018,
    "XAI results support functional-site association, faithfulness, "
    "and stability of residue-level evidence; they do not establish "
    "biological causality.",
    ha="center",
    fontsize=7.5,
)


fig.tight_layout(
    rect=[
        0,
        0.085,
        1,
        0.945,
    ],
    h_pad=2.4,
    w_pad=2.8,
)


# =========================================================
# Save outputs
# =========================================================

png = (
    OUT
    / "Figure_5_xai_biological_evidence.png"
)

pdf = (
    OUT
    / "Figure_5_xai_biological_evidence.pdf"
)

svg = (
    OUT
    / "Figure_5_xai_biological_evidence.svg"
)


fig.savefig(
    png,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

fig.savefig(
    svg,
    bbox_inches="tight",
)

plt.close(fig)


# =========================================================
# Numerical verification output
# =========================================================

print("CREATED:")
print(png)
print(pdf)
print(svg)


print()
print("PANEL A — FUNCTIONAL-SITE ENRICHMENT")

for modality in modalities:
    print(modality.upper())

    for category in categories:
        d = panel_a[modality][category]

        print(
            f"  {category:<8} "
            f"n={d['n_proteins']:4d}  "
            f"mean={d['mean']:+.9f}  "
            f"CI=[{d['low']:.9f}, {d['high']:.9f}]"
        )


print()
print("PANEL B — DELETION FAITHFULNESS")

for modality in modalities:
    d = panel_b[modality]

    print(
        f"{modality:<9} "
        f"n={d['n_proteins']:4d}  "
        f"targets={d['n_targets']:5d}  "
        f"mean={d['mean']:.9f}  "
        f"CI=[{d['low']:.9f}, {d['high']:.9f}]"
    )


print()
print("PANEL C — CONFLICT-DRIVER ENRICHMENT")

print(
    f"n={panel_c['n_proteins']}  "
    f"targets={panel_c['n_targets']}  "
    f"mean={panel_c['mean']:+.9f}  "
    f"CI=[{panel_c['low']:.9f}, "
    f"{panel_c['high']:.9f}]  "
    f"fraction_positive="
    f"{panel_c['fraction_positive']:.9f}"
)


print()
print("PANEL D — EXPLANATION STABILITY")

for modality in modalities:
    d = panel_d[modality]

    print(
        f"{modality.upper()}  "
        f"n={d['n_proteins']}  "
        f"targets={d['n_targets']}"
    )

    print(
        f"  Spearman "
        f"mean={d['spearman']:.9f}  "
        f"CI=[{d['spearman_low']:.9f}, "
        f"{d['spearman_high']:.9f}]"
    )

    print(
        f"  Jaccard  "
        f"mean={d['jaccard']:.9f}  "
        f"CI=[{d['jaccard_low']:.9f}, "
        f"{d['jaccard_high']:.9f}]"
    )