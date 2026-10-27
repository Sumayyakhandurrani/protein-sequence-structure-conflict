from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(".")
OUT = ROOT / "data/manuscript/figures_v1"
OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# Helpers
# =========================================================

def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    *,
    linewidth=1.2,
    linestyle="-",
    title_size=8.2,
    body_size=6.8,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=linewidth,
        linestyle=linestyle,
        facecolor="white",
        edgecolor="black",
        transform=ax.transAxes,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(patch)

    ax.text(
        x + 0.012,
        y + h - 0.018,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        zorder=4,
    )

    ax.text(
        x + 0.012,
        y + h - 0.057,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=body_size,
        linespacing=1.18,
        zorder=4,
    )

    return patch


def add_arrow(
    ax,
    start,
    end,
    *,
    linewidth=1.2,
    mutation_scale=10,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color="black",
        transform=ax.transAxes,
        clip_on=False,
        zorder=2,
    )
    ax.add_patch(arrow)
    return arrow


def add_line(
    ax,
    x1,
    y1,
    x2,
    y2,
    *,
    linewidth=1.1,
    linestyle="-",
):
    ax.plot(
        [x1, x2],
        [y1, y2],
        transform=ax.transAxes,
        color="black",
        linewidth=linewidth,
        linestyle=linestyle,
        clip_on=False,
        zorder=1,
    )


# =========================================================
# Canvas
# =========================================================

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
})

fig, ax = plt.subplots(
    figsize=(16.5, 9.0)
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


# =========================================================
# Title
# =========================================================

fig.suptitle(
    "Sequence–structure disagreement as a reliability signal for "
    "protein molecular-function prediction",
    fontsize=15,
    fontweight="bold",
    y=0.975,
)


# =========================================================
# Section headings
# =========================================================

headings = [
    (0.105, "A  Data and benchmark"),
    (0.340, "B–C  Predictive experts"),
    (0.565, "D  Uncertainty and conflict"),
    (0.755, "E  Reliability question"),
    (0.920, "F  Validation"),
]

for x, text in headings:
    ax.text(
        x,
        0.925,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.7,
        fontweight="bold",
    )


# =========================================================
# A — Data and benchmark
# =========================================================

add_box(
    ax,
    0.015,
    0.735,
    0.180,
    0.145,
    "Reviewed protein data",
    "UniProtKB/Swiss-Prot\n"
    "Experimental GO Molecular Function\n"
    "evidence: EXP, IDA, IPI, IMP, IGI, IEP\n\n"
    "Structural evidence from the frozen\n"
    "structure-processing pipeline",
)

add_arrow(
    ax,
    (0.105, 0.735),
    (0.105, 0.700),
)


add_box(
    ax,
    0.015,
    0.520,
    0.180,
    0.180,
    "GO-MF target construction",
    "Child-to-parent GO propagation\n"
    "GO:0003674 root removed\n\n"
    "TRAIN-only vocabulary\n"
    "494 frozen GO-MF labels\n"
    "Minimum TRAIN support = 75",
)

add_arrow(
    ax,
    (0.105, 0.520),
    (0.105, 0.485),
)


add_box(
    ax,
    0.015,
    0.235,
    0.180,
    0.250,
    "Graph-V2 benchmark",
    "Protein-similarity graph\n"
    "Connected components assigned wholly\n"
    "to partitions\n\n"
    "TRAIN        38,031\n"
    "VAL            4,754\n"
    "TEST           4,754\n"
    "Paired TEST    4,731\n\n"
    "Graph/audit criterion:\n"
    "≥30% sequence identity and\n"
    "≥80% bidirectional coverage",
)


ax.text(
    0.105,
    0.205,
    "Leakage-controlled benchmark; the criterion does not imply\n"
    "a mathematical guarantee of zero cross-split homology.",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=6.4,
    fontstyle="italic",
)


# =========================================================
# Benchmark → two expert branches
# =========================================================

# Clean orthogonal branching rather than diagonal lines crossing boxes.
add_line(
    ax,
    0.195,
    0.360,
    0.215,
    0.360,
)

add_line(
    ax,
    0.215,
    0.360,
    0.215,
    0.680,
)

add_line(
    ax,
    0.215,
    0.360,
    0.215,
    0.385,
)

add_arrow(
    ax,
    (0.215, 0.680),
    (0.235, 0.680),
)

add_arrow(
    ax,
    (0.215, 0.385),
    (0.235, 0.385),
)


# =========================================================
# B — Sequence expert
# =========================================================

add_box(
    ax,
    0.235,
    0.570,
    0.210,
    0.275,
    "B  Sequence expert",
    "Amino-acid sequence\n"
    "↓\n"
    "Frozen ESM-2\n"
    "facebook/esm2_t12_35M_UR50D\n"
    "480-D residue representation\n"
    "length-weighted protein mean\n"
    "↓\n"
    "MLP: 480 → 512 → 494\n"
    "LayerNorm • GELU • Dropout\n"
    "↓\n"
    "Independent temperature scaling\n"
    "T_seq = 1.10698",
)

ax.text(
    0.340,
    0.548,
    r"$p_{\mathrm{seq}}$",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
)


# =========================================================
# C — Structure expert
# =========================================================

add_box(
    ax,
    0.235,
    0.245,
    0.210,
    0.260,
    "C  Structure-derived / geometric expert",
    "Residue graph\n"
    "21-D residue-identity encoding\n"
    "Cα geometry\n"
    "Spatial edges ≤10 Å + sequential edges\n"
    "Edge features: distance, sequential\n"
    "indicator, absolute sequence separation\n"
    "↓\n"
    "2-layer GNN • hidden = 128\n"
    "Global mean pooling → 494 outputs\n"
    "↓\n"
    "Calibration evaluated; fitted T rejected\n"
    "T_struct = 1.0 (unscaled)",
)

ax.text(
    0.340,
    0.223,
    r"$p_{\mathrm{struct}}$",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
)

ax.text(
    0.340,
    0.192,
    "pLDDT is metadata/QC only; it is not an input to the primary model.",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=6.3,
    fontstyle="italic",
)


# =========================================================
# Expert outputs → D
# =========================================================

# Both calibrated expert outputs feed a shared junction.
# From that junction, downstream analysis computes expert
# uncertainty and sequence–structure conflict.

junction_x = 0.465
junction_y = 0.495


# Sequence expert -> shared junction
add_line(
    ax,
    0.445,
    0.680,
    junction_x,
    0.680,
)

add_line(
    ax,
    junction_x,
    0.680,
    junction_x,
    junction_y,
)


# Structure-derived expert -> shared junction
add_line(
    ax,
    0.445,
    0.385,
    junction_x,
    0.385,
)

add_line(
    ax,
    junction_x,
    0.385,
    junction_x,
    junction_y,
)


# Small junction marker
ax.plot(
    junction_x,
    junction_y,
    marker="o",
    markersize=3.5,
    color="black",
    transform=ax.transAxes,
    zorder=3,
)


# Shared expert probabilities -> uncertainty
add_line(
    ax,
    junction_x,
    junction_y,
    0.475,
    junction_y,
)

add_line(
    ax,
    0.475,
    junction_y,
    0.475,
    0.590,
)

add_arrow(
    ax,
    (0.475, 0.590),
    (0.485, 0.590),
    linewidth=1.1,
    mutation_scale=9,
)


# Shared expert probabilities -> sequence–structure conflict
add_line(
    ax,
    0.475,
    junction_y,
    0.475,
    0.395,
)

add_arrow(
    ax,
    (0.475, 0.395),
    (0.485, 0.395),
    linewidth=1.1,
    mutation_scale=9,
)

# =========================================================
# D — Uncertainty and conflict
# =========================================================

add_box(
    ax,
    0.485,
    0.515,
    0.165,
    0.155,
    "Expert uncertainty",
    "For each expert probability p:\n\n"
    r"$U(p)=1-2|p-0.5|$"
    "\n\n"
    "Per protein × GO-MF label",
    body_size=7.1,
)


add_box(
    ax,
    0.485,
    0.320,
    0.165,
    0.150,
    "Sequence–structure conflict",
    "Primary conflict measure:\n\n"
    r"$JS[Bern(p_{seq}),Bern(p_{struct})]$"
    "\n\n"
    "Bernoulli Jensen–Shannon divergence\n"
    "Natural logarithms",
    body_size=6.7,
)


# =========================================================
# D → E
# =========================================================

# Merge both evidence streams before the reliability question.
add_line(
    ax,
    0.650,
    0.590,
    0.670,
    0.590,
)

add_line(
    ax,
    0.650,
    0.395,
    0.670,
    0.395,
)

add_line(
    ax,
    0.670,
    0.395,
    0.670,
    0.590,
)

add_arrow(
    ax,
    (0.670, 0.4925),
    (0.695, 0.4925),
)


# =========================================================
# E — Reliability question
# =========================================================

add_box(
    ax,
    0.695,
    0.405,
    0.155,
    0.275,
    "Central reliability question",
    "Does sequence–structure disagreement\n"
    "provide information about prediction\n"
    "failure beyond expert uncertainty alone?\n\n"
    "MA\n"
    "Expert uncertainty features\n\n"
    "versus\n\n"
    "MB\n"
    "Expert uncertainty features\n"
    "+ sequence–structure conflict",
    linewidth=1.6,
    title_size=8.2,
    body_size=6.8,
)


ax.text(
    0.7725,
    0.375,
    "Primary emphasis: structure-expert failure\n"
    "Sequence-expert reliability: asymmetric control",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=6.4,
    fontstyle="italic",
)


# =========================================================
# E → F
# =========================================================

add_arrow(
    ax,
    (0.850, 0.542),
    (0.875, 0.542),
)


# =========================================================
# F — Validation framework
# =========================================================

validation_x = 0.885
validation_w = 0.105

validation_boxes = [
    (
        0.745,
        0.125,
        "Core evaluation",
        "Reliability\n"
        "Conflict-aware arbitration\n"
        "High-confidence conflict",
        "-",
    ),
    (
        0.590,
        0.125,
        "Robustness",
        "Low-homology strata\n"
        "Structural degradation\n"
        "Sequence degradation\n"
        "GO-MF frequency strata",
        "--",
    ),
    (
        0.455,
        0.105,
        "Controls",
        "M0 / MA / MB ablation\n"
        "Randomized-conflict control",
        "--",
    ),
    (
        0.295,
        0.130,
        "Biological / XAI",
        "Functional-site enrichment\n"
        "Rank localization\n"
        "Deletion faithfulness\n"
        "Conflict-focused XAI\n"
        "Conflict-driver deletion\n"
        "Explanation stability",
        "--",
    ),
    (
        0.145,
        0.120,
        "Sensitivity / efficiency",
        "Alternative GCN backbone\n"
        "Runtime • GPU memory\n"
        "Parameter counts",
        "--",
    ),
]


# Validation spine sits away from box boundaries.
spine_x = 0.870

add_line(
    ax,
    spine_x,
    0.190,
    spine_x,
    0.807,
    linewidth=1.0,
)


for y, h, title, body, ls in validation_boxes:
    center_y = y + h / 2

    add_arrow(
        ax,
        (spine_x, center_y),
        (validation_x, center_y),
        linewidth=0.95,
        mutation_scale=8,
    )

    add_box(
        ax,
        validation_x,
        y,
        validation_w,
        h,
        title,
        body,
        linewidth=1.05,
        linestyle=ls,
        title_size=7.2,
        body_size=5.9,
    )


# =========================================================
# Confirmatory-status legend
# =========================================================

legend_x = 0.695

add_line(
    ax,
    legend_x,
    0.245,
    legend_x + 0.030,
    0.245,
    linewidth=1.4,
)

ax.text(
    legend_x + 0.037,
    0.245,
    "Frozen / confirmatory core",
    transform=ax.transAxes,
    ha="left",
    va="center",
    fontsize=6.5,
)


add_line(
    ax,
    legend_x,
    0.215,
    legend_x + 0.030,
    0.215,
    linewidth=1.2,
    linestyle="--",
)

ax.text(
    legend_x + 0.037,
    0.215,
    "Designated post-confirmatory robustness / XAI",
    transform=ax.transAxes,
    ha="left",
    va="center",
    fontsize=6.5,
)


# =========================================================
# Footer
# =========================================================

fig.text(
    0.5,
    0.027,
    "Arrows indicate computational analysis flow, not biological causality. "
    "Core models and primary TEST analyses were frozen before TEST evaluation; "
    "designated robustness and explainability analyses were performed "
    "post-confirmatory.",
    ha="center",
    va="bottom",
    fontsize=7.0,
)


# =========================================================
# Save
# =========================================================

fig.subplots_adjust(
    left=0.025,
    right=0.985,
    top=0.925,
    bottom=0.075,
)

png = OUT / "Figure_1_study_design.png"
pdf = OUT / "Figure_1_study_design.pdf"
svg = OUT / "Figure_1_study_design.svg"

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
# Content verification
# =========================================================

print("CREATED:")
print(png)
print(pdf)
print(svg)

print()
print("FIGURE 1 CONTENT CHECK")
print("GO-MF labels: 494")
print("Minimum TRAIN support: 75")
print("TRAIN: 38,031")
print("VAL: 4,754")
print("TEST: 4,754")
print("Paired TEST: 4,731")
print("T_seq: 1.106977105140686")
print("T_struct deployed: 1.0")
print("T_struct fitted but rejected: 1.0261014699935913")
print("Uncertainty: U(p) = 1 - 2|p - 0.5|")
print(
    "Conflict: Bernoulli Jensen-Shannon divergence, "
    "natural logarithm"
)
print(
    "Structure branch wording: "
    "structure-derived / geometric expert"
)
print("pLDDT explicitly excluded as primary-model input")
print("No mathematical zero-cross-split-homology guarantee stated")
print("No performance results included")
print(
    "Confirmatory and designated post-confirmatory "
    "analyses visually distinguished"
)