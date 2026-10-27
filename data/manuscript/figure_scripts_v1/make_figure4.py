from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(".")
OUT = ROOT / "data/manuscript/figures_v1"
OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load frozen machine-readable artifacts
# =========================================================

# E8 — authoritative FINAL TEST GO-frequency analysis
e8_path = (
    ROOT
    / "data/final_evaluation/final_test_go_frequency_v1/"
      "go_frequency_test_summary.csv"
)

with open(e8_path, newline="") as f:
    e8_rows = list(csv.DictReader(f))


# E12 — alternative structure-backbone FINAL TEST analysis
e12_rel_path = (
    ROOT
    / "data/analysis/backbone_sensitivity_v1/final_test_v1/"
      "reliability_summary.json"
)

e12_boot_path = (
    ROOT
    / "data/analysis/backbone_sensitivity_v1/final_test_v1/bootstrap/"
      "bootstrap_summary.json"
)

with open(e12_rel_path) as f:
    e12_rel = json.load(f)

with open(e12_boot_path) as f:
    e12_boot = json.load(f)


# =========================================================
# Panel A — GO-frequency robustness
# =========================================================

expected_bands = [
    "75-99",
    "100-249",
    "250-499",
    "500-999",
    ">=1000",
]

e8_map = {row["band"]: row for row in e8_rows}

missing_bands = [
    band for band in expected_bands
    if band not in e8_map
]

if missing_bands:
    raise ValueError(
        f"Missing expected E8 support bands: {missing_bands}"
    )

frequency_n_labels = np.array([
    int(e8_map[band]["n_labels"])
    for band in expected_bands
])

frequency_delta = np.array([
    float(e8_map[band]["delta_ap"])
    for band in expected_bands
])

frequency_low = np.array([
    float(e8_map[band]["delta_ap_ci_low"])
    for band in expected_bands
])

frequency_high = np.array([
    float(e8_map[band]["delta_ap_ci_high"])
    for band in expected_bands
])


# =========================================================
# Panel B — structure-backbone sensitivity
# =========================================================

# Primary structure-GNN FINAL TEST values are frozen headline values
# from the primary structure-failure reliability analysis.
primary_delta = 0.11161403
primary_low = 0.10182952
primary_high = 0.12151900


# Alternative GCN values are read directly from the frozen E12
# FINAL TEST bootstrap artifact.
e12_ap = e12_boot["metrics"]["average_precision"]

alternative_delta = float(
    e12_ap["observed_delta"]
)

alternative_low = float(
    e12_ap["ci_2.5"]
)

alternative_high = float(
    e12_ap["ci_97.5"]
)


# =========================================================
# Artifact consistency checks
# =========================================================

if (
    e12_rel["alternative_structure_backbone"]
    != "GCN-style normalized-neighbor propagation"
):
    raise ValueError(
        "Unexpected E12 alternative structure backbone."
    )

if (
    e12_rel["conflict_definition"]
    != "per-label Bernoulli Jensen-Shannon divergence using natural logarithms"
):
    raise ValueError(
        "Unexpected E12 conflict definition."
    )

if int(e12_rel["evaluation_proteins"]) != 4731:
    raise ValueError(
        "Unexpected E12 FINAL TEST protein count."
    )

if int(e12_rel["labels"]) != 494:
    raise ValueError(
        "Unexpected E12 label count."
    )


# Cross-check E12 reliability summary against bootstrap artifact.
# Tiny numerical differences are expected because the stored
# summaries were produced through slightly different numerical paths.
e12_summary_delta = float(
    e12_rel[
        "binary_delta_MB_minus_MA"
    ]["average_precision"]
)

if not np.isclose(
    e12_summary_delta,
    alternative_delta,
    atol=1e-6,
    rtol=0,
):
    raise ValueError(
        "E12 reliability and bootstrap Delta AP values disagree."
    )


# =========================================================
# Figure setup
# =========================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, (axA, axB) = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.8),
    gridspec_kw={
        "width_ratios": [1.6, 1.0],
    },
)


# =========================================================
# Panel A — GO-frequency robustness
# =========================================================

x = np.arange(len(expected_bands))

lower_err = frequency_delta - frequency_low
upper_err = frequency_high - frequency_delta

axA.errorbar(
    x,
    frequency_delta,
    yerr=np.vstack([
        lower_err,
        upper_err,
    ]),
    fmt="o",
    capsize=4,
    linewidth=1.5,
    markersize=6,
)

axA.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.6,
)

axA.set_xticks(x)

axA.set_xticklabels([
    f"{band.replace('>=', '≥')}\n(n={n:,})"
    for band, n in zip(
        expected_bands,
        frequency_n_labels,
    )
])

axA.set_xlabel(
    "TRAIN GO-MF support band"
)

axA.set_ylabel(
    "ΔAP: conflict − uncertainty"
)

axA.set_ylim(
    -0.005,
    0.15,
)

axA.set_title(
    "A  Conflict reliability across GO-frequency strata",
    loc="left",
    fontweight="bold",
)

for i, value in enumerate(frequency_delta):
    axA.text(
        i,
        frequency_high[i] + 0.004,
        f"{value:+.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )


# =========================================================
# Panel B — structure-backbone sensitivity
# =========================================================

backbone_labels = [
    "Primary GNN",
    "Alternative GCN",
]

backbone_delta = np.array([
    primary_delta,
    alternative_delta,
])

backbone_low = np.array([
    primary_low,
    alternative_low,
])

backbone_high = np.array([
    primary_high,
    alternative_high,
])

x = np.arange(
    len(backbone_labels)
)

lower_err = (
    backbone_delta
    - backbone_low
)

upper_err = (
    backbone_high
    - backbone_delta
)

axB.errorbar(
    x,
    backbone_delta,
    yerr=np.vstack([
        lower_err,
        upper_err,
    ]),
    fmt="o",
    capsize=4,
    linewidth=1.5,
    markersize=6,
)

axB.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.6,
)

axB.set_xticks(x)

axB.set_xticklabels(
    backbone_labels
)

axB.set_ylabel(
    "ΔAP: conflict − uncertainty"
)

axB.set_ylim(
    -0.005,
    0.15,
)

axB.set_title(
    "B  Structure-backbone sensitivity",
    loc="left",
    fontweight="bold",
)

for i, value in enumerate(backbone_delta):
    axB.text(
        i,
        backbone_high[i] + 0.007,
        f"{ value:+.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )


# =========================================================
# Shared formatting
# =========================================================

for ax in (axA, axB):
    ax.tick_params(
        axis="x",
        pad=6,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.18,
    )


# =========================================================
# Figure title and explanatory notes
# =========================================================

fig.suptitle(
    "Conflict reliability persists across GO-frequency strata "
    "and an alternative structure backbone",
    fontsize=14,
    fontweight="bold",
    y=0.99,
)


# First explanatory line
fig.text(
    0.5,
    0.055,
    "ΔAP = AP with conflict − AP with uncertainty only; "
    "positive values indicate incremental reliability information.",
    ha="center",
    fontsize=8,
)


# Second explanatory line
fig.text(
    0.5,
    0.025,
    "Whiskers: 95% whole-protein bootstrap CI. "
    "Backbone comparison tests persistence, not architecture superiority.",
    ha="center",
    fontsize=7.5,
)


# Reserve enough bottom space for both explanatory lines.
fig.tight_layout(
    rect=[
        0,
        0.105,
        1,
        0.94,
    ],
    w_pad=3.0,
)


# =========================================================
# Save
# =========================================================

png = (
    OUT
    / "Figure_4_frequency_backbone_robustness.png"
)

pdf = (
    OUT
    / "Figure_4_frequency_backbone_robustness.pdf"
)

svg = (
    OUT
    / "Figure_4_frequency_backbone_robustness.svg"
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
# Verification output
# =========================================================

print("CREATED:")
print(png)
print(pdf)
print(svg)


print()
print(
    "PANEL A — E8 FINAL TEST"
)

for band, n, delta, low, high in zip(
    expected_bands,
    frequency_n_labels,
    frequency_delta,
    frequency_low,
    frequency_high,
):
    print(
        f"{band:>7}  "
        f"n_labels={n:3d}  "
        f"DeltaAP={delta:+.9f}  "
        f"CI=[{low:.9f}, {high:.9f}]"
    )


print()
print(
    "PANEL B — BACKBONE SENSITIVITY"
)

print(
    "Primary GNN     "
    f"DeltaAP={primary_delta:+.9f}  "
    f"CI=[{primary_low:.9f}, {primary_high:.9f}]"
)

print(
    "Alternative GCN "
    f"DeltaAP={alternative_delta:+.9f}  "
    f"CI=[{alternative_low:.9f}, {alternative_high:.9f}]"
)
