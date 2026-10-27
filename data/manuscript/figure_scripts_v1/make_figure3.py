from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "data/manuscript/figures_v1"
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================
# Load frozen machine-readable artifacts
# =========================================================

with open(
    ROOT / "data/robustness/structural_degradation_v1/analysis/"
           "structural_degradation_summary.json"
) as f:
    e6_perf = json.load(f)

with open(
    ROOT / "data/robustness/structural_degradation_v1/analysis/"
           "structural_degradation_reliability.json"
) as f:
    e6_rel = json.load(f)

with open(
    ROOT / "data/robustness/sequence_degradation_v1/analysis/"
           "sequence_degradation_summary.json"
) as f:
    e7_perf = json.load(f)

with open(
    ROOT / "data/robustness/sequence_degradation_v1/"
           "reliability_analysis/bootstrap/bootstrap_summary.json"
) as f:
    e7_boot = json.load(f)


# =========================================================
# Panel A — frozen E5 ledger values
# =========================================================
# Source:
# data/manuscript/experiment_ledger_v1/E5_low_homology.txt
#
# >=30% contains only 8 proteins and is descriptive only.

homology_labels = [
    "No hit",
    "<20%",
    "20–25%",
    "25–30%",
    "≥30%",
]

homology_n = np.array([
    4194,
    151,
    248,
    130,
    8,
])

homology_delta = np.array([
    0.111423136,
    0.163638518,
    0.080405227,
    0.099559637,
    0.185478594,
])

homology_low = np.array([
    0.100871242,
    0.104759189,
    0.049387401,
    0.050477405,
    np.nan,
])

homology_high = np.array([
    0.121924135,
    0.222993918,
    0.118438904,
    0.150505482,
    np.nan,
])


# =========================================================
# Extract E6/E7 performance
# =========================================================

e6_conditions = ["clean", "sigma_0.5", "sigma_1.0", "sigma_2.0"]
e6_perf_map = {x["condition"]: x for x in e6_perf}

structure_ap = np.array([
    e6_perf_map[c]["structure_micro_ap"]
    for c in e6_conditions
])

e7_conditions = ["clean", "mask_10pct", "mask_20pct", "mask_40pct"]
e7_perf_map = {x["condition"]: x for x in e7_perf}

sequence_ap = np.array([
    e7_perf_map[c]["micro_average_precision"]
    for c in e7_conditions
])


# =========================================================
# Extract E6 reliability
# =========================================================

e6_rel_map = {x["condition"]: x for x in e6_rel}

structure_rel_delta = np.array([
    e6_rel_map[c]["delta_binary"]["average_precision"]
    for c in e6_conditions
])


# =========================================================
# Extract E7 reliability bootstrap
# =========================================================

e7_results = e7_boot["results"]

sequence_rel_delta = []
sequence_rel_low = []
sequence_rel_high = []

for c in e7_conditions:
    ap = e7_results[c]["binary"]["average_precision"]

    sequence_rel_delta.append(
        float(ap["observed_delta"])
    )
    sequence_rel_low.append(
        float(ap["ci95"][0])
    )
    sequence_rel_high.append(
        float(ap["ci95"][1])
    )

sequence_rel_delta = np.array(sequence_rel_delta)
sequence_rel_low = np.array(sequence_rel_low)
sequence_rel_high = np.array(sequence_rel_high)


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

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11.5, 7.5),
)

axA, axB, axC, axD = axes.flat


# =========================================================
# Panel A — low-homology robustness
# =========================================================

x = np.arange(len(homology_labels))

# Four inferential strata with bootstrap CIs
for i in range(4):
    yerr = np.array([
        [homology_delta[i] - homology_low[i]],
        [homology_high[i] - homology_delta[i]],
    ])

    axA.errorbar(
        x[i],
        homology_delta[i],
        yerr=yerr,
        fmt="o",
        capsize=4,
        linewidth=1.4,
        markersize=6,
    )

# Descriptive n=8 stratum: open marker, no CI
axA.plot(
    x[4],
    homology_delta[4],
    marker="o",
    markersize=7,
    markerfacecolor="none",
    markeredgewidth=1.5,
    linestyle="none",
)

axA.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.6,
)

axA.set_xticks(x)
axA.set_xticklabels([
    "No hit\n(n=4,194)",
    "<20%\n(n=151)",
    "20–25%\n(n=248)",
    "25–30%\n(n=130)",
    "≥30%\n(n=8; descriptive)",
])

axA.set_ylabel("ΔAP: conflict − uncertainty")
axA.set_ylim(-0.005, 0.25)

axA.set_title(
    "A  Low-homology robustness — FINAL TEST",
    loc="left",
    fontweight="bold",
)


# =========================================================
# Panel B — targeted expert degradation
# =========================================================

x_struct = np.arange(4)
x_seq = np.arange(4)

axB.plot(
    x_struct,
    structure_ap,
    marker="o",
    linewidth=1.8,
    label="Structure expert",
)

axB.plot(
    x_seq,
    sequence_ap,
    marker="s",
    linewidth=1.8,
    label="Sequence expert",
)

axB.set_xticks(x_struct)

axB.set_xticklabels([
    "Clean",
    "0.5 Å / 10%",
    "1.0 Å / 20%",
    "2.0 Å / 40%",
])

axB.set_ylabel("Micro AUPRC")
axB.set_xlabel("Coordinate noise / sequence masking")
axB.set_ylim(0.30, 0.66)

axB.set_title(
    "B  Targeted expert degradation — CAL-EVAL",
    loc="left",
    fontweight="bold",
)

axB.legend(
    frameon=False,
    loc="lower left",
)


# =========================================================
# Panel C — structure reliability under degradation
# =========================================================

x = np.arange(4)

axC.plot(
    x,
    structure_rel_delta,
    marker="o",
    linewidth=1.8,
)

axC.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.6,
)

axC.set_xticks(x)

axC.set_xticklabels([
    "Clean",
    "0.5 Å",
    "1.0 Å",
    "2.0 Å",
])

axC.set_ylabel("ΔAP: conflict − uncertainty")
axC.set_ylim(-0.01, 0.15)

axC.set_title(
    "C  Structure-side reliability — CAL-EVAL",
    loc="left",
    fontweight="bold",
)

for i, value in enumerate(structure_rel_delta):
    axC.text(
        i,
        value + 0.007,
        f"{value:+.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )


# =========================================================
# Panel D — sequence reliability under degradation
# =========================================================

x = np.arange(4)

lower_err = sequence_rel_delta - sequence_rel_low
upper_err = sequence_rel_high - sequence_rel_delta

axD.errorbar(
    x,
    sequence_rel_delta,
    yerr=np.vstack([lower_err, upper_err]),
    fmt="o-",
    capsize=4,
    linewidth=1.5,
    markersize=6,
)

axD.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.7,
)

axD.set_xticks(x)

axD.set_xticklabels([
    "Clean",
    "10%",
    "20%",
    "40%",
])

axD.set_ylabel("ΔAP: conflict − uncertainty")
axD.set_xlabel("Sequence masking")
axD.set_ylim(-0.006, 0.004)

axD.set_title(
    "D  Sequence-side reliability — CAL-EVAL",
    loc="left",
    fontweight="bold",
)

for i, value in enumerate(sequence_rel_delta):
    if value >= 0:
        y = sequence_rel_high[i] + 0.00035
        va = "bottom"
    else:
        y = sequence_rel_low[i] - 0.00035
        va = "top"

    axD.text(
        i,
        y,
        f"{value:+.4f}",
        ha="center",
        va=va,
        fontsize=7.5,
    )


# =========================================================
# Shared formatting
# =========================================================

for ax in axes.flat:
    ax.tick_params(axis="x", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        alpha=0.18,
    )

fig.suptitle(
    "Conflict reliability is robust across low homology "
    "but asymmetric across modalities",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.5,
    0.030,
    "ΔAP = AP with conflict − AP with uncertainty only; "
    "positive values indicate incremental reliability information.",
    ha="center",
    fontsize=8,
)

fig.text(
    0.5,
    0.012,
    "Whiskers: 95% whole-protein bootstrap CI where available. "
    "Panel C shows point estimates only; E6 has no bootstrap CI artifact.",
    ha="center",
    fontsize=7.5,
)

fig.tight_layout(
    rect=[0, 0.065, 1, 0.96],
    h_pad=2.4,
    w_pad=2.8,
)


# =========================================================
# Save
# =========================================================

png = OUT / "Figure_3_robustness_asymmetry.png"
pdf = OUT / "Figure_3_robustness_asymmetry.pdf"
svg = OUT / "Figure_3_robustness_asymmetry.svg"

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

print("CREATED:")
print(png)
print(pdf)
print(svg)
