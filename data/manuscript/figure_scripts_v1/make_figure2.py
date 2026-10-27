from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "data/manuscript/figures_v1"
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================
# Load frozen source artifacts
# =========================================================

with open(
    ROOT / "data/final_evaluation/final_test_primary_v1/primary_results.json"
) as f:
    primary = json.load(f)

rel_boot = pd.read_csv(
    ROOT
    / "data/final_evaluation/final_test_primary_v1/bootstrap/"
      "reliability_bootstrap_summary.csv"
)

arb_boot = pd.read_csv(
    ROOT
    / "data/final_evaluation/final_test_primary_v1/bootstrap/"
      "arbitration_bootstrap_summary.csv"
)

randomized = pd.read_csv(
    ROOT
    / "data/analysis/randomized_conflict_control_v1/"
      "randomized_conflict_summary.csv"
)

with open(
    ROOT
    / "data/conflict/high_confidence_conflict_v1/final_test_results.json"
) as f:
    highconf = json.load(f)


# =========================================================
# Extract frozen values
# =========================================================

# Panel A — primary structure reliability
rel = primary["primary_reliability"]

ma_ap = float(rel["MA_binary"]["average_precision"])
mb_ap = float(rel["MB_binary"]["average_precision"])

row = rel_boot.loc[
    rel_boot["metric"] == "average_precision"
].iloc[0]

rel_delta = float(row["observed_delta"])
rel_low = float(row["ci_2.5"])
rel_high = float(row["ci_97.5"])


# Panel B — randomized-conflict control
row = randomized.loc[
    randomized["expert"] == "structure"
].iloc[0]

rand_ma = float(row["MA_AP"])
rand_shuffled = float(row["MB_random_AP"])
rand_real = float(row["MB_real_AP"])
real_minus_random = float(row["real_minus_random_AP"])


# Panel C — arbitration
arb = primary["arbitration_v1"]["results"]

a0 = float(arb["A0_sequence"]["micro_ap"])
a1 = float(arb["A1_uncertainty_gate"]["micro_ap"])
a2 = float(arb["A2_conflict_gate"]["micro_ap"])

row = arb_boot.loc[
    arb_boot["metric"] == "micro_ap"
].iloc[0]

arb_delta = float(row["observed_delta"])
arb_low = float(row["ci_2.5"])
arb_high = float(row["ci_97.5"])


# Panel D — high-confidence conflict
hc = highconf["primary_results"]

seq_better = 100.0 * float(hc["sequence_better_fraction"])
struct_better = 100.0 * float(hc["structure_better_fraction"])
hc_pairs = int(hc["pairs"])
hc_fraction = float(hc["fraction_all_pairs_percent"])


# =========================================================
# Figure setup
# =========================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

axA, axB, axC, axD = axes.flat


# =========================================================
# Panel A
# =========================================================

labels = [
    "Uncertainty\n(MA)",
    "+ JS conflict\n(MB)",
]

values = [ma_ap, mb_ap]

bars = axA.bar(labels, values)

axA.set_ylabel("Average precision")
axA.set_ylim(0, 0.56)

axA.set_title(
    "A  Structure-failure reliability",
    loc="left",
    fontweight="bold",
)

for bar, value in zip(bars, values):
    axA.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.012,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

axA.text(
    0.03,
    0.96,
    f"ΔAP = {rel_delta:+.3f}  "
    f"[95% CI {rel_low:.3f}, {rel_high:.3f}]",
    transform=axA.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
)



# =========================================================
# Panel B
# =========================================================

labels = [
    "Uncertainty\nonly",
    "+ shuffled\nconflict",
    "+ paired\nconflict",
]

values = [
    rand_ma,
    rand_shuffled,
    rand_real,
]

bars = axB.bar(labels, values)

axB.set_ylabel("Average precision")
axB.set_ylim(0, 0.52)

axB.set_title(
    "B  Randomized-conflict control",
    loc="left",
    fontweight="bold",
)

for bar, value in zip(bars, values):
    axB.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.010,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

axB.text(
    0.03,
    0.96,
    f"Paired − shuffled: ΔAP = {real_minus_random:+.3f}",
    transform=axB.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
)



# =========================================================
# Panel C
# =========================================================

labels = [
    "Sequence\n(A0)",
    "Uncertainty gate\n(A1)",
    "Conflict gate\n(A2)",
]

values = [a0, a1, a2]

bars = axC.bar(labels, values)

axC.set_ylabel("Micro AUPRC")
axC.set_ylim(0, 0.70)

axC.set_title(
    "C  Conflict-aware arbitration",
    loc="left",
    fontweight="bold",
)

for bar, value in zip(bars, values):
    axC.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.012,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

axC.text(
    0.03,
    0.96,
    f"A2 − A1 = {arb_delta:+.3f}  "
    f"[95% CI {arb_low:.3f}, {arb_high:.3f}]",
    transform=axC.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
)



# =========================================================
# Panel D
# =========================================================

axD.barh(
    ["Closer to\nground truth"],
    [seq_better],
    label="Sequence",
)

axD.barh(
    ["Closer to\nground truth"],
    [struct_better],
    left=[seq_better],
    label="Structure",
)

axD.set_xlim(0, 100)

axD.set_xlabel(
    "High-confidence conflict pairs (%)"
)

axD.set_title(
    "D  High-confidence conflict",
    loc="left",
    fontweight="bold",
)

axD.text(
    seq_better / 2,
    0,
    f"Sequence\n{seq_better:.1f}%",
    ha="center",
    va="center",
)

axD.text(
    seq_better + struct_better / 2,
    0,
    f"Structure\n{struct_better:.1f}%",
    ha="center",
    va="center",
)

axD.text(
    0.02,
    0.12,
    f"n = {hc_pairs:,} pairs\n"
    f"{hc_fraction:.3f}% of paired TEST protein-label pairs",
    transform=axD.transAxes,
    ha="left",
    va="bottom",
    fontsize=8,
)



# =========================================================
# Shared formatting
# =========================================================

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        alpha=0.18,
    )

fig.suptitle(
    "Sequence–structure conflict provides incremental "
    "reliability information",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.5,
    0.01,
    "Frozen outputs; intervals shown are whole-protein "
    "bootstrap 95% confidence intervals.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.06, 1, 0.96],
    h_pad=1.8,
    w_pad=2.5,
)


# =========================================================
# Save
# =========================================================

png = OUT / "Figure_2_core_conflict_reliability.png"
pdf = OUT / "Figure_2_core_conflict_reliability.pdf"
svg = OUT / "Figure_2_core_conflict_reliability.svg"

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
