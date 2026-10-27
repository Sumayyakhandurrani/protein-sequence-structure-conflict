from pathlib import Path
import pandas as pd


INPUT = Path(
    "data/processed/structure_ready_proteins.pkl"
)

OUTPUT = Path(
    "data/processed/structure_ready_qc_summary.csv"
)


df = pd.read_pickle(INPUT)

df["global_metric"] = pd.to_numeric(
    df["global_metric"],
    errors="coerce"
)


print("=" * 60)
print("STRUCTURE-READY QC ANALYSIS")
print("=" * 60)

print(f"\nTotal proteins: {len(df):,}")


# =========================================================
# LENGTH DISTRIBUTION
# =========================================================

print("\n" + "=" * 60)
print("LENGTH DISTRIBUTION")
print("=" * 60)

thresholds = [
    30,
    50,
    100,
    500,
    1000,
    1500,
    2000,
    3000,
    5000,
    10000,
]

for t in thresholds:

    below_equal = (df["length"] <= t).sum()

    pct = (
        100 * below_equal / len(df)
    )

    print(
        f"Length <= {t:5d}: "
        f"{below_equal:6,d} "
        f"({pct:6.2f}%)"
    )


print("\nLong proteins:")

for t in [1000, 1500, 2000, 3000, 5000]:

    n = (df["length"] > t).sum()

    print(
        f"Length > {t:4d}: "
        f"{n:6,d} "
        f"({100*n/len(df):6.2f}%)"
    )


print("\nVery short proteins:")

for t in [30, 50, 100]:

    n = (df["length"] < t).sum()

    print(
        f"Length < {t:3d}: "
        f"{n:6,d} "
        f"({100*n/len(df):6.2f}%)"
    )


# =========================================================
# LENGTH QUANTILES
# =========================================================

print("\nLength quantiles:")

for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
]:

    value = df["length"].quantile(q)

    print(
        f"{q*100:6.1f}% : "
        f"{value:8.1f}"
    )


# =========================================================
# ALPHAFOLD CONFIDENCE
# =========================================================

print("\n" + "=" * 60)
print("ALPHAFOLD GLOBAL CONFIDENCE")
print("=" * 60)

confidence_thresholds = [
    50,
    60,
    70,
    80,
    90,
]

for t in confidence_thresholds:

    below = (
        df["global_metric"] < t
    ).sum()

    print(
        f"Global metric < {t}: "
        f"{below:6,d} "
        f"({100*below/len(df):6.2f}%)"
    )


print("\nConfidence bands:")

bands = [
    (0, 50),
    (50, 70),
    (70, 90),
    (90, 101),
]

for low, high in bands:

    mask = (
        (df["global_metric"] >= low)
        &
        (df["global_metric"] < high)
    )

    n = mask.sum()

    print(
        f"{low:2d} <= metric < {high:3d}: "
        f"{n:6,d} "
        f"({100*n/len(df):6.2f}%)"
    )


# =========================================================
# COMBINED CANDIDATE LENGTH POLICIES
# =========================================================

print("\n" + "=" * 60)
print("CANDIDATE LENGTH POLICIES")
print("=" * 60)

policies = [
    (30, 1000),
    (30, 1500),
    (30, 2000),
    (50, 1000),
    (50, 1500),
    (50, 2000),
]

summary = []

for minimum, maximum in policies:

    mask = (
        (df["length"] >= minimum)
        &
        (df["length"] <= maximum)
    )

    subset = df[mask]

    n = len(subset)

    pct = 100 * n / len(df)

    summary.append({
        "min_length": minimum,
        "max_length": maximum,
        "proteins": n,
        "percent_retained": pct,
        "mean_global_metric":
            subset["global_metric"].mean(),
        "median_length":
            subset["length"].median(),
    })

    print(
        f"{minimum:3d}-{maximum:4d}: "
        f"{n:6,d} proteins "
        f"({pct:6.2f}%)"
    )


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT,
    index=False
)


# =========================================================
# EXTREME PROTEINS
# =========================================================

print("\n" + "=" * 60)
print("TOP 20 LONGEST PROTEINS")
print("=" * 60)

cols = [
    "uniprot_id",
    "length",
    "global_metric",
    "n_propagated_labels",
]

print(
    df.nlargest(
        20,
        "length"
    )[cols].to_string(
        index=False
    )
)


print(
    "\nSaved policy summary to:"
)

print(OUTPUT)