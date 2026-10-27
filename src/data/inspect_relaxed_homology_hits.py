from pathlib import Path
import pandas as pd


BASE = Path("data/splits/mmseqs30_graph")
AUDIT = BASE / "audit_relaxed50"

FILES = {
    "val_vs_train": AUDIT / "val_vs_train.tsv",
    "test_vs_train": AUDIT / "test_vs_train.tsv",
    "test_vs_val": AUDIT / "test_vs_val.tsv",
}

COLUMNS = [
    "query",
    "target",
    "fident",
    "alnlen",
    "qcov",
    "tcov",
    "evalue",
    "bits",
]


all_rows = []


for comparison, path in FILES.items():

    df = pd.read_csv(
        path,
        sep="\t",
        names=COLUMNS
    )

    df["comparison"] = comparison

    all_rows.append(df)


hits = pd.concat(
    all_rows,
    ignore_index=True
)


# ---------------------------------------------------------
# Basic checks
# ---------------------------------------------------------

print("=" * 80)
print("RELAXED HOMOLOGY HIT INSPECTION")
print("=" * 80)

print(f"\nTotal relaxed hits: {len(hits):,}")


print("\nIdentity >= 90%:")
print(
    f"{(hits['fident'] >= 0.90).sum():,}"
)

print("\nIdentity >= 95%:")
print(
    f"{(hits['fident'] >= 0.95).sum():,}"
)

print("\nIdentity >= 99%:")
print(
    f"{(hits['fident'] >= 0.99).sum():,}"
)

print("\nIdentity == 100%:")
print(
    f"{(hits['fident'] >= 0.999999).sum():,}"
)


# ---------------------------------------------------------
# Why did these fail the primary 80% coverage criterion?
# ---------------------------------------------------------

high = hits[
    hits["fident"] >= 0.90
].copy()


high["min_coverage"] = high[
    ["qcov", "tcov"]
].min(axis=1)

high["max_coverage"] = high[
    ["qcov", "tcov"]
].max(axis=1)


print("\n" + "=" * 80)
print("HIGH-IDENTITY HITS (>=90%)")
print("=" * 80)

print(f"Number of hits: {len(high):,}")


if len(high) > 0:

    print("\nMinimum-coverage statistics:")

    print(
        high["min_coverage"].describe()
    )

    print(
        "\nHigh-identity hits with "
        "one/both coverage <80%:"
    )

    print(
        (
            high["min_coverage"] < 0.80
        ).sum()
    )


# ---------------------------------------------------------
# Show strongest examples
# ---------------------------------------------------------

top = hits.sort_values(
    [
        "fident",
        "bits"
    ],
    ascending=[
        False,
        False
    ]
).head(30)


print("\n" + "=" * 80)
print("TOP 30 RELAXED CROSS-SPLIT HITS")
print("=" * 80)

print(
    top[
        [
            "comparison",
            "query",
            "target",
            "fident",
            "alnlen",
            "qcov",
            "tcov",
            "evalue",
            "bits",
        ]
    ].to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

high.to_csv(
    AUDIT / "high_identity_relaxed_hits.csv",
    index=False
)

top.to_csv(
    AUDIT / "top30_relaxed_hits.csv",
    index=False
)


print("\nSaved:")
print(
    AUDIT /
    "high_identity_relaxed_hits.csv"
)

print(
    AUDIT /
    "top30_relaxed_hits.csv"
)