from pathlib import Path
import pandas as pd

BASE = Path("data/splits/mmseqs30_s75/audit")

FILES = {
    "val_vs_train": BASE / "val_vs_train.tsv",
    "test_vs_train": BASE / "test_vs_train.tsv",
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

print("=" * 70)
print("CROSS-SPLIT HOMOLOGY AUDIT")
print("=" * 70)

summary = []

for name, path in FILES.items():
    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    if not path.exists():
        print("File not found:", path)
        continue

    if path.stat().st_size == 0:
        print("Hits: 0")
        print("PASS: No cross-split hits satisfying search criteria.")

        summary.append({
            "comparison": name,
            "hits": 0,
            "queries_with_hits": 0,
            "targets_with_hits": 0,
        })

        continue

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS
    )

    print(f"Total hits: {len(df):,}")
    print(
        f"Unique query proteins with hits: "
        f"{df['query'].nunique():,}"
    )
    print(
        f"Unique training proteins hit: "
        f"{df['target'].nunique():,}"
    )

    print("\nIdentity statistics:")
    print(df["fident"].describe())

    print("\nQuery coverage statistics:")
    print(df["qcov"].describe())

    print("\nTarget coverage statistics:")
    print(df["tcov"].describe())

    print("\nTop 20 strongest hits:")

    strongest = df.sort_values(
        ["fident", "bits"],
        ascending=[False, False]
    ).head(20)

    print(
        strongest.to_string(index=False)
    )

    summary.append({
        "comparison": name,
        "hits": len(df),
        "queries_with_hits": df["query"].nunique(),
        "targets_with_hits": df["target"].nunique(),
    })

summary_df = pd.DataFrame(summary)

summary_file = (
    BASE / "homology_audit_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(index=False)
)

print("\nSaved:")
print(summary_file)