from pathlib import Path
import pandas as pd


AUDIT_DIR = Path(
    "data/splits/mmseqs30_graph/audit_relaxed50"
)

FILES = {
    "val_vs_train": AUDIT_DIR / "val_vs_train.tsv",
    "test_vs_train": AUDIT_DIR / "test_vs_train.tsv",
    "test_vs_val": AUDIT_DIR / "test_vs_val.tsv",
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


rows = []


print("=" * 75)
print("RELAXED HOMOLOGY AUDIT")
print("Identity >= 30%, bidirectional coverage >= 50%")
print("=" * 75)


for comparison, path in FILES.items():

    print(f"\n{comparison.upper()}")
    print("-" * 75)

    if not path.exists():

        print("FILE NOT FOUND")

        rows.append({
            "comparison": comparison,
            "qualifying_hits": None,
            "query_proteins_with_hits": None,
            "target_proteins_with_hits": None,
            "mean_identity": None,
            "max_identity": None,
            "mean_qcov": None,
            "mean_tcov": None,
        })

        continue


    if path.stat().st_size == 0:

        print("Qualifying hits: 0")

        rows.append({
            "comparison": comparison,
            "qualifying_hits": 0,
            "query_proteins_with_hits": 0,
            "target_proteins_with_hits": 0,
            "mean_identity": None,
            "max_identity": None,
            "mean_qcov": None,
            "mean_tcov": None,
        })

        continue


    df = pd.read_csv(
        path,
        sep="\t",
        names=COLUMNS
    )


    print(
        f"Qualifying hits: "
        f"{len(df):,}"
    )

    print(
        f"Query proteins with hits: "
        f"{df['query'].nunique():,}"
    )

    print(
        f"Target proteins with hits: "
        f"{df['target'].nunique():,}"
    )

    print(
        f"Mean identity: "
        f"{df['fident'].mean():.4f}"
    )

    print(
        f"Maximum identity: "
        f"{df['fident'].max():.4f}"
    )

    print(
        f"Mean query coverage: "
        f"{df['qcov'].mean():.4f}"
    )

    print(
        f"Mean target coverage: "
        f"{df['tcov'].mean():.4f}"
    )


    rows.append({
        "comparison": comparison,

        "qualifying_hits":
            len(df),

        "query_proteins_with_hits":
            df["query"].nunique(),

        "target_proteins_with_hits":
            df["target"].nunique(),

        "mean_identity":
            df["fident"].mean(),

        "max_identity":
            df["fident"].max(),

        "mean_qcov":
            df["qcov"].mean(),

        "mean_tcov":
            df["tcov"].mean(),
    })


summary = pd.DataFrame(rows)


print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

print(
    summary.to_string(
        index=False
    )
)


OUTPUT = (
    AUDIT_DIR /
    "relaxed_homology_audit_summary.csv"
)

summary.to_csv(
    OUTPUT,
    index=False
)


print("\nSaved:")
print(OUTPUT)