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


all_violations = []


print("=" * 80)
print("PRIMARY-CRITERION VIOLATIONS FOUND BY RELAXED SEARCH")
print("Required: identity >= 30%, qcov >= 80%, tcov >= 80%")
print("=" * 80)


for comparison, path in FILES.items():

    df = pd.read_csv(
        path,
        sep="\t",
        names=COLUMNS
    )

    # relaxed search already required fident >= 0.30,
    # but explicitly apply all primary conditions again
    violations = df[
        (df["fident"] >= 0.30)
        &
        (df["qcov"] >= 0.80)
        &
        (df["tcov"] >= 0.80)
    ].copy()

    violations["comparison"] = comparison

    all_violations.append(violations)

    print(f"\n{comparison.upper()}")
    print("-" * 80)

    print(
        f"Primary-criterion hits found: "
        f"{len(violations):,}"
    )

    print(
        f"Unique query proteins: "
        f"{violations['query'].nunique():,}"
    )

    print(
        f"Unique target proteins: "
        f"{violations['target'].nunique():,}"
    )

    if len(violations) > 0:

        print(
            f"Identity range: "
            f"{violations['fident'].min():.3f} - "
            f"{violations['fident'].max():.3f}"
        )

        print(
            f"qcov range: "
            f"{violations['qcov'].min():.3f} - "
            f"{violations['qcov'].max():.3f}"
        )

        print(
            f"tcov range: "
            f"{violations['tcov'].min():.3f} - "
            f"{violations['tcov'].max():.3f}"
        )


violations = pd.concat(
    all_violations,
    ignore_index=True
)


print("\n" + "=" * 80)
print("TOTAL")
print("=" * 80)

print(
    f"Total primary-criterion cross-split hits: "
    f"{len(violations):,}"
)

print(
    f"Unique query proteins involved: "
    f"{violations['query'].nunique():,}"
)

print(
    f"Unique target proteins involved: "
    f"{violations['target'].nunique():,}"
)


if len(violations) > 0:

    top = violations.sort_values(
        ["fident", "bits"],
        ascending=[False, False]
    )

    print("\nTop 30 violations:")

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
        ].head(30).to_string(index=False)
    )


OUTPUT = (
    AUDIT_DIR /
    "primary_criterion_violations_from_relaxed_search.csv"
)

violations.to_csv(
    OUTPUT,
    index=False
)


print("\nSaved:")
print(OUTPUT)