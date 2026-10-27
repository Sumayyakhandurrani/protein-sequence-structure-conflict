from pathlib import Path
import pandas as pd

AUDIT_DIR = Path(
    "data/splits/mmseqs30_graph/audit"
)

FILES = {
    "val_vs_train": AUDIT_DIR / "val_vs_train.tsv",
    "test_vs_train": AUDIT_DIR / "test_vs_train.tsv",
    "test_vs_val": AUDIT_DIR / "test_vs_val.tsv",
}

rows = []

for comparison, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        hits = 0
        queries = 0
        targets = 0
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None
        )

        hits = len(df)
        queries = df[0].nunique()
        targets = df[1].nunique()

    rows.append({
        "comparison": comparison,
        "qualifying_hits": hits,
        "query_proteins_with_hits": queries,
        "target_proteins_with_hits": targets,
    })


summary = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("FINAL PRIMARY HOMOLOGY AUDIT")
print("=" * 70)

print(summary.to_string(index=False))

if (summary["qualifying_hits"] == 0).all():
    print(
        "\nPASS: No cross-partition qualifying "
        "homology hits detected."
    )
else:
    print(
        "\nFAIL: Cross-partition homology "
        "violations detected."
    )


output = (
    AUDIT_DIR /
    "homology_audit_summary.csv"
)

summary.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)