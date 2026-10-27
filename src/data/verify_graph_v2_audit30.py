from pathlib import Path
import pandas as pd


BASE = Path("data/splits/mmseqs30_graph_v2/audit30")

FILES = {
    "VAL->TRAIN": BASE / "val_vs_train.tsv",
    "TEST->TRAIN": BASE / "test_vs_train.tsv",
    "TEST->VAL": BASE / "test_vs_val.tsv",
}

SUMMARY_PATH = BASE / "graph_v2_audit_summary.csv"
VIOLATIONS_PATH = BASE / "graph_v2_primary_violations.csv"

MIN_IDENTITY = 0.30
MIN_QCOV = 0.80
MIN_TCOV = 0.80

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


def normalize_fraction(series):
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()

    if len(valid) == 0:
        return x

    if valid.max() > 1.0:
        x = x / 100.0

    return x


print("=" * 70)
print("VERIFY GRAPH-V2 AUDIT30")
print("=" * 70)

summary_rows = []
violation_frames = []

for comparison, path in FILES.items():

    print(f"\nChecking {comparison}...")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing audit file: {path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
    )

    df["fident"] = normalize_fraction(df["fident"])
    df["qcov"] = normalize_fraction(df["qcov"])
    df["tcov"] = normalize_fraction(df["tcov"])

    violations = df[
        (df["fident"] >= MIN_IDENTITY)
        & (df["qcov"] >= MIN_QCOV)
        & (df["tcov"] >= MIN_TCOV)
    ].copy()

    violations.insert(
        0,
        "comparison",
        comparison,
    )

    n_candidates = len(df)
    n_violations = len(violations)

    result = "PASS" if n_violations == 0 else "FAIL"

    summary_rows.append(
        {
            "comparison": comparison,
            "candidate_pairs": n_candidates,
            "primary_violations": n_violations,
            "result": result,
        }
    )

    if n_violations > 0:
        violation_frames.append(violations)

    print(f"Candidate pairs: {n_candidates:,}")
    print(f"Primary violations: {n_violations:,}")
    print(f"Result: {result}")


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

if violation_frames:

    all_violations = pd.concat(
        violation_frames,
        ignore_index=True,
    )

else:

    all_violations = pd.DataFrame(
        columns=["comparison", *COLUMNS]
    )


all_violations.to_csv(
    VIOLATIONS_PATH,
    index=False,
)

total_candidates = int(
    summary["candidate_pairs"].sum()
)

total_violations = int(
    summary["primary_violations"].sum()
)


print("\n" + "=" * 70)
print("AUDIT30 SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print(
    f"\nTotal candidate pairs: "
    f"{total_candidates:,}"
)

print(
    "Total primary cross-split violations: "
    f"{total_violations:,}"
)

if total_violations == 0:
    print("\nOVERALL: PASS")
else:
    print("\nOVERALL: FAIL")
    raise RuntimeError(
        "Cross-split primary homology violations detected."
    )


print("\nSaved:")
print(SUMMARY_PATH)
print(VIOLATIONS_PATH)

print("\nDONE")