from pathlib import Path
import pandas as pd

BATCH_DIR = Path(
    "data/xai/functional_sites_v1/uniprot_batches"
)

SITE_FILE = Path(
    "data/xai/functional_sites_v1/functional_sites_raw.csv"
)

BENCHMARK_FILE = Path(
    "data/splits/mmseqs30_graph_v2/test.pkl"
)

OUT_DIR = Path(
    "data/xai/functional_sites_v1"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Load exact benchmark TEST sequences
# ------------------------------------------------------------

bench = pd.read_pickle(BENCHMARK_FILE)

assert "uniprot_id" in bench.columns
assert "sequence" in bench.columns

benchmark_seq = dict(
    zip(
        bench["uniprot_id"].astype(str),
        bench["sequence"].astype(str),
    )
)

# ------------------------------------------------------------
# 2. Parse sequences from downloaded UniProt records
# ------------------------------------------------------------

uniprot_seq = {}

for path in sorted(BATCH_DIR.glob("batch_*.dat")):

    accession = None
    sequence_parts = []
    in_sequence = False

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            if line.startswith("AC   ") and accession is None:
                values = [
                    x.strip()
                    for x in line[5:].split(";")
                    if x.strip()
                ]

                if values:
                    accession = values[0]

            elif line.startswith("SQ   "):
                in_sequence = True
                sequence_parts = []

            elif in_sequence and line.startswith("     "):
                part = "".join(
                    x for x in line.strip()
                    if x.isalpha()
                )
                sequence_parts.append(part)

            elif line.rstrip() == "//":

                if accession is not None:
                    uniprot_seq[accession] = "".join(
                        sequence_parts
                    )

                accession = None
                sequence_parts = []
                in_sequence = False

# ------------------------------------------------------------
# 3. Validate proteins carrying functional annotations
# ------------------------------------------------------------

sites = pd.read_csv(SITE_FILE)

site_ids = sorted(
    sites["uniprot_id"]
    .dropna()
    .astype(str)
    .unique()
)

rows = []

for uid in site_ids:

    bseq = benchmark_seq.get(uid)
    useq = uniprot_seq.get(uid)

    benchmark_present = bseq is not None
    uniprot_present = useq is not None

    exact_match = (
        benchmark_present
        and uniprot_present
        and bseq == useq
    )

    rows.append(
        {
            "uniprot_id": uid,
            "benchmark_present": benchmark_present,
            "uniprot_present": uniprot_present,
            "benchmark_length":
                len(bseq) if bseq is not None else None,
            "uniprot_length":
                len(useq) if useq is not None else None,
            "exact_sequence_match": exact_match,
        }
    )

qc = pd.DataFrame(rows)

qc.to_csv(
    OUT_DIR / "functional_site_sequence_qc.csv",
    index=False,
)

# ------------------------------------------------------------
# 4. Coordinate bounds check
# ------------------------------------------------------------

length_map = {
    uid: len(seq)
    for uid, seq in benchmark_seq.items()
}

sites["benchmark_length"] = (
    sites["uniprot_id"].map(length_map)
)

sites["coordinate_in_bounds"] = (
    sites["coordinate_valid"].astype(bool)
    & sites["start"].notna()
    & sites["end"].notna()
    & (sites["start"] >= 1)
    & (sites["end"] >= sites["start"])
    & (sites["end"] <= sites["benchmark_length"])
)

match_map = dict(
    zip(
        qc["uniprot_id"],
        qc["exact_sequence_match"],
    )
)

sites["exact_sequence_match"] = (
    sites["uniprot_id"]
    .map(match_map)
    .fillna(False)
)

sites["retain_primary"] = (
    sites["exact_sequence_match"]
    & sites["coordinate_in_bounds"]
)

sites.to_csv(
    OUT_DIR / "functional_sites_qc.csv",
    index=False,
)

retained = sites[
    sites["retain_primary"]
].copy()

retained.to_csv(
    OUT_DIR / "functional_sites_primary.csv",
    index=False,
)

# ------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------

n_exact = int(qc["exact_sequence_match"].sum())
n_mismatch = len(qc) - n_exact

n_bounds = int(
    sites["coordinate_in_bounds"].sum()
)

n_retained = int(
    sites["retain_primary"].sum()
)

retained_proteins = retained[
    "uniprot_id"
].nunique()

print("=" * 72)
print("E10 FUNCTIONAL-SITE SEQUENCE / COORDINATE QC")
print("=" * 72)

print("Annotated proteins:", len(site_ids))
print("Exact sequence matches:", n_exact)
print("Sequence mismatches/missing:", n_mismatch)

print()
print("Raw functional features:", len(sites))
print("Features within benchmark bounds:", n_bounds)
print("Primary retained features:", n_retained)
print(
    "Proteins retained for primary reference:",
    retained_proteins,
)

print()
print("ACT_SITE retained:",
      (retained["feature_type"] == "ACT_SITE").sum())

print("BINDING retained:",
      (retained["feature_type"] == "BINDING").sum())

print()

if n_mismatch:
    print("Sequence mismatch examples:")
    print(
        qc.loc[
            ~qc["exact_sequence_match"]
        ].head(20).to_string(index=False)
    )

print()
print("Outputs:")
print(
    " data/xai/functional_sites_v1/"
    "functional_site_sequence_qc.csv"
)
print(
    " data/xai/functional_sites_v1/"
    "functional_sites_qc.csv"
)
print(
    " data/xai/functional_sites_v1/"
    "functional_sites_primary.csv"
)

print()
print("No model predictions modified.")
print("No XAI results computed.")
print("=" * 72)