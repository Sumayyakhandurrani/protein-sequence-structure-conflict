from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

# ============================================================
# Frozen constants
# ============================================================

BASE = Path("data/xai/functional_sites_v1/conflict_deletion_v1")
PER_PROTEIN = BASE / "per_protein"

PROTOCOL = BASE / "AGGREGATION_PROTOCOL_V1.txt"
MANIFEST = BASE / "PER_PROTEIN_SHA256.txt"

EXPECTED_PROTOCOL_SHA = (
    "b96a20097f80fe7dafbee2d66cf482b3994e35ff349a96945d6ca4c283b795d9"
)

EXPECTED_MANIFEST_SHA = (
    "47df91321507008bbec4957c73e082deb8d04278b27a590169dff101cf784416"
)

EXPECTED_PROTEINS = 2447
EXPECTED_TARGETS = 23087

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42

LABEL_OUT = BASE / "conflict_deletion_label_level_v1.csv"
PROTEIN_OUT = BASE / "conflict_deletion_protein_level_v1.csv"
SUMMARY_OUT = BASE / "conflict_deletion_summary_v1.csv"
RUNINFO_OUT = BASE / "AGGREGATION_RUN_INFO_V1.txt"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================
# Verify frozen inputs
# ============================================================

protocol_sha = sha256_file(PROTOCOL)
manifest_sha = sha256_file(MANIFEST)

if protocol_sha != EXPECTED_PROTOCOL_SHA:
    raise RuntimeError(
        f"Aggregation protocol SHA mismatch:\n"
        f"expected {EXPECTED_PROTOCOL_SHA}\n"
        f"observed {protocol_sha}"
    )

if manifest_sha != EXPECTED_MANIFEST_SHA:
    raise RuntimeError(
        f"Production manifest SHA mismatch:\n"
        f"expected {EXPECTED_MANIFEST_SHA}\n"
        f"observed {manifest_sha}"
    )

print("Aggregation protocol: SHA VERIFIED")
print("Production manifest: SHA VERIFIED")


# ============================================================
# Verify every production NPZ against frozen manifest
# ============================================================

manifest_entries = []

for line in MANIFEST.read_text().splitlines():
    line = line.strip()
    if not line:
        continue

    digest, filename = line.split(maxsplit=1)
    filename = filename.strip()

    manifest_entries.append((digest, filename))

if len(manifest_entries) != EXPECTED_PROTEINS:
    raise RuntimeError(
        f"Manifest contains {len(manifest_entries)} files; "
        f"expected {EXPECTED_PROTEINS}"
    )

actual_files = sorted(p.name for p in PER_PROTEIN.glob("*.npz"))
manifest_files = sorted(filename for _, filename in manifest_entries)

if actual_files != manifest_files:
    raise RuntimeError(
        "Current per-protein file set differs from frozen manifest."
    )

for expected_sha, filename in manifest_entries:
    p = PER_PROTEIN / filename

    observed_sha = sha256_file(p)

    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Frozen production file SHA mismatch: {filename}"
        )

print(f"Frozen production files verified: {len(manifest_entries)}")


# ============================================================
# Read label-level results
# ============================================================

label_rows = []
seen_proteins = set()
seen_pairs = set()

for _, filename in manifest_entries:
    p = PER_PROTEIN / filename

    with np.load(p, allow_pickle=False) as z:
        uid = str(z["uniprot_id"].item())
        labels = np.asarray(z["positive_label_indices"], dtype=np.int64)

        top_reduction = np.asarray(
            z["top_conflict_reduction"], dtype=np.float64
        )

        random_reduction = np.asarray(
            z["random_mean_conflict_reduction"], dtype=np.float64
        )

        stored_enrichment = np.asarray(
            z["conflict_deletion_enrichment"], dtype=np.float64
        )

        if uid in seen_proteins:
            raise RuntimeError(f"Duplicate protein: {uid}")

        seen_proteins.add(uid)

        n = len(labels)

        if not (
            top_reduction.shape == (n,)
            and random_reduction.shape == (n,)
            and stored_enrichment.shape == (n,)
        ):
            raise RuntimeError(f"Shape mismatch: {uid}")

        if not (
            np.all(np.isfinite(top_reduction))
            and np.all(np.isfinite(random_reduction))
            and np.all(np.isfinite(stored_enrichment))
        ):
            raise RuntimeError(f"Non-finite scientific value: {uid}")

        recomputed = top_reduction - random_reduction

        # Float32 storage can introduce tiny round-off differences.
        if not np.allclose(
            stored_enrichment,
            recomputed,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"Stored enrichment identity failed: {uid}"
            )

        for j, label_idx in enumerate(labels):
            pair = (uid, int(label_idx))

            if pair in seen_pairs:
                raise RuntimeError(
                    f"Duplicate protein-label target: {pair}"
                )

            seen_pairs.add(pair)

            label_rows.append(
                {
                    "uniprot_id": uid,
                    "label_index": int(label_idx),
                    "top_conflict_reduction": float(top_reduction[j]),
                    "random_mean_conflict_reduction": float(
                        random_reduction[j]
                    ),
                    "conflict_deletion_enrichment": float(
                        stored_enrichment[j]
                    ),
                }
            )


label_df = pd.DataFrame(label_rows)

if len(seen_proteins) != EXPECTED_PROTEINS:
    raise RuntimeError(
        f"Observed {len(seen_proteins)} proteins; "
        f"expected {EXPECTED_PROTEINS}"
    )

if len(label_df) != EXPECTED_TARGETS:
    raise RuntimeError(
        f"Observed {len(label_df)} targets; "
        f"expected {EXPECTED_TARGETS}"
    )

print(f"Proteins validated: {len(seen_proteins)}")
print(f"Protein-label targets validated: {len(label_df)}")


# ============================================================
# Protein-level aggregation
# ============================================================

protein_df = (
    label_df
    .groupby("uniprot_id", sort=True)
    .agg(
        n_positive_labels=("label_index", "size"),
        mean_conflict_deletion_enrichment=(
            "conflict_deletion_enrichment", "mean"
        ),
    )
    .reset_index()
)

if len(protein_df) != EXPECTED_PROTEINS:
    raise RuntimeError(
        f"Protein aggregation produced {len(protein_df)} rows; "
        f"expected {EXPECTED_PROTEINS}"
    )

protein_values = protein_df[
    "mean_conflict_deletion_enrichment"
].to_numpy(dtype=np.float64)

if not np.all(np.isfinite(protein_values)):
    raise RuntimeError("Non-finite protein-level enrichment detected.")


# ============================================================
# Frozen whole-protein bootstrap
# ============================================================

rng = np.random.default_rng(BOOTSTRAP_SEED)

bootstrap_means = np.empty(
    BOOTSTRAP_REPLICATES,
    dtype=np.float64,
)

n_proteins = len(protein_values)

for b in range(BOOTSTRAP_REPLICATES):
    indices = rng.integers(
        0,
        n_proteins,
        size=n_proteins,
    )

    bootstrap_means[b] = np.mean(
        protein_values[indices]
    )

if len(bootstrap_means) != BOOTSTRAP_REPLICATES:
    raise RuntimeError("Bootstrap replicate count mismatch.")

ci_low, ci_high = np.percentile(
    bootstrap_means,
    [2.5, 97.5],
)

mean_value = float(np.mean(protein_values))
median_value = float(np.median(protein_values))
fraction_positive = float(np.mean(protein_values > 0))


# ============================================================
# Summary
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "n_proteins": EXPECTED_PROTEINS,
            "n_protein_label_targets": EXPECTED_TARGETS,
            "mean_protein_enrichment": mean_value,
            "median_protein_enrichment": median_value,
            "fraction_proteins_positive": fraction_positive,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "mean_enrichment_ci95_low": float(ci_low),
            "mean_enrichment_ci95_high": float(ci_high),
        }
    ]
)


# ============================================================
# Write outputs
# ============================================================

label_df.to_csv(LABEL_OUT, index=False)
protein_df.to_csv(PROTEIN_OUT, index=False)
summary_df.to_csv(SUMMARY_OUT, index=False)

implementation_sha = sha256_file(Path(__file__))

run_info = f"""E10 CONFLICT-DELETION AGGREGATION V1
========================================

STATUS: COMPLETE

Proteins: {EXPECTED_PROTEINS}
Protein-label targets: {EXPECTED_TARGETS}

Frozen production manifest SHA256:
{manifest_sha}

Aggregation protocol SHA256:
{protocol_sha}

Aggregation implementation SHA256:
{implementation_sha}

Bootstrap unit: protein
Bootstrap replicates: {BOOTSTRAP_REPLICATES}
Bootstrap seed: {BOOTSTRAP_SEED}
CI: percentile 2.5%, 97.5%

Validation:
- frozen manifest verified
- every NPZ SHA verified
- protein count verified
- target count verified
- unique protein IDs verified
- unique protein-label targets verified
- finite scientific values verified
- enrichment algebra verified
- protein-level equal weighting verified
- bootstrap replicate count verified

No frozen production NPZ was modified.
"""

RUNINFO_OUT.write_text(run_info)


# ============================================================
# Scientific result
# ============================================================

print()
print("=" * 72)
print("E10 CONFLICT-DELETION V1 — PRIMARY RESULT")
print("=" * 72)
print(f"Proteins: {EXPECTED_PROTEINS}")
print(f"Protein-label targets: {EXPECTED_TARGETS}")
print(f"Mean protein enrichment: {mean_value:.10f}")
print(f"Median protein enrichment: {median_value:.10f}")
print(f"Fraction proteins > 0: {fraction_positive:.10f}")
print(
    "95% whole-protein bootstrap CI for mean: "
    f"[{ci_low:.10f}, {ci_high:.10f}]"
)
print("=" * 72)

print()
print("Outputs written:")
print(LABEL_OUT)
print(PROTEIN_OUT)
print(SUMMARY_OUT)
print(RUNINFO_OUT)
