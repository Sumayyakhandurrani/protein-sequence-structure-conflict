from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

BASE = Path("data")

STABILITY_DIR = (
    BASE
    / "xai/functional_sites_v1/"
      "explanation_stability_v1"
)

POPULATION_PATH = (
    STABILITY_DIR
    / "stability_population_ids.txt"
)

SEQUENCE_DIR = (
    STABILITY_DIR
    / "per_protein/sequence"
)

STRUCTURE_DIR = (
    STABILITY_DIR
    / "per_protein/structure"
)

SEQUENCE_MANIFEST = (
    STABILITY_DIR
    / "SEQUENCE_PER_PROTEIN_SHA256.txt"
)

STRUCTURE_MANIFEST = (
    STABILITY_DIR
    / "STRUCTURE_PER_PROTEIN_SHA256.txt"
)

OUT_DIR = (
    STABILITY_DIR
    / "aggregation_v1"
)

EXPECTED_PROTEINS = 200
EXPECTED_TARGETS = 1770

EXPECTED_SEQUENCE_MANIFEST_SHA = (
    "7ac635b5574b1e4260250fee7c7bf83fffbe42b5c001679cb08e6d8a10ea10aa"
)

EXPECTED_STRUCTURE_MANIFEST_SHA = (
    "af66478f469af0f81f669b3497520cbfbfe0e44dc627fce3fbbf945ae484c1b2"
)

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def verify_manifest_hashes(
    manifest_path,
    expected_manifest_sha,
):

    actual_manifest_sha = sha256_file(
        manifest_path
    )

    if actual_manifest_sha != expected_manifest_sha:

        raise RuntimeError(
            f"Manifest SHA mismatch:\n"
            f"{manifest_path}\n"
            f"expected={expected_manifest_sha}\n"
            f"actual={actual_manifest_sha}"
        )


    lines = [
        line.strip()
        for line in manifest_path
        .read_text()
        .splitlines()
        if line.strip()
    ]


    if len(lines) != EXPECTED_PROTEINS:

        raise RuntimeError(
            f"{manifest_path}: expected "
            f"{EXPECTED_PROTEINS} entries, "
            f"found {len(lines)}."
        )


    for line in lines:

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            raise RuntimeError(
                f"Malformed manifest line: {line}"
            )

        expected_sha, file_name = parts

        file_path = Path(file_name.strip())

        if not file_path.exists():
            raise RuntimeError(
                f"Missing frozen output: "
                f"{file_path}"
            )

        actual_sha = sha256_file(
            file_path
        )

        if actual_sha != expected_sha:

            raise RuntimeError(
                f"Frozen output SHA mismatch:\n"
                f"{file_path}"
            )


def percentile_ci(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return np.nan, np.nan


    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = len(values)

    boot_means = np.empty(
        BOOTSTRAP_RESAMPLES,
        dtype=np.float64,
    )


    for b in range(
        BOOTSTRAP_RESAMPLES
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        boot_means[b] = np.mean(
            values[indices]
        )


    low, high = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    return float(low), float(high)


def load_modality(
    modality,
    directory,
    population_ids,
):

    target_rows = []
    protein_rows = []

    attempted_targets = 0
    eligible_targets = 0


    for uid in population_ids:

        path = (
            directory
            / f"{uid}.npz"
        )

        if not path.exists():

            raise RuntimeError(
                f"{modality}: missing output "
                f"for {uid}"
            )


        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            file_uid = str(
                z["uniprot_id"].item()
            )

            labels = np.asarray(
                z["positive_label_indices"],
                dtype=np.int32,
            )

            eligible = np.asarray(
                z["eligible"],
                dtype=bool,
            )

            spearman = np.asarray(
                z["spearman"],
                dtype=np.float64,
            )

            jaccard = np.asarray(
                z["top10_jaccard"],
                dtype=np.float64,
            )


        if file_uid != uid:

            raise RuntimeError(
                f"{modality}: ID mismatch "
                f"for {uid}"
            )


        n_targets = len(labels)

        if not (
            eligible.shape
            == spearman.shape
            == jaccard.shape
            == (n_targets,)
        ):

            raise RuntimeError(
                f"{modality}: shape mismatch "
                f"for {uid}"
            )


        attempted_targets += n_targets

        eligible_idx = np.flatnonzero(
            eligible
        )

        eligible_targets += len(
            eligible_idx
        )


        protein_spearman = []
        protein_jaccard = []


        for i in range(n_targets):

            row = {
                "modality": modality,
                "uniprot_id": uid,
                "label_index": int(
                    labels[i]
                ),
                "eligible": bool(
                    eligible[i]
                ),
                "spearman": float(
                    spearman[i]
                ),
                "top10_jaccard": float(
                    jaccard[i]
                ),
            }

            target_rows.append(row)


        for i in eligible_idx:

            # Undefined Spearman values are retained
            # as NaN at target level, as required by
            # the frozen protocol. They are not
            # converted into numerical values.
            if np.isfinite(
                spearman[i]
            ):
                protein_spearman.append(
                    float(spearman[i])
                )

            if not np.isfinite(
                jaccard[i]
            ):
                raise RuntimeError(
                    f"{modality}: non-finite "
                    f"eligible Jaccard for {uid}"
                )

            protein_jaccard.append(
                float(jaccard[i])
            )


        if len(eligible_idx) > 0:

            mean_spearman = (
                float(
                    np.mean(
                        protein_spearman
                    )
                )
                if protein_spearman
                else np.nan
            )

            mean_jaccard = float(
                np.mean(
                    protein_jaccard
                )
            )

            protein_rows.append(
                {
                    "modality": modality,
                    "uniprot_id": uid,
                    "attempted_targets": (
                        n_targets
                    ),
                    "eligible_targets": (
                        len(eligible_idx)
                    ),
                    "finite_spearman_targets": (
                        len(
                            protein_spearman
                        )
                    ),
                    "mean_spearman": (
                        mean_spearman
                    ),
                    "mean_top10_jaccard": (
                        mean_jaccard
                    ),
                }
            )


    return (
        pd.DataFrame(target_rows),
        pd.DataFrame(protein_rows),
        attempted_targets,
        eligible_targets,
    )


def summarize_modality(
    modality,
    protein_df,
    attempted_targets,
    eligible_targets,
):

    proteins_with_eligible = len(
        protein_df
    )


    spearman_values = (
        protein_df[
            "mean_spearman"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    spearman_values = (
        spearman_values[
            np.isfinite(
                spearman_values
            )
        ]
    )


    jaccard_values = (
        protein_df[
            "mean_top10_jaccard"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    if not np.isfinite(
        jaccard_values
    ).all():

        raise RuntimeError(
            f"{modality}: non-finite "
            "protein-level Jaccard."
        )


    spearman_low, spearman_high = (
        percentile_ci(
            spearman_values
        )
    )

    jaccard_low, jaccard_high = (
        percentile_ci(
            jaccard_values
        )
    )


    return {
        "modality": modality,

        "frozen_proteins_attempted": (
            EXPECTED_PROTEINS
        ),

        "protein_label_targets_attempted": (
            attempted_targets
        ),

        "eligible_protein_label_targets": (
            eligible_targets
        ),

        "eligibility_fraction": (
            eligible_targets
            / attempted_targets
        ),

        "proteins_with_at_least_one_eligible_target": (
            proteins_with_eligible
        ),

        "proteins_with_finite_spearman": (
            len(spearman_values)
        ),

        "spearman_mean_protein_level": (
            float(
                np.mean(
                    spearman_values
                )
            )
        ),

        "spearman_median_protein_level": (
            float(
                np.median(
                    spearman_values
                )
            )
        ),

        "spearman_bootstrap_ci95_low": (
            spearman_low
        ),

        "spearman_bootstrap_ci95_high": (
            spearman_high
        ),

        "jaccard_mean_protein_level": (
            float(
                np.mean(
                    jaccard_values
                )
            )
        ),

        "jaccard_median_protein_level": (
            float(
                np.median(
                    jaccard_values
                )
            )
        ),

        "jaccard_bootstrap_ci95_low": (
            jaccard_low
        ),

        "jaccard_bootstrap_ci95_high": (
            jaccard_high
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "E10 — EXPLANATION STABILITY V1 "
        "— FROZEN AGGREGATION"
    )
    print("=" * 78)


    verify_manifest_hashes(
        SEQUENCE_MANIFEST,
        EXPECTED_SEQUENCE_MANIFEST_SHA,
    )

    print(
        "Sequence per-protein outputs: "
        "SHA VERIFIED"
    )


    verify_manifest_hashes(
        STRUCTURE_MANIFEST,
        EXPECTED_STRUCTURE_MANIFEST_SHA,
    )

    print(
        "Structure per-protein outputs: "
        "SHA VERIFIED"
    )


    population_ids = [
        x.strip()
        for x in POPULATION_PATH
        .read_text()
        .splitlines()
        if x.strip()
    ]


    if (
        len(population_ids)
        != EXPECTED_PROTEINS
    ):

        raise RuntimeError(
            "Unexpected frozen population size."
        )


    if (
        len(set(population_ids))
        != EXPECTED_PROTEINS
    ):

        raise RuntimeError(
            "Duplicate population IDs."
        )


    all_target_dfs = []
    all_protein_dfs = []
    summaries = []


    for modality, directory in [
        (
            "sequence",
            SEQUENCE_DIR,
        ),
        (
            "structure",
            STRUCTURE_DIR,
        ),
    ]:

        (
            target_df,
            protein_df,
            attempted_targets,
            eligible_targets,
        ) = load_modality(
            modality,
            directory,
            population_ids,
        )


        if (
            attempted_targets
            != EXPECTED_TARGETS
        ):

            raise RuntimeError(
                f"{modality}: expected "
                f"{EXPECTED_TARGETS} targets, "
                f"found {attempted_targets}"
            )


        summary = summarize_modality(
            modality,
            protein_df,
            attempted_targets,
            eligible_targets,
        )


        all_target_dfs.append(
            target_df
        )

        all_protein_dfs.append(
            protein_df
        )

        summaries.append(
            summary
        )


    target_out = pd.concat(
        all_target_dfs,
        ignore_index=True,
    )

    protein_out = pd.concat(
        all_protein_dfs,
        ignore_index=True,
    )

    summary_out = pd.DataFrame(
        summaries
    )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    target_path = (
        OUT_DIR
        / "stability_target_level.csv"
    )

    protein_path = (
        OUT_DIR
        / "stability_protein_level.csv"
    )

    summary_path = (
        OUT_DIR
        / "stability_summary.csv"
    )


    target_out.to_csv(
        target_path,
        index=False,
    )

    protein_out.to_csv(
        protein_path,
        index=False,
    )

    summary_out.to_csv(
        summary_path,
        index=False,
    )


    print()
    print("=" * 78)
    print("PRIMARY RESULTS")
    print("=" * 78)

    print(
        summary_out.to_string(
            index=False
        )
    )

    print()
    print(
        "Bootstrap resamples:",
        BOOTSTRAP_RESAMPLES,
    )

    print(
        "Bootstrap seed:",
        BOOTSTRAP_SEED,
    )

    print()
    print("Outputs:")
    print(target_path)
    print(protein_path)
    print(summary_path)


if __name__ == "__main__":
    main()
