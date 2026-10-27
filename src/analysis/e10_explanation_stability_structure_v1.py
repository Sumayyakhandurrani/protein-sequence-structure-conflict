from pathlib import Path
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr


# ============================================================
# IMPORT FROZEN STRUCTURE MODEL
# ============================================================

MODEL_CODE_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
)

if str(MODEL_CODE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODEL_CODE_DIR),
    )

from structure_gnn import StructureGNN


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

NUM_LABELS = 494
NODE_DIM = 21
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10

TEMPERATURE = 1.0

GLOBAL_SEED = 42
TOLERANCE = 0.01

EXPECTED_PARAMETER_COUNT = 298990

EXPECTED_MODEL_SHA = (
    "f1434b60a67e7dd695f289f7094d528b"
    "b91a28f249b28efd0d26af28d698991e"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PATHS
# ============================================================

BASE = Path("data")

POPULATION_PATH = (
    BASE
    / "xai/functional_sites_v1/"
      "explanation_stability_v1/"
      "stability_population_ids.txt"
)

MODEL_PATH = (
    BASE
    / "models/structure_gnn_v1_full/"
      "final_model.pt"
)

GRAPH_DIR = (
    BASE
    / "structures/structure_protocol_v1/"
      "graphs/test"
)

ATTR_DIR = (
    BASE
    / "xai/functional_sites_v1/"
      "structure_attributions_v1/"
      "per_protein"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sigmoid(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return (
        1.0
        / (1.0 + np.exp(-x))
    )


def candidate_order(
    uid,
    length,
):

    key = (
        f"explanation_stability_v1|"
        f"{GLOBAL_SEED}|{uid}|structure"
    ).encode("utf-8")

    seed = int.from_bytes(
        hashlib.sha256(
            key
        ).digest()[:8],
        byteorder="little",
        signed=False,
    )

    return np.random.default_rng(
        seed
    ).permutation(length)


def predict(
    model,
    x,
    edge_index,
    edge_attr,
):

    batch = torch.zeros(
        x.shape[0],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():

        logits = model(
            x,
            edge_index,
            edge_attr,
            batch,
            1,
        )

    return (
        logits[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def top_fraction_indices(
    values,
    original_positions,
):

    n = len(values)

    k = max(
        1,
        int(np.ceil(0.10 * n)),
    )

    order = np.lexsort(
        (
            original_positions,
            -values,
        )
    )

    return set(
        original_positions[
            order[:k]
        ].tolist()
    )


# ============================================================
# PRODUCTION DRIVER
# ============================================================

EXPECTED_POPULATION_SIZE = 200
EXPECTED_TARGET_COUNT = 1770

OUT_DIR = (
    BASE
    / "xai/functional_sites_v1/"
      "explanation_stability_v1/"
      "per_protein/structure"
)


def atomic_save_npz(path, **arrays):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp.npz"
    )

    np.savez_compressed(
        tmp,
        **arrays,
    )

    os.replace(
        tmp,
        path,
    )


def validate_existing_output(
    path,
    uid,
    expected_length,
    expected_labels,
):

    try:

        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            required = {
                "uniprot_id",
                "length",
                "positive_label_indices",
                "selected_base_position0",
                "clean_target_probability",
                "perturbed_target_probability",
                "absolute_probability_shift",
                "eligible",
                "spearman",
                "top10_jaccard",
                "clean_logit_max_abs_diff",
                "base_logit_abs_diff",
            }

            if not required.issubset(
                set(z.files)
            ):
                return False

            if str(
                z["uniprot_id"].item()
            ) != uid:
                return False

            if int(
                z["length"].item()
            ) != expected_length:
                return False

            labels = np.asarray(
                z["positive_label_indices"],
                dtype=np.int32,
            )

            if not np.array_equal(
                labels,
                expected_labels,
            ):
                return False

            n = len(expected_labels)

            for key in [
                "selected_base_position0",
                "clean_target_probability",
                "perturbed_target_probability",
                "absolute_probability_shift",
                "eligible",
                "spearman",
                "top10_jaccard",
                "base_logit_abs_diff",
            ]:

                if np.asarray(
                    z[key]
                ).shape != (n,):
                    return False

            selected = np.asarray(
                z["selected_base_position0"],
                dtype=np.int32,
            )

            eligible = np.asarray(
                z["eligible"],
                dtype=bool,
            )

            shifts = np.asarray(
                z["absolute_probability_shift"],
                dtype=np.float64,
            )

            jaccard = np.asarray(
                z["top10_jaccard"],
                dtype=np.float64,
            )

            base_diff = np.asarray(
                z["base_logit_abs_diff"],
                dtype=np.float64,
            )

            if not np.all(eligible):
                return False

            if np.any(selected < 0):
                return False

            if np.any(
                selected >= expected_length
            ):
                return False

            if not np.isfinite(
                shifts
            ).all():
                return False

            if np.any(
                shifts > TOLERANCE + 1e-12
            ):
                return False

            if not np.isfinite(
                jaccard
            ).all():
                return False

            if np.any(
                (jaccard < 0.0)
                | (jaccard > 1.0)
            ):
                return False

            if not np.isfinite(
                base_diff
            ).all():
                return False

            clean_diff = float(
                np.asarray(
                    z["clean_logit_max_abs_diff"]
                ).item()
            )

            if not np.isfinite(clean_diff):
                return False

            # Spearman may legitimately be NaN for
            # degenerate attribution vectors.

    except Exception:
        return False

    return True


def select_base_positions(
    uid,
    clean_target_logits,
    clean_attribution,
    length,
):

    order = candidate_order(
        uid,
        length,
    )

    clean_p = sigmoid(
        clean_target_logits
        / TEMPERATURE
    )

    selected = np.full(
        len(clean_target_logits),
        -1,
        dtype=np.int32,
    )

    selected_p = np.full(
        len(clean_target_logits),
        np.nan,
        dtype=np.float64,
    )

    selected_shift = np.full(
        len(clean_target_logits),
        np.nan,
        dtype=np.float64,
    )

    for pos in order:

        perturbed_logits = (
            clean_target_logits
            - clean_attribution[:, pos]
        )

        perturbed_p = sigmoid(
            perturbed_logits
            / TEMPERATURE
        )

        shift = np.abs(
            perturbed_p
            - clean_p
        )

        newly = (
            (selected < 0)
            & (shift <= TOLERANCE)
        )

        selected[newly] = int(pos)

        selected_p[newly] = (
            perturbed_p[newly]
        )

        selected_shift[newly] = (
            shift[newly]
        )

        if np.all(selected >= 0):
            break

    return (
        selected,
        clean_p,
        selected_p,
        selected_shift,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise RuntimeError(
            "--limit must be positive."
        )


    print("=" * 78)
    print(
        "E10 — EXPLANATION STABILITY V1 "
        "— STRUCTURE PRODUCTION"
    )
    print("=" * 78)

    print("Device:", DEVICE)
    print("Tolerance:", TOLERANCE)
    print("Temperature:", TEMPERATURE)
    print()


    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA is required."
        )


    if sha256_file(
        MODEL_PATH
    ) != EXPECTED_MODEL_SHA:

        raise RuntimeError(
            "Structure checkpoint SHA mismatch."
        )


    print(
        "Structure checkpoint SHA: VERIFIED"
    )


    ids = [
        x.strip()
        for x in POPULATION_PATH
        .read_text()
        .splitlines()
        if x.strip()
    ]


    if len(ids) != EXPECTED_POPULATION_SIZE:
        raise RuntimeError(
            "Unexpected stability population size."
        )


    if len(set(ids)) != len(ids):
        raise RuntimeError(
            "Duplicate stability population IDs."
        )


    metadata = {}
    total_targets = 0


    for uid in ids:

        graph_path = (
            GRAPH_DIR / f"{uid}.npz"
        )

        attr_path = (
            ATTR_DIR / f"{uid}.npz"
        )


        if not graph_path.exists():
            raise RuntimeError(
                f"{uid}: missing frozen graph."
            )


        if not attr_path.exists():
            raise RuntimeError(
                f"{uid}: missing frozen "
                "structure attribution."
            )


        with np.load(
            graph_path,
            allow_pickle=False,
        ) as g:

            graph_uid = str(
                g["uniprot_id"].item()
            )

            x_shape = g["x"].shape


        if graph_uid != uid:
            raise RuntimeError(
                f"{uid}: graph ID mismatch."
            )


        if (
            len(x_shape) != 2
            or x_shape[1] != NODE_DIM
        ):
            raise RuntimeError(
                f"{uid}: unexpected graph "
                "node-feature shape."
            )


        L = int(x_shape[0])


        with np.load(
            attr_path,
            allow_pickle=False,
        ) as z:

            attr_uid = str(
                z["uniprot_id"].item()
            )

            attr_length = int(
                z["length"].item()
            )

            labels = np.asarray(
                z["positive_label_indices"],
                dtype=np.int32,
            )

            attr_shape = (
                z["attribution_logit_drop"].shape
            )


        if attr_uid != uid:
            raise RuntimeError(
                f"{uid}: attribution ID mismatch."
            )


        if attr_length != L:
            raise RuntimeError(
                f"{uid}: graph/attribution "
                "length mismatch."
            )


        if attr_shape != (
            len(labels),
            L,
        ):
            raise RuntimeError(
                f"{uid}: attribution shape mismatch."
            )


        if len(labels) <= 0:
            raise RuntimeError(
                f"{uid}: zero target labels."
            )


        total_targets += len(labels)

        metadata[uid] = (
            L,
            labels,
        )


    if total_targets != EXPECTED_TARGET_COUNT:
        raise RuntimeError(
            "Unexpected stability target count: "
            f"{total_targets}"
        )


    print(
        "Population proteins:",
        len(ids),
    )

    print(
        "Population targets:",
        total_targets,
    )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    completed = []
    invalid_existing = []


    for uid in ids:

        L, labels = metadata[uid]

        out_path = (
            OUT_DIR / f"{uid}.npz"
        )

        if not out_path.exists():
            continue


        if validate_existing_output(
            out_path,
            uid,
            L,
            labels,
        ):
            completed.append(uid)

        else:
            invalid_existing.append(uid)


    if invalid_existing:

        raise RuntimeError(
            "Invalid existing structure stability "
            "outputs detected: "
            + ", ".join(
                invalid_existing[:10]
            )
        )


    completed_set = set(completed)

    todo = [
        uid
        for uid in ids
        if uid not in completed_set
    ]


    if args.limit is not None:
        todo = todo[:args.limit]


    print(
        "Validated completed outputs:",
        len(completed),
    )

    print(
        "Remaining frozen population:",
        len(ids) - len(completed),
    )

    print(
        "Scheduled this invocation:",
        len(todo),
    )


    if args.validate_only:

        print()
        print("=" * 78)
        print("VALIDATION ONLY: PASS")
        print("=" * 78)

        print(
            "No structure model loaded."
        )

        print(
            "No stability outcome generated."
        )

        return


    if not todo:

        print()
        print("Nothing to process.")
        return


    # --------------------------------------------------------
    # Load frozen model once
    # --------------------------------------------------------

    print()
    print(
        "Loading frozen structure GNN..."
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )


    if checkpoint["epoch"] != 21:
        raise RuntimeError(
            "Unexpected checkpoint epoch."
        )


    model = StructureGNN(
        node_dim=NODE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_labels=NUM_LABELS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()


    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )


    if (
        num_parameters
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Unexpected structure "
            "parameter count."
        )


    print(
        "Structure parameters:",
        num_parameters,
    )


    run_start = time.time()
    processed_this_run = 0


    # --------------------------------------------------------
    # Production loop
    # --------------------------------------------------------

    for number, uid in enumerate(
        todo,
        start=1,
    ):

        protein_start = time.time()

        graph_path = (
            GRAPH_DIR / f"{uid}.npz"
        )

        attr_path = (
            ATTR_DIR / f"{uid}.npz"
        )


        with np.load(
            graph_path,
            allow_pickle=False,
        ) as g:

            x_np = np.asarray(
                g["x"],
                dtype=np.float32,
            )

            edge_index_np = np.asarray(
                g["edge_index"],
                dtype=np.int64,
            )

            edge_attr_np = np.asarray(
                g["edge_attr"],
                dtype=np.float32,
            )

            graph_uid = str(
                g["uniprot_id"].item()
            )


        if graph_uid != uid:
            raise RuntimeError(
                f"{uid}: graph ID mismatch."
            )


        L = x_np.shape[0]


        if x_np.shape != (
            L,
            NODE_DIM,
        ):
            raise RuntimeError(
                f"{uid}: unexpected node "
                "feature shape."
            )


        if not np.allclose(
            x_np.sum(axis=1),
            1.0,
            rtol=0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{uid}: clean node features "
                "are not one-hot."
            )


        with np.load(
            attr_path,
            allow_pickle=False,
        ) as z:

            positive_labels = np.asarray(
                z["positive_label_indices"],
                dtype=np.int32,
            )

            clean_target_logits = np.asarray(
                z["clean_target_logits"],
                dtype=np.float64,
            )

            clean_attribution = np.asarray(
                z["attribution_logit_drop"],
                dtype=np.float64,
            )


        expected_L, expected_labels = (
            metadata[uid]
        )


        if L != expected_L:
            raise RuntimeError(
                f"{uid}: metadata length mismatch."
            )


        if not np.array_equal(
            positive_labels,
            expected_labels,
        ):
            raise RuntimeError(
                f"{uid}: target-label mismatch."
            )


        if clean_attribution.shape != (
            len(positive_labels),
            L,
        ):
            raise RuntimeError(
                f"{uid}: attribution shape mismatch."
            )


        # ----------------------------------------------------
        # Frozen candidate selection
        # ----------------------------------------------------

        (
            selected,
            clean_p,
            selected_p,
            selected_shift,
        ) = select_base_positions(
            uid,
            clean_target_logits,
            clean_attribution,
            L,
        )


        eligible = (
            selected >= 0
        )


        if not np.all(eligible):
            raise RuntimeError(
                f"{uid}: unexpected ineligible "
                "structure target."
            )


        # ----------------------------------------------------
        # Device graph
        # ----------------------------------------------------

        x_clean = torch.from_numpy(
            x_np
        ).float().to(
            DEVICE
        )

        edge_index = torch.from_numpy(
            edge_index_np
        ).long().to(
            DEVICE
        )

        edge_attr = torch.from_numpy(
            edge_attr_np
        ).float().to(
            DEVICE
        )


        # ----------------------------------------------------
        # Clean reproduction
        # ----------------------------------------------------

        clean_logits = predict(
            model,
            x_clean,
            edge_index,
            edge_attr,
        )


        clean_target_reproduced = (
            clean_logits[
                positive_labels
            ]
        )


        clean_diff = float(
            np.max(
                np.abs(
                    clean_target_reproduced
                    - clean_target_logits
                )
            )
        )


        if not np.allclose(
            clean_target_reproduced,
            clean_target_logits,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"{uid}: clean structure "
                "logits failed reproduction; "
                f"max_diff={clean_diff}"
            )


        # ----------------------------------------------------
        # Group targets by selected base
        # ----------------------------------------------------

        groups = {}

        for target_row, base_pos in enumerate(
            selected
        ):

            groups.setdefault(
                int(base_pos),
                [],
            ).append(
                target_row
            )


        spearman_values = np.full(
            len(positive_labels),
            np.nan,
            dtype=np.float64,
        )

        jaccard_values = np.full(
            len(positive_labels),
            np.nan,
            dtype=np.float64,
        )

        base_reproduction_diff = np.full(
            len(positive_labels),
            np.nan,
            dtype=np.float64,
        )


        # ----------------------------------------------------
        # Perturbed explanations
        # ----------------------------------------------------

        for base_pos, target_rows in sorted(
            groups.items()
        ):

            group_labels = (
                positive_labels[
                    target_rows
                ]
            )


            x_base = x_clean.clone()

            x_base[
                base_pos
            ] = 0.0


            base_logits_all = predict(
                model,
                x_base,
                edge_index,
                edge_attr,
            )


            base_target_logits = (
                base_logits_all[
                    group_labels
                ]
            )


            expected_base_logits = (
                clean_target_logits[
                    target_rows
                ]
                - clean_attribution[
                    target_rows,
                    base_pos
                ]
            )


            diffs = np.abs(
                base_target_logits
                - expected_base_logits
            )


            if not np.allclose(
                base_target_logits,
                expected_base_logits,
                rtol=1e-5,
                atol=1e-5,
            ):
                raise RuntimeError(
                    f"{uid}: base structure "
                    "perturbation failed reproduction; "
                    f"base={base_pos + 1}, "
                    f"max_diff={float(diffs.max())}"
                )


            base_reproduction_diff[
                target_rows
            ] = diffs


            K = len(group_labels)

            perturbed_attr = np.full(
                (
                    K,
                    L,
                ),
                np.nan,
                dtype=np.float32,
            )


            for second_pos in range(L):

                if second_pos == base_pos:
                    continue


                x_double = x_base.clone()

                x_double[
                    second_pos
                ] = 0.0


                double_logits = predict(
                    model,
                    x_double,
                    edge_index,
                    edge_attr,
                )


                perturbed_attr[
                    :,
                    second_pos
                ] = (
                    base_target_logits
                    - double_logits[
                        group_labels
                    ]
                )


            comparison_mask = np.ones(
                L,
                dtype=bool,
            )

            comparison_mask[
                base_pos
            ] = False


            original_positions = np.arange(
                L,
                dtype=np.int32,
            )[
                comparison_mask
            ]


            for local_row, target_row in enumerate(
                target_rows
            ):

                clean_vec = (
                    clean_attribution[
                        target_row,
                        comparison_mask
                    ]
                )

                perturbed_vec = (
                    perturbed_attr[
                        local_row,
                        comparison_mask
                    ]
                )


                if not np.isfinite(
                    perturbed_vec
                ).all():
                    raise RuntimeError(
                        f"{uid}: non-finite "
                        "perturbed explanation."
                    )


                rho = spearmanr(
                    clean_vec,
                    perturbed_vec,
                ).statistic


                clean_top = top_fraction_indices(
                    clean_vec,
                    original_positions,
                )

                perturbed_top = top_fraction_indices(
                    perturbed_vec,
                    original_positions,
                )


                intersection = (
                    clean_top
                    & perturbed_top
                )

                union = (
                    clean_top
                    | perturbed_top
                )


                jac = (
                    len(intersection)
                    / len(union)
                )


                spearman_values[
                    target_row
                ] = rho

                jaccard_values[
                    target_row
                ] = jac


        if not np.isfinite(
            jaccard_values
        ).all():
            raise RuntimeError(
                f"{uid}: non-finite Jaccard."
            )


        if not np.isfinite(
            base_reproduction_diff
        ).all():
            raise RuntimeError(
                f"{uid}: missing base "
                "reproduction QC."
            )


        out_path = (
            OUT_DIR / f"{uid}.npz"
        )


        atomic_save_npz(
            out_path,

            uniprot_id=np.asarray(uid),

            length=np.asarray(
                L,
                dtype=np.int32,
            ),

            positive_label_indices=(
                positive_labels.astype(
                    np.int32,
                    copy=False,
                )
            ),

            selected_base_position0=(
                selected.astype(
                    np.int32,
                    copy=False,
                )
            ),

            clean_target_probability=(
                clean_p.astype(
                    np.float32
                )
            ),

            perturbed_target_probability=(
                selected_p.astype(
                    np.float32
                )
            ),

            absolute_probability_shift=(
                selected_shift.astype(
                    np.float32
                )
            ),

            eligible=(
                eligible.astype(
                    np.bool_
                )
            ),

            spearman=(
                spearman_values.astype(
                    np.float32
                )
            ),

            top10_jaccard=(
                jaccard_values.astype(
                    np.float32
                )
            ),

            clean_logit_max_abs_diff=(
                np.asarray(
                    clean_diff,
                    dtype=np.float32,
                )
            ),

            base_logit_abs_diff=(
                base_reproduction_diff.astype(
                    np.float32
                )
            ),
        )


        if not validate_existing_output(
            out_path,
            uid,
            L,
            positive_labels,
        ):
            raise RuntimeError(
                f"{uid}: saved output failed "
                "post-save validation."
            )


        processed_this_run += 1

        elapsed = (
            time.time()
            - protein_start
        )


        print(
            f"[{number}/{len(todo)}] "
            f"{uid} "
            f"L={L} "
            f"K={len(positive_labels)} "
            f"bases={len(groups)} "
            f"finite_rho="
            f"{int(np.isfinite(spearman_values).sum())}/"
            f"{len(spearman_values)} "
            f"time={elapsed:.2f}s",
            flush=True,
        )


        del (
            x_clean,
            edge_index,
            edge_attr,
        )


    wall_time = (
        time.time()
        - run_start
    )


    print()
    print("=" * 78)
    print(
        "STRUCTURE STABILITY INVOCATION COMPLETE"
    )
    print("=" * 78)

    print(
        "Processed this invocation:",
        processed_this_run,
    )

    print(
        "Wall time seconds:",
        f"{wall_time:.2f}",
    )

    print(
        "Output directory:",
        OUT_DIR,
    )

    print(
        "Frozen model modified: NO"
    )

    print(
        "Frozen attribution files modified: NO"
    )


if __name__ == "__main__":
    main()
