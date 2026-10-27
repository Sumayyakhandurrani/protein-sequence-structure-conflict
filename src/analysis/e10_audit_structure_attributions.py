from pathlib import Path
import numpy as np

POPULATION_PATH = Path(
    "data/xai/functional_sites_v1/"
    "xai_attribution_eligible_ids.txt"
)

ATTR_DIR = Path(
    "data/xai/functional_sites_v1/"
    "structure_attributions_v1/per_protein"
)

TEST_IDS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "labels/test_uniprot_ids.txt"
)

TEST_LABELS_PATH = Path(
    "data/splits/mmseqs30_graph_v2/"
    "labels/test_labels.npy"
)

STRUCTURE_IDS_PATH = Path(
    "data/structures/structure_protocol_v1/"
    "test_structure_eligible_ids.txt"
)

FROZEN_LOGITS_PATH = Path(
    "data/final_evaluation/"
    "structure_expert_v1_test/"
    "test_raw_logits.npy"
)

GRAPH_DIR = Path(
    "data/structures/structure_protocol_v1/"
    "graphs/test"
)

RTOL = 1e-5
ATOL = 1e-5


def read_ids(path):
    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


population = read_ids(POPULATION_PATH)
test_ids = read_ids(TEST_IDS_PATH)
structure_ids = read_ids(STRUCTURE_IDS_PATH)

Y = np.load(
    TEST_LABELS_PATH,
    mmap_mode="r",
)

frozen_logits = np.load(
    FROZEN_LOGITS_PATH,
    mmap_mode="r",
)

test_row = {
    uid: i
    for i, uid in enumerate(test_ids)
}

structure_row = {
    uid: i
    for i, uid in enumerate(structure_ids)
}


files = sorted(
    ATTR_DIR.glob("*.npz")
)

file_ids = {
    p.stem
    for p in files
}

population_set = set(population)

missing = sorted(
    population_set - file_ids
)

extra = sorted(
    file_ids - population_set
)


print("=" * 78)
print("E10 STRUCTURE ATTRIBUTION — FULL INTEGRITY AUDIT")
print("=" * 78)

print("Frozen population             :", len(population))
print("Attribution files             :", len(files))
print("Missing population files      :", len(missing))
print("Extra attribution files       :", len(extra))


if missing:
    print("First missing IDs:", missing[:10])

if extra:
    print("First extra IDs:", extra[:10])


errors = []

total_targets = 0
total_values = 0

negative_values = 0
positive_values = 0
zero_values = 0

global_min = np.inf
global_max = -np.inf

global_max_clean_diff = 0.0
worst_clean_uid = None
worst_clean_label = None
worst_clean_ratio = 0.0

clean_failing_elements = 0
clean_failing_proteins = 0

length_sum = 0


for number, uid in enumerate(
    population,
    start=1,
):

    path = ATTR_DIR / f"{uid}.npz"

    if not path.exists():
        errors.append(
            f"{uid}: missing attribution file"
        )
        continue


    if uid not in test_row:
        errors.append(
            f"{uid}: missing TEST row"
        )
        continue

    if uid not in structure_row:
        errors.append(
            f"{uid}: missing structure row"
        )
        continue


    tr = test_row[uid]
    sr = structure_row[uid]

    expected_labels = (
        np.flatnonzero(
            Y[tr] == 1
        )
        .astype(np.int64)
    )


    graph_path = (
        GRAPH_DIR / f"{uid}.npz"
    )

    if not graph_path.exists():
        errors.append(
            f"{uid}: missing graph"
        )
        continue


    with np.load(
        graph_path,
        allow_pickle=False,
    ) as g:
        graph_length = int(
            g["x"].shape[0]
        )


    with np.load(
        path,
        allow_pickle=False,
    ) as z:

        stored_uid = str(
            z["uniprot_id"].item()
        )

        stored_test_row = int(
            z["test_row"].item()
        )

        stored_structure_row = int(
            z["structure_row"].item()
        )

        stored_length = int(
            z["length"].item()
        )

        labels = (
            z["positive_label_indices"]
            .astype(np.int64)
        )

        attr = (
            z["attribution_logit_drop"]
            .astype(np.float32)
        )

        clean_targets = (
            z["clean_target_logits"]
            .astype(np.float32)
        )


    if stored_uid != uid:
        errors.append(
            f"{uid}: stored UID={stored_uid}"
        )


    if stored_test_row != tr:
        errors.append(
            f"{uid}: TEST row mismatch"
        )


    if stored_structure_row != sr:
        errors.append(
            f"{uid}: structure row mismatch"
        )


    if stored_length != graph_length:
        errors.append(
            f"{uid}: length mismatch "
            f"{stored_length} vs graph {graph_length}"
        )


    if not np.array_equal(
        labels,
        expected_labels,
    ):
        errors.append(
            f"{uid}: positive-label mismatch"
        )


    expected_shape = (
        len(expected_labels),
        graph_length,
    )

    if attr.shape != expected_shape:
        errors.append(
            f"{uid}: attribution shape "
            f"{attr.shape} expected {expected_shape}"
        )


    if clean_targets.shape != (
        len(expected_labels),
    ):
        errors.append(
            f"{uid}: clean-target shape mismatch"
        )


    if not np.isfinite(attr).all():
        errors.append(
            f"{uid}: non-finite attribution"
        )


    if not np.isfinite(clean_targets).all():
        errors.append(
            f"{uid}: non-finite clean target logits"
        )


    frozen_targets = np.asarray(
        frozen_logits[
            sr,
            expected_labels,
        ],
        dtype=np.float32,
    )


    diff = np.abs(
        clean_targets
        - frozen_targets
    )

    allowed = (
        ATOL
        + RTOL
        * np.abs(
            frozen_targets
        )
    )

    failed = (
        diff > allowed
    )

    n_failed = int(
        failed.sum()
    )

    clean_failing_elements += n_failed

    if n_failed > 0:
        clean_failing_proteins += 1

        errors.append(
            f"{uid}: {n_failed} clean target "
            "logits fail reproduction tolerance"
        )


    if diff.size:

        local_max = float(
            diff.max()
        )

        if local_max > global_max_clean_diff:
            global_max_clean_diff = local_max


        ratio = (
            diff / allowed
        )

        local_ratio_index = int(
            np.argmax(ratio)
        )

        local_ratio = float(
            ratio[
                local_ratio_index
            ]
        )

        if local_ratio > worst_clean_ratio:
            worst_clean_ratio = local_ratio
            worst_clean_uid = uid
            worst_clean_label = int(
                expected_labels[
                    local_ratio_index
                ]
            )


    total_targets += len(labels)
    total_values += int(
        attr.size
    )

    length_sum += graph_length


    negative_values += int(
        np.sum(attr < 0)
    )

    positive_values += int(
        np.sum(attr > 0)
    )

    zero_values += int(
        np.sum(attr == 0)
    )


    if attr.size:
        global_min = min(
            global_min,
            float(attr.min()),
        )

        global_max = max(
            global_max,
            float(attr.max()),
        )


print()
print("Protein-label targets         :", total_targets)
print("Attribution values            :", total_values)
print("Total graph residues          :", length_sum)

print()
print("Attribution minimum           :", global_min)
print("Attribution maximum           :", global_max)
print("Negative attribution values   :", negative_values)
print("Positive attribution values   :", positive_values)
print("Exactly-zero values           :", zero_values)

print()
print(
    "Max clean-target abs diff     :",
    global_max_clean_diff,
)

print(
    "Worst clean diff/tolerance    :",
    worst_clean_ratio,
)

print(
    "Worst clean protein           :",
    worst_clean_uid,
)

print(
    "Worst clean label index       :",
    worst_clean_label,
)

print(
    "Failing clean-logit elements  :",
    clean_failing_elements,
)

print(
    "Failing clean-logit proteins  :",
    clean_failing_proteins,
)

print()
print("Audit errors                  :", len(errors))

if errors:
    print()
    print("First audit errors:")
    for x in errors[:20]:
        print("  -", x)


expected_ok = (
    len(population) == 2447
    and len(files) == 2447
    and len(missing) == 0
    and len(extra) == 0
    and total_targets == 23087
    and total_values == 12758968
    and length_sum == 1279145
    and clean_failing_elements == 0
    and clean_failing_proteins == 0
    and len(errors) == 0
)


print()
print("=" * 78)

print(
    "STRUCTURE ATTRIBUTION AUDIT:",
    "PASS"
    if expected_ok
    else "FAIL",
)

print("=" * 78)
