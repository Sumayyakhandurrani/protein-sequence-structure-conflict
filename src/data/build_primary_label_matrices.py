from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pandas as pd


BASE = Path("data/splits/mmseqs30_graph_v2")
LABEL_DIR = BASE / "labels"

TRAIN_FILE = BASE / "train.pkl"
VAL_FILE = BASE / "val.pkl"
TEST_FILE = BASE / "test.pkl"

VOCAB_FILE = BASE / "primary_go_mf_labels.csv"

EXPECTED = {
    "train": 38031,
    "val": 4754,
    "test": 4754,
}

EXPECTED_LABELS = 494


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def normalize_go_terms(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(x) for x in value]

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed]

        except Exception:
            pass

        return [
            x.strip()
            for x in value.replace(";", ",").split(",")
            if x.strip()
        ]

    return [str(value)]


print("=" * 70)
print("BUILD PRIMARY LABEL MATRICES")
print("=" * 70)

LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\nLoading frozen primary vocabulary...")

vocab = pd.read_csv(VOCAB_FILE)

required_vocab = {
    "go_id",
    "train_protein_count",
    "label_index",
}

if not required_vocab.issubset(vocab.columns):
    raise ValueError(
        f"Missing vocabulary columns: "
        f"{required_vocab - set(vocab.columns)}"
    )

if len(vocab) != EXPECTED_LABELS:
    raise ValueError(
        f"Expected {EXPECTED_LABELS} labels, "
        f"found {len(vocab)}"
    )

vocab = vocab.sort_values(
    "label_index"
).reset_index(drop=True)

expected_indices = list(
    range(EXPECTED_LABELS)
)

if vocab["label_index"].tolist() != expected_indices:
    raise ValueError(
        "label_index is not contiguous 0..493"
    )

go_to_index = dict(
    zip(
        vocab["go_id"],
        vocab["label_index"],
    )
)

label_set = set(go_to_index)


def build_matrix(split_name, path):
    print(f"\nProcessing {split_name}...")

    df = pd.read_pickle(path)

    if len(df) != EXPECTED[split_name]:
        raise ValueError(
            f"{split_name}: expected "
            f"{EXPECTED[split_name]:,}, "
            f"found {len(df):,}"
        )

    required_cols = {
        "uniprot_id",
        "propagated_go_terms",
    }

    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{split_name}: missing columns "
            f"{required_cols - set(df.columns)}"
        )

    if df["uniprot_id"].duplicated().any():
        raise ValueError(
            f"{split_name}: duplicate protein IDs"
        )

    Y = np.zeros(
        (len(df), EXPECTED_LABELS),
        dtype=np.uint8,
    )

    for row_idx, terms in enumerate(
        df["propagated_go_terms"]
    ):
        terms = normalize_go_terms(terms)

        for go_id in terms:
            idx = go_to_index.get(go_id)

            if idx is not None:
                Y[row_idx, idx] = 1

    if not np.isin(Y, [0, 1]).all():
        raise ValueError(
            f"{split_name}: non-binary values found"
        )

    npy_path = LABEL_DIR / f"{split_name}_labels.npy"
    ids_path = LABEL_DIR / f"{split_name}_uniprot_ids.txt"

    np.save(
        npy_path,
        Y,
    )

    with open(
        ids_path,
        "w",
        encoding="utf-8",
    ) as f:
        for uid in df["uniprot_id"].astype(str):
            f.write(f"{uid}\n")

    print(
        f"{split_name}: {Y.shape}"
    )

    return df, Y, npy_path, ids_path


train_df, train_Y, train_npy, train_ids = build_matrix(
    "train",
    TRAIN_FILE,
)

val_df, val_Y, val_npy, val_ids = build_matrix(
    "val",
    VAL_FILE,
)

test_df, test_Y, test_npy, test_ids = build_matrix(
    "test",
    TEST_FILE,
)


# ============================================================
# TRAINING SUPPORT CONSISTENCY CHECK
# ============================================================

train_support = train_Y.sum(axis=0)

frozen_support = (
    vocab["train_protein_count"]
    .to_numpy(dtype=np.int64)
)

if not np.array_equal(
    train_support,
    frozen_support,
):
    mismatch = np.where(
        train_support != frozen_support
    )[0]

    raise ValueError(
        "Training support does not match frozen "
        f"vocabulary for {len(mismatch)} labels."
    )


# ============================================================
# ID ORDER CHECKS
# ============================================================

def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


if read_ids(train_ids) != train_df["uniprot_id"].astype(str).tolist():
    raise ValueError("Train ID order mismatch")

if read_ids(val_ids) != val_df["uniprot_id"].astype(str).tolist():
    raise ValueError("Validation ID order mismatch")

if read_ids(test_ids) != test_df["uniprot_id"].astype(str).tolist():
    raise ValueError("Test ID order mismatch")


# ============================================================
# LABEL MAP
# ============================================================

LABEL_MAP_FILE = LABEL_DIR / "primary_label_map.csv"

vocab[
    [
        "go_id",
        "train_protein_count",
        "label_index",
    ]
].to_csv(
    LABEL_MAP_FILE,
    index=False,
)


# ============================================================
# METADATA
# ============================================================

METADATA_FILE = LABEL_DIR / "label_metadata.json"

metadata = {
    "n_labels": EXPECTED_LABELS,
    "label_policy": {
        "type": "GO Molecular Function",
        "selection_basis": "training support only",
        "minimum_train_support": 75,
    },
    "matrix_shapes": {
        "train": list(train_Y.shape),
        "val": list(val_Y.shape),
        "test": list(test_Y.shape),
    },
    "dtype": str(train_Y.dtype),
    "checksums": {
        "vocabulary_csv": sha256_file(VOCAB_FILE),
        "train_labels_npy": sha256_file(train_npy),
        "val_labels_npy": sha256_file(val_npy),
        "test_labels_npy": sha256_file(test_npy),
        "train_ids": sha256_file(train_ids),
        "val_ids": sha256_file(val_ids),
        "test_ids": sha256_file(test_ids),
        "label_map": sha256_file(LABEL_MAP_FILE),
    },
    "test_policy": (
        "Test matrix is generated mechanically from the "
        "frozen benchmark, but must not be used for model "
        "selection, calibration, hyperparameter tuning, "
        "or threshold selection."
    ),
}

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print("\n" + "=" * 70)
print("LABEL MATRIX SUMMARY")
print("=" * 70)

print(
    f"Train:      {train_Y.shape}"
)
print(
    f"Validation: {val_Y.shape}"
)
print(
    f"Test:       {test_Y.shape}"
)

print("\nProtein ID ordering: PASS")
print("Vocabulary mapping: PASS")
print("Matrix binary check: PASS")
print(
    "Training support exactly matches "
    "frozen vocabulary: PASS"
)

print("\nSaved:")
print(train_npy)
print(val_npy)
print(test_npy)
print(train_ids)
print(val_ids)
print(test_ids)
print(LABEL_MAP_FILE)
print(METADATA_FILE)

print("\nDONE")
