from pathlib import Path
import ast
import pandas as pd


BASE = Path("data/splits/mmseqs30")

TRAIN_FILE = BASE / "train.pkl"
VAL_FILE = BASE / "val.pkl"
TEST_FILE = BASE / "test.pkl"

FREQ_FILE = BASE / "train_go_term_frequencies.csv"

OUTPUT = BASE / "label_threshold_evaluation.csv"


THRESHOLDS = [
    20,
    30,
    50,
    75,
    100,
]


print("Loading datasets...")

train = pd.read_pickle(TRAIN_FILE)
val = pd.read_pickle(VAL_FILE)
test = pd.read_pickle(TEST_FILE)

freq = pd.read_csv(FREQ_FILE)


print(f"Train proteins: {len(train):,}")
print(f"Val proteins:   {len(val):,}")
print(f"Test proteins:  {len(test):,}")


# ---------------------------------------------------------
# Make sure propagated_go_terms are Python lists
# ---------------------------------------------------------

def normalize_terms(x):

    if isinstance(x, list):
        return x

    if isinstance(x, tuple):
        return list(x)

    if isinstance(x, set):
        return list(x)

    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)

            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)

        except Exception:
            pass

    return []


for df in [train, val, test]:

    df["propagated_go_terms"] = (
        df["propagated_go_terms"]
        .apply(normalize_terms)
    )


# ---------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------

def evaluate_split(df, vocab):

    vocab = set(vocab)

    selected = df[
        "propagated_go_terms"
    ].apply(
        lambda terms: set(terms) & vocab
    )

    n_with_label = (
        selected.apply(len) > 0
    ).sum()

    total_labels = selected.apply(len).sum()

    return {
        "proteins": len(df),

        "proteins_with_selected_label":
            n_with_label,

        "coverage_percent":
            100 * n_with_label / len(df),

        "mean_selected_labels_per_protein":
            selected.apply(len).mean(),

        "median_selected_labels_per_protein":
            selected.apply(len).median(),

        "total_positive_assignments":
            total_labels,
    }


# ---------------------------------------------------------
# Held-out label support
# ---------------------------------------------------------

def label_support(df, vocab):

    vocab = set(vocab)

    counts = {
        go_id: 0
        for go_id in vocab
    }

    for terms in df["propagated_go_terms"]:

        for go_id in set(terms) & vocab:
            counts[go_id] += 1

    return pd.Series(counts)


# ---------------------------------------------------------
# Evaluate thresholds
# ---------------------------------------------------------

rows = []


print("\n" + "=" * 70)
print("LABEL THRESHOLD EVALUATION")
print("=" * 70)


for threshold in THRESHOLDS:

    vocab = freq.loc[
        freq["train_protein_count"] >= threshold,
        "go_id"
    ].tolist()

    n_labels = len(vocab)

    train_result = evaluate_split(
        train,
        vocab
    )

    val_result = evaluate_split(
        val,
        vocab
    )

    test_result = evaluate_split(
        test,
        vocab
    )

    val_support = label_support(
        val,
        vocab
    )

    test_support = label_support(
        test,
        vocab
    )


    print(
        f"\n{'=' * 70}"
    )

    print(
        f"TRAIN SUPPORT >= {threshold}"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"Selected labels: {n_labels:,}"
    )


    print("\nProtein coverage:")

    print(
        f"Train: "
        f"{train_result['proteins_with_selected_label']:,} / "
        f"{train_result['proteins']:,} "
        f"({train_result['coverage_percent']:.2f}%)"
    )

    print(
        f"Val:   "
        f"{val_result['proteins_with_selected_label']:,} / "
        f"{val_result['proteins']:,} "
        f"({val_result['coverage_percent']:.2f}%)"
    )

    print(
        f"Test:  "
        f"{test_result['proteins_with_selected_label']:,} / "
        f"{test_result['proteins']:,} "
        f"({test_result['coverage_percent']:.2f}%)"
    )


    print("\nMean selected labels per protein:")

    print(
        f"Train: "
        f"{train_result['mean_selected_labels_per_protein']:.2f}"
    )

    print(
        f"Val:   "
        f"{val_result['mean_selected_labels_per_protein']:.2f}"
    )

    print(
        f"Test:  "
        f"{test_result['mean_selected_labels_per_protein']:.2f}"
    )


    print("\nValidation label support:")

    print(
        f"Zero positives: "
        f"{(val_support == 0).sum():,}"
    )

    print(
        f"< 5 positives: "
        f"{(val_support < 5).sum():,}"
    )

    print(
        f"< 10 positives: "
        f"{(val_support < 10).sum():,}"
    )


    print("\nTest label support:")

    print(
        f"Zero positives: "
        f"{(test_support == 0).sum():,}"
    )

    print(
        f"< 5 positives: "
        f"{(test_support < 5).sum():,}"
    )

    print(
        f"< 10 positives: "
        f"{(test_support < 10).sum():,}"
    )


    rows.append({

        "train_support_threshold":
            threshold,

        "n_labels":
            n_labels,

        "train_coverage_percent":
            train_result["coverage_percent"],

        "val_coverage_percent":
            val_result["coverage_percent"],

        "test_coverage_percent":
            test_result["coverage_percent"],

        "train_mean_labels":
            train_result[
                "mean_selected_labels_per_protein"
            ],

        "val_mean_labels":
            val_result[
                "mean_selected_labels_per_protein"
            ],

        "test_mean_labels":
            test_result[
                "mean_selected_labels_per_protein"
            ],

        "val_zero_positive_labels":
            int((val_support == 0).sum()),

        "test_zero_positive_labels":
            int((test_support == 0).sum()),

        "val_labels_lt5":
            int((val_support < 5).sum()),

        "test_labels_lt5":
            int((test_support < 5).sum()),

        "val_labels_lt10":
            int((val_support < 10).sum()),

        "test_labels_lt10":
            int((test_support < 10).sum()),
    })


# ---------------------------------------------------------
# Save comparison
# ---------------------------------------------------------

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT,
    index=False
)


print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


print("\nSaved:")
print(OUTPUT)