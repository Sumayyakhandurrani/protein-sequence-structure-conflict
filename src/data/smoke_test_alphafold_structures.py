from pathlib import Path
import hashlib
import json
import urllib.request

import pandas as pd


# ============================================================
# PATHS
# ============================================================

MANIFEST = Path(
    "data/structures/manifests/train_structure_manifest.csv"
)

OUTPUT_DIR = Path(
    "data/structures/smoke_test"
)

PDB_DIR = OUTPUT_DIR / "pdb"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def select_nearest(df, target_length, target_confidence):
    """
    Select protein nearest to requested length/confidence target.
    Both dimensions are normalized so one does not dominate.
    """

    x = df.copy()

    length_scale = max(
        float(x["length"].std()),
        1.0
    )

    conf_scale = max(
        float(x["global_metric"].std()),
        1.0
    )

    x["_distance"] = (
        (
            (x["length"] - target_length)
            / length_scale
        ) ** 2
        +
        (
            (x["global_metric"] - target_confidence)
            / conf_scale
        ) ** 2
    )

    return x.sort_values(
        ["_distance", "uniprot_id"]
    ).iloc[0]


# ============================================================
# LOAD
# ============================================================

print("=" * 78)
print("ALPHAFOLD STRUCTURE SMOKE TEST — TRAIN ONLY")
print("=" * 78)

df = pd.read_csv(MANIFEST)

assert len(df) == 38031
assert df["uniprot_id"].is_unique

required = [
    "uniprot_id",
    "length",
    "model_id",
    "pdb_url",
    "global_metric",
]

for col in required:
    assert col in df.columns


df = df.dropna(
    subset=[
        "length",
        "pdb_url",
        "global_metric",
    ]
).copy()


print("\nTRAIN manifest proteins:", len(df))
print("TEST USED: NO")


# ============================================================
# TARGETS
# ============================================================

targets = [
    ("short_low",       75,  55),
    ("short_medium",    75,  75),
    ("short_high",      75,  92),

    ("medium_low",     300,  55),
    ("medium_medium",  300,  75),
    ("medium_high",    300,  92),

    ("long_low",       800,  55),
    ("long_medium",    800,  75),
    ("long_high",      800,  92),

    ("xlong_low",     1500,  55),
    ("xlong_medium",  1500,  75),
    ("xlong_high",    1500,  92),
]


# ============================================================
# SELECT UNIQUE PROTEINS
# ============================================================

selected = []
used = set()

for category, target_len, target_conf in targets:

    candidates = df[
        ~df["uniprot_id"].isin(used)
    ]

    row = select_nearest(
        candidates,
        target_len,
        target_conf
    )

    used.add(row["uniprot_id"])

    selected.append(
        {
            "category": category,
            "target_length": target_len,
            "target_global_metric": target_conf,
            "uniprot_id": row["uniprot_id"],
            "length": int(row["length"]),
            "global_metric": float(
                row["global_metric"]
            ),
            "model_id": row["model_id"],
            "pdb_url": row["pdb_url"],
        }
    )


selected_df = pd.DataFrame(selected)


print("\nSelected smoke-test proteins:")
print(
    selected_df[
        [
            "category",
            "uniprot_id",
            "length",
            "global_metric",
        ]
    ].to_string(index=False)
)


# ============================================================
# DOWNLOAD
# ============================================================

print("\n" + "=" * 78)
print("DOWNLOAD")
print("=" * 78)

download_records = []

for _, row in selected_df.iterrows():

    uid = row["uniprot_id"]

    destination = (
        PDB_DIR
        / f"{uid}.pdb"
    )

    print(
        f"\n{uid}"
        f" | length={row['length']}"
        f" | global={row['global_metric']:.2f}"
    )

    if destination.exists():

        size = destination.stat().st_size

        if size > 0:
            print(
                f"  Already exists: {size:,} bytes"
            )

        else:
            destination.unlink()
            size = 0

    else:
        size = 0

    if size == 0:

        try:

            urllib.request.urlretrieve(
                row["pdb_url"],
                destination
            )

        except Exception as e:

            print(
                "  DOWNLOAD FAILED:",
                repr(e)
            )

            download_records.append(
                {
                    "uniprot_id": uid,
                    "status": "failed",
                    "error": repr(e),
                }
            )

            continue

        size = destination.stat().st_size

        print(
            f"  Downloaded: {size:,} bytes"
        )

    if size < 100:

        status = "suspicious_file_size"

        print(
            "  WARNING: suspiciously small file"
        )

    else:
        status = "ok"

    download_records.append(
        {
            "uniprot_id": uid,
            "status": status,
            "file": str(destination),
            "bytes": int(size),
            "sha256": (
                sha256(destination)
                if size > 0
                else None
            ),
        }
    )


# ============================================================
# SAVE
# ============================================================

selected_path = (
    OUTPUT_DIR
    / "smoke_test_selection.csv"
)

selected_df.to_csv(
    selected_path,
    index=False
)


download_df = pd.DataFrame(
    download_records
)

download_path = (
    OUTPUT_DIR
    / "smoke_test_download_status.csv"
)

download_df.to_csv(
    download_path,
    index=False
)


n_ok = int(
    (
        download_df["status"] == "ok"
    ).sum()
)

n_failed = int(
    (
        download_df["status"] == "failed"
    ).sum()
)


summary = {
    "protocol": "structure_v1",
    "source_split": "train",
    "selected_proteins": len(selected_df),
    "successful_downloads": n_ok,
    "failed_downloads": n_failed,
    "test_used": False,
}


summary_path = (
    OUTPUT_DIR
    / "smoke_test_summary.json"
)

with open(summary_path, "w") as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 78)
print("SMOKE TEST DOWNLOAD SUMMARY")
print("=" * 78)

print(
    "Selected:",
    len(selected_df)
)

print(
    "Successful:",
    n_ok
)

print(
    "Failed:",
    n_failed
)

print("TEST USED: NO")

print("\nSaved:")
print(selected_path)
print(download_path)
print(summary_path)

print("=" * 78)