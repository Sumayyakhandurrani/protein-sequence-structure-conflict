from pathlib import Path
import argparse
import hashlib
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import gzip

# ============================================================
# CONFIG
# ============================================================

MANIFEST_DIR = Path(
    "data/structures/manifests"
)

STRUCTURE_ROOT = Path(
    "data/structures/alphafold_v6"
)

MIN_VALID_BYTES = 1000

MAX_RETRIES = 5

TIMEOUT_SECONDS = 60

CHECKPOINT_EVERY = 100

MAX_WORKERS = 8

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


def looks_like_pdb(path):
    """
    Basic integrity check for an AlphaFold PDB.

    Do NOT assume ATOM records occur within the first
    N lines. Long AlphaFold PDB headers can push the
    first ATOM record beyond line 100.
    """

    if not path.exists():
        return False

    if path.stat().st_size < MIN_VALID_BYTES:
        return False

    try:

        with open(
            path,
            "r",
            errors="replace"
        ) as f:

            for line in f:

                if line.startswith("ATOM"):
                    return True

    except Exception:
        return False

    return False

def download_one(url, destination):

    temp_path = destination.with_suffix(
        ".pdb.part"
    )

    if temp_path.exists():
        temp_path.unlink()

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "protein-conflict-research/structure-v1"
        }
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            with urllib.request.urlopen(
                request,
                timeout=TIMEOUT_SECONDS
            ) as response:

                data = response.read()

            # ------------------------------------------------
            # Detect gzip by MAGIC BYTES, not filename/header.
            #
            # gzip files begin:
            #     1f 8b
            # ------------------------------------------------

            was_gzip = (
                len(data) >= 2
                and
                data[:2] == b"\x1f\x8b"
            )

            if was_gzip:

                try:
                    data = gzip.decompress(data)

                except Exception as e:
                    raise RuntimeError(
                        "Response was gzip-compressed "
                        "but decompression failed."
                    ) from e

            # ------------------------------------------------
            # Write normalized plain-text PDB
            # ------------------------------------------------

            with open(
                temp_path,
                "wb"
            ) as f:

                f.write(data)

            # ------------------------------------------------
            # Validate normalized PDB
            # ------------------------------------------------

            if not looks_like_pdb(
                temp_path
            ):

                raise RuntimeError(
                    "Downloaded response failed "
                    "basic PDB validation after "
                    "optional gzip decompression."
                )

            temp_path.replace(
                destination
            )

            return True, None

        except Exception as e:

            last_error = repr(e)

            if temp_path.exists():
                temp_path.unlink()

            if attempt < MAX_RETRIES:

                sleep_seconds = min(
                    2 ** attempt,
                    30
                )

                time.sleep(
                    sleep_seconds
                )

    return False, last_error

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    required=True,
    choices=[
        "train",
        "val",
        "test",
    ],
)

args = parser.parse_args()

split = args.split


# ============================================================
# PATHS
# ============================================================

manifest_path = (
    MANIFEST_DIR
    / f"{split}_structure_manifest.csv"
)

output_dir = (
    STRUCTURE_ROOT
    / split
)

pdb_dir = (
    output_dir
    / "pdb"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

pdb_dir.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint_path = (
    output_dir
    / "download_checkpoint.csv"
)

summary_path = (
    output_dir
    / "download_summary.json"
)


# ============================================================
# LOAD MANIFEST
# ============================================================

print("=" * 80)
print(
    f"ALPHAFOLD V6 DOWNLOAD — "
    f"{split.upper()}"
)
print("=" * 80)

manifest = pd.read_csv(
    manifest_path
)

required = [
    "uniprot_id",
    "model_id",
    "pdb_url",
    "length",
    "global_metric",
]

for col in required:

    if col not in manifest.columns:

        raise RuntimeError(
            f"Missing column: {col}"
        )


expected = (
    38031
    if split == "train"
    else 4754
)

assert len(manifest) == expected

assert manifest[
    "uniprot_id"
].is_unique


print(
    "\nManifest proteins:",
    len(manifest)
)

print(
    "Destination:",
    pdb_dir
)

print(
    "TEST SPLIT PROCESSING:",
    "YES" if split == "test" else "NO"
)


# ============================================================
# EXISTING CHECKPOINT
# ============================================================

records = {}

if checkpoint_path.exists():

    old = pd.read_csv(
        checkpoint_path
    )

    for _, row in old.iterrows():

        records[
            row["uniprot_id"]
        ] = row.to_dict()

    print(
        "Existing checkpoint rows:",
        len(records)
    )

else:

    print(
        "Existing checkpoint rows: 0"
    )


# ============================================================
# DOWNLOAD
# ============================================================

downloaded_now = 0
already_valid = 0
failed_now = 0

start_time = time.time()


def save_checkpoint():

    rows = []

    for uid in manifest[
        "uniprot_id"
    ]:

        if uid in records:
            rows.append(
                records[uid]
            )

    pd.DataFrame(
        rows
    ).to_csv(
        checkpoint_path,
        index=False
    )


# ============================================================
# CHECK EXISTING FILES AND BUILD DOWNLOAD QUEUE
# ============================================================

download_queue = []

for _, row in manifest.iterrows():

    uid = row["uniprot_id"]

    destination = (
        pdb_dir
        / f"{uid}.pdb"
    )

    # --------------------------------------------------------
    # Existing valid file
    # --------------------------------------------------------

    if looks_like_pdb(
        destination
    ):

        already_valid += 1

        records[uid] = {
            "uniprot_id":
                uid,

            "status":
                "ok",

            "bytes":
                destination.stat().st_size,

            "error":
                "",
        }

    else:

        # Remove corrupt/incomplete destination.
        if destination.exists():
            destination.unlink()

        download_queue.append(
            {
                "uniprot_id":
                    uid,

                "pdb_url":
                    row["pdb_url"],

                "destination":
                    destination,
            }
        )


print(
    "\nAlready valid:",
    already_valid
)

print(
    "Remaining to download:",
    len(download_queue)
)

print(
    "Parallel workers:",
    MAX_WORKERS
)


# Save the existing-file state before network work starts.
save_checkpoint()


# ============================================================
# PARALLEL DOWNLOAD WORKER
# ============================================================

def worker(item):

    uid = item[
        "uniprot_id"
    ]

    destination = item[
        "destination"
    ]

    ok, error = download_one(
        item["pdb_url"],
        destination
    )

    file_bytes = (
        destination.stat().st_size
        if ok
        and destination.exists()
        else 0
    )

    return {
        "uniprot_id":
            uid,

        "ok":
            ok,

        "bytes":
            file_bytes,

        "error":
            error,
    }


# ============================================================
# PARALLEL DOWNLOAD
# ============================================================

completed_downloads = 0

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(
            worker,
            item
        ):
        item["uniprot_id"]

        for item in download_queue
    }


    for future in as_completed(
        futures
    ):

        uid = futures[
            future
        ]

        completed_downloads += 1


        try:

            result = future.result()

            if result["ok"]:

                downloaded_now += 1

                records[uid] = {
                    "uniprot_id":
                        uid,

                    "status":
                        "ok",

                    "bytes":
                        result["bytes"],

                    "error":
                        "",
                }

            else:

                failed_now += 1

                records[uid] = {
                    "uniprot_id":
                        uid,

                    "status":
                        "failed",

                    "bytes":
                        0,

                    "error":
                        result["error"],
                }


        except Exception as e:

            failed_now += 1

            records[uid] = {
                "uniprot_id":
                    uid,

                "status":
                    "failed",

                "bytes":
                    0,

                "error":
                    repr(e),
            }


        # ----------------------------------------------------
        # Progress / checkpoint
        # ----------------------------------------------------

        if (
            completed_downloads
            % CHECKPOINT_EVERY
            == 0

            or

            completed_downloads
            == len(download_queue)
        ):

            save_checkpoint()

            elapsed = (
                time.time()
                - start_time
            )

            processed_total = (
                already_valid
                +
                completed_downloads
            )

            rate = (
                completed_downloads
                / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(download_queue)
                -
                completed_downloads
            )

            eta_seconds = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"[{processed_total:>6}/"
                f"{len(manifest)}]"
                f" downloaded_now="
                f"{downloaded_now}"
                f" existing="
                f"{already_valid}"
                f" failed="
                f"{failed_now}"
                f" download_rate="
                f"{rate:.2f}/s"
                f" ETA="
                f"{eta_seconds/60:.1f} min",
                flush=True
            )



# ============================================================
# FINAL CHECKPOINT
# ============================================================

save_checkpoint()

checkpoint = pd.read_csv(
    checkpoint_path
)

n_ok = int(
    (
        checkpoint["status"]
        == "ok"
    ).sum()
)

n_failed = int(
    (
        checkpoint["status"]
        == "failed"
    ).sum()
)


# ============================================================
# FINAL FILE AUDIT
# ============================================================

valid_files = 0
total_bytes = 0

for uid in manifest[
    "uniprot_id"
]:

    path = (
        pdb_dir
        / f"{uid}.pdb"
    )

    if looks_like_pdb(path):

        valid_files += 1
        total_bytes += (
            path.stat().st_size
        )


elapsed = (
    time.time()
    - start_time
)


summary = {

    "protocol":
        "structure_v1",

    "alphafold_version":
        "v6",

    "split":
        split,

    "manifest_proteins":
        int(
            len(manifest)
        ),

    "valid_pdb_files":
        int(
            valid_files
        ),

    "checkpoint_ok":
        n_ok,

    "checkpoint_failed":
        n_failed,

    "downloaded_this_run":
        downloaded_now,

    "already_valid_this_run":
        already_valid,

    "failed_this_run":
        failed_now,

    "total_bytes":
        int(total_bytes),

    "runtime_seconds":
        float(elapsed),

    "test_used":
        False,
}


with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print(
    f"{split.upper()} DOWNLOAD SUMMARY"
)
print("=" * 80)

print(
    "Manifest proteins:",
    len(manifest)
)

print(
    "Valid PDB files:",
    valid_files
)

print(
    "Checkpoint OK:",
    n_ok
)

print(
    "Checkpoint failed:",
    n_failed
)

print(
    "Downloaded this run:",
    downloaded_now
)

print(
    "Already valid:",
    already_valid
)

print(
    "Failed this run:",
    failed_now
)

print(
    "Total size:",
    f"{total_bytes / (1024**3):.2f} GiB"
)

print(
    "Runtime:",
    f"{elapsed / 60:.1f} minutes"
)

print("\nTEST USED: NO")

print("\nSaved:")
print(checkpoint_path)
print(summary_path)

print("=" * 80)


# ============================================================
# STRICT COMPLETION CHECK
# ============================================================

if valid_files != len(manifest):

    raise RuntimeError(
        f"{len(manifest) - valid_files} "
        "structures are still missing or invalid. "
        "Rerun the same command to retry/resume."
    )