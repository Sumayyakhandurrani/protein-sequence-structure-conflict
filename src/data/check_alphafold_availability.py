import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


INPUT_FILE = Path(
    "data/processed/full_mf_accessions.txt"
)

OUTPUT_FILE = Path(
    "data/processed/alphafold_availability.csv"
)

CHECKPOINT_FILE = Path(
    "data/processed/alphafold_availability_checkpoint.csv"
)

ERROR_FILE = Path(
    "data/processed/alphafold_errors.csv"
)

BASE_URL = "https://alphafold.ebi.ac.uk/api/prediction/"


# ---------------------------------------------------------
# HTTP session
# ---------------------------------------------------------

session = requests.Session()

retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1,
    status_forcelist=[
        429,
        500,
        502,
        503,
        504,
    ],
    allowed_methods=["GET"],
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retry)
)


# ---------------------------------------------------------
# Read protein IDs
# ---------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    accessions = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Proteins to check: {len(accessions):,}")


# ---------------------------------------------------------
# Resume only genuinely completed records
# ---------------------------------------------------------

records = []

if CHECKPOINT_FILE.exists():

    old = pd.read_csv(CHECKPOINT_FILE)

    # Old script could accidentally mark network failures
    # as False. We will keep existing rows for now, but
    # the safest resume point is the existing checkpoint.

    records = old.to_dict("records")

    processed_ids = set(
        old["uniprot_id"].astype(str)
    )

    print(
        f"Checkpoint contains: "
        f"{len(processed_ids):,}"
    )

else:

    processed_ids = set()


remaining = [
    accession
    for accession in accessions
    if accession not in processed_ids
]

print(f"Remaining: {len(remaining):,}")


errors = []


# ---------------------------------------------------------
# Check one accession
# ---------------------------------------------------------

def check_accession(accession):

    url = BASE_URL + accession

    try:

        response = session.get(
            url,
            timeout=(10, 20)
        )

        # Genuine AlphaFold absence
        if response.status_code == 404:

            return {
                "status": "not_available",
                "alphafold_available": False,
                "model_id": None,
                "cif_url": None,
                "pdb_url": None,
                "global_metric": None,
            }

        response.raise_for_status()

        data = response.json()

        if (
            isinstance(data, list)
            and len(data) > 0
        ):

            model = data[0]

            return {
                "status": "available",
                "alphafold_available": True,

                "model_id":
                    model.get("entryId")
                    or model.get("entry_id"),

                "cif_url":
                    model.get("cifUrl")
                    or model.get("cif_url"),

                "pdb_url":
                    model.get("pdbUrl")
                    or model.get("pdb_url"),

                "global_metric":
                    model.get("globalMetricValue")
                    or model.get("global_metric_value"),
            }

        # Successful response but empty result
        return {
            "status": "not_available",
            "alphafold_available": False,
            "model_id": None,
            "cif_url": None,
            "pdb_url": None,
            "global_metric": None,
        }


    except requests.exceptions.RequestException as e:

        # IMPORTANT:
        # network failure != structure unavailable

        return {
            "status": "network_error",
            "error": str(e)
        }


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

consecutive_errors = 0


for i, accession in enumerate(
    tqdm(remaining),
    start=1
):

    result = check_accession(accession)


    if result["status"] == "network_error":

        errors.append({
            "uniprot_id": accession,
            "error": result["error"]
        })

        consecutive_errors += 1

        print(
            f"\nNetwork error for {accession}"
        )


        # Stop automatically if network appears down
        if consecutive_errors >= 5:

            print(
                "\nFive consecutive network errors."
            )

            print(
                "Stopping safely instead of "
                "marking proteins unavailable."
            )

            break

        time.sleep(2)

        continue


    # Successful request
    consecutive_errors = 0


    records.append({
        "uniprot_id": accession,
        "alphafold_available":
            result["alphafold_available"],
        "status":
            result["status"],
        "model_id":
            result["model_id"],
        "cif_url":
            result["cif_url"],
        "pdb_url":
            result["pdb_url"],
        "global_metric":
            result["global_metric"],
    })


    # Checkpoint every 100 successful proteins
    if len(records) % 100 == 0:

        pd.DataFrame(records).to_csv(
            CHECKPOINT_FILE,
            index=False
        )


# ---------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------

result_df = pd.DataFrame(records)

result_df = result_df.drop_duplicates(
    subset=["uniprot_id"],
    keep="last"
)

result_df.to_csv(
    CHECKPOINT_FILE,
    index=False
)


if errors:

    pd.DataFrame(errors).to_csv(
        ERROR_FILE,
        index=False
    )


# ---------------------------------------------------------
# Final output only when complete
# ---------------------------------------------------------

completed_ids = set(
    result_df["uniprot_id"]
)

remaining_count = (
    len(accessions)
    - len(completed_ids)
)


print("\n==============================")
print("ALPHAFOLD CHECK STATUS")
print("==============================")

print(
    f"Successfully checked: "
    f"{len(result_df):,}"
)

print(
    f"Still remaining: "
    f"{remaining_count:,}"
)


if "alphafold_available" in result_df.columns:

    available = (
        result_df["alphafold_available"]
        .fillna(False)
        .sum()
    )

    print(
        f"Available among checked: "
        f"{int(available):,}"
    )


print(
    f"Network errors this run: "
    f"{len(errors):,}"
)


if remaining_count == 0:

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nFULL CHECK COMPLETE"
    )

    print(
        f"Saved final file to:\n"
        f"{OUTPUT_FILE}"
    )

else:

    print(
        "\nCheck is incomplete."
    )

    print(
        "Fix internet/DNS and run "
        "the same script again."
    )

    print(
        "It will resume from the checkpoint."
    )