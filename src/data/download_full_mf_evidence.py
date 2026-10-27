import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


INPUT_FILE = Path(
    "data/processed/swissprot_mf_candidates.csv"
)

OUTPUT_FILE = Path(
    "data/processed/full_mf_evidence_raw.csv"
)

FAILED_FILE = Path(
    "data/processed/full_mf_failed_batches.txt"
)

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

BATCH_SIZE = 100

EXPERIMENTAL_CODES = {
    "EXP",
    "IDA",
    "IPI",
    "IMP",
    "IGI",
    "IEP",
}


session = requests.Session()

retry = Retry(
    total=6,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retry)
)


def parse_records(text):

    records = []
    current_accession = None

    for line in text.splitlines():

        if line.startswith("AC   "):

            ids = [
                x.strip()
                for x in line[5:].split(";")
                if x.strip()
            ]

            if current_accession is None and ids:
                current_accession = ids[0]

        elif line.startswith("DR   GO;"):

            if current_accession is None:
                continue

            parts = [
                x.strip()
                for x in line[5:].split(";")
            ]

            if len(parts) < 4:
                continue

            go_id = parts[1]
            aspect_field = parts[2]
            evidence_field = parts[3]

            if ":" not in aspect_field:
                continue

            aspect, go_name = aspect_field.split(":", 1)

            evidence_code = evidence_field.split(":")[0]

            # Molecular Function only
            if aspect != "F":
                continue

            # Experimental evidence only
            if evidence_code not in EXPERIMENTAL_CODES:
                continue

            records.append({
                "uniprot_id": current_accession,
                "go_id": go_id,
                "go_name": go_name,
                "evidence_code": evidence_code,
                "evidence_raw": evidence_field,
            })

        elif line.startswith("//"):
            current_accession = None

    return records


def download_batch(ids):

    query = " OR ".join(
        f"accession:{x}"
        for x in ids
    )

    params = {
        "query": query,
        "format": "txt",
        "size": 500,
    }

    response = session.get(
        BASE_URL,
        params=params,
        timeout=180,
    )

    response.raise_for_status()

    return response.text


df = pd.read_csv(INPUT_FILE)

ids = (
    df["uniprot_id"]
    .drop_duplicates()
    .tolist()
)

print(f"MF candidate proteins: {len(ids):,}")

batches = [
    ids[i:i + BATCH_SIZE]
    for i in range(0, len(ids), BATCH_SIZE)
]

print(f"Total batches: {len(batches):,}")

all_records = []
failed_batches = []


for batch_index, batch in enumerate(
    tqdm(batches),
    start=1
):

    try:

        text = download_batch(batch)

        parsed = parse_records(text)

        all_records.extend(parsed)

    except Exception as e:

        print(
            f"\nFailed batch {batch_index}: {e}"
        )

        failed_batches.append(
            (batch_index, batch)
        )

    # Save checkpoint every 100 batches
    if batch_index % 100 == 0:

        checkpoint = pd.DataFrame(all_records)

        checkpoint.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"\nCheckpoint saved at batch "
            f"{batch_index:,}"
        )

    time.sleep(0.1)


result = pd.DataFrame(all_records)

result = result.drop_duplicates()

result.to_csv(
    OUTPUT_FILE,
    index=False
)


with open(
    FAILED_FILE,
    "w",
    encoding="utf-8"
) as f:

    for batch_number, batch in failed_batches:

        f.write(
            f"{batch_number}\t"
            + ",".join(batch)
            + "\n"
        )


print("\n==============================")
print("DOWNLOAD COMPLETE")
print("==============================")

print(
    f"Experimental MF annotations: "
    f"{len(result):,}"
)

print(
    f"Proteins with experimental MF: "
    f"{result['uniprot_id'].nunique():,}"
)

print(
    f"Unique GO-MF terms: "
    f"{result['go_id'].nunique():,}"
)

print("\nEvidence distribution:")

print(
    result["evidence_code"]
    .value_counts()
)

print(
    f"\nFailed batches: "
    f"{len(failed_batches):,}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)