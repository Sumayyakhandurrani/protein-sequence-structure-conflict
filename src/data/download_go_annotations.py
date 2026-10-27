import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/swissprot_candidates.csv")
OUTPUT_FILE = Path("data/processed/go_annotations_raw.csv")

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

# Small batch -> reliable and avoids extremely long URLs
BATCH_SIZE = 50


# ---------------------------------------------------------
# HTTP SESSION WITH AUTOMATIC RETRIES
# ---------------------------------------------------------

session = requests.Session()

retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("https://", adapter)


# ---------------------------------------------------------
# LOAD OUR 20,000 PROTEINS
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

accessions = df["uniprot_id"].drop_duplicates().tolist()

print(f"Candidate proteins: {len(accessions):,}")


# ---------------------------------------------------------
# FUNCTION TO DOWNLOAD ONE BATCH
# ---------------------------------------------------------

def download_batch(ids):

    query = " OR ".join(
        f"accession:{accession}"
        for accession in ids
    )

    params = {
        "query": query,
        "format": "txt",
        "size": 500,
    }

    response = session.get(
        BASE_URL,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    return response.text


# ---------------------------------------------------------
# PARSE GO LINES FROM UNIPROT FLAT FILE
# ---------------------------------------------------------

def parse_uniprot_text(text):

    records = []

    current_accession = None

    for line in text.splitlines():

        # Primary accession occurs in AC line
        #
        # Example:
        # AC   P12345; Q12345;
        #
        if line.startswith("AC   "):

            accessions_part = line[5:].strip()

            accession_list = [
                x.strip()
                for x in accessions_part.split(";")
                if x.strip()
            ]

            if current_accession is None and accession_list:
                current_accession = accession_list[0]

        # GO database cross-reference
        #
        # Example conceptually:
        #
        # DR   GO; GO:000xxxx; F:some function; IDA:...
        #
        elif line.startswith("DR   GO;"):

            if current_accession is None:
                continue

            parts = [
                part.strip()
                for part in line[5:].split(";")
            ]

            if len(parts) < 4:
                continue

            # parts[0] = GO
            # parts[1] = GO:xxxxxxx
            # parts[2] = F:function name
            # parts[3] = IDA:source

            go_id = parts[1]
            aspect_and_name = parts[2]
            evidence_field = parts[3]

            if ":" in aspect_and_name:

                aspect, go_name = aspect_and_name.split(
                    ":",
                    1
                )

            else:
                aspect = ""
                go_name = aspect_and_name

            evidence_code = evidence_field.split(":")[0]

            records.append({
                "uniprot_id": current_accession,
                "go_id": go_id,
                "aspect": aspect,
                "go_name": go_name,
                "evidence_code": evidence_code,
                "evidence_raw": evidence_field,
            })

        # End of one UniProt record
        elif line.startswith("//"):

            current_accession = None

    return records


# ---------------------------------------------------------
# DOWNLOAD ALL BATCHES
# ---------------------------------------------------------

all_records = []

batches = [
    accessions[i:i + BATCH_SIZE]
    for i in range(0, len(accessions), BATCH_SIZE)
]

print(f"Number of batches: {len(batches)}")
print("\nDownloading GO annotations...\n")


for batch in tqdm(batches):

    try:

        text = download_batch(batch)

        records = parse_uniprot_text(text)

        all_records.extend(records)

    except Exception as e:

        print("\nProblem with batch:")
        print(batch[:5])
        print(e)

    # Be polite to the UniProt server
    time.sleep(0.15)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

go_df = pd.DataFrame(all_records)

if len(go_df) == 0:

    raise RuntimeError(
        "No GO annotations were downloaded."
    )


go_df = go_df.drop_duplicates()

go_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==============================")
print("DOWNLOAD COMPLETE")
print("==============================")

print(f"Total GO annotations: {len(go_df):,}")

print(
    f"Proteins with any GO annotation: "
    f"{go_df['uniprot_id'].nunique():,}"
)

print("\nGO aspects:")

print(
    go_df["aspect"]
    .value_counts()
)

print("\nEvidence codes:")

print(
    go_df["evidence_code"]
    .value_counts()
    .head(20)
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)