import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


OUTPUT_FILE = Path(
    "data/processed/quickgo_mf_experimental_raw.csv"
)

BASE_URL = (
    "https://www.ebi.ac.uk/QuickGO/"
    "services/annotation/search"
)

EXPERIMENTAL_CODES = [
    "EXP",
    "IDA",
    "IPI",
    "IMP",
    "IGI",
    "IEP",
]

headers = {
    "Accept": "application/json"
}


def get_page(page):

    params = {
        "aspect": "molecular_function",
        "evidenceCode": ",".join(EXPERIMENTAL_CODES),
        "geneProductType": "protein",
        "page": page,
        "limit": 100,
    }

    response = requests.get(
        BASE_URL,
        headers=headers,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


print("Testing QuickGO connection...")

first = get_page(1)

page_info = first.get(
    "pageInfo",
    {}
)

total = page_info.get(
    "total",
    0
)

total_pages = page_info.get(
    "totalPages",
    1
)

print(f"Annotations found: {total:,}")
print(f"Pages: {total_pages:,}")

records = []


def extract_results(data):

    output = []

    for item in data.get("results", []):

        gene_product = item.get(
            "geneProductId",
            ""
        )

        # QuickGO normally returns things such as:
        # UniProtKB:P12345

        if ":" in gene_product:
            database, accession = (
                gene_product.split(":", 1)
            )
        else:
            database = ""
            accession = gene_product

        output.append({
            "database": database,
            "uniprot_id": accession,
            "go_id": item.get("goId"),
            "evidence_code": item.get(
                "goEvidence"
            ),
            "qualifier": item.get(
                "qualifier"
            ),
            "reference": item.get(
                "reference"
            ),
            "assigned_by": item.get(
                "assignedBy"
            ),
            "date": item.get(
                "date"
            ),
        })

    return output


records.extend(
    extract_results(first)
)


for page in tqdm(
    range(2, total_pages + 1)
):

    try:

        data = get_page(page)

        records.extend(
            extract_results(data)
        )

        time.sleep(0.1)

    except Exception as e:

        print(
            f"\nFailed page {page}: {e}"
        )

        time.sleep(2)


df = pd.DataFrame(records)

print("\n==============================")
print("RAW QUICKGO RESULT")
print("==============================")

print(
    f"Annotations downloaded: "
    f"{len(df):,}"
)

print(
    f"Unique proteins: "
    f"{df['uniprot_id'].nunique():,}"
)

print("\nDatabase distribution:")

print(
    df["database"]
    .value_counts()
    .head(10)
)

print("\nEvidence distribution:")

print(
    df["evidence_code"]
    .value_counts()
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)