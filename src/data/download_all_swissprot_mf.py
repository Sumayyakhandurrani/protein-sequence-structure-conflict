import re
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OUTPUT_FILE = Path(
    "data/processed/all_swissprot_mf_raw.tsv"
)

BASE_URL = (
    "https://rest.uniprot.org/"
    "uniprotkb/search"
)


# ---------------------------------------------------------
# RETRY SESSION
# ---------------------------------------------------------

session = requests.Session()

retries = Retry(
    total=5,
    backoff_factor=0.5,
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
    HTTPAdapter(
        max_retries=retries
    )
)


# ---------------------------------------------------------
# NEXT PAGE LINK
# ---------------------------------------------------------

NEXT_LINK_RE = re.compile(
    r'<(.+)>; rel="next"'
)


def get_next_link(headers):

    if "Link" not in headers:
        return None

    match = NEXT_LINK_RE.match(
        headers["Link"]
    )

    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------
# INITIAL QUERY
# ---------------------------------------------------------

params = {
    "query": (
        "reviewed:true AND go:*"
    ),
    "format": "tsv",

    # accession + Molecular Function
    "fields": (
        "accession,go_f"
    ),

    "size": 500,
}


print(
    "Downloading reviewed Swiss-Prot "
    "proteins with GO annotations..."
)

response = session.get(
    BASE_URL,
    params=params,
    timeout=120,
)

response.raise_for_status()


# ---------------------------------------------------------
# SAVE FIRST PAGE
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

lines = response.text.splitlines()

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as out:

    # keep header once
    for line in lines:
        out.write(line + "\n")


try:
    total = int(
        response.headers.get(
            "x-total-results",
            "0"
        )
    )

except ValueError:
    total = 0


downloaded = max(
    0,
    len(lines) - 1
)

print(
    f"Downloaded: "
    f"{downloaded:,}"
    f" / "
    f"{total:,}"
)


# ---------------------------------------------------------
# PAGINATION
# ---------------------------------------------------------

next_url = get_next_link(
    response.headers
)


while next_url:

    response = session.get(
        next_url,
        timeout=120,
    )

    response.raise_for_status()

    lines = response.text.splitlines()

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as out:

        # Every page has a header.
        # Skip it after page 1.
        for line in lines[1:]:
            out.write(line + "\n")

    downloaded += max(
        0,
        len(lines) - 1
    )

    print(
        f"Downloaded: "
        f"{downloaded:,}"
        f" / "
        f"{total:,}"
    )

    next_url = get_next_link(
        response.headers
    )


print("\n==============================")
print("DOWNLOAD COMPLETE")
print("==============================")

print(
    f"Entries downloaded: "
    f"{downloaded:,}"
)

print(
    f"Saved to:\n"
    f"{OUTPUT_FILE}"
)