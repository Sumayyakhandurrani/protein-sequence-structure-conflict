from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
import time

IDS_FILE = Path(
    "data/final_evaluation/final_test_primary_v1/"
    "paired_test_uniprot_ids.txt"
)

OUT_DIR = Path(
    "data/xai/functional_sites_v1/uniprot_batches"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

ids = [
    x.strip()
    for x in IDS_FILE.read_text().splitlines()
    if x.strip()
]

assert len(ids) == 4731
assert len(set(ids)) == 4731

BATCH_SIZE = 100
batches = [
    ids[i:i+BATCH_SIZE]
    for i in range(0, len(ids), BATCH_SIZE)
]

print("=" * 75)
print("E10 TARGETED UNIPROT FEATURE DOWNLOAD")
print("=" * 75)
print("Proteins:", len(ids))
print("Batch size:", BATCH_SIZE)
print("Batches:", len(batches))
print()

for i, batch in enumerate(batches, start=1):

    out = OUT_DIR / f"batch_{i:03d}.dat"

    # Resume safely if a previously completed batch exists.
    if out.exists():
        text = out.read_text(
            encoding="utf-8",
            errors="replace"
        )

        n_id = sum(
            x.startswith("ID   ")
            for x in text.splitlines()
        )

        n_end = sum(
            x == "//"
            for x in text.splitlines()
        )

        if n_id == len(batch) and n_end == len(batch):
            print(
                f"[{i:02d}/{len(batches)}] "
                f"already complete ({n_end} records)"
            )
            continue

    query = " OR ".join(
        f"accession:{uid}"
        for uid in batch
    )

    url = (
        "https://rest.uniprot.org/uniprotkb/stream"
        "?format=txt&query="
        + quote(f"({query})")
    )

    print(
        f"[{i:02d}/{len(batches)}] "
        f"requesting {len(batch)} proteins..."
    )

    success = False

    for attempt in range(1, 4):

        try:
            req = Request(
                url,
                headers={
                    "User-Agent":
                    "protein-conflict-research/1.0"
                }
            )

            with urlopen(req, timeout=120) as r:
                data = r.read()

            text = data.decode(
                "utf-8",
                errors="replace"
            )

            n_id = sum(
                x.startswith("ID   ")
                for x in text.splitlines()
            )

            n_end = sum(
                x == "//"
                for x in text.splitlines()
            )

            if n_id != n_end:
                raise RuntimeError(
                    f"incomplete response: "
                    f"ID={n_id}, //={n_end}"
                )

            if n_id != len(batch):
                raise RuntimeError(
                    f"requested {len(batch)}, "
                    f"received {n_id}"
                )

            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(out)

            print(
                f"          saved {n_end} complete records"
            )

            success = True
            break

        except Exception as e:
            print(
                f"          attempt {attempt}/3 failed: {e}"
            )

            if attempt < 3:
                time.sleep(5 * attempt)

    if not success:
        raise RuntimeError(
            f"Batch {i} failed after 3 attempts."
        )

    time.sleep(1)

print()
print("=" * 75)
print("DOWNLOAD COMPLETE")
print("=" * 75)
