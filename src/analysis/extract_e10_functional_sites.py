from pathlib import Path
import re
import pandas as pd

BATCH_DIR = Path(
    "data/xai/functional_sites_v1/uniprot_batches"
)

OUT_DIR = Path(
    "data/xai/functional_sites_v1"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "functional_sites_raw.csv"
SUMMARY = OUT_DIR / "functional_sites_extraction_summary.txt"

# ---------------------------------------------------------------------
# Parse UniProt feature-table locations.
#
# Accepted primary categories were frozen before analysis:
#   ACT_SITE
#   BINDING
#
# We preserve raw locations and evidence.
# Ambiguous locations such as <10, >20, ? etc. are retained in the
# raw table but marked coordinate_valid=False.
# ---------------------------------------------------------------------

def parse_location(location):
    location = location.strip()

    # Exact single residue
    m = re.fullmatch(r"(\d+)", location)
    if m:
        pos = int(m.group(1))
        return pos, pos, True

    # Exact range
    m = re.fullmatch(r"(\d+)\.\.(\d+)", location)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))

        if start <= end:
            return start, end, True

    return None, None, False


rows = []

batch_files = sorted(BATCH_DIR.glob("batch_*.dat"))

for batch_file in batch_files:

    current_accession = None
    current_feature = None

    with batch_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for raw_line in f:

            line = raw_line.rstrip("\n")

            # Primary accession = first accession on first AC line.
            if line.startswith("AC   ") and current_accession is None:

                accession_text = line[5:].strip()

                accessions = [
                    x.strip()
                    for x in accession_text.split(";")
                    if x.strip()
                ]

                if accessions:
                    current_accession = accessions[0]

            # New feature-table record
            if line.startswith("FT   "):

                feature_type = line[5:13].strip()
                remainder = line[13:].strip()

                if feature_type in {"ACT_SITE", "BINDING"}:

                    start, end, valid = parse_location(remainder)

                    current_feature = {
                        "uniprot_id": current_accession,
                        "feature_type": feature_type,
                        "location_raw": remainder,
                        "start": start,
                        "end": end,
                        "coordinate_valid": valid,
                        "description": "",
                        "evidence": "",
                        "ligand": "",
                        "ligand_id": "",
                        "source_batch": batch_file.name,
                    }

                    rows.append(current_feature)

                elif feature_type:
                    current_feature = None

                else:
                    # Continuation line for current feature
                    if current_feature is None:
                        continue

                    text = remainder

                    if text.startswith('/note="'):
                        value = text[len('/note="'):]
                        current_feature["description"] += value

                    elif text.startswith('/evidence="'):
                        value = text[len('/evidence="'):]
                        current_feature["evidence"] += value

                    elif text.startswith('/ligand="'):
                        value = text[len('/ligand="'):]
                        current_feature["ligand"] += value

                    elif text.startswith('/ligand_id="'):
                        value = text[len('/ligand_id="'):]
                        current_feature["ligand_id"] += value

                    else:
                        # Continuation of a quoted qualifier.
                        if (
                            current_feature["description"]
                            and not current_feature["description"].endswith('"')
                        ):
                            current_feature["description"] += " " + text

                        elif (
                            current_feature["evidence"]
                            and not current_feature["evidence"].endswith('"')
                        ):
                            current_feature["evidence"] += " " + text

                        elif (
                            current_feature["ligand"]
                            and not current_feature["ligand"].endswith('"')
                        ):
                            current_feature["ligand"] += " " + text

                        elif (
                            current_feature["ligand_id"]
                            and not current_feature["ligand_id"].endswith('"')
                        ):
                            current_feature["ligand_id"] += " " + text

            if line == "//":
                current_accession = None
                current_feature = None


df = pd.DataFrame(rows)

# Remove surrounding quotation marks where present.
for col in [
    "description",
    "evidence",
    "ligand",
    "ligand_id",
]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.strip('"')
    )

# Safety checks
assert len(batch_files) == 48
assert len(df) > 0

df.to_csv(
    OUT_CSV,
    index=False,
)

act = df[df["feature_type"] == "ACT_SITE"]
binding = df[df["feature_type"] == "BINDING"]

valid = df[df["coordinate_valid"]]
invalid = df[~df["coordinate_valid"]]

proteins_any = df["uniprot_id"].nunique()
proteins_act = act["uniprot_id"].nunique()
proteins_binding = binding["uniprot_id"].nunique()

summary = f"""
E10 FUNCTIONAL-SITE EXTRACTION
========================================
Batch files: {len(batch_files)}

Total retained features: {len(df)}
ACT_SITE features: {len(act)}
BINDING features: {len(binding)}

Proteins with any retained feature: {proteins_any}
Proteins with ACT_SITE: {proteins_act}
Proteins with BINDING: {proteins_binding}

Coordinate-valid features: {len(valid)}
Coordinate-invalid/ambiguous features: {len(invalid)}

Output:
{OUT_CSV}
""".strip()

SUMMARY.write_text(summary + "\n")

print(summary)

if len(invalid):
    print()
    print("First coordinate-invalid examples:")
    print(
        invalid[
            [
                "uniprot_id",
                "feature_type",
                "location_raw",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

print()
print("No model predictions used.")
print("No XAI results computed.")
print("No benchmark files modified.")