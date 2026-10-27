from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/all_swissprot_mf_raw.tsv")
OUTPUT = Path("data/processed/swissprot_mf_candidates.csv")

df = pd.read_csv(INPUT, sep="\t")

print("Total downloaded entries:", len(df))

mf_col = "Gene Ontology (molecular function)"

mf = df[
    df[mf_col].notna()
].copy()

mf = mf.rename(
    columns={
        "Entry": "uniprot_id",
        mf_col: "mf_terms_raw"
    }
)

print("Proteins with MF annotations:", len(mf))
print("Unique proteins:", mf["uniprot_id"].nunique())

mf.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved to:")
print(OUTPUT)

print("\nFirst five:")
print(mf.head().to_string())