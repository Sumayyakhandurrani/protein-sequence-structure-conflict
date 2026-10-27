from pathlib import Path
import pandas as pd


BASE = Path("data/splits/mmseqs30_s75")


for split in ["train", "val", "test"]:

    input_file = BASE / f"{split}.pkl"
    output_file = BASE / f"{split}.fasta"

    print(f"Loading {split}...")

    df = pd.read_pickle(input_file)

    assert "uniprot_id" in df.columns
    assert "sequence" in df.columns

    assert df["uniprot_id"].is_unique
    assert df["sequence"].notna().all()

    with open(output_file, "w") as f:

        for row in df.itertuples():

            f.write(
                f">{row.uniprot_id}\n"
            )

            seq = str(row.sequence)

            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    print(
        f"{split}: {len(df):,} sequences -> {output_file}"
    )


print("\nFASTA export complete.")