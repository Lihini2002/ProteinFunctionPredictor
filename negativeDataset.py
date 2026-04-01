import re
import requests
import pandas as pd
from io import StringIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# -----------------------------------------------
# 1. Extract positives from GAF file
# -----------------------------------------------
positives = set()
descriptions = []

with open("data.gaf") as f:
    for line in f:
        if line.startswith("!"):
            continue
        cols = line.strip().split("\t")
        protein_id = cols[0]
        protein_descriptions = cols[1]
        go_term = cols[3]

        if go_term == "GO:0008381":
            if re.match(r"^UniProtKB", protein_id):
                db, accession = protein_id.split(":", 1)
                positives.add(accession)
                descriptions.append(protein_descriptions)

desc_series = pd.Series(descriptions)
unique_desc = desc_series.unique()
pd.Series(unique_desc).to_csv("unique_descriptions.csv", index=False)
print(f"Positives found: {len(positives)}")

# -----------------------------------------------
# 2. Fetch TRULY RANDOM Swiss-Prot negatives
#    Only requirement: reviewed + NOT mechanosensitive GO term
# -----------------------------------------------
query = (
    'reviewed:true '
    'AND NOT go:0008381 '         # not mechanosensitive channel
    'AND NOT go:0097495 '         # not mechanosensitive ion channel (child term)
    'AND NOT protein_name:mechanosensitive '
    'AND NOT protein_name:piezo '
    'AND NOT protein_name:TMC '
    'AND NOT protein_name:SH3 '
)

url = "https://rest.uniprot.org/uniprotkb/search"
params = {
    "query": query,
    "format": "tsv",
    "fields": "accession,protein_name,sequence",
    "size": 500
}

all_df = []

while url:
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(r.text)
        raise Exception(f"UniProt error: {r.status_code}")

    print("Total results available:", r.headers.get("X-Total-Results", "unknown"))
    df = pd.read_csv(StringIO(r.text), sep="\t")
    df.columns = ["Entry", "Protein names", "Sequence"]
    all_df.append(df)

    total_rows = sum(len(d) for d in all_df)
    print(f"  Fetched {total_rows} so far...")

    if total_rows >= 5000:  # fetch more since we'll subsample
        break

    url = None
    params = None
    link_header = r.headers.get("Link", "")
    match = re.search(r'<(https://[^>]+)>;\s*rel="next"', link_header)
    if match:
        url = match.group(1)
        print(f"  Next page found")
    else:
        print("  No more pages")
        break

# -----------------------------------------------
# 3. Filter out positives
# -----------------------------------------------
final_df = pd.concat(all_df, ignore_index=True)
print(f"Total proteins fetched: {len(final_df)}")

positive_ids = set(line.strip() for line in open("positiveUniprot.txt"))
final_df = final_df[~final_df["Entry"].isin(positive_ids)]

# Drop rows with missing sequences
final_df = final_df.dropna(subset=["Sequence"])
print(f"After removing positives: {len(final_df)}")

# -----------------------------------------------
# 4. Random sample — match positive count for balance
# -----------------------------------------------
n_positives = len(positive_ids)
n_samples = min(n_positives * 2, len(final_df))  # 1:2 ratio max
sample_df = final_df.sample(n=n_samples, random_state=42)
print(f"Sampled {n_samples} random negatives (ratio 1:{n_samples//n_positives})")

# -----------------------------------------------
# 5. Save FASTA + TXT
# -----------------------------------------------
with open("random_negatives.fasta", "w") as fasta_f, open("random_negatives.txt", "w") as txt_f:
    for _, row in sample_df.iterrows():
        acc = row["Entry"]
        seq = row["Sequence"]
        header = f">{acc} {row['Protein names']}"

        fasta_f.write(header + "\n")
        for i in range(0, len(seq), 60):
            fasta_f.write(seq[i:i+60] + "\n")

        txt_f.write(acc + "\n")

print(f"Saved {n_samples} random negatives to random_negatives.fasta")