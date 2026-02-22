
# base_url = "www.ebi.ac.uk/proteins/api" 
import re
import requests
from Bio import SeqIO
from io import StringIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
# class negativeDatasetBuilder: 
#     def __init__(self):
#         pass

#     # def random_uniprot(self, n=1000): 
#     #     organism='9606',  # Human
#     #     exclude_go='GO:0008381'

#     # extract positives 
#     def extractPositives():
#         positives = set()
#         with open("data.gaf") as f:
#             for line in f:
#                 if line.startswith("!"):
#                     continue

#                 cols = line.strip().split("\t")
#                 protein_id = cols[0]
#                 go_term = cols[3]

#                 if go_term == "GO:0008381":
#                     positives.add(protein_id)
        
#         file_path = "positiveUniprot.txt"

#         with open(file_path, 'w') as file:
#             data_to_write = '\n'.join(str(item) for item in positives)
#             file.write(data_to_write)

 

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

                pattern = r"^UniProtKB"
                if go_term == "GO:0008381": 
                    if re.match(pattern,protein_id):
                        db, accession = protein_id.split(":", 1)
                        positives.add(accession)   
                        descriptions.append(protein_descriptions) 

# convert to pandas series
desc_series = pd.Series(descriptions)

# unique descriptions
unique_desc = desc_series.unique()
pd.Series(unique_desc).to_csv("unique_descriptions.csv", index=False)

               
    # Uncomment this part to rewrite the file    
# file_path = "positiveUniprot.txt"

# with open(file_path, 'w') as file:
#             data_to_write = '\n'.join(str(item) for item in positives)
#             file.write(data_to_write)   


# # getting protein seqeunces using uniprot IDS> 
# url = 'https://rest.uniprot.org/uniprotkb/stream'

# with open("positiveUniprot.txt") as f:
#     uniprot_ids = [line.strip() for line in f if line.strip()]

# url = "lp[]"
# batch_size = 100  
# all_sequences = {}

# for i in range(0, len(uniprot_ids), batch_size):
#     batch_ids = uniprot_ids[i:i+batch_size]

#     query = " OR ".join([f"accession:{uid}" for uid in batch_ids])
#     params = {
#         "format": "fasta",
#         "query": query
#     }

#     response = requests.get(url, params=params)
#     response.raise_for_status()

#     for record in SeqIO.parse(StringIO(response.text), "fasta"):
#         uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
#         all_sequences[uniprot_id] = str(record.seq)


# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord

# records = [
#     SeqRecord(Seq(seq), id=uid, description="")
#     for uid, seq in all_sequences.items()
# ]

# SeqIO.write(records, "positive_sequences.fasta", "fasta")



# getting the negative dataset from uniprot now 
# reviewed:true
# annotation:(type:transmembrane)
#NOT mechanoreceptor
# NOT piezo
# NOT mechanosensitive
# NOT stretch-activated
# ENaC/DEG
# K2P
# -not annotation:(go:0008381)
# family:"ion channel" OR family:"transporter"'


# First method i suppose  

# import requests
# import pandas as pd
# import random
# from io import StringIO

# import requests
# import pandas as pd
# from io import StringIO

# # -------------------------------
# # 1️⃣ Query (API-safe)
# # -------------------------------
# query = 'reviewed:true AND keyword:KW-0812 AND (protein_name:channel OR protein_name:receptor OR protein_name:ion) AND NOT protein_name:mechanosensitive AND NOT go:0008381'


# url = "https://rest.uniprot.org/uniprotkb/search"

# # -------------------------------
# # 2️⃣ Load positives
# # -------------------------------
# positives = set()
# with open("positiveUniprot.txt") as f:
#     for line in f:
#         positives.add(line.strip())

# # -------------------------------
# # 3️⃣ Query UniProt
# # -------------------------------
# params = {
#     "query": query,
#     "format": "tsv",
#     "fields": "accession,protein_name,sequence",
#     "size": 500
# }

# response = requests.get(url, params=params)

# if response.status_code != 200:
#     print(response.text)
#     raise Exception("UniProt error:", response.status_code)

# df = pd.read_csv(StringIO(response.text), sep="\t")
# df.columns = ["Entry", "Protein names", "Sequence"]

# print("Total returned:", len(df))

# # -------------------------------
# # 4️⃣ Remove positives
# # -------------------------------
# df = df[~df["Entry"].isin(positives)]
# print("After removing positives:", len(df))

# if len(df) < 700:
#     raise Exception("Not enough negatives — relax query")

# # -------------------------------
# # 5️⃣ Random sample
# # -------------------------------
# sample_df = df.sample(n=700, random_state=42)

# # -------------------------------
# # 6️⃣ Save FASTA + ID list
# # -------------------------------
# neg_accessions = []

# with open("hard_negatives.fasta", "w") as fasta_f:
#     for _, row in sample_df.iterrows():
#         acc = row["Entry"]
#         neg_accessions.append(acc)

#         header = f">{acc} {row['Protein names']}"
#         seq = row["Sequence"]

#         fasta_f.write(header + "\n")
#         for i in range(0, len(seq), 60):
#             fasta_f.write(seq[i:i+60] + "\n")

# with open("hard_negatives.txt", "w") as txt_f:
#     txt_f.write("\n".join(neg_accessions))

# print("Saved:")
# print(" hard_negatives.fasta")
# print(" hard_negatives.txt")



# import requests

# url = "https://rest.uniprot.org/uniprotkb/search"
# query = (
#     'reviewed:true '
#     'AND keyword:KW-0812 '
#     'AND (protein_name:channel OR protein_name:receptor) '
#     'AND NOT protein_name:mechanosensitive '
#     'AND NOT go:0008381'
# )


# params = {
#     "query": query,
#     "format": "tsv",
#     "size": 20

# }
# r = requests.get(url, params=params)

# print("STATUS:", r.status_code)
# print("URL:", r.url)
# print("RESPONSE:", r.text[:500])



import requests
import pandas as pd
from io import StringIO

# Query
query = (
    'reviewed:true '
    'AND keyword:KW-0812 '
    'AND (protein_name:channel OR protein_name:transporter OR protein_name:receptor) '
    'AND NOT protein_name:mechanosensitive '
    'AND NOT go:0008381'
)

url = "https://rest.uniprot.org/uniprotkb/search"
params = {
    "query": query,
    "format": "tsv",
    "fields": "accession,protein_name,sequence",
    "size": 200  # max per request
}

all_df = []

while url:
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(r.text)
        raise Exception("UniProt error", r.status_code)
    
    df = pd.read_csv(StringIO(r.text), sep="\t")
    df.columns = ["Entry", "Protein names", "Sequence"]
    all_df.append(df)

    total_rows = sum(len(d) for d in all_df)
    if total_rows >= 800:
        break

    # check if there is a next page
    next_link = None
if "Link" in r.headers:
    links = r.headers["Link"].split(",")
    for l in links:
        l = l.strip()
        if 'rel="next"' in l:
            start = l.find("<")
            end = l.find(">")
            if start != -1 and end != -1:
                next_link = l[start+1:end]
            break


# combine all batches
final_df = pd.concat(all_df, ignore_index=True)
print("Total proteins fetched:", len(final_df))

positives = set(line.strip() for line in open("positiveUniprot.txt"))
final_df = final_df[~final_df["Entry"].isin(positives)]

# -------------------------------
# 5️⃣ Sample 700 (or max available)
# -------------------------------
n_samples = min(800, len(final_df))
sample_df = final_df.sample(n=n_samples, random_state=42)

# -------------------------------
# 6️⃣ Save FASTA + TXT
# -------------------------------
with open("hard_negatives.fasta", "w") as fasta_f, open("hard_negatives.txt", "w") as txt_f:
    for _, row in sample_df.iterrows():
        acc = row["Entry"]
        seq = row["Sequence"]
        header = f">{acc} {row['Protein names']}"

        fasta_f.write(header + "\n")
        for i in range(0, len(seq), 60):
            fasta_f.write(seq[i:i+60] + "\n")
        
        txt_f.write(acc + "\n")

print(f"Saved {n_samples} hard negatives")