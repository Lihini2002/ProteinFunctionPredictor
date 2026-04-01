# Import the function from feature_extraction.py
from descriptors import fasta_to_feature_csv

# Now you can call it
fasta_file = "combined_dataset.fasta"
output_csv = "descriptors.csv"

fasta_to_feature_csv(fasta_file, output_csv) 



