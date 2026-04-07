from collections import Counter

# /string of 20 amino acids
AA = "ACDEFGHIKLMNPQRSTVWY"


# amino acid composition 
# counts how many times each amino acid appears in teh seuence. 
def aac(sequence: str) -> dict:
    seq = sequence.upper()
    L = len(seq)
    counts = Counter(seq)
    return {f"AAC_{aa}": counts.get(aa, 0) / L for aa in AA}



# Mean hydrophobicity 
# These are the Kyte-Doolittle hydrophobicity values for the 20 amino acids.
# Positive numbers --> hydrophobic (water-repelling) amino acids.
# Negative numbers ___> hydrophilic (water-attracting) amino acids.

# for every amino acid in the sequence this retrieves its hydrophobicity 
KD = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

def mean_hydrophobicity(sequence: str) -> dict:
    seq = sequence.upper()
    values = [KD.get(aa, 0) for aa in seq if aa in AA]
    if not values:  # handle empty sequence
        return {"hydro_mean": 0.0, "hydro_std": 0.0}
    mean_val = sum(values) / len(values)
    std_val  = (sum((x - mean_val)**2 for x in values) / len(values))**0.5
    return {"hydro_mean": mean_val, "hydro_std": std_val}

# uncomment if accuracy is too low 
# def aa_composition_features(seq: str, pa: ProteinAnalysis) -> OrderedDict:
#     """
#     20 individual aa fractions + 9 biochemical group fractions.
#     Mechanoreceptors are enriched in hydrophobic and helix-breaking
#     residues compared with average proteins.
#     """
#     f = OrderedDict()
#     comp = pa.get_amino_acids_percent()
 
#     for aa in sorted('ACDEFGHIKLMNPQRSTVWY'):
#         f[f'aa_frac_{aa}'] = comp.get(aa, 0.0)
 
#     f['frac_hydrophobic']  = sum(comp.get(a, 0) for a in 'AVILMFWP')
#     f['frac_charged']      = sum(comp.get(a, 0) for a in 'DEKR')
#     f['frac_polar']        = sum(comp.get(a, 0) for a in 'STNQHY')
#     f['frac_helix_break']  = sum(comp.get(a, 0) for a in 'GP')
#     f['frac_aromatic']     = sum(comp.get(a, 0) for a in 'FWY')
#     f['frac_tiny']         = sum(comp.get(a, 0) for a in 'GAS')
#     f['frac_positive']     = sum(comp.get(a, 0) for a in 'KR')
#     f['frac_negative']     = sum(comp.get(a, 0) for a in 'DE')
#     f['charge_ratio']      = (
#         f['frac_positive'] / max(f['frac_negative'], 1e-6)
#     )
 
#     return f

# uncomment if accuracy is too low 
# Charge distribution features 
# def charge_distribution_features(seq: str) -> OrderedDict:
#     """
#     Where charges sit along the sequence.
#     The positive-inside rule: intracellular loops of TM proteins
#     are enriched in K/R.
#     Mechanogating relies on electrostatic coupling to the membrane.
#     """
#     f = OrderedDict()
#     n = len(seq)
 
#     def net_charge(s: str) -> int:
#         return sum(
#             1 if aa in POSITIVE_AA else -1 if aa in NEGATIVE_AA else 0
#             for aa in s
#         )
 
#     def pos_density(s: str) -> float:
#         return sum(1 for aa in s if aa in POSITIVE_AA) / max(len(s), 1)
 
#     def neg_density(s: str) -> float:
#         return sum(1 for aa in s if aa in NEGATIVE_AA) / max(len(s), 1)
 
#     f['charge_net']         = float(net_charge(seq))
#     f['charge_density_pos'] = pos_density(seq)
#     f['charge_density_neg'] = neg_density(seq)
 
#     # Split into N-terminal / middle / C-terminal thirds
#     t1, t2    = n // 3, 2 * n // 3
#     nterm     = seq[:t1]
#     middle    = seq[t1:t2]
#     cterm     = seq[t2:]
 
#     f['charge_nterm']  = float(net_charge(nterm))
#     f['charge_middle'] = float(net_charge(middle))
#     f['charge_cterm']  = float(net_charge(cterm))
 
#     # Asymmetry across thirds
#     f['charge_asymmetry'] = float(
#         np.std([f['charge_nterm'], f['charge_middle'], f['charge_cterm']])
#     )
 
#     # Longest consecutive charged run
#     f['charge_max_run'] = float(
#         max((len(m.group()) for m in re.finditer(r'[DEKR]+', seq)), default=0)
#     )
 
#     return f
 
def charge_features(sequence: str) -> dict:
    L = len(sequence)
    pos = sum(sequence.count(aa) for aa in "KRH")
    neg = sum(sequence.count(aa) for aa in "DE")
    return {
        "charge_net": (pos - neg) / L,
        "charge_density": (pos + neg) / L
    }

# import re 
# MOTIFS = {
#     "motif_proline_hinge": r"P..P",
#     "motif_glycine_rich": r"GG.",
#     "motif_pos_cluster": r"[KR]{3,}",
#     "motif_aromatic_pair": r"F.Y",
#     "motif_hydrophobic_stretch": r"[AILMFWV]{6,}"
# }
# def motif_features(sequence:str) -> dict:
#     L = len(sequence)
#     feats = {}

#     for name, pattern in MOTIFS.items():
#         matches = re.findall(pattern, sequence)
#         feats[name + "_count"] = len(matches)
#         feats[name + "_density"] = len(matches) / max(L, 1)

#     return feats
    




def gp_features(sequence: str) -> dict:
    L = len(sequence)
    g = sequence.count('G')
    p = sequence.count('P')
    return {
        "gly_fraction": g / L,
        "pro_fraction": p / L,
        "gp_fraction": (g + p) / L
    }

# For mechanoreceptors or GPCRs, aromatic belts 
# contribute to membrane anchoring and gating mechanics.
# Aromatic residues are often at the lipid-water interface in membrane proteins.
def aromatic_features(sequence: str) -> dict:
    L = len(sequence)
    arom = sum(sequence.count(aa) for aa in "FWY")
    return {
        "aromatic_fraction": arom / L
    }

# long hydrophobic stretches. 
# This usually refers to a continuous stretch of hydrophobic residues,
# which can approximate transmembrane segments.
def hydrophobic_stretches(sequence: str, threshold=1.6, window=18) -> dict:
    scores = [KD.get(aa, 0) for aa in sequence]
    count = 0
    for i in range(len(scores) - window + 1):
        if sum(scores[i:i+window]) / window > threshold:
            count += 1
    return {"hydrophobic_stretch_count": count}


def low_complexity_fraction(sequence: str, window=12) -> dict:
    import math
    lc = 0
    for i in range(len(sequence)-window+1):
        w = sequence[i:i+window]
        probs = [w.count(a)/window for a in set(w)]
        entropy = -sum(p*math.log2(p) for p in probs)
        if entropy < 2.2:  # low complexity threshold
            lc += 1
    return {"low_complexity_frac": lc / len(sequence)}


from itertools import product
from collections import Counter

AA = "ACDEFGHIKLMNPQRSTVWY"

def dipeptide_composition(sequence: str) -> dict:
    seq = sequence.upper()
    L = len(seq) - 1
    pairs = [a+b for a,b in product(AA, AA)]
    counts = Counter(seq[i:i+2] for i in range(L))
    return {f"DPC_{p}": counts.get(p, 0)/L for p in pairs}



def pI_feature(sequence: str) -> dict:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        analysis = ProteinAnalysis(sequence)
        return {"pI": analysis.isoelectric_point()}
    except Exception:
        # Default pI if calculation fails
        return {"pI": 7.0}

def extract_features(sequence: str) -> dict:
    """
    Compute all hand-coded descriptors for a protein sequence.
    Returns a single dictionary of features.
    """
    features = {}

    # --- Basic sequence features ---
    features.update(aac(sequence))                  # Amino acid composition
    features.update(dipeptide_composition(sequence))  # Dipeptide composition

    # --- Physicochemical / biochemical ---
    features.update(mean_hydrophobicity(sequence)) # Kyte-Doolittle hydrophobicity
    features.update(charge_features(sequence))     # Net charge & charge density
    features.update(gp_features(sequence))         # Glycine / Proline fractions
    features.update(aromatic_features(sequence))   # F, W, Y aromatic fraction
    features.update(hydrophobic_stretches(sequence)) # Long hydrophobic stretches
    features.update(low_complexity_fraction(sequence)) # Low-complexity fraction
    features.update(pI_feature(sequence))          # Isoelectric point

    # You can add more here if you implement other scales (flexibility, BLOSUM62, ZSCALE...)

    return features

def read_fasta_with_labels(fasta_file):
    sequences = []
    
    with open(fasta_file, 'r') as f:
        name = None
        seq_lines = []
        label = None

        for line in f:
            line = line.strip()

            if line.startswith('>'):
                # Save previous record
                if name is not None:
                    sequences.append((name, ''.join(seq_lines), int(label)))

                # Parse new header
                header = line[1:]
                parts = header.split()

                name = parts[0]

                # Expecting "label=0" or "label=1"
                for p in parts:
                    if p.startswith('label='):
                        label = p.split('=')[1]

                seq_lines = []

            else:
                seq_lines.append(line)

        # Save last record
        if name is not None:
            sequences.append((name, ''.join(seq_lines), int(label)))

    return sequences

import joblib
import csv
def fasta_to_feature_csv(fasta_file, output_csv):
    sequences = read_fasta_with_labels(fasta_file)  # Your FASTA reader
    # Get feature names from a dummy sequence
    feature_names = list(extract_features('ACDEFGHIKLMNPQRSTVWY').keys())

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['name'] + feature_names + ['Label']
        )
        writer.writeheader()

        for name, seq, label in sequences:
            features = extract_features(seq)
            features['name'] = name
            features['Label'] = label
            writer.writerow(features)
    
    print(f"Feature extraction complete. Saved to {output_csv}")