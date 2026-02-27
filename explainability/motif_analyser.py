# explainability/motif_analyzer.py

"""
Detects biological motifs and patterns in protein sequences
"""

from collections import Counter
import re
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

class MotifAnalyzer:
    """
    Analyzes protein sequences for known motifs and patterns
    """
    
    def __init__(self, positive_sequences, negative_sequences):
        """
        Initialize with training sequences to learn discriminative patterns
        
        Args:
            positive_sequences: List of positive training sequences (strings or SeqRecords)
            negative_sequences: List of negative training sequences
        """
        
        # Convert SeqRecords to strings if needed
        self.pos_seqs = [str(s.seq) if hasattr(s, 'seq') else str(s) for s in positive_sequences]
        self.neg_seqs = [str(s.seq) if hasattr(s, 'seq') else str(s) for s in negative_sequences]
        
        # Learn discriminative k-mers
        self.discriminative_kmers = self._find_discriminative_kmers(k=3, top_n=50)
        
        # Known mechanosensitive motifs (from literature)
        # add more into this.
        self.known_motifs = {
            'KRK': 'Positive charge cluster (mechanosensing domain)',
            'GXXXG': 'Helix-helix packing motif (transmembrane)',
            'DED': 'Negative charge cluster',
            'LVLG': 'Mechanosensitive channel signature',
            '[RK]XX[RK]': 'Voltage sensor-like pattern',
            'TXXTXP': 'Ion selectivity filter',
        }
        
        print(f"✓ MotifAnalyzer initialized")
        print(f"  Found {len(self.discriminative_kmers)} discriminative k-mers")
    
    def _find_discriminative_kmers(self, k=3, top_n=50):
        """
        Find k-mers enriched in positives vs negatives
        
        Returns:
            dict: {kmer: enrichment_score}
        """
        
        # Count k-mers in positive set
        pos_kmers = Counter()
        for seq in self.pos_seqs:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if kmer.isalpha():  # Valid amino acids only
                    pos_kmers[kmer] += 1
        
        # Count k-mers in negative set
        neg_kmers = Counter()
        for seq in self.neg_seqs:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if kmer.isalpha():
                    neg_kmers[kmer] += 1
        
        # Calculate enrichment (odds ratio)
        enrichment = {}
        for kmer in pos_kmers:
            pos_freq = pos_kmers[kmer] / len(self.pos_seqs)
            neg_freq = neg_kmers.get(kmer, 1) / len(self.neg_seqs)
            
            if neg_freq > 0:
                odds_ratio = pos_freq / neg_freq
                if odds_ratio > 2.0:  # At least 2x enriched
                    enrichment[kmer] = odds_ratio
        
        # Return top N
        top = sorted(enrichment.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(top)
    
    def analyze(self, sequence):
        """
        Find all motifs and patterns in a sequence
        
        Args:
            sequence: Protein sequence (string)
        
        Returns:
            dict with found motifs
        """
        
        results = {
            'known_motifs': [],
            'discriminative_kmers': [],
            'charge_clusters': [],
            'hydrophobic_regions': [],
            'composition': {}
        }
        
        # 1. Search for known motifs
        for motif, description in self.known_motifs.items():
            # Convert to regex
            pattern = motif.replace('X', '.')
            
            for match in re.finditer(pattern, sequence):
                results['known_motifs'].append({
                    'motif': motif,
                    'position': match.start(),
                    'end': match.end(),
                    'sequence': match.group(),
                    'description': description
                })
        
        # 2. Search for discriminative k-mers
        for kmer, enrichment in self.discriminative_kmers.items():
            positions = [i for i in range(len(sequence) - len(kmer) + 1)
                        if sequence[i:i+len(kmer)] == kmer]
            
            if positions:
                results['discriminative_kmers'].append({
                    'kmer': kmer,
                    'count': len(positions),
                    'positions': positions[:10],  # First 10 positions
                    'enrichment': enrichment
                })
        
        # 3. Find charge clusters
        results['charge_clusters'] = self._find_charge_clusters(sequence)
        
        # 4. Find hydrophobic regions (potential transmembrane)
        results['hydrophobic_regions'] = self._find_hydrophobic_regions(sequence)
        
        # 5. Composition analysis
        results['composition'] = self._analyze_composition(sequence)
        
        return results
    
    def _find_charge_clusters(self, sequence, min_length=3):
        """Find clusters of charged amino acids"""
        
        clusters = []
        
        # Positive charge clusters (K, R)
        for match in re.finditer(r'[KR]{3,}', sequence):
            clusters.append({
                'type': 'positive',
                'position': match.start(),
                'end': match.end(),
                'length': len(match.group()),
                'sequence': match.group()
            })
        
        # Negative charge clusters (D, E)
        for match in re.finditer(r'[DE]{3,}', sequence):
            clusters.append({
                'type': 'negative',
                'position': match.start(),
                'end': match.end(),
                'length': len(match.group()),
                'sequence': match.group()
            })
        
        return clusters
    
    def _find_hydrophobic_regions(self, sequence, window=19, threshold=0.6):
        """
        Find hydrophobic regions (potential transmembrane helices)
        
        Args:
            window: Window size (typical TM helix ~19-25 residues)
            threshold: Fraction of hydrophobic residues required
        """
        
        hydrophobic = set('AVILMFYW')
        regions = []
        
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i+window]
            hydro_frac = sum(aa in hydrophobic for aa in window_seq) / window
            
            if hydro_frac >= threshold:
                # Check if extending existing region
                if regions and regions[-1]['end'] >= i:
                    regions[-1]['end'] = i + window
                    regions[-1]['length'] = regions[-1]['end'] - regions[-1]['start']
                else:
                    regions.append({
                        'start': i,
                        'end': i + window,
                        'length': window,
                        'hydrophobicity': hydro_frac,
                        'sequence': window_seq
                    })
        
        return regions
    
    def _analyze_composition(self, sequence):
        """Analyze amino acid composition"""
        
        try:
            analyzed = ProteinAnalysis(sequence)
            
            return {
                'length': len(sequence),
                'molecular_weight': analyzed.molecular_weight(),
                'aromaticity': analyzed.aromaticity(),
                'instability_index': analyzed.instability_index(),
                'isoelectric_point': analyzed.isoelectric_point(),
                'gravy': analyzed.gravy(),  # Hydrophobicity
                'secondary_structure': analyzed.secondary_structure_fraction()
            }
        except Exception as e:
            print(f"Warning: Could not analyze composition: {e}")
            return {'length': len(sequence)}
    
    def summarize_findings(self, analysis):
        """
        Create human-readable summary
        
        Args:
            analysis: Output from analyze()
        
        Returns:
            list of summary strings
        """
        
        summary = []
        
        # Known motifs
        if analysis['known_motifs']:
            summary.append(f"Found {len(analysis['known_motifs'])} known mechanosensitive motifs")
            for motif in analysis['known_motifs'][:3]:
                summary.append(f"  • {motif['motif']} at position {motif['position']}: {motif['description']}")
        else:
            summary.append("No known mechanosensitive motifs detected")
        
        # Discriminative k-mers
        strong_kmers = [k for k in analysis['discriminative_kmers'] if k['enrichment'] > 3.0]
        if strong_kmers:
            summary.append(f"Found {len(strong_kmers)} strongly discriminative patterns")
            for kmer in strong_kmers[:3]:
                summary.append(f"  • '{kmer['kmer']}' appears {kmer['count']} times (enrichment: {kmer['enrichment']:.1f}x)")
        
        # Charge clusters
        if analysis['charge_clusters']:
            summary.append(f"Found {len(analysis['charge_clusters'])} charge clusters")
            for cluster in analysis['charge_clusters'][:2]:
                summary.append(f"  • {cluster['type'].capitalize()} cluster at {cluster['position']}: {cluster['sequence']}")
        
        # Hydrophobic regions
        if analysis['hydrophobic_regions']:
            summary.append(f"Found {len(analysis['hydrophobic_regions'])} potential transmembrane regions")
            for region in analysis['hydrophobic_regions'][:2]:
                summary.append(f"  • Positions {region['start']}-{region['end']} ({region['hydrophobicity']:.1%} hydrophobic)")
        
        return summary