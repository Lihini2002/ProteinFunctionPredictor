# scripts/01_prepare_dataset.py

"""
Prepare dataset from FASTA files
Split into train/val/test sets

"""

from Bio import SeqIO
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def load_sequences_from_fasta(fasta_file, label):
    """
    Load sequences from FASTA file
    
    Args:
        fasta_file: Path to FASTA file
        label: 0 or 1 (negative or positive)
    
    Returns:
        List of dicts with id, sequence, label
    """
    sequences = []
    
    for record in SeqIO.parse(fasta_file, 'fasta'):
        sequences.append({
            'id': record.id,
            'sequence': str(record.seq),
            'label': label,
            'length': len(record.seq)
        })
    
    return sequences

def filter_sequences(sequences, min_length=50, max_length=1500):
    """Filter sequences by length"""
    
    filtered = [s for s in sequences 
                if min_length <= s['length'] <= max_length]
    
    print(f"Length filtering: {len(sequences)} → {len(filtered)}")
    
    return filtered

def prepare_dataset():
    """
    Main function to prepare dataset
    """
    
    print("="*70)
    print("PREPARING DATASET")
    print("="*70)
    
    # Load positive sequences
    print("\n1. Loading positive sequences...")
    positives = load_sequences_from_fasta('positive_sequences.fasta', label=1)
    print(f"   ✓ Loaded {len(positives)} positive sequences")
    
    # Load negative sequences
    print("\n2. Loading negative sequences...")
    negatives = load_sequences_from_fasta('random_negatives.fasta', label=0)
    print(f"   ✓ Loaded {len(negatives)} negative sequences")
    
    # Filter by length
    print("\n3. Filtering sequences...")
    positives = filter_sequences(positives)
    negatives = filter_sequences(negatives)
    
    # Statistics
    pos_lengths = [s['length'] for s in positives]
    neg_lengths = [s['length'] for s in negatives]
    
    print(f"\n   Positive lengths: {min(pos_lengths)} - {max(pos_lengths)} aa (μ={np.mean(pos_lengths):.0f})")
    print(f"   Negative lengths: {min(neg_lengths)} - {max(neg_lengths)} aa (μ={np.mean(neg_lengths):.0f})")
    
    # Combine
    all_data = positives + negatives
    
    # Create arrays
    ids = [s['id'] for s in all_data]
    sequences = [s['sequence'] for s in all_data]
    labels = [s['label'] for s in all_data]
    
    print(f"\n4. Total dataset: {len(all_data)} proteins")

    def save_to_fasta(sequences_list, output_file):
        with open(output_file, 'w') as f:
            for s in sequences_list:
            # Write header line with id and label
                f.write(f">{s['id']} label={s['label']}\n")
            # Write sequence in 60-character chunks (standard FASTA format)
                seq = s['sequence']
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + '\n')
        print(f"   ✓ Saved {len(sequences_list)} sequences to {output_file}")

        # Save combined dataset
    print("\n5. Saving combined dataset...")
    save_to_fasta(all_data, 'combined_dataset.fasta')

    print(f"   Positive: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"   Negative: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    # Split into train/val/test
    print("\n5. Splitting dataset...")
    
    # First split: train+val vs test
    train_val_ids, test_ids, train_val_seqs, test_seqs, train_val_labels, test_labels = train_test_split(
        ids, sequences, labels,
        test_size=0.15,
        random_state=42,
        stratify=labels
    )
    
    # Second split: train vs val
    train_ids, val_ids, train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        train_val_ids, train_val_seqs, train_val_labels,
        test_size=0.176,  # 0.15 of original = 0.15/0.85 ≈ 0.176
        random_state=42,
        stratify=train_val_labels
    )
    
    print(f"   Train: {len(train_ids)} ({sum(train_labels)} pos, {len(train_labels)-sum(train_labels)} neg)")
    print(f"   Val:   {len(val_ids)} ({sum(val_labels)} pos, {len(val_labels)-sum(val_labels)} neg)")
    print(f"   Test:  {len(test_ids)} ({sum(test_labels)} pos, {len(test_labels)-sum(test_labels)} neg)")
    
    # Save datasets
    print("\n6. Saving datasets...")
    
    train_data = {
        'ids': train_ids,
        'sequences': train_seqs,
        'labels': train_labels
    }
    
    val_data = {
        'ids': val_ids,
        'sequences': val_seqs,
        'labels': val_labels
    }
    
    test_data = {
        'ids': test_ids,
        'sequences': test_seqs,
        'labels': test_labels
    }
    
    with open('data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open('data/val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    
    with open('data/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print("   ✓ Saved train_data.pkl")
    print("   ✓ Saved val_data.pkl")
    print("   ✓ Saved test_data.pkl")
    
    # Create summary
    summary = {
        'total_proteins': len(all_data),
        'positive_count': sum(labels),
        'negative_count': len(labels) - sum(labels),
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'test_size': len(test_ids),
        'avg_length_pos': np.mean(pos_lengths),
        'avg_length_neg': np.mean(neg_lengths)
    }
    
    with open('data/dataset_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    
    print("\n" + "="*70)
    print("✓ DATASET PREPARATION COMPLETE")
    print("="*70)
    print("\nNext step: Extract ESM-2 embeddings")
    print("Run: scripts/02_extract_embeddings.py (on Google Colab with GPU)")

if __name__ == "__main__":
    prepare_dataset()