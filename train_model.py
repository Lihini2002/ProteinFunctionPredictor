# scripts/03_train_model.py

"""
Train ANN classifier on pre-computed ESM-2 embeddings
This runs fast on CPU!
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from models.esm_ann_classifier import SimpleANNClassifier

# ============================================================================
# Dataset Class
# ============================================================================

class EmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings"""
    
    def __init__(self, embeddings_dict, labels, ids):
        self.ids = ids
        self.embeddings = [embeddings_dict[id] for id in ids]
        self.labels = labels
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'embedding': torch.FloatTensor(self.embeddings[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
        }

# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, val_loader, 
                num_epochs=50, learning_rate=0.001, device='cpu'):
    """
    Train the ANN classifier
    """
    
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    best_val_auc = 0
    patience_counter = 0
    patience_limit = 10
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    print("Hi")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_probs.extend(outputs.cpu().numpy().flatten())
                val_preds.extend((outputs > 0.5).cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"  ✓ New best model (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        # Learning rate scheduling
        scheduler.step(val_auc)
    
    # Load best model
    model.load_state_dict(torch.load('models/best_model.pth'))
    
    return model, history

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set
    """
    
    model = model.to(device)
    model.eval()
    
    test_preds = []
    test_probs = []
    test_labels = []
    test_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label']
            
            outputs = model(embeddings)
            
            test_probs.extend(outputs.cpu().numpy().flatten())
            test_preds.extend((outputs > 0.5).cpu().numpy().flatten())
            test_labels.extend(labels.numpy().flatten())
            test_ids.extend(batch['id'])
    
    # Metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    print(f"\nAccuracy: {test_acc:.4f}")
    print(f"ROC AUC:  {test_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=['Negative', 'Positive (Mechanosensitive)']
    ))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {test_auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved ROC curve to results/roc_curve.png")
    
    return {
        'accuracy': test_acc,
        'auc': test_auc,
        'predictions': test_preds,
        'probabilities': test_probs,
        'labels': test_labels,
        'ids': test_ids
    }

# ============================================================================
# Main
# ============================================================================

def main():
    """
    Main training pipeline
    """
    
    print("="*70)
    print("ESM-2 + ANN CLASSIFIER TRAINING")
    print("="*70)
    
    # Load embeddings
    print("\n1. Loading embeddings...")
    with open('data/protein_embeddings.pkl', 'rb') as f:
        embedding_data = pickle.load(f)
    
    embedding_dim = embedding_data['embedding_dim']
    print(f"   ✓ Embedding dimension: {embedding_dim}")
    print(f"   ✓ Model: {embedding_data['model_name']}")
    
    # Create datasets
    print("\n2. Creating datasets...")
    train_dataset = EmbeddingDataset(
        embedding_data['train']['embeddings'],
        embedding_data['train']['labels'],
        embedding_data['train']['ids']
    )
    
    val_dataset = EmbeddingDataset(
        embedding_data['val']['embeddings'],
        embedding_data['val']['labels'],
        embedding_data['val']['ids']
    )
    
    test_dataset = EmbeddingDataset(
        embedding_data['test']['embeddings'],
        embedding_data['test']['labels'],
        embedding_data['test']['ids']
    )
    
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val:   {len(val_dataset)}")
    print(f"   Test:  {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("\n3. Initializing model...")
    model = SimpleANNClassifier(
        input_dim=embedding_dim,
        hidden_dims=[512, 256, 128],
        dropout=0.3
    )
    
    # Train
    print("\n4. Training model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    model, history = train_model(
        model, train_loader, val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device
    )
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.plot(history['val_auc'], label='Val AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Training Metrics')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved training curves to results/training_curves.png")
    
    # Evaluate
    print("\n5. Evaluating on test set...")
    results = evaluate_model(model, test_loader, device=device)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'test_accuracy': results['accuracy'],
        'test_auc': results['auc'],
        'history': history
    }, 'models/final_model.pth')
    
    print("\n✓ Saved final model to models/final_model.pth")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test AUC:      {results['auc']:.4f}")

if __name__ == "__main__":
    main()