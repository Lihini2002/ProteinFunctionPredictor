# models/esm_ann_classifier.py

"""
ESM-2 (frozen) + ANN classifier architecture
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

class ESM2_ANN_Classifier(nn.Module):
    """
    Complete model: ESM-2 (frozen) + ANN head
    
    Architecture:
    Input Sequence → ESM-2 (frozen) → Embedding → ANN → Prediction
    """
    
    def __init__(self, 
                 esm_model_name="facebook/esm2_t30_150M_UR50D",
                 freeze_esm=True,
                #  even if we reduce hidden dims to a really really low number it is still overfit 
                # embeddings are really powerful. 
                 hidden_dims=[512, 256, 128],
                #  initially the dropout was set to be 0.3 
                # stronger dropout can 
                 dropout=0.6):
        """
        Args:
            esm_model_name: ESM-2 model to use
            freeze_esm: Whether to freeze ESM-2 weights (recommended)
            hidden_dims: Hidden layer dimensions for ANN
            dropout: Dropout probability
        """
        super(ESM2_ANN_Classifier, self).__init__()
        
        print(f"Initializing ESM-2 + ANN Classifier...")
        print(f"  ESM-2 model: {esm_model_name}")
        print(f"  Freeze ESM-2: {freeze_esm}")
        
        # Load ESM-2
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        
        # Freeze ESM-2 if specified
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
            print("  ✓ ESM-2 frozen (will not be trained)")
        
        # Get embedding dimension
        self.embedding_dim = self.esm_model.config.hidden_size
        
        # Build ANN classifier head
        layers = []
        prev_dim = self.embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.ann_classifier = nn.Sequential(*layers)
        
        # Print architecture
        esm_params = sum(p.numel() for p in self.esm_model.parameters())
        ann_params = sum(p.numel() for p in self.ann_classifier.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n  Model Statistics:")
        print(f"    ESM-2 parameters: {esm_params:,}")
        print(f"    ANN parameters: {ann_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        print(f"    Total parameters: {esm_params + ann_params:,}")
        print(f"\n  ✓ Model initialized")
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Tokenized sequences (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            predictions: (batch_size, 1)
        """
        
        # Get ESM-2 embeddings
        with torch.set_grad_enabled(self.training and not self.esm_model.training):
            esm_output = self.esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Extract embeddings
        hidden_states = esm_output.last_hidden_state  # (batch, seq_len, hidden_dim)
        
        # Mean pooling (exclude padding)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask  # (batch, hidden_dim)
        
        # Classify with ANN
        predictions = self.ann_classifier(embeddings)
        
        return predictions
    
    def get_embedding(self, input_ids, attention_mask):
        """
        Get ESM-2 embedding without classification
        Useful for explainability
        """
        with torch.no_grad():
            esm_output = self.esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        hidden_states = esm_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings


# Simple ANN classifier (for pre-computed embeddings)
class SimpleANNClassifier(nn.Module):
    """
    Simpler version: Just ANN on pre-computed embeddings
    Use this if you pre-computed embeddings on Colab
    """
    
    def __init__(self, input_dim=640, hidden_dims=[512, 256, 128], dropout=0.3):
        super(SimpleANNClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        
        print(f"SimpleANN initialized: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, x):
        return self.network(x)