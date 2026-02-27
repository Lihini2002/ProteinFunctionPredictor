# explainability/feature_attribution.py

"""
Explains which embedding dimensions are important for predictions
Uses SHAP (SHapley Additive exPlanations)
"""

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureAttributor:
    """
    Explains which parts of the ESM-2 embedding drove the prediction
    """
    
    def __init__(self, model, background_embeddings, device='cpu'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained ANN classifier
            background_embeddings: Sample of training embeddings for SHAP baseline
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Use subset of background for efficiency
        if len(background_embeddings) > 100:
            # Randomly using a subset of the embeddings to make a background embedding 
        
            indices = np.random.Generator(len(background_embeddings), 100, replace=False)
            background = background_embeddings[indices]
        else:
            background = background_embeddings
        
        # Create SHAP explainer
        self.explainer = shap.DeepExplainer(
            
            self.model,
            torch.FloatTensor(background).to(device)
        )
        
        print(f"✓ FeatureAttributor initialized with {len(background)} background samples")
    
    def explain(self, embedding, top_k=15):
        """
        Explain which embedding dimensions are important
        
        Args:
            embedding: Single embedding vector (numpy array)
            top_k: Number of top features to return
        
        Returns:
            dict with attribution information
        """
        
        # Convert to tensor
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(embedding_tensor)
        
        # Extract importance (absolute values)
        if isinstance(shap_values, list):
            importance = np.abs(shap_values[0][0])
        else:
            importance = np.abs(shap_values[0])
        
        # Get top features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_values = importance[top_indices]
        
        # Calculate statistics
        total_importance = np.sum(importance)
        top_contribution = np.sum(top_values) / total_importance if total_importance > 0 else 0
        
        return {
            'top_features': top_indices.tolist(),
            'importance_scores': top_values.tolist(),
            'all_importances': importance,
            'mean_importance': float(np.mean(importance)),
            'max_importance': float(np.max(importance)),
            'std_importance': float(np.std(importance)),
            'top_contribution': float(top_contribution),
            'num_dimensions': len(importance)
        }
    
    def visualize(self, embedding, save_path=None):
        """
        Create visualization of feature importance
        
        Args:
            embedding: Single embedding vector
            save_path: Path to save figure (optional)
        
        Returns:
            matplotlib figure
        """
        
        explanation = self.explain(embedding, top_k=20)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Top features bar chart
        top_feats = explanation['top_features']
        top_scores = explanation['importance_scores']
        
        axes[0].barh(
            range(len(top_feats)),
            top_scores,
            color=plt.cm.viridis(np.linspace(0, 1, len(top_feats)))
        )
        axes[0].set_yticks(range(len(top_feats)))
        axes[0].set_yticklabels([f"Dim {i}" for i in top_feats])
        axes[0].set_xlabel('Importance Score', fontsize=12)
        axes[0].set_title('Top 20 Most Important Embedding Dimensions', fontsize=13)
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Distribution of all importances
        all_imp = explanation['all_importances']
        
        axes[1].hist(all_imp, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(
            explanation['mean_importance'],
            color='r',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {explanation["mean_importance"]:.4f}'
        )
        axes[1].axvline(
            explanation['max_importance'],
            color='g',
            linestyle='--',
            linewidth=2,
            label=f'Max: {explanation["max_importance"]:.4f}'
        )
        axes[1].set_xlabel('Importance Score', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Distribution of Feature Importances', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        return fig
    
    def get_concentration_score(self, embedding):
        """
        Calculate how concentrated the importance is
        High concentration = more confident
        
        Returns:
            float between 0 and 1
        """
        explanation = self.explain(embedding, top_k=10)
        
        # Top 10 features contribute what % of total importance?
        concentration = explanation['top_contribution']
        
        return concentration