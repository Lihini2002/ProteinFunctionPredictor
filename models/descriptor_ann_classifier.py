# models/descriptor_ann_classifier.py

"""
ANN classifier for handcrafted protein descriptors
(AAC, DPC, physicochemical features, motifs, loops, etc.)
"""

import torch
import torch.nn as nn


class DescriptorANNClassifier(nn.Module):
    """
    Input: Descriptor feature vector from CSV
    Output: Binary prediction (mechanoreceptor / not)

    Designed specifically for:
    - Sparse handcrafted features
    - Low information density
    - High correlation between features
    """

    def __init__(self ,  
        input_dim: int,
        hidden_dims: tuple = (64, 32),
        dropout: float = 0.5,):
        """
        Args:
            input_dim: number of descriptor features (e.g. 432+)
            dropout: regularization
        """
        super(DescriptorANNClassifier, self).__init__()

        print(f"Initializing Descriptor ANN...")
        print(f"Input features: {input_dim}")

        self.network = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 1)  # NO SIGMOID
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print("✓ Descriptor ANN ready")

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        """
        return self.network(x)




# Monte Carlo uncertainity output. 
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> dict:
        """
        Monte Carlo Dropout inference.
        Keeps dropout ACTIVE across n_samples forward passes.
        Variance across passes = epistemic uncertainty.
 
        Parameters
        ----------
        x         : (batch_size, input_dim) — single protein or batch
        n_samples : number of stochastic forward passes
 
        Returns
        -------
        dict with keys:
          mean_prob    : (batch_size,) mean predicted probability
          std_prob     : (batch_size,) std across samples = uncertainty
          raw_samples  : (n_samples, batch_size) all probability samples
          confidence   : str tier per sample — 'high' / 'uncertain' / 'low'
        """
        # Force train mode so dropout is active — do NOT call model.eval()
        self.train()
 
        probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                logit = self.forward(x)               # (B, 1)
                prob  = torch.sigmoid(logit).squeeze(-1)  # (B,)
                probs.append(prob)
 
        probs    = torch.stack(probs)                 # (n_samples, B)
        mean_p   = probs.mean(dim=0)                  # (B,)
        std_p    = probs.std(dim=0)                   # (B,)
 
        # Confidence tiering
        # High   : model consistently says positive/negative
        # Low    : model consistently says negative
        # Uncertain: model is unsure — flag for wet-lab validation
        tiers = []
        for mp, sp in zip(mean_p.tolist(), std_p.tolist()):
            if mp >= 0.6 and sp < 0.15:
                tiers.append("high_confidence_positive")
            elif mp < 0.4 and sp < 0.15:
                tiers.append("high_confidence_negative")
            else:
                tiers.append("uncertain_flag_for_wetlab")
 
        return {
            "mean_prob":   mean_p,
            "std_prob":    std_p,
            "raw_samples": probs,
            "confidence":  tiers,
        }
    


# ======================================================================
# Training utilities
# ======================================================================
 
def build_optimizer(model: DescriptorANNClassifier, lr: float = 8e-4,
                    weight_decay: float = 5e-3):
    
    """
    AdamW with mild weight decay.
    weight_decay acts as L2 regularisation on weights (not biases).
    """
    return torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
 
 
def build_scheduler(optimizer, n_epochs: int = 100):
    """
    Cosine annealing — smoothly reduces LR to near-zero over training.
    Better than StepLR for small datasets where you want to fully
    explore the loss landscape before cooling.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )
 
 
def build_loss(y_train: torch.Tensor = None, device: str = 'cpu'):
    """
    BCEWithLogitsLoss with automatic class-weight balancing.
    Corrects for mechanoreceptor/non-mechanoreceptor imbalance.
 
    Parameters
    ----------
    y_train : 1-D tensor of training labels (0/1) for computing weight.
              If None, uses weight=1 (balanced dataset assumed).
    """
    if y_train is not None:
        n_pos  = y_train.sum().item()
        n_neg  = len(y_train) - n_pos
        # pos_weight = n_neg / n_pos
        # means each positive counts as (n_neg/n_pos) negatives
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                                  dtype=torch.float32).to(device)
        print(f"  Class weight: pos_weight = {pos_weight.item():.2f} "
              f"({int(n_neg)} neg / {int(n_pos)} pos)")
    else:
        pos_weight = None
 
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
 
 
# ======================================================================
# Full training loop
# ======================================================================
 
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)
 
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
 
        # Gradient clipping — prevents exploding gradients on small batches
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
        optimizer.step()
        total_loss += loss.item()
 
    return total_loss / len(loader)
 
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Ensure outputs folder exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@torch.no_grad()
def evaluate(model, loader, criterion, device,
             threshold: float = 0.5,
             plot_curves: bool = True,
             prefix: str = "model") -> dict:
    """
    Evaluate model and optionally plot ROC and PR-AUC curves.
    """
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.
 
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)
 
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item()
 
        all_logits.append(logits.squeeze(1).cpu())
        all_labels.append(y_batch.squeeze(1).cpu())
 
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs  = torch.sigmoid(logits)
    preds  = (probs >= threshold).float()
 
    tp = ((preds == 1) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
 
    precision  = tp / (tp + fp + 1e-8)
    recall     = tp / (tp + fn + 1e-8)
    f1         = 2 * precision * recall / (precision + recall + 1e-8)
 
    # Matthews Correlation Coefficient
    mcc_num    = (tp * tn) - (fp * fn)
    mcc_den    = torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-8)
    mcc        = mcc_num / mcc_den

    probs_np = probs.numpy()
    labels_np = labels.numpy()

    # --- Plot ROC and PR curves ---
    if plot_curves:
        # ROC
        fpr, tpr, _ = roc_curve(labels_np, probs_np)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        roc_path = os.path.join(OUTPUT_DIR, f"{prefix}_roc_curve.png")
        plt.savefig(roc_path)
        plt.close()

        # Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probs_np)
        pr_auc = auc(recall_curve, precision_curve)
        plt.figure()
        plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        pr_path = os.path.join(OUTPUT_DIR, f"{prefix}_pr_curve.png")
        plt.savefig(pr_path)
        plt.close()

        print(f"ROC curve saved to {roc_path}, PR curve saved to {pr_path}")
 
    return {
        "loss":      total_loss / len(loader),
        "precision": precision.item(),
        "recall":    recall.item(),
        "f1":        f1.item(),
        "mcc":       mcc.item(),
        "probs":     probs_np,
        "labels":    labels_np,
        "roc_auc":   roc_auc if plot_curves else None,
        "pr_auc":    pr_auc if plot_curves else None
    }


# def evaluate(model, loader, criterion, device,
#              threshold: float = 0.5) -> dict:
#     """
#     Evaluate model. Uses model.eval() so dropout is OFF —
#     deterministic predictions for reporting metrics.
#     """
#     model.eval()
#     all_logits = []
#     all_labels = []
#     total_loss = 0.
 
#     for X_batch, y_batch in loader:
#         X_batch = X_batch.to(device)
#         y_batch = y_batch.to(device).float().unsqueeze(1)
 
#         logits = model(X_batch)
#         loss   = criterion(logits, y_batch)
#         total_loss += loss.item()
 
#         all_logits.append(logits.squeeze(1).cpu())
#         all_labels.append(y_batch.squeeze(1).cpu())
 
#     logits = torch.cat(all_logits)
#     labels = torch.cat(all_labels)
#     probs  = torch.sigmoid(logits)
#     preds  = (probs >= threshold).float()
 
#     tp = ((preds == 1) & (labels == 1)).sum().float()
#     tn = ((preds == 0) & (labels == 0)).sum().float()
#     fp = ((preds == 1) & (labels == 0)).sum().float()
#     fn = ((preds == 0) & (labels == 1)).sum().float()
 
#     precision  = tp / (tp + fp + 1e-8)
#     recall     = tp / (tp + fn + 1e-8)
#     f1         = 2 * precision * recall / (precision + recall + 1e-8)
 
#     # Matthews Correlation Coefficient — gold standard for binary
#     # classification with imbalanced classes
#     mcc_num    = (tp * tn) - (fp * fn)
#     mcc_den    = torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-8)
#     mcc        = mcc_num / mcc_den
 
#     return {
#         "loss":      total_loss / len(loader),
#         "precision": precision.item(),
#         "recall":    recall.item(),
#         "f1":        f1.item(),
#         "mcc":       mcc.item(),          # report this as primary metric
#         "probs":     probs.numpy(),
#         "labels":    labels.numpy(),
#     }
 
 
def find_optimal_threshold(probs: torch.Tensor,
                            labels: torch.Tensor) -> float:
    """
    Sweep thresholds on validation set and return the one
    that maximises MCC. Default 0.5 is rarely optimal with
    imbalanced mechanoreceptor datasets.
    """
    best_mcc = -1.
    best_thr = 0.5
 
    for thr in torch.arange(0.1, 0.9, 0.01):
        preds = (probs >= thr).float()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        num = (tp * tn) - (fp * fn)
        den = torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-8)
        mcc = (num / den).item()
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr.item()
 
    print(f"  Optimal threshold: {best_thr:.2f}  (val MCC = {best_mcc:.4f})")
    return best_thr
 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Make sure outputs folder exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add this function to plot and save ROC & PR curves
def plot_and_save_metrics(probs, labels, prefix="model"):
    """
    probs  : numpy array of predicted probabilities
    labels : numpy array of true labels
    prefix : str prefix for filenames
    """
    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_path = os.path.join(OUTPUT_DIR, f"{prefix}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to {roc_path}")
    
    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure()
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    pr_path = os.path.join(OUTPUT_DIR, f"{prefix}_pr_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"PR-AUC curve saved to {pr_path}")
 
# ======================================================================
# Sanity check
# ======================================================================
 
if __name__ == "__main__":
    import numpy as np
 
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_DIM = 75     # your hand-crafted feature count
    BATCH     = 32
    N_EPOCHS  = 5      # just for smoke test
 
    # Fake data — 200 samples, 75 features, binary labels
    X_fake = torch.randn(200, INPUT_DIM)
    y_fake = torch.randint(0, 2, (200,))
 
    dataset = torch.utils.data.TensorDataset(X_fake, y_fake)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH,
                                          shuffle=True)
 
    model     = DescriptorANNClassifier(INPUT_DIM).to(device)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, N_EPOCHS)
    criterion = build_loss(y_fake, device)
 
    print(f"\nRunning {N_EPOCHS} smoke-test epochs on {device}...")
    for epoch in range(1, N_EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        metrics = evaluate(model, loader, criterion, device)
        print(f"  Epoch {epoch:2d} | loss {loss:.4f} | "
              f"F1 {metrics['f1']:.4f} | MCC {metrics['mcc']:.4f}")
 
    # MC Dropout uncertainty on 5 proteins
    print("\nMC Dropout uncertainty (5 test proteins):")
    x_test = torch.randn(5, INPUT_DIM).to(device)
    result  = model.predict_with_uncertainty(x_test, n_samples=100)
    for i, (mp, sp, tier) in enumerate(
        zip(result["mean_prob"], result["std_prob"], result["confidence"])
    ):
        print(f"  Protein {i+1}: prob={mp:.3f}  unc={sp:.3f}  → {tier}")
 
    print("\nSmoke test passed.")

 
 