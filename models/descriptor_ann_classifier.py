# models/descriptor_ann_classifier.py

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import numpy as np


# ==============================================================
# Model
# ==============================================================

class DescriptorANNClassifier(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dims=(32, 16),
                 dropout=0.3):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==============================================================
# Training utilities
# ==============================================================

def build_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


def build_loss(y_train, device):
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                              dtype=torch.float32).to(device)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    all_probs, all_labels = [], []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float().unsqueeze(1)

        logits = model(X)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    mcc = ((tp * tn) - (fp * fn)) / torch.sqrt(
        (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-8
    )

    return mcc.item()


# ==============================================================
# Cross-Validation + Hyperparameter Search
# ==============================================================

def cross_validate_model(X, y, device, param_grid,
                         n_splits=5, epochs=50, batch_size=32):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score = -1
    best_params = None

    for params in param_grid:
        fold_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32),
                ),
                batch_size=batch_size,
                shuffle=True
            )

            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.float32),
                ),
                batch_size=batch_size,
                shuffle=False
            )

            model = DescriptorANNClassifier(
                input_dim=X.shape[1],
                hidden_dims=params["hidden_dims"],
                dropout=params["dropout"],
            ).to(device)

            optimizer = build_optimizer(
                model,
                lr=params["lr"],
                weight_decay=params["weight_decay"]
            )

            criterion = build_loss(
                torch.tensor(y_train), device
            )

            for _ in range(epochs):
                train_one_epoch(model, train_loader,
                                optimizer, criterion, device)

            mcc = evaluate(model, val_loader,
                           criterion, device)

            fold_scores.append(mcc)

        mean_mcc = np.mean(fold_scores)
        print(f"{params} → Mean MCC: {mean_mcc:.4f}")

        if mean_mcc > best_score:
            best_score = mean_mcc
            best_params = params

    print("\nBest Hyperparameters:")
    print(best_params, "MCC:", best_score)

    return best_params