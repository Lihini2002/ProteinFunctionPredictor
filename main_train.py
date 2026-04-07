import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score
)

from models.descriptor_ann_classifier import (
    DescriptorANNClassifier,
    build_optimizer,
    build_loss,
    train_one_epoch,
    evaluate,
    cross_validate_model
)


# ==============================================================
# Data loading and splitting
# ==============================================================

def load_and_split(csv_path: str, random_state: int = 42):
    df = pd.read_csv(csv_path)

    feature_cols = df.columns[1:-1].tolist()
    label_col = df.columns[-1]

    df = df.dropna()

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=0.176,  # ~15% of original
        stratify=y_tmp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ==============================================================
# Final evaluation — runs inference directly (no evaluate())
# since evaluate() only returns MCC scalar
# ==============================================================

@torch.no_grad()
def final_evaluation(model, X_test, y_test, device, threshold=0.5):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    logits = model(X_t).squeeze(1)
    probs  = torch.sigmoid(logits).cpu().numpy()
    preds  = (probs >= threshold).astype(int)
    labels = y_test.astype(int)

    mcc  = matthews_corrcoef(labels, preds)
    f1   = f1_score(labels, preds)
    roc  = roc_auc_score(labels, probs)
    pr   = average_precision_score(labels, probs)

    print("\nFINAL TEST RESULTS")
    print(f"MCC    : {mcc:.4f}")
    print(f"F1     : {f1:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print(f"PR  AUC: {pr:.4f}")
    print("\nConfusion Matrix")
    print(confusion_matrix(labels, preds))
    print("\nClassification Report")
    print(classification_report(labels, preds))

    return mcc, f1, roc, pr


# ==============================================================
# Save artefacts
# ==============================================================

def save_artefacts(model, scaler, feature_names, input_dim,
                   threshold=0.5, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{out_dir}/model_weights.pt")

    import pickle, json
    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump({
            "feature_names": feature_names,
            "input_dim"    : input_dim,
            "threshold"    : threshold,
        }, f, indent=2)

    print(f"\nArtefacts saved to '{out_dir}/'")


# ==============================================================
# Main
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    required=True)
    p.add_argument("--epochs", type=int,   default=60)
    p.add_argument("--batch",  type=int,   default=32)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 16])
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------------
    # 1) Load and split — test set stays completely untouched
    # ----------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        load_and_split(args.csv)

    # Merge train + val for CV
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.hstack([y_train, y_val])

    # Scale once — fit on CV portion only, apply to test
    scaler = StandardScaler()
    X_cv   = scaler.fit_transform(X_cv)
    X_test = scaler.transform(X_test)

    # ----------------------------------------------------------
    # 2) Hyperparameter search via cross-validation
    # ----------------------------------------------------------
    param_grid = [
        {"hidden_dims": (32, 16), "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3},
        {"hidden_dims": (64, 32), "dropout": 0.4, "lr": 8e-4, "weight_decay": 5e-3},
        {"hidden_dims": (48, 24), "dropout": 0.35, "lr": 5e-4, "weight_decay": 1e-2},
    ]

    best_params = cross_validate_model(X_cv, y_cv, device, param_grid)
    print("\nBest params from CV:", best_params)

    # ----------------------------------------------------------
    # 3) Train final model on full CV data with best params
    # ----------------------------------------------------------
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_cv,  dtype=torch.float32),
            torch.tensor(y_cv,  dtype=torch.float32),
        ),
        batch_size=args.batch,
        shuffle=True,
    )

    model = DescriptorANNClassifier(
        input_dim   = X_cv.shape[1],
        hidden_dims = best_params["hidden_dims"],
        dropout     = best_params["dropout"],
    ).to(device)

    optimizer = build_optimizer(
        model,
        lr           = best_params["lr"],
        weight_decay = best_params["weight_decay"],
    )

    criterion = build_loss(torch.tensor(y_cv), device=device)

    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    # ----------------------------------------------------------
    # 4) Evaluate on held-out test set (one shot)
    # ----------------------------------------------------------
    final_evaluation(model, X_test, y_test, device, threshold=0.5)

    # ----------------------------------------------------------
    # 5) Save artefacts
    # ----------------------------------------------------------
    save_artefacts(
        model        = model,
        scaler       = scaler,
        feature_names= feature_names,
        input_dim    = X_cv.shape[1],
        threshold    = 0.5,
        out_dir      = "outputs",
    )


if __name__ == "__main__":
    main()