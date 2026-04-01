

"""
main.py
-------
Full training pipeline for the mechanoreceptor descriptor ANN.
 
Dataset layout (CSV):
    - All columns except last : hand-crafted feature values
    - Last column             : binary label (1 = mechanoreceptor, 0 = not)
 
Split strategy:
    - 70% train
    - 15% validation  (threshold tuning, early stopping)
    - 15% test        (final honest evaluation — touched once)
 
Usage:
    python main.py --csv features.csv
    python main.py --csv features.csv --epochs 150 --batch 32 --lr 0.001
"""
 
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
import warnings
warnings.filterwarnings("ignore")
 
# Local import — assumes descriptor_ann_classifier.py is in same folder
from models.descriptor_ann_classifier import (
    DescriptorANNClassifier,
    build_optimizer,
    build_scheduler,
    build_loss,
    train_one_epoch,
    evaluate,
    find_optimal_threshold,
)
 
 
 

# ======================================================================
# 1. Data loading and splitting
# ======================================================================
 
def load_and_split(csv_path: str,
                   val_size: float = 0.15,
                   test_size: float = 0.15,
                   random_state: int = 42):
    """
    Load CSV, split features from label (last column),
    stratify-split into train / val / test.
 
    Returns numpy arrays: X_train, X_val, X_test, y_train, y_val, y_test
    and the list of feature names.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
 
    print(f"  Raw shape        : {df.shape}")
    print(f"  Columns          : {df.shape[1]} total")
 
    # Last column is the label
    feature_cols = df.columns[:-1].tolist()
    label_col    = df.columns[-1]
 
    print(f"  Feature columns  : {len(feature_cols)}")
    print(f"  Label column     : '{label_col}'")
 
    # Drop rows with any NaN
    before = len(df)
    df = df.dropna()
    after  = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with NaN values")
 
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)
 
    # Class distribution
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"  Positives        : {n_pos}  ({100*n_pos/len(y):.1f}%)")
    print(f"  Negatives        : {n_neg}  ({100*n_neg/len(y):.1f}%)")
 
    # --- Split 1: carve out test set first ---
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,              # preserve class ratio in every split
        random_state=random_state
    )
 
    # --- Split 2: split remaining into train and val ---
    # val fraction relative to the tmp pool
    val_fraction_of_tmp = val_size / (1.0 - test_size)
 
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_fraction_of_tmp,
        stratify=y_tmp,
        random_state=random_state
    )
 
    print(f"\n  Split summary:")
    print(f"    Train : {len(X_train):5d} samples  "
          f"({int(y_train.sum())} pos / {int((y_train==0).sum())} neg)")
    print(f"    Val   : {len(X_val):5d} samples  "
          f"({int(y_val.sum())} pos / {int((y_val==0).sum())} neg)")
    print(f"    Test  : {len(X_test):5d} samples  "
          f"({int(y_test.sum())} pos / {int((y_test==0).sum())} neg)")
 
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

def preprocess(X_train, X_val, X_test):
    """
    Fit StandardScaler on train only.
    Apply same transform to val and test — no leakage.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print("\n  StandardScaler fitted on train set only.")
    return X_train, X_val, X_test, scaler
 
 
 # ======================================================================
# 3. Build DataLoaders
# ======================================================================
 
def make_loaders(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 batch_size: int = 32):
    """
    Convert numpy arrays to TensorDatasets and DataLoaders.
    Train loader shuffles; val/test do not.
    """
    def to_tensors(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
 
    train_ds = to_tensors(X_train, y_train)
    val_ds   = to_tensors(X_val,   y_val)
    test_ds  = to_tensors(X_test,  y_test)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, drop_last=False)
 
    return train_loader, val_loader, test_loader


# 4. Training loop with early stopping
# ======================================================================
 
def train(model, train_loader, val_loader,
          optimizer, scheduler, criterion,
          device, n_epochs: int, patience: int = 20):
    """
    Train with early stopping on validation MCC.
    Saves best weights in memory (no disk required).
 
    Returns history dict and best model state.
    """
    history = {"train_loss": [], "val_loss": [],
               "val_f1": [],    "val_mcc": []}
 
    best_val_mcc   = -1.0
    best_state     = None
    patience_count = 0
 
    print(f"\n{'='*60}")
    print(f"Training  |  {n_epochs} epochs  |  patience={patience}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} | {'Train loss':>10} | {'Val loss':>9} | "
          f"{'Val F1':>7} | {'Val MCC':>7}")
    print(f"{'-'*60}")
 
    for epoch in range(1, n_epochs + 1):
 
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
 
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_mcc"].append(val_metrics["mcc"])
 
        # Print every 10 epochs or last epoch
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"  {epoch:4d} | {train_loss:10.4f} | "
                  f"{val_metrics['loss']:9.4f} | "
                  f"{val_metrics['f1']:7.4f} | "
                  f"{val_metrics['mcc']:7.4f}")
 
        # Early stopping on val MCC
        if val_metrics["mcc"] > best_val_mcc:
            best_val_mcc = val_metrics["mcc"]
            best_state   = {k: v.clone() for k, v in
                            model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
 
        if patience_count >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best val MCC = {best_val_mcc:.4f})")
            break
 
    print(f"\n  Best val MCC : {best_val_mcc:.4f}")
    return history, best_state
 
 

def final_evaluation(model, test_loader, criterion,
                      device, threshold: float,
                      feature_names: list):
    """
    Evaluate on the held-out test set using the optimal threshold
    found on the validation set.
    Print full classification report and confusion matrix.
    """
    print(f"\n{'='*60}")
    print(f"FINAL TEST EVALUATION  (threshold = {threshold:.2f})")
    print(f"{'='*60}")
 
    metrics = evaluate(model, test_loader, criterion,
                       device, threshold=threshold)
 
    probs  = torch.tensor(metrics["probs"])
    labels = torch.tensor(metrics["labels"])
    preds  = (probs >= threshold).int().numpy()
 
    print(f"\n  MCC       : {metrics['mcc']:.4f}   ← primary metric")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  PR_AUC    : {metrics['pr_auc']:.4f}")
    print(f"  ROC_AUC    : {metrics['roc_auc']:.4f}")
 
    # AUC-ROC and AUC-PR (more informative than accuracy for imbalanced)
    try:
        auc_roc = roc_auc_score(labels.numpy(), probs.numpy())
        auc_pr  = average_precision_score(labels.numpy(), probs.numpy())
        print(f"  AUC-ROC   : {auc_roc:.4f}")
        print(f"  AUC-PR    : {auc_pr:.4f} ")
    except Exception:
        pass
 
    # Confusion matrix
    cm = confusion_matrix(labels.numpy(), preds)
    print(f"\n  Confusion matrix:")
    print(f"               Pred Neg  Pred Pos")
    print(f"    True Neg :   {cm[0,0]:5d}     {cm[0,1]:5d}")
    print(f"    True Pos :   {cm[1,0]:5d}     {cm[1,1]:5d}")
 
    # Full sklearn report
    print(f"\n  Classification report:")
    print(classification_report(
        labels.numpy(), preds,
        target_names=["Non-mechanoreceptor", "Mechanoreceptor"]
    ))
 
    return metrics


 
# ======================================================================
# 6. MC Dropout uncertainty on test set
# ======================================================================
 
def uncertainty_analysis(model, test_loader, device,
                          n_samples: int = 100):
    """
    Run MC Dropout on the test set.
    Prints distribution of confidence tiers.
    """
    print(f"\n{'='*60}")
    print(f"MC DROPOUT UNCERTAINTY  ({n_samples} samples per protein)")
    print(f"{'='*60}")
 
    all_mean  = []
    all_std   = []
    all_tiers = []
    all_labels = []
 
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        result  = model.predict_with_uncertainty(
            X_batch, n_samples=n_samples
        )
        all_mean.extend(result["mean_prob"].cpu().tolist())
        all_std.extend(result["std_prob"].cpu().tolist())
        all_tiers.extend(result["confidence"])
        all_labels.extend(y_batch.tolist())
 
    # Tier distribution
    from collections import Counter
    tier_counts = Counter(all_tiers)
    total = len(all_tiers)
 
    print(f"\n  Confidence tier distribution ({total} test proteins):")
    for tier, count in tier_counts.most_common():
        pct = 100 * count / total
        print(f"    {tier:<40} : {count:4d}  ({pct:.1f}%)")
 
    # Mean uncertainty per true class
    pos_std = [s for s, l in zip(all_std, all_labels) if l == 1]
    neg_std = [s for s, l in zip(all_std, all_labels) if l == 0]
    if pos_std:
        print(f"\n  Mean uncertainty — true positives  : "
              f"{np.mean(pos_std):.4f}")
    if neg_std:
        print(f"  Mean uncertainty — true negatives  : "
              f"{np.mean(neg_std):.4f}")
    print(f"  Mean uncertainty — all             : "
          f"{np.mean(all_std):.4f}")
    print(f"\n  Proteins flagged for wet-lab:")
    n_flag = tier_counts.get("uncertain_flag_for_wetlab", 0)
    print(f"    {n_flag} / {total} proteins "
          f"({100*n_flag/max(total,1):.1f}%) need experimental validation")
 
    return all_mean, all_std, all_tiers



 
# ======================================================================
# 6. MC Dropout uncertainty on test set
# ======================================================================
 
def uncertainty_analysis(model, test_loader, device,
                          n_samples: int = 100):
    """
    Run MC Dropout on the test set.
    Prints distribution of confidence tiers.
    """
    print(f"\n{'='*60}")
    print(f"MC DROPOUT UNCERTAINTY  ({n_samples} samples per protein)")
    print(f"{'='*60}")
 
    all_mean  = []
    all_std   = []
    all_tiers = []
    all_labels = []
 
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        result  = model.predict_with_uncertainty(
            X_batch, n_samples=n_samples
        )
        all_mean.extend(result["mean_prob"].cpu().tolist())
        all_std.extend(result["std_prob"].cpu().tolist())
        all_tiers.extend(result["confidence"])
        all_labels.extend(y_batch.tolist())
 
    # Tier distribution
    from collections import Counter
    tier_counts = Counter(all_tiers)
    total = len(all_tiers)
 
    print(f"\n  Confidence tier distribution ({total} test proteins):")
    for tier, count in tier_counts.most_common():
        pct = 100 * count / total
        print(f"    {tier:<40} : {count:4d}  ({pct:.1f}%)")
 
    # Mean uncertainty per true class
    pos_std = [s for s, l in zip(all_std, all_labels) if l == 1]
    neg_std = [s for s, l in zip(all_std, all_labels) if l == 0]
    if pos_std:
        print(f"\n  Mean uncertainty — true positives  : "
              f"{np.mean(pos_std):.4f}")
    if neg_std:
        print(f"  Mean uncertainty — true negatives  : "
              f"{np.mean(neg_std):.4f}")
    print(f"  Mean uncertainty — all             : "
          f"{np.mean(all_std):.4f}")
    print(f"\n  Proteins flagged for wet-lab:")
    n_flag = tier_counts.get("uncertain_flag_for_wetlab", 0)
    print(f"    {n_flag} / {total} proteins "
          f"({100*n_flag/max(total,1):.1f}%) need experimental validation")
 
    return all_mean, all_std, all_tiers

def save_artefacts(model, scaler, threshold,
                   feature_names, history, out_dir: str = "outputs"):
    """
    Save model weights, scaler, threshold, and feature names
    so the model can be reloaded for inference.
    """
    import pickle, json
    os.makedirs(out_dir, exist_ok=True)
 
    # Model weights
    torch.save(model.state_dict(),
               os.path.join(out_dir, "model_weights.pt"))
 
    # Scaler
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
 
    # Threshold and feature names
    meta = {
        "threshold":     threshold,
        "feature_names": feature_names,
        "input_dim":     len(feature_names),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
 
    # Training history as CSV
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, "training_history.csv"), index=False
    )
 
    print(f"\n  Artefacts saved to '{out_dir}/':")
    print(f"    model_weights.pt")
    print(f"    scaler.pkl")
    print(f"    meta.json  (threshold={threshold:.3f}, "
          f"input_dim={len(feature_names)})")
    print(f"    training_history.csv")
 
# ======================================================================
# 8. Main
# ======================================================================
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Train mechanoreceptor ANN on hand-crafted descriptors."
    )
    p.add_argument("--csv",       required=True,
                   help="Path to features CSV. Last column = label.")
    p.add_argument("--epochs",    type=int,   default=150)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--dropout",   type=float, default=0.3)
    p.add_argument("--patience",  type=int,   default=20,
                   help="Early stopping patience on val MCC.")
    p.add_argument("--hidden",    type=int,   nargs="+",
                   default=[64, 32],
                   help="Hidden layer widths e.g. --hidden 64 32")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--out",       default="outputs",
                   help="Directory to save model artefacts.")
    p.add_argument("--mc_samples", type=int,  default=100,
                   help="MC Dropout samples for uncertainty estimation.")
    return p.parse_args()
 
 
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
 
    print(f"\n{'='*60}")
    print(f"Mechanoreceptor ANN  |  device={device}")
    print(f"{'='*60}\n")
 
    # ── 1. Load & split ──────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names) = load_and_split(args.csv, random_state=args.seed)
 
    input_dim = X_train.shape[1]
 
    # ── 2. Preprocess ────────────────────────────────────────────────
    X_train, X_val, X_test, scaler = preprocess(X_train, X_val, X_test)
 
    # ── 3. DataLoaders ───────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        batch_size=args.batch
    )
 
    # ── 4. Model ─────────────────────────────────────────────────────
    print(f"\n")
    model = DescriptorANNClassifier(
        input_dim   = input_dim,
        hidden_dims = tuple(args.hidden),
        dropout     = args.dropout,
    ).to(device)
 
    optimizer = build_optimizer(model, lr=args.lr)
    scheduler = build_scheduler(optimizer, n_epochs=args.epochs)
    criterion = build_loss(
        torch.tensor(y_train), device=device
    )
 
    # ── 5. Train ─────────────────────────────────────────────────────
    history, best_state = train(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion,
        device, n_epochs=args.epochs, patience=args.patience
    )
 
    # Restore best weights
    model.load_state_dict(best_state)
    print("  Best weights restored.")
 
    # ── 6. Optimal threshold on val set ──────────────────────────────
    print(f"\n  Finding optimal decision threshold on val set...")
    val_metrics = evaluate(model, val_loader, criterion, device)
    threshold   = find_optimal_threshold(
        torch.tensor(val_metrics["probs"]),
        torch.tensor(val_metrics["labels"])
    )
 
    # ── 7. Final test evaluation ──────────────────────────────────────
    final_evaluation(
        model, test_loader, criterion, device, threshold, feature_names
    )
 
    # ── 8. MC Dropout uncertainty ─────────────────────────────────────
    uncertainty_analysis(
        model, test_loader, device,
        n_samples=args.mc_samples
    )
 
    # ── 9. Save artefacts ─────────────────────────────────────────────
    save_artefacts(
        model, scaler, threshold,
        feature_names, history, out_dir=args.out
    )
 
    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"{'='*60}\n")
 
 
if __name__ == "__main__":
    main()