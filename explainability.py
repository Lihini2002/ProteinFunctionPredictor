import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import seaborn as sns
 
from models.descriptor_ann_classifier import DescriptorANNClassifier


# ======================================================================
# Feature group mapping — matches actual descriptor code output
# ======================================================================

FEATURE_GROUPS = {
    "AAC_"                      : "AA composition",
    "DPC_"                      : "Dipeptide composition",
    
    "hydro_"                    : "Hydrophobicity",
    "charge_"                   : "Charge",
    "gly_fraction"              : "Gly/Pro",
    "pro_fraction"              : "Gly/Pro",
    "gp_fraction"               : "Gly/Pro",
    "aromatic_fraction"         : "Aromatic",
    "hydrophobic_stretch_count" : "Hydrophobic stretches",
    "low_complexity_frac"       : "Low complexity",
    "pI"                        : "Physicochemical",
}

FEATURE_BIOLOGY = {
    # --- AA composition (20 features) ---
    "AAC_G"  : "Glycine fraction — helix flexibility, bending in TM segments",
    "AAC_P"  : "Proline fraction — helix-breaking kinks, mechanogating",
    "AAC_L"  : "Leucine fraction — hydrophobic TM core packing",
    "AAC_W"  : "Tryptophan fraction — aromatic membrane anchor at interface",
    "AAC_F"  : "Phenylalanine fraction — aromatic belt, lipid-water interface",
    "AAC_Y"  : "Tyrosine fraction — aromatic belt, gating residues",
    "AAC_K"  : "Lysine fraction — positive charge, intracellular loop signal",
    "AAC_R"  : "Arginine fraction — positive-inside rule, G-protein coupling",
    "AAC_D"  : "Aspartate fraction — negative charge, calcium binding sites",
    "AAC_E"  : "Glutamate fraction — negative charge, pH sensing",
    "AAC_C"  : "Cysteine fraction — disulfide bonds, redox-sensitive gating",
    "AAC_H"  : "Histidine fraction — pH-sensitive gating, zinc coordination",
    "AAC_V"  : "Valine fraction — hydrophobic TM packing, beta-sheet",
    "AAC_I"  : "Isoleucine fraction — hydrophobic TM core",
    "AAC_A"  : "Alanine fraction — helix-forming, small hydrophobic",
    "AAC_S"  : "Serine fraction — phosphorylation sites, polar loops",
    "AAC_T"  : "Threonine fraction — phosphorylation, polar loops",
    "AAC_N"  : "Asparagine fraction — N-glycosylation sites in ECL",
    "AAC_Q"  : "Glutamine fraction — polar, hydrogen bonding in loops",
    "AAC_M"  : "Methionine fraction — hydrophobic core, oxidation sensor",

    # --- Hydrophobicity (2 features) ---
    "hydro_mean" : "Mean KD hydrophobicity — overall membrane affinity of protein",
    "hydro_std"  : "Std of KD hydrophobicity — variability between TM and loop regions",

    # --- Charge (2 features) ---
    "charge_net"     : "Net charge per residue (K+R+H - D-E) / length",
    "charge_density" : "Total charged residue density — electrostatic coupling to membrane",

    # --- Gly/Pro (3 features) ---
    "gly_fraction" : "Glycine fraction — helix breaker, creates flexible bends in TM",
    "pro_fraction" : "Proline fraction — introduces kinks critical for mechanogating",
    "gp_fraction"  : "Combined G+P fraction — overall helix-breaking capacity",

    # --- Aromatic (1 feature) ---
    "aromatic_fraction" : "F+W+Y fraction — aromatic belt anchoring TM helices at membrane interface",

    # --- Hydrophobic stretches (1 feature) ---
    "hydrophobic_stretch_count" : "Count of windows (18aa) with mean KD>1.6 — TM helix proxy",

    # --- Low complexity (1 feature) ---
    "low_complexity_frac" : "Fraction of low-entropy windows — disordered/repeat regions",

    # --- Physicochemical (1 feature) ---
    "pI" : "Isoelectric point — pH at net zero charge, membrane targeting signal",

    # --- Dipeptide composition: only highlight biologically notable ones ---
    "DPC_GP" : "Gly-Pro dipeptide — sharp helix-breaking turn motif",
    "DPC_PG" : "Pro-Gly dipeptide — reverse helix-breaking turn",
    "DPC_LL" : "Leu-Leu dipeptide — hydrophobic TM core packing",
    "DPC_LV" : "Leu-Val dipeptide — hydrophobic TM core",
    "DPC_KR" : "Lys-Arg dipeptide — NLS-like basic cluster, nuclear import",
    "DPC_RR" : "Arg-Arg dipeptide — strong NLS signal, basic cluster",
    "DPC_KK" : "Lys-Lys dipeptide — NLS-like basic cluster",
    "DPC_DE" : "Asp-Glu dipeptide — acidic cluster, ER retention signal",
    "DPC_WF" : "Trp-Phe dipeptide — dual aromatic anchoring at TM boundary",
    "DPC_FW" : "Phe-Trp dipeptide — dual aromatic anchoring",
    "DPC_CG" : "Cys-Gly dipeptide — flexible disulfide loop",
    "DPC_GG" : "Gly-Gly dipeptide — maximum backbone flexibility",
    "DPC_PP" : "Pro-Pro dipeptide — rigid kink, structural constraint",
}


def get_feature_group(feat_name: str) -> str:
    """Map a feature name to its biological group label."""
    for prefix, group in FEATURE_GROUPS.items():
        if feat_name.startswith(prefix) or feat_name == prefix:
            return group
    return "Other"


# ======================================================================
# 1. Load artefacts
# ======================================================================
 
def load_artefacts(model_dir: str):
    """
    Load model weights, scaler, threshold, and feature names
    saved by main.py.
    """
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
 
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
 
    model = DescriptorANNClassifier(
        input_dim   = meta["input_dim"],
        hidden_dims = (64, 32),      # must match what you trained with
        dropout     = 0.3,
    )
    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "model_weights.pt"),
            map_location="cpu"
        )
    )
    model.eval()
 
    print(f"Model loaded from '{model_dir}'")
    print(f"  Input dim   : {meta['input_dim']}")
    print(f"  Threshold   : {meta['threshold']:.3f}")
    print(f"  Features    : {len(meta['feature_names'])}")
 
    return model, scaler, meta["threshold"], meta["feature_names"]
 


def load_and_predict(csv_path: str, model, scaler,
                     threshold: float, feature_names: list,
                     device: str = "cpu"):
    """
    Load features CSV, run model inference, return predictions
    alongside raw feature matrix.
 
    Returns
    -------
    df_raw     : original dataframe (features + label)
    X_scaled   : scaled feature matrix (numpy, shape N x D)
    X_tensor   : torch tensor version of X_scaled
    probs      : predicted probabilities (numpy, shape N)
    preds      : binary predictions at threshold (numpy, shape N)
    labels     : true labels if available, else None
    """
    df = pd.read_csv(csv_path).dropna()
 
    # Separate features and label
    # Last column is label (same as main.py convention)
    has_label   = True
    label_col   = df.columns[-1]
    feat_cols   = df.columns[:-1].tolist()
 
    # Guard: if columns don't match saved feature names use saved order
    if set(feat_cols) != set(feature_names):
        print("  WARNING: CSV columns differ from saved feature names.")
        print("  Using saved feature_names order — missing cols set to 0.")
        X_raw = np.zeros((len(df), len(feature_names)), dtype=np.float32)
        for i, fn in enumerate(feature_names):
            if fn in df.columns:
                X_raw[:, i] = df[fn].values.astype(np.float32)
    else:
        X_raw = df[feature_names].values.astype(np.float32)
 
    labels = df[label_col].values.astype(np.float32) if has_label else None
 
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    X_tensor = torch.tensor(X_scaled)
 
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        probs  = torch.sigmoid(logits).numpy()
 
    preds = (probs >= threshold).astype(int)
 
    # Print summary
    print(f"\nPrediction summary ({len(probs)} proteins):")
    print(f"  Predicted mechanoreceptors     : {preds.sum()}")
    print(f"  Predicted non-mechanoreceptors : {(preds==0).sum()}")
    if labels is not None:
        correct = (preds == labels.astype(int)).sum()
        print(f"  Correct predictions            : "
              f"{correct} / {len(preds)}")
 
    return df, X_scaled, X_tensor, probs, preds, labels
 
 
# ======================================================================
# 3. SHAP explainer setup
# ======================================================================
def build_shap_explainer(model, X_train_scaled: np.ndarray,
                          n_background: int = 50):
    """
    Build a SHAP KernelExplainer.

    KernelExplainer treats the model as a black box — it only needs
    a predict function f(X) -> array of probabilities. No backprop,
    no PyTorch internals. Works with any model.

    Background dataset: use kmeans-summarised background (shap.kmeans)
    rather than random samples. kmeans(X, k) compresses X into k
    representative centroids weighted by cluster size — gives more
    accurate SHAP values than random sampling for the same compute cost.

    n_background: 50 is a good default. More = more accurate but slower.
    With 431 features and 1300 proteins, 50 background + nsamples=512
    takes ~2-5 min on CPU for the full dataset.
    """

    # Wrap model: numpy array in → numpy probabilities out
    # KernelExplainer cannot call PyTorch directly
    def model_predict(X_numpy: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X_numpy.astype(np.float32))
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor).squeeze(1)
            probs  = torch.sigmoid(logits).numpy()
        return probs   # shape (N,) — probability of mechanoreceptor

    # Summarise background with kmeans — faster and more representative
    # than a random subset
    background = shap.kmeans(X_train_scaled, n_background)

    explainer = shap.KernelExplainer(
        model_predict,
        background,
        link="identity"    # SHAP values in probability space (0-1)
                           # not log-odds space — easier to interpret
    )

    print(f"\nSHAP KernelExplainer built.")
    print(f"  Type            : KernelExplainer (model-agnostic)")
    print(f"  Background      : kmeans({n_background} centroids from train set)")
    print(f"  Link function   : identity (SHAP values in probability space)")
    print(f"  Note: slower than DeepExplainer — batch in compute_shap_values")

    return explainer
 
# ======================================================================
# 4. Compute SHAP values
# ======================================================================
 
def compute_shap_values(explainer, X_scaled: np.ndarray,
                         batch_size: int = 32,
                         nsamples: int = 512) -> np.ndarray:
    """
    Compute SHAP values in batches using KernelExplainer.

    Parameters
    ----------
    X_scaled  : numpy array (N, D) — already scaled, NOT a tensor
    batch_size: proteins per batch — keep small (16-32) for memory
    nsamples  : SHAP perturbation samples per protein per feature.
                Higher = more accurate, slower.
                512 is a good balance. Use 256 for a quick run,
                1000+ for final paper-quality values.

    Note: KernelExplainer returns shape (N, D) for a single-output
    model with link='identity'. No list unwrapping needed.
    """
    print(f"\nComputing SHAP values for {len(X_scaled)} proteins...")
    print(f"  nsamples per call : {nsamples}")
    print(f"  batch size        : {batch_size}")
    estimated_mins = len(X_scaled) * nsamples / 50000
    print(f"  Estimated time    : ~{estimated_mins:.1f} min on CPU")

    all_shap = []

    for i in range(0, len(X_scaled), batch_size):
        batch = X_scaled[i:i + batch_size]
        sv    = explainer.shap_values(batch, nsamples=nsamples, silent=True)
        all_shap.append(sv)
        done = min(i + batch_size, len(X_scaled))
        print(f"  [{done:4d}/{len(X_scaled)}] batches done", end="\r")

    print()   # newline after progress
    shap_values = np.vstack(all_shap)    # (N, D)
    print(f"  SHAP values shape : {shap_values.shape}")
    return shap_values
# ======================================================================
# 5. Global feature importance plot
# ======================================================================
 
def plot_global_importance(shap_values: np.ndarray,
                            feature_names: list,
                            out_dir: str,
                            top_n: int = 25):
    """
    Bar plot of mean absolute SHAP value per feature.
    Groups features by biological category.
    This answers: "What does the model rely on overall?"
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)    # (D,)
    importance_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": mean_abs_shap,
        "group"     : [get_feature_group(f) for f in feature_names],
    }).sort_values("importance", ascending=False)
 
    top_df = importance_df.head(top_n)
 
    # Colour by biological group
    groups        = top_df["group"].unique().tolist()
    palette       = sns.color_palette("tab10", len(groups))
    group_colours = dict(zip(groups, palette))
    bar_colours   = [group_colours[g] for g in top_df["group"]]
 
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        top_df["feature"][::-1],
        top_df["importance"][::-1],
        color=bar_colours[::-1],
        edgecolor="none", height=0.7
    )
 
    ax.set_xlabel("Mean |SHAP value|  (impact on mechanoreceptor prediction)",
                  fontsize=11)
    ax.set_title(f"Top {top_n} most important features\n"
                 f"(global — averaged over all proteins)",
                 fontsize=12, fontweight="bold")
 
    # Legend for groups
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=group_colours[g], label=g) for g in groups
        if g in top_df["group"].values
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=9, title="Feature group")
 
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "shap_global_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")
 
    # Print top 10 with biology
    print(f"\n  Top 10 most important features (biological interpretation):")
    print(f"  {'Feature':<35} {'Importance':>10}  Biological meaning")
    print(f"  {'-'*80}")
    for _, row in importance_df.head(10).iterrows():
        bio = FEATURE_BIOLOGY.get(row["feature"], "—")
        print(f"  {row['feature']:<35} {row['importance']:>10.4f}  {bio}")
 
    return importance_df
 
 
# ======================================================================
# 6. SHAP beeswarm (summary) plot
# ======================================================================
 
def plot_beeswarm(shap_values: np.ndarray,
                  X_scaled: np.ndarray,
                  feature_names: list,
                  out_dir: str,
                  top_n: int = 20):
    """
    SHAP beeswarm plot — shows both direction and magnitude.
    Each dot = one protein. Colour = feature value (red=high, blue=low).
    Position on x-axis = SHAP value (right = pushes toward mechanoreceptor).
 
    This answers: "High values of this feature push toward which class?"
    """
    # Select top_n features by mean absolute SHAP
    mean_abs   = np.abs(shap_values).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]
    top_names  = [feature_names[i] for i in top_idx]
    top_shap   = shap_values[:, top_idx]
    top_X      = X_scaled[:, top_idx]
 
    # Use shap's built-in summary_plot
    shap.summary_plot(
        top_shap, top_X,
        feature_names=top_names,
        show=False,
        plot_size=(10, 8),
        color_bar_label="Feature value (scaled)"
    )
    out_path = os.path.join(out_dir, "shap_beeswarm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
 
 
# ======================================================================
# 7. Per-protein waterfall explanation
# ======================================================================
 
def explain_single_protein(shap_values: np.ndarray,
                             X_scaled: np.ndarray,
                             feature_names: list,
                             protein_idx: int,
                             prob: float,
                             pred: int,
                             true_label: float,
                             out_dir: str,
                             top_n: int = 15):
    """
    Waterfall plot for one protein.
    Shows which features pushed the prediction up or down
    from the baseline (average prediction across training set).
 
    This answers: "Why was THIS specific protein classified as
    mechanoreceptor / non-mechanoreceptor?"
    """
    sv    = shap_values[protein_idx]          # (D,) SHAP values
    x_val = X_scaled[protein_idx]             # (D,) feature values
 
    # Sort by absolute SHAP value for this protein
    order      = np.argsort(np.abs(sv))[::-1][:top_n]
    top_sv     = sv[order]
    top_fnames = [feature_names[i] for i in order]
    top_xvals  = x_val[order]
 
    # Waterfall: bars right = pushed toward mechanoreceptor
    colours = ["#E8593C" if v > 0 else "#378ADD" for v in top_sv]
 
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(len(top_sv)),
        top_sv[::-1],
        color=colours[::-1],
        edgecolor="none", height=0.65
    )
    ax.set_yticks(range(len(top_sv)))
    ax.set_yticklabels(
        [f"{top_fnames[::-1][i]}  ({top_xvals[::-1][i]:.2f})"
         for i in range(len(top_sv))],
        fontsize=9
    )
 
    pred_label = "Mechanoreceptor" if pred == 1 else "Non-mechanoreceptor"
    true_str   = ("Mechanoreceptor" if true_label == 1
                  else "Non-mechanoreceptor") if true_label is not None else "?"
    correct    = ("✓ Correct" if (pred == int(true_label))
                  else "✗ Incorrect") if true_label is not None else ""
 
    ax.set_xlabel("SHAP value  (red = toward mechanoreceptor, "
                  "blue = away from mechanoreceptor)",
                  fontsize=10)
    ax.set_title(
        f"Protein {protein_idx}  |  Pred: {pred_label}  "
        f"(prob={prob:.3f})  |  True: {true_str}  {correct}",
        fontsize=11, fontweight="bold"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
 
    plt.tight_layout()
    out_path = os.path.join(
        out_dir, f"shap_waterfall_protein_{protein_idx}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
 
    # Print biological narrative
    print(f"\n  Protein {protein_idx} — biological narrative:")
    print(f"  Prediction : {pred_label}  (confidence = {prob:.3f})")
    print(f"  True label : {true_str}  {correct}")
    print(f"\n  Top driving features:")
    for fname, sv_val, xv in zip(top_fnames, top_sv, top_xvals):
        direction = "TOWARD mech." if sv_val > 0 else "AWAY from mech."
        bio       = FEATURE_BIOLOGY.get(fname, "")
        print(f"    {fname:<38} SHAP={sv_val:+.4f}  "
              f"val={xv:.3f}  → {direction}")
        if bio:
            print(f"      Biology: {bio}")
 
    return out_path
 
 
# ======================================================================
# 8. Class-conditional SHAP comparison
# ======================================================================
 
def plot_class_comparison(shap_values: np.ndarray,
                           feature_names: list,
                           preds: np.ndarray,
                           out_dir: str,
                           top_n: int = 20):
    """
    Compare mean SHAP values for predicted positives vs negatives.
    Shows which features are systematically different between classes.
    This answers: "What makes the model think something IS or ISN'T
    a mechanoreceptor at the population level?"
    """
    pos_mask = preds == 1
    neg_mask = preds == 0
 
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        print("  Skipping class comparison — one class is empty.")
        return
 
    mean_pos = shap_values[pos_mask].mean(axis=0)    # (D,)
    mean_neg = shap_values[neg_mask].mean(axis=0)    # (D,)
    delta    = mean_pos - mean_neg                    # (D,)
 
    # Top features by absolute difference between classes
    order    = np.argsort(np.abs(delta))[::-1][:top_n]
    top_d    = delta[order]
    top_n_   = [feature_names[i] for i in order]
 
    colours = ["#E8593C" if v > 0 else "#378ADD" for v in top_d]
 
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(top_d)),
        top_d[::-1],
        color=colours[::-1],
        edgecolor="none", height=0.65
    )
    ax.set_yticks(range(len(top_d)))
    ax.set_yticklabels(top_n_[::-1], fontsize=9)
    ax.set_xlabel(
        "Mean SHAP (predicted mechanoreceptors) − "
        "Mean SHAP (predicted non-mechanoreceptors)",
        fontsize=10
    )
    ax.set_title(
        "Feature SHAP difference between classes\n"
        "(red = higher in mechanoreceptors, blue = higher in non-mech.)",
        fontsize=11, fontweight="bold"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
 
    out_path = os.path.join(out_dir, "shap_class_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
 
    return delta, top_n_
 
 
# ======================================================================
# 9. Feature group summary
# ======================================================================
 
def plot_group_importance(shap_values: np.ndarray,
                           feature_names: list,
                           out_dir: str):
    """
    Aggregate SHAP importance by biological feature group.
    Gives a high-level view: "Which type of information matters most?"
    e.g. Is topology more important than charge distribution?
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    groups   = [get_feature_group(f) for f in feature_names]
 
    group_imp = {}
    for g, imp in zip(groups, mean_abs):
        group_imp[g] = group_imp.get(g, 0.0) + imp
 
    group_df = pd.DataFrame(
        group_imp.items(), columns=["group", "total_importance"]
    ).sort_values("total_importance", ascending=False)
 
    palette = sns.color_palette("tab10", len(group_df))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        group_df["group"],
        group_df["total_importance"],
        color=palette, edgecolor="none"
    )
    ax.set_ylabel("Total SHAP importance (sum of mean |SHAP|)", fontsize=10)
    ax.set_title("Importance by biological feature group", fontsize=11,
                 fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
 
    out_path = os.path.join(out_dir, "shap_group_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
 
    print(f"\n  Feature group importance:")
    for _, row in group_df.iterrows():
        print(f"    {row['group']:<30} {row['total_importance']:.4f}")
 
    return group_df
 
 
# ======================================================================
# 10. Save SHAP values to CSV for downstream analysis
# ======================================================================
 
def save_shap_csv(shap_values: np.ndarray,
                  feature_names: list,
                  probs: np.ndarray,
                  preds: np.ndarray,
                  labels,
                  out_dir: str):
    """
    Save full SHAP matrix as CSV.
    Rows = proteins, columns = SHAP value per feature + prediction info.
    Useful for custom downstream analysis.
    """
    df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{f}" for f in feature_names]
    )
    df.insert(0, "predicted_prob",  probs)
    df.insert(1, "predicted_label", preds)
    if labels is not None:
        df.insert(2, "true_label", labels.astype(int))
 
    out_path = os.path.join(out_dir, "shap_values.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({df.shape[0]} proteins × "
          f"{df.shape[1]} columns)")
    return df
 
 
# ======================================================================
# Main
# ======================================================================
 
def parse_args():
    p = argparse.ArgumentParser(
        description="SHAP explainability for mechanoreceptor ANN."
    )
    p.add_argument("--csv",         required=True,
                   help="Features CSV (same format as training)" , default="descriptors.csv" )
    p.add_argument("--model_dir",   default="outputs",
                   help="Directory with model_weights.pt, scaler.pkl, meta.json")
    p.add_argument("--out_dir",     default="outputs/explainability",
                   help="Where to save plots and SHAP CSV.")
    p.add_argument("--top_n",       type=int, default=20,
                   help="Top N features to show in plots.")
    p.add_argument("--n_background",type=int, default=100,
                   help="Background samples for SHAP DeepExplainer.")
    p.add_argument("--explain_idx", type=int, nargs="+",
                   default=[0, 1, 2],
                   help="Indices of proteins to explain individually.")
    p.add_argument("--mc_samples",  type=int, default=100,
                   help="MC Dropout samples for uncertainty.")
    p.add_argument("--nsamples", type=int, default=512,
               help="SHAP perturbation samples per protein. "
                    "Higher = more accurate, slower. "
                    "Use 256 for quick run, 1000 for final results.")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"      # SHAP DeepExplainer works on CPU
 
    print(f"\n{'='*60}")
    print(f"SHAP Explainability Pipeline")
    print(f"{'='*60}\n")
 
    # ── 1. Load artefacts ─────────────────────────────────────────────
    model, scaler, threshold, feature_names = load_artefacts(args.model_dir)
 
    # ── 2. Load CSV and predict ───────────────────────────────────────
    df, X_scaled, X_tensor, probs, preds, labels = load_and_predict(
        args.csv, model, scaler, threshold, feature_names, device
    )
 
    # ── 3. Build SHAP explainer ───────────────────────────────────────
    explainer = build_shap_explainer(
        model, X_tensor, n_background=args.n_background
    )
 
    # ── 4. Compute SHAP values ────────────────────────────────────────
    shap_values = compute_shap_values(explainer, X_scaled)
 
    # ── 5. Global importance bar chart ───────────────────────────────
    print(f"\n--- Global feature importance ---")
    importance_df = plot_global_importance(
        shap_values, feature_names, args.out_dir, top_n=args.top_n
    )
 
    # ── 6. Beeswarm plot ──────────────────────────────────────────────
    print(f"\n--- Beeswarm (direction + magnitude) ---")
    plot_beeswarm(
        shap_values, X_scaled, feature_names,
        args.out_dir, top_n=args.top_n
    )
 
    # ── 7. Feature group summary ──────────────────────────────────────
    print(f"\n--- Feature group importance ---")
    plot_group_importance(shap_values, feature_names, args.out_dir)
 
    # ── 8. Class-conditional comparison ──────────────────────────────
    print(f"\n--- Class-conditional SHAP comparison ---")
    plot_class_comparison(
        shap_values, feature_names, preds, args.out_dir,
        top_n=args.top_n
    )
 
    # ── 9. Per-protein waterfall explanations ─────────────────────────
    print(f"\n--- Per-protein explanations ---")
    for idx in args.explain_idx:
        if idx >= len(probs):
            print(f"  Skipping index {idx} — out of range.")
            continue
        explain_single_protein(
            shap_values, X_scaled, feature_names,
            protein_idx = idx,
            prob        = probs[idx],
            pred        = preds[idx],
            true_label  = labels[idx] if labels is not None else None,
            out_dir     = args.out_dir,
            top_n       = 15,
        )
 
    # ── 10. Save SHAP CSV ─────────────────────────────────────────────
    print(f"\n--- Saving SHAP matrix ---")
    save_shap_csv(
        shap_values, feature_names, probs, preds, labels, args.out_dir
    )
 
    # ── 11. MC Dropout on flagged proteins ───────────────────────────
    # Identify uncertain predictions and print which ones need wet-lab
    print(f"\n--- MC Dropout on all proteins ---")
    result = model.predict_with_uncertainty(
        X_tensor, n_samples=args.mc_samples
    )
    uncertain_idx = [
        i for i, tier in enumerate(result["confidence"])
        if tier == "uncertain_flag_for_wetlab"
    ]
    print(f"  {len(uncertain_idx)} proteins flagged as uncertain:")
    print(f"  Indices: {uncertain_idx[:20]}"
          f"{'...' if len(uncertain_idx) > 20 else ''}")
    print(f"  These are the candidates most worth wet-lab follow-up.")
 
    print(f"\n{'='*60}")
    print(f"Explainability complete.")
    print(f"Outputs in: {args.out_dir}/")
    print(f"  shap_global_importance.png")
    print(f"  shap_beeswarm.png")
    print(f"  shap_group_importance.png")
    print(f"  shap_class_comparison.png")
    print(f"  shap_waterfall_protein_N.png  (one per --explain_idx)")
    print(f"  shap_values.csv")
    print(f"{'='*60}\n")
 
 
if __name__ == "__main__":
    main()
 








