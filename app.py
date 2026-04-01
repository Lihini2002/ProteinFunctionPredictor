"""
app.py
------
Streamlit UI for the Mechanoreceptor Classifier.

Tabs:
  1. About      — background on mechanoreceptors
  2. Predict    — enter a sequence, get prediction + SHAP + descriptor insights

Run:
    streamlit run app.py

Requirements:
    pip install streamlit shap matplotlib seaborn pandas numpy torch scikit-learn biopython
    
    Place in same folder as:
        descriptor_ann_classifier.py
        outputs/model_weights.pt
        outputs/scaler.pkl
        outputs/meta.json
"""
from PIL import Image

import os
import json
import math
import pickle
import warnings
import re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from collections import Counter
from itertools import product

from models.descriptor_ann_classifier import DescriptorANNClassifier



# ─────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MechanoDB",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────
# Custom CSS — clean dark-science aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8b949e;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background-color: #21262d !important;
    color: #58a6ff !important;
}

/* Text area */
.stTextArea textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 8px !important;
}
.stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1) !important;
}

/* Buttons */
.stButton > button {
    background-color: #238636 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: background-color 0.15s !important;
}
.stButton > button:hover {
    background-color: #2ea043 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
}

/* Info / success / warning boxes */
.stAlert {
    border-radius: 8px !important;
    border-left-width: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

/* Divider */
hr {
    border-color: #30363d !important;
    margin: 2rem 0 !important;
}

/* Code blocks */
code {
    background-color: #161b22 !important;
    color: #79c0ff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div style="display:flex; align-items:baseline; gap:12px;">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:11px;
                     color:#8b949e; letter-spacing:0.1em;">v1.0 · BINARY CLASSIFIER</span>
    </div>
    <h1 style="font-family:'IBM Plex Sans',sans-serif; font-size:2.2rem;
               font-weight:600; color:#e6edf3; margin:4px 0 0 0;
               letter-spacing:-0.02em;">
        MechanoDB
        <span style="color:#58a6ff;">.</span>
    </h1>
    <p style="font-family:'IBM Plex Sans',sans-serif; font-size:15px;
              color:#8b949e; margin:6px 0 0 0; font-weight:300;">
        Mechanoreceptor protein identification via hand-crafted biophysical descriptors
    </p>
</div>
<hr/>
""", unsafe_allow_html=True)

from descriptors import (
   aac , 
   mean_hydrophobicity, 
   charge_features, 
   gp_features, 
   aromatic_features, 
   hydrophobic_stretches, 
   low_complexity_fraction, 
   dipeptide_composition, 
   pI_feature
)
 
# ─────────────────────────────────────────────────────────────
# Constants — feature group mapping matching your descriptor code
# ─────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "AAC_"                      : "AA Composition",
    "DPC_"                      : "Dipeptide Composition",
    "hydro_"                    : "Hydrophobicity",
    "charge_"                   : "Charge",
    "gly_fraction"              : "Gly/Pro",
    "pro_fraction"              : "Gly/Pro",
    "gp_fraction"               : "Gly/Pro",
    "aromatic_fraction"         : "Aromatic",
    "hydrophobic_stretch_count" : "Hydrophobic Stretches",
    "low_complexity_frac"       : "Low Complexity",
    "pI"                        : "Physicochemical",
}

FEATURE_BIOLOGY = {
    "hydro_mean"               : "Mean KD hydrophobicity — overall membrane affinity",
    "hydro_std"                : "Hydrophobicity variability — TM vs loop contrast",
    "charge_net"               : "Net charge per residue — electrostatic identity",
    "charge_density"           : "Total charged residue density",
    "gly_fraction"             : "Glycine — helix breaker, flexible TM bends",
    "pro_fraction"             : "Proline — mechanogating kink residue",
    "gp_fraction"              : "G+P combined — total helix-breaking capacity",
    "aromatic_fraction"        : "F+W+Y — aromatic belt at membrane interface",
    "hydrophobic_stretch_count": "TM helix proxy via hydrophobic window count",
    "low_complexity_frac"      : "Disordered / low-entropy regions",
    "pI"                       : "Isoelectric point — membrane targeting signal",
    "AAC_G"                    : "Glycine — helix flexibility",
    "AAC_P"                    : "Proline — helix kinks",
    "AAC_L"                    : "Leucine — TM core packing",
    "AAC_W"                    : "Tryptophan — membrane anchor",
    "AAC_F"                    : "Phenylalanine — aromatic belt",
    "AAC_K"                    : "Lysine — positive-inside rule",
    "AAC_R"                    : "Arginine — G-protein coupling",
    "AAC_C"                    : "Cysteine — disulfide / redox gating",
}



MODEL_DIR = "outputs"

KD_SCALE = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}
AA = "ACDEFGHIKLMNPQRSTVWY"
# ─────────────────────────────────────────────────────────────
# Descriptor extraction — matches your actual descriptor code
# ─────────────────────────────────────────────────────────────
def extract_features(sequence: str) -> dict:
    """
    Compute all hand-coded descriptors for a protein sequence.
    Returns a single dictionary of features.
    """
    # Clean sequence: remove invalid characters
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    if len(seq) < 10:
        return None

    features = {}

    # --- Amino acid composition (AAC) ---
    features.update(aac(seq))

    # --- Dipeptide composition (DPC) ---
    features.update(dipeptide_composition(seq))

    # --- Hydrophobicity ---
    hydro_vals = mean_hydrophobicity(seq)
    features["hydro_mean"] = hydro_vals["hydro_mean"]
    features["hydro_std"]  = hydro_vals["hydro_std"]

    # --- Charge features ---
    charge_vals = charge_features(seq)
    features["charge_net"]     = charge_vals["charge_net"]
    features["charge_density"] = charge_vals["charge_density"]

    # --- Glycine/Proline fractions ---
    gp_vals = gp_features(seq)
    features["gly_fraction"] = gp_vals["gly_fraction"]
    features["pro_fraction"] = gp_vals["pro_fraction"]
    features["gp_fraction"]  = gp_vals["gp_fraction"]

    # --- Aromatic fraction ---
    aromatic_vals = aromatic_features(seq)
    features["aromatic_fraction"] = aromatic_vals["aromatic_fraction"]

    # --- Hydrophobic stretches ---
    features.update(hydrophobic_stretches(seq))

    # --- Low complexity regions ---
    features.update(low_complexity_fraction(seq))

    # --- Isoelectric point (pI) ---
    features.update(pI_feature(seq))

    return features

def get_feature_group(name: str) -> str:
    for prefix, group in FEATURE_GROUPS.items():
        if name.startswith(prefix) or name == prefix:
            return group
    return "Other"


# ─────────────────────────────────────────────────────────────
# Model loading — cached so it only loads once
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_artefacts():
    """Load model, scaler, threshold, feature names — once per session."""
    try:
        # Load meta information
        with open(os.path.join(MODEL_DIR, "meta.json")) as f:
            meta = json.load(f)
        
        # Load scaler
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        # Initialize model
        model = DescriptorANNClassifier(
            input_dim=meta["input_dim"],
            hidden_dims=(64, 32),
            dropout=0.3,
        )

        # Load model weights
        model.load_state_dict(
            torch.load(
                os.path.join(MODEL_DIR, "model_weights.pt"),
                map_location="cpu"
            )
        )
        model.eval()

        return model, scaler, meta["threshold"], meta["feature_names"], None

    except Exception as e:
        return None, None, None, None, str(e)

# ─────────────────────────────────────────────────────────────
# Prediction + SHAP
# ─────────────────────────────────────────────────────────────

def run_prediction(sequence: str, model, scaler, threshold, feature_names):
    """Extract features, scale, predict, return everything needed."""
    feats = extract_features(sequence)
    if feats is None:
        return None

    # Align to saved feature order
    X_raw = np.array(
        [feats.get(f, 0.0) for f in feature_names],
        dtype=np.float32
    ).reshape(1, -1)

    X_scaled = scaler.transform(X_raw).astype(np.float32)

    model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(X_scaled)).squeeze()
        prob  = torch.sigmoid(logit).item()

    pred = int(prob >= threshold)

    # MC Dropout uncertainty
    result = model.predict_with_uncertainty(
        torch.tensor(X_scaled), n_samples=100
    )
    uncertainty = result["std_prob"].item()
    confidence_tier = result["confidence"][0]

    return {
        "feats"           : feats,
        "X_scaled"        : X_scaled,
        "X_raw"           : X_raw,
        "prob"            : prob,
        "pred"            : pred,
        "uncertainty"     : uncertainty,
        "confidence_tier" : confidence_tier,
        "sequence"        : re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]',
                                   '', sequence.upper()),
    }


@st.cache_data(show_spinner=False, ttl=600)
def compute_local_shap(_model, _scaler, sequence: str,
                        feature_names: tuple, n_bg: int = 30):
    """
    KernelExplainer SHAP for a single sequence.
    Cached by sequence string so re-running same input is instant.
    _model and _scaler prefixed with _ so Streamlit doesn't try to hash them.
    """
    feats  = extract_features(sequence)
    X_raw  = np.array([feats.get(f, 0.0) for f in feature_names],
                      dtype=np.float32).reshape(1, -1)
    X_sc   = _scaler.transform(X_raw).astype(np.float32)

    def predict_fn(X):
        t = torch.tensor(X.astype(np.float32))
        _model.eval()
        with torch.no_grad():
            return torch.sigmoid(_model(t).squeeze(1)).numpy()

    # Background: small random normal (fast for single-protein explanation)
    np.random.seed(42)
    background = np.random.randn(n_bg, X_sc.shape[1]).astype(np.float32)

    explainer   = shap.KernelExplainer(predict_fn, background,
                                        link="identity")
    shap_values = explainer.shap_values(X_sc, nsamples=256, silent=True)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return shap_values.flatten()   # (D,)


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
RED_COL   = "#f85149"
AMBER     = "#d29922"


def _fig_defaults(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_SEC, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(TEXT_SEC)
    ax.yaxis.label.set_color(TEXT_SEC)
    ax.title.set_color(TEXT_PRI)


def plot_waterfall(shap_values, feature_names, top_n=15):
    """Local waterfall for one protein."""
    order    = np.argsort(np.abs(shap_values))[::-1][:top_n]
    sv       = shap_values[order]
    names    = [feature_names[i] for i in order]
    colours  = [GREEN if v > 0 else RED_COL for v in sv]

    fig, ax = plt.subplots(figsize=(9, 6))
    _fig_defaults(fig, ax)

    y_pos = range(len(sv))
    ax.barh(list(y_pos), sv[::-1], color=colours[::-1],
            height=0.65, edgecolor="none")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names[::-1], fontsize=8.5,
                       fontfamily="monospace", color=TEXT_SEC)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_xlabel("SHAP value  (green → mechanoreceptor,  red → away)",
                  fontsize=9, color=TEXT_SEC)
    ax.set_title("Local explanation — why this prediction?",
                 fontsize=11, color=TEXT_PRI, pad=12)

    # Annotate values
    for i, (v, y) in enumerate(zip(sv[::-1], y_pos)):
        ax.text(v + (0.002 if v >= 0 else -0.002), y,
                f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=7.5, color=TEXT_SEC,
                fontfamily="monospace")

    plt.tight_layout()
    return fig


def plot_hydrophobicity_profile(sequence, window=9):
    """Sliding-window hydrophobicity along the sequence."""
    vals = [KD_SCALE.get(aa, 0) for aa in sequence]
    n = len(vals)
    if n < window:
        return None

    smoothed = []
    for i in range(n - window + 1):
        smoothed.append(np.mean(vals[i:i+window]))
    x = np.arange(len(smoothed)) + window // 2

    fig, ax = plt.subplots(figsize=(10, 3))
    _fig_defaults(fig, ax)

    # Fill above/below zero
    ax.fill_between(x, smoothed, 0,
                    where=[v > 0 for v in smoothed],
                    color=AMBER, alpha=0.4, label="Hydrophobic")
    ax.fill_between(x, smoothed, 0,
                    where=[v <= 0 for v in smoothed],
                    color=BLUE, alpha=0.4, label="Hydrophilic")
    ax.plot(x, smoothed, color=TEXT_SEC, linewidth=0.8)
    ax.axhline(1.6, color=RED_COL, linewidth=0.8,
               linestyle="--", alpha=0.7, label="TM threshold (1.6)")
    ax.axhline(0, color=BORDER, linewidth=0.5)

    ax.set_xlabel(f"Sequence position  (window={window})", fontsize=9)
    ax.set_ylabel("KD score", fontsize=9)
    ax.set_title("Hydrophobicity profile", fontsize=10, color=TEXT_PRI)
    ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER,
              labelcolor=TEXT_SEC)
    plt.tight_layout()
    return fig


def plot_aa_composition(feats):
    """Bar chart of amino acid fractions."""
    aac = {aa: feats.get(f"AAC_{aa}", 0) for aa in AA}
    sorted_aa = sorted(aac.items(), key=lambda x: -x[1])
    labels = [x[0] for x in sorted_aa]
    values = [x[1] for x in sorted_aa]

    # Colour by biochemical group
    group_colours = {
        "AVILMFWP": AMBER,   # hydrophobic
        "DEKR":     RED_COL, # charged
        "STNQHY":   BLUE,    # polar
        "GP":       GREEN,   # helix-break
    }
    bar_colours = []
    for aa in labels:
        col = TEXT_SEC
        for group, c in group_colours.items():
            if aa in group:
                col = c
                break
        bar_colours.append(col)

    fig, ax = plt.subplots(figsize=(10, 3))
    _fig_defaults(fig, ax)
    ax.bar(labels, values, color=bar_colours, edgecolor="none", width=0.7)
    ax.set_ylabel("Fraction", fontsize=9)
    ax.set_title("Amino acid composition", fontsize=10, color=TEXT_PRI)

    legend_handles = [
        mpatches.Patch(color=AMBER,   label="Hydrophobic (AVILMFWP)"),
        mpatches.Patch(color=RED_COL, label="Charged (DEKR)"),
        mpatches.Patch(color=BLUE,    label="Polar (STNQHY)"),
        mpatches.Patch(color=GREEN,   label="Helix-break (GP)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5,
              facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_SEC,
              ncol=2)
    plt.tight_layout()
    return fig


def plot_shap_group_bar(shap_values, feature_names):
    """Aggregate SHAP by feature group."""
    group_shap = {}
    for sv, fn in zip(shap_values, feature_names):
        g = get_feature_group(fn)
        group_shap[g] = group_shap.get(g, 0.0) + abs(sv)

    # Remove DPC from this plot — it dominates by count
    group_shap.pop("Dipeptide Composition", None)

    sorted_g = sorted(group_shap.items(), key=lambda x: -x[1])
    labels = [x[0] for x in sorted_g]
    values = [x[1] for x in sorted_g]

    palette = [BLUE, GREEN, AMBER, RED_COL,
               "#bc8cff", "#79c0ff", "#56d364", "#ffa657"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    _fig_defaults(fig, ax)
    bars = ax.bar(labels, values,
                  color=palette[:len(labels)], edgecolor="none", width=0.6)
    ax.set_ylabel("Total |SHAP|", fontsize=9)
    ax.set_title("Feature group contribution  (excluding dipeptides)",
                 fontsize=10, color=TEXT_PRI)
    plt.xticks(rotation=25, ha="right", fontsize=8.5, color=TEXT_SEC)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
model, scaler, threshold, feature_names, load_error = load_model_artefacts()


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_about, tab_predict = st.tabs(["About", " Predict"])

with tab_about:
    # Define two columns: left (main content) and right (clinical relevance)
    col_l, col_r = st.columns([2, 1])

    # LEFT COLUMN — Mechanoreceptor background and families
    with col_l:

        st.markdown(r"""
<h2 style="font-family:'IBM Plex Sans',sans-serif; font-size:1.6rem;
           font-weight:600; color:#e6edf3; margin-bottom:0.5rem;">
    What are mechanoreceptors?
</h2>
<p style="color:#8b949e; font-size:15px; line-height:1.8; margin-bottom:1.5rem;">
    Mechanoreceptors are proteins that convert mechanical stimuli — membrane
    tension, stretch, pressure, and osmotic force — into biological signals.
    They are the molecular basis of touch, proprioception, hearing, and
    cardiovascular regulation.
</p> 

<h2 style="font-family:'IBM Plex Sans',sans-serif; font-size:1.6rem;
           font-weight:600; color:#e6edf3; margin:1.2rem 0 0.5rem 0;">
    Molecular monoatomic transmembrane mechanoreceptors
</h2>

<p style="color:#8b949e; font-size:15px; line-height:1.8; margin-bottom:1.5rem;">
    Molecular monoatomic transmembrane mechanoreceptors are single-protein,
    membrane-spanning ion channels that convert mechanical forces on the cell
    membrane into electrical or biochemical signals by opening their pore in
    response to tension.
</p>


<h3 style="font-family:'IBM Plex Mono',monospace; font-size:13px;
           color:#58a6ff; letter-spacing:0.05em; text-transform:uppercase;
           margin-bottom:0.5rem;">
    Key families
</h3>
""", unsafe_allow_html=True)

        families = {
            "Piezo1 / Piezo2": ("Large pore-forming channels (~2500 aa, 24–38 TM helices). "
                                "Activated by membrane tension. Piezo1 governs red blood cell "
                                "volume; Piezo2 mediates touch and proprioception. Mutations "
                                "cause hereditary xerocytosis and PIEZO2 syndrome."),
            "TRP channels (TRPV, TRPM, TRPA)": ("6-TM tetrameric channels gated by mechanical "
                                                  "force, temperature, and chemical stimuli. "
                                                  "TRPV4 responds to osmotic stress and is "
                                                  "implicated in skeletal dysplasias."),
            "OSCA / TMEM63": ("Plant and animal mechanosensitive channels. Two-TM architecture. "
                               "Key role in osmosensing and cell volume regulation."),
            "K2P channels (TREK, TRAAK)": ("Two-pore domain potassium channels. Directly "
                                            "activated by membrane stretch and lipid bilayer "
                                            "deformation. Important in pain and anaesthesia."),
            "ENaC / Degenerin": ("Epithelial sodium channels. Baroreceptor function in kidney "
                                  "and cardiovascular system. Activated by shear stress and touch."),
            "MscL / MscS (bacterial)": ("Prokaryotic mechanosensitive channels that protect "
                                         "against osmotic shock by acting as emergency pressure "
                                         "valves. Structural archetypes for the whole field."),
        }

        for name, desc in families.items():
            st.markdown(f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:8px;
            padding:14px 16px; margin-bottom:10px;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:12px;
                color:#58a6ff; margin-bottom:4px;">{name}</div>
    <div style="color:#8b949e; font-size:13px; line-height:1.6;">{desc}</div>
</div>
""", unsafe_allow_html=True)

    # RIGHT COLUMN — Clinical relevance
    with col_r:
        st.markdown("<h3 style='font-family:monospace ;color:#58a6ff; font-size:13px; margin-top:1.5rem;'>GO Term </h3>", unsafe_allow_html=True)

        st.markdown("<h3 style='font-family:monospace ;color:#B6DFFC; font-size:25px; border: 2px solid #FFFBB5 ;display: inline-block; padding: 8px; '>GO:0098782 </h3>", unsafe_allow_html=True)

        st.markdown("""
<h3 style="font-family:'IBM Plex Mono',monospace; font-size:13px;
           color:#58a6ff; letter-spacing:0.05em; text-transform:uppercase;
           margin-bottom:0.75rem;">
    Clinical relevance
</h3>
""", unsafe_allow_html=True)

        diseases = [
            ("Hereditary xerocytosis",    "PIEZO1 gain-of-function"),
            ("PIEZO2 syndrome",           "PIEZO2 loss-of-function"),
            ("Skeletal dysplasias",       "TRPV4 missense variants"),
            ("Hypertension",              "ENaC overactivation"),
            ("Dehydrated stomatocytosis", "PIEZO1 GOF mutations"),
            ("Proprioception disorders",  "PIEZO2 LOF mutations"),
        ]

        for disease, cause in diseases:
            st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:start;
            padding:8px 0; border-bottom:1px solid #21262d;">
    <div style="color:#e6edf3; font-size:13px;">{disease}</div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:10px;
                color:#d29922; text-align:right; max-width:140px;
                line-height:1.4;">{cause}</div>
</div>
""", unsafe_allow_html=True)
            
        metrics = [
    ("MCC", 0.8981),
    ("F1", 0.9297),
    ("Precision", 0.8958),
    ("Recall", 0.9663),
    ("PR AUC", 0.9851),
    ("ROC_AUC", 0.9930),
    ("AUC-ROC", 0.9930),
    ("AUC-PR", 0.9851),
]
        st.markdown("<h3 style='color:#58a6ff; font-size:15px; margin-top:1.5rem;'>MODEL PERFORMANCE METRICS</h3>", unsafe_allow_html=True)
        for metric, value in metrics:
            st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:6px 0; border-bottom:1px solid #21262d;">
            <div style="color:#e6edf3; font-size:14px;">{metric}</div>
            <div style="color:#72B879; font-size:14px;">{value:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        roc_img = Image.open("outputs/model_roc_curve.png") 
        st.markdown("<h3 style='color:#58a6ff; font-size:15px; margin-top:1.5rem;'>ROC AUC CURVE</h3>", unsafe_allow_html=True)

        st.image(roc_img, caption="ROC-AUC Curve", width="stretch")  

# Display the image

# Load the ROC-AUC image (replace with your file path)
     
        st.markdown("<br/>", unsafe_allow_html=True); 

# ══════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab_predict:

    # ── Model status banner ───────────────────────────────────
    if load_error:
        st.error(f"⚠ Model failed to load: `{load_error}`  "
                 f"— make sure `{MODEL_DIR}/` contains "
                 f"`model_weights.pt`, `scaler.pkl`, `meta.json`")
    else:
        st.markdown(f"""
<div style="background:#161b22; border:1px solid #238636;
            border-radius:6px; padding:8px 14px; margin-bottom:1.5rem;
            display:inline-block; font-family:'IBM Plex Mono',monospace;
            font-size:11px; color:#3fb950;">
    ● model loaded · {len(feature_names)} features · threshold={threshold:.2f}
</div>
""", unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────
    col_input, col_spacer = st.columns([3, 1])
    with col_input:
        st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
            color:#8b949e; letter-spacing:0.05em; text-transform:uppercase;
            margin-bottom:6px;">
    Protein sequence (FASTA or raw)
</div>
""", unsafe_allow_html=True)

        example_seq = (
            """MDFRNSFKSHSSYKQIRSPGDQSETSTPEHRPILHDPDMDHHKTESSSSFHEDCRDAPVE
            RDPSYNFWQDNKTSEQAAAAGTSGREPTVMTRKSGRISRSFNFGSGKPPPMEESPTKMAG
            GEQRQWGGGGEITVDVDQENEEDASRHTLPTPASTARTSFDASRELRVSFKVREAGSTTF
            TGSVASSSSTTPSSSSSATLRTNQDTQQQQEDEVVRCTSNTSFQRKSELISRVKTRSRLQ
            DPPREEDTPYSGWRSGQLKSGLLGDIDEEDDPLADEDVPDEYKRGKLDAITLLEWLSLVA
            IIAALACSLSIPSWKKVRLWNLHLWKWEVFLLVLICGRLVSGWGIRIIVFFIERNFLLRK
            RVLYFVYGVRRAVQNCLWLGLVLLAWHFLFDKKVQRETKSKFLPYVTKILVCFLLSTILW
            LIKTLVVKVMASSFHVSTYFDRIQEAMFNQYVIETLSGPPMIEMSRIEEEEEKAQEEIFK
            MQNAGANLPPDLCAAALAPGKSGRVMNPKLSPIIPKTTADSGISMEHLHRMNHKNISAWN
            MKRLMKIVRHVSLTTLDEQMLESTYEDESTRQIRSEKEAKAAARKIFKNVAQRGAKHIYM
            DDLIRFLREDEAMKTMSLFEGAPETRRISKSALKNWLVNAFRERRALALTLNDTKTAVNK
            LHHMINIVTAIVIVVIWLVLLEIASSKVLLFVSSQVVLLAFIFGNTVKTVFESIIFLFIV
            HPYDVGDRCEIDDVQLVVEEMNILTTVFLRYDNLKIMYPNSLLWQKSISNYYRSPDMGDA
            IEFCVHITTPVEKIATIRQRISNYIDNKPEYWHPSAKIIVKNVEGLNMIRLVIWPDHRFN
            HQDMLERWSRRSVLVEEVIKILLELDIQHRFYPHDINVKTMPTVVSSRVPQGWSENPA"""
        )

        seq_input = st.text_area(
            label="sequence_input",
            value="",
            height=160,
            placeholder=f"Paste sequence here…\n\nExample (Unknown protein fragment):\n{example_seq[:80]}…",
            label_visibility="collapsed",
        )

        col_btn1, col_btn2, _ = st.columns([1, 1, 3])
        with col_btn1:
            run_btn = st.button("▶  Run prediction", type="primary")
        with col_btn2:
            load_btn = st.button("Load example")
            if load_btn:
                seq_input = example_seq
                

    # ── Run prediction ────────────────────────────────────────
    if (run_btn or load_btn) and seq_input.strip() :
        if load_error:
            st.error("Cannot predict — model not loaded.")
        else:
            # Strip FASTA header if present
            lines = seq_input.strip().split("\n")
            clean_lines = [l for l in lines if not l.startswith(">")]
            raw_seq = "".join(clean_lines).strip()

            if len(re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '',
                          raw_seq.upper())) < 20:
                st.warning("Sequence too short — need at least 20 valid amino acids.")
            else:
                with st.spinner("Computing descriptors…"):
                    result = run_prediction(
                        raw_seq, model, scaler, threshold, feature_names
                    )

                if result is None:
                    st.error("Feature extraction failed — check sequence.")
                else:
                    seq     = result["sequence"]
                    prob    = result["prob"]
                    pred    = result["pred"]
                    unc     = result["uncertainty"]
                    tier    = result["confidence_tier"]
                    feats   = result["feats"]

          
                    # ── Prediction banner ─────────────────────
                    if pred == 1:
                        banner_col  = "#238636"
                        banner_text = "MECHANORECEPTOR"
                        banner_icon = "✓"
                        banner_bg   = "#0d1117"
                    else:
                        banner_col  = "#da3633"
                        banner_text = "NON-MECHANORECEPTOR"
                        banner_icon = "✗"
                        banner_bg   = "#0d1117"

                    tier_colours = {
                        "high_confidence_positive" : "#3fb950",
                        "high_confidence_negative" : "#f85149",
                        "uncertain_flag_for_wetlab": "#d29922",
                    }
                    tier_labels = {
                        "high_confidence_positive" : "High confidence",
                        "high_confidence_negative" : "High confidence",
                        "uncertain_flag_for_wetlab": "⚠ Flag for wet-lab",
                    }

                    st.markdown(f"""
<div style="background:{banner_bg}; border:2px solid {banner_col};
            border-radius:10px; padding:20px 28px; margin:1.5rem 0;
            display:flex; align-items:center; gap:20px;">
    <div style="font-size:2.5rem; color:{banner_col};">{banner_icon}</div>
    <div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.3rem;
                    font-weight:600; color:{banner_col}; letter-spacing:0.05em;">
            {banner_text}
        </div>
        <div style="color:#8b949e; font-size:13px; margin-top:4px;">
            Probability: <span style="color:#e6edf3; font-family:monospace;">
            {prob:.4f}</span> &nbsp;|&nbsp;
            Uncertainty (MC Dropout): <span style="color:#e6edf3;
            font-family:monospace;">{unc:.4f}</span> &nbsp;|&nbsp;
            <span style="color:{tier_colours.get(tier,'#8b949e')};">
            {tier_labels.get(tier, tier)}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

                    # ── Quick metrics ─────────────────────────
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Length",      f"{len(seq)} aa")
                    m2.metric("Hydro mean",  f"{feats['hydro_mean']:.3f}")
                    m3.metric("pI",          f"{feats['pI']:.2f}")
                    m4.metric("TM proxies",  int(feats['hydrophobic_stretch_count']))
                    m5.metric("Aromatic",    f"{feats['aromatic_fraction']:.3f}")

                    st.markdown("<br/>", unsafe_allow_html=True)

                    # ── Two-column layout: plots + SHAP ───────
                    col_plots, col_shap = st.columns([1.1, 1])

                    with col_plots:
                        st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
            color:#8b949e; letter-spacing:0.05em; text-transform:uppercase;
            margin-bottom:10px;">Molecule insights</div>
""", unsafe_allow_html=True)

                        # Hydrophobicity profile
                        fig_hydro = plot_hydrophobicity_profile(seq)
                        if fig_hydro:
                            st.pyplot(fig_hydro, use_container_width=True)
                            plt.close()

                        st.markdown("<br/>", unsafe_allow_html=True)

                        # AA composition
                        fig_aac = plot_aa_composition(feats)
                        st.pyplot(fig_aac, use_container_width=True)
                        plt.close()

                    with col_shap:
                        st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
            color:#8b949e; letter-spacing:0.05em; text-transform:uppercase;
            margin-bottom:10px;">SHAP local explanation</div>
""", unsafe_allow_html=True)

                        with st.spinner("Running SHAP KernelExplainer…"):
                            shap_vals = compute_local_shap(
                                model, scaler, seq,
                                tuple(feature_names), n_bg=30
                            )

                        # Waterfall
                        fig_wf = plot_waterfall(shap_vals, feature_names, top_n=15)
                        st.pyplot(fig_wf, use_container_width=True)
                        plt.close()

                        st.markdown("<br/>", unsafe_allow_html=True)

                        # Group-level bar
                        fig_grp = plot_shap_group_bar(shap_vals, feature_names)
                        st.pyplot(fig_grp, use_container_width=True)
                        plt.close()

                    # ── Descriptor table ──────────────────────
                    st.markdown("<hr/>", unsafe_allow_html=True)
                    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
            color:#8b949e; letter-spacing:0.05em; text-transform:uppercase;
            margin-bottom:10px;">Descriptor summary</div>
""", unsafe_allow_html=True)

                    # Non-DPC features for the table
                    summary_feats = {
                        k: v for k, v in feats.items()
                        if not k.startswith("DPC_")
                    }

                    # Build a clean table with biology annotations
                    rows = []
                    for fname, fval in summary_feats.items():
                        rows.append({
                            "Feature"     : fname,
                            "Group"       : get_feature_group(fname),
                            "Value"       : round(fval, 5),
                            "Biology"     : FEATURE_BIOLOGY.get(fname, "—"),
                            "SHAP impact" : round(float(
                                shap_vals[list(feature_names).index(fname)]
                            ), 5) if fname in feature_names else 0.0,
                        })

                    df_summary = pd.DataFrame(rows).sort_values(
                        "SHAP impact", key=abs, ascending=False
                    )

                    st.dataframe(
                        df_summary,
                        use_container_width=True,
                        height=320,
                        hide_index=True,
                        column_config={
                            "SHAP impact": st.column_config.NumberColumn(
                                format="%.4f"
                            ),
                            "Value": st.column_config.NumberColumn(
                                format="%.4f"
                            ),
                        }
                    )

                    # ── Top SHAP features narrative ───────────
                    top5_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
                    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
            color:#8b949e; letter-spacing:0.05em; text-transform:uppercase;
            margin:1rem 0 8px 0;">Top 5 drivers</div>
""", unsafe_allow_html=True)

                    for rank, idx in enumerate(top5_idx, 1):
                        fname  = feature_names[idx]
                        sv     = shap_vals[idx]
                        bio    = FEATURE_BIOLOGY.get(fname, "—")
                        colour = GREEN if sv > 0 else RED_COL
                        direc  = "→ mechanoreceptor" if sv > 0 else "→ away"

                        st.markdown(f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:6px;
            padding:10px 14px; margin-bottom:7px; display:flex;
            align-items:center; gap:14px;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:18px;
                color:#30363d; min-width:24px;">#{rank}</div>
    <div style="flex:1;">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <code style="font-size:12px; color:#79c0ff;">{fname}</code>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:12px;
                         color:{colour};">{sv:+.4f}  {direc}</span>
        </div>
        <div style="color:#8b949e; font-size:12px; margin-top:3px;">{bio}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    elif run_btn:
        st.warning("Please enter a protein sequence first.")
                           
