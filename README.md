# Protein Function Predictor

## Overview

The **Protein Function Predictor** is a **binary classification model** designed to predict whether a protein functions as a **Molecular Monoatomic Transmembrane Mechanoreceptor** (GO:0098782). These receptors are single-protein channels that open in response to mechanical tension, mediating biochemical signals across membranes.

Understanding and predicting these mechanoreceptors is critical for research into a variety of human disorders, including:

* Hereditary xerocytosis
* PIEZO2 syndrome
* Skeletal dysplasias
* Hypertension
* Dehydrated stomatocytosis
* Proprioception disorders
* PIEZO1 gain-of-function (GOF) mutations
* PIEZO2 loss-of-function (LOF) mutations
* TRPV4 missense variants
* ENaC overactivation

This tool aims to assist researchers by providing a computational method to identify potential mechanoreceptor proteins and improve functional annotation studies.

---

## Model Architecture

The model is implemented as an **Artificial Neural Network (ANN)** trained on **handcrafted protein descriptors**. Its architecture is as follows:

```text
Input Feature Vector (Descriptors)
           │
       48 Neurons
Linear + ReLU + Dropout (0.35)
           │
       24 Neurons
Linear + ReLU + Dropout (0.35)
           │
    Single Neuron Output (Logit)
```

* **Loss Function:** Binary Cross-Entropy with `BCEWithLogitsLoss`
* **Optimizer:** `AdamW` with weight decay
* **Regularization:** Dropout (0.35)
* **Hyperparameters (from cross-validation):**

  * Hidden dimensions: (48, 24)
  * Dropout: 0.35
  * Learning rate: 0.0005
  * Weight decay: 0.01

Additionally, a **similarity feature** was added for positive predictions, which identifies the closest known positive relative from the training set. This feature enhances the model's reliability for identifying mechanoreceptors.

---

## Input Descriptors

The model uses **handcrafted descriptors** to encode sequence and physicochemical properties of proteins:

| Descriptor                  | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `hydro_mean`                | Mean Kyte-Doolittle hydrophobicity — overall membrane affinity |
| `hydro_std`                 | Hydrophobicity variability — TM vs loop contrast               |
| `charge_net`                | Net charge per residue — electrostatic identity                |
| `charge_density`            | Total charged residue density                                  |
| `gly_fraction`              | Glycine — helix breaker, flexible TM bends                     |
| `pro_fraction`              | Proline — mechanogating kink residue                           |
| `gp_fraction`               | Glycine + Proline combined — total helix-breaking capacity     |
| `aromatic_fraction`         | F+W+Y — aromatic belt at membrane interface                    |
| `hydrophobic_stretch_count` | TM helix proxy via hydrophobic window count                    |
| `low_complexity_frac`       | Disordered / low-entropy regions                               |
| `pI`                        | Isoelectric point — membrane targeting signal                  |
| `AAC_G`                     | Glycine — helix flexibility                                    |
| `AAC_P`                     | Proline — helix kinks                                          |
| `AAC_L`                     | Leucine — TM core packing                                      |
| `AAC_W`                     | Tryptophan — membrane anchor                                   |
| `AAC_F`                     | Phenylalanine — aromatic belt                                  |
| `AAC_K`                     | Lysine — positive-inside rule                                  |
| `AAC_R`                     | Arginine — G-protein coupling                                  |
| `AAC_C`                     | Cysteine — disulfide / redox gating                            |

These features allow the ANN to capture **structural, biochemical, and sequence-level information** relevant to mechanosensing.

---

## Final Test Results

The model achieves **high predictive performance**:

| Metric                                 | Score  |
| -------------------------------------- | ------ |
| MCC (Matthews Correlation Coefficient) | 0.9033 |
| F1 Score                               | 0.9333 |
| ROC AUC                                | 0.9926 |
| PR AUC                                 | 0.9837 |

These results indicate **robust classification ability**, particularly for identifying rare mechanoreceptor proteins.

---

## Functions
Feature Extraction: Converts protein sequences into numerical descriptors such as amino acid composition, hydrophobicity, charge distribution, and secondary structure propensities.
Prediction: Uses an ANN with ReLU activations and dropout to output a logit representing the probability of the protein being a mechanoreceptor.
Similarity-Based Confidence: Boosts prediction confidence by comparing predicted positives to the most similar known positive proteins from the training set.
Explainability: Provides contribution scores for each descriptor, showing which features influence the classification.
Output: Produces a probability and a binary classification for each protein.

---

## Impact

By enabling **computational identification of mechanoreceptor proteins**, this predictor can:

* Accelerate **functional annotation** of newly sequenced proteins
* Guide **experimental studies** into mechanotransduction
* Assist in **disease research** where mechanoreceptor dysfunction plays a role

---

## Requirements

* Python ≥ 3.10
* Additional libraries found in the requirements.txt file. 

---

## Usage Example

```python
from predictor import ProteinFunctionPredictor
import torch

# construct dataset from start and make the fasta files. 
python3 negativeDataset.py

# see model training metrics
python3 main_train.py

# run streamlit model to see UI 
streamlit run app.py


```

---

## References

* GO:0098782 — Molecular monoatomic transmembrane mechanoreceptor
* PIEZO1 / PIEZO2 mechanoreceptors literature
* ANN-based protein classification approaches

