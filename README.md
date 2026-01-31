# Selecting Color Spaces Using Explainable Artificial Intelligence

This repository contains the source code and data for the paper **"Selecting Color Spaces Using Explainable Artificial Intelligence"**.

## Overview

In computer vision, selecting the appropriate color space is crucial for robust object detection, especially under varying illumination conditions. This project employs **Explainable Artificial Intelligence (X-AI)** techniques—specifically **SHAP (SHapley Additive exPlanations)**—to audit and select the optimal color spaces for identifying objects in the **IEEE-VSSS (Very Small Size Soccer)** competition.

We propose a methodology to:
1.  Transform RGB data into multiple color spaces (LCh, HSV, HSL, YCrCb, XYZ, Lab, Luv, YUV, OKlab).
2.  Train a **Support Vector Machine (SVM)** classifier.
3.  Use **KernelExplainer** to determine the most important channels.
4.  Derive simple, human-readable decision rules for color classification.

## Key Results

The core contribution of this work is a method to select the optimal color space for classification. We found a very strong correlation between our proposed SHAP-based selection metric and the actual accuracy of the SVM classifier:

*   **$k=1$ Channel:** Spearman's $\rho(9)= 0.96$ ($p < .001$).
*   **$k=2$ Channels:** Spearman's $\rho(9)= 0.80$ ($p < .001$).

### Optimal Color Spaces
| Channels ($k$) | Best Space (Proposed) | Best Space (Actual Accuracy) | Accuracy |
| :--- | :--- | :--- | :--- |
| **1** | **OKLab** (Channel `a`) | **OKLab** (Channel `a`) | **86.99%** |
| **2** | **LCh** (`C`, `h`) | **LCh** (`C`, `h`) / **HSL** | **99.32%** |
| **3** | All | All | **100.00%** |

## Repository Structure

*   `src/`: Source code for models and analysis.
    *   `main_svm_pro.py`: **Main script** that generates the optimal accuracy results (Table 4 in the paper).
    *   `main_svm.py`: Basic SVM implementation.
    *   `main_torch.py`: Neural Network implementation (PyTorch).
    *   `main.KNN.py`: K-Nearest Neighbors implementation.
    *   `main.py`: LightGBM implementation.
    *   `compare_models.py`: Script to compare accuracy and execution time.
    *   `compare_shap.py`: Advanced script to compare SHAP values and stability.
*   `dataframes/` & `real_data_regras/`: Datasets containing color samples from VSSS environments.
*   `main.tex`: LaTeX source of the paper.

## Installation

This project requires Python 3.8+ and the following libraries:

```bash
pip install numpy pandas opencv-python scikit-learn shap torch lightgbm colorspacious matplotlib
```

Or using `uv`:

```bash
uv pip install numpy pandas opencv-python scikit-learn shap torch lightgbm colorspacious matplotlib
```

## Usage

### Reproduce Model Comparison
To run the comprehensive comparison (Accuracy, Time, SHAP Metrics) presented in the paper:

```bash
python3 src/compare_shap.py
```

To run a quick benchmark of Accuracy and Time:

```bash
python3 src/compare_models.py
```

### Train Individual Models
You can run individual model scripts to see specific training details and plots:

```bash
python3 src/main_svm.py
```

## Authors

*   **Carlos Lara-Álvarez** (CIMAT, Zacatecas)
*   **Julio De La Torre-Vanegas** (CIMAT, Zacatecas)
*   **Jose Torres-Jimenez** (CINVESTAV-Tamaulipas)
*   **Hector Cardona-Reyes** (CIMAT, Zacatecas)
