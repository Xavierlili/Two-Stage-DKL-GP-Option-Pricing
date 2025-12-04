<div align="center">

# **Two-Stage Deep Kernel Learning + Gaussian Process for S&P 500 Option Pricing**

A hybrid **Deep Kernel Learning (DKL)** + **Gaussian Process Regression (GPR)** framework for robust, data-driven valuation of S&P 500 index options — implemented as a clean, reproducible, and scalable research pipeline.

</div>

---

# 🧠 Overview

This project implements a **two-stage DKL–GP architecture** for modeling:

1. **Stage 1:** Implied volatility surface  
2. **Stage 2:** Option mid-prices using DKL-GP + predicted IV

The architecture leverages:

- 🧩 Neural feature extractors  
- 📈 Nonparametric Gaussian Process modelling  
- 🧮 Chronological train/val/test splits  
- 🔍 Variational inference for scalability  
- ⚙️ End-to-end clean pipeline in PyTorch + GPyTorch  

---

# 📐 Model Architecture

                    ┌──────────────────────┐
                    │  Stage 1: IV Model   │
                    │  (DKL + GP)          │
                    └──────────┬───────────┘
                               │  predicts IV_hat
                               ▼
                    ┌──────────────────────┐
                    │ Stage 2: Price Model │
                    │ (DKL + GP using      │
                    │  IV_hat + features)  │
                    └──────────────────────┘



A deep neural encoder maps raw features to a latent space where a GP performs regression:

\[
f(x) = \mathcal{GP}(m(x), k_{\theta}(x, x'))
\]

Feature extractor:

\[
\phi(x) = \text{MLP}(x)
\]

GP operates on \(\phi(x)\):

\[
y = f(\phi(x)) + \epsilon
\]

---

# ✨ Key Features

### 🔹 Two-stage nonparametric modelling
- Stage 1 learns **implied volatility** without parametric assumptions.
- Stage 2 learns **option prices** using predicted IV.

### 🔹 Deep Kernel Learning
Combines neural networks + Gaussian Processes for strong representation power.

### 🔹 Scalable variational GP
Handles **hundreds of thousands** of option observations using inducing points.

### 🔹 Fully chronological evaluation
Avoids look-ahead bias inherent in financial time series.

### 🔹 Clean & reproducible code structure
No Colab noise. Ready for academic or production usage.

---

# 📁 Project Structure

Two-Stage-DKL-GP-Option-Pricing/
│
├── src/
│ └── iv_dkl_pipeline.py # Main training pipeline
│
├── data/
│ └── SPX500.csv # Underlying options dataset (not included)
│
├── paper/
│ └── CN39523dissertation.pdf # Full dissertation
│
├── results/ # Training curves, figures, metrics
│
├── notebooks/ # (Optional) Colab / Jupyter notebooks
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore

---

# 🚀 Getting Started

## 🔧 Installation

```bash
git clone https://github.com/Xavierlili/Two-Stage-DKL-GP-Option-Pricing.git
cd Two-Stage-DKL-GP-Option-Pricing
pip install -r requirements.txt

📊 Usage
Train both stages (IV → price model)
python src/iv_dkl_pipeline.py \
    --data-path data/SPX500.csv \
    --results-dir results/

What this does:

Loads & preprocesses option data

Trains Stage 1 DKL-GP implied volatility model

Generates IV predictions for all samples

Trains Stage 2 DKL-GP price model

Saves metrics to results/metrics_summary.csv

## 📈 Results

### Mean Absolute Error (MAE)

| Model           | Validation MAE | Test MAE |
|-----------------|----------------|----------|
| **Stage 1 — Implied Volatility** | — | **0.0126** |
| **Stage 2 — Price Model** | **6.95** | **7.86** |

### Root Mean Squared Error (RMSE)

| Model           | Validation RMSE | Test RMSE |
|-----------------|------------------|-----------|
| **Stage 1 — Implied Volatility** | — | **0.0249** |
| **Stage 2 — Price Model** | **22.93** | **19.85** |

### R² Scores

| Model           | Validation R² | Test R² |
|-----------------|----------------|----------|
| **Stage 1 — Implied Volatility** | — | **96.83%** |
| **Stage 2 — Price Model** | **0.998** | **0.998** |

> These results are taken from the full dissertation analysis (Chapter 5–6).  
> The price model achieves near-perfect generalization on both validation and hold-out datasets.

🧩 Method Details
Feature set

Strike, moneyness, log-moneyness

Time to maturity (T)

Underlying index (SPX)

Risk-free rate

Call/put flag

Stage 1 output: predicted IV

Why two stages?

Implied volatility is a smooth function of (strike, maturity, underlying), and modelling it separately reduces noise and regularizes the price model.

Why DKL + GP?

Neural networks capture nonlinear structure

GP provides uncertainty and smooth priors

Variational inference scales to large financial datasets

📑 Citation

If you use this repository, please cite:

@misc{two_stage_dkl_gp_2025,
  author       = {Xavier Li},
  title        = {Two-Stage Deep Kernel Learning + Gaussian Process for S\&P 500 Option Pricing},
  year         = {2025},
  howpublished = {\url{https://github.com/Xavierlili/Two-Stage-DKL-GP-Option-Pricing}},
}

🙌 Acknowledgements

GPyTorch

PyTorch

Gaussian Process literature

Deep Kernel Learning framework
