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

$$
f(x) = \mathcal{GP}\big(m(x),\, k_\theta(x, x')\big)
$$

Feature extractor:

$$
\phi(x) = \text{MLP}(x)
$$

GP operates on the extracted features:

$$
y = f(\phi(x)) + \epsilon
$$


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

# 📁 Project Structure

```text
Two-Stage-DKL-GP-Option-Pricing/
│
├── src/
│   └── iv_dkl_pipeline.py          # Main training pipeline
│
├── data/
│   └── SPX500.csv                  # Underlying options dataset (not included)
│
├── paper/
│   └── CN39523dissertation.pdf     # Full dissertation
│
├── results/                        # Training curves, figures, metrics
│
├── notebooks/                      # (Optional) Colab / Jupyter notebooks
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore



# 🚀 Installation
git clone https://github.com/Xavierlili/Two-Stage-DKL-GP-Option-Pricing.git
cd Two-Stage-DKL-GP-Option-Pricing
pip install -r requirements.txt
