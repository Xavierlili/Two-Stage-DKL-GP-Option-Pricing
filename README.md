# Two-Stage-DKL-GP-Option-Pricing
Deep Kernel Learning + Gaussian Process pipeline for S&amp;P 500 option pricing (two-stage IV + price model).

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

