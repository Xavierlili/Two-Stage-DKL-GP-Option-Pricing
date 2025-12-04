"""
iv_dkl_pipeline.py

Clean core pipeline for two-stage DKL + GP option pricing:

Stage 1: Implied Volatility (IV) model
Stage 2: Price model using predicted IV

Assumptions about data/SPX500.csv:
- Contains at least:
    date, exdate, cp_flag, strike_price,
    best_bid, best_offer,
    impl_volatility, spx_close, r

Chronological split to avoid look-ahead bias.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ======================================================
# 0. Utility: metrics
# ======================================================

def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ======================================================
# 1. Data loading & preprocessing
# ======================================================

def load_and_preprocess(data_path: str) -> pd.DataFrame:
    """Load CSV and construct basic derived features."""

    df = pd.read_csv(data_path, low_memory=False)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")
    df = df.dropna(subset=["date", "exdate"]).reset_index(drop=True)

    # Force numeric types
    for col in ["strike_price", "best_bid", "best_offer", "spx_close", "r", "impl_volatility"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strike unit sanity check (many SPX datasets use *1000)
    med_strike = df["strike_price"].median()
    if np.isfinite(med_strike) and med_strike > 5e5:
        df["strike_price"] = df["strike_price"] / 1000.0

    # Mid price
    df["c_mid"] = 0.5 * (df["best_bid"] + df["best_offer"])

    # Time to maturity (trading years, 252 days)
    start = df["date"].values.astype("datetime64[D]")
    end = df["exdate"].values.astype("datetime64[D]")
    bdays = np.busday_count(start, end)
    df["T"] = np.maximum(bdays, 0) / 252.0
    df["sqrt_T"] = np.sqrt(np.clip(df["T"], 1e-8, None))

    # Call/put flag
    if "cp_flag" in df.columns:
        df["is_call"] = df["cp_flag"].astype(str).str.upper().eq("C").astype(int)
    else:
        # fallback: assume all calls
        df["is_call"] = 1

    # Moneyness
    df["moneyness"] = df["spx_close"] / df["strike_price"]
    df["log_moneyness"] = np.log(df["moneyness"].replace(0, np.nan))
    df["abs_log_m"] = df["log_moneyness"].abs()
    df["log_m_sq"] = df["log_moneyness"] ** 2
    df["lm_sqrtT"] = df["log_moneyness"] * df["sqrt_T"]

    # Sort by date (for chronological splits)
    df = df.sort_values("date").reset_index(drop=True)

    # Drop blatantly invalid rows
    df = df.dropna(
        subset=["impl_volatility", "c_mid", "spx_close", "strike_price", "r", "T"]
    ).reset_index(drop=True)

    # Reasonable IV range (very conservative)
    df = df[(df["impl_volatility"] > 0) & (df["impl_volatility"] < 1.0)].reset_index(drop=True)

    return df


def chronological_splits(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices for train / val / test based on date order."""

    dates = df["date"].dt.date.to_numpy()
    unique_days = np.array(sorted(np.unique(dates)))
    n_days = len(unique_days)

    train_end = int(train_frac * n_days)
    val_end = int((train_frac + val_frac) * n_days)

    train_days = set(unique_days[:train_end])
    val_days = set(unique_days[train_end:val_end])
    test_days = set(unique_days[val_end:])

    idx = np.arange(len(df))
    idx_train = idx[np.isin(dates, list(train_days))]
    idx_val = idx[np.isin(dates, list(val_days))]
    idx_test = idx[np.isin(dates, list(test_days))]

    return idx_train, idx_val, idx_test


@dataclass
class FeatureScaler:
    X_scaler: StandardScaler
    y_scaler: StandardScaler


def build_feature_matrix(df: pd.DataFrame, feature_cols) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=float)
    return X


# ======================================================
# 2. DKL + GP model definitions
# ======================================================

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 32, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPRegressionLayer(gpytorch.models.ApproximateGP):
    def __init__(self, latent_dim: int = 32, n_inducing: int = 512, device: torch.device = torch.device("cpu")):
        inducing_points = torch.randn(n_inducing, latent_dim, device=device)
        variational_distribution = CholeskyVariationalDistribution(n_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


@dataclass
class DKLModelBundle:
    encoder: FeatureExtractor
    gp_layer: GPRegressionLayer
    likelihood: GaussianLikelihood
    scaler: FeatureScaler
    device: torch.device


# ======================================================
# 3. DKL training loop (generic)
# ======================================================

def train_dkl_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    latent_dim: int = 32,
    n_inducing: int = 512,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    patience: int = 10,
) -> Tuple[DKLModelBundle, Dict[str, float]]:
    """
    Generic DKL + variational GP trainer.
    Returns trained model bundle and final validation metrics (in original y-scale).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale features and targets
    X_scaler = StandardScaler().fit(X_train)
    X_train_t = X_scaler.transform(X_train)
    X_val_t = X_scaler.transform(X_val)

    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_t = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_t = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    scaler = FeatureScaler(X_scaler=X_scaler, y_scaler=y_scaler)

    # Torch tensors
    Xtr = torch.tensor(X_train_t, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train_t, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val_t, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    encoder = FeatureExtractor(in_dim=X_train_t.shape[1], latent_dim=latent_dim, p_drop=0.2).to(device)
    gp_layer = GPRegressionLayer(latent_dim=latent_dim, n_inducing=n_inducing, device=device).to(device)
    likelihood = GaussianLikelihood().to(device)

    params = list(encoder.parameters()) + list(gp_layer.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs // 2))
    mll = VariationalELBO(likelihood, gp_layer, num_data=len(Xtr))

    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    def _predict_raw(X_tensor: torch.Tensor) -> np.ndarray:
        encoder.eval()
        gp_layer.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_dist = gp_layer(encoder(X_tensor))
            pred_dist = likelihood(f_dist)
            mean = pred_dist.mean
            if mean.dim() > 1:
                mean = mean.mean(dim=-1)
            return mean.detach().cpu().numpy().ravel()

    for epoch in range(1, epochs + 1):
        encoder.train()
        gp_layer.train()
        likelihood.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = gp_layer(encoder(xb))
            loss = -mll(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        scheduler.step()
        avg_loss = total_loss / len(Xtr)

        # Validation: compute MAE in original scale
        y_val_pred_t = _predict_raw(Xva)
        y_val_pred = scaler.y_scaler.inverse_transform(y_val_pred_t.reshape(-1, 1)).ravel()
        metrics = compute_regression_metrics(y_val, y_val_pred)
        val_mae = metrics["MAE"]

        print(
            f"[Epoch {epoch:02d}] -ELBO={avg_loss:.4f} | "
            f"Val MAE={val_mae:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}"
        )

        if val_mae + 1e-6 < best_val_mae:
            best_val_mae = val_mae
            no_improve = 0
            best_state = {
                "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "gp": {k: v.detach().cpu() for k, v in gp_layer.state_dict().items()},
                "lik": {k: v.detach().cpu() for k, v in likelihood.state_dict().items()},
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val MAE={best_val_mae:.4f}")
                break

    # Restore best state
    if best_state is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_state["encoder"].items()})
        gp_layer.load_state_dict({k: v.to(device) for k, v in best_state["gp"].items()})
        likelihood.load_state_dict({k: v.to(device) for k, v in best_state["lik"].items()})

        # Final validation metrics after restoring
        y_val_pred_t = _predict_raw(Xva)
        y_val_pred = scaler.y_scaler.inverse_transform(y_val_pred_t.reshape(-1, 1)).ravel()
        metrics = compute_regression_metrics(y_val, y_val_pred)
    else:
        metrics = {"MAE": best_val_mae, "RMSE": np.nan, "R2": np.nan}

    bundle = DKLModelBundle(
        encoder=encoder,
        gp_layer=gp_layer,
        likelihood=likelihood,
        scaler=scaler,
        device=device,
    )
    return bundle, metrics


def dkl_predict(bundle: DKLModelBundle, X: np.ndarray) -> np.ndarray:
    X_t = bundle.scaler.X_scaler.transform(X)
    Xt = torch.tensor(X_t, dtype=torch.float32, device=bundle.device)

    bundle.encoder.eval()
    bundle.gp_layer.eval()
    bundle.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        feats = bundle.encoder(Xt)
        f_dist = bundle.gp_layer(feats)
        pred_dist = bundle.likelihood(f_dist)
        mean_t = pred_dist.mean
        if mean_t.dim() > 1:
            mean_t = mean_t.mean(dim=-1)
        mean_t = mean_t.detach().cpu().numpy().ravel()
        preds = bundle.scaler.y_scaler.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    return preds


# ======================================================
# 4. Stage 1 & Stage 2 pipeline
# ======================================================

def run_pipeline(data_path: str, results_dir: str = None) -> None:
    print(f"Loading data from: {data_path}")
    df = load_and_preprocess(data_path)
    print(f"Data shape after preprocessing: {df.shape}")

    idx_train, idx_val, idx_test = chronological_splits(df)
    print(f"Train size: {len(idx_train)}, Val size: {len(idx_val)}, Test size: {len(idx_test)}")

    # ---------- Stage 1: IV model ----------
    iv_feature_cols = [
        "strike_price",
        "T",
        "spx_close",
        "r",
        "is_call",
        "moneyness",
        "log_moneyness",
        "sqrt_T",
        "abs_log_m",
        "log_m_sq",
        "lm_sqrtT",
    ]
    iv_feature_cols = [c for c in iv_feature_cols if c in df.columns]

    X_iv = build_feature_matrix(df, iv_feature_cols)
    y_iv = df["impl_volatility"].to_numpy()

    X_iv_train, y_iv_train = X_iv[idx_train], y_iv[idx_train]
    X_iv_val, y_iv_val = X_iv[idx_val], y_iv[idx_val]
    X_iv_test, y_iv_test = X_iv[idx_test], y_iv[idx_test]

    print("\n=== Stage 1: Training IV DKL-GP model ===")
    iv_model, iv_val_metrics = train_dkl_regressor(
        X_iv_train,
        y_iv_train,
        X_iv_val,
        y_iv_val,
        latent_dim=32,
        n_inducing=512,
        epochs=40,
        batch_size=2048,
        lr=1e-2,
        weight_decay=1e-4,
        patience=10,
    )
    print(f"[Stage 1] Validation metrics: {iv_val_metrics}")

    # Predictions on all data for Stage 2
    iv_pred_all = dkl_predict(iv_model, X_iv)
    df["iv_hat"] = iv_pred_all

    iv_test_pred = iv_pred_all[idx_test]
    iv_test_metrics = compute_regression_metrics(y_iv_test, iv_test_pred)
    print(f"[Stage 1] Test metrics: {iv_test_metrics}")

    # ---------- Stage 2: Price model ----------
    price_feature_cols = list(iv_feature_cols) + ["iv_hat"]
    price_feature_cols = [c for c in price_feature_cols if c in df.columns]

    X_price = build_feature_matrix(df, price_feature_cols)
    y_price = df["c_mid"].to_numpy()

    X_p_train, y_p_train = X_price[idx_train], y_price[idx_train]
    X_p_val, y_p_val = X_price[idx_val], y_price[idx_val]
    X_p_test, y_p_test = X_price[idx_test], y_price[idx_test]

    print("\n=== Stage 2: Training Price DKL-GP model ===")
    price_model, price_val_metrics = train_dkl_regressor(
        X_p_train,
        y_p_train,
        X_p_val,
        y_p_val,
        latent_dim=16,
        n_inducing=256,
        epochs=60,
        batch_size=2048,
        lr=3e-3,
        weight_decay=1e-4,
        patience=12,
    )
    print(f"[Stage 2] Validation metrics: {price_val_metrics}")

    price_test_pred = dkl_predict(price_model, X_p_test)
    price_test_metrics = compute_regression_metrics(y_p_test, price_test_pred)
    print(f"[Stage 2] Test metrics: {price_test_metrics}")

    # Optionally: save metrics to disk
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        metrics = {
            "stage1_val": iv_val_metrics,
            "stage1_test": iv_test_metrics,
            "stage2_val": price_val_metrics,
            "stage2_test": price_test_metrics,
        }
        metrics_df = pd.json_normalize(metrics)
        metrics_path = os.path.join(results_dir, "metrics_summary.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")


# ======================================================
# 5. CLI entry point
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage DKL-GP option pricing pipeline.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/SPX500.csv",
        help="Path to options CSV dataset.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save metrics/plots (optional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data_path, args.results_dir)
