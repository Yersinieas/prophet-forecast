"""
forecast.py
-----------
Main forecasting script using Facebook Prophet.

Usage:
    python src/forecast.py --data data/sample_data.csv --periods 90

Expects a CSV with two columns:
    ds  : datestamp (YYYY-MM-DD)
    y   : numeric value to forecast
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ── 1. Load Data ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    df = pd.read_csv(filepath)
    assert "ds" in df.columns and "y" in df.columns, \
        "CSV must have 'ds' (datestamp) and 'y' (value) columns."
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df


# ── 2. Train / Test Split ─────────────────────────────────────────────────────

def train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split dataframe into train and test sets by time."""
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


# ── 3. Fit Model ──────────────────────────────────────────────────────────────

def fit_model(train: pd.DataFrame) -> Prophet:
    """Fit a Prophet model on training data."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,   # 95% uncertainty interval
    )
    model.fit(train)
    print("Model fitted successfully.")
    return model


# ── 4. Forecast ───────────────────────────────────────────────────────────────

def forecast(model: Prophet, periods: int) -> pd.DataFrame:
    """Generate future forecast for `periods` days ahead."""
    future = model.make_future_dataframe(periods=periods)
    forecast_df = model.predict(future)
    print(f"Forecast generated for {periods} future periods.")
    return forecast_df


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(model: Prophet, test: pd.DataFrame, forecast_df: pd.DataFrame):
    """Compare forecast against held-out test data."""
    # Merge test actuals with forecast predictions
    merged = test.merge(forecast_df[["ds", "yhat"]], on="ds", how="inner")
    if merged.empty:
        print("No overlapping dates for evaluation.")
        return

    mae  = mean_absolute_error(merged["y"], merged["yhat"])
    rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
    print(f"\n--- Evaluation Results ---")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    return {"mae": mae, "rmse": rmse}


# ── 6. Plot & Save ────────────────────────────────────────────────────────────

def save_plots(model: Prophet, forecast_df: pd.DataFrame, output_dir: str = "outputs"):
    """Save forecast and components plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Forecast plot
    fig1 = model.plot(forecast_df)
    fig1.suptitle("Prophet Forecast", fontsize=14)
    fig1.savefig(os.path.join(output_dir, "forecast.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Components plot (trend, seasonality)
    fig2 = model.plot_components(forecast_df)
    fig2.savefig(os.path.join(output_dir, "components.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Plots saved to {output_dir}/")


# ── 7. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prophet time-series forecaster")
    parser.add_argument("--data",    type=str, default="data/sample_data.csv",
                        help="Path to input CSV (requires ds and y columns)")
    parser.add_argument("--periods", type=int, default=90,
                        help="Number of future days to forecast")
    parser.add_argument("--output",  type=str, default="outputs",
                        help="Directory to save plots and results")
    args = parser.parse_args()

    # Pipeline
    df             = load_data(args.data)
    train, test    = train_test_split(df, test_ratio=0.2)
    model          = fit_model(train)
    forecast_df    = forecast(model, periods=args.periods)
    evaluate(model, test, forecast_df)
    save_plots(model, forecast_df, output_dir=args.output)

    # Save forecast CSV
    out_csv = os.path.join(args.output, "forecast_output.csv")
    forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_csv, index=False)
    print(f"Forecast CSV saved to {out_csv}")


if __name__ == "__main__":
    main()
