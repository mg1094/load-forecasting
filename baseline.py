#!/usr/bin/env python3
"""
Run baselines (ARIMA, Prophet, XGBoost) against the LSTM model for comparison.
Outputs a benchmark table for README.

Usage:
    python baseline.py
"""

import numpy as np
import pandas as pd
import json
import os
import time
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def load_data(filepath="data/pjm_load.csv"):
    """Load PJM data and extract hourly load."""
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    # Use the last 7 days as test set
    train = df.iloc[:-168]["load_mw"]
    test = df.iloc[-168:]["load_mw"]

    return train.values, test.values


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def baseline_persistence(train, test):
    """Persist same hour from previous day."""
    predictions = []
    for i in range(len(test)):
        if i < 24:
            predictions.append(train[-24 + i])
        else:
            predictions.append(test[i - 24])
    return np.array(predictions)


def baseline_arima(train, test):
    """ARIMA(5,1,2) forecast."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("  ⚠️  statsmodels not installed. Run: pip install statsmodels")
        return None

    model = ARIMA(train, order=(5, 1, 2))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    return forecast


def baseline_prophet(train, test):
    """Prophet forecast."""
    try:
        from prophet import Prophet
    except ImportError:
        print("  ⚠️  prophet not installed. Run: pip install prophet")
        return None

    df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=len(train), freq="h"), "y": train})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=len(test), freq="h")
    forecast = model.predict(future)
    return forecast.tail(len(test))["yhat"].values


def baseline_xgboost(train, test):
    """XGBoost with lag features."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  ⚠️  xgboost not installed. Run: pip install xgboost")
        return None

    # Build features: lag-1, lag-24, lag-168, hour, day_of_week
    def build_features(series, start_idx=0):
        X, y = [], []
        lookback = max(168, start_idx) + 1
        for i in range(lookback, len(series)):
            X.append([
                series[i - 1],
                series[i - 24] if i >= 24 else series[0],
                series[i - 168] if i >= 168 else series[0],
                (i % 24) / 23.0,  # hour normalized
                ((i // 24) % 7) / 6.0,  # day_of_week normalized
            ])
            y.append(series[i])
        return np.array(X), np.array(y)

    full = np.concatenate([train, test])
    X_train, y_train = build_features(train)
    X_test, y_test = build_features(full, start_idx=len(train))

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
    model.fit(X_train, y_train)
    return model.predict(X_test[-len(test):])


def run_benchmark():
    print("=" * 60)
    print("📊 LOAD FORECASTING BENCHMARK")
    print("=" * 60)

    # Load data
    data_files = ["data/pjm_load.csv", "data/load_data.csv"]
    for f in data_files:
        if os.path.exists(f):
            train, test = load_data(f)
            print(f"✅ Dataset: {f} ({len(train):,} train, {len(test)} test)")
            break
    else:
        print("❌ No data found. Run: python download_pjm.py")
        return

    results = []

    # 1. Persistence baseline
    print("\n1. 🔨 Persistence (same hour yesterday)...")
    t0 = time.time()
    pred_persist = baseline_persistence(train, test)
    elapsed = time.time() - t0
    results.append({
        "model": "Naive (persistence)",
        "mae": mean_absolute_error(test, pred_persist),
        "rmse": np.sqrt(mean_squared_error(test, pred_persist)),
        "mape": mape(test, pred_persist),
        "r2": r2_score(test, pred_persist),
        "time_s": elapsed,
    })
    print(f"   MAE: {results[-1]['mae']:.0f} MW, RMSE: {results[-1]['rmse']:.0f} MW")

    # 2. ARIMA
    print("\n2. 📈 ARIMA(5,1,2)...")
    t0 = time.time()
    pred_arima = baseline_arima(train, test)
    elapsed = time.time() - t0
    if pred_arima is not None:
        results.append({
            "model": "ARIMA(5,1,2)",
            "mae": mean_absolute_error(test, pred_arima),
            "rmse": np.sqrt(mean_squared_error(test, pred_arima)),
            "mape": mape(test, pred_arima),
            "r2": r2_score(test, pred_arima),
            "time_s": elapsed,
        })
        print(f"   MAE: {results[-1]['mae']:.0f} MW, RMSE: {results[-1]['rmse']:.0f} MW")

    # 3. Prophet
    print("\n3. 🔮 Prophet...")
    t0 = time.time()
    pred_prophet = baseline_prophet(train, test)
    elapsed = time.time() - t0
    if pred_prophet is not None:
        results.append({
            "model": "Prophet",
            "mae": mean_absolute_error(test, pred_prophet),
            "rmse": np.sqrt(mean_squared_error(test, pred_prophet)),
            "mape": mape(test, pred_prophet),
            "r2": r2_score(test, pred_prophet),
            "time_s": elapsed,
        })
        print(f"   MAE: {results[-1]['mae']:.0f} MW, RMSE: {results[-1]['rmse']:.0f} MW")

    # 4. XGBoost
    print("\n4. 🌲 XGBoost...")
    t0 = time.time()
    pred_xgb = baseline_xgboost(train, test)
    elapsed = time.time() - t0
    if pred_xgb is not None:
        results.append({
            "model": "XGBoost",
            "mae": mean_absolute_error(test, pred_xgb),
            "rmse": np.sqrt(mean_squared_error(test, pred_xgb)),
            "mape": mape(test, pred_xgb),
            "r2": r2_score(test, pred_xgb),
            "time_s": elapsed,
        })
        print(f"   MAE: {results[-1]['mae']:.0f} MW, RMSE: {results[-1]['rmse']:.0f} MW")

    # 5. LSTM (if trained model exists)
    print("\n5. 🤖 LSTM...")
    model_path = "models/lstm_model_pytorch.pth"
    if os.path.exists(model_path):
        # Run a quick LSTM evaluation
        try:
            from data_preprocessor import DataPreprocessor
            from lstm_model_pytorch import LSTMLoadPredictor

            t0 = time.time()
            preprocessor = DataPreprocessor(sequence_length=24, prediction_steps=1)
            df_data = pd.read_csv("data/pjm_load.csv" if os.path.exists("data/pjm_load.csv") else "data/load_data.csv")
            df_data["datetime"] = pd.to_datetime(df_data["datetime"])
            data_dict = preprocessor.prepare_data(df_data)

            predictor = LSTMLoadPredictor(
                sequence_length=24,
                n_features=data_dict["X_train"].shape[2],
                prediction_steps=1,
            )
            predictor.load_model(model_path)
            lstm_metrics, lstm_preds, lstm_true = predictor.evaluate(
                data_dict["X_test"], data_dict["y_test"], preprocessor
            )
            elapsed = time.time() - t0

            results.append({
                "model": "LSTM (ours)",
                "mae": lstm_metrics["MAE"],
                "rmse": lstm_metrics["RMSE"],
                "mape": lstm_metrics["MAPE"],
                "r2": lstm_metrics["R²"],
                "time_s": elapsed,
            })
            print(f"   MAE: {results[-1]['mae']:.0f} MW, RMSE: {results[-1]['rmse']:.0f} MW")
        except Exception as e:
            print(f"   ⚠️  Could not evaluate LSTM: {e}")
            print("   Run: python train_pytorch.py")
    else:
        print(f"   ⚠️  No trained model found at {model_path}")
        print("   Run: python train_pytorch.py")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)

    # Header
    print(f"{'Model':<25s} {'MAE':>10s} {'RMSE':>10s} {'MAPE':>8s} {'R²':>8s} {'Time':>8s}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['model']:<25s} "
            f"{r['mae']:>8.0f} MW "
            f"{r['rmse']:>8.0f} MW "
            f"{r['mape']:>6.1f}% "
            f"{r['r2']:>7.3f} "
            f"{r['time_s']:>7.1f}s"
        )

    # Save
    os.makedirs("models", exist_ok=True)
    with open("models/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n✅ Saved to models/benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
