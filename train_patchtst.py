#!/usr/bin/env python3
"""
Train PatchTST model for long-term load forecasting.

PatchTST excels at long-horizon prediction (72h+), complementing
the LSTM model which works best for short-term (1-24h).

Usage:
    python train_patchtst.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json

from data_preprocessor import DataPreprocessor
from patchtst_model import PatchTSTPredictor


def main():
    parser = argparse.ArgumentParser(description="训练 PatchTST 日前负荷预测模型")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--no-revin", action="store_true",
                        help="关闭 RevIN（用于消融对比）")
    parser.add_argument("--tag", default="", help="输出文件后缀，便于区分实验")
    args = parser.parse_args()

    print("🔥 PatchTST Long-Term Load Forecasting")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Data ──────────────────────────────────────────────────
    data_file = None
    for candidate in ["data/pjm_real.csv", "data/pjm_load.csv", "data/load_data.csv"]:
        if os.path.exists(candidate):
            data_file = candidate
            break
    if data_file is None:
        print("No data found. Run: python fetch_real_pjm.py")
        return
    print(f"   Dataset: {data_file}")

    # ── Preprocessing ─────────────────────────────────────────
    # PatchTST needs longer lookback for long-term prediction
    SEQ_LEN = 96    # 96 hours lookback (4 days)
    PRED_LEN = 24   # 24 hours ahead
    preprocessor = DataPreprocessor(sequence_length=SEQ_LEN, prediction_steps=PRED_LEN)

    print(f"\n1. 📊 Loading {data_file}...")
    # 走 preprocessor.load_data，统一列名（load_mw → load）
    df = preprocessor.load_data(data_file)
    print(f"   {len(df):,} rows, {df['datetime'].min()} → {df['datetime'].max()}")

    print(f"\n2. 🔧 Preprocessing (seq_len={SEQ_LEN}, pred_len={PRED_LEN})...")
    data = preprocessor.prepare_data(df, test_size=0.2, val_size=0.1)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    preprocessor.save_scalers(f"models/scalers_patchtst{args.tag}.pkl")

    # ── Build ─────────────────────────────────────────────────
    print("\n3. 🤖 Building PatchTST...")
    predictor = PatchTSTPredictor(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        n_vars=X_train.shape[2],
        use_revin=not args.no_revin,
    )
    predictor.build_model(
        d_model=args.d_model,
        n_heads=max(1, args.d_model // 16),
        n_layers=args.n_layers,
        patch_len=16,
        stride=8,
        dropout=0.1,
        learning_rate=0.001,
    )
    print(f"   RevIN: {'ON' if predictor.use_revin else 'OFF'}")

    # ── Train ─────────────────────────────────────────────────
    print("\n4. 🚀 Training...")
    history = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        save_path=f"models/patchtst_model{args.tag}.pth",
    )

    # ── Evaluate ──────────────────────────────────────────────
    print("\n5. 📈 Evaluating...")
    metrics, predictions, y_true = predictor.evaluate(X_test, y_test, preprocessor)

    # ── Plot ──────────────────────────────────────────────────
    print("\n6. 📊 Plotting...")
    predictor.plot_history()

    # Prediction vs actual
    n_show = min(168, len(predictions))
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.plot(y_true[:n_show], label="Actual", linewidth=2)
    plt.plot(predictions[:n_show], label="PatchTST", linewidth=2, alpha=0.8)
    plt.title(f"PatchTST: {PRED_LEN}h-Ahead Load Forecast", fontsize=14, fontweight="bold")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 1, 2)
    errors = predictions[:n_show] - y_true[:n_show]
    plt.plot(errors, color="red", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title("Prediction Error", fontsize=14, fontweight="bold")
    plt.ylabel("Error (MW)")
    plt.xlabel("Hour")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"models/patchtst_predictions{args.tag}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Save ──────────────────────────────────────────────────
    with open(f"models/patchtst_metrics{args.tag}.json", "w") as f:
        json.dump({**{k: float(v) for k, v in metrics.items()},
                   "revin": predictor.use_revin}, f, indent=2)

    print("\n" + "=" * 60)
    print("🎉 PatchTST training complete!")
    print(f"   RMSE: {metrics['RMSE']:.2f} MW")
    print(f"   MAE:  {metrics['MAE']:.2f} MW")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   R²:   {metrics['R²']:.4f}")
    print(f"   Model: models/patchtst_model.pth")


if __name__ == "__main__":
    main()
