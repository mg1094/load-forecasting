#!/usr/bin/env python3
"""
Quantile GBDT baseline for day-ahead load forecasting.

这是电力负荷预测比赛（GEFCom2014/2017、Kaggle、IISE 2025）验证过的主线做法：
梯度提升树 + 分位数损失 + 日历/温度/滞后特征，输出预测区间而不只是点预测。

模型用 sklearn 的 HistGradientBoostingRegressor(loss="quantile")，
与 LightGBM 同族（直方图 GBDT），但不需要额外的 native 依赖。

评测口径：
  - 点预测 = P50（中位数）
  - 概率预测 = 9 个分位点，用 pinball loss 打分（GEFCom 官方口径）
  - 区间 = P10–P90，报告覆盖率

Usage:
    python baseline_quantile.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HORIZONS = range(1, 25)          # 预测未来 1–24 小时（日前场景）
DATA_FILES = ["data/pjm_real.csv", "data/pjm_load.csv", "data/load_data.csv"]


def load_dataset():
    for f in DATA_FILES:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df["datetime"] = pd.to_datetime(df["datetime"])
            if "load" not in df.columns and "load_mw" in df.columns:
                df = df.rename(columns={"load_mw": "load"})
            df = df.sort_values("datetime").reset_index(drop=True)
            print(f"✅ 数据集: {f} ({len(df):,} 小时, {df['datetime'].min()} → {df['datetime'].max()})")
            return df, f
    raise FileNotFoundError("找不到数据，请先运行 fetch_real_pjm.py")


def build_features(df):
    """构造监督学习表：对每个起点 t，预测 t+1 … t+24。"""
    df = df.copy()

    # —— 起点可用的负荷历史（滞后 / 滚动统计）——
    df["lag_1"] = df["load"].shift(1)
    df["lag_2"] = df["load"].shift(2)
    df["lag_24"] = df["load"].shift(24)
    df["lag_48"] = df["load"].shift(48)
    df["lag_168"] = df["load"].shift(168)
    df["roll_mean_24"] = df["load"].shift(1).rolling(24).mean()
    df["roll_mean_168"] = df["load"].shift(1).rolling(168).mean()
    df["roll_std_24"] = df["load"].shift(1).rolling(24).std()

    # —— 度日电（U 型温度响应）——
    if "cdd" not in df.columns or "hdd" not in df.columns:
        base = 65.0 if np.nanmedian(df["temperature"]) > 45 else 18.0
        df["cdd"] = np.maximum(df["temperature"] - base, 0.0)
        df["hdd"] = np.maximum(base - df["temperature"], 0.0)

    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    # —— 目标时刻的日历特征（已知）——
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    origin_cols = ["lag_1", "lag_2", "lag_24", "lag_48", "lag_168",
                   "roll_mean_24", "roll_mean_168", "roll_std_24"]
    target_time_cols = ["hour", "day_of_week", "month", "is_weekend", "is_holiday"]
    weather_cols = [c for c in ["temperature", "cdd", "hdd", "humidity", "wind_speed"] if c in df.columns]

    rows = []
    load = df["load"].values
    n = len(df)
    max_lag = 168
    for h in HORIZONS:
        idx = np.arange(max_lag, n - h)          # 起点 t，目标 t+h
        block = df.iloc[idx]
        X = pd.DataFrame({
            "horizon": h,
            "origin_hour": block["hour"].values,
            **{c: block[c].values for c in origin_cols},
        })
        target = block.shift(-h)
        for c in target_time_cols + weather_cols:
            X[f"t_{c}"] = target[c].values
        X["y"] = load[idx + h]
        X["origin_idx"] = idx
        rows.append(X)

    data = pd.concat(rows, ignore_index=True).dropna().reset_index(drop=True)
    feature_cols = [c for c in data.columns if c not in ("y", "origin_idx")]
    return data, feature_cols


def pinball_loss(y_true, y_pred, q):
    """GEFCom 官方评分口径的分位数损失。"""
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def calibrate_width(p50, p10, p90, y, target=0.8, grid=None):
    """在验证集上找区间外扩系数 k，使 P10–P90 覆盖率逼近 nominal。

    独立训练的分位数 GBDT 往往区间过窄（欠覆盖）；按中位数两侧等比例外扩
    是最简单有效的保形校准（conformal 思想）。
    """
    if grid is None:
        grid = np.arange(1.0, 3.01, 0.01)
    best_k, best_gap = 1.0, 1e9
    for k in grid:
        lo = p50 - k * (p50 - p10)
        hi = p50 + k * (p90 - p50)
        cov = np.mean((y >= lo) & (y <= hi))
        gap = abs(cov - target)
        if gap < best_gap:
            best_gap, best_k = gap, float(k)
    return best_k


def main():
    print("=" * 66)
    print("📊 分位数 GBDT 基线（日前 1–24h，概率预测）")
    print("=" * 66)

    df, data_file = load_dataset()
    print("\n1. 🔧 构造特征…")
    data, feature_cols = build_features(df)
    print(f"   {len(data):,} 条样本 × {len(feature_cols)} 个特征")

    # —— 按时间切分（同一起点的 24 个 horizon 必须落在同一侧，避免泄漏）——
    origin_max = data["origin_idx"].max()
    tr_end = int(origin_max * 0.7)
    val_end = int(origin_max * 0.8)
    tr = data[data["origin_idx"] < tr_end]
    va = data[(data["origin_idx"] >= tr_end) & (data["origin_idx"] < val_end)]
    te = data[data["origin_idx"] >= val_end]
    print(f"   训练 {len(tr):,} | 验证 {len(va):,} | 测试 {len(te):,}")

    X_tr, y_tr = tr[feature_cols].values, tr["y"].values
    X_va, y_va = va[feature_cols].values, va["y"].values
    X_te, y_te = te[feature_cols].values, te["y"].values

    print("\n2. 🤖 训练分位数 GBDT（9 个分位点）…")
    preds, val_preds = {}, {}
    t0 = time.time()
    for q in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=400, learning_rate=0.06,
            max_depth=None, max_leaf_nodes=63, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        preds[q] = model.predict(X_te)
        if q in (0.1, 0.5, 0.9):          # 校准只需要这三条
            val_preds[q] = model.predict(X_va)
        print(f"   q={q:.1f} 完成")
    train_time = time.time() - t0

    # —— 点预测 = P50 ——
    pred_median = preds[0.5]

    # —— 持续性基线：目标时刻的"昨天同一时刻" = y(t+h-24) ——
    # 注意不能拿起点(t)的 lag_24 当预测，那是 (t-24) 对 (t+h)，跨了 24+h 小时，不公平。
    load_series = df["load"].values
    target_idx = te["origin_idx"].values + te["horizon"].values
    pers_shift = load_series[target_idx - 24]

    def mape(y, p):
        mask = y != 0
        return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)

    results = {}
    results["persistence"] = {
        "MAE": float(mean_absolute_error(y_te, pers_shift)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pers_shift))),
        "MAPE": mape(y_te, pers_shift),
        "R2": float(r2_score(y_te, pers_shift)),
    }
    results["quantile_gbdt"] = {
        "MAE": float(mean_absolute_error(y_te, pred_median)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred_median))),
        "MAPE": mape(y_te, pred_median),
        "R2": float(r2_score(y_te, pred_median)),
        "train_time_s": float(train_time),
    }

    # —— 概率预测指标 ——
    pinball = {str(q): float(pinball_loss(y_te, preds[q], q)) for q in QUANTILES}
    mean_pinball = float(np.mean(list(pinball.values())))

    lo_raw, hi_raw = preds[0.1], preds[0.9]
    coverage_raw = float(np.mean((y_te >= lo_raw) & (y_te <= hi_raw)))
    width_raw = float(np.mean(hi_raw - lo_raw))

    # —— 保形校准：验证集定外扩系数，测试集评估 ——
    k = calibrate_width(val_preds[0.5], val_preds[0.1], val_preds[0.9], y_va, target=0.8)
    p50 = preds[0.5]
    lo_cal = p50 - k * (p50 - preds[0.1])
    hi_cal = p50 + k * (preds[0.9] - p50)
    coverage_cal = float(np.mean((y_te >= lo_cal) & (y_te <= hi_cal)))
    width_cal = float(np.mean(hi_cal - lo_cal))

    print("\n" + "=" * 66)
    print("📊 结果（测试集，真实 PJME 数据）")
    print("=" * 66)
    print(f"{'模型':<28s} {'MAE':>9s} {'RMSE':>9s} {'MAPE':>8s} {'R²':>7s}")
    print("-" * 66)
    for name, key in [("持续性 (lag-24)", "persistence"), ("分位数 GBDT (P50)", "quantile_gbdt")]:
        r = results[key]
        print(f"{name:<28s} {r['MAE']:>7.0f}MW {r['RMSE']:>7.0f}MW {r['MAPE']:>6.1f}% {r['R2']:>7.3f}")

    print(f"\n概率预测: mean pinball = {mean_pinball:,.0f} MW")
    print(f"          P10–P90 覆盖率 = {coverage_raw:.1%} （目标 80%），平均宽度 {width_raw:,.0f} MW")
    print(f"    校准后 (k={k:.2f}) 覆盖率 = {coverage_cal:.1%}，平均宽度 {width_cal:,.0f} MW")
    print(f"训练耗时: {train_time:.0f}s（9 个模型）")

    # —— 分 horizon 的误差（日前预测的关键视角）——
    print("\n分 horizon MAE（MW）:")
    per_h = {}
    for h in HORIZONS:
        mask = te["horizon"].values == h
        per_h[h] = {
            "gbdt": float(mean_absolute_error(y_te[mask], pred_median[mask])),
            "persistence": float(mean_absolute_error(y_te[mask], pers_shift[mask])),
        }
    for h in [1, 4, 8, 12, 18, 24]:
        print(f"   t+{h:>2d}h: GBDT {per_h[h]['gbdt']:>7.0f} | 持续性 {per_h[h]['persistence']:>7.0f}")

    # —— 出图 ——
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 中文字体（macOS 自带），否则标签会显示成方框
        plt.rcParams["font.sans-serif"] = [
            "Arial Unicode MS", "PingFang HK", "Heiti TC", "Hiragino Sans GB", "DejaVu Sans"
        ]
        plt.rcParams["axes.unicode_minus"] = False

        os.makedirs("models", exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))

        # 上：测试集最后一周的预测区间（固定 horizon=24 的样本，即"日前 24h"视图）
        mask_h24 = te["horizon"].values == 24
        n_show = min(168, mask_h24.sum())
        idx_h24 = np.where(mask_h24)[0][-n_show:]
        t = np.arange(n_show)
        axes[0].fill_between(t, lo_cal[idx_h24], hi_cal[idx_h24],
                             alpha=0.25, color="tab:orange", label="P10–P90 (校准后)")
        axes[0].plot(t, y_te[idx_h24], label="实际负荷", lw=2, color="black")
        axes[0].plot(t, pred_median[idx_h24], label="GBDT P50", lw=2, color="tab:blue")
        axes[0].set_title(f"日前 24h 预测：测试集最后 {n_show} 天（覆盖率 "
                          f"{np.mean((y_te[idx_h24] >= lo_cal[idx_h24]) & (y_te[idx_h24] <= hi_cal[idx_h24])):.0%}）")
        axes[0].set_ylabel("负荷 (MW)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 下：分 horizon MAE
        hs = list(HORIZONS)
        axes[1].plot(hs, [per_h[h]["gbdt"] for h in hs], marker="o", label="Gradient Boosting (P50)")
        axes[1].plot(hs, [per_h[h]["persistence"] for h in hs], marker="s",
                     linestyle="--", label="Persistence (昨天同时刻)")
        axes[1].set_title("分 horizon MAE")
        axes[1].set_xlabel("预测步长 (小时)")
        axes[1].set_ylabel("MAE (MW)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xticks([1, 4, 8, 12, 16, 20, 24])

        plt.tight_layout()
        plt.savefig("models/quantile_forecast.png", dpi=150, bbox_inches="tight")
        print("📈 已保存 models/quantile_forecast.png")
    except Exception as e:
        print(f"⚠️  出图失败（不影响指标）: {e}")

    os.makedirs("models", exist_ok=True)
    with open("models/quantile_benchmark.json", "w") as f:
        json.dump({
            "dataset": data_file,
            "point": results,
            "pinball_by_quantile": pinball,
            "mean_pinball": mean_pinball,
            "interval": {
                "nominal": 0.8,
                "coverage_raw": coverage_raw,
                "width_raw": width_raw,
                "calibration_k": k,
                "coverage_calibrated": coverage_cal,
                "width_calibrated": width_cal,
            },
            "per_horizon_mae": per_h,
        }, f, indent=2)
    print("\n✅ 已保存 models/quantile_benchmark.json")


if __name__ == "__main__":
    main()
