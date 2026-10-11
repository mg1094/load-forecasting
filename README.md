<p align="center">
  <h1 align="center">⚡ Load Forecasting with LSTM</h1>
  <p align="center">
    <strong>Production-grade electricity load forecasting — LSTM + PatchTST, pure PyTorch</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" alt="PyTorch">
    <img src="https://img.shields.io/badge/Model-LSTM-red.svg" alt="LSTM">
    <img src="https://img.shields.io/badge/Dataset-PJM-green.svg" alt="PJM">
    <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
    <img src="https://img.shields.io/github/stars/mg1094/load-forecasting.svg" alt="Stars">
  </p>
</p>

---

## 🎯 One-liner

**Day-ahead electricity load forecasting on real PJM data — quantile GBDT at 2.7% MAPE with calibrated prediction intervals, plus LSTM / PatchTST in pure PyTorch.**

---

## 🧠 Why this project

Most load forecasting tutorials use toy datasets and overfit immediately.
This project gives you:

- ✅ **Real data** — 35,061 hours of actual PJM East (PJME) load, 2014–2018, merged with real Philadelphia weather
- ✅ **Pure PyTorch** — clean, single-framework codebase
- ✅ **Production-ready** — early stopping, LR scheduling, checkpointing, scaler persistence
- ✅ **Multiple baselines** — persistence, ARIMA, Prophet, XGBoost vs LSTM vs PatchTST
- ✅ **Probabilistic forecasting** — quantile GBDT with pinball loss, P10–P90 intervals, conformal calibration
- ✅ **Full pipeline** — `fetch → preprocess → train → evaluate → predict`

---

## 📊 Benchmark

**Dataset**: real PJME hourly load + real Philadelphia weather, 35,061 h (2014-08 → 2018-08).
Chronological split — test set = last 20% (≈14 months). M1 MacBook Pro, CPU only.

### Day-ahead forecast (1–24h, same protocol for all models)

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | R² |
|-------|----------|-----------|----------|-----|
| **Persistence** (昨天同时刻) | 2,300 | 3,124 | 7.3% | 0.724 |
| **LSTM** (24h head) | 1,457 | 2,077 | 4.6% | 0.878 |
| **Quantile GBDT** (P50) | **859** | **1,161** | **2.7%** | **0.962** |
| **PatchTST + RevIN** | 1,554 | 2,167 | 4.9% | 0.867 |

> 分位数 GBDT 的 per-horizon MAE：t+1h 780 MW → t+24h 954 MW（误差随步长缓慢上升）。
> 这张表印证了电力预测比赛的长期经验：**GBDT + 特征工程仍是日前负荷的强基线**，
> 深度学习模型要赢它需要更好的结构与调参。
>
> PatchTST 那行的配置是 `--d-model 64 --n-layers 2 --batch-size 128`（327 万参数，
> 第 20 轮早停），**不是**脚本默认的 `d_model=128 / n_layers=3`（611 万参数）——
> 默认档在 M1 CPU 上单轮就要 20 分钟以上，没跑完。它可以复现：
> ```
> python train_patchtst.py --epochs 40 --batch-size 128 --d-model 64 --n-layers 2 --tag _small
> ```
> 它打赢了持续性基线，但**输给了 GBDT（859 MW）和 LSTM（1,457 MW）**。
> 这个结果不意外：在 3.5 万小时这种规模上，PatchTST 的展平头参数量（611 万）相对数据量偏大，
> 而论文里的优势场景本来就是更长的回看窗口和更大的数据量。想让 Transformer 赢，
> 该改的是结构（通道独立 + 小线性头），不是加训练轮数。

### Short-term forecast (1h ahead)

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | R² |
|-------|----------|-----------|----------|-----|
| **LSTM** (24h lookback → 1h) | **179** | **240** | **0.57%** | **0.998** |

> 1 小时超前本质上是在预测"上一小时负荷 + 小扰动"，所以指标天然远好于日前，
> 两个口径不可直接比较。

### Probabilistic forecast (day-ahead, quantile GBDT)

| Metric | Value |
|---|---|
| Mean pinball loss (9 quantiles) | 344 MW |
| P10–P90 coverage, raw | 60.5% |
| P10–P90 coverage, conformal-calibrated (k=2.13) | **83.4%** (nominal 80%) |
| Mean interval width | 1,949 MW → 4,150 MW after calibration |

> 独立训练的分位数模型区间系统性偏窄（欠覆盖），这是已知现象；
> `baseline_quantile.py` 在验证集上标定一个外扩系数做保形校准。
> 预测区间图见 `models/quantile_forecast.png`。

**⚠️ 天气用的是完美预报**：测试集里的温度与真实时刻对齐，而真实日前预测只能拿到
数值天气预报。所以上面这些数字应视为"协作变量理想"下的上限。

---

## 📁 Project Structure

```
load-forecasting/
├── data/
│   ├── pjm_real.csv            # Real PJME load + real weather (2014–2018)  ← 推荐
│   ├── pjm_load.csv            # Legacy synthetic data (2020–2023)
│   └── load_data.csv           # Original simple simulation
├── models/                     # Saved models + scalers + charts
├── fetch_real_pjm.py           # Download real PJME load + open-meteo weather
├── download_pjm.py             # Legacy synthetic data generator
├── data_generator.py           # Original data generator
├── data_preprocessor.py        # Feature engineering + sequence builder
├── lstm_model_pytorch.py       # LSTM model (short-term, 1-24h)
├── patchtst_model.py           # PatchTST model (long-term, RevIN + instance norm)
├── train_pytorch.py            # Train LSTM (--horizon 1 | 24)
├── train_patchtst.py           # Train PatchTST
├── predict_pytorch.py          # Run prediction
├── baseline.py                 # Persistence/ARIMA/Prophet/XGBoost vs LSTM
├── baseline_quantile.py        # Quantile GBDT: pinball loss + prediction intervals
├── requirements.txt
└── README.md
```

### Data

| 文件 | 来源 | 说明 |
|---|---|---|
| `pjm_real.csv` | PJME 小时负荷（Kaggle "Hourly Energy Consumption" 的公开镜像）+ open-meteo 费城历史天气 | 35,061 小时，2014-08 → 2018-08，负荷 19,255–56,609 MW |
| `pjm_load.csv` | 合成 | 正弦 + 噪声生成，**天气与负荷无真实因果**，仅供离线演示 |
| `load_data.csv` | 合成 | 最早期的最简模拟数据 |

> ⚠️ 合成数据的天气特征是独立随机生成的，模型学不到真实的负荷-天气耦合。
> 项目里所有对外可引用的指标都应基于 `pjm_real.csv`。

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- `pip install -r requirements.txt`

### 1. Get data (or use pre-generated)

```bash
# Real PJME load (Kaggle mirror) + real Philadelphia weather (open-meteo, no key)
python fetch_real_pjm.py            # → data/pjm_real.csv
python fetch_real_pjm.py --years 6  # custom span

# OR: legacy synthetic data (no network needed)
python download_pjm.py              # → data/pjm_load.csv
```

### 2. Train a model

```bash
# Short-term (1h ahead, 3min training)
python train_pytorch.py

# Day-ahead (1–24h, same protocol as the GBDT baseline)
python train_pytorch.py --horizon 24

# Long-term (24h ahead, 8min training, higher accuracy)
python train_patchtst.py
```

### 3. Run benchmarks

```bash
python baseline.py            # persistence / ARIMA / Prophet / XGBoost vs LSTM
python baseline_quantile.py   # quantile GBDT: pinball loss + P10–P90 intervals
```

### 4. Predict

```bash
python predict_pytorch.py
```

Output: prediction chart + metrics saved to `models/`.

---

## 🔧 Features

- Multi-variate input (load + real weather + calendar + holidays + degree days)
- Early stopping with patience-based termination
- LR scheduling (ReduceLROnPlateau)
- Model checkpointing (best val loss)
- tqdm progress bars during training
- Multi-step prediction (1h to 24h, both LSTM and PatchTST)
- RevIN (reversible instance normalization) in PatchTST
- Quantile GBDT baseline with pinball loss + conformal-calibrated intervals
- Training history plots (loss + MAE curves)
- Dual-model architecture: LSTM for short-term, PatchTST for long-term

---

## 📈 Model Architecture

```
Input: (batch, 24h, 15 features)
  ↓
LSTM Layer 1 (64 hidden, batch_first)
  ↓ Dropout(0.2)
LSTM Layer 2 (32 hidden)
  ↓ Dropout(0.2)
Linear(32 → prediction_steps)
  ↓
Output: (batch, 1 or 24)  ← next-hour or day-ahead load
```

14 input features (+ target `load`):
- `temperature`, `humidity`, `wind_speed`, `precipitation`
- `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos`
- `is_weekend`, `is_holiday`（真实美国联邦假日）
- `cdd`, `hdd`（度日电，制冷/采暖需求，基准 65°F）

PatchTST 额外带 **RevIN**（可逆实例归一化）：每个样本在进入 patch embedding 前做归一化、
预测头输出后再反归一化，缓解负荷序列的水平漂移。

---

## 🛠️ Evaluation Metrics

| Metric | Meaning | When to use |
|--------|---------|-------------|
| **MAE** | Average absolute error in MW | Intuitive, outlier-tolerant |
| **RMSE** | Root mean squared error in MW | Penalizes large errors |
| **R²** | Variance explained (0–1) | Overall goodness-of-fit |
| **MAPE** | Mean absolute % error | Compare across scales |

---

## 💡 Why this architecture works

Electricity load has 3 dominant patterns:

1. **Daily cycle** (24h) — morning ramp + evening peak
2. **Weekly cycle** (168h) — weekday vs weekend
3. **Seasonal cycle** (8,760h) — summer AC peak, winter heating peak

**LSTM** learns these via gating — selectively remembering and forgetting time steps.
Best for short-term (1–24h ahead) where local patterns dominate.

**PatchTST** learns these via attention — seeing the entire lookback window at once
through patch segmentation. Best for long-term (24h+ ahead) where global
structure matters more than local detail.

Choose based on your prediction horizon:
- `< 24h ahead` → LSTM (faster, simpler)
- `≥ 24h ahead` → PatchTST (more accurate, handles longer dependencies)

---

## 📝 Known Limitations

- 2-layer LSTM works well for 1–24h ahead; for >72h ahead, consider Transformer-based models (Informer, PatchTST)
- **天气用的是"完美预报"**：`pjm_real.csv` 里的温度和真实时刻对齐。真实部署里日前预测只能拿到
  数值天气预报（NWP），协变量误差会显著抬高误差——这是负荷预测的主要误差源之一
- 分位数模型的原始 P10–P90 区间偏窄（欠覆盖），脚本里用验证集保形校准（外扩系数 k）修正，
  但校准系数是全局的，没有按 horizon / 季节分别标定
- PatchTST 的预测头把 `n_vars × n_patches × d_model` 全展平（默认配置约 610 万参数），在 3.5 万
  小时数据上偏大：611 万参数档在 M1 CPU 上单轮超过 20 分钟，退到 327 万参数档跑完 20 轮后
  MAE 仍是 1,554 MW，**没打过 GBDT**。这条路要见效得改结构（通道独立 + 小线性头），
  而不是加参数或加轮数

---

## 🔜 Roadmap

- [x] Add PatchTST model (2024 SOTA for long-term forecasting)
- [x] Add ARIMA / Prophet / XGBoost baseline scripts
- [x] Add Quantile Loss for prediction intervals（`baseline_quantile.py` + 保形校准）
- [x] Add holiday calendar integration
- [x] Switch to real PJME load + real weather data
- [x] Add RevIN to PatchTST
- [ ] Add Streamlit Web Demo
- [ ] Add Optuna hyperparameter tuning
- [ ] Add CI/CD + tests
- [ ] 接入时序基础模型对照（Chronos-Bolt / TimesFM 零样本）

---

## 📄 License

MIT © [mg1094](https://github.com/mg1094)
