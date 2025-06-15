# 电力负荷预测 - LSTM模型

基于LSTM深度学习模型的电力负荷预测系统。

## 🔥 PyTorch版本（推荐）

现在提供了PyTorch版本，具有更直观的语法和更好的调试体验！

## 项目结构
```
load_forecasting/
├── data/                    # 数据文件夹
├── models/                  # 保存的模型
├── utils/                   # 工具函数
├── notebooks/               # Jupyter笔记本
├── data_generator.py        # 数据生成器
├── data_preprocessor.py     # 数据预处理
├── lstm_model.py           # LSTM模型定义 (TensorFlow)
├── lstm_model_pytorch.py   # LSTM模型定义 (PyTorch)
├── train.py                # 训练脚本 (TensorFlow)
├── train_pytorch.py        # 训练脚本 (PyTorch)
├── predict.py              # 预测脚本 (TensorFlow)
├── predict_pytorch.py      # 预测脚本 (PyTorch)
└── requirements.txt        # 依赖包
```

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 生成示例数据
```bash
python data_generator.py
```

3. 训练模型 (PyTorch版本)
```bash
python train_pytorch.py
```

4. 或使用TensorFlow版本
```bash
python train.py
```

5. 进行预测
```bash
# 使用PyTorch版本
python predict_pytorch.py
```

## 功能特点

- 🔥 基于LSTM的深度学习模型
- 🚀 支持TensorFlow和PyTorch双框架
- 📊 自动数据预处理和特征工程
- 📈 多种评估指标 (RMSE, MAE, R², MAPE)
- 🎯 支持多步预测
- 📋 详细的可视化分析
- ⚡ 模型保存和加载

## 评估指标说明

本项目使用以下核心指标来评估模型的性能：

| 指标 | 全称 (英文) | 公式 | 解释 |
| :--- | :--- | :--- | :--- |
| **MAE** | Mean Absolute Error | `(1/n) Σ|y - ŷ|` | **平均绝对误差**：预测值与真实值之间绝对误差的平均值。单位与原数据相同 (MW)，直观反映平均预测误差的大小。对异常值不敏感。 |
| **RMSE** | Root Mean Squared Error | `sqrt((1/n) Σ(y - ŷ)²)` | **均方根误差**：均方误差的平方根。单位与原数据相同 (MW)，但由于对误差进行了平方，它会放大较大误差的惩罚，对异常值更敏感。 |
| **R²** | Coefficient of Determination | `1 - (Σ(y - ŷ)²) / (Σ(y - ȳ)²) ` | **决定系数**：表示模型对数据方差的解释程度。值越接近1，说明模型的拟合优度越好。0表示模型等同于用均值进行预测。 |
| **MAPE** | Mean Absolute Percentage Error | `(100/n) Σ|(y - ŷ) / y|` | **平均绝对百分比误差**：将误差转换为百分比形式，更便于跨不同量级的数据进行比较。但当真实值接近0时该指标可能失效。 |

*其中 `y` 是真实值, `ŷ` 是预测值, `ȳ` 是真实值的平均值, `n` 是样本数量。*

## 模型特性

- 支持多变量时间序列预测
- 自动处理季节性和趋势
- 包含天气、时间等外部特征
- 支持不同时间窗口的预测