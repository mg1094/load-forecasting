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
