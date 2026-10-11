#!/usr/bin/env python3
"""
PyTorch版本的LSTM负荷预测模型训练脚本
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessor import DataPreprocessor
from lstm_model_pytorch import LSTMLoadPredictor
import json
import torch

def main():
    parser = argparse.ArgumentParser(description="训练 LSTM 负荷预测模型")
    parser.add_argument("--horizon", type=int, default=1,
                        help="预测步数：1=短期(1h ahead，默认)，24=日前(1-24h，与 GBDT/PatchTST 同口径)")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    horizon = args.horizon

    print(f"🔥 PyTorch LSTM负荷预测模型训练 (horizon={horizon}h)")
    print("=" * 50)

    # 中文字体（macOS 自带），否则图上的中文标签会显示成方框
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS", "PingFang HK", "Heiti TC", "Hiragino Sans GB", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查数据文件（优先真实 PJM 数据，回退到合成数据）
    candidates = [
        ('data/pjm_real.csv', '真实 PJM 数据 (Kaggle 镜像 + open-meteo 天气)'),
        ('data/pjm_load.csv', 'PJM 风格合成数据'),
        ('data/load_data.csv', '原始模拟数据'),
    ]
    data_file = None
    for path, label in candidates:
        if os.path.exists(path):
            data_file = path
            print(f"📡 Using dataset: {path} —— {label}")
            break
    if data_file is None:
        print("数据文件不存在!")
        print("请先运行: python fetch_real_pjm.py")
        return
    
    # 1. 数据预处理
    print("\n1. 📊 数据预处理...")
    preprocessor = DataPreprocessor(sequence_length=24, prediction_steps=horizon)
    
    df = preprocessor.load_data(data_file)
    print(f"✅ 加载数据: {df.shape}")
    print(f"📅 时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    
    data = preprocessor.prepare_data(df, test_size=0.2, val_size=0.1)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    print(f"✅ 训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"✅ 验证集: X={X_val.shape}, y={y_val.shape}")
    print(f"✅ 测试集: X={X_test.shape}, y={y_test.shape}")
    
    tag = "" if horizon == 1 else f"_{horizon}h"
    preprocessor.save_scalers(f'models/scalers_pytorch{tag}.pkl')

    # 2. 构建模型
    print("\n2. 🤖 构建PyTorch LSTM模型...")
    predictor = LSTMLoadPredictor(
        sequence_length=24,
        n_features=X_train.shape[2],
        prediction_steps=horizon
    )
    
    # 构建模型
    predictor.build_model(
        hidden_sizes=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # 3. 训练模型
    print("\n3. 🚀 开始训练...")
    history = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,      # 减少轮数便于演示
        batch_size=32,
        patience=10,
        save_path=f'models/lstm_model_pytorch{tag}.pth'
    )
    
    # 4. 评估模型
    print("\n4. 📈 模型评估...")
    metrics, predictions, y_true = predictor.evaluate(X_test, y_test, preprocessor)
    
    # 5. 可视化结果
    print("\n5. 📊 生成可视化图表...")
    
    # 绘制训练历史
    predictor.plot_training_history()
    
    # 绘制预测结果
    plt.figure(figsize=(15, 10))
    n_samples = min(168, len(predictions))  # 显示一周数据
    
    # 子图1: 预测对比
    plt.subplot(3, 1, 1)
    plt.plot(y_true[:n_samples], label='实际负荷', linewidth=2, alpha=0.8)
    plt.plot(predictions[:n_samples], label='预测负荷', linewidth=2, alpha=0.8)
    plt.title('PyTorch LSTM 负荷预测结果对比', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (小时)')
    plt.ylabel('负荷 (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 误差分析
    plt.subplot(3, 1, 2)
    errors = predictions[:n_samples] - y_true[:n_samples]
    plt.plot(errors, color='red', alpha=0.7, linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('预测误差分析', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (小时)')
    plt.ylabel('误差 (MW)')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 散点图
    plt.subplot(3, 1, 3)
    plt.scatter(y_true[:n_samples], predictions[:n_samples], alpha=0.6, s=20)
    
    # 添加理想预测线
    min_val = min(y_true[:n_samples].min(), predictions[:n_samples].min())
    max_val = max(y_true[:n_samples].max(), predictions[:n_samples].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.title('实际值 vs 预测值', fontsize=14, fontweight='bold')
    plt.xlabel('实际负荷 (MW)')
    plt.ylabel('预测负荷 (MW)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'models/pytorch_prediction_results{tag}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. 保存结果
    print("\n6. 💾 保存结果...")
    with open(f'models/pytorch_evaluation_metrics{tag}.json', 'w') as f:
        json.dump({**metrics, 'horizon': horizon}, f, indent=2)

    # 保存训练历史
    with open(f'models/pytorch_training_history{tag}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("🎉 PyTorch LSTM 训练完成!")
    print("="*60)
    print(f"📁 模型文件: models/lstm_model_pytorch.pth")
    print(f"📁 预处理器: models/scalers_pytorch.pkl")
    print(f"📁 评估指标: models/pytorch_evaluation_metrics.json")
    print(f"📊 训练历史: models/pytorch_training_history.json")
    print(f"📈 可视化图表: models/pytorch_prediction_results.png")
    
    print(f"\n📊 最终性能指标:")
    print(f"🎯 RMSE: {metrics['RMSE']:.2f} MW")
    print(f"🎯 MAE:  {metrics['MAE']:.2f} MW")
    print(f"🎯 MAPE: {metrics['MAPE']:.2f}%")
    print(f"🎯 R²:   {metrics['R²']:.4f}")
    
    # 检查GPU使用情况
    if torch.cuda.is_available():
        print(f"\n🚀 使用了GPU加速: {torch.cuda.get_device_name()}")
    else:
        print(f"\n💻 使用CPU训练")
        
    print("\n✅ 可以开始使用模型进行预测了!")

if __name__ == "__main__":
    main()