import os
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessor import DataPreprocessor
from lstm_model import LSTMLoadPredictor
import json

def main():
    print("=== LSTM负荷预测模型训练 ===")
    
    data_file = 'data/load_data.csv'
    if not os.path.exists(data_file):
        print("请先运行: python data_generator.py")
        return
    
    # 数据预处理
    print("1. 数据预处理...")
    preprocessor = DataPreprocessor(sequence_length=24, prediction_steps=1)
    df = preprocessor.load_data(data_file)
    
    data = preprocessor.prepare_data(df)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    preprocessor.save_scalers()
    
    # 构建模型
    print("2. 构建LSTM模型...")
    predictor = LSTMLoadPredictor(sequence_length=24, n_features=X_train.shape[2], prediction_steps=1)
    predictor.build_lstm_model()
    
    # 训练
    print("3. 开始训练...")
    predictor.train(X_train, y_train, X_val, y_val, epochs=50, patience=10)
    
    # 评估
    print("4. 模型评估...")
    metrics, predictions, y_true = predictor.evaluate(X_test, y_test, preprocessor)
    
    # 可视化
    predictor.plot_training_history()
    
    print("训练完成!")
    print(f"RMSE: {metrics['RMSE']:.2f} MW")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

if __name__ == "__main__":
    main()