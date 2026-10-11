#!/usr/bin/env python3
"""
PyTorch版本的LSTM负荷预测脚本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessor import DataPreprocessor
from lstm_model_pytorch import LSTMLoadPredictor
import torch
import os

def predict_future_load(hours_ahead=24):
    """
    预测未来负荷
    
    Args:
        hours_ahead: 预测未来多少小时
    """
    print(f"🔮 PyTorch LSTM 负荷预测 - 预测未来 {hours_ahead} 小时")
    print("=" * 60)
    
    # 1. 加载数据和预处理器
    print("1. 📊 加载数据和预处理器...")
    preprocessor = DataPreprocessor(sequence_length=24, prediction_steps=1)
    
    # 加载原始数据（优先真实 PJM 数据）
    for _cand in ['data/pjm_real.csv', 'data/pjm_load.csv', 'data/load_data.csv']:
        if os.path.exists(_cand):
            break
    df = preprocessor.load_data(_cand)
    print(f"✅ 原始数据: {_cand} {df.shape}")
    
    # 加载预处理器
    preprocessor.load_scalers('models/scalers_pytorch.pkl')
    print("✅ 预处理器加载完成")
    
    # 2. 加载训练好的模型
    print("\n2. 🤖 加载PyTorch模型...")
    predictor = LSTMLoadPredictor(
        sequence_length=preprocessor.sequence_length, 
        n_features=len(preprocessor.feature_columns) + 1, 
        prediction_steps=preprocessor.prediction_steps
    )
    predictor.load_model('models/lstm_model_pytorch.pth')
    print("✅ 模型加载完成")
    
    # 3. 准备预测输入
    print(f"\n3. 🔍 准备预测数据 (使用最近 {preprocessor.sequence_length} 小时数据)...")
    
    # 准备特征
    df_with_features = preprocessor.prepare_features(df)
    
    # 标准化数据
    df_scaled = preprocessor.transform_data(df_with_features)
    
    # 获取最后的序列
    model_columns = ['load'] + preprocessor.feature_columns
    last_sequence_scaled = df_scaled[model_columns].tail(preprocessor.sequence_length).values
    
    input_sequence = last_sequence_scaled.reshape(1, preprocessor.sequence_length, len(model_columns))
    print(f"✅ 输入序列形状: {input_sequence.shape}")
    
    # 4. 进行预测
    print(f"\n4. 🚀 开始预测未来 {hours_ahead} 小时...")
    
    predictions_scaled = []
    current_sequence = torch.FloatTensor(input_sequence).to(predictor.device)
    
    last_known_datetime = df['datetime'].iloc[-1]
    
    for i in range(hours_ahead):
        with torch.no_grad():
            pred = predictor.model(current_sequence)
        
        pred_value = pred.item()
        predictions_scaled.append(pred_value)
        
        # 构造下一时间步的特征
        next_datetime = last_known_datetime + pd.Timedelta(hours=i + 1)
        
        # 创建一个包含下一时间步的单行DataFrame
        # 注意：天气用最后一个已知值近似（真实部署应换成数值天气预报）
        last_temp = df_with_features['temperature'].iloc[-1]
        next_step_df = pd.DataFrame([{
            'datetime': next_datetime,
            'hour': next_datetime.hour,
            'day_of_week': next_datetime.dayofweek,
            'month': next_datetime.month,
            'is_weekend': next_datetime.dayofweek >= 5,
            'temperature': last_temp,
            'humidity': df_with_features['humidity'].iloc[-1],
            'wind_speed': df_with_features['wind_speed'].iloc[-1],
            'precipitation': df_with_features['precipitation'].iloc[-1],
        }])
        
        # 创建时间特征
        next_step_df = preprocessor.create_time_features(next_step_df)
        
        # 标准化新特征
        next_features_scaled = preprocessor.feature_scaler.transform(next_step_df[preprocessor.feature_columns])
        
        # 组合成新的一行 (load + features)
        new_row = np.concatenate(([pred_value], next_features_scaled.flatten()))
        new_row = new_row.reshape(1, 1, -1)
        new_row_tensor = torch.FloatTensor(new_row).to(predictor.device)
        
        # 滑动窗口
        current_sequence = torch.cat((current_sequence[:, 1:, :], new_row_tensor), dim=1)
        
        if (i + 1) % 6 == 0:
            print(f"  ✅ 已预测 {i + 1} 小时")
    
    # 5. 反标准化预测结果
    print("\n5. 📈 处理预测结果...")
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions_original = preprocessor.inverse_transform_load(predictions_scaled)
    
    # 获取最近的真实数据用于对比
    recent_hours = min(hours_ahead * 3, 168) # 显示更多历史
    recent_data = df.tail(recent_hours)
    
    # 6. 可视化结果
    print("\n6. 📊 生成预测图表...")
    
    plt.figure(figsize=(15, 8))
    
    # 创建时间轴
    future_times = pd.date_range(start=last_known_datetime + pd.Timedelta(hours=1), 
                                periods=hours_ahead, freq='H')
    
    # 绘制历史数据
    plt.plot(pd.to_datetime(recent_data['datetime']), recent_data['load'], 
             label='历史负荷', linewidth=2, alpha=0.8, color='blue')
    
    # 绘制预测数据
    plt.plot(future_times, predictions_original, 
             label=f'预测负荷 ({hours_ahead}小时)', linewidth=2, alpha=0.8, color='red', marker='o', markersize=4)
    
    # 添加分界线
    plt.axvline(x=last_known_datetime, color='green', linestyle='--', alpha=0.7, 
                label='预测起点')
    
    plt.title(f'PyTorch LSTM 负荷预测 - 未来 {hours_ahead} 小时', fontsize=16, fontweight='bold')
    plt.xlabel('时间')
    plt.ylabel('负荷 (MW)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'models/pytorch_prediction_{hours_ahead}h.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 输出预测结果
    print(f"\n7. 📋 预测结果摘要:")
    print("=" * 40)
    print(f"📅 预测时间范围: {future_times[0]} 到 {future_times[-1]}")
    print(f"🔢 预测点数: {len(predictions_original)}")
    print(f"📊 平均预测负荷: {predictions_original.mean():.2f} MW")
    print(f"📈 最大预测负荷: {predictions_original.max():.2f} MW")
    print(f"📉 最小预测负荷: {predictions_original.min():.2f} MW")
    
    # 保存预测结果到CSV
    result_df = pd.DataFrame({
        'datetime': future_times,
        'predicted_load': predictions_original
    })
    
    result_file = f'models/pytorch_predictions_{hours_ahead}h.csv'
    result_df.to_csv(result_file, index=False)
    print(f"\n💾 预测结果已保存到: {result_file}")
    
    return predictions_original, future_times

def main():
    """主函数"""
    try:
        predict_future_load(hours_ahead=24)
    except KeyboardInterrupt:
        print("\n\n👋 预测已取消")
    except Exception as e:
        print(f"\n❌ 预测过程中出现错误: {e}")
        print("请检查模型和预处理器文件是否存在。")

if __name__ == "__main__":
    main()