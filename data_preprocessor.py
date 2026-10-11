import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self, sequence_length=24, prediction_steps=1):
        """
        初始化数据预处理器
        
        Args:
            sequence_length: 输入序列长度（小时）
            prediction_steps: 预测步数
        """
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.load_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, filepath):
        """加载数据（兼容 load_mw / load 两种列名）"""
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        # 真实 PJM 数据用 load_mw，原始模拟数据用 load —— 统一成 load
        if 'load' not in df.columns and 'load_mw' in df.columns:
            df = df.rename(columns={'load_mw': 'load'})
        df = df.sort_values('datetime').reset_index(drop=True)
        return df

    @staticmethod
    def degree_day_base(temperature):
        """按单位猜度日基准：华氏用 65°F，摄氏用 18°C。"""
        median_temp = float(np.nanmedian(temperature))
        return 65.0 if median_temp > 45 else 18.0

    def create_time_features(self, df):
        """创建时间特征（含节假日与度日电）"""
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 度日电：刻画负荷-温度的 U 型关系（制冷/采暖需求）
        if 'temperature' in df.columns and 'cdd' not in df.columns:
            base = self.degree_day_base(df['temperature'])
            df['cdd'] = np.maximum(df['temperature'] - base, 0.0)
            df['hdd'] = np.maximum(base - df['temperature'], 0.0)

        # 节假日：数据里没有就现算
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = self._compute_holiday(df)

        return df

    @staticmethod
    def _compute_holiday(df):
        """用 holidays 库算美国联邦节假日；不可用时回退为 0。"""
        try:
            import holidays
            if 'datetime' not in df.columns:
                return 0
            years = range(int(df['datetime'].dt.year.min()), int(df['datetime'].dt.year.max()) + 1)
            us = holidays.US(years=list(years))
            return df['datetime'].dt.date.map(lambda d: int(d in us)).astype(int)
        except ImportError:
            return 0

    def prepare_features(self, df):
        """准备特征"""
        df = self.create_time_features(df)

        # 选择特征列：天气 + 日历 + 节假日 + 度日电
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'precipitation',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'is_weekend', 'is_holiday',
            'cdd', 'hdd',
        ]

        return df
    
    def create_sequences(self, data, target_col='load'):
        """创建序列数据"""
        sequences = []
        targets = []
        
        # 转换为numpy数组
        if isinstance(data, pd.DataFrame):
            data_values = data.values
            target_col_idx = data.columns.get_loc(target_col)
        else:
            data_values = data
            target_col_idx = 0
        
        for i in range(len(data_values) - self.sequence_length - self.prediction_steps + 1):
            # 输入序列
            seq = data_values[i:(i + self.sequence_length)]
            sequences.append(seq)
            
            # 目标值
            target_start = i + self.sequence_length
            target_end = target_start + self.prediction_steps
            target = data_values[target_start:target_end, target_col_idx]
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def fit_scalers(self, df):
        """拟合标准化器"""
        # 拟合负荷标准化器
        self.load_scaler.fit(df[['load']])
        
        # 拟合特征标准化器
        self.feature_scaler.fit(df[self.feature_columns])
        
    def transform_data(self, df):
        """转换数据"""
        df = df.copy()
        
        # 标准化负荷
        df['load'] = self.load_scaler.transform(df[['load']]).flatten()
        
        # 标准化特征
        df[self.feature_columns] = self.feature_scaler.transform(df[self.feature_columns])
        
        return df
    
    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        """准备训练数据"""
        # 准备特征
        df = self.prepare_features(df)
        
        # 划分训练集和测试集
        train_df, test_df = train_test_split(
            df, test_size=test_size, shuffle=False, random_state=42
        )
        
        # 从训练集中划分验证集
        train_df, val_df = train_test_split(
            train_df, test_size=val_size/(1-test_size), shuffle=False, random_state=42
        )
        
        # 拟合标准化器（只用训练数据）
        self.fit_scalers(train_df)
        
        # 标准化数据
        train_df = self.transform_data(train_df)
        val_df = self.transform_data(val_df)
        test_df = self.transform_data(test_df)
        
        # 选择建模用的列
        model_columns = ['load'] + self.feature_columns
        
        # 创建序列
        X_train, y_train = self.create_sequences(train_df[model_columns])
        X_val, y_val = self.create_sequences(val_df[model_columns])
        X_test, y_test = self.create_sequences(test_df[model_columns])
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_df': train_df, 'val_df': val_df, 'test_df': test_df
        }
    
    def inverse_transform_load(self, scaled_load):
        """反标准化负荷数据"""
        return self.load_scaler.inverse_transform(scaled_load.reshape(-1, 1)).flatten()
    
    def save_scalers(self, filepath='models/scalers.pkl'):
        """保存标准化器"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        scalers = {
            'load_scaler': self.load_scaler,
            'feature_scaler': self.feature_scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'prediction_steps': self.prediction_steps
        }
        joblib.dump(scalers, filepath)
        print(f"标准化器已保存到: {filepath}")
    
    def load_scalers(self, filepath='models/scalers.pkl'):
        """加载标准化器"""
        scalers = joblib.load(filepath)
        self.load_scaler = scalers['load_scaler']
        self.feature_scaler = scalers['feature_scaler']
        self.feature_columns = scalers['feature_columns']
        self.sequence_length = scalers['sequence_length']
        self.prediction_steps = scalers['prediction_steps']
        print(f"标准化器已从 {filepath} 加载")

if __name__ == "__main__":
    # 测试数据预处理
    preprocessor = DataPreprocessor(sequence_length=24, prediction_steps=1)
    
    # 检查数据文件是否存在
    if os.path.exists('data/load_data.csv'):
        df = preprocessor.load_data('data/load_data.csv')
        print(f"数据形状: {df.shape}")
        
        # 准备数据
        data = preprocessor.prepare_data(df)
        
        print(f"训练集形状: X_train={data['X_train'].shape}, y_train={data['y_train'].shape}")
        print(f"验证集形状: X_val={data['X_val'].shape}, y_val={data['y_val'].shape}")
        print(f"测试集形状: X_test={data['X_test'].shape}, y_test={data['y_test'].shape}")
        
        # 保存标准化器
        preprocessor.save_scalers()
    else:
        print("未找到数据文件，请先运行: python fetch_real_pjm.py")