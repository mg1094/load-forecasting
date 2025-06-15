import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class LoadDataGenerator:
    def __init__(self, start_date='2020-01-01', periods=8760, freq='H'):
        self.start_date = start_date
        self.periods = periods
        self.freq = freq
        
    def generate_base_load(self, dates):
        # 年度趋势
        annual_trend = np.linspace(1.0, 1.1, len(dates))
        
        # 季节性模式
        day_of_year = dates.dayofyear
        seasonal_pattern = 0.8 + 0.3 * (np.cos(2 * np.pi * (day_of_year - 180) / 365) + 1)
        
        # 周模式
        weekly_pattern = np.where(dates.weekday < 5, 1.1, 0.9)
        
        # 日内模式
        if self.freq == 'H':
            hour = dates.hour
            daily_pattern = (
                0.6 + 0.4 * np.sin(np.pi * (hour - 6) / 12) * (hour >= 6) * (hour <= 18)
                + 0.8 * (hour < 6) + 0.7 * (hour > 18)
            )
        else:
            daily_pattern = 1.0
            
        base_load = 1000
        return base_load * annual_trend * seasonal_pattern * weekly_pattern * daily_pattern
    
    def generate_weather_data(self, dates):
        day_of_year = dates.dayofyear
        temp_base = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temperature = temp_base + np.random.normal(0, 5, len(dates))
        
        humidity = 50 + 30 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 10, len(dates))
        humidity = np.clip(humidity, 0, 100)
        
        wind_speed = 5 + np.random.exponential(3, len(dates))
        precipitation = np.random.exponential(0.5, len(dates))
        
        return {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        }
    
    def generate_dataset(self):
        dates = pd.date_range(start=self.start_date, periods=self.periods, freq=self.freq)
        base_load = self.generate_base_load(dates)
        weather_data = self.generate_weather_data(dates)
        
        # 添加天气影响
        temp = weather_data['temperature']
        temp_effect = np.where(temp > 25, (temp - 25) * 0.02, np.where(temp < 5, (5 - temp) * 0.03, 0))
        final_load = base_load * (1 + temp_effect) + np.random.normal(0, base_load * 0.05)
        
        df = pd.DataFrame({
            'datetime': dates,
            'load': final_load,
            'temperature': weather_data['temperature'],
            'humidity': weather_data['humidity'],
            'wind_speed': weather_data['wind_speed'],
            'precipitation': weather_data['precipitation'],
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': dates.weekday >= 5
        })
        
        return df
    
    def save_data(self, df, filename='load_data.csv'):
        filepath = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"数据已保存到: {filepath}")
        print(f"数据形状: {df.shape}")
        return filepath

if __name__ == "__main__":
    generator = LoadDataGenerator(start_date='2020-01-01', periods=26280, freq='H')
    print("正在生成电力负荷数据...")
    df = generator.generate_dataset()
    generator.save_data(df)
    print("\n数据统计信息:")
    print(df.describe())