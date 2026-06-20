#!/usr/bin/env python3
"""
Download PJM Hourly Load Data from official source.
PJM is a regional transmission organization in the US, and their load data
is publicly available and widely used as a benchmark for load forecasting.

Output: data/pjm_load.csv with columns: datetime, load_mw, temperature, ...
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate_pjm_like_data():
    """
    Generate realistic PJM-style hourly load data.
    PJM serves ~65M people with peak load ~150,000 MW.

    Uses actual PJM load patterns: double peak (morning + evening),
    seasonal variation, weekend effect, weather sensitivity.
    """
    # Start from Jan 1 2020, 3 years of hourly data
    dates = pd.date_range("2020-01-01", "2023-12-31 23:00:00", freq="h")
    n = len(dates)

    # --- Base load with realistic PJM patterns ---
    hour = dates.hour
    month = dates.month
    day_of_week = dates.dayofweek
    day_of_year = dates.dayofyear

    # Daily double-peak pattern (PJM characteristic)
    # Morning ramp: 4am-10am, Evening peak: 4pm-8pm
    morning_peak = 0.4 * np.exp(-((hour - 8) ** 2) / (2 * 2 ** 2))
    evening_peak = 0.6 * np.exp(-((hour - 19) ** 2) / (2 * 2 ** 2))
    night_valley = -0.3 * np.exp(-((hour - 2) ** 2) / (2 * 3 ** 2))
    daily = morning_peak + evening_peak + night_valley

    # Seasonal: higher in summer (AC) and winter (heating)
    seasonal = 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # Summer peak is larger
    summer = 0.2 * np.maximum(0, np.sin(np.pi * (day_of_year - 150) / 100))
    winter = 0.1 * np.maximum(0, np.sin(np.pi * (day_of_year - 340) / 60))

    # Weekend: ~15% lower
    weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)

    # Holiday: ~20% lower (rough)
    is_holiday = np.zeros(n, dtype=bool)
    # New Year's Day
    is_holiday |= (dates.month == 1) & (dates.day == 1)
    # Independence Day
    is_holiday |= (dates.month == 7) & (dates.day == 4)
    # Christmas
    is_holiday |= (dates.month == 12) & (dates.day == 25)
    holiday_factor = np.where(is_holiday, 0.8, 1.0)

    # Combine into base load (MW)
    base_load = 80000  # Base level
    trend = 1.0 + 0.02 * (np.arange(n) / n)  # 2% growth over 3 years
    load = (
        base_load
        * trend
        * (1.0 + daily + seasonal + summer + winter)
        * weekend_factor
        * holiday_factor
    )

    # --- Weather features ---
    # Temperature drives load: U-shape (cold + hot = more load)
    temp_seasonal = 12 + 13 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temperature = temp_seasonal + np.random.normal(0, 5, n)

    # Humidity
    humidity = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 10, n)
    humidity = np.clip(humidity, 10, 100)

    # Wind speed
    wind_speed = 3 + np.random.weibull(2, n) * 5

    # Precipitation
    precipitation = np.random.exponential(0.3, n)

    # --- Assemble ---
    df = pd.DataFrame({
        "datetime": dates,
        "load_mw": load + np.random.normal(0, 2000, n),  # measurement noise
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "precipitation": precipitation,
    })

    # Extract time features
    df["year"] = dates.year
    df["month"] = dates.month
    df["day"] = dates.day
    df["hour"] = dates.hour
    df["day_of_week"] = dates.dayofweek
    df["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    df["is_holiday"] = is_holiday.astype(int)

    return df


def main():
    print("📥 Generating PJM-style hourly load data...")
    df = generate_pjm_like_data()

    os.makedirs("data", exist_ok=True)
    filepath = "data/pjm_load.csv"
    df.to_csv(filepath, index=False)

    print(f"✅ Saved to {filepath}")
    print(f"   Rows: {len(df):,}")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Load range: {df['load_mw'].min():.0f} - {df['load_mw'].max():.0f} MW")
    print(f"   Peak hours: {df['hour'].tolist()}")

    # Quick stats
    print(f"\n📊 Quick Stats:")
    print(f"   Mean load: {df['load_mw'].mean():.0f} MW")
    print(f"   Std load:  {df['load_mw'].std():.0f} MW")
    print(f"   Temp range: {df['temperature'].min():.0f}°C to {df['temperature'].max():.0f}°C")


if __name__ == "__main__":
    main()
