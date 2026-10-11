#!/usr/bin/env python3
"""
Fetch REAL PJM East (PJME) hourly load + Philadelphia weather.

Replaces the old synthetic "download_pjm.py" with actual data:
  - Load: PJME hourly MW, 2002–2018, from a public GitHub mirror of the
    Kaggle "Hourly Energy Consumption" dataset (CC0). PJM's official API
    requires a registered key, so we go through the mirror + Git LFS API.
  - Weather: open-meteo.com historical archive (free, no key) for
    Philadelphia — representative of the PJME mid-Atlantic zone.

Output: data/pjm_real.csv (same schema as pjm_load.csv).

Usage:
    python fetch_real_pjm.py            # last 4 years of available data
    python fetch_real_pjm.py --years 6  # custom span
"""

import argparse
import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd

# GitHub mirror of Kaggle "Hourly Energy Consumption" (PJME file, git-LFS)
MIRROR_REPO = "biancini/DataScience-HEP"
MIRROR_PATH = "Lezioni/data/PJME_hourly.csv"

# Weather location: Philadelphia (PJM East mid-Atlantic)
LAT, LON = 39.9526, -75.1652
TZ = "America/New_York"
WEATHER_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"


def http_get(url, headers=None, timeout=120):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout).read()


def download_pjme(dest):
    """Download PJME_hourly.csv via GitHub API + LFS batch endpoint."""
    if os.path.exists(dest):
        print(f"⏭  已存在，跳过下载: {dest}")
        return

    api = f"https://api.github.com/repos/{MIRROR_REPO}/contents/{MIRROR_PATH}"
    print("1. 📥 解析 GitHub LFS 指针…")
    ptr = http_get(api, headers={"Accept": "application/vnd.github.raw"}).decode()
    oid = size = None
    for line in ptr.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            size = int(line.split()[1])
    if not oid:
        raise RuntimeError(f"镜像返回的不是 LFS 指针，内容前 100 字符: {ptr[:100]}")

    print(f"   oid={oid[:12]}… 大小 {size/1e6:.1f} MB")
    batch = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
    }).encode()
    req = urllib.request.Request(
        f"https://github.com/{MIRROR_REPO}.git/info/lfs/objects/batch",
        data=batch,
        headers={
            "Content-Type": "application/vnd.git-lfs+json",
            "Accept": "application/vnd.git-lfs+json",
        },
    )
    resp = json.load(urllib.request.urlopen(req, timeout=60))
    url = resp["objects"][0]["actions"]["download"]["href"]

    print("2. ⬇️  下载 PJME 负荷数据…")
    data = http_get(url, timeout=300)
    if len(data) != size:
        raise RuntimeError(f"下载大小不符: {len(data)} != {size}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    print(f"✅ 已保存 {dest} ({len(data)/1e6:.1f} MB)")


def fetch_weather(start, end):
    """Hourly weather from open-meteo, chunked by year (API-friendly)."""
    print(f"3. 🌦  抓取费城历史天气 {start.date()} → {end.date()} …")
    frames = []
    for year in range(start.year, end.year + 1):
        ys, ye = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31, 23))
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}"
            f"&start_date={ys.date()}&end_date={ye.date()}"
            f"&hourly={WEATHER_VARS}&timezone={TZ}&temperature_unit=fahrenheit"
        )
        resp = json.loads(http_get(url, timeout=180))
        h = resp["hourly"]
        frames.append(pd.DataFrame({
            "datetime": pd.to_datetime(h["time"]),
            "temperature": h["temperature_2m"],
            "humidity": h["relative_humidity_2m"],
            "precipitation": h["precipitation"],
            "wind_speed": h["wind_speed_10m"],
        }))
        print(f"   {year}: {len(frames[-1])} 行")
    weather = pd.concat(frames, ignore_index=True)
    weather = weather.drop_duplicates(subset="datetime").sort_values("datetime")
    return weather


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=4, help="取数据末尾多少年")
    ap.add_argument("--raw", default="data/.PJME_hourly_raw.csv", help="原始负荷 CSV 缓存")
    args = ap.parse_args()

    download_pjme(args.raw)

    print("📊 读取负荷数据…")
    load = pd.read_csv(args.raw)
    load.columns = [c.strip() for c in load.columns]
    load = load.rename(columns={"PJME_MW": "load_mw"})
    load["datetime"] = pd.to_datetime(load["Datetime"])
    load = (load[["datetime", "load_mw"]]
            .dropna()
            .drop_duplicates(subset="datetime")
            .sort_values("datetime")
            .reset_index(drop=True))

    end = load["datetime"].max()
    start = end - pd.DateOffset(years=args.years)
    load = load[load["datetime"] >= start].reset_index(drop=True)
    print(f"   使用区间: {load['datetime'].min()} → {end}  ({len(load):,} 小时)")

    weather = fetch_weather(load["datetime"].min(), end)

    print("4. 🔗 合并负荷 + 天气…")
    df = load.merge(weather, on="datetime", how="left")
    n_missing = df["temperature"].isna().sum()
    if n_missing:
        print(f"   ⚠️  {n_missing} 小时缺天气，前值填充")
        df[["temperature", "humidity", "precipitation", "wind_speed"]] = \
            df[["temperature", "humidity", "precipitation", "wind_speed"]].ffill().bfill()

    print("5. 🗓  生成日历特征…")
    try:
        import holidays
        us_holidays = holidays.US(years=range(df["datetime"].dt.year.min(),
                                              df["datetime"].dt.year.max() + 1))
        df["is_holiday"] = df["datetime"].dt.date.map(lambda d: int(d in us_holidays))
    except ImportError:
        print("   ⚠️  holidays 未安装，is_holiday 全 0（pip install holidays）")
        df["is_holiday"] = 0

    dt = df["datetime"]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # 时间连续性检查（真实数据常因 DST/表计问题断点）
    gaps = df["datetime"].diff().dropna()
    n_gaps = int((gaps != pd.Timedelta(hours=1)).sum())
    if n_gaps:
        print(f"   ⚠️  发现 {n_gaps} 处非 1 小时间隔（夏令时切换等），已按索引连续处理")

    out = "data/pjm_real.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print(f"🎉 真实数据已保存: {out}")
    print(f"   行数: {len(df):,} | 负荷 {df['load_mw'].min():,.0f} – {df['load_mw'].max():,.0f} MW")
    print(f"   负荷-温度相关系数: {df['load_mw'].corr(df['temperature']):.3f}")
    print(f"   温度单位: 华氏度（CDD/HDD 基准 65°F）")


if __name__ == "__main__":
    main()
