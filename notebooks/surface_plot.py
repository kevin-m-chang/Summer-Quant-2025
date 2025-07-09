import numpy as np
import polars as pl
from pathlib import Path
from Heston_Calibration_Class import Data_Class
import os

def plot_surface_for_date(date: str, calib_dir: Path, out_dir: Path):
    day_dir = calib_dir / f"date={date}"
    files = list(day_dir.glob("*.parquet"))
    if not files:
        print(f"[Surface] {date} → no data, skipping")
        return False
    df = pl.concat([pl.read_parquet(p) for p in files]).to_pandas()
    if df.empty:
        print(f"[Surface] {date} → empty DataFrame, skipping")
        return False
    data = Data_Class()
    data.S             = float(df["S"].iloc[0])
    data.K             = df["strike"].values.astype(float)
    data.T             = df["T"].values.astype(float)
    data.r             = np.full_like(data.K, float(df["r"].iloc[0]), dtype=float)
    data.q             = np.zeros_like(data.K, dtype=float)
    data.market_prices = df["close"].values.astype(float)
    data.flag          = np.where(df["option_type"]=="call","c","p")
    data.calculate_implied_vol()
    calib_iv = df["calib_iv"].values.astype(float) if "calib_iv" in df.columns else None
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(str(out_dir))
    try:
        data.plot_save_surface(calib_iv, date_today=date)
        print(f"[Surface] {date} → saved {date}.png in {out_dir}")
    finally:
        os.chdir(prev_cwd)
    return True
