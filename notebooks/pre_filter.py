# pre_filter.py
import os, shutil, stat
from pathlib import Path
import pandas as pd
import polars as pl
import yfinance as yf

def pre_filter_options(
    underlying: str,
    input_pattern: str,
    output_dir: str,
    min_window_start: int = 1,
    min_T_days: int = 7,
    max_T_days: int = 180,
    min_price: float = 0.1,
    min_volume: int = 0
):
    """
    Scan CSVs matching `input_pattern`, filter and enrich, and write out
    partitioned parquet to `output_dir`.

    - underlying: e.g. "SPXW", "TSLA", used in the ticker‐regex
    - input_pattern: glob like "…/spxw_filtered/*.csv"
    - output_dir: folder for date=… parquet slices
    - thresholds: drop anything below these
    """
    INPUT_PATTERN = input_pattern
    OUTPUT_DIR = Path(output_dir)

    # clean up old
    def _on_rm(func, path, exc): 
        os.chmod(path, stat.S_IWRITE); func(path)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, onerror=_on_rm)

    # derive date bounds
    pat = rf"O:{underlying}(\d{{6}})([CP])(\d{{8}})"
    # -- you must tell Polars that window_start really is Int64, etc. --
    dtypes = {
        "ticker": pl.Utf8,
        "volume": pl.Int64,
        "open":   pl.Float64,
        "close":  pl.Float64,
        "high":   pl.Float64,
        "low":    pl.Float64,
        "window_start": pl.Int64,
        "transactions": pl.Int64
    }
    lf = pl.scan_csv(INPUT_PATTERN, dtypes=dtypes).filter(pl.col("window_start") >= min_window_start)
    lf_dates = lf.with_columns([
        pl.col("ticker").str.extract(pat, 1)
            .str.strptime(pl.Date, "%y%m%d", strict=False)
            .alias("expiration"),
        pl.col("window_start").cast(pl.Datetime("ns")).dt.date().alias("date")
    ]).filter(pl.col("expiration").is_not_null())
    bounds = lf_dates.select([
        pl.col("date").min().alias("min"),
        pl.col("date").max().alias("max")
    ]).collect()
    min_date, max_date = bounds["min"][0], bounds["max"][0]

    # build S & r time series
    all_dates = pd.DataFrame({"date": pd.date_range(min_date, max_date).date})
    def fetch(ticker, colname, scale=1.0):
        try:
            pd_df = (
                yf.Ticker(ticker)
                  .history(start=min_date, end=max_date + pd.Timedelta(days=1))
                  ["Close"]
                  .reset_index()
                  .rename(columns={"Close": colname})
            )
            pd_df["date"] = pd_df["Date"].dt.date
            pd_df[colname] *= scale
            merged = all_dates.merge(pd_df[["date", colname]], on="date", how="left").ffill()
        except Exception as e:
            print(f"Warning: Could not fetch {ticker} due to {e}. Filling with NaN.")
            merged = all_dates.copy()
            merged[colname] = float('nan')
        return merged
    spx = pl.from_pandas(fetch("^GSPC", "S", 1.0))
    irx = pl.from_pandas(fetch("^IRX", "r", 1/100.0))
    # Ensure date columns are pl.Date for joins
    spx = spx.with_columns([pl.col("date").cast(pl.Date)])
    irx = irx.with_columns([pl.col("date").cast(pl.Date)])

    # final pipeline
    spx_lazy = spx.lazy(); irx_lazy = irx.lazy()
    lf = (
        pl.scan_csv(INPUT_PATTERN, dtypes=dtypes)
          .filter(pl.col("window_start") > min_window_start)
          .with_columns([
              # call/put
              pl.col("ticker")
                .str.extract(pat, 2)
                .replace({"C": "call", "P": "put"})
                .alias("option_type"),
              # expiration & date
              pl.col("ticker").str.extract(pat, 1)
                .str.strptime(pl.Date, "%y%m%d", strict=False)
                .alias("expiration"),
              pl.col("ticker").str.extract(pat, 3)
                .cast(pl.Int64).alias("strike_raw"),
              pl.col("window_start").cast(pl.Datetime("ns")).dt.date().alias("date")
          ])
          .filter(
              pl.col("expiration").is_not_null() &
              pl.col("strike_raw").is_not_null() &
              pl.col("date").is_not_null()
          )
          .with_columns([
              (pl.col("strike_raw")/1_000).alias("strike"),
              ((pl.col("expiration").cast(pl.Datetime("ns"))
                - pl.col("date").cast(pl.Datetime("ns")))
               .dt.total_days()/365.0).alias("T")
          ])
          .filter(
              (pl.col("T") >= min_T_days/365) &
              (pl.col("T") <= max_T_days/365) &
              (pl.col("close") > min_price) &
              (pl.col("volume") >= min_volume)
          )
          .join(spx_lazy, on="date", how="left")
          .join(irx_lazy, on="date", how="left")
          # put-call parity: only OTM
          .with_columns([
             (pl.col("S") * (pl.col("r") * pl.col("T")).exp()).alias("F")
          ])
          .filter(
             ((pl.col("option_type")=="call") & (pl.col("strike") >= pl.col("F"))) |
             ((pl.col("option_type")=="put")  & (pl.col("strike") <= pl.col("F")))
          )
          .drop("F")
          .select(["date","strike","T","close","option_type","S","r"])
    )
    df = lf.collect()
    df.write_parquet(
        OUTPUT_DIR,
        compression="snappy",
        partition_by="date"
    )
    print(f"✍️  wrote pre-filtered surface to {OUTPUT_DIR}")
