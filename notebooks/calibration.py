import polars as pl
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from Heston_Calibration_Class import Data_Class
from Heston_COS_METHOD import heston_cosine_method
from Levenberg_Marquardt import levenberg_Marquardt

def calibrate_one_date(part_dir: Path, output_dir: Path, initial_guess=None, N=100, L=10, I=500, w=1e-3, precision=0.01, params_to_calibrate=None, accel_mag=1, min_acc=1e-3):
    date = part_dir.name.split('=')[1]
    dfs = [pl.read_parquet(f) for f in part_dir.glob("*.parquet")]
    df_pl = pl.concat(dfs)
    S       = float(df_pl["S"][0])
    strikes = df_pl["strike"].to_numpy()
    T       = df_pl["T"].to_numpy()
    prices  = df_pl["close"].to_numpy()
    flags   = np.where(df_pl["option_type"].to_numpy() == "call", "c", "p")
    r       = float(df_pl["r"][0])
    q       = 0.0
    r_arr   = np.full_like(strikes, r, dtype=float)
    q_arr   = np.full_like(strikes, q, dtype=float)
    data = Data_Class()
    data.S             = S
    data.K             = strikes
    data.T             = T
    data.r             = r_arr
    data.q             = q_arr
    data.market_prices = prices
    data.flag          = flags
    data.calculate_implied_vol()
    market_vol = data.market_vol
    if initial_guess is None:
        initial_guess = np.array([0.04, 0.50, -0.70, 1.0, 0.04]).reshape(5,1)
    if params_to_calibrate is None:
        params_to_calibrate = ['vbar','sigma','rho','kappa','v0']
    calib_params, acc, rej, RMSE, lm_logs = levenberg_Marquardt(
        data, initial_guess, I, w, N, L, precision,
        params_to_calibrate, accel_mag, min_acc, return_logs=True
    )
    h_prices = heston_cosine_method(
        data.S, strikes, T, N, L, r_arr, q_arr,
        calib_params[0,0], calib_params[4,0],
        calib_params[1,0], calib_params[2,0],
        calib_params[3,0], flags
    )
    from py_vollib_vectorized import vectorized_implied_volatility as calculate_iv
    calib_iv = (
        calculate_iv(
            h_prices, S, strikes, T, r_arr, flags, q_arr,
            model='black_scholes_merton', return_as='numpy'
        ) * 100
    ).ravel()
    out = pl.DataFrame({
        "date": [date]*len(strikes),
        "strike": strikes,
        "T": T,
        "close": prices,
        "option_type": df_pl["option_type"].to_numpy(),
        "S": [S]*len(strikes),
        "r": [r]*len(strikes),
        "q": [q]*len(strikes),
        "market_vol": market_vol,
        "calib_iv": calib_iv,
        "calib_vbar": float(calib_params[0,0]),
        "calib_sigma": float(calib_params[1,0]),
        "calib_rho": float(calib_params[2,0]),
        "calib_kappa": float(calib_params[3,0]),
        "calib_v0": float(calib_params[4,0]),
        "calib_rmse": float(RMSE[-1]) if len(RMSE)>0 else np.nan,
        "calib_acc": int(acc),
        "calib_rej": int(rej),
    })
    out_dir = output_dir / f"date={date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_dir/"part-0.parquet", compression="snappy")
    print(f"✓ Finished calibration for {date}")
    return date, lm_logs

def parallel_calibrate(input_dir: Path, output_dir: Path, n_jobs: int = -1, **calib_kwargs):
    # Optional: filter by date range
    start_date = calib_kwargs.pop('start_date', None)
    end_date = calib_kwargs.pop('end_date', None)
    date_dirs = sorted(input_dir.glob("date=*"))
    if start_date or end_date:
        filtered = []
        for d in date_dirs:
            dstr = d.name.split('=')[1]
            try:
                from datetime import datetime
                ddate = datetime.strptime(dstr, "%Y-%m-%d").date()
                if (not start_date or ddate >= start_date) and (not end_date or ddate <= end_date):
                    filtered.append(d)
            except Exception:
                continue
        date_dirs = filtered
    output_dir.mkdir(parents=True, exist_ok=True)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(calibrate_one_date)(d, output_dir, **calib_kwargs) for d in date_dirs
    )
    print(f"\n✅ Calibrated {len(results)} dates.")
    return results
