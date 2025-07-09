import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from py_vollib_vectorized import vectorized_implied_volatility as calculate_iv

def hybrid_kmeans_pca_atm(
    df: pd.DataFrame,
    n_clusters: int = 200,
    n_pca_extremes: int = 0,
    atm_width: float = 0.05,
    atm_frac: float = 0.20,
    num_m_bins: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    df = df.copy()
    df['days']      = (df['T'] * 365).round().astype(int)
    S, r            = df['S'].iloc[0], df['r'].iloc[0]
    q               = df.get('q', 0.0)
    df['moneyness'] = df['strike'] / S
    flags = np.where(df['option_type']=='call','c','p')
    iv = calculate_iv(
        df['close'].values, S, df['strike'].values, df['T'].values,
        r, flags, q, model='black_scholes_merton', return_as='numpy'
    )/100.0
    iv = np.nan_to_num(iv, nan=1e-6)
    sqrtT = np.sqrt(df['T'])
    d1    = (
        np.log(S/df['strike']) +
        (r - q + 0.5*iv**2)*df['T']
    )/(iv*sqrtT + 1e-12)
    df['vega'] = S * np.exp(-q*df['T']) * sqrtT * np.exp(-0.5*d1**2)/np.sqrt(2*np.pi)
    max_days     = df['days'].max() or 1
    df['m_rank'] = df['moneyness'].rank(pct=True)
    df['d_norm'] = df['days'] / max_days
    X  = np.vstack([df['m_rank'], df['d_norm']]).T
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster'] = km.fit_predict(X)
    idx_k        = df.groupby('cluster')['vega'].idxmax().values
    samp_k       = df.loc[idx_k]
    if n_pca_extremes > 0:
        pca     = PCA(n_components=1, random_state=random_state)
        df['pc1'] = pca.fit_transform(X).ravel()
        half     = n_pca_extremes//2
        samp_pc  = pd.concat([
            df.nlargest(half, 'pc1'),
            df.nsmallest(half, 'pc1')
        ])
    else:
        samp_pc = df.head(0)
    combined = pd.concat([samp_k, samp_pc]) \
                 .drop_duplicates(subset=['strike','T','option_type'])
    total_target = n_clusters + n_pca_extremes
    atm_target   = int(atm_frac * total_target)
    in_atm       = combined[np.abs(combined['moneyness']-1) <= atm_width]
    if len(in_atm) < atm_target:
        need   = atm_target - len(in_atm)
        extras = (df[np.abs(df['moneyness']-1) <= atm_width]
                  .drop(combined.index, errors='ignore')
                  .nlargest(need, 'vega'))
        combined = pd.concat([combined, extras])
    edges = np.linspace(0, 1, num_m_bins+1)
    for i in range(num_m_bins):
        lo, hi = edges[i], edges[i+1]
        mask_comb = combined['moneyness'].rank(pct=True).between(lo, hi, inclusive='left')
        if not mask_comb.any():
            mask_df   = df['m_rank'].between(lo, hi, inclusive='left')
            if mask_df.any():
                pick = df[mask_df].nlargest(1, 'vega')
                combined = pd.concat([combined, pick])
    if len(combined) < total_target:
        need      = total_target - len(combined)
        leftovers = df.drop(combined.index, errors='ignore')
        extra     = leftovers.nlargest(need, 'vega')
        combined  = pd.concat([combined, extra])
    to_drop = ['days','moneyness','m_rank','d_norm','vega','cluster','pc1']
    return combined.drop(columns=[c for c in to_drop if c in combined.columns]) \
                   .reset_index(drop=True)
