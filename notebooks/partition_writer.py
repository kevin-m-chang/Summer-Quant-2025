import polars as pl
from pathlib import Path
from joblib import Parallel, delayed
from sampling import hybrid_kmeans_pca_atm
import os
import shutil
import stat

def process_date(part_dir: Path, output_dir: Path, **sampling_kwargs):
    df = pl.read_parquet(str(part_dir/"0.parquet")).to_pandas()
    sampled = hybrid_kmeans_pca_atm(df, **sampling_kwargs)
    sampled['date'] = part_dir.name.split('=')[1]
    out_dir = output_dir / part_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(sampled).write_parquet(str(out_dir/"0.parquet"), compression='snappy')
    return len(sampled)

def parallel_sample(input_dir: Path, output_dir: Path, n_jobs: int = -1, **sampling_kwargs):
    date_dirs = sorted(input_dir.glob("date=*"))
    # Clean up output dir
    def _on_rm_error(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    if output_dir.exists():
        shutil.rmtree(output_dir, onerror=_on_rm_error)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_date)(d, output_dir, **sampling_kwargs) for d in date_dirs
    )
    print("Hybrid KMeans+PCA+Vega Weighted sampling done!")
    print("Total picks:", sum(sizes))
    print("Sample sizes (first 5):", sizes[:5])
    return sizes
