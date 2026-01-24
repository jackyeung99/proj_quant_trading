from pathlib import Path
import shutil
import pandas as pd

from qbt.storage.paths import StoragePaths


# --------------------
# CONFIG
# --------------------
MAIN_REPO = Path(__file__).resolve().parents[1]   # proj_quant_trading
DASH_REPO = MAIN_REPO.parent / "qbt_dashboard"

SRC_ROOT = MAIN_REPO            
DST_ROOT = DASH_REPO

paths = StoragePaths(root="results")


# --------------------
# HELPERS
# --------------------
def key_to_local_path(root: Path, key: str) -> Path:
    """Convert StoragePaths key -> local filesystem path."""
    return root / key


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_parquet_to_csv_table(src: Path, dst: Path):
    """For normal tables like runs/metrics: index not needed."""
    print(f"Converting {src} -> {dst}")
    df = pd.read_parquet(src)
    ensure_dir(dst.parent)
    df.to_csv(dst, index=False)


def convert_parquet_to_csv_timeseries(src: Path, dst: Path):
    """For timeseries: preserve datetime index in first column."""
    print(f"Converting {src} -> {dst}")
    df = pd.read_parquet(src)
    ensure_dir(dst.parent)
    df.to_csv(dst, index=True)


def copy_json(src: Path, dst: Path):
    print(f"Copying {src} -> {dst}")
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


# --------------------
# MAIN
# --------------------
def main():
    if not DASH_REPO.exists():
        raise RuntimeError(f"Dashboard repo not found: {DASH_REPO}")

    dst_results = DST_ROOT / paths.root

    print("Cleaning dashboard results directory...")
    if dst_results.exists():
        shutil.rmtree(dst_results)
    ensure_dir(dst_results)

    # ---- export runs ----
    runs_key = paths.runs_key()
    src_runs = key_to_local_path(SRC_ROOT, runs_key)
    if src_runs.exists():
        dst_runs = key_to_local_path(DST_ROOT, runs_key).with_suffix(".csv")
        convert_parquet_to_csv_table(src_runs, dst_runs)
    else:
        print(f"Missing: {src_runs}")

    # ---- export metrics ----
    metrics_key = paths.metrics_key()
    src_metrics = key_to_local_path(SRC_ROOT, metrics_key)
    if src_metrics.exists():
        dst_metrics = key_to_local_path(DST_ROOT, metrics_key).with_suffix(".csv")
        convert_parquet_to_csv_table(src_metrics, dst_metrics)
    else:
        print(f"Missing: {src_metrics}")

    # ---- export meta + timeseries ----
    ts_root = SRC_ROOT / paths.root / "timeseries"
    runs_root = SRC_ROOT / paths.root / "runs"

    if runs_root.exists():
        for run_dir in runs_root.iterdir():
            meta_src = run_dir / "meta.json"
            if meta_src.exists():
                meta_dst = DST_ROOT / paths.root / "runs" / run_dir.name / "meta.json"
                copy_json(meta_src, meta_dst)

    if ts_root.exists():
        print("Exporting timeseries...")
        for pq in ts_root.rglob("*.parquet"):
            rel = pq.relative_to(SRC_ROOT)
            out_csv = (DST_ROOT / rel).with_suffix(".csv")
            convert_parquet_to_csv_timeseries(pq, out_csv)

    print("\n✅ Dashboard results exported successfully.")
    print(f"Target: {dst_results}")


if __name__ == "__main__":
    main()
