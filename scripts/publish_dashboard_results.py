from pathlib import Path
import shutil
from typing import Optional


# --------------------
# CONFIG
# --------------------
MAIN_REPO = Path(__file__).resolve().parents[1]   # proj_quant_trading
DASH_REPO = MAIN_REPO.parent / "qbt_dashboard"

SRC_ROOT = MAIN_REPO / "artifacts" / "backtesting_results"
DST_ROOT = DASH_REPO / "results"   # per your updated route


# --------------------
# HELPERS
# --------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, *, overwrite: bool = False) -> None:
    """
    Copy a single file. If overwrite=False and dst exists, skip.
    """
    ensure_dir(dst.parent)
    if dst.exists() and not overwrite:
        # keep existing results
        return
    shutil.copy2(src, dst)


def copy_tree(src_dir: Path, dst_dir: Path, *, overwrite: bool = False) -> None:
    """
    Copy a directory tree preserving structure.
    If overwrite=False, existing files are preserved.
    """
    for p in src_dir.rglob("*"):
        rel = p.relative_to(src_dir)
        out = dst_dir / rel
        if p.is_dir():
            ensure_dir(out)
        else:
            copy_file(p, out, overwrite=overwrite)


# --------------------
# MAIN
# --------------------
def main(experiment: Optional[str] = None, *, overwrite: bool = False) -> None:
    if not DASH_REPO.exists():
        raise RuntimeError(f"Dashboard repo not found: {DASH_REPO}")

    if not SRC_ROOT.exists():
        raise RuntimeError(f"Source artifacts not found: {SRC_ROOT}")

    ensure_dir(DST_ROOT)

    # If experiment is provided, copy only that partition (merge into existing)
    if experiment is not None:
        src_exp = SRC_ROOT / f"experiment={experiment}"
        if not src_exp.exists():
            raise RuntimeError(f"Experiment not found: {src_exp}")

        dst_exp = DST_ROOT / src_exp.name

        print(f"Copying one experiment (no delete): {src_exp} -> {dst_exp}")
        ensure_dir(dst_exp)

        copy_tree(src_exp, dst_exp, overwrite=overwrite)

        print("\n✅ Export complete.")
        print(f"Target: {dst_exp}")
        print(f"Overwrite: {overwrite}")
        return

    # Otherwise: copy all experiments (merge into results/)
    print(f"Copying ALL experiments (no delete): {SRC_ROOT} -> {DST_ROOT}")

    # DO NOT delete DST_ROOT
    # Just merge/copy each experiment folder into results/
    for exp_dir in SRC_ROOT.glob("experiment=*"):
        if exp_dir.is_dir():
            dst_exp = DST_ROOT / exp_dir.name
            ensure_dir(dst_exp)
            copy_tree(exp_dir, dst_exp, overwrite=overwrite)

    print("\n✅ Export complete.")
    print(f"Target: {DST_ROOT}")
    print(f"Overwrite: {overwrite}")


if __name__ == "__main__":
    # Option A: copy everything, keep existing files
    main(overwrite=False)

    # Option B: copy only one experiment, keep existing files
    # main(experiment="macro_var", overwrite=False)

    # Option C: copy only one experiment, overwrite existing files inside it
    # main(experiment="macro_var", overwrite=True)