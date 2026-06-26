"""Phase 10a: Build λjk (firm-pair common ownership measure).

λjk = Σᵢ βᵢⱼ βᵢₖ / Σᵢ βᵢⱼ²

Computes pre-merger (2007-2008) and post-merger (2010-2011) λjk,
then Δλjk = λ_post - λ_pre as the treatment variable.

Output: lambda_jk_pre.parquet, lambda_jk_post.parquet, lambda_jk_delta.parquet
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

DATA_ROOT = Path("/home/kun/Documents/沃顿数据")
OUTPUT = DATA_ROOT / "数据和清洗" / "data_clean" / "output"
BETA_CSV = OUTPUT / "beta_aggregated.csv"
PANEL_CSV = OUTPUT / "analysis_panel.csv"
LAMBDA_DIR = OUTPUT / "lambda_jk"
LAMBDA_DIR.mkdir(exist_ok=True)

PRE_YEARS = [2007, 2008]
POST_YEARS = [2010, 2011]
MIN_BETA = 1e-8


def load_firm_list() -> set:
    """Get all gvkeys that appear in the ExecuComp analysis panel."""
    chunks = pd.read_csv(PANEL_CSV, usecols=["gvkey"], chunksize=200_000)
    firms = set()
    for c in chunks:
        firms.update(c["gvkey"].unique())
    print(f"Analysis panel gvkeys: {len(firms):,}")
    return firms


def load_beta_period(years: list, valid_firms: set) -> pd.DataFrame:
    """Load beta_aggregated for given years, keep only valid firms."""
    chunks = pd.read_csv(
        BETA_CSV,
        usecols=["gvkey", "mgrno", "quarter", "beta"],
        chunksize=500_000,
    )
    rows = []
    for c in chunks:
        yr = c["quarter"].str[:4].astype(int)
        mask = yr.isin(years) & c["gvkey"].isin(valid_firms) & (c["beta"] > MIN_BETA)
        rows.append(c[mask])
    df = pd.concat(rows, ignore_index=True)
    print(f"  Rows loaded: {len(df):,}")
    print(f"  Firms: {df['gvkey'].nunique():,}")
    print(f"  Institutions: {df['mgrno'].nunique():,}")
    return df


def build_lambda_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build λjk matrix from ownership data.
    
    λjk = Σᵢ βᵢⱼ βᵢₖ / Σᵢ βᵢⱼ²
    
    Uses sparse matrix multiplication for efficiency.
    Returns DataFrame with columns (gvkey_j, gvkey_k, lambda_jk).
    Keeps only j ≠ k, and only pairs where lambda > 0.
    """
    # Average beta by (firm, institution) over the period
    avg = df.groupby(["gvkey", "mgrno"], as_index=False)["beta"].mean()
    print(f"  Avg ownership pairs: {len(avg):,}")

    # Build integer mappings
    gvkeys = sorted(avg["gvkey"].unique())
    mgrnos = sorted(avg["mgrno"].unique())
    gvkey_to_idx = {g: i for i, g in enumerate(gvkeys)}
    mgrno_to_idx = {m: i for i, m in enumerate(mgrnos)}
    n_firms = len(gvkeys)
    n_institutions = len(mgrnos)
    print(f"  Matrix: {n_firms} firms × {n_institutions} institutions")

    # Build sparse ownership matrix B (n_firms × n_institutions)
    row_idx = avg["gvkey"].map(gvkey_to_idx).values
    col_idx = avg["mgrno"].map(mgrno_to_idx).values
    beta_vals = avg["beta"].values.astype(np.float32)

    B = sparse.csr_matrix(
        (beta_vals, (row_idx, col_idx)),
        shape=(n_firms, n_institutions),
        dtype=np.float32,
    )
    del avg, row_idx, col_idx, beta_vals
    gc.collect()

    # Compute λ = B @ B.T  (n_firms × n_firms)
    # λ_jk = Σᵢ βᵢⱼ βᵢₖ
    L = B @ B.T  # sparse result, but may be dense
    print(f"  Raw λ matrix computed: {L.shape}")

    # Convert to dense for normalization
    L_dense = L.toarray() if hasattr(L, 'toarray') else np.array(L)

    # Normalize: λ_jk = λ_jk / diag_j where diag_j = Σᵢ βᵢⱼ²
    diag = np.diag(L_dense).copy()
    diag[diag < 1e-10] = np.nan  # avoid division by zero
    L_normalized = L_dense / diag[:, np.newaxis]
    print(f"  λ values: min={np.nanmin(L_normalized):.6f}, "
          f"max={np.nanmax(L_normalized):.6f}, "
          f"mean={np.nanmean(L_normalized):.6f}")

    # Extract all pairs (j ≠ k) with non-zero λ
    n = n_firms
    # Remove diagonal, get indices of non-zero (above threshold) values
    mask = ~np.eye(n, dtype=bool) & (L_normalized > 1e-8) & ~np.isnan(L_normalized)
    j_idx, k_idx = np.where(mask)
    values = L_normalized[mask]

    result = pd.DataFrame({
        "gvkey_j": [gvkeys[j] for j in j_idx],
        "gvkey_k": [gvkeys[k] for k in k_idx],
        "lambda_jk": values,
    })
    print(f"  Non-zero firm-pairs: {len(result):,}")
    print(f"  Connected firms: {result['gvkey_j'].nunique():,} "
          f"→ {result['gvkey_k'].nunique():,}")
    return result


def main():
    print("=" * 60)
    print("Phase 10a: Build λjk (firm-pair common ownership)")
    print("=" * 60)

    # Valid firms
    valid_firms = load_firm_list()

    # Pre-merger λ
    print(f"\n--- Pre-merger ({PRE_YEARS}) ---")
    beta_pre = load_beta_period(PRE_YEARS, valid_firms)
    lambda_pre = build_lambda_matrix(beta_pre)
    lambda_pre.to_parquet(LAMBDA_DIR / "lambda_pre.parquet")
    print(f"  Saved: {LAMBDA_DIR / 'lambda_pre.parquet'}")
    del beta_pre, lambda_pre
    gc.collect()

    # Post-merger λ
    print(f"\n--- Post-merger ({POST_YEARS}) ---")
    beta_post = load_beta_period(POST_YEARS, valid_firms)
    lambda_post = build_lambda_matrix(beta_post)
    lambda_post.to_parquet(LAMBDA_DIR / "lambda_post.parquet")
    print(f"  Saved: {LAMBDA_DIR / 'lambda_post.parquet'}")
    del beta_post, lambda_post
    gc.collect()

    # Compute Δλ = λ_post - λ_pre
    print("\n--- Computing Δλ ---")
    pre = pd.read_parquet(LAMBDA_DIR / "lambda_pre.parquet")
    post = pd.read_parquet(LAMBDA_DIR / "lambda_post.parquet")
    merged = pre.merge(
        post, on=["gvkey_j", "gvkey_k"],
        how="outer", suffixes=("_pre", "_post"),
    )
    merged["lambda_jk_pre"] = merged["lambda_jk_pre"].fillna(0.0)
    merged["lambda_jk_post"] = merged["lambda_jk_post"].fillna(0.0)
    merged["delta_lambda"] = merged["lambda_jk_post"] - merged["lambda_jk_pre"]
    merged = merged[merged["delta_lambda"].abs() > 1e-8].copy()
    merged = merged.reset_index(drop=True)

    print(f"  Firm-pairs with Δλ ≠ 0: {len(merged):,}")
    print(f"  Δλ: min={merged['delta_lambda'].min():.6f}, "
          f"max={merged['delta_lambda'].max():.6f}, "
          f"mean={merged['delta_lambda'].mean():.6f}, "
          f"SD={merged['delta_lambda'].std():.6f}")
    print(f"  Δλ > 0: {(merged['delta_lambda']>0).sum():,} ({(merged['delta_lambda']>0).mean()*100:.1f}%)")

    merged.to_parquet(LAMBDA_DIR / "lambda_delta.parquet")
    # Also save CSV for easy inspection
    merged.to_csv(LAMBDA_DIR / "lambda_delta.csv", index=False)
    print(f"  Saved: {LAMBDA_DIR / 'lambda_delta.parquet'}")
    print(f"  Saved: {LAMBDA_DIR / 'lambda_delta.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
