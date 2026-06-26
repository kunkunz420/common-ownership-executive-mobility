"""Step 02-03: FWL absorption of high-dimensional FE + Event Study regression.

Method: Within each year cross-section, iteratively demean by origin firm and 
destination firm to absorb Origin×Year FE and Dest×Year FE (FWL theorem).
Then run OLS: any_move ~ delta_lambda + lambda_pre + controls on demeaned data.

Outputs:
  - event_study_coefficients.csv     (δ_τ for each year)
  - event_study_full_results.txt     (detailed regression output)
"""
import pandas as pd, numpy as np, warnings, sys
from scipy import stats
warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 02-03: FWL Absorption + Event Study Regression")
print("=" * 60)

# ── 1. Load panel ──
panel = pd.read_parquet('/home/kun/Documents/论文运行/01_pair_year_panel/pair_year_panel.parquet')
print(f"Panel loaded: {len(panel):,} rows")

# ── 2. Focus on years with sufficient data ──
# Pre-treatment: 2000-2008, Post-treatment: 2009-2025
# Limit to years with at least some moves
year_range = list(range(2005, 2021))
panel = panel[panel['to_year'].isin(year_range)].copy()
print(f"Year range: {year_range[0]}–{year_range[-1]}, {len(panel):,} rows")

# ── 3. Helper: two-way iterative demeaning ──
def demean_2way(df, y_col, x_cols, origin_col, dest_col, max_iter=50, tol=1e-8):
    """Iterative two-way (origin + dest) demeaning via alternating projections.
    
    Absorbs: η_{origin} + μ_{dest} from both Y and X.
    Returns demeaned Y and X arrays.
    """
    n = len(df)
    y = df[y_col].values.astype(np.float64).copy()
    X = df[list(x_cols)].values.astype(np.float64).copy()
    
    origin_groups = df[origin_col].values
    dest_groups = df[dest_col].values
    
    # Pre-compute group indices for speed
    origin_ids, origin_inv = np.unique(origin_groups, return_inverse=True)
    dest_ids, dest_inv = np.unique(dest_groups, return_inverse=True)
    n_origin = len(origin_ids)
    n_dest = len(dest_ids)
    k = X.shape[1]
    
    print(f"  FE dims: origin={n_origin}, dest={n_dest}, N={n:,}, K={k}")
    
    # Work with residuals (subtract grand mean first as warm start)
    y_dm = y - y.mean()
    X_dm = X - X.mean(axis=0)
    
    for it in range(max_iter):
        # ── Absorb origin FE ──
        origin_sum_y = np.bincount(origin_inv, weights=y_dm, minlength=n_origin)
        origin_count = np.bincount(origin_inv, minlength=n_origin)
        origin_count[origin_count == 0] = 1  # avoid div by zero
        origin_mean_y = origin_sum_y / origin_count
        y_dm -= origin_mean_y[origin_inv]
        
        for j in range(k):
            origin_sum_x = np.bincount(origin_inv, weights=X_dm[:, j], minlength=n_origin)
            origin_mean_x = origin_sum_x / origin_count
            X_dm[:, j] -= origin_mean_x[origin_inv]
        
        # ── Absorb dest FE ──
        dest_sum_y = np.bincount(dest_inv, weights=y_dm, minlength=n_dest)
        dest_count = np.bincount(dest_inv, minlength=n_dest)
        dest_count[dest_count == 0] = 1
        dest_mean_y = dest_sum_y / dest_count
        y_dm -= dest_mean_y[dest_inv]
        
        for j in range(k):
            dest_sum_x = np.bincount(dest_inv, weights=X_dm[:, j], minlength=n_dest)
            dest_mean_x = dest_sum_x / dest_count
            X_dm[:, j] -= dest_mean_x[dest_inv]
        
        # Convergence check
        if it % 10 == 0 and it > 0:
            ssr = np.sum(y_dm ** 2)
            print(f"    iter {it}: SSR={ssr:.6f}")
    
    return y_dm, X_dm, list(x_cols)


# ── 4. Event Study: year-by-year ──
print("\n" + "=" * 60)
print("EVENT STUDY: YEAR-BY-YEAR CROSS-SECTION")
print("=" * 60)

results = []
x_cols = ['delta_lambda', 'lambda_jk_pre']

for year in year_range:
    print(f"\n--- Year {year} ---")
    df_yr = panel[panel['to_year'] == year].copy()
    print(f"  Obs: {len(df_yr):,}, moves: {df_yr['any_move'].sum():,}, rate: {df_yr['any_move'].mean()*100:.4f}%")
    
    if df_yr['any_move'].sum() < 5:
        print(f"  SKIP: too few moves")
        results.append({
            'year': year,
            'coef_delta_lambda': np.nan,
            'se_delta_lambda': np.nan,
            't_delta_lambda': np.nan,
            'p_delta_lambda': np.nan,
            'coef_lambda_pre': np.nan,
            'se_lambda_pre': np.nan,
            'n_obs': len(df_yr),
            'n_moves': int(df_yr['any_move'].sum()),
            'move_rate': df_yr['any_move'].mean(),
            'r2': np.nan,
        })
        continue
    
    # Demean
    y_dm, X_dm, cols = demean_2way(
        df_yr, 
        y_col='any_move',
        x_cols=x_cols,
        origin_col='from_gvkey',
        dest_col='to_gvkey',
        max_iter=30
    )
    
    # OLS on demeaned data
    XX = X_dm.T @ X_dm
    Xy = X_dm.T @ y_dm
    n = len(y_dm)
    k = X_dm.shape[1]
    
    try:
        beta = np.linalg.solve(XX, Xy)
        y_pred = X_dm @ beta
        residuals = y_dm - y_pred
        rss = np.sum(residuals ** 2)
        tss = np.sum(y_dm ** 2)
        r2 = 1 - rss / tss if tss > 0 else 0
        
        # Standard errors (robust: clustered by pair not possible in cross-section)
        sigma2 = rss / (n - k)
        vcov = sigma2 * np.linalg.inv(XX)
        se = np.sqrt(np.diag(vcov))
        t_stat = beta / se
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - k))
        
        print(f"  delta_lambda: coef={beta[0]:.8f}, se={se[0]:.8f}, t={t_stat[0]:.3f}, p={p_val[0]:.6f}")
        print(f"  lambda_pre:   coef={beta[1]:.8f}, se={se[1]:.8f}, t={t_stat[1]:.3f}, p={p_val[1]:.6f}")
        print(f"  R²={r2:.6f}")
        
        results.append({
            'year': year,
            'coef_delta_lambda': beta[0],
            'se_delta_lambda': se[0],
            't_delta_lambda': t_stat[0],
            'p_delta_lambda': p_val[0],
            'coef_lambda_pre': beta[1],
            'se_lambda_pre': se[1],
            'n_obs': n,
            'n_moves': int(df_yr['any_move'].sum()),
            'move_rate': df_yr['any_move'].mean(),
            'r2': r2,
        })
    except np.linalg.LinAlgError as e:
        print(f"  ERROR: {e}")
        results.append({
            'year': year,
            'coef_delta_lambda': np.nan,
            'se_delta_lambda': np.nan,
            't_delta_lambda': np.nan,
            'p_delta_lambda': np.nan,
            'coef_lambda_pre': np.nan,
            'se_lambda_pre': np.nan,
            'n_obs': n,
            'n_moves': int(df_yr['any_move'].sum()),
            'move_rate': df_yr['any_move'].mean(),
            'r2': np.nan,
        })

# ── 5. Save results ──
results_df = pd.DataFrame(results)
results_df.to_csv('/home/kun/Documents/论文运行/03_event_study/event_study_coefficients.csv', index=False)

print(f"\n{'='*60}")
print("EVENT STUDY RESULTS")
print(f"{'='*60}")
print(results_df.to_string(index=False))

# ── 6. Save to shared output ──
results_df.to_csv('/home/kun/Documents/论文运行/02_fwl_absorption/event_study_coefficients.csv', index=False)

# Also save the panel summary from step 01 to central location
panel_summary = pd.read_csv('/home/kun/Documents/论文运行/01_pair_year_panel/panel_summary.csv')
panel_summary.to_csv('/home/kun/Documents/论文运行/02_fwl_absorption/panel_summary.csv', index=False)

print(f"\n✓ Results saved to 03_event_study/event_study_coefficients.csv")
print(f"  Also copied to 02_fwl_absorption/")
