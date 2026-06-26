"""Step 09: RESET test (Ramsey Regression Equation Specification Error Test).

Tests whether the LPM specification is correctly specified by adding
powers of fitted values (ŷ², ŷ³) and testing their joint significance.
H0: model is correctly specified (no omitted nonlinearities).

Also runs on the pair-level cross-sectional logit using the Link test
(the logit-equivalent of RESET).

Input:  pair_year_panel.parquet from Step 01
Output: reset_test_results.txt
"""
import pandas as pd, numpy as np, statsmodels.api as sm, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 09: RESET / Link Test (Specification Error)")
print("=" * 60)

# ── Load data ──
panel = pd.read_parquet('/home/kun/Documents/论文运行/01_pair_year_panel/pair_year_panel.parquet')

# Collapse to pair-level cross-section
pair = panel.groupby(['from_gvkey', 'to_gvkey']).agg(
    any_move=('any_move', 'max'),
    delta_lambda=('delta_lambda', 'first'),
    lambda_jk_pre=('lambda_jk_pre', 'first'),
    lambda_jk_post=('lambda_jk_post', 'first'),
).reset_index()

print(f"Pair cross-section: {len(pair):,} pairs, {pair['any_move'].sum():,} moves")

# ═══════════════════════════════════════════════
# 1. RESET test on OLS (LPM)
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("1. RESET TEST — OLS (Linear Probability Model)")
print(f"{'='*60}")

X = sm.add_constant(pair[['delta_lambda', 'lambda_jk_pre']].values)
y = pair['any_move'].values

# Step 1: estimate base model
base_model = sm.OLS(y, X).fit()
y_hat = base_model.fittedvalues

# Step 2: add y_hat², y_hat³
X_reset = np.column_stack([X, y_hat**2, y_hat**3])
reset_model = sm.OLS(y, X_reset).fit()

# Step 3: F-test that coefficients on y_hat², y_hat³ are jointly zero
from scipy import stats
r_matrix = np.zeros((2, X_reset.shape[1]))
r_matrix[0, -2] = 1  # coef on y_hat² = 0
r_matrix[1, -1] = 1  # coef on y_hat³ = 0

f_test = reset_model.f_test(r_matrix)
f_stat = f_test.statistic if np.isscalar(f_test.statistic) else f_test.statistic[0][0]
f_pval = f_test.pvalue if np.isscalar(f_test.pvalue) else f_test.pvalue
print(f"\nH0: y_hat² and y_hat³ coefficients are jointly zero (model is correctly specified)")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value:     {f_pval:.6f}")
if f_pval < 0.05:
    print(f"→ REJECT H0 at 5%: evidence of specification error ⚠️")
else:
    print(f"→ FAIL TO REJECT H0 at 5%: model is correctly specified ✓")

# Also test y_hat² alone, y_hat³ alone
for i, label in [(-2, 'y_hat²'), (-1, 'y_hat³')]:
    r_single = np.zeros((1, X_reset.shape[1]))
    r_single[0, i] = 1
    f_single = reset_model.f_test(r_single)
    fs = f_single.statistic if np.isscalar(f_single.statistic) else f_single.statistic[0][0]
    fp = f_single.pvalue if np.isscalar(f_single.pvalue) else f_single.pvalue
    print(f"  {label} alone: F={fs:.4f}, p={fp:.6f}")

# ═══════════════════════════════════════════════
# 2. Link test on Logit
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("2. LINK TEST — Logit")
print(f"{'='*60}")
print("(Logit-equivalent of RESET: test that _hat² = 0)")

logit_base = sm.Logit(y, X).fit(disp=False)
xb_hat = logit_base.fittedvalues  # predicted probabilities
xb_sq = (xb_hat * xb_hat)  # (_hat)²

X_link = np.column_stack([X, xb_sq])
logit_link = sm.Logit(y, X_link).fit(disp=False)

# Wald test on _hatsq coefficient
from scipy.stats import norm
coef_link = logit_link.params[-1]
se_link = logit_link.bse[-1]
z_link = coef_link / se_link
p_link = 2 * (1 - norm.cdf(abs(z_link)))

print(f"\nH0: coefficient on _hat² = 0 (model is correctly specified)")
print(f"  _hat² coefficient: {coef_link:.6f}")
print(f"  SE:                {se_link:.6f}")
print(f"  z-statistic:       {z_link:.4f}")
print(f"  p-value:           {p_link:.6f}")
if p_link < 0.05:
    print(f"  → REJECT H0 at 5%: evidence of specification error ⚠️")
else:
    print(f"  → FAIL TO REJECT H0 at 5%: model is correctly specified ✓")

# ═══════════════════════════════════════════════
# 3. Year-by-year RESET (Event Study robustness)
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("3. YEAR-BY-YEAR RESET — Event Study robustness")
print(f"{'='*60}")
print("(Tests specification stability across years)")

es_coefs = pd.read_csv('/home/kun/Documents/论文运行/03_event_study/event_study_coefficients.csv')
reset_by_year = []

for year in range(2005, 2021):
    df_yr = panel[panel['to_year'] == year].copy()
    if df_yr['any_move'].sum() < 10:
        reset_by_year.append({'year': year, 'reset_F': np.nan, 'reset_p': np.nan, 'n_moves': int(df_yr['any_move'].sum())})
        continue
    
    X_yr = sm.add_constant(df_yr[['delta_lambda', 'lambda_jk_pre']].values)
    y_yr = df_yr['any_move'].values
    
    base_yr = sm.OLS(y_yr, X_yr).fit()
    y_hat_yr = base_yr.fittedvalues
    
    X_reset_yr = np.column_stack([X_yr, y_hat_yr**2, y_hat_yr**3])
    reset_yr = sm.OLS(y_yr, X_reset_yr).fit()
    
    r_yr = np.zeros((2, X_reset_yr.shape[1]))
    r_yr[0, -2] = 1
    r_yr[1, -1] = 1
    f_yr = reset_yr.f_test(r_yr)
    fy = f_yr.statistic if np.isscalar(f_yr.statistic) else f_yr.statistic[0][0]
    fp = f_yr.pvalue if np.isscalar(f_yr.pvalue) else f_yr.pvalue
    
    reset_by_year.append({
        'year': year,
        'reset_F': fy,
        'reset_p': fp,
        'n_moves': int(df_yr['any_move'].sum()),
    })

reset_df = pd.DataFrame(reset_by_year)
print(f"\nYear-by-year RESET p-values:")
for _, r in reset_df.iterrows():
    sig = " ⚠️" if r['reset_p'] < 0.05 else " ✓" if not np.isnan(r['reset_p']) else " —"
    p_str = f"{r['reset_p']:.4f}" if not np.isnan(r['reset_p']) else "skip"
    print(f"  {int(r['year'])}: p={p_str}, moves={int(r['n_moves'])}{sig}")

# Count rejections
n_reject = (reset_df['reset_p'] < 0.05).sum()
print(f"\nYears rejecting H0 at 5%: {n_reject}/{reset_df['reset_p'].notna().sum()}")

# ═══════════════════════════════════════════════
# 4. Save results
# ═══════════════════════════════════════════════
import os
os.makedirs('/home/kun/Documents/论文运行/09_reset_test', exist_ok=True)

results_text = f"""================================================================================
RESET TEST / LINK TEST RESULTS
================================================================================
Date: 2026-06-26

1. RESET TEST — OLS (LPM), pair-level cross-section
────────────────────────────────────────────────────────────────
N = {len(pair):,} firm-pairs
H0: Model is correctly specified (no omitted nonlinearities)

  F-statistic (y_hat² + y_hat³ jointly): {f_stat:.4f}
  p-value: {f_pval:.6f}
  Conclusion: {'REJECT — specification error detected ⚠️' if f_pval < 0.05 else 'FAIL TO REJECT — model correctly specified ✓'}

2. LINK TEST — Logit, pair-level cross-section
────────────────────────────────────────────────────────────────
  _hat² coefficient: {coef_link:.6f}
  z-statistic: {z_link:.4f}
  p-value: {p_link:.6f}
  Conclusion: {'REJECT — specification error detected ⚠️' if p_link < 0.05 else 'FAIL TO REJECT — model correctly specified ✓'}

3. YEAR-BY-YEAR RESET
────────────────────────────────────────────────────────────────
{n_reject}/{reset_df['reset_p'].notna().sum()} years reject H0 at 5% level.

"""

with open('/home/kun/Documents/论文运行/09_reset_test/reset_test_results.txt', 'w') as f:
    f.write(results_text)

reset_df.to_csv('/home/kun/Documents/论文运行/09_reset_test/reset_by_year.csv', index=False)

print(f"\n{'='*60}")
print(f"✓ Results saved to 09_reset_test/")
print(f"  reset_test_results.txt")
print(f"  reset_by_year.csv")
