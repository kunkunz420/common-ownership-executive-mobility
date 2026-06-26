"""Step 08: Pair-level cross-sectional logit (baseline replication).

Replicates the core pair-level result: AnyMove_jk ~ delta_lambda + lambda_pre.

This is the simplest specification — a cross-sectional logit on 1.07M pairs.
Serves as the baseline against which the richer Event Study is compared.
"""
import pandas as pd, numpy as np, statsmodels.api as sm, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 08: Pair-Level Cross-Sectional Logit (Baseline)")
print("=" * 60)

# Load pair panel
panel = pd.read_parquet('/home/kun/Documents/论文运行/01_pair_year_panel/pair_year_panel.parquet')

# Collapse to pair-level cross-section
pair = panel.groupby(['from_gvkey', 'to_gvkey']).agg(
    any_move=('any_move', 'max'),
    n_moves=('n_moves', 'sum'),
    delta_lambda=('delta_lambda', 'first'),
    lambda_jk_pre=('lambda_jk_pre', 'first'),
    lambda_jk_post=('lambda_jk_post', 'first'),
).reset_index()

print(f"\nPair cross-section: {len(pair):,} pairs")
print(f"  Pairs with any move: {pair['any_move'].sum():,} ({pair['any_move'].mean()*100:.4f}%)")
print(f"  Total moves: {pair['n_moves'].sum():,}")

# Summary stats
print(f"\nLambda summary:")
for col in ['delta_lambda', 'lambda_jk_pre', 'lambda_jk_post']:
    print(f"  {col}: mean={pair[col].mean():.6f}, std={pair[col].std():.6f}")

print(f"\nMean lambda by move status:")
print(pair.groupby('any_move')[['delta_lambda', 'lambda_jk_pre', 'lambda_jk_post']].mean())

# Logit regression
X = sm.add_constant(pair[['delta_lambda', 'lambda_jk_pre']].values)
y = pair['any_move'].values

logit_model = sm.Logit(y, X)
logit_result = logit_model.fit(disp=False)

print(f"\n{'='*60}")
print("LOGIT REGRESSION RESULTS")
print(f"{'='*60}")
print(logit_result.summary().tables[1])

# Marginal effects at means
coefs = logit_result.params
xb_mean = coefs[0] + coefs[1]*pair['delta_lambda'].mean() + coefs[2]*pair['lambda_jk_pre'].mean()
p_mean = 1 / (1 + np.exp(-xb_mean))
marginal = p_mean * (1 - p_mean) * coefs

print(f"\nMarginal effects (at means):")
print(f"  delta_lambda: {marginal[1]:.8f}")
print(f"  lambda_jk_pre: {marginal[2]:.8f}")
print(f"  Baseline prob (at mean X): {p_mean:.8f}")

# Also OLS for comparability with Event Study (LPM)
ols_model = sm.OLS(y, X)
ols_result = ols_model.fit()

print(f"\n{'='*60}")
print("OLS (Linear Probability Model) — for Event Study comparison")
print(f"{'='*60}")
print(ols_result.summary().tables[1])

# Save results
results_text = f"""
================================================================================
PAIR-LEVEL CROSS-SECTIONAL LOGIT — BASELINE REPLICATION
================================================================================
N = {len(pair):,} firm-pairs
Pairs with moves: {pair['any_move'].sum():,}

LOGIT RESULTS
────────────────────────────────────────────────────────────────
{logit_result.summary().tables[1].as_text()}

MARGINAL EFFECTS (at means)
────────────────────────────────────────────────────────────────
  delta_lambda: {marginal[1]:.8f}
  lambda_jk_pre: {marginal[2]:.8f}

OLS (LPM) RESULTS — for Event Study comparison
────────────────────────────────────────────────────────────────
{ols_result.summary().tables[1].as_text()}

MEAN LAMBDA BY MOVE STATUS
────────────────────────────────────────────────────────────────
No move:  delta_lambda={pair[pair['any_move']==0]['delta_lambda'].mean():.6f}
Move:     delta_lambda={pair[pair['any_move']==1]['delta_lambda'].mean():.6f}
Difference: {pair[pair['any_move']==1]['delta_lambda'].mean() - pair[pair['any_move']==0]['delta_lambda'].mean():.6f}
"""

with open('/home/kun/Documents/论文运行/08_baseline_logit/baseline_logit_results.txt', 'w') as f:
    f.write(results_text)

# Save coefficient table
coef_table = pd.DataFrame({
    'Model': ['Logit', 'Logit', 'OLS (LPM)', 'OLS (LPM)'],
    'Variable': ['delta_lambda', 'lambda_jk_pre', 'delta_lambda', 'lambda_jk_pre'],
    'Coefficient': [logit_result.params[1], logit_result.params[2], ols_result.params[1], ols_result.params[2]],
    'SE': [logit_result.bse[1], logit_result.bse[2], ols_result.bse[1], ols_result.bse[2]],
    'p_value': [logit_result.pvalues[1], logit_result.pvalues[2], ols_result.pvalues[1], ols_result.pvalues[2]],
    'N': [len(pair)] * 4,
})
coef_table.to_csv('/home/kun/Documents/论文运行/08_baseline_logit/baseline_coefficients.csv', index=False)

print(f"\n✓ Results saved to 08_baseline_logit/")
print(f"  baseline_logit_results.txt")
print(f"  baseline_coefficients.csv")
