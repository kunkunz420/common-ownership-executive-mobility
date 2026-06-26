"""Step 10: Pair-Level Robustness Checks (memory-optimized)."""
import pandas as pd, numpy as np, statsmodels.api as sm, warnings, os
from scipy import stats
warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 10: Pair-Level Robustness Checks")
print("=" * 60)

# ── Load only pair cross-section (not full panel) ──
panel = pd.read_parquet('/home/kun/Documents/论文运行/01_pair_year_panel/pair_year_panel.parquet',
                         columns=['from_gvkey','to_gvkey','to_year','any_move','n_moves',
                                  'delta_lambda','lambda_jk_pre'])
pair = panel.groupby(['from_gvkey', 'to_gvkey']).agg(
    any_move=('any_move', 'max'),
    n_moves=('n_moves', 'sum'),
    delta_lambda=('delta_lambda', 'first'),
    lambda_jk_pre=('lambda_jk_pre', 'first'),
).reset_index()
del panel  # free memory

print(f"Base: {len(pair):,} pairs, {pair['any_move'].sum():,} with moves")
results = []

# ── Helper: OLS ──
def ols_fit(data, label):
    X = sm.add_constant(data[['delta_lambda', 'lambda_jk_pre']].astype(float).values)
    y = data['any_move'].astype(float).values
    m = sm.OLS(y, X).fit()
    return {'spec': label, 'n_pairs': len(data), 'n_moves': int(data['any_move'].sum()),
            'coef_delta': m.params[1], 'se_delta': m.bse[1], 
            't_delta': m.tvalues[1], 'p_delta': m.pvalues[1],
            'coef_lambda_pre': m.params[2], 'p_lambda_pre': m.pvalues[2], 'r2': m.rsquared}

# ── Helper: Logit ──
def logit_fit(data, label):
    X = sm.add_constant(data[['delta_lambda', 'lambda_jk_pre']].astype(float).values)
    y = data['any_move'].astype(float).values
    m = sm.Logit(y, X).fit(disp=False)
    return {'spec': label + ' (logit)', 'n_pairs': len(data), 'n_moves': int(data['any_move'].sum()),
            'coef_delta': m.params[1], 'p_delta': m.pvalues[1]}

# ── 0. Baseline ──
r = ols_fit(pair, '0. Baseline')
results.append(r)
results.append(logit_fit(pair, '0. Baseline'))
print(f"\nBaseline: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), λ_pre={r['coef_lambda_pre']:.6f} (p={r['p_lambda_pre']:.6f}), R²={r['r2']:.6f}")

# ── 1. Winsorize ──
for pct in [0.01, 0.05]:
    lo = pair['delta_lambda'].quantile(pct)
    hi = pair['delta_lambda'].quantile(1 - pct)
    w = pair.copy()
    w['delta_lambda'] = w['delta_lambda'].clip(lo, hi)
    r = ols_fit(w, f'1. Winsorize {int(pct*100)}%')
    results.append(r)
    print(f"Winsorize {int(pct*100)}%: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f})")
del w

# ── 2. Only positive Δλ ──
pos = pair[pair['delta_lambda'] > 0]
r = ols_fit(pos, '2. Only Δλ > 0')
results.append(r); results.append(logit_fit(pos, '2. Only Δλ > 0'))
print(f"Δλ > 0: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), N={len(pos):,}")

# ── 3. Only negative Δλ ──
neg = pair[pair['delta_lambda'] < 0]
if len(neg) > 100:
    r = ols_fit(neg, '3. Only Δλ < 0')
    results.append(r)
    print(f"Δλ < 0: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), N={len(neg):,}")

# ── 4. Quartile split ──
pair['q_delta'] = pd.qcut(pair['delta_lambda'], 4, labels=['Q1','Q2','Q3','Q4'])
for q in ['Q1', 'Q4']:
    sub = pair[pair['q_delta'] == q]
    r = ols_fit(sub, f'4. Quartile {q}')
    results.append(r)
    print(f"{q}: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), moves={r['n_moves']}")

# ── 5. Only λ_pre > 0 ──
nz = pair[pair['lambda_jk_pre'] > 0]
r = ols_fit(nz, '5. Only λ_pre > 0')
results.append(r); results.append(logit_fit(nz, '5. Only λ_pre > 0'))
print(f"λ_pre > 0: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), N={len(nz):,}")

# ── 6. Exclude top 1% moves ──
cap = pair['n_moves'].quantile(0.99)
trim = pair[pair['n_moves'] <= cap]
r = ols_fit(trim, '6. Exclude top 1% moves')
results.append(r)
print(f"Trim top 1%: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), N={len(trim):,}")

# ── 7. Only pairs where at least one firm had any move ──
active_orig = pair.groupby('from_gvkey')['any_move'].transform('max')
active_dest = pair.groupby('to_gvkey')['any_move'].transform('max')
active = pair[(active_orig > 0) | (active_dest > 0)]
r = ols_fit(active, '7. Active firms only')
results.append(r); results.append(logit_fit(active, '7. Active firms only'))
print(f"Active firms: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), N={len(active):,}")

# ── 8. Clustered SE (origin) ──
X = sm.add_constant(pair[['delta_lambda', 'lambda_jk_pre']].astype(float).values)
y = pair['any_move'].astype(float).values
m_cl = sm.OLS(y, X).fit()
cov_cl = m_cl.get_robustcov_results(cov_type='cluster', groups=pair['from_gvkey'].values)
r = {'spec': '8. Clustered SE (origin)', 'n_pairs': len(pair), 'n_moves': int(pair['any_move'].sum()),
     'coef_delta': cov_cl.params[1], 'se_delta': cov_cl.bse[1],
     't_delta': cov_cl.tvalues[1], 'p_delta': cov_cl.pvalues[1],
     'coef_lambda_pre': cov_cl.params[2], 'p_lambda_pre': cov_cl.pvalues[2], 'r2': cov_cl.rsquared}
results.append(r)
print(f"Clustered SE: Δλ={r['coef_delta']:.6f} (SE_cl={r['se_delta']:.6f}, p={r['p_delta']:.6f})")

# ── 9. Time-split: pre-2010 vs post-2010 moves ──
panel_small = pd.read_parquet('/home/kun/Documents/论文运行/01_pair_year_panel/pair_year_panel.parquet',
                               columns=['from_gvkey','to_gvkey','to_year','any_move','n_moves',
                                        'delta_lambda','lambda_jk_pre'])
for label, yr_cond in [('Pre-2010', panel_small['to_year'] < 2010), 
                        ('Post-2009', panel_small['to_year'] >= 2010)]:
    sub = panel_small[yr_cond].groupby(['from_gvkey', 'to_gvkey']).agg(
        any_move=('any_move', 'max'), n_moves=('n_moves', 'sum'),
        delta_lambda=('delta_lambda', 'first'), lambda_jk_pre=('lambda_jk_pre', 'first'),
    ).reset_index()
    r = ols_fit(sub, f'9. Moves {label}')
    results.append(r)
    print(f"Moves {label}: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), moves={r['n_moves']}")
del panel_small

# ── 10. Exclude extreme Δλ (beyond 3σ) ──
mu, sd = pair['delta_lambda'].mean(), pair['delta_lambda'].std()
center = pair[(pair['delta_lambda'] > mu - 3*sd) & (pair['delta_lambda'] < mu + 3*sd)]
r = ols_fit(center, '10. Exclude Δλ > 3σ')
results.append(r)
print(f"Exclude outliers: Δλ={r['coef_delta']:.6f} (p={r['p_delta']:.6f}), N={len(center):,}")

# ═══════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════
os.makedirs('/home/kun/Documents/论文运行/10_robustness', exist_ok=True)
rdf = pd.DataFrame(results)
rdf.to_csv('/home/kun/Documents/论文运行/10_robustness/robustness_results.csv', index=False)

print(f"\n{'='*80}")
print(f"{'Specification':<35s} {'N':>10s} {'Δλ coef':>10s} {'p':>12s}")
print(f"{'='*80}")
for _, r in rdf.iterrows():
    dc = f"{r['coef_delta']:.6f}" if not pd.isna(r['coef_delta']) else "N/A"
    dp = f"{r['p_delta']:.6f}***" if not pd.isna(r['p_delta']) and r['p_delta'] < 0.01 else (
        f"{r['p_delta']:.6f} **" if not pd.isna(r['p_delta']) and r['p_delta'] < 0.05 else (
        f"{r['p_delta']:.6f}  *" if not pd.isna(r['p_delta']) and r['p_delta'] < 0.10 else (
        f"{r['p_delta']:.6f}" if not pd.isna(r['p_delta']) else "N/A")))
    print(f"{r['spec']:<35s} {r.get('n_pairs',0):>10,} {dc:>10s} {dp:>12s}")

stable = rdf[rdf['p_delta'].notna()]
n_sig = (stable['p_delta'] < 0.05).sum()
print(f"\n{len(stable)} specs, {n_sig}/{len(stable)} significant (p<0.05)")

summary = f"""================================================================================
PAIR-LEVEL ROBUSTNESS CHECKS — SUMMARY
================================================================================
{len(stable)} specifications tested. {n_sig}/{len(stable)} show significant Δλ (p<0.05).
Baseline Δλ: {rdf.iloc[0]['coef_delta']:.6f} (p={rdf.iloc[0]['p_delta']:.6f})
================================================================================
"""
with open('/home/kun/Documents/论文运行/10_robustness/robustness_summary.txt', 'w') as f:
    f.write(summary)
print(f"\n✓ Saved to 10_robustness/")
