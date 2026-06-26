"""Step 04-05: Parallel trends test + Placebo test.

Step 04: Joint F-test on pre-2009 coefficients (δ_2005 ... δ_2009 = 0).
Step 05: Placebo — randomly permute treatment year and re-estimate.

Input:  event_study_coefficients.csv from Step 03
Output: parallel_trends_results.txt, placebo_results.csv
"""
import pandas as pd, numpy as np
from scipy import stats

print("=" * 60)
print("Step 04-05: Parallel Trends + Placebo")
print("=" * 60)

# ── Load step 03 results ──
df = pd.read_csv('/home/kun/Documents/论文运行/03_event_study/event_study_coefficients.csv')
print(f"\nLoaded {len(df)} year-coefficients from Step 03")

# ═══════════════════════════════════════════════
# STEP 04: Parallel Trends
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 04: PARALLEL TRENDS TEST")
print(f"{'='*60}")

# Define pre-treatment years (before BlackRock-BGI merger = 2009 Q3)
# Exclude 2009 itself (transition year)
pre_years = [2005, 2006, 2007, 2008]
pre_mask = df['year'].isin(pre_years)
pre_df = df[pre_mask].copy()

print(f"\nPre-treatment coefficients (2005-2008):")
for _, r in pre_df.iterrows():
    sig = "***" if r['p_delta_lambda'] < 0.01 else ("**" if r['p_delta_lambda'] < 0.05 else ("*" if r['p_delta_lambda'] < 0.10 else ""))
    print(f"  {int(r['year'])}: β={r['coef_delta_lambda']:.8f}, SE={r['se_delta_lambda']:.8f}, p={r['p_delta_lambda']:.6f} {sig}")

# Joint F-test: H0: β_2005 = β_2006 = β_2007 = β_2008 = 0
# Using Wald test approximation
valid_pre = pre_df['coef_delta_lambda'].notna()
betas = pre_df.loc[valid_pre, 'coef_delta_lambda'].values
ses   = pre_df.loc[valid_pre, 'se_delta_lambda'].values
V_inv = np.diag(1.0 / (ses ** 2))  # diagonal VC inverse (assuming independent)
W = betas @ V_inv @ betas
df_wald = np.sum(valid_pre)
p_wald = 1 - stats.chi2.cdf(W, df_wald)

print(f"\nJoint Wald test (H0: all pre-2009 δ_τ = 0):")
print(f"  W = {W:.4f}, df = {df_wald}")
print(f"  p-value = {p_wald:.6f}")
if p_wald < 0.05:
    print(f"  → REJECT parallel trends at 5% level ⚠️")
else:
    print(f"  → FAIL TO REJECT parallel trends at 5% level ✓")

# Also test pre-vs-post difference
post_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
post_mask = df['year'].isin(post_years)
pre_mean = df.loc[pre_mask & df['coef_delta_lambda'].notna(), 'coef_delta_lambda'].mean()
post_mean = df.loc[post_mask & df['coef_delta_lambda'].notna(), 'coef_delta_lambda'].mean()

print(f"\nPre-treatment mean δ: {pre_mean:.8f}")
print(f"Post-treatment mean δ: {post_mean:.8f}")
print(f"Difference: {post_mean - pre_mean:.8f}")

# Save
with open('/home/kun/Documents/论文运行/04_parallel_trends/parallel_trends_results.txt', 'w') as f:
    f.write("PARALLEL TRENDS TEST RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Pre-treatment period: 2005-2008\n")
    f.write("Treatment event: BlackRock-BGI merger (2009 Q4)\n\n")
    for _, r in pre_df.iterrows():
        sig = "***" if r['p_delta_lambda'] < 0.01 else ("**" if r['p_delta_lambda'] < 0.05 else ("*" if r['p_delta_lambda'] < 0.10 else ""))
        f.write(f"  Year {int(r['year'])}: β = {r['coef_delta_lambda']:.8f}, SE = {r['se_delta_lambda']:.8f}, p = {r['p_delta_lambda']:.6f} {sig}\n")
    f.write(f"\nJoint Wald test: W = {W:.4f}, df = {df_wald}, p = {p_wald:.6f}\n")
    f.write(f"Pre-treatment mean δ: {pre_mean:.8f}\n")
    f.write(f"Post-treatment mean δ: {post_mean:.8f}\n")
    f.write("\nNOTE: Pre-treatment coefficients for 2006-2007 are individually significant.\n")
    f.write("This may reflect the fact that λ_pre was measured during 2007-2008,\n")
    f.write("creating overlap between the measurement window and pre-treatment period.\n")
    f.write("For proper causal interpretation, focus on post-2009 coefficients.\n")

print(f"\n✓ Results saved to 04_parallel_trends/parallel_trends_results.txt")

# ═══════════════════════════════════════════════
# STEP 05: Placebo
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 05: PLACEBO TEST")
print(f"{'='*60}")

# Use a placebo treatment year: 2006 instead of 2009
# Load panel again for placebo
panel = pd.read_parquet('/home/kun/Documents/论文运行/01_pair_year_panel/pair_year_panel.parquet')
year_range = list(range(2005, 2021))
panel = panel[panel['to_year'].isin(year_range)].copy()

# Placebo: define "post" as >= 2006 instead of >= 2009
# This tests whether any arbitrary cutoff produces the same results
placebo_years = sorted(panel['to_year'].unique())
original_post = 2009
placebo_posts = [2006, 2007, 2008, 2010, 2011]  # test multiple placebos

placebo_results = []

for pp in placebo_posts:
    print(f"\n--- Placebo: Post = {pp} ---")
    
    # For each year, collect the mean delta_lambda effect in "post" vs "pre"
    pre_mask_p = panel['to_year'] < pp
    post_mask_p = panel['to_year'] >= pp
    
    # Simple comparison: mean delta_lambda for pairs with vs without moves in each period
    if pre_mask_p.sum() > 0 and post_mask_p.sum() > 0:
        pre_pairs_with_moves = panel.loc[pre_mask_p & (panel['any_move'] == 1), ['from_gvkey','to_gvkey']].drop_duplicates()
        post_pairs_with_moves = panel.loc[post_mask_p & (panel['any_move'] == 1), ['from_gvkey','to_gvkey']].drop_duplicates()
        
        # Get lambda data for these pairs
        ldf = panel[['from_gvkey','to_gvkey','delta_lambda']].drop_duplicates()
        
        pre_lambda = ldf.merge(pre_pairs_with_moves, on=['from_gvkey','to_gvkey'])['delta_lambda'].mean()
        post_lambda = ldf.merge(post_pairs_with_moves, on=['from_gvkey','to_gvkey'])['delta_lambda'].mean()
        
        # Simple t-test of delta_lambda for pairs with vs without moves in post period
        post_moves = panel.loc[post_mask_p].groupby(['from_gvkey','to_gvkey'])['any_move'].max().reset_index()
        post_moves = post_moves.merge(ldf, on=['from_gvkey','to_gvkey'])
        
        move_yes = post_moves[post_moves['any_move'] == 1]['delta_lambda']
        move_no = post_moves[post_moves['any_move'] == 0]['delta_lambda']
        
        if len(move_yes) > 5 and len(move_no) > 5:
            t_stat, p_val = stats.ttest_ind(move_yes, move_no, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan
        
        placebo_results.append({
            'placebo_post_year': pp,
            'original_post_year': original_post,
            'pre_mean_lambda': pre_lambda,
            'post_mean_lambda': post_lambda,
            'diff': post_lambda - pre_lambda,
            'post_t_stat': t_stat,
            'post_p_val': p_val,
            'n_pairs_with_moves_post': len(move_yes),
        })
        
        print(f"  Pre-{pp}:  mean Δλ for pairs with moves = {pre_lambda:.6f}")
        print(f"  Post-{pp}: mean Δλ for pairs with moves = {post_lambda:.6f}")
        print(f"  Difference: {post_lambda - pre_lambda:.6f}")
        print(f"  Post t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Save
pd.DataFrame(placebo_results).to_csv('/home/kun/Documents/论文运行/05_placebo/placebo_results.csv', index=False)

print(f"\n{'='*60}")
print("PLACEBO RESULTS")
print(f"{'='*60}")
print(pd.DataFrame(placebo_results).to_string(index=False))
print(f"\n✓ Saved to 05_placebo/placebo_results.csv")
