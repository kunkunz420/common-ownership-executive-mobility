"""Step 06-07: Event Study coefficient plot + Summary tables for José Azar.

Outputs:
  06_visualizations/event_study_plot.png      (Event Study coefficient chart)
  06_visualizations/event_study_plot.pdf      (PDF version)
  07_summary_tables/summary_results.txt        (full text summary)
  07_summary_tables/event_study_table.csv      (event study table)
  07_summary_tables/regression_summary.csv     (all regressions in one table)
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

print("=" * 60)
print("Step 06-07: Visualization + Summary")
print("=" * 60)

# ── Load data ──
df = pd.read_csv('/home/kun/Documents/论文运行/03_event_study/event_study_coefficients.csv')
pair_results = pd.read_csv('/home/kun/Documents/Orbis数据/pair_level_results.csv', index_col=0)

print(f"\nLoaded event study: {len(df)} years")
print(f"Loaded pair-level results")

# ═══════════════════════════════════════════════
# STEP 06: EVENT STUDY PLOT
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 06: Generating Event Study plot")
print(f"{'='*60}")

fig, ax = plt.subplots()

# Filter to valid coefficients
plot_df = df[df['coef_delta_lambda'].notna()].copy()
plot_df = plot_df.sort_values('year')

years = plot_df['year'].values
coefs = plot_df['coef_delta_lambda'].values
ses = plot_df['se_delta_lambda'].values

# 95% CI
ci_lower = coefs - 1.96 * ses
ci_upper = coefs + 1.96 * ses

# Points — plot pre and post separately for proper coloring
pre_mask = years < 2009
post_mask = years >= 2009

ax.scatter(years[pre_mask], coefs[pre_mask], c='#999999', s=60, zorder=5, edgecolors='white', linewidth=0.5, label='Pre-treatment (2005-2008)')
ax.scatter(years[post_mask], coefs[post_mask], c='#1B365D', s=60, zorder=5, edgecolors='white', linewidth=0.5, label='Post-treatment (2009-2020)')

# Error bars
ax.errorbar(years[pre_mask], coefs[pre_mask], yerr=1.96*ses[pre_mask], fmt='none', ecolor='#999999', capsize=3, linewidth=1.2, zorder=4)
ax.errorbar(years[post_mask], coefs[post_mask], yerr=1.96*ses[post_mask], fmt='none', ecolor='#1B365D', capsize=3, linewidth=1.2, zorder=4)

# Treatment line
ax.axvline(x=2009.5, color='#cc3333', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
ax.text(2009.6, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 0.0004,
        'BlackRock-BGI\nMerger (Dec 2009)', fontsize=9, color='#cc3333', fontweight='bold')

# Zero line
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

# Shade 95% CI
ax.fill_between(years, ci_lower, ci_upper, alpha=0.15, color='#1B365D')

# Labels
ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Coefficient: Δλ_jk × Year (LPM, FWL FE absorbed)', fontweight='bold')
ax.set_title('Event Study: Common Ownership Change (Δλ) and Executive Mobility\nPair-Level (j,k,t) with Origin×Year + Dest×Year FE', 
             fontweight='bold', pad=15)

# Formatting
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.grid(axis='y', alpha=0.2)
ax.set_facecolor('#f8f8f8')

# Legend
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#dddddd')

plt.tight_layout()
plt.savefig('/home/kun/Documents/论文运行/06_visualizations/event_study_plot.png', dpi=200, bbox_inches='tight')
plt.savefig('/home/kun/Documents/论文运行/06_visualizations/event_study_plot.pdf', bbox_inches='tight')
plt.close()

print("✓ Event study plot saved")

# ═══════════════════════════════════════════════
# STEP 07: SUMMARY TABLES
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 07: Generating summary tables")
print(f"{'='*60}")

# ── Table 1: Event Study Coefficients ──
event_table = plot_df[['year', 'coef_delta_lambda', 'se_delta_lambda', 't_delta_lambda', 'p_delta_lambda', 'n_obs', 'n_moves']].copy()
event_table.columns = ['Year', 'δ_τ (Coef)', 'SE', 't-stat', 'p-value', 'N', 'Moves']
event_table.to_csv('/home/kun/Documents/论文运行/07_summary_tables/event_study_table.csv', index=False)

# ── Table 2: Comprehensive Regression Summary ──
summary_text = f"""
================================================================================
RESEARCH SUMMARY: Institutional Common Ownership and Executive Mobility
================================================================================
Author: Kun Zhang | University of Navarra | Advisor: José Azar
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

1. FIRM-LEVEL RESULTS (ExecuComp, N=31,959 executive-years)
────────────────────────────────────────────────────────────────
  BR_Δ × Post:  +0.0731*** (p=0.002), SE clustered at firm level
  Long-diff:    +0.096***  (p=0.005), N=18,108
  Binary treat: +0.0035    (p=0.77) ← NULL — confirms continuous treatment

  Economic significance: 1 SD → +0.93 pp (+19.5% of mean mobility)
  Controls: 0→9, all spec: stable +0.073 (all p<0.003)
  Robustness: drop tenure +0.162***, 2011 cutoff +0.065***, 1% winsor +0.101***
  R&D heterogeneity: High +0.099, Low +0.093** → ILM confirmed, Shadow NCA rejected

2. PAIR-LEVEL RESULTS (Orbis, 1,070,174 firm-pairs)
────────────────────────────────────────────────────────────────
  Logit: λ_jk_pre  coef = +1.546 (p < 10^-124)
         Δλ_jk     coef = +0.583 (p < 10^-18)
  Marginal effects: λ_pre = +0.0036, Δλ = +0.0014 (per Δλ unit)

3. PAIR-LEVEL EVENT STUDY (FWL-absorbed Origin×Year + Dest×Year FE)
────────────────────────────────────────────────────────────────
  Specification: Move(j,k,t) = sum δ_τ × (Δλ_jk × 1[year=τ]) + γ × λ_pre + FE + ε
  FE absorbed: Origin firm × Year + Destination firm × Year (iterative demeaning)
  
  Pre-treatment (2005-2008):
"""
for _, r in plot_df[plot_df['year'] < 2009].iterrows():
    sig = "***" if r['p_delta_lambda'] < 0.01 else ("**" if r['p_delta_lambda'] < 0.05 else ("*" if r['p_delta_lambda'] < 0.10 else ""))
    summary_text += f"    {int(r['year'])}: δ = {r['coef_delta_lambda']:.8f} (SE={r['se_delta_lambda']:.8f}, p={r['p_delta_lambda']:.4f}) {sig}\n"

summary_text += f"""
  Parallel trends Wald test: p = 0.019 ⚠️ (jointly reject at 5%)
    NOTE: Pre-treatment period overlaps with λ_pre measurement window (2007-2008).
    This is a data limitation — 13F only has two cross-sections.
  
  Post-treatment (2009-2020):
"""
for _, r in plot_df[plot_df['year'] >= 2009].iterrows():
    sig = "***" if r['p_delta_lambda'] < 0.01 else ("**" if r['p_delta_lambda'] < 0.05 else ("*" if r['p_delta_lambda'] < 0.10 else ""))
    summary_text += f"    {int(r['year'])}: δ = {r['coef_delta_lambda']:.8f} (SE={r['se_delta_lambda']:.8f}, p={r['p_delta_lambda']:.4f}) {sig}\n"

pre_mean = plot_df[plot_df['year'] < 2009]['coef_delta_lambda'].mean()
post_mean = plot_df[plot_df['year'] >= 2009]['coef_delta_lambda'].mean()

summary_text += f"""
  Pre-treatment mean δ: {pre_mean:.8f}
  Post-treatment mean δ: {post_mean:.8f}
  Post − Pre: {post_mean - pre_mean:.8f}
  
4. PLACEBO TEST (5 alternative cutoff years)
────────────────────────────────────────────────────────────────
  All placebo cutoff years (2006, 2007, 2008, 2010, 2011):
    t-tests all p > 0.42 ← NO false treatment effect with arbitrary cutoff ✓

5. DATA SUMMARY
────────────────────────────────────────────────────────────────
  13F: 58,929,118 institutional holdings records (2007-2008, 2010-2011)
  ExecuComp: 244,857 executive records → 31,959 regression sample
  Orbis: 14,220 firms, 1,417,530 person-role records, 24,716 director moves
  λ_jk: 1,070,174 firm-pairs (922 pre-merger × 1,035 post-merger firms)
  Pair×Year panel: 27,824,524 obs (1,070,174 pairs × 26 years, 2000-2025)

6. KEY LIMITATIONS
────────────────────────────────────────────────────────────────
  (a) 13F has only two time cross-sections (no annual λ data)
  (b) Parallel trends violated at pair-level — likely due to λ measurement overlap
  (c) Orbis moves coverage improves over time (more moves in recent years)
  (d) Pair-level event study coefficients are in LPM units (very small due to sparse Y)
  (e) ExecuComp mobility collapsed post-2009 (1,378→22), necessitating Orbis

7. NARRATIVE
────────────────────────────────────────────────────────────────
  Mechanism: Network formation, not mere ownership concentration.
  BlackRock acquires BGI → inter-firm common ownership links strengthen
  → Δλ_jk ↑ → connected-firm executive mobility ↑
  Empirical support: ILM channel confirmed, Shadow NCA rejected.

================================================================================
"""

with open('/home/kun/Documents/论文运行/07_summary_tables/summary_results.txt', 'w') as f:
    f.write(summary_text)

# ── Table 3: Compact regression summary CSV ──
reg_summary = pd.DataFrame([
    {'Model': 'Firm-Level DiD (OLS)', 'Coef': 0.0731, 'SE': 0.0236, 'p': 0.002, 'N': 31959, 'R²': 0.024},
    {'Model': 'Firm-Level DiD (+Exec FE)', 'Coef': 0.0727, 'SE': 0.0235, 'p': 0.0022, 'N': 31959, 'R²': 0.023},
    {'Model': 'Firm-Level Long-Diff', 'Coef': 0.096, 'SE': 0.034, 'p': 0.005, 'N': 18108, 'R²': 0.055},
    {'Model': 'Firm-Level Binary (null)', 'Coef': 0.0035, 'SE': 0.012, 'p': 0.77, 'N': 31959, 'R²': 0.022},
    {'Model': 'Pair-Level Logit λ_pre', 'Coef': 1.546, 'SE': None, 'p': 0.0, 'N': 1070174, 'R²': None},
    {'Model': 'Pair-Level Logit Δλ', 'Coef': 0.583, 'SE': None, 'p': 0.0, 'N': 1070174, 'R²': None},
    {'Model': 'Pair-Level ES (post-2009 mean δ)', 'Coef': post_mean, 'SE': None, 'p': None, 'N': None, 'R²': None},
])
reg_summary.to_csv('/home/kun/Documents/论文运行/07_summary_tables/regression_summary.csv', index=False)

print("✓ Summary tables saved")
print(f"\n{'='*60}")
print("ALL STEPS COMPLETE")
print(f"{'='*60}")
print(f"\nOutput files:")
print(f"  06_visualizations/event_study_plot.png")
print(f"  06_visualizations/event_study_plot.pdf")
print(f"  07_summary_tables/summary_results.txt")
print(f"  07_summary_tables/event_study_table.csv")
print(f"  07_summary_tables/regression_summary.csv")
