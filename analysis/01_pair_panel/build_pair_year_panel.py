"""Step 01: Build pair × year panel for Event Study.

Inputs:
  - orbis_moves_with_lambda.parquet  (Orbis moves + lambda_jk data)
  - lambda_delta.parquet              (full pair universe with lambda)

Outputs:
  - pair_year_panel.parquet    (pair × year panel: move indicator + lambda + FE keys)
  - panel_summary.csv          (summary statistics)
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ──
print("=" * 60)
print("Step 01: Building pair × year panel")
print("=" * 60)

moves = pd.read_parquet('/home/kun/Documents/Orbis数据/orbis_moves_with_lambda.parquet')
ldf   = pd.read_parquet('/home/kun/Documents/沃顿数据/数据和清洗/data_clean/output/lambda_jk/lambda_delta.parquet')

print(f"\nMoves with lambda: {len(moves):,} rows")
print(f"  Persons: {moves['person_bvd'].nunique():,}")
print(f"  Unique pairs: {moves.groupby(['from_gvkey','to_gvkey']).ngroups:,}")
print(f"Full pair universe: {len(ldf):,} pairs")

# ── 2. Extract year from move date ──
def extract_year(d):
    if pd.isna(d) or str(d).strip() == '':
        return None
    d = str(d).strip()
    # Formats: YYYY-MM-DD, YYYY-MM, or YYYY
    if '-' in d:
        try:
            return int(d.split('-')[0])
        except:
            return None
    try:
        return int(d)
    except:
        return None

moves['to_year'] = moves['to_date'].apply(extract_year)
moves_yr = moves[moves['to_year'].notna()].copy()
moves_yr['to_year'] = moves_yr['to_year'].astype(int)
print(f"\nMoves with valid year: {len(moves_yr):,}")
print(f"Year range: {moves_yr['to_year'].min()} – {moves_yr['to_year'].max()}")

# ── 3. Count moves per pair per year ──
move_counts = moves_yr.groupby(['from_gvkey', 'to_gvkey', 'to_year']).size().reset_index(name='n_moves')
move_counts['any_move'] = (move_counts['n_moves'] > 0).astype(int)
print(f"\nPair × year observations with moves: {len(move_counts):,}")

# ── 4. Build full panel: cartesian product of pairs × years ──
# Get unique years that have moves
move_years = sorted(move_counts['to_year'].unique())
print(f"Years with moves: {move_years}")

# Get all pairs from lambda data
pairs = ldf[['gvkey_j', 'gvkey_k', 'lambda_jk_pre', 'lambda_jk_post', 'delta_lambda']].copy()
pairs.columns = ['from_gvkey', 'to_gvkey', 'lambda_jk_pre', 'lambda_jk_post', 'delta_lambda']

# Create pair × year index
# To keep it manageable, only include years where we have reasonable coverage
# and pairs that have lambda data
year_range = list(range(2000, 2026))
pair_idx = pairs[['from_gvkey', 'to_gvkey']].drop_duplicates()

# Cross join pairs × years
panel = pair_idx.assign(key=1).merge(
    pd.DataFrame({'to_year': year_range, 'key': 1}), on='key'
).drop('key', axis=1)

print(f"\nFull panel (all pairs × all years): {len(panel):,} rows")

# ── 5. Merge move counts onto panel ──
panel = panel.merge(move_counts, on=['from_gvkey', 'to_gvkey', 'to_year'], how='left')
panel['n_moves'] = panel['n_moves'].fillna(0).astype(int)
panel['any_move'] = panel['any_move'].fillna(0).astype(int)

# ── 6. Merge lambda data ──
panel = panel.merge(
    pairs[['from_gvkey', 'to_gvkey', 'lambda_jk_pre', 'lambda_jk_post', 'delta_lambda']],
    on=['from_gvkey', 'to_gvkey'], how='inner'
)

# ── 7. Define treatment period ──
panel['post'] = (panel['to_year'] >= 2009).astype(int)  # 2009 Q3 base, 2010+ = post

# ── 8. Summary statistics ──
print(f"\n{'='*60}")
print("PANEL SUMMARY")
print(f"{'='*60}")
print(f"Total pair × year observations: {len(panel):,}")
print(f"Pairs with moves (any year): {panel[panel['n_moves']>0].groupby(['from_gvkey','to_gvkey']).ngroups:,}")
print(f"Total moves in panel: {panel['n_moves'].sum():,}")

print(f"\nBy year:")
yr_summary = panel.groupby('to_year').agg(
    obs=('any_move', 'count'),
    moves=('any_move', 'sum'),
    move_rate=('any_move', 'mean')
).round(6)
yr_summary['move_rate_pct'] = (yr_summary['move_rate'] * 100).round(4)
print(yr_summary)

print(f"\nLambda statistics:")
for col in ['lambda_jk_pre', 'lambda_jk_post', 'delta_lambda']:
    vals = panel.drop_duplicates(['from_gvkey','to_gvkey'])[col]
    print(f"  {col}: mean={vals.mean():.6f}, std={vals.std():.6f}, N={len(vals):,}")

# ── 9. Create FE columns for FWL absorption ──
# Origin × Year FE
panel['origin_year'] = panel['from_gvkey'].astype(str) + '_' + panel['to_year'].astype(str)
# Dest × Year FE
panel['dest_year'] = panel['to_gvkey'].astype(str) + '_' + panel['to_year'].astype(str)

# ── 10. Save ──
panel.to_parquet('/home/kun/Documents/论文运行/01_pair_year_panel/pair_year_panel.parquet', index=False)

# Save codebook
codebook = pd.DataFrame({
    'variable': ['from_gvkey', 'to_gvkey', 'to_year', 'n_moves', 'any_move',
                 'lambda_jk_pre', 'lambda_jk_post', 'delta_lambda', 'post',
                 'origin_year', 'dest_year'],
    'description': [
        'Origin firm gvkey',
        'Destination firm gvkey',
        'Year of move (or observation)',
        'Number of director moves j→k in year t',
        '1 if any move j→k in year t (outcome variable)',
        'Common ownership λ_jk before BlackRock-BGI merger (2007-2008)',
        'Common ownership λ_jk after BlackRock-BGI merger (2010-2011)',
        'Δλ = λ_post - λ_pre (treatment intensity)',
        '1 if year >= 2009 (post-treatment period)',
        'Origin firm × Year FE key',
        'Destination firm × Year FE key'
    ]
})
codebook.to_csv('/home/kun/Documents/论文运行/01_pair_year_panel/codebook.csv', index=False)

# Save summary
yr_summary.to_csv('/home/kun/Documents/论文运行/01_pair_year_panel/panel_summary.csv')

print(f"\n✓ Outputs saved to 01_pair_year_panel/")
print(f"  pair_year_panel.parquet ({len(panel):,} rows)")
print(f"  codebook.csv")
print(f"  panel_summary.csv")
