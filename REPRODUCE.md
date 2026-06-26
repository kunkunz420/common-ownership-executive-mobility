# Reproduction Guide

> This document provides a complete walkthrough for reproducing the analysis pipeline.

---

## Requirements

```bash
Python 3.12+
pip install pandas numpy statsmodels matplotlib pyarrow scipy
System RAM ≥ 16 GB
```

---

## Step 1: WRDS Data Pipeline

Constructs the λ_jk common ownership measure and the main analysis panel.

```bash
# Run in order:
python3 01_build_link_table.py          # gvkey crosswalk table
python3 02_clean_13f_mhhi.py            # 13F data cleaning
python3 03_clean_executive.py           # ExecuComp cleaning
python3 04_merge_panel.py               # Panel merge
python3 10a_build_lambda_jk.py          # Build λ_jk (core step)
python3 08_directional_mobility_did.py  # Firm-level DiD
python3 09_long_diff_verify.py          # Long-difference verification
python3 robustness_checks.py            # Robustness checks
```

Output directory: `wharton_data/data_clean/output/`

---

## Step 2: Orbis Pair-Level Analysis

### 2.1 Build Pair × Year Panel

```bash
python3 analysis/01_pair_panel/build_pair_year_panel.py
```

**Inputs:** `orbis_moves_with_lambda.parquet`, `lambda_delta.parquet`  
**Output:** `pair_year_panel.parquet` (27,824,524 rows)

Steps:
1. Extract move years from Orbis job-transition records
2. Construct the Cartesian pair × year product
3. Merge λ_jk data
4. Generate Origin×Year and Dest×Year fixed-effect keys

### 2.2 FWL Absorption + Event Study

```bash
python3 analysis/02_fwl/fwl_event_study.py
```

Methodology:
- Split panel by year (2005–2020)
- Apply Gauss-Seidel iterative demeaning to each annual cross-section:
  - Subtract within-group mean for origin firm
  - Subtract within-group mean for destination firm
  - Alternate until convergence (30 iterations)
- OLS regression: `any_move ~ delta_lambda + lambda_jk_pre`

### 2.3 Parallel Trends + Placebo Tests

```bash
python3 analysis/04_parallel_trends/parallel_trends_placebo.py
```

- **Parallel trends:** Joint Wald test, H₀: δ₂₀₀₅ = δ₂₀₀₆ = δ₂₀₀₇ = δ₂₀₀₈ = 0
- **Placebo:** Treatment cutoff reassigned to 2006, 2007, 2008, 2010, and 2011

### 2.4 Baseline Logit

```bash
python3 analysis/08_logit/baseline_logit.py
```

Cross-sectional logit: `AnyMove_jk ~ delta_lambda + lambda_jk_pre`

### 2.5 RESET Specification Tests

```bash
python3 analysis/09_reset/reset_test.py
```

- RESET test (OLS specification check)
- Link test (logit specification check)

### 2.6 Robustness Checks

```bash
python3 analysis/10_robustness/robustness_checks.py
```

17 checks: winsorization, positive/negative Δλ subsamples, quartile splits, λ_pre > 0 restriction, clustered standard errors, time splits, outlier exclusion, and others.

### 2.7 Visualization

```bash
python3 analysis/06_plots/plot_and_summary.py
```

Produces the event-study coefficient plot (δ_τ vs. year with 95% CI), with a 2009 reference line.

---

## File Dependency Map

```
01_build_link_table.py ──▶ 02_clean_13f ──▶ 10a_build_lambda_jk ──▶ lambda_delta.parquet
                                                                              │
03_clean_executive.py ──▶ 04_merge_panel ──▶ 08_directional_mobility ───────┘
                                              09_long_diff_verify
                                              robustness_checks

lambda_delta.parquet ───┐
orbis_moves.parquet ────┤
                         ├──▶ 01_pair_panel ──▶ pair_year_panel.parquet
                         │
pair_year_panel ─────────┤
                         ├──▶ 02_fwl ──▶ 03_event_study
                         ├──▶ 04_parallel_trends + 05_placebo
                         ├──▶ 08_logit
                         ├──▶ 09_reset
                         └──▶ 10_robustness ──▶ 06_plots ──▶ 07_tables
```

---

## Key Results at a Glance

| Level | Specification | Coefficient | *p*-value | *N* |
|---|---|---|---|---|
| Firm | BR_Δ × Post | +0.0731 | 0.002 | 31,959 |
| Firm | Long-difference | +0.096 | 0.005 | 18,108 |
| Pair | Logit Δλ_jk | +0.582 | < 10⁻¹⁸ | 1,070,174 |
| Pair | Pre-2010 time split | +0.0001 | 0.063 | — |
| Pair | Post-2009 time split | +0.0017 | < 0.001 | — |

---

## Data Sources

| Database | Scale |
|---|---|
| Thomson Reuters 13F | 58,929,118 records |
| ExecuComp | 244,857 records |
| Compustat | 25,874 firms (231,365 firm-year observations) |
| CRSP Stock / Names | 1,821,454 monthly observations · 40,518 PERMNOs |
| Orbis (Bureau van Dijk) | 14,220 firms · 1,417,530 person-role records |

**Note:** Raw data files are under license and not included in this repository. Contact the author for data access.
