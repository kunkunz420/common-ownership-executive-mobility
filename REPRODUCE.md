# 完整复现指南

> 本文档可以使得复现我的完整的工作流程。

---

## 环境要求

```bash
Python 3.12+
pip install pandas numpy statsmodels matplotlib pyarrow scipy
系统内存 ≥ 16 GB
```

---

## 第一步：WRDS 数据管道

构建 λ_jk 共同所有权指标和分析面板。

```bash
# 按顺序执行：
python3 01_build_link_table.py      # gvkey 匹配表
python3 02_clean_13f_mhhi.py        # 13F 清洗
python3 03_clean_executive.py       # ExecuComp 清洗
python3 04_merge_panel.py           # 合并面板
python3 10a_build_lambda_jk.py      # 构建 λ_jk（核心）
python3 08_directional_mobility_did.py  # Firm-level DiD
python3 09_long_diff_verify.py      # 长差分验证
python3 robustness_checks.py        # 稳健性
```

输出目录：`沃顿数据/数据和清洗/data_clean/output/`

---

## 第二步：Orbis 配对层面分析

### 2.1 构建 Pair × Year 面板

```bash
python3 analysis/01_pair_panel/build_pair_year_panel.py
```

输入：`orbis_moves_with_lambda.parquet`、`lambda_delta.parquet`
输出：`pair_year_panel.parquet` (27,824,524 行)

步骤：
1. 从 Orbis 跳槽记录提取年份
2. 构建 pair × year 笛卡尔积
3. 合并 λ_jk 数据
4. 生成 Origin×Year 和 Dest×Year FE 键

### 2.2 FWL 吸收 + Event Study

```bash
python3 analysis/02_fwl/fwl_event_study.py
```

方法论：
- 逐年切分面板 (2005-2020)
- 对每年截面做 Gauss-Seidel 迭代 demean：
  - 减去 origin firm 组内均值
  - 减去 dest firm 组内均值
  - 交替至收敛 (30 次迭代)
- OLS 回归：`any_move ~ delta_lambda + lambda_jk_pre`

### 2.3 平行趋势 + 安慰剂

```bash
python3 analysis/04_parallel_trends/parallel_trends_placebo.py
```

- 平行趋势：Joint Wald 检验，H0: δ_{2005}=δ_{2006}=δ_{2007}=δ_{2008}=0
- 安慰剂：将处理截止日置换至 2006/2007/2008/2010/2011

### 2.4 基准 Logit

```bash
python3 analysis/08_logit/baseline_logit.py
```

截面 logit：`AnyMove_jk ~ delta_lambda + lambda_jk_pre`

### 2.5 RESET 规范检验

```bash
python3 analysis/09_reset/reset_test.py
```

- RESET test (OLS 规范检验)
- Link test (Logit 规范检验)

### 2.6 稳健性检验

```bash
python3 analysis/10_robustness/robustness_checks.py
```

17 项检验：winsorize、正/负 Δλ 子样本、四分位、λ_pre>0、聚类 SE、时间拆分、异常值剔除等

### 2.7 可视化

```bash
python3 analysis/06_plots/plot_and_summary.py
```

生成 Event Study 系数图（δ_τ vs 年份，95% CI），2009 年红线。

---






## 文件依赖关系

```
01_build_link_table.py ──▶ 02_clean_13f ──▶ 10a_build_lambda_jk ──▶ lambda_delta.parquet
                                                                          │
03_clean_executive.py ──▶ 04_merge_panel ──▶ 08_directional_mobility ──┘
                                             09_long_diff_verify
                                             robustness_checks

lambda_delta.parquet ──┐
orbis_moves.parquet ───┤
                        ├──▶ 01_pair_panel ──▶ pair_year_panel.parquet
                        │
pair_year_panel ────────┤
                        ├──▶ 02_fwl ──▶ 03_event_study
                        ├──▶ 04_parallel_trends + 05_placebo
                        ├──▶ 08_logit
                        ├──▶ 09_reset
                        └──▶ 10_robustness ──▶ 06_plots ──▶ 07_tables
```

---

## 关键结果速查

| 层面 | 规格 | 系数 | p 值 | N |
|------|------|------|------|---|
| Firm | BR_Δ × Post | +0.0731 | 0.002 | 31,959 |
| Firm | 长差分 | +0.096 | 0.005 | 18,108 |
| Pair | Logit Δλ_jk | +0.582 | <10⁻¹⁸ | 1,070,174 |
| Pair | Pre-2010 时间拆分 | +0.0001 | 0.063 | — |
| Pair | Post-2009 时间拆分 | +0.0017 | <0.001 | — |

---

## 数据来源

| 数据库 | 规模 |
|--------|------|
| Thomson Reuters 13F | 58,929,118 条 |
| ExecuComp | 244,857 条 |
| Compustat | 25,874 企业 (231,365 行 firm-year) |
| CRSP Stock / Names | 1,821,454 行月度数据 + 40,518 个 PERMNO |
| Orbis (BvD) | 14,220 企业，1,417,530 条 person-role |

