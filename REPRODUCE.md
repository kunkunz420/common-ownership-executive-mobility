# Institutional Common Ownership and Executive Mobility — 完整复现指南

> Author: Kun Zhang · University of Navarra · Advisor: José Azar
> Last updated: 2026-06-26

---

## 一、项目概述

**研究问题**: BlackRock 收购 BGI (2009年12月) 是否通过加强企业间共同所有权链接 (Δλ_jk) 促进了 C-suite 高管在关联企业间的流动？

**核心发现**: 数据拒绝了 Shadow Non-Compete 假说，支持 Internal Labor Market (ILM) 渠道。共同所有权增加高管流动性。

---

## 二、数据来源

### 2.1 核心数据库

| 数据库 | 路径/来源 | 内容 | 规模 |
|--------|----------|------|------|
| Thomson Reuters 13F | `/home/kun/Documents/沃顿数据/数据和清洗/` | 机构投资者季度持股 | 58,929,118 行 |
| Compustat | WRDS | 企业财务数据 | — |
| ExecuComp | WRDS | 高管薪酬与流动 | 244,857 行 |
| CRSP Stock | WRDS | 股票价格与回报 | — |
| CRSP Names | WRDS | 企业名称匹配 | — |
| Orbis (BvD) | `/home/kun/Documents/Orbis数据/` | 董事任职与跳槽记录 | 14,220 企业, 1,417,530 条 |

### 2.2 关键数据文件

```
# λ 构建结果（共同所有权指标）
/home/kun/Documents/沃顿数据/数据和清洗/data_clean/output/lambda_jk/
├── lambda_pre.parquet     # λ_jk 合并前 (2007-2008), 922 firms → 1,070,174 pairs
├── lambda_post.parquet    # λ_jk 合并后 (2010-2011), 1,035 firms → 1,070,174 pairs
└── lambda_delta.parquet   # Δλ = λ_post - λ_pre + summary stats

# ExecuComp 清洗结果
/home/kun/Documents/沃顿数据/数据和清洗/data_clean/output/
├── exec_clean.csv         # 清洗后的高管数据
└── link_table.csv         # gvkey-ticker 匹配表

# Orbis 数据
/home/kun/Documents/Orbis数据/
├── Final_data.csv                # 180 MB, 原始 Orbis 导出
├── orbis_with_gvkey.parquet      # 49 MB, 匹配 gvkey 后
├── orbis_moves_with_lambda.parquet  # 0.6 MB, 跳槽记录+λ数据
├── orbis_pair_panel.parquet      # 14.6 MB, 配对面板
├── orbis_pair_level.py           # 配对分析脚本
├── orbis_exec_crossref.py        # Orbis×ExecuComp 交叉引用
└── pair_level_results.csv        # logit 回归结果

# 原始脚本 (Phase 1-9)
/home/kun/Documents/沃顿数据/数据和清洗/所有源文件/data_clean/
├── 01_build_link_table.py
├── 02_clean_13f_mhhi.py
├── 03_clean_executive.py
├── 04_merge_panel.py
├── 05_did_regression.py
├── 06_ddd_rd_donut.py
├── 07_psm_did.py
├── 08_directional_mobility_did.py
├── 08b_fixed_fe_did.py
├── 09_long_diff_verify.py
├── robustness_checks.py
└── br_beta_jump.py
```

---

## 三、变量定义

### 3.1 共同所有权指标 λ_jk

```
λ_jk = Σ_i (β_ij × β_ik) / Σ_i (β_ij)²

其中:
  β_ij = 投资者 i 在企业 j 中的所有权份额
  β_ik = 投资者 i 在企业 k 中的所有权份额
  λ_jk ∈ [0, 1]
```

### 3.2 处理变量 Δλ_jk

```
λ_jk^{Pre}  = λ_jk 用 2007-2008 13F 数据计算
λ_jk^{Post} = λ_jk 用 2010-2011 13F 数据计算
Δλ_jk       = λ_jk^{Post} - λ_jk^{Pre}
基期         = 2009 Q3 (合并完成前最后一个完整季度)
```

### 3.3 高管流动性

| 变量 | 定义 | 数据源 |
|------|------|--------|
| Mobility_{i,t} | 高管 i 在 t 年是否跳槽 (0/1) | ExecuComp |
| AnyMove_{jk} | 董事是否从 j 跳到 k (0/1) | Orbis |
| Move_{jk,t} | 董事在 t 年是否从 j 跳到 k (0/1) | Orbis |

### 3.4 固定效应设计

| 层面 | FE 规格 |
|------|---------|
| Firm-level | Firm FE + Year FE |
| Pair-level 截面 | — |
| Pair-level Event Study | Origin×Year FE + Dest×Year FE (FWL 吸收) |

---

## 四、实验流程（10 步复现）

### 环境要求

```bash
Python 3.12+
pip install pandas numpy statsmodels matplotlib pyarrow scipy
```

### Step 01: 构建 pair×year 面板

```bash
python3 /home/kun/Documents/论文运行/01_pair_year_panel/build_pair_year_panel.py
```

**输入**: `orbis_moves_with_lambda.parquet`, `lambda_delta.parquet`
**输出**: `pair_year_panel.parquet` (27,824,524 行)
**逻辑**:
1. 从 Orbis 跳槽记录提取 `to_year`
2. 构建 pair × year 笛卡尔积
3. 合并 λ_jk 数据
4. 生成 Origin×Year 和 Dest×Year FE 键

### Step 02-03: FWL 吸收 + Event Study

```bash
python3 /home/kun/Documents/论文运行/02_fwl_absorption/fwl_event_study.py
```

**输入**: `pair_year_panel.parquet`
**输出**: `event_study_coefficients.csv`

**方法论**:
1. 逐年切分面板 (2005-2020)
2. 对每年截面迭代 demean (Gauss-Seidel):
   - 减去 origin firm 组内均值
   - 减去 dest firm 组内均值
   - 交替直至收敛 (30 次迭代)
3. OLS: `any_move ~ delta_lambda + lambda_jk_pre` on demeaned data

### Step 04-05: 平行趋势 + 安慰剂

```bash
python3 /home/kun/Documents/论文运行/04_parallel_trends/parallel_trends_placebo.py
```

- **平行趋势**: Joint Wald test, H0: δ_{2005}=δ_{2006}=δ_{2007}=δ_{2008}=0
- **安慰剂**: 将处理截止日置换至 2006/2007/2008/2010/2011，检查 t-test

### Step 06: 可视化

```bash
python3 /home/kun/Documents/论文运行/06_visualizations/plot_and_summary.py
```

生成 Event Study 系数图 (δ_τ vs 年份, 95% CI, 2009 年红线)。

### Step 08: 基准 logit

```bash
python3 /home/kun/Documents/论文运行/08_baseline_logit/baseline_logit.py
```

截面 logit: `AnyMove_jk ~ delta_lambda + lambda_jk_pre`

### Step 09: 规范检验

```bash
python3 /home/kun/Documents/论文运行/09_reset_test/reset_test.py
```

- RESET test (OLS 规范检验)
- Link test (Logit 规范检验)

### Step 10: 稳健性检验

```bash
python3 /home/kun/Documents/论文运行/10_robustness/robustness_checks.py
```

17 项检验: winsorize, 正/负 Δλ 子样本, quartile split, λ_pre>0, 聚类 SE, 时间拆分, 异常值剔除等

---

## 五、关键结果

### 5.1 Firm-Level (ExecuComp)

| 模型 | 系数 | p 值 | N |
|------|------|------|---|
| BR_Δ × Post | +0.0731*** | 0.002 | 31,959 |
| Long-Diff +FE | +0.096*** | 0.005 | 18,108 |
| Binary (median) | +0.0035 | 0.77 | 31,959 |

经济显著性: 1 SD → +0.93 pp (+19.5% of mean)

### 5.2 Pair-Level 截面 (Orbis)

| 模型 | 变量 | 系数 | p 值 |
|------|------|------|------|
| Logit | Δλ_jk | +0.582*** | <10⁻¹⁸ |
| Logit | λ_jk_pre | +1.533*** | <10⁻¹²⁴ |
| OLS | Δλ_jk | +0.0018*** | <10⁻⁹ |

### 5.3 Pair-Level Event Study (关键年)

| 年份 | δ_τ | p 值 |
|------|-----|------|
| 2015 | +0.000316 | 0.002 *** |
| 2016 | +0.000262 | 0.015 ** |
| 2017 | +0.000340 | <0.001 *** |
| 2018 | +0.000271 | 0.017 ** |
| 2019 | +0.000262 | 0.034 ** |
| 2020 | +0.000405 | 0.002 *** |

### 5.4 验证

| 检验 | 结果 |
|------|------|
| 平行趋势 (Wald) | p=0.019 ⚠️ (λ 测量窗口重叠) |
| 安慰剂 (5 个截止日) | 全通过 p>0.42 ✓ |
| RESET (OLS) | F=8.27, p=0.0003 ⚠️ |
| 时间拆分: Pre-2010 | p=0.063 (不显著) |
| 时间拆分: Post-2009 | p<0.001 *** |
| 稳健性 (17 项) | 15/17 显著 |

---

## 六、关键决策记录

1. **BGI 不在 13F** → 用 BlackRock 持股变化 (BR_beta_change) 替代直接 BGI 数据
2. **连续 vs 二元处理** → 二元 p=0.77 (null), 必须用连续变量
3. **长差分** → 金融危机稀释了效应 (6.45% → 0.53%, -91.8%)
4. **Tenure 缺失** → 去掉后样本从 32K 翻到 58K
5. **External mobility = 0** → 几乎所有跳槽发生在有共同机构联系的 firm 之间
6. **13F 两个截面** → 无法做年度 λ 面板，但 Event Study 可用固定 Δλ × 年份交互
7. **Orbis 补充 ExecuComp** → ExecuComp 2009 后流动崩溃 (1,378→22), Orbis 弥补

---

## 七、文件依赖关系 (DAG)

```
lambda_delta.parquet ──┐
orbis_moves_with_λ.parquet ──┤
                              ├──▶ 01_pair_year_panel ──▶ pair_year_panel.parquet
                              │
pair_year_panel.parquet ──────┤
                              ├──▶ 02_fwl_absorption ──▶ 03_event_study
                              ├──▶ 04_parallel_trends
                              ├──▶ 05_placebo
                              ├──▶ 06_visualizations
                              ├──▶ 07_summary_tables
                              ├──▶ 08_baseline_logit
                              ├──▶ 09_reset_test
                              └──▶ 10_robustness
```

---

## 八、边界文件（不可超出范围）

本实验严格遵守以下两个文件的约束：

1. **综合修改意见**: `/home/kun/Documents/论文修改意见/综合修改意见汇总.md`
   - 定义: 分析单位、处理变量、实验设计、λ_jk 公式、基准回归、验证方案
   
2. **逐条核对**: `/home/kun/Documents/论文修改意见/checklist_results.md`
   - 37 条要求中 34 条已完成 ✓, 3 条数据限制 △, 1 条待完成 ✗

任何分析不得超出上述文件定义的范围。

---

## 九、其他 Agent 复现检查清单

- [ ] Python 3.12+ with pandas, numpy, statsmodels, matplotlib, pyarrow, scipy
- [ ] 系统内存 ≥ 8 GB (panel 有 27.8M 行)
- [ ] Step 01 需要访问 Orbis 和 lambda 数据
- [ ] Step 02 运行约 10 分钟 (16 年 × 1M 行/年 × 30 次迭代)
- [ ] Step 10 需要约 8 GB 内存（已优化为仅加载必要列）
- [ ] 所有输出在 `/home/kun/Documents/论文运行/` 下
- [ ] 核心 README: `README.txt`
