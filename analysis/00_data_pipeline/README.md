# WRDS 数据管道 (Phase 1-10)

这些脚本构建了从原始 WRDS 数据到分析面板的完整管道。将脚本复制到 `/home/kun/Documents/沃顿数据/数据和清洗/所有源文件/data_clean/` 下运行。

## 执行顺序

| Phase | 脚本 | 功能 | 产出 |
|-------|------|------|------|
| 1 | `01_build_link_table.py` | 构建 gvkey-ticker 匹配表 | 8,150 gvkeys |
| 2 | `02_clean_13f_mhhi.py` | 清洗 13F 机构持股数据 | 57,888 MHHI obs |
| 3 | `03_clean_executive.py` | 清洗 ExecuComp 高管数据 | 244,857 行 |
| 4 | `04_merge_panel.py` | 合并形成分析面板 | executive-year panel |
| 8 | `08_directional_mobility_did.py` | **方向性流动 DiD (首次显著)** | +0.073*** |
| 9 | `09_long_diff_verify.py` | **长差分验证** | +0.096*** |
| 10a | `10a_build_lambda_jk.py` | **构建 λ_jk 配对数据** | lambda_delta.parquet |
| — | `robustness_checks.py` | 稳健性检验 | 10 项检验 |
| — | `br_beta_jump.py` | BlackRock beta 跳跃分析 | +0.013 (p=0.001) |

## 数据要求

这些脚本需要访问以下 WRDS 数据库：
- Compustat Fundamentals Annual
- ExecuComp
- Thomson Reuters 13F
- CRSP Stock / CRSP Names

数据文件不在本仓库中（受 WRDS 许可限制）。获取数据后，按上述顺序执行脚本即可复现。
