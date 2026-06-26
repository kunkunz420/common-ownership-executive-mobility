"""Supplemental robustness checks for IESE conference Q&A defense.

1. Drop tenure → recovers ~70% of lost observations
2. Post=2011 cutoff (not 2010)
3. CEO vs non-CEO subsamples
4. Winsorized BR_beta_change (drop top/bottom 1%)
5. All on long-difference window
"""

import gc
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

from lib.config import PANEL_CSV, EXEC_CLEAN_CSV, OUTPUT, LOGS, TR13F_CSV, BLACKROCK_PATTERNS
from lib.helpers import setup_logger

logger = setup_logger("robustness", LOGS / "robustness_checks.log")

RANDOM_SEED = 42


def rebuild_panel():
    """Same panel build as Phase 8b / Phase 9."""
    logger.info("Rebuilding panel...")

    br_mgrnos = set()
    for chunk in pd.read_csv(TR13F_CSV, chunksize=100_000, low_memory=False, usecols=["mgrno", "mgrname"]):
        names = chunk["mgrname"].str.upper().fillna("")
        mask = names.str.contains("|".join(BLACKROCK_PATTERNS))
        br_mgrnos.update(chunk.loc[mask, "mgrno"].unique())

    betas = pd.read_csv(OUTPUT / "beta_aggregated.csv", low_memory=False)
    br_firms = set(betas[betas["mgrno"].isin(br_mgrnos)]["gvkey"].unique())
    del betas; gc.collect()

    betas2 = pd.read_csv(OUTPUT / "beta_aggregated.csv", low_memory=False)
    br_betas = betas2[betas2["mgrno"].isin(br_mgrnos)].copy()
    br_betas["year"] = br_betas["quarter"].str[:4].astype(int)
    br_annual = br_betas.groupby(["gvkey", "year"], as_index=False)["beta"].sum()
    br_annual.rename(columns={"beta": "br_beta"}, inplace=True)
    pre = br_annual[br_annual["year"].between(2007, 2008)].groupby("gvkey")["br_beta"].mean().reset_index()
    pre.columns = ["gvkey", "br_beta_pre"]
    post = br_annual[br_annual["year"].between(2010, 2011)].groupby("gvkey")["br_beta"].mean().reset_index()
    post.columns = ["gvkey", "br_beta_post"]
    br_change = pre.merge(post, on="gvkey", how="outer")
    br_change["br_beta_pre"] = br_change["br_beta_pre"].fillna(0)
    br_change["br_beta_post"] = br_change["br_beta_post"].fillna(0)
    br_change["br_beta_change"] = br_change["br_beta_post"] - br_change["br_beta_pre"]
    del br_betas, br_annual, betas2; gc.collect()

    panel = pd.read_csv(PANEL_CSV, low_memory=False)
    panel = panel.merge(br_change[["gvkey", "br_beta_change"]], on="gvkey", how="left")
    panel["br_beta_change"] = panel["br_beta_change"].fillna(0)
    panel["gvkey_cat"] = panel["gvkey"].astype(str)
    panel["year_cat"] = panel["year"].astype(str)
    panel["sic2"] = panel["sich"].fillna(0).astype(int) // 100
    panel["sic2_cat"] = panel["sic2"].astype(str)
    panel["rd_intensity"] = panel["rd_intensity"].fillna(0)
    panel = panel[panel["year"].between(2005, 2014)].copy()

    # Identify CEO
    panel["is_ceo"] = (panel["co_per_rol"].astype(str).str.upper().str.contains("CEO")).astype(int)

    return panel


def run_checks(panel):
    logger.info("=" * 70)
    logger.info("ROBUSTNESS CHECKS")
    logger.info("=" * 70)

    exec_ctrls = "age + tenure + male"
    exec_no_tenure = "age + male"
    firm_ctrls = "log_assets + leverage + rd_intensity"
    ind_yr_fe = "C(sic2_cat) + C(year_cat)"

    # ── Long-diff window ──
    ld = panel[~panel["year"].between(2008, 2011)].copy()
    ld["post_ld"] = (ld["year"] >= 2012).astype(int)
    ld["br_change_x_post_ld"] = ld["br_beta_change"] * ld["post_ld"]

    # ── Check 1: Drop tenure (biggest N killer) ──────────────────
    logger.info("--- Check 1: Drop tenure from controls ---")
    # Use looser drop: keep rows even if tenure is missing
    loose_vars = ["mobility_event", "br_beta_change", "age", "male",
                  "log_assets", "leverage", "rd_intensity"]
    ld_loose = ld.dropna(subset=loose_vars).copy()
    logger.info("  With tenure in dropna: %d rows", len(ld.dropna(subset=loose_vars + ["tenure"])))
    logger.info("  Without tenure in dropna: %d rows", len(ld_loose))

    for spec_name, formula, df_use in [
        ("+Firm (no tenure)",
         f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + {exec_no_tenure} + {firm_ctrls}",
         ld_loose),
        ("+FE (no tenure)",
         f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + {exec_no_tenure} + {firm_ctrls} + {ind_yr_fe}",
         ld_loose),
        ("+Firm (with tenure, baseline)",
         f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + {exec_ctrls} + {firm_ctrls}",
         ld.dropna(subset=loose_vars + ["tenure"])),
    ]:
        try:
            m = smf.ols(formula, data=df_use).fit(
                cov_type="cluster", cov_kwds={"groups": df_use["gvkey_cat"]}
            )
            b = m.params.get("br_change_x_post_ld", np.nan)
            se = m.bse.get("br_change_x_post_ld", np.nan)
            p = m.pvalues.get("br_change_x_post_ld", np.nan)
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            logger.info("  %-30s | coef=%+9.6f  SE=%.6f  p=%.4f  %s  N=%d  R²=%.4f",
                        spec_name, b, se, p, stars, int(m.nobs), m.rsquared)
        except Exception as e:
            logger.info("  %-30s | FAILED: %s", spec_name, str(e)[:150])

    # ── Check 2: Post=2011 cutoff ────────────────────────────────
    logger.info("--- Check 2: Post=2011 instead of Post=2010 ---")
    panel_p11 = panel[panel["year"] != 2009].copy()  # drop ambiguous 2009
    panel_p11["post_2011"] = (panel_p11["year"] >= 2011).astype(int)
    panel_p11["br_x_post11"] = panel_p11["br_beta_change"] * panel_p11["post_2011"]
    key_vars = ["mobility_event", "br_beta_change", "age", "tenure", "male",
                "log_assets", "leverage", "rd_intensity"]
    panel_p11 = panel_p11.dropna(subset=key_vars)

    for spec_name, formula in [
        ("OLS", f"mobility_event ~ br_beta_change + post_2011 + br_x_post11"),
        ("+Firm", f"mobility_event ~ br_beta_change + post_2011 + br_x_post11 + {exec_ctrls} + {firm_ctrls}"),
        ("+FE", f"mobility_event ~ br_beta_change + post_2011 + br_x_post11 + {exec_ctrls} + {firm_ctrls} + {ind_yr_fe}"),
    ]:
        try:
            m = smf.ols(formula, data=panel_p11).fit(
                cov_type="cluster", cov_kwds={"groups": panel_p11["gvkey_cat"]}
            )
            b = m.params.get("br_x_post11", np.nan)
            se = m.bse.get("br_x_post11", np.nan)
            p = m.pvalues.get("br_x_post11", np.nan)
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            logger.info("  %-20s | coef=%+9.6f  SE=%.6f  p=%.4f  %s  N=%d  R²=%.4f",
                        spec_name, b, se, p, stars, int(m.nobs), m.rsquared)
        except Exception as e:
            logger.info("  %-20s | FAILED: %s", spec_name, str(e)[:150])

    # ── Check 3: CEO vs non-CEO ──────────────────────────────────
    logger.info("--- Check 3: CEO vs non-CEO subsamples (long-diff) ---")
    ld_kv = ld.dropna(subset=["mobility_event", "br_beta_change", "age", "tenure", "male",
                               "log_assets", "leverage", "rd_intensity"])
    for label, mask in [("CEO", ld_kv["is_ceo"] == 1), ("Non-CEO", ld_kv["is_ceo"] == 0)]:
        df_sub = ld_kv[mask]
        if len(df_sub) < 500:
            logger.info("  %-8s | insufficient N (%d)", label, len(df_sub))
            continue
        try:
            m = smf.ols(
                f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + "
                f"{exec_ctrls} + {firm_ctrls} + {ind_yr_fe}",
                data=df_sub,
            ).fit(cov_type="cluster", cov_kwds={"groups": df_sub["gvkey_cat"]})
            b = m.params.get("br_change_x_post_ld", np.nan)
            se = m.bse.get("br_change_x_post_ld", np.nan)
            p = m.pvalues.get("br_change_x_post_ld", np.nan)
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            logger.info("  %-8s | coef=%+9.6f  SE=%.6f  p=%.4f  %s  N=%d  mob=%.4f",
                        label, b, se, p, stars, int(m.nobs), df_sub["mobility_event"].mean())
        except Exception as e:
            logger.info("  %-8s | FAILED: %s", label, str(e)[:150])

    # ── Check 4: Winsorized BR_beta_change ──────────────────────
    logger.info("--- Check 4: Winsorize BR_beta_change (drop top/bottom 1%) ---")
    p1 = ld_kv["br_beta_change"].quantile(0.01)
    p99 = ld_kv["br_beta_change"].quantile(0.99)
    ld_w = ld_kv[(ld_kv["br_beta_change"] >= p1) & (ld_kv["br_beta_change"] <= p99)].copy()
    ld_w["br_change_x_post_ld"] = ld_w["br_beta_change"] * ld_w["post_ld"]
    logger.info("  Winsorized: drop BR_change < %.6f or > %.6f, keep %d rows (from %d)",
                p1, p99, len(ld_w), len(ld_kv))

    for spec_name, formula in [
        ("OLS", f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld"),
        ("+Firm", f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + {exec_ctrls} + {firm_ctrls}"),
        ("+FE", f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + {exec_ctrls} + {firm_ctrls} + {ind_yr_fe}"),
    ]:
        try:
            m = smf.ols(formula, data=ld_w).fit(
                cov_type="cluster", cov_kwds={"groups": ld_w["gvkey_cat"]}
            )
            b = m.params.get("br_change_x_post_ld", np.nan)
            se = m.bse.get("br_change_x_post_ld", np.nan)
            p = m.pvalues.get("br_change_x_post_ld", np.nan)
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            logger.info("  %-20s | coef=%+9.6f  SE=%.6f  p=%.4f  %s  N=%d  R²=%.4f",
                        spec_name, b, se, p, stars, int(m.nobs), m.rsquared)
        except Exception as e:
            logger.info("  %-20s | FAILED: %s", spec_name, str(e)[:150])

    logger.info("\nRobustness checks complete.")


def main():
    np.random.seed(RANDOM_SEED)
    panel = rebuild_panel()
    run_checks(panel)


if __name__ == "__main__":
    main()
