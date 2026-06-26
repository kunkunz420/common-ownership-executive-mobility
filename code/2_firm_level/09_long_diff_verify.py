"""Phase 9: Reproduce long-difference results from working paper Tables 7.1 and 5.4.

- Drop 2008-2011 entirely
- Compare 2005-2007 (pure pre-crisis) vs 2012-2014 (pure post-crisis)
- Re-run continuous-treatment DiD and R&D heterogeneity
"""

import gc
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

from lib.config import (
    PANEL_CSV, EXEC_CLEAN_CSV, OUTPUT, LOGS, TR13F_CSV, BLACKROCK_PATTERNS,
)
from lib.helpers import setup_logger

logger = setup_logger("phase9", LOGS / "09_long_diff_verify.log")

RANDOM_SEED = 42

# Long-difference window
PRE_START, PRE_END = 2005, 2007
POST_START, POST_END = 2012, 2014
# Drop years
DROP_START, DROP_END = 2008, 2011


# ═══════════════════════════════════════════════════════════════════
# Step 1: Rebuild enhanced panel (same logic as Phase 8b)
# ═══════════════════════════════════════════════════════════════════

def rebuild_panel():
    logger.info("=" * 60)
    logger.info("Rebuilding enhanced panel (same logic as Phase 8b)")
    logger.info("=" * 60)

    # --- Get BR mgrnos ---
    br_mgrnos = set()
    for chunk in pd.read_csv(
        TR13F_CSV, chunksize=100_000, low_memory=False,
        usecols=["mgrno", "mgrname"],
    ):
        names = chunk["mgrname"].str.upper().fillna("")
        mask = names.str.contains("|".join(BLACKROCK_PATTERNS))
        br_mgrnos.update(chunk.loc[mask, "mgrno"].unique())
    logger.info("BlackRock mgrnos: %d", len(br_mgrnos))

    # --- Get BR firms ---
    betas = pd.read_csv(OUTPUT / "beta_aggregated.csv", low_memory=False)
    br_firms = set(betas[betas["mgrno"].isin(br_mgrnos)]["gvkey"].unique())
    del betas; gc.collect()
    logger.info("BR firms: %d", len(br_firms))

    # --- BR beta change (firm-level) ---
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
    logger.info("BR beta change: %d firms, mean=%.6f, median=%.6f, SD=%.6f",
                len(br_change),
                br_change["br_beta_change"].mean(),
                br_change["br_beta_change"].median(),
                br_change["br_beta_change"].std())
    del br_betas, br_annual, betas2; gc.collect()

    # --- Merge into panel ---
    panel = pd.read_csv(PANEL_CSV, low_memory=False)
    panel = panel.merge(
        br_change[["gvkey", "br_beta_change", "br_beta_pre", "br_beta_post"]],
        on="gvkey", how="left",
    )
    panel["br_beta_change"] = panel["br_beta_change"].fillna(0)

    # --- DiD variables ---
    panel["post"] = (panel["year"] >= 2010).astype(int)  # won't matter, we redefine below
    panel["br_change_x_post"] = panel["br_beta_change"] * panel["post"]

    panel["gvkey_cat"] = panel["gvkey"].astype(str)
    panel["year_cat"] = panel["year"].astype(str)
    panel["sic2"] = panel["sich"].fillna(0).astype(int) // 100
    panel["sic2_cat"] = panel["sic2"].astype(str)
    panel["rd_intensity"] = panel["rd_intensity"].fillna(0)

    # --- Restrict to full analysis window (2005-2014) first ---
    panel = panel[panel["year"].between(2005, 2014)].copy()

    key_vars = ["mobility_event", "br_beta_change", "post", "age", "tenure", "male",
                "log_assets", "leverage", "rd_intensity"]
    panel = panel.dropna(subset=key_vars).reset_index(drop=True)
    logger.info("Full panel: %d rows, %d firms, %d execs",
                len(panel), panel["gvkey"].nunique(), panel["execid"].nunique())

    return panel


# ═══════════════════════════════════════════════════════════════════
# Step 2: Long-difference filter + DiD
# ═══════════════════════════════════════════════════════════════════

def run_long_diff(panel):
    logger.info("=" * 60)
    logger.info("LONG-DIFFERENCE DiD")
    logger.info("Drop %d-%d, compare %d-%d vs %d-%d",
                DROP_START, DROP_END, PRE_START, PRE_END, POST_START, POST_END)
    logger.info("=" * 60)

    # Filter
    ld = panel[~panel["year"].between(DROP_START, DROP_END)].copy()
    # Redefine post for long-difference: 2012-2014 = 1, 2005-2007 = 0
    ld["post_ld"] = (ld["year"] >= POST_START).astype(int)
    ld["br_change_x_post_ld"] = ld["br_beta_change"] * ld["post_ld"]

    logger.info("Long-diff sample: %d rows, %d firms, %d execs",
                len(ld), ld["gvkey"].nunique(), ld["execid"].nunique())
    logger.info("  Pre (2005-2007): %d rows, mobility=%.4f",
                (ld["post_ld"] == 0).sum(),
                ld.loc[ld["post_ld"] == 0, "mobility_event"].mean())
    logger.info("  Post (2012-2014): %d rows, mobility=%.4f",
                (ld["post_ld"] == 1).sum(),
                ld.loc[ld["post_ld"] == 1, "mobility_event"].mean())

    exec_ctrls = "age + tenure + male"
    firm_ctrls = "log_assets + leverage + rd_intensity"
    ind_yr_fe = "C(sic2_cat) + C(year_cat)"

    results = {}

    # ── Table 7.1 Panel A: Continuous treatment ──────────────────
    specs = [
        ("OLS", f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld"),
        ("+Exec", f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + {exec_ctrls}"),
        ("+Firm", f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + {exec_ctrls} + {firm_ctrls}"),
        ("+Ind+Year FE", f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + {exec_ctrls} + {firm_ctrls} + {ind_yr_fe}"),
    ]

    logger.info("--- Table 7.1 Panel A: Continuous Treatment ---")
    for spec_name, formula in specs:
        try:
            m = smf.ols(formula, data=ld).fit(
                cov_type="cluster", cov_kwds={"groups": ld["gvkey_cat"]}
            )
            b = m.params.get("br_change_x_post_ld", np.nan)
            se = m.bse.get("br_change_x_post_ld", np.nan)
            p = m.pvalues.get("br_change_x_post_ld", np.nan)
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            logger.info("  %-18s | coef=%+9.6f  SE=%.6f  p=%.4f  %s  N=%d  R²=%.4f",
                        spec_name, b, se, p, stars, int(m.nobs), m.rsquared)
            results[f"Mobility_{spec_name}"] = {
                "coef": b, "se": se, "p": p, "stars": stars,
                "nobs": int(m.nobs), "rsquared": m.rsquared,
            }
        except Exception as e:
            logger.info("  %-18s | FAILED: %s", spec_name, str(e)[:150])

    # ── Table 7.1 Panel B: Binary treatment (comparison) ─────────
    ld["br_change_med"] = ld.loc[ld["br_beta_change"] != 0, "br_beta_change"].median()
    ld["high_br"] = (ld["br_beta_change"] > ld["br_change_med"]).astype(int)
    ld["high_br_x_post_ld"] = ld["high_br"] * ld["post_ld"]

    logger.info("--- Table 7.1 Panel B: Binary Treatment ---")
    for spec_name, formula in [
        ("OLS", f"mobility_event ~ high_br + post_ld + high_br_x_post_ld"),
        ("+Firm+FE", f"mobility_event ~ high_br + post_ld + high_br_x_post_ld + {exec_ctrls} + {firm_ctrls} + {ind_yr_fe}"),
    ]:
        try:
            m = smf.ols(formula, data=ld).fit(
                cov_type="cluster", cov_kwds={"groups": ld["gvkey_cat"]}
            )
            b = m.params.get("high_br_x_post_ld", np.nan)
            se = m.bse.get("high_br_x_post_ld", np.nan)
            p = m.pvalues.get("high_br_x_post_ld", np.nan)
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            logger.info("  %-18s | coef=%+9.6f  SE=%.6f  p=%.4f  %s  N=%d  R²=%.4f",
                        f"Binary-{spec_name}", b, se, p, stars, int(m.nobs), m.rsquared)
        except Exception as e:
            logger.info("  %-18s | FAILED: %s", f"Binary-{spec_name}", str(e)[:150])

    # ── R&D Heterogeneity (Table 5.4 Panel B) ────────────────────
    logger.info("--- Table 5.4 Panel B: R&D Heterogeneity (Long-diff) ---")
    ld["rd_high"] = (ld["rd_intensity"] > ld["rd_intensity"].median()).astype(int)

    for rd_label, rd_val in [("High-RD", 1), ("Low-RD", 0)]:
        ld_rd = ld[ld["rd_high"] == rd_val]
        try:
            m = smf.ols(
                f"mobility_event ~ br_beta_change + post_ld + br_change_x_post_ld + "
                f"{exec_ctrls} + {firm_ctrls} + {ind_yr_fe}",
                data=ld_rd,
            ).fit(cov_type="cluster", cov_kwds={"groups": ld_rd["gvkey_cat"]})
            b = m.params.get("br_change_x_post_ld", np.nan)
            se = m.bse.get("br_change_x_post_ld", np.nan)
            p = m.pvalues.get("br_change_x_post_ld", np.nan)
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            logger.info("  %-6s | coef=%+9.6f  SE=%.6f  p=%.4f  %s  N=%d  R²=%.4f  mob_pre=%.4f  mob_post=%.4f",
                        rd_label, b, se, p, stars, int(m.nobs), m.rsquared,
                        ld_rd.loc[ld_rd["post_ld"] == 0, "mobility_event"].mean(),
                        ld_rd.loc[ld_rd["post_ld"] == 1, "mobility_event"].mean())
        except Exception as e:
            logger.info("  %-6s | FAILED: %s", rd_label, str(e)[:150])

    # ── Unconditional mean ───────────────────────────────────────
    full_mobility_mean = panel["mobility_event"].mean()
    logger.info("--- Summary Stats ---")
    logger.info("Unconditional mobility mean (full 2005-2014): %.4f", full_mobility_mean)
    logger.info("BR_beta_change SD: %.6f", panel["br_beta_change"].std())
    logger.info("BR_beta_change mean (BR_change > 0): %.6f",
                panel.loc[panel["br_beta_change"] > 0, "br_beta_change"].mean())

    return results


# ═══════════════════════════════════════════════════════════════════
# Step 3: Comparison with working paper claims
# ═══════════════════════════════════════════════════════════════════

def compare_with_paper(results):
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Reproduced vs Working Paper Claims")
    logger.info("=" * 70)

    # Table 7.1 Panel A claims
    claims = {
        "Mobility_OLS": ("+0.0961", "0.0342", "0.0049", 18108),
        "Mobility_+Exec": ("+0.0984", "0.0341", "0.0039", 18108),
        "Mobility_+Firm": ("+0.0995", "0.0341", "0.0036", 18108),
        "Mobility_+Ind+Year FE": ("+0.0956", "0.0340", "0.0049", 18108),
    }

    for key, (wp_coef, wp_se, wp_p, wp_n) in claims.items():
        if key in results:
            r = results[key]
            match_coef = abs(r["coef"] - float(wp_coef)) < 0.001
            match_p = abs(r["p"] - float(wp_p)) < 0.01 if not np.isnan(r["p"]) else False
            status = "✅ MATCH" if (match_coef and r["nobs"] == wp_n) else "⚠️ DIFF"
            logger.info(
                "%s | WP: %s (%s) p=%s N=%d | REPRO: %+.4f (%s) p=%.4f N=%d | %s",
                key, wp_coef, wp_se, wp_p, wp_n,
                r["coef"], "nan" if np.isnan(r["se"]) else f"{r['se']:.4f}",
                r["p"] if not np.isnan(r["p"]) else -1,
                r["nobs"],
                status,
            )
        else:
            logger.info("%s | NOT FOUND in results", key)


def main():
    np.random.seed(RANDOM_SEED)

    panel = rebuild_panel()
    results = run_long_diff(panel)
    compare_with_paper(results)

    logger.info("\nPhase 9 done.")


if __name__ == "__main__":
    main()
