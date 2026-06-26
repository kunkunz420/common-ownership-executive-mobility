"""Check if BlackRock's ownership share (beta) jumped at the 2009 merger.

If BlackRock absorbed BGI's portfolio, BlackRock's combined beta in firms
where BGI also held stock should INCREASE discretely at 2009.

Uses beta_aggregated.csv (pre-computed) + BlackRock mgrnos from 13F scan.
"""

import gc
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from lib.helpers import setup_logger, log_shape
from lib.memguard import check as mem_check
import lib.config as cfg

logger = setup_logger("br_jump", cfg.LOGS / "br_beta_jump.log")

CHUNKSIZE = 100_000
BR_PATTERNS = ["BLACKROCK", "BLACK ROCK", "BLK "]


def get_br_mgrnos():
    """Get BlackRock mgrno set from 13F (streaming)."""
    logger.info("Finding BlackRock mgrnos...")
    br_mgrnos = set()
    for chunk in pd.read_csv(cfg.TR13F_CSV, chunksize=CHUNKSIZE, low_memory=False,
                             usecols=["mgrno", "mgrname"]):
        names = chunk["mgrname"].str.upper().fillna("")
        mask = names.str.contains("|".join(BR_PATTERNS))
        br_mgrnos.update(chunk.loc[mask, "mgrno"].unique())
    logger.info("BlackRock mgrnos: %s", sorted(br_mgrnos))
    return br_mgrnos


def compute_br_beta_panel(br_mgrnos):
    """Load aggregated betas, filter to BlackRock, aggregate to firm-quarter."""
    logger.info("Loading beta_aggregated.csv...")
    betas = pd.read_csv(cfg.OUTPUT / "beta_aggregated.csv")
    log_shape(betas, "All beta rows", logger)

    # Filter to BlackRock managers
    br_betas = betas[betas["mgrno"].isin(br_mgrnos)].copy()
    log_shape(br_betas, "BlackRock beta rows", logger)
    del betas; gc.collect()

    # Aggregate: sum beta across BlackRock entities per firm-quarter
    br_betas["quarter_period"] = br_betas["quarter"].apply(lambda x: pd.Period(x, freq="Q"))
    br_betas["year"] = br_betas["quarter_period"].apply(lambda q: q.year)

    br_firm_q = br_betas.groupby(["gvkey", "quarter", "year"], as_index=False)["beta"].sum()
    br_firm_q.rename(columns={"beta": "br_beta"}, inplace=True)
    log_shape(br_firm_q, "BlackRock firm-quarter beta", logger)

    # Identify firms ever held by BlackRock
    br_firms = set(br_firm_q["gvkey"].unique())
    logger.info("Firms ever held by BlackRock: %d", len(br_firms))

    # Add pre-2008 indicator
    # Identify firms held by BR in 2008 specifically
    br_2008 = set(br_firm_q[br_firm_q["year"] == 2008]["gvkey"].unique())
    logger.info("Firms held by BR in 2008: %d", len(br_2008))

    br_firm_q["br_held_2008"] = br_firm_q["gvkey"].isin(br_2008).astype(int)
    br_firm_q["post"] = (br_firm_q["year"] >= 2009).astype(int)
    br_firm_q["did_interact"] = br_firm_q["post"] * br_firm_q["br_held_2008"]

    # Annualize for DiD
    annual = br_firm_q.groupby(["gvkey", "year"], as_index=False).agg(
        br_beta=("br_beta", "mean"),
        br_held_2008=("br_held_2008", "first"),
    )
    annual["post"] = (annual["year"] >= 2009).astype(int)
    annual["did_interact"] = annual["post"] * annual["br_held_2008"]
    annual["gvkey_cat"] = annual["gvkey"].astype(str)
    annual["year_cat"] = annual["year"].astype(str)
    log_shape(annual, "Annual BlackRock beta", logger)

    return annual, br_2008


def run_did(annual, br_2008):
    """Run DiD: did BlackRock beta jump for BR-held firms post-2009?"""
    logger.info("=== BlackRock Beta DiD ===")

    # Filter to firms that appear at least once in 2007-2010 window
    window_firms = set(annual[annual["year"].between(2007, 2010)]["gvkey"].unique())
    annual_w = annual[annual["gvkey"].isin(window_firms)].copy()
    log_shape(annual_w, "Annual BR beta (2007-2010 window firms)", logger)

    # Descriptive
    for label, mask in [("BR-held 2008", annual_w["br_held_2008"] == 1),
                         ("Not BR-held 2008", annual_w["br_held_2008"] == 0)]:
        for post_val, post_label in [(0, "Pre-2009"), (1, "Post-2009")]:
            subset = annual_w[mask & (annual_w["post"] == post_val)]
            if len(subset):
                logger.info("  %s / %s: br_beta mean=%.4f, median=%.4f, N=%d, firms=%d",
                            label, post_label,
                            subset["br_beta"].mean(),
                            subset["br_beta"].median(),
                            len(subset), subset["gvkey"].nunique())

    # Raw DiD
    pre_t = annual_w[(annual_w["br_held_2008"] == 1) & (annual_w["post"] == 0)]["br_beta"]
    post_t = annual_w[(annual_w["br_held_2008"] == 1) & (annual_w["post"] == 1)]["br_beta"]
    pre_c = annual_w[(annual_w["br_held_2008"] == 0) & (annual_w["post"] == 0)]["br_beta"]
    post_c = annual_w[(annual_w["br_held_2008"] == 0) & (annual_w["post"] == 1)]["br_beta"]
    logger.info("  Treated: pre=%.4f → post=%.4f (Δ=%+.4f, N=%d/%d)",
                pre_t.mean(), post_t.mean(), post_t.mean() - pre_t.mean(),
                len(pre_t), len(post_t))
    logger.info("  Control: pre=%.4f → post=%.4f (Δ=%+.4f, N=%d/%d)",
                pre_c.mean(), post_c.mean(), post_c.mean() - pre_c.mean(),
                len(pre_c), len(post_c))
    raw_did = (post_t.mean() - pre_t.mean()) - (post_c.mean() - pre_c.mean())
    logger.info("  Raw DiD = %+.4f", raw_did)

    # DiD models
    logger.info("=== DiD specifications ===")
    for spec_name, formula, filt in [
        ("All years", "br_beta ~ br_held_2008 + post + did_interact", slice(None)),
        ("2006-2011", "br_beta ~ br_held_2008 + post + did_interact",
         annual_w["year"].between(2006, 2011)),
        ("2007-2011", "br_beta ~ br_held_2008 + post + did_interact",
         annual_w["year"].between(2007, 2011)),
        ("2008-2010 (tight)", "br_beta ~ br_held_2008 + post + did_interact",
         annual_w["year"].between(2008, 2010)),
        ("2006-2011 + yr FE", "br_beta ~ br_held_2008 + post + did_interact + C(year_cat)",
         annual_w["year"].between(2006, 2011)),
    ]:
        df = annual_w.loc[filt].copy()
        try:
            m = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["gvkey_cat"]}
            )
            did = m.params.get("did_interact", np.nan)
            did_se = m.bse.get("did_interact", np.nan)
            did_p = m.pvalues.get("did_interact", np.nan)
            logger.info("  %s: DiD=%+.4f (SE=%.4f, p=%.4f), R2=%.3f, N=%d",
                        spec_name, did, did_se, did_p, m.rsquared, int(m.nobs))
        except Exception as e:
            logger.info("  %s: FAILED (%s)", spec_name, str(e)[:120])

    # Year-by-year mean
    logger.info("=== Year-by-year BR beta ===")
    yearly = annual_w.groupby(["year", "br_held_2008"])["br_beta"].agg(["mean", "median", "count"])
    for (yr, tr), row in yearly.iterrows():
        logger.info("  %d | held_2008=%d: mean=%.4f, median=%.4f, N=%d",
                    yr, tr, row["mean"], row["median"], int(row["count"]))

    # Also: look at the DISTRIBUTION of beta changes from 2008 to 2009
    logger.info("=== Distribution of BR beta change (2008→2009) ===")
    beta_08 = annual_w[annual_w["year"] == 2008][["gvkey", "br_beta"]].rename(columns={"br_beta": "beta_08"})
    beta_09 = annual_w[annual_w["year"] == 2009][["gvkey", "br_beta"]].rename(columns={"br_beta": "beta_09"})
    delta = beta_08.merge(beta_09, on="gvkey", how="inner")
    delta["delta"] = delta["beta_09"] - delta["beta_08"]
    delta["held_2008"] = delta["gvkey"].isin(br_2008).astype(int)

    for label, mask in [("All firms", slice(None)),
                         ("BR-held 2008", delta["held_2008"] == 1),
                         ("Not BR-held 2008", delta["held_2008"] == 0)]:
        d = delta[mask]["delta"]
        logger.info("  %s: mean=%.4f, median=%.4f, std=%.4f, p25=%.4f, p75=%.4f, >0=%.1f%%, N=%d",
                    label, d.mean(), d.median(), d.std(),
                    d.quantile(0.25), d.quantile(0.75),
                    (d > 0).mean() * 100, len(d))

    # T-test
    from scipy import stats
    t_delta = delta[delta["held_2008"] == 1]["delta"]
    c_delta = delta[delta["held_2008"] == 0]["delta"]
    if len(t_delta) > 1 and len(c_delta) > 1:
        t_stat, t_p = stats.ttest_ind(t_delta, c_delta)
        logger.info("  t-test (treated vs control delta): t=%.4f, p=%.4f", t_stat, t_p)

    return delta


def main():
    mem_check("[BR beta jump start]")

    br_mgrnos = get_br_mgrnos()
    annual, br_2008 = compute_br_beta_panel(br_mgrnos)
    delta = run_did(annual, br_2008)

    # Final summary: DID the jump happen?
    logger.info("=== VERDICT ===")
    treated_delta = delta[delta["held_2008"] == 1]["delta"]
    logger.info("BlackRock beta change 2008→2009 for treated firms:")
    logger.info("  mean=%.4f, median=%.4f, >0=%.1f%%",
                treated_delta.mean(), treated_delta.median(),
                (treated_delta > 0).mean() * 100)

    mem_check("[BR beta jump end]")


if __name__ == "__main__":
    main()
