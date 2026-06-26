"""Phase 8: Identification Pivot — Directional Mobility + Continuous Treatment + Dual Mechanism.

Tackles the null reduced-form by:
  Step 1: Track destination firms for mobility events → poach vs external
  Step 2: Replace binary high_mhhi with continuous BR_beta_change (first-stage validated)
  Step 3: DiD on directional outcomes (poach_mobility, external_mobility)
  Step 4: Triple interaction with continuous R&D intensity (dual mechanism)

Key hypotheses:
  H1: β(poach) < 0 — common owners suppress intra-portfolio executive poaching
  H2: β(external) ≈ 0 — no effect on moves outside the common-owner network
  H3: The poaching suppression is concentrated in high-R&D firms (trade-secret / shadow NCA)
  H3-alt: Low-R&D firms show positive effect (internal labour market reallocation)

Author: Kun Zhang
Date:   2026-05-17
"""

import gc
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from scipy import stats

from lib.config import (
    PANEL_CSV, EXEC_CLEAN_CSV, OUTPUT, LOGS,
    MERGER_YEAR, BLACKROCK_PATTERNS, TR13F_CSV,
)
from lib.helpers import setup_logger, log_shape
from lib.memguard import check as mem_check, guard as mem_guard

logger = setup_logger("phase8", LOGS / "08_directional_mobility.log")

RANDOM_SEED = 42
CHUNKSIZE = 100_000
POST_CUTOFF = 2010  # post = 2010+ (see Phase 6 finding: merger closed Dec 2009, 13F reporting lag)
PRE_START = 2005
POST_END = 2014


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Track destination firms for directional mobility
# ═══════════════════════════════════════════════════════════════════════════════

def build_directional_mobility():
    """Identify WHERE each executive went after leaving their firm.

    Uses the execid × gvkey × year panel structure of ExecuComp.
    When an exec's gvkey changes between year t and year t+1 AND leftco is
    populated in year t, the NEW gvkey is the destination firm.

    Returns
    -------
    dest_map : pd.DataFrame
        Columns: execid, year_moved, src_gvkey, dest_gvkey, dest_observed
    """
    logger.info("=" * 60)
    logger.info("Step 1: Building directional mobility destination map")
    logger.info("=" * 60)

    exec_df = pd.read_csv(EXEC_CLEAN_CSV, low_memory=False)
    log_shape(exec_df, "Exec clean loaded", logger)

    # Sort by exec and year to track moves
    exec_df = exec_df.sort_values(["execid", "year"]).reset_index(drop=True)

    # For each exec, identify when gvkey changes
    exec_df["next_gvkey"] = exec_df.groupby("execid")["gvkey"].shift(-1)
    exec_df["next_year"] = exec_df.groupby("execid")["year"].shift(-1)
    exec_df["prev_gvkey"] = exec_df.groupby("execid")["gvkey"].shift(1)

    # --- A mobility event occurs when:
    #   1. mobility_event == 1 (leftco populated, not retired/deceased)
    #   2. next_gvkey is different from current gvkey (actual firm change) OR
    #      next_gvkey is NaN (exec disappears from ExecuComp)
    exec_df["firm_change"] = (
        (exec_df["next_gvkey"] != exec_df["gvkey"]) & exec_df["next_gvkey"].notna()
    )
    exec_df["disappears"] = exec_df["next_gvkey"].isna()

    # Build destination map: only for mobility_event == 1 rows
    mobility_rows = exec_df[exec_df["mobility_event"] == 1].copy()

    dest_records = []
    for _, row in mobility_rows.iterrows():
        execid = row["execid"]
        src_gvkey = row["gvkey"]
        year_moved = row["year"]

        # Look forward: find the next gvkey this exec appears at
        future = exec_df[
            (exec_df["execid"] == execid) & (exec_df["year"] > year_moved)
        ]
        if len(future) > 0:
            # Take the EARLIEST future observation as destination
            dest_row = future.iloc[0]
            dest_gvkey = dest_row["gvkey"]
            if dest_gvkey != src_gvkey:
                dest_records.append({
                    "execid": execid,
                    "year_moved": int(year_moved),
                    "src_gvkey": int(src_gvkey),
                    "dest_gvkey": int(dest_gvkey),
                    "dest_observed": True,
                })
            else:
                # Same gvkey — probably left and came back, or data error
                dest_records.append({
                    "execid": execid,
                    "year_moved": int(year_moved),
                    "src_gvkey": int(src_gvkey),
                    "dest_gvkey": np.nan,
                    "dest_observed": False,
                })
        else:
            # Executive disappears (left S&P 1500, or truly exited labour market)
            dest_records.append({
                "execid": execid,
                "year_moved": int(year_moved),
                "src_gvkey": int(src_gvkey),
                "dest_gvkey": np.nan,
                "dest_observed": False,
            })

    dest_map = pd.DataFrame(dest_records)
    n_obs = dest_map["dest_observed"].sum()
    n_miss = (~dest_map["dest_observed"]).sum()
    logger.info("Destination map: %d mobility events", len(dest_map))
    logger.info("  Destination observed: %d (%.1f%%)", n_obs, 100 * n_obs / len(dest_map))
    logger.info("  Destination missing:  %d (%.1f%%)", n_miss, 100 * n_miss / len(dest_map))

    return dest_map


def classify_destination_by_ownership(dest_map: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference destination firms with BlackRock ownership / MHHI data.

    For each observed destination, look up:
      - br_held_2008: whether BR held the destination firm pre-merger
      - mhhi_pre2008 at the destination
      - mhhi_delta_avg at the destination in the year of move

    Classify as:
      - poach_mobility: destination has BR presence OR high MHHI
      - external_mobility: destination has zero/low BR presence AND low MHHI
    """
    logger.info("Classifying destination firms by BR/MHHI ownership...")

    # Load MHHI data for destination firms
    mhhi = pd.read_csv(OUTPUT / "mhhi_delta.csv", low_memory=False)
    mhhi["year"] = mhhi["quarter"].str[:4].astype(int)

    # Compute pre-2008 baseline MHHI per firm
    mhhi_pre = mhhi[mhhi["year"] == 2008].groupby("gvkey")["mhhi_delta"].mean().reset_index()
    mhhi_pre.columns = ["gvkey", "dest_mhhi_pre2008"]

    # Compute annual MHHI for year-of-move lookup
    mhhi_annual = mhhi.groupby(["gvkey", "year"])["mhhi_delta"].mean().reset_index()
    mhhi_annual.columns = ["gvkey", "year", "dest_mhhi_annual"]

    # Load BR beta data to identify BR-held firms
    br_mgrnos = _get_br_mgrnos()
    br_firms = _get_br_firms(br_mgrnos)

    # Merge destination info
    dest = dest_map.copy()
    dest["dest_gvkey_int"] = dest["dest_gvkey"].fillna(-1).astype(int)

    # Merge pre-2008 MHHI at destination
    dest = dest.merge(mhhi_pre, left_on="dest_gvkey_int", right_on="gvkey", how="left")
    dest.drop(columns=["gvkey"], inplace=True, errors="ignore")

    # Merge year-of-move MHHI at destination
    dest = dest.merge(
        mhhi_annual,
        left_on=["dest_gvkey_int", "year_moved"],
        right_on=["gvkey", "year"],
        how="left",
    )
    dest.drop(columns=["gvkey", "year"], inplace=True, errors="ignore")

    # BR presence at destination
    dest["dest_br_held"] = dest["dest_gvkey_int"].isin(br_firms).astype(int)
    dest.loc[dest["dest_gvkey_int"] == -1, "dest_br_held"] = np.nan

    # --- Classification ---
    # Use available info: prefer BR beta, fall back to MHHI, mark missing as unknown
    median_mhhi_dest = dest["dest_mhhi_pre2008"].median()

    conditions = [
        # BR-held destination → poach
        dest["dest_br_held"] == 1,
        # Not BR-held but high MHHI → poach (broader common ownership)
        (dest["dest_br_held"] == 0) & (dest["dest_mhhi_pre2008"] > median_mhhi_dest),
        # Not BR-held and low MHHI → external
        (dest["dest_br_held"] == 0) & (dest["dest_mhhi_pre2008"] <= median_mhhi_dest),
    ]
    choices = ["poach", "poach", "external"]
    dest["move_type"] = np.select(conditions, choices, default="unknown")

    logger.info("Move type distribution:")
    for mt in ["poach", "external", "unknown"]:
        n = (dest["move_type"] == mt).sum()
        logger.info("  %s: %d (%.1f%%)", mt, n, 100 * n / len(dest))

    return dest


def _get_br_mgrnos() -> set:
    """Stream 13F to identify BlackRock manager numbers."""
    br_mgrnos = set()
    for chunk in pd.read_csv(
        TR13F_CSV, chunksize=CHUNKSIZE, low_memory=False,
        usecols=["mgrno", "mgrname"],
    ):
        names = chunk["mgrname"].str.upper().fillna("")
        mask = names.str.contains("|".join(BLACKROCK_PATTERNS))
        br_mgrnos.update(chunk.loc[mask, "mgrno"].unique())
    logger.info("BlackRock mgrnos: %d found", len(br_mgrnos))
    return br_mgrnos


def _get_br_firms(br_mgrnos: set) -> set:
    """Identify firms ever held by BlackRock (from beta_aggregated)."""
    betas = pd.read_csv(OUTPUT / "beta_aggregated.csv", low_memory=False)
    br_betas = betas[betas["mgrno"].isin(br_mgrnos)]
    br_firms = set(br_betas["gvkey"].unique())
    logger.info("Firms ever held by BR: %d", len(br_firms))
    del betas, br_betas
    gc.collect()
    return br_firms


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Compute continuous BR_beta_change treatment
# ═══════════════════════════════════════════════════════════════════════════════

def compute_br_beta_change() -> pd.DataFrame:
    """Compute firm-level BlackRock beta change around the merger.

    br_beta_change = mean BR beta (2010-2011) - mean BR beta (2007-2008)

    Rationale: merger closed Dec 2009, BR holdings jump appears in 2010 13F.
    Using 2007-2008 as pre (2 years) and 2010-2011 as post (2 years) to
    average out quarterly noise.
    """
    logger.info("=" * 60)
    logger.info("Step 2: Computing BR_beta_change (continuous treatment)")
    logger.info("=" * 60)

    br_mgrnos = _get_br_mgrnos()

    betas = pd.read_csv(OUTPUT / "beta_aggregated.csv", low_memory=False)
    br_betas = betas[betas["mgrno"].isin(br_mgrnos)].copy()
    del betas
    gc.collect()

    br_betas["year"] = br_betas["quarter"].str[:4].astype(int)

    # Sum beta across BR entities per firm-year
    br_annual = (
        br_betas.groupby(["gvkey", "year"], as_index=False)["beta"].sum()
    )
    br_annual.rename(columns={"beta": "br_beta"}, inplace=True)

    # Pre-merger: average BR beta in 2007-2008
    pre = br_annual[br_annual["year"].between(2007, 2008)]
    pre_avg = pre.groupby("gvkey")["br_beta"].mean().reset_index()
    pre_avg.columns = ["gvkey", "br_beta_pre"]

    # Post-merger: average BR beta in 2010-2011
    post = br_annual[br_annual["year"].between(2010, 2011)]
    post_avg = post.groupby("gvkey")["br_beta"].mean().reset_index()
    post_avg.columns = ["gvkey", "br_beta_post"]

    # Merge and compute change
    br_change = pre_avg.merge(post_avg, on="gvkey", how="outer")
    br_change["br_beta_pre"] = br_change["br_beta_pre"].fillna(0.0)
    br_change["br_beta_post"] = br_change["br_beta_post"].fillna(0.0)
    br_change["br_beta_change"] = br_change["br_beta_post"] - br_change["br_beta_pre"]

    # Also record raw levels for descriptives
    br_change["br_beta_2008"] = br_change["br_beta_pre"]  # alias

    n_firms = len(br_change)
    logger.info("BR beta change computed for %d firms", n_firms)
    logger.info("  Mean Δ:   %+.6f", br_change["br_beta_change"].mean())
    logger.info("  Median Δ: %+.6f", br_change["br_beta_change"].median())
    logger.info("  >0:       %.1f%%", (br_change["br_beta_change"] > 0).mean() * 100)
    logger.info("  p25, p75: %.6f, %.6f",
                br_change["br_beta_change"].quantile(0.25),
                br_change["br_beta_change"].quantile(0.75))

    # Log distribution
    for label, mask in [
        ("BR present in both periods",
         (br_change["br_beta_pre"] > 0) & (br_change["br_beta_post"] > 0)),
        ("BR appeared post-merger only",
         (br_change["br_beta_pre"] == 0) & (br_change["br_beta_post"] > 0)),
        ("BR disappeared post-merger",
         (br_change["br_beta_pre"] > 0) & (br_change["br_beta_post"] == 0)),
    ]:
        logger.info("  %s: %d firms", label, mask.sum())

    del br_annual, pre, post, pre_avg, post_avg, br_betas
    gc.collect()
    return br_change


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Merge directional mobility + continuous treatment into analysis panel
# ═══════════════════════════════════════════════════════════════════════════════

def build_enhanced_panel(dest_map: pd.DataFrame, br_change: pd.DataFrame) -> pd.DataFrame:
    """Merge destination classification and BR_beta_change into the analysis panel."""
    logger.info("=" * 60)
    logger.info("Step 3: Building enhanced analysis panel")
    logger.info("=" * 60)

    # Load base panel
    panel = pd.read_csv(PANEL_CSV, low_memory=False)
    log_shape(panel, "Base panel loaded", logger)

    # --- Merge BR_beta_change (firm-level, time-invariant) ---
    panel = panel.merge(
        br_change[["gvkey", "br_beta_change", "br_beta_pre", "br_beta_post"]],
        on="gvkey", how="left",
    )
    n_br = panel["br_beta_change"].notna().sum()
    logger.info("BR beta change merged: %d / %d rows (%.1f%%)",
                n_br, len(panel), 100 * n_br / len(panel))

    # Firms with no BR beta change get 0 (no BR exposure = no treatment)
    panel["br_beta_change"] = panel["br_beta_change"].fillna(0.0)
    panel["br_beta_pre"] = panel["br_beta_pre"].fillna(0.0)
    panel["br_beta_post"] = panel["br_beta_post"].fillna(0.0)

    # --- Merge directional mobility from dest_map ---
    # dest_map is at execid × year_moved × src_gvkey level
    # Merge onto panel by (execid, gvkey, year) where year == year_moved
    dest_merge = dest_map[["execid", "year_moved", "src_gvkey", "move_type"]].copy()
    dest_merge.rename(columns={"src_gvkey": "gvkey"}, inplace=True)

    panel = panel.merge(
        dest_merge,
        left_on=["execid", "gvkey", "year"],
        right_on=["execid", "gvkey", "year_moved"],
        how="left",
    )
    # Clean up
    panel.drop(columns=["year_moved"], inplace=True, errors="ignore")

    # --- Construct directional mobility dummies ---
    # For exec-year rows where mobility_event == 1:
    #   poach_mobility   = 1 if move_type == "poach"
    #   external_mobility = 1 if move_type == "external"
    #   unobs_mobility   = 1 if move_type == "unknown" (dest not observed)
    # For exec-year rows where mobility_event == 0:
    #   all three remain 0 (stayer)
    panel["poach_mobility"] = (
        (panel["mobility_event"] == 1) & (panel["move_type"] == "poach")
    ).astype(int)
    panel["external_mobility"] = (
        (panel["mobility_event"] == 1) & (panel["move_type"] == "external")
    ).astype(int)
    panel["unobs_mobility"] = (
        (panel["mobility_event"] == 1) & (panel["move_type"] == "unknown")
    ).astype(int)

    logger.info("Directional mobility constructed:")
    logger.info("  poach_mobility:    %d events", panel["poach_mobility"].sum())
    logger.info("  external_mobility: %d events", panel["external_mobility"].sum())
    logger.info("  unobs_mobility:    %d events", panel["unobs_mobility"].sum())

    # --- Post period ---
    panel["post"] = (panel["year"] >= POST_CUTOFF).astype(int)

    # --- Construct DiD interactions ---
    # Continuous treatment × Post
    panel["br_change_x_post"] = panel["br_beta_change"] * panel["post"]

    # Binary treatment (for comparison / reference)
    median_br_change = panel.loc[panel["br_beta_change"] != 0, "br_beta_change"].median()
    panel["high_br_change"] = (panel["br_beta_change"] > median_br_change).astype(int)
    panel["high_br_x_post"] = panel["high_br_change"] * panel["post"]
    logger.info("Median BR change (non-zero firms): %.6f", median_br_change)
    logger.info("High BR change firms: %d", panel["high_br_change"].sum())

    # --- Ensure categorical variables for FE ---
    panel["gvkey_cat"] = panel["gvkey"].astype(str)
    panel["year_cat"] = panel["year"].astype(str)
    panel["sic2"] = panel["sich"].fillna(0).astype(int) // 100
    panel["sic2_cat"] = panel["sic2"].astype(str)

    # --- R&D intensity ---
    panel["rd_intensity"] = panel["rd_intensity"].fillna(0.0)

    # --- Restrict to analysis window ---
    panel = panel[panel["year"].between(PRE_START, POST_END)].copy()
    log_shape(panel, "After restricting to %d-%d" % (PRE_START, POST_END), logger)

    # Drop rows with missing key variables
    key_vars = [
        "mobility_event", "br_beta_change", "post", "age", "tenure", "male",
        "log_assets", "leverage", "rd_intensity",
    ]
    panel = panel.dropna(subset=key_vars).reset_index(drop=True)
    log_shape(panel, "After dropping missing key vars", logger)

    logger.info("Enhanced panel ready:")
    logger.info("  Years: %d - %d", panel["year"].min(), panel["year"].max())
    logger.info("  Firms: %d", panel["gvkey"].nunique())
    logger.info("  Execs: %d", panel["execid"].nunique())
    logger.info("  Mobility rate: %.4f", panel["mobility_event"].mean())
    logger.info("  Poach rate:    %.4f", panel["poach_mobility"].mean())
    logger.info("  External rate: %.4f", panel["external_mobility"].mean())

    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Run the regressions
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_models(df: pd.DataFrame) -> dict:
    """Estimate all directional mobility DiD models.

    Returns dict of results keyed by model name.
    """
    logger.info("=" * 60)
    logger.info("Step 4: Running DiD models")
    logger.info("=" * 60)

    exec_ctrls = "age + tenure + male"
    firm_ctrls = "log_assets + leverage + rd_intensity"
    ind_yr_fe = "C(sic2_cat):C(year_cat)"

    results = {}

    # ── Outcome set ──────────────────────────────────────────────────
    outcomes = {
        "mobility": "mobility_event",
        "poach": "poach_mobility",
        "external": "external_mobility",
    }

    # ── Model A-C: Continuous treatment (BR_beta_change × Post) ──────
    logger.info("--- Models with CONTINUOUS treatment (br_beta_change × post) ---")
    for label, yvar in outcomes.items():
        # Baseline: treatment + post + interaction, no FE
        formula = f"{yvar} ~ br_beta_change + post + br_change_x_post"
        try:
            m = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["gvkey_cat"]}
            )
            key = f"{label}_ct_ols"
            results[key] = _extract_result(m, "br_change_x_post", label, "CT-OLS")
        except Exception as e:
            logger.info(f"  {label} CT-OLS: FAILED ({e})")

        # + Exec controls
        formula = f"{yvar} ~ br_beta_change + post + br_change_x_post + {exec_ctrls}"
        try:
            m = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["gvkey_cat"]}
            )
            key = f"{label}_ct_exec"
            results[key] = _extract_result(m, "br_change_x_post", label, "CT+Exec")
        except Exception as e:
            logger.info(f"  {label} CT+Exec: FAILED ({e})")

        # + Firm controls
        formula = f"{yvar} ~ br_beta_change + post + br_change_x_post + {exec_ctrls} + {firm_ctrls}"
        try:
            m = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["gvkey_cat"]}
            )
            key = f"{label}_ct_firm"
            results[key] = _extract_result(m, "br_change_x_post", label, "CT+Firm")
        except Exception as e:
            logger.info(f"  {label} CT+Firm: FAILED ({e})")

        # + Industry-Year FE (preferred specification)
        formula = (
            f"{yvar} ~ br_beta_change + post + br_change_x_post + "
            f"{exec_ctrls} + {firm_ctrls} + {ind_yr_fe}"
        )
        try:
            m = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["gvkey_cat"]}
            )
            key = f"{label}_ct_fe"
            results[key] = _extract_result(m, "br_change_x_post", label, "CT+FE")
        except Exception as e:
            logger.info(f"  {label} CT+FE: FAILED ({e})")

    # ── Model D: Binary treatment (for comparison) ──────────────────
    logger.info("--- Models with BINARY treatment (high_br_change × post) ---")
    for label, yvar in outcomes.items():
        formula = (
            f"{yvar} ~ high_br_change + post + high_br_x_post + "
            f"{exec_ctrls} + {firm_ctrls} + {ind_yr_fe}"
        )
        try:
            m = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["gvkey_cat"]}
            )
            key = f"{label}_bin_fe"
            results[key] = _extract_result(m, "high_br_x_post", label, "Bin+FE")
        except Exception as e:
            logger.info(f"  {label} Bin+FE: FAILED ({e})")

    # ── Model E: Triple interaction (R&D heterogeneity) ─────────────
    logger.info("--- Triple interaction: BR_beta_change × Post × RD_intensity ---")
    # Standardize R&D intensity for interpretable coefficients
    df = df.copy()
    df["rd_std"] = (df["rd_intensity"] - df["rd_intensity"].mean()) / df["rd_intensity"].std()
    df["rd_high"] = (df["rd_intensity"] > df["rd_intensity"].median()).astype(int)

    # Continuous triple
    df["post_x_rd"] = df["post"] * df["rd_std"]
    df["br_x_rd"] = df["br_beta_change"] * df["rd_std"]
    df["br_x_post"] = df["br_beta_change"] * df["post"]
    df["triple"] = df["br_beta_change"] * df["post"] * df["rd_std"]

    for label, yvar in outcomes.items():
        formula = (
            f"{yvar} ~ br_beta_change + post + rd_std + "
            f"br_x_post + post_x_rd + br_x_rd + triple + "
            f"{exec_ctrls} + {firm_ctrls} + {ind_yr_fe}"
        )
        try:
            m = smf.ols(formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["gvkey_cat"]}
            )
            key = f"{label}_triple"
            results[key] = _extract_result(m, "triple", label, "DDD-cont")
            # Also log the main DiD at mean RD
            br_post = m.params.get("br_x_post", np.nan)
            br_post_se = m.bse.get("br_x_post", np.nan)
            logger.info("  %s DDD: br_x_post (at mean RD) = %+.6f (SE=%.6f)",
                        label, br_post, br_post_se)
        except Exception as e:
            logger.info(f"  {label} DDD: FAILED ({e})")

    # ── Sub-group DiDs by R&D level ─────────────────────────────────
    logger.info("--- Sub-group DiDs by R&D intensity ---")
    for rd_label, rd_mask in [("High-RD", df["rd_high"] == 1),
                               ("Low-RD", df["rd_high"] == 0)]:
        df_rd = df[rd_mask]
        if len(df_rd) < 100:
            logger.info(f"  {rd_label}: insufficient N ({len(df_rd)})")
            continue
        for label, yvar in outcomes.items():
            formula = (
                f"{yvar} ~ br_beta_change + post + br_change_x_post + "
                f"{exec_ctrls} + {firm_ctrls} + {ind_yr_fe}"
            )
            try:
                m = smf.ols(formula, data=df_rd).fit(
                    cov_type="cluster", cov_kwds={"groups": df_rd["gvkey_cat"]}
                )
                key = f"{label}_ct_{rd_label.replace('-','')}"
                results[key] = _extract_result(m, "br_change_x_post", label, f"CT+FE ({rd_label})")
            except Exception as e:
                logger.info(f"  {label} {rd_label}: FAILED ({e})")

    return results


def _extract_result(model, coef_name: str, outcome_label: str, spec_label: str) -> dict:
    """Extract key statistics from a fitted model."""
    b = model.params.get(coef_name, np.nan)
    se = model.bse.get(coef_name, np.nan)
    p = model.pvalues.get(coef_name, np.nan)
    t = b / se if se > 1e-10 else np.nan
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

    if not np.isnan(b):
        logger.info(
            "  %-22s | %s | coef=%+8.6f  SE=%.6f  p=%.4f  %s  N=%d  R²=%.4f",
            spec_label, outcome_label, b, se, p, stars,
            int(model.nobs), model.rsquared,
        )

    return {
        "outcome": outcome_label,
        "spec": spec_label,
        "coef": b,
        "se": se,
        "pvalue": p,
        "t_stat": t,
        "stars": stars,
        "nobs": int(model.nobs),
        "rsquared": model.rsquared,
        "model": model,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    mem_check("[Phase 8 start]")
    np.random.seed(RANDOM_SEED)

    # Step 1: Build destination map
    dest_map = build_directional_mobility()

    # Classify destinations
    dest_map = classify_destination_by_ownership(dest_map)

    # Step 2: Compute continuous treatment
    br_change = compute_br_beta_change()

    # Step 3: Build enhanced panel
    panel = build_enhanced_panel(dest_map, br_change)

    # Step 4: Run all models
    results = run_all_models(panel)

    # ── Summary Comparison Table ─────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Directional Mobility DiD Comparison")
    logger.info("=" * 70)

    summary_rows = [
        ("poach", "CT+FE", "H1: Poaching suppression"),
        ("external", "CT+FE", "H2: External mobility (no effect)"),
        ("poach", "DDD-cont", "H3: Poaching × R&D intensity"),
        ("external", "DDD-cont", "External × R&D intensity"),
    ]

    for outcome, spec, hypothesis in summary_rows:
        key = f"{outcome}_{spec.lower().replace('+','').replace('-','').replace(' ','_').replace('(','').replace(')','')}"
        # Map to actual result keys
        result_keys = [k for k in results if outcome in k and spec.lower().replace('+','') in k.lower().replace('+','').replace(' ','')]
        for rk in result_keys:
            r = results[rk]
            logger.info(
                "  %-12s | %-20s | coef=%+8.6f  SE=%.6f  p=%.4f  %s  N=%d  [%s]",
                r["outcome"], r["spec"],
                r["coef"], r["se"], r["pvalue"], r["stars"], r["nobs"],
                hypothesis,
            )

    # ── Statistical test: H0: β(poach) = β(external) ────────────
    logger.info("\n--- Wald test: β(poach) = β(external) ---")
    poach_key = None
    ext_key = None
    for k in results:
        if "poach_ct_fe" in k.lower().replace(" ", "_"):
            poach_key = k
        if "external_ct_fe" in k.lower().replace(" ", "_"):
            ext_key = k

    if poach_key and ext_key:
        b_poach = results[poach_key]["coef"]
        b_ext = results[ext_key]["coef"]
        se_poach = results[poach_key]["se"]
        se_ext = results[ext_key]["se"]
        diff = b_poach - b_ext
        # Conservative: assume independence → SE_diff = sqrt(se1^2 + se2^2)
        se_diff = np.sqrt(se_poach**2 + se_ext**2)
        t_diff = diff / se_diff if se_diff > 1e-10 else np.nan
        p_diff = 2 * (1 - stats.norm.cdf(abs(t_diff))) if not np.isnan(t_diff) else np.nan
        logger.info(
            "  β(poach) - β(external) = %+.6f  SE=%.6f  t=%.3f  p=%.4f",
            diff, se_diff, t_diff, p_diff,
        )
        logger.info(
            "  → %s",
            "Directional effect CONFIRMED ✅" if p_diff < 0.05
            else "Directional effect SUGGESTIVE ⚠️" if p_diff < 0.10
            else "No directional difference detected"
        )

    # ── Save regression table ───────────────────────────────────
    _save_tables(results, panel)

    mem_check("[Phase 8 end]")
    logger.info("Phase 8 done.")


def _save_tables(results: dict, panel: pd.DataFrame):
    """Write LaTeX regression tables and summary CSVs."""
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # --- Main DiD comparison table ---
    latex_lines = []
    latex_lines.append(r"\begin{table}[ht]")
    latex_lines.append(r"\caption{Directional Mobility DiD: Poaching vs. External}")
    latex_lines.append(r"\label{tab:directional_did}")
    latex_lines.append(r"\begin{tabular}{lccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"& \textbf{Poach} & \textbf{External} & \textbf{Any Mobility} \\")
    latex_lines.append(r"\midrule")

    for spec_label, spec_key in [
        ("Baseline (OLS)", "ct_ols"),
        ("+ Executive controls", "ct_exec"),
        ("+ Firm controls", "ct_firm"),
        ("+ Industry-Year FE", "ct_fe"),
    ]:
        row = f"  {spec_label} "
        for outcome in ["poach", "external", "mobility"]:
            key = f"{outcome}_{spec_key}"
            if key in results:
                r = results[key]
                row += f"& {r['coef']:.4f}{r['stars']} ({r['se']:.4f}) "
            else:
                row += "& "
        row += r"\\"
        latex_lines.append(row)

    latex_lines.append(r"\midrule")
    # N and R² from preferred spec
    for outcome in ["poach", "external", "mobility"]:
        key = f"{outcome}_ct_fe"
        if key in results:
            r = results[key]
            latex_lines.append(
                f"  {outcome}: N={r['nobs']:,}, $R^2$={r['rsquared']:.3f} \\\\"
            )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    with open(OUTPUT / "directional_did_table.tex", "w") as f:
        f.write("\n".join(latex_lines))
    logger.info("Directional DiD table saved to %s", OUTPUT / "directional_did_table.tex")

    # --- Triple interaction table ---
    latex_lines2 = []
    latex_lines2.append(r"\begin{table}[ht]")
    latex_lines2.append(r"\caption{Dual Mechanism: R\&D Heterogeneity (Triple Interaction)}")
    latex_lines2.append(r"\label{tab:dual_mechanism}")
    latex_lines2.append(r"\begin{tabular}{lccc}")
    latex_lines2.append(r"\toprule")
    latex_lines2.append(r"& \textbf{Poach} & \textbf{External} & \textbf{Any Mobility} \\")
    latex_lines2.append(r"\midrule")

    for spec_label, spec_key in [
        ("Triple (continuous RD)", "triple"),
        ("High-RD sub-group", "ct_highrd"),
        ("Low-RD sub-group", "ct_lowrd"),
    ]:
        row = f"  {spec_label} "
        for outcome in ["poach", "external", "mobility"]:
            key = f"{outcome}_{spec_key}"
            if key in results:
                r = results[key]
                row += f"& {r['coef']:.4f}{r['stars']} ({r['se']:.4f}) "
            else:
                row += "& "
        row += r"\\"
        latex_lines2.append(row)

    latex_lines2.append(r"\bottomrule")
    latex_lines2.append(r"\end{tabular}")
    latex_lines2.append(r"\end{table}")

    with open(OUTPUT / "dual_mechanism_table.tex", "w") as f:
        f.write("\n".join(latex_lines2))
    logger.info("Dual mechanism table saved to %s", OUTPUT / "dual_mechanism_table.tex")

    # --- Summary CSV for inspection ---
    summary = []
    for key, r in results.items():
        summary.append({
            "key": key,
            "outcome": r["outcome"],
            "spec": r["spec"],
            "coef": r["coef"],
            "se": r["se"],
            "pvalue": r["pvalue"],
            "stars": r["stars"],
            "nobs": r["nobs"],
            "rsquared": r["rsquared"],
        })
    pd.DataFrame(summary).to_csv(OUTPUT / "phase8_results_summary.csv", index=False)
    logger.info("Full results summary saved to %s", OUTPUT / "phase8_results_summary.csv")


if __name__ == "__main__":
    main()
