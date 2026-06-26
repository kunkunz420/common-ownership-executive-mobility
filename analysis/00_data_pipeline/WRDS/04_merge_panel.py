"""Phase 4: Merge executive panel with CRSP market data and MHHI Delta.

Produces analysis_panel.csv — the main dataset for DiD regression.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from lib.config import (
    EXEC_CLEAN_CSV, LINK_TABLE_CSV, MHHI_DELTA_CSV,
    CRSP_STOCK_CSV, CRSP_NAMES_CSV, PANEL_CSV, LOGS, OUTPUT,
    MERGER_YEAR, PRE_START, POST_END,
)
from lib.helpers import setup_logger, read_csv_safe, log_shape

logger = setup_logger("phase4", LOGS / "04_merge_panel.log")


def load_exec_clean() -> pd.DataFrame:
    df = read_csv_safe(EXEC_CLEAN_CSV)
    log_shape(df, "Exec clean", logger)
    return df


def load_link_table() -> pd.DataFrame:
    df = read_csv_safe(LINK_TABLE_CSV)
    # Keep one PERMNO per gvkey (first match)
    df = df.drop_duplicates(subset="gvkey")[["gvkey", "PERMNO", "cusip8"]]
    log_shape(df, "Link table (deduped)", logger)
    return df


def build_crsp_annual(crsp_path: Path, names_path: Path) -> pd.DataFrame:
    """Load CRSP monthly, compute annual market data per PERMNO-year."""
    logger.info("Loading CRSP monthly stock...")
    crsp_cols = ["PERMNO", "MthCalDt", "MthPrc", "MthRet", "MthVol", "ShrOut", "vwretd"]
    crsp = read_csv_safe(crsp_path, usecols=crsp_cols)
    crsp = crsp.drop_duplicates(subset=["PERMNO", "MthCalDt"])
    log_shape(crsp, "CRSP monthly", logger)

    crsp["MthCalDt"] = pd.to_datetime(crsp["MthCalDt"], errors="coerce")
    crsp["year"] = crsp["MthCalDt"].dt.year
    crsp["month"] = crsp["MthCalDt"].dt.month

    # Market cap in millions
    crsp["mktcap"] = crsp["MthPrc"].abs() * crsp["ShrOut"] / 1000

    # Annual aggregates
    annual = crsp.groupby(["PERMNO", "year"]).agg(
        mktcap_dec=("mktcap", lambda x: x.iloc[-1] if len(x) else np.nan),
        mktcap_avg=("mktcap", "mean"),
        ann_ret=("MthRet", lambda x: np.prod(1 + x / 100) - 1 if x.notna().any() else np.nan),
        ann_vol=("MthRet", lambda x: x.std() if len(x) >= 6 else np.nan),
        n_months=("month", "count"),
    ).reset_index()
    log_shape(annual, "CRSP annual", logger)

    # Merge CRSP names for SIC
    logger.info("Loading CRSP names...")
    names = read_csv_safe(names_path, usecols=["PERMNO", "HdrSICCD"])
    names = names.drop_duplicates(subset="PERMNO")
    annual = annual.merge(names, on="PERMNO", how="left")
    annual["sic_crsp"] = annual["HdrSICCD"].astype(float)

    logger.info("CRSP annual + names: %d rows", len(annual))
    return annual


def build_mhhi_annual(mhhi_path: Path) -> pd.DataFrame:
    """Convert quarterly MHHI Delta to annual per gvkey.

    Also computes pre-merger (2008) baseline MHHI per firm as a
    time-invariant treatment intensity measure.
    """
    mhhi = read_csv_safe(mhhi_path)
    mhhi["year"] = mhhi["quarter"].str[:4].astype(int)
    mhhi["q"] = mhhi["quarter"].str[5:6]

    annual = mhhi.groupby(["gvkey", "year"]).agg(
        mhhi_delta_avg=("mhhi_delta", "mean"),
        mhhi_delta_q4=("mhhi_delta", lambda x: x.iloc[-1] if len(x) else np.nan),
        mhhi_delta_max=("mhhi_delta", "max"),
        n_quarters=("q", "count"),
    ).reset_index()

    # Pre-merger baseline: average MHHI Delta in 2008 per firm
    pre_mhhi = (
        mhhi[mhhi["year"] == 2008]
        .groupby("gvkey", as_index=False)["mhhi_delta"]
        .mean()
        .rename(columns={"mhhi_delta": "mhhi_pre2008"})
    )
    annual = annual.merge(pre_mhhi, on="gvkey", how="left")
    logger.info("firms with pre-2008 MHHI: %d", pre_mhhi["gvkey"].nunique())

    log_shape(annual, "MHHI Delta annual", logger)
    return annual


def main():
    exec_df = load_exec_clean()
    link = load_link_table()
    mhhi = build_mhhi_annual(MHHI_DELTA_CSV)
    crsp = build_crsp_annual(CRSP_STOCK_CSV, CRSP_NAMES_CSV)

    # Step 1: Merge link table (gvkey → PERMNO)
    panel = exec_df.merge(link, on="gvkey", how="left")
    n_miss_link = panel["PERMNO"].isna().sum()
    logger.info("After link table merge: %d rows, %d missing PERMNO (%.1f%%)",
                len(panel), n_miss_link, 100 * n_miss_link / len(panel))

    # Step 2: Merge CRSP annual market data
    panel = panel.merge(crsp, on=["PERMNO", "year"], how="left")
    n_miss_crsp = panel["mktcap_avg"].isna().sum()
    logger.info("After CRSP merge: %d rows, %d missing mktcap (%.1f%%)",
                len(panel), n_miss_crsp, 100 * n_miss_crsp / len(panel))

    # Step 3: Merge MHHI Delta annual
    panel = panel.merge(mhhi, on=["gvkey", "year"], how="left")
    n_miss_mhhi = panel["mhhi_delta_avg"].isna().sum()
    logger.info("After MHHI Delta merge: %d rows, %d missing mhhi (%.1f%%)",
                len(panel), n_miss_mhhi, 100 * n_miss_mhhi / len(panel))

    # Step 4: Create DiD variables
    panel["post"] = (panel["year"] >= MERGER_YEAR).astype(int)
    panel["event_year"] = panel["year"] - MERGER_YEAR

    # Treatment intensity: above-median pre-2008 MHHI Delta (time-invariant)
    panel["treat_intensity"] = panel["mhhi_pre2008"].fillna(0)
    median_pre = panel["mhhi_pre2008"].median()
    panel["high_mhhi"] = (panel["mhhi_pre2008"] > median_pre).astype(int)
    logger.info("MHHI pre2008 median = %.6f; high_mhhi=1: %d rows",
                median_pre, panel["high_mhhi"].sum())

    # Step 5: Select and order columns
    out_cols = [
        # Identifiers
        "gvkey", "PERMNO", "cusip8", "year", "execid",
        "exec_fullname", "co_per_rol",
        # Executive vars
        "gender", "male", "age", "tenure",
        "becameceo", "leftco", "reason", "mobility_event",
        "salary", "bonus", "tdc1", "shrown_tot_pct",
        # Firm controls (Compustat)
        "at", "leverage", "rd_intensity", "log_assets",
        "sich", "sale", "csho", "prcc_f",
        # CRSP market data
        "mktcap_dec", "mktcap_avg", "ann_ret", "ann_vol", "n_months",
        "sic_crsp",
        # MHHI Delta
        "mhhi_delta_avg", "mhhi_delta_q4", "mhhi_delta_max",
        "mhhi_pre2008", "n_quarters",
        # DiD variables
        "post", "event_year", "high_mhhi", "treat_intensity",
    ]
    available = [c for c in out_cols if c in panel.columns]
    panel = panel[available]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_CSV, index=False)
    logger.info("Analysis panel saved to %s (%d rows x %d cols)",
                PANEL_CSV, len(panel), panel.shape[1])
    logger.info("Phase 4 done.")


if __name__ == "__main__":
    main()
