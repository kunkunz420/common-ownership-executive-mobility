"""Phase 1: Build gvkey ↔ cusip ↔ permno Link Table.

CRSP Names (permno–cusip) + Compustat (gvkey–cusip) → unified mapping.
CUSIP (8-digit) is the bridge.
"""

import pandas as pd
from pathlib import Path

from lib.config import COMPUSTAT_CSV, CRSP_NAMES_CSV, LINK_TABLE_CSV, LOGS
from lib.helpers import setup_logger, clean_cusip8, read_csv_safe, log_shape

logger = setup_logger("phase1", LOGS / "01_link_table.log")


def load_compustat_keys(path: Path) -> pd.DataFrame:
    """Extract unique gvkey–cusip pairs from Compustat NA."""
    logger.info("Loading Compustat NA for gvkey-cusip mapping...")
    cols = ["gvkey", "cusip", "conm", "tic", "sich"]
    df = read_csv_safe(path, usecols=cols).drop_duplicates()
    log_shape(df, "Compustat raw", logger)
    df["cusip8"] = clean_cusip8(df["cusip"])
    # one gvkey may have multiple cusip8 over time — keep unique pairs
    mapping = df[["gvkey", "cusip8", "conm", "tic", "sich"]].drop_duplicates()
    log_shape(mapping, "Compustat gvkey–cusip8 unique", logger)
    return mapping


def load_crsp_names(path: Path) -> pd.DataFrame:
    """Extract permno–cusip mapping from CRSP Names.

    CRSP Names has one row per PERMNO per CUSIP assignment period.
    We keep the latest CUSIP per PERMNO (largest SecInfoEndDt).
    """
    logger.info("Loading CRSP Names for permno-cusip mapping...")
    cols = ["PERMNO", "PERMCO", "CUSIP", "Ticker", "IssuerNm",
            "SecInfoStartDt", "SecInfoEndDt", "HdrSICCD"]
    df = read_csv_safe(path, usecols=cols)
    log_shape(df, "CRSP Names raw", logger)

    df["cusip8"] = clean_cusip8(df["CUSIP"])

    # Parse date columns
    df["SecInfoStartDt"] = pd.to_datetime(df["SecInfoStartDt"], errors="coerce")
    df["SecInfoEndDt"] = pd.to_datetime(df["SecInfoEndDt"], errors="coerce")

    # For each permno, keep the CUSIP with the most recent SecInfoEndDt
    df = df.sort_values(["PERMNO", "SecInfoEndDt"], ascending=[True, False])
    mapping = df.drop_duplicates(subset=["PERMNO", "cusip8"], keep="first")
    # Further dedup: one PERMNO → one primary cusip8
    mapping = mapping.drop_duplicates(subset=["PERMNO"], keep="first")

    log_shape(mapping, "CRSP Names permno–cusip8 unique", logger)
    return mapping[["PERMNO", "PERMCO", "cusip8", "Ticker", "IssuerNm", "HdrSICCD"]]


def build_link_table(comp: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    """Merge Compustat and CRSP on cusip8."""
    logger.info("Merging Compustat ↔ CRSP on cusip8...")
    merged = comp.merge(crsp, on="cusip8", how="inner", indicator=True)
    log_shape(merged, "After cusip8 merge", logger)

    # Report match stats
    n_comp = comp["gvkey"].nunique()
    n_crsp = crsp["PERMNO"].nunique()
    n_matched_gvkey = merged["gvkey"].nunique()
    n_matched_permno = merged["PERMNO"].nunique()
    logger.info("Match: %d/%d gvkeys, %d/%d permnos linked",
                n_matched_gvkey, n_comp, n_matched_permno, n_crsp)

    # Check for duplicated mappings
    dup_gvkey = merged.groupby("gvkey").size()
    multi_match = (dup_gvkey > 1).sum()
    logger.info("gvkeys with >1 permno: %d (%.1f%%)",
                multi_match, multi_match / n_matched_gvkey * 100)

    # Rename for clarity
    out = merged.rename(columns={
        "conm": "comp_name",
        "tic": "comp_ticker",
        "sich": "comp_sich",
        "Ticker": "crsp_ticker",
        "IssuerNm": "crsp_name",
        "HdrSICCD": "crsp_siccd",
    })

    cols = ["gvkey", "PERMNO", "PERMCO", "cusip8",
            "comp_name", "comp_ticker", "comp_sich",
            "crsp_ticker", "crsp_name", "crsp_siccd"]
    return out[cols].drop_duplicates().reset_index(drop=True)


def main():
    comp = load_compustat_keys(COMPUSTAT_CSV)
    crsp = load_crsp_names(CRSP_NAMES_CSV)
    link = build_link_table(comp, crsp)

    LINK_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    link.to_csv(LINK_TABLE_CSV, index=False)
    logger.info("Link table saved to %s (%d rows)", LINK_TABLE_CSV, len(link))
    logger.info("Phase 1 done.")


if __name__ == "__main__":
    main()
