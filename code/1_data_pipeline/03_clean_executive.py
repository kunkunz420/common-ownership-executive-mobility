"""Phase 3: Clean ExecuComp Executive Data.

Computes tenure, flags mobility events (leftco), and merges
with Compustat firm-level controls.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from lib.config import (
    EXECUCOMP_CSV, COMPUSTAT_CSV, EXEC_CLEAN_CSV, LOGS, OUTPUT,
    MOBILITY_EXCLUDED_REASONS,
)
from lib.helpers import setup_logger, read_csv_safe, log_shape

logger = setup_logger("phase3", LOGS / "03_exec_clean.log")


def load_execucomp(path: Path) -> pd.DataFrame:
    """Load and initial-clean ExecuComp Annual Compensation."""
    cols = [
        "gvkey", "year", "execid", "exec_fullname", "gender",
        "becameceo", "co_per_rol", "leftco", "reason",
        "age", "bonus", "salary", "shrown_tot_pct", "tdc1",
    ]
    df = read_csv_safe(path, usecols=cols)
    log_shape(df, "ExecuComp raw", logger)
    return df


def clean_execucomp(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived executive variables."""
    df = df.copy()

    # -- Parse dates --
    df["becameceo_dt"] = pd.to_datetime(df["becameceo"], errors="coerce")
    df["leftco_dt"] = pd.to_datetime(df["leftco"], errors="coerce")

    # -- Tenure (years since becameceo) --
    df["becameceo_year"] = df["becameceo_dt"].dt.year
    df["tenure"] = df["year"] - df["becameceo_year"]
    df["tenure"] = df["tenure"].clip(lower=0)

    # -- Gender flag --
    df["male"] = (df["gender"].str.upper() == "MALE").astype(int)

    # -- Mobility event --
    # Exec left within [year, year+2] for non-retirement, non-death reasons
    has_left = df["leftco_dt"].notna()
    reason_upper = df["reason"].fillna("").str.upper().str.strip()
    valid_exit = ~reason_upper.isin([r.upper() for r in MOBILITY_EXCLUDED_REASONS])
    df["mobility_event"] = (has_left & valid_exit).astype(int)

    # -- Log mobility stats --
    n_mobile = df["mobility_event"].sum()
    n_left = has_left.sum()
    logger.info("mobility_event=1: %d (%.1f%% of all rows)", n_mobile, 100 * n_mobile / len(df))
    logger.info("leftco non-null: %d rows", n_left)
    logger.info("reason distribution (non-null leftco):")
    for reason, cnt in df.loc[has_left, "reason"].value_counts().head(10).items():
        logger.info("  %s: %d", reason, cnt)

    return df


def merge_firm_controls(exec_df: pd.DataFrame, comp_path: Path) -> pd.DataFrame:
    """Merge with Compustat NA for firm-level controls."""
    logger.info("Loading Compustat for firm controls...")
    cols = ["gvkey", "fyear", "at", "dltt", "lt", "sale", "xrd", "csho", "prcc_f", "sich"]
    comp = read_csv_safe(comp_path, usecols=cols)
    comp = comp.drop_duplicates(subset=["gvkey", "fyear"])
    log_shape(comp, "Compustat firm controls", logger)

    # Compute derived ratios
    comp["leverage"] = comp["dltt"] / comp["at"].replace(0, np.nan)
    comp["rd_intensity"] = comp["xrd"].fillna(0) / comp["sale"].replace(0, np.nan)
    comp["log_assets"] = np.log(comp["at"].clip(lower=1))

    merged = exec_df.merge(
        comp, left_on=["gvkey", "year"],
        right_on=["gvkey", "fyear"], how="left"
    )
    log_shape(merged, "After Compustat firm control merge", logger)

    # Report merge quality
    n_miss = merged["at"].isna().sum()
    logger.info("rows missing Compustat: %d (%.1f%%)", n_miss, 100 * n_miss / len(merged))

    return merged


def main():
    exec_df = load_execucomp(EXECUCOMP_CSV)
    exec_df = clean_execucomp(exec_df)

    # Merge firm controls
    panel = merge_firm_controls(exec_df, COMPUSTAT_CSV)

    # Select and order output columns
    out_cols = [
        "gvkey", "year", "execid", "co_per_rol", "exec_fullname",
        "gender", "male", "age", "tenure",
        "becameceo", "leftco", "reason", "mobility_event",
        "salary", "bonus", "tdc1", "shrown_tot_pct",
        "at", "leverage", "rd_intensity", "log_assets",
        "sich", "sale", "csho", "prcc_f",
    ]
    available = [c for c in out_cols if c in panel.columns]
    panel = panel[available]

    EXEC_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(EXEC_CLEAN_CSV, index=False)
    logger.info("ExecuComp clean panel saved to %s (%d rows)", EXEC_CLEAN_CSV, len(panel))
    logger.info("Phase 3 done.")


if __name__ == "__main__":
    main()
