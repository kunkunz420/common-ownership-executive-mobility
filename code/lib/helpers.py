"""Shared utility functions for the data cleaning pipeline."""

import logging
import sys
from pathlib import Path

import pandas as pd


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Create logger that writes to both file and stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def clean_cusip8(cusip_series: pd.Series) -> pd.Series:
    """Extract first 8 characters of CUSIP for cross-database matching."""
    return cusip_series.astype(str).str.strip().str[:8].str.upper()


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with memory-efficient defaults."""
    defaults = dict(low_memory=False)
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def log_shape(df: pd.DataFrame, label: str, logger: logging.Logger) -> None:
    """Log row and column count of a dataframe."""
    logger.info(f"{label}: {len(df):,d} rows × {df.shape[1]} cols")


def summarize_missing(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Log columns with missing rate > 1%."""
    missing = df.isnull().mean()
    for col in missing[missing > 0.01].index:
        logger.info("  Missing: %s = %.1f%%" % (col, missing[col] * 100))
