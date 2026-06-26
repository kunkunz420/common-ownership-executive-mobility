"""Central paths, parameters, and constants for the data cleaning pipeline."""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────
ROOT = Path("/home/kun/文档/沃顿数据")
DATA = ROOT
OUTPUT = ROOT / "data_clean" / "output"
LOGS = ROOT / "data_clean" / "logs"

# ── Raw data paths ────────────────────────────────────────────
COMPUSTAT_CSV = DATA / "comp_na_daily_all" / "vcnbtnnbogjbzcyg.csv"
EXECUCOMP_CSV = DATA / "comp_execucomp" / "yu0tpsek6ag9swri.csv"
TR13F_CSV = DATA / "tr_13f" / "xhsbigsc1zkjzbwb_all.csv"
CRSP_STOCK_CSV = DATA / "crsp_a_stock" / "wfn9ourtsz2anown.csv"
CRSP_NAMES_CSV = DATA / "crsp_a_stock_name" / "f5t6ddtcfuetf5ja.csv"

# ── Output paths ──────────────────────────────────────────────
LINK_TABLE_CSV = OUTPUT / "link_table.csv"
MHHI_DELTA_CSV = OUTPUT / "mhhi_delta.csv"
EXEC_CLEAN_CSV = OUTPUT / "exec_clean.csv"
PANEL_CSV = OUTPUT / "analysis_panel.csv"
PANEL_DTA = OUTPUT / "analysis_panel.dta"
BETA_CSV = OUTPUT / "beta_aggregated.csv"
LAMBDA_DIR = OUTPUT / "lambda_jk"

# ── Time parameters ───────────────────────────────────────────
MERGER_YEAR = 2009
PRE_START = 2000
POST_END = 2018

# ── BlackRock-BGI merger identifiers ──────────────────────────
BLACKROCK_PATTERNS = [
    "BLACKROCK",
    "BLACK ROCK",
    "BLK",
]
BGI_PATTERNS = [
    "BARCLAYS GLOBAL INVESTORS",
    "BARCLAYS GLOBAL",
    "BGI",
    "BARCLAYS GLOBAL INV",
]

# ── Compustat filters (applied in WRDS query) ─────────────────
# consol='C', indfmt='INDL'|'FS', datafmt='STD', curcd='USD', costat='A'|'I'
COMPUSTAT_FILTERS = dict(
    consol="C",
    datafmt="STD",
    curcd="USD",
)

# ── ExecuComp mobility: reason codes that suggest a voluntary move ──
# RETIRED, DECEASED → NOT a mobility event (retention / natural exit)
# RESIGNED, NO REASON, OTHER → potential mobility to competitor
MOBILITY_EXCLUDED_REASONS = {"RETIRED", "DECEASED", "DECEASED ", "RETIRED "}

# ── MHHI Delta computation ────────────────────────────────────
SIC_DIGITS = 2  # 2-digit SIC for peer grouping

# ── Column name mappings (raw → clean) ────────────────────────
COMPUSTAT_RENAME = {
    "gvkey": "gvkey",
    "datadate": "datadate",
    "fyear": "fyear",
    "conm": "conm",
    "tic": "ticker",
    "cusip": "cusip",
    "at": "at",
    "dltt": "dltt",
    "lt": "lt",
    "sale": "sale",
    "xrd": "xrd",
    "csho": "csho",
    "prcc_f": "prcc_f",
    "sich": "sich",
    "ajp": "ajp",
    "bspr": "bspr",
}

EXECUCOMP_RENAME = {
    "gvkey": "gvkey",
    "year": "year",
    "execid": "execid",
    "exec_fullname": "exec_fullname",
    "gender": "gender",
    "becameceo": "becameceo",
    "co_per_rol": "co_per_rol",
    "leftco": "leftco",
    "reason": "reason",
    "age": "age",
    "bonus": "bonus",
    "salary": "salary",
    "shrown_tot_pct": "shrown_tot_pct",
    "tdc1": "tdc1",
}
