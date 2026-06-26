"""Phase 2 (v3): MHHI Delta — streaming, memory-safe version.

Key fix: NEVER loads all 13F data into memory. Instead:
  Phase A — stream 13F in 50K chunks, aggregate beta per firm-inst-quarter
            directly, flush to disk frequently. Peak: ~400 MB.
  Phase B — read aggregated betas (much smaller), compute MHHI.

Uses ALL 13F institutional holdings, not just BlackRock/BGI.
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.sparse import csr_matrix

import lib.config as cfg
from lib.helpers import setup_logger, clean_cusip8, log_shape, read_csv_safe
from lib.memguard import check as mem_check, needs_gb

logger = setup_logger("phase2", cfg.LOGS / "02_mhhi_delta.log")

CHUNKSIZE = 50_000
BETA_AGG_CSV = cfg.OUTPUT / "beta_aggregated.csv"
BETA_MIN_SUM = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Stream 13F, aggregate beta per firm-inst-quarter
# ═══════════════════════════════════════════════════════════════════════════

def load_lookups():
    """Load small lookup tables that fit easily in memory."""
    # Link table: cusip8 → (gvkey, PERMNO)
    logger.info("Loading link table...")
    link = read_csv_safe(cfg.LINK_TABLE_CSV, usecols=["gvkey", "PERMNO", "cusip8"])
    link = link.drop_duplicates(subset=["cusip8", "gvkey"])
    cusip_to_gvkey = dict(zip(link["cusip8"], link["gvkey"]))
    cusip_to_permno = dict(zip(link["cusip8"], link["PERMNO"]))
    logger.info("  %d unique cusip8 codes", len(cusip_to_gvkey))
    del link; gc.collect()

    # Market cap: (PERMNO, quarter) → mkt_cap
    logger.info("Loading CRSP market cap...")
    mkt = read_csv_safe(cfg.CRSP_STOCK_CSV, usecols=["PERMNO", "MthCalDt", "MthPrc", "ShrOut"])
    mkt["MthCalDt"] = pd.to_datetime(mkt["MthCalDt"])
    mkt["mkt_cap"] = mkt["MthPrc"].abs() * mkt["ShrOut"] * 1_000
    mkt["quarter"] = mkt["MthCalDt"].dt.to_period("Q")
    mkt = mkt.sort_values(["PERMNO", "quarter", "MthCalDt"])
    mkt = mkt.drop_duplicates(subset=["PERMNO", "quarter"], keep="last")
    mcap_lookup = {}
    for row in mkt.itertuples(index=False):
        mcap_lookup[(row.PERMNO, str(row.quarter))] = row.mkt_cap
    logger.info("  %d PERMNO-quarter entries", len(mcap_lookup))
    del mkt; gc.collect()

    # SIC: gvkey → sich2
    logger.info("Loading Compustat SIC...")
    comp = read_csv_safe(cfg.COMPUSTAT_CSV, usecols=["gvkey", "fyear", "sich"])
    comp["sich"] = pd.to_numeric(comp["sich"], errors="coerce")
    comp = comp.dropna(subset=["sich"])
    comp["sich2"] = comp["sich"].astype(int).astype(str).str[:cfg.SIC_DIGITS]
    comp = comp.sort_values("fyear").drop_duplicates(subset="gvkey", keep="last")
    sic_lookup = dict(zip(comp["gvkey"], comp["sich2"]))
    logger.info("  %d gvkeys with SIC", len(sic_lookup))
    del comp; gc.collect()

    return cusip_to_gvkey, cusip_to_permno, mcap_lookup, sic_lookup


def stream_aggregate(tr13f_path, c2gvkey, c2permno, mcap, sic):
    """Stream 13F, filter + aggregate beta per key. Flush to disk.

    Returns nothing — writes BETA_AGG_CSV to disk incrementally.
    """
    logger.info("Streaming 13F for beta aggregation...")
    BETA_AGG_CSV.unlink(missing_ok=True)

    # Accumulate: key = (gvkey, mgrno, quarter_str, sich2) → beta_sum
    accum = defaultdict(float)
    total_in = 0
    total_kept = 0
    first_flush = True

    for i, chunk in enumerate(pd.read_csv(tr13f_path, chunksize=CHUNKSIZE, low_memory=False)):
        total_in += len(chunk)

        # Clean CUSIP and filter
        chunk["cusip8"] = clean_cusip8(chunk["cusip"])
        mask = chunk["cusip8"].isin(c2gvkey)
        chunk = chunk[mask].copy()
        if not len(chunk):
            continue

        # Map to gvkey and PERMNO
        chunk["gvkey"] = chunk["cusip8"].map(c2gvkey)
        chunk["PERMNO"] = chunk["cusip8"].map(c2permno)

        # Parse numeric fields
        chunk["shares"] = pd.to_numeric(chunk["shares"], errors="coerce")
        chunk["prc"] = pd.to_numeric(chunk["prc"], errors="coerce")
        chunk = chunk.dropna(subset=["shares", "prc"])
        chunk = chunk[chunk["shares"] > 0]

        # Quarter
        chunk["fdate"] = pd.to_datetime(chunk["fdate"])
        chunk["quarter"] = chunk["fdate"].dt.to_period("Q").astype(str)

        # Merge market cap via (PERMNO, quarter) lookup
        keys = list(zip(chunk["PERMNO"], chunk["quarter"]))
        chunk["mkt_cap"] = [mcap.get(k) for k in keys]
        chunk = chunk.dropna(subset=["mkt_cap"])

        # Holding value and beta
        chunk["hold_value"] = chunk["shares"].abs() * chunk["prc"].abs()
        chunk["beta"] = (chunk["hold_value"] / chunk["mkt_cap"]).clip(upper=1.0)

        # SIC
        chunk["sich2"] = chunk["gvkey"].map(sic)
        chunk = chunk.dropna(subset=["sich2"])

        # Aggregate into accum dict
        for row in chunk.itertuples(index=False):
            key = (row.gvkey, row.mgrno, row.quarter, row.sich2)
            accum[key] += row.beta

        total_kept += len(chunk)
        del chunk

        # Flush accum to disk every 50 chunks
        if (i + 1) % 50 == 0:
            _flush_accum(accum, first_flush)
            first_flush = False
            accum.clear()
            gc.collect()
            logger.info(f"  Chunk {i+1}: {total_in:,} read, {total_kept:,} kept, accum flushed")

    # Final flush
    _flush_accum(accum, first_flush)
    accum.clear()
    gc.collect()

    logger.info(f"Beta aggregation done: {total_in:,} → {total_kept:,} rows")
    return BETA_AGG_CSV


def _flush_accum(accum, write_header):
    """Write accumulated beta dict to disk, then clear."""
    if not accum:
        return
    rows = [(k[0], k[1], k[2], k[3], v) for k, v in accum.items()]
    df = pd.DataFrame(rows, columns=["gvkey", "mgrno", "quarter", "sich2", "beta"])
    df.to_csv(BETA_AGG_CSV, mode="w" if write_header else "a",
              header=write_header, index=False)
    logger.info(f"  Flushed {len(df):,} aggregated rows to disk")
    del df


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Compute MHHI Delta from aggregated betas
# ═══════════════════════════════════════════════════════════════════════════

def compute_mhhi_delta():
    """Read aggregated betas from disk, compute MHHI Delta per quarter."""
    logger.info("Phase B: Computing MHHI Delta from aggregated betas...")

    # Read all aggregated betas — this should be much smaller than raw 13F
    betas = pd.read_csv(BETA_AGG_CSV)
    log_shape(betas, "Aggregated betas", logger)
    logger.info("  Memory: %.1f MB", betas.memory_usage(deep=True).sum() / 1e6)

    betas["quarter"] = betas["quarter"].apply(lambda x: pd.Period(x, freq="Q"))

    quarters = sorted(betas["quarter"].unique())
    logger.info("  %d quarters to process", len(quarters))

    results = []
    for qi, q in enumerate(quarters):
        hq = betas[betas["quarter"] == q]
        firms = hq["gvkey"].unique()
        insts = hq["mgrno"].unique()

        firm_idx = {f: i for i, f in enumerate(firms)}
        inst_idx = {m: i for i, m in enumerate(insts)}

        row = hq["gvkey"].map(firm_idx).values
        col = hq["mgrno"].map(inst_idx).values
        data = hq["beta"].values

        B_sparse = csr_matrix((data, (row, col)), shape=(len(firms), len(insts)))

        # SIC2 per firm
        firm_sic = hq[["gvkey", "sich2"]].drop_duplicates().set_index("gvkey")["sich2"]
        firm_list = list(firms)

        sic_groups = defaultdict(list)
        for fi, f in enumerate(firm_list):
            s = firm_sic.get(f)
            if s is not None:
                sic_groups[s].append(fi)

        mhhi_per_firm = np.zeros(len(firms))

        for sic, group_fi in sic_groups.items():
            if len(group_fi) < 2:
                continue

            Bg = B_sparse[group_fi, :]
            row_sums = Bg.sum(axis=1).A1

            cross = Bg @ Bg.T
            if hasattr(cross, "toarray"):
                cross_dense = cross.toarray()
            else:
                cross_dense = np.asarray(cross)

            cross_row_sums = cross_dense.sum(axis=1)
            self_dots = np.diag(cross_dense)
            peer_sums = cross_row_sums - self_dots

            valid = row_sums > BETA_MIN_SUM
            mhhi_per_firm[np.array(group_fi)[valid]] = (
                peer_sums[valid] / row_sums[valid]
            )

        q_results = pd.DataFrame({
            "gvkey": firm_list,
            "quarter": q,
            "mhhi_delta": mhhi_per_firm,
        })
        results.append(q_results)

        if (qi + 1) % 10 == 0:
            logger.info("  Processed %d/%d quarters...", qi+1, len(quarters))

    mhhi = pd.concat(results, ignore_index=True)
    log_shape(mhhi, "MHHI Delta output", logger)
    return mhhi


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    mem_check("[Phase 2 start]")
    needs_gb(3.0)

    # Phase A: Stream + aggregate
    c2gvkey, c2permno, mcap, sic = load_lookups()
    mem_check("[After lookups]")

    stream_aggregate(cfg.TR13F_CSV, c2gvkey, c2permno, mcap, sic)
    del c2gvkey, c2permno, mcap, sic
    gc.collect()
    mem_check("[After Phase A]")

    # Phase B: MHHI from aggregated betas
    mhhi = compute_mhhi_delta()
    mem_check("[After Phase B]")

    # Summary
    logger.info("MHHI Delta summary:")
    logger.info("  mean = %.6f", mhhi["mhhi_delta"].mean())
    logger.info("  median = %.6f", mhhi["mhhi_delta"].median())
    logger.info("  p95 = %.6f", mhhi["mhhi_delta"].quantile(0.95))
    logger.info("  > 0 = %.1f%%", (mhhi["mhhi_delta"] > 1e-10).mean() * 100)
    logger.info("  Quarters: %d", mhhi["quarter"].nunique())
    logger.info("  Firms: %d", mhhi["gvkey"].nunique())

    mhhi.to_csv(cfg.MHHI_DELTA_CSV, index=False)
    logger.info("MHHI Delta saved to %s", cfg.MHHI_DELTA_CSV)
    logger.info("Phase 2 (v3 streaming) done.")


if __name__ == "__main__":
    main()
