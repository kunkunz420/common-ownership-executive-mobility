"""Memory safety guard — call before any heavy data processing.

Usage:
    from memguard import guard, check
    guard(80)   # abort if RAM usage exceeds 80%
    check()     # print current memory status

Or as a standalone check in bash:
    python3 -c "from memguard import guard; guard(75)"
"""

import os
import sys
import psutil

GB = 1024**3
WARN_PCT = 70   # warn threshold
KILL_PCT = 85   # hard abort threshold


def _mem():
    """Return (used_gb, total_gb, pct, avail_gb)."""
    v = psutil.virtual_memory()
    return v.used / GB, v.total / GB, v.percent, v.available / GB


def check(prefix="[memguard]"):
    """Print current memory status."""
    used, total, pct, avail = _mem()
    swap = psutil.swap_memory()
    print(f"{prefix} RAM: {used:.1f}/{total:.1f} GB ({pct:.0f}%) | avail: {avail:.1f} GB | swap: {swap.used/GB:.1f}/{swap.total/GB:.1f} GB")
    if pct > WARN_PCT:
        print(f"{prefix} ⚠️  WARNING: Memory usage above {WARN_PCT}%!", file=sys.stderr)
    return pct


def guard(max_pct=80, hard_kill=True):
    """Abort if memory usage exceeds max_pct."""
    used, total, pct, avail = _mem()
    if pct > max_pct:
        msg = f"MEMGUARD: {pct:.0f}% RAM used (limit={max_pct}%, avail={avail:.1f} GB). Aborting."
        print(msg, file=sys.stderr)
        if hard_kill:
            os._exit(1)
        else:
            raise MemoryError(msg)
    return pct


def needs_gb(required_gb):
    """Check if at least required_gb is available. Raise if not."""
    _, _, _, avail = _mem()
    if avail < required_gb:
        msg = f"MEMGUARD: need {required_gb:.1f} GB free, only {avail:.1f} GB available."
        print(msg, file=sys.stderr)
        raise MemoryError(msg)
    return True


# ── Auto-guard on import ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-pct", type=int, default=80, help="Abort if RAM usage above this %")
    p.add_argument("--need-gb", type=float, help="Abort if free RAM below this GB")
    args = p.parse_args()

    check()
    if args.need_gb:
        needs_gb(args.need_gb)
    guard(args.max_pct)
    print("[memguard] OK — within limits")
