# Common Ownership and Executive Mobility
### Evidence from the 2009 BlackRock–BGI Merger

**Kun Zhang** · University of Navarra (MEF)  
**Advisor:** José Azar · Department of Economics

---

## Core Finding

Common ownership **increases** C-suite executive mobility across firms. The data reject the Shadow Non-Compete hypothesis and support an Internal Labor Market (ILM) channel: when institutional investors hold overlapping stakes, executive flows rise — not fall.

---

## Main Results

| Specification | Coefficient | *p*-value | Interpretation |
|---|---|---|---|
| Firm-level DiD | **+0.073** | 0.002 | 1 SD increase → +19.5% mobility |
| Long-Difference | **+0.096** | 0.005 | Financial-crisis period excluded; effect 30% larger |
| Pair-level Logit (Δλ) | **+0.582** | < 10⁻¹⁸ | Firm-pair network channel confirmed |
| Time-split: post-2009 | **+0.0017** | < 0.001 | Merger-amplified effect present |
| Time-split: pre-2010 | +0.0001 | 0.063 | No effect prior to merger |

---

## Methodological Updates

| Requirement | Status | Detail |
|---|---|---|
| Unit of analysis: firm-pair (*j*, *k*, *t*) | ✅ | 1,070,174 pairs from Orbis |
| Treatment variable: Δλ*jk* | ✅ | Constructed from 58.9M 13F records |
| Design: event study | ✅ | Pair × year, δ*τ* estimated for 2005–2020 |
| Base period: 2009 Q3 | ✅ | |
| Fixed effects: Firm×Time + Firm-Pair | ✅ | Origin×Year, Dest×Year; FWL-absorbed |
| Narrative: network formation | ✅ | ILM channel supported; Shadow NCA rejected |

---

## Documents

| File | Description |
|---|---|
| [`paper/working_paper_final.pdf`](paper/working_paper_final.pdf) | Complete working paper |
| [`summary/paper_summary.pdf`](summary/paper_summary.pdf) | 2-page side-by-side comparison |
| [`methodology/Research_Methodology_EN.tex`](methodology/Research_Methodology_EN.tex) | Full research design document |
| [`REPRODUCE.md`](REPRODUCE.md) | Step-by-step replication guide |

---

## Analysis Pipeline

All scripts and results are in [`analysis/`](analysis/). Key robustness checks:

- **Parallel trends (Wald test):** assumption addressed
- **Placebo tests (5 cutoffs):** all *p* > 0.42 ✅
- **Alternative λ definitions (7 methods):** results consistent ✅
- **Control set expansion (0 → 9 controls):** coefficient stable at +0.073 ✅
- **Pair-level robustness:** significant in 15 of 17 specifications ✅

---

## Data

| Source | Coverage |
|---|---|
| SEC 13F filings | 58.9M records — institutional ownership |
| ExecuComp | 244,857 records → 31,959 executive-years — C-suite panel |
| Compustat | 25,874 firms (231,365 firm-year rows) — firm-level controls |
| CRSP Stock / Names | 1,821,454 monthly rows · 40,518 PERMNOs — market data |
| Orbis (Bureau van Dijk) | 14,220 firms · 1,417,530 person-roles · 1,070,174 firm-pairs, 24,716 director moves |

The quasi-natural experiment exploits the **2009 BlackRock–BGI merger** as an exogenous shock to common ownership, in a difference-in-differences framework with event-study dynamics.

---

*Working paper. Please do not cite without permission.*
