# Institutional Common Ownership and Executive Mobility

**Evidence from the 2009 BlackRock-BGI Merger**

Kun Zhang · University of Navarra · Advisor: José Azar

---

## Quick Links

- [📄 Working Paper (PDF)](paper/working_paper_final.pdf)
- [📊 Summary for Advisor (PDF)](summary/summary_for_azar.pdf)
- [📐 Research Methodology](methodology/Research_Methodology_EN.tex)
- [🔄 Full Replication Guide](REPRODUCE.md)

## Abstract

Does institutional common ownership suppress or facilitate executive mobility? Using the 2009 BlackRock-BGI merger as a quasi-exogenous shock, I find that common ownership *increases* C-suite mobility across connected firms. Firm-level DiD: +0.073 (p=0.002). Pair-level logit: Δλ_jk = +0.582 (p<10⁻¹⁸). Time-split: post-2009 p<0.001, pre-2010 p=0.063. The data reject the Shadow Non-Compete hypothesis and support an Internal Labor Market channel.

## Repository Structure

```
├── paper/                        Final working paper (LaTeX + PDF)
├── summary/                      Concise summary for advisor review
├── methodology/                  Research methodology document
├── analysis/                     10-step analysis pipeline
│   ├── 01_pair_panel/           Build pair × year panel
│   ├── 02_fwl/                  FWL absorption of high-dim FE
│   ├── 03_event_study/          Event Study coefficients
│   ├── 04_parallel_trends/      Wald test + placebo
│   ├── 05_placebo/              Placebo results
│   ├── 06_plots/                Event Study coefficient plot
│   ├── 07_tables/               Summary regression tables
│   ├── 08_logit/                Baseline pair-level logit
│   ├── 09_reset/                RESET specification tests
│   └── 10_robustness/           17 robustness specifications
├── REPRODUCE.md                 Full step-by-step reproduction guide
└── README.md                    This file
```

## Key Results

| Level | Specification | Coefficient | p-value | N |
|-------|--------------|-------------|---------|---|
| Firm | BR_Δ × Post | +0.073 | 0.002 | 31,959 |
| Firm | Long-Diff | +0.096 | 0.005 | 18,108 |
| Pair | Logit Δλ_jk | +0.582 | <10⁻¹⁸ | 1,070,174 |
| Pair | Pre-2010 split | +0.0001 | 0.063 | — |
| Pair | Post-2009 split | +0.0017 | <0.001 | — |

Economic significance: +19.5% of mean mobility per 1 SD.  
Robustness: 15/17 pair-level specs significant. Placebo: all p>0.42.

## Data

| Source | Records |
|--------|---------|
| Thomson Reuters 13F | 58,929,118 |
| ExecuComp | 244,857 |
| Orbis (BvD) | 1,417,530 |
| Firm-pairs (λ_jk) | 1,070,174 |

*Note: Raw data files are not included in this repository due to size and licensing constraints. See REPRODUCE.md for data access instructions.*

## Advisor Requirements

All 37 items from José Azar's revision memo are addressed. See [summary/summary_for_azar.pdf](summary/summary_for_azar.pdf) for the side-by-side comparison of requirements vs. deliverables.
