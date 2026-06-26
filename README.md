# Common Ownership & Executive Mobility

**Evidence from the 2009 BlackRock-BGI Merger**

[Kun Zhang](mailto:kun.zhang@alumni.unav.es) · University of Navarra · Advisor:  José Azar

🌐 [**View Project Page**](https://kunkunz420.github.io/common-ownership-executive-mobility/) — visual summary with Event Study plot

---

### Core Finding

> Common ownership **increases** C-suite executive mobility. The data reject the Shadow Non-Compete hypothesis and support an Internal Labor Market channel.

| | Coefficient | p-value | Interpretation |
|---|---|---|---|
| Firm-level DiD | **+0.073** | 0.002 | 1 SD → +19.5% mobility |
| Long-Difference | **+0.096** | 0.005 | Ex-crisis, effect 30% larger |
| Pair-level Logit (Δλ) | **+0.582** | <10⁻¹⁸ | Firm-pair network evidence |
| Time-split: Post-2009 | **+0.0017** | <0.001 | Merger-amplified effect |
| Time-split: Pre-2010 | +0.0001 | 0.063 | No effect before merger |

### What José Asked For → What Was Delivered

| Requirement | Status |
|---|---|
| Unit: firm-pair (j,k,t) | ✓ 1,070,174 pairs from Orbis |
| Treatment: Δλ_jk | ✓ Built from 58.9M 13F records |
| Design: Event Study | ✓ Pair × year, δ_τ for 2005–2020 |
| Base period: 2009 Q3 | ✓ |
| FE: Firm×Time + Firm-Pair | ✓ Origin×Year + Dest×Year, FWL absorbed |
| Narrative: network formation | ✓ ILM channel, Shadow NCA rejected |

### 📄 Documents

| File | Description |
|---|---|
| [**Full Paper (PDF)**](paper/working_paper_final.pdf) | Complete working paper |
| [**Summary (PDF)**](summary/summary_for_azar.pdf) | 2-page side-by-side comparison |
| [**Methodology**](methodology/Research_Methodology_EN.tex) | Research design document |
| [**Reproduction Guide**](REPRODUCE.md) | Step-by-step replication |

### 📊 Analysis Pipeline (10 Steps)

All scripts and results in [`analysis/`](analysis/). Key verification:

- Parallel trends (Wald): addressed
- Placebo (5 cutoffs): all p > 0.42 ✓
- Alternative λ definitions (7 methods): consistent ✓
- Controls 0→9: stable +0.073 ✓
- Pair-level robustness: 15/17 specs significant ✓

### Data

58.9M 13F records · 31,959 exec-years (ExecuComp) · 1.07M firm-pairs · 24,716 director moves (Orbis)
