# Common Ownership and Executive Mobility
### Evidence from the 2009 BlackRock–BGI Merger

**Kun Zhang** · University of Navarra (MEF) · Advisor: José Azar

---

## What I Updated (Per Your Feedback)

| Your Requirement | Implementation |
|---|---|---|
| Unit of analysis: firm-pair (`j,k,t`) | 1,070,174 pairs from Orbis |
| Treatment variable: `Δλ`~jk~ | Built from 58.9M 13F records, base period 2009 Q3 |
| Design: event study with `δ`~τ~ | Pair × year, 2005–2020 |
| FE: Origin×Year + Dest×Year | FWL-absorbed, Gauss-Seidel iteration |
| Narrative: network formation | ILM channel confirmed, Shadow NCA rejected |

---

## Core Results

Common ownership **increases** C-suite executive mobility.

| Specification | Coefficient | *p*-value | *N* |
|---|---|---|---|---|
| Firm-level DiD (BR`Δ` × Post) | +0.073 | 0.002 | 31,959 |
| Long-difference | +0.096 | 0.005 | 18,108 |
| Pair-level Logit (`Δλ`~jk~) | +0.582 | < 10⁻¹⁸ | 1,070,174 |
| Post-2009 time split | +0.0017 | < 0.001 | — |
| Pre-2010 time split | +0.0001 | 0.063 | — |

1 SD increase in BR`Δ` → +0.93 pp (+19.5% of mean mobility rate)

---

## Robustness

- Placebo tests (5 cutoffs): all *p* > 0.42 ✅
- Alternative λ definitions (7 methods): consistent ✅
- Control set expansion (0 → 9 controls): coefficient stable at +0.073 ✅
- Pair-level: 15/17 specifications significant ✅

---

## Code

All analysis scripts are in [`code/`](code/), organized by stage:

| Folder | What it does |
|---|---|
| `1_data_pipeline/` | WRDS cleaning: 13F → λ_jk |
| `2_firm_level/` | Firm-level DiD and long-difference |
| `3_pair_level/` | Pair panel, FWL event study, logit, RESET |
| `4_robustness/` | Parallel trends, placebo, 17 robustness checks |
| `5_output/` | Event-study coefficient plot |

See [`REPRODUCE.md`](REPRODUCE.md) for step-by-step instructions.

---

## Paper

| File | Description |
|---|---|
| [`paper/working_paper_final.pdf`](paper/working_paper_final.pdf) | Full working paper |
| [`paper/summary.pdf`](paper/summary.pdf) | 2-page summary |

---

## Data

Raw data (13F, ExecuComp, Orbis) cannot be redistributed due to licensing.  
See [`data/README.md`](data/README.md) for sources and access. A small sample is in [`data/sample/`](data/sample/).
