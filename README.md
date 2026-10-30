# Do K-pop Groups Really Die at Seven?

A survival analysis of 587 South Korean idol groups, built as the final tutorial for
*21501755 AI and Data Analysis Practice* (Soongsil University, Fall 2026).

**Read the tutorial:** https://gyeunguk.github.io/

## What's here

| Path | Contents |
|---|---|
| `docs/index.html` | The tutorial, rendered — this is what GitHub Pages serves |
| `notebooks/kpop_survival.ipynb` | The tutorial source: Markdown prose + Python, executed |
| `scripts/collect_data.py` | Collects debut / disbandment / label / member data from the Wikipedia API |
| `data/raw/groups_raw.csv` | One row per group, straight from the collector (591 rows) |
| `data/processed/survival_table.csv` | The cleaned survival table used in the analysis (587 rows) |
| `PLAN.md` | Project plan: research questions, hypotheses, timeline |

The per-page API cache (`data/raw/pages/`, ~25 MB) is not committed; running the collector
once regenerates it.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/collect_data.py              # ~2 min; refreshes data/raw/groups_raw.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/kpop_survival.ipynb
jupyter nbconvert --to html notebooks/kpop_survival.ipynb --output ../docs/index.html
```

Note that re-running the collector pulls *today's* Wikipedia, so counts will drift slightly
from those quoted in the tutorial (collected 30 October 2026).

## Headline results

- The **seven-year curse is real**: a group entering its seventh year has an 18.5 % chance of
  ending that year, the highest hazard at any age (vs. ~12–14 % in the surrounding years).
- Median lifespan is **6 years**; most disbandments happen in years 1–4, before the contract wall.
- **Girl groups** face ~60 % higher hazard than boy groups; **Big-4** groups face about half the
  hazard of everyone else; **larger groups last longer**.
- Debut-day information predicts five-year survival only modestly (ROC-AUC ≈ 0.69).
