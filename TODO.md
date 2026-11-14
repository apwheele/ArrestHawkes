# TODO / progress log

Status file so a later session can pick up where the last one stopped.
The first unchecked item is where to resume.

## Pipeline

- [x] uv project (`arresthawkes` package under `src/`), kernel `arresthawkes` registered
- [x] `scripts/01_seattle_data.py` -- SPD car prowls + arrests, 2019-01-01 to 2026-06-30, downloaded 2026-11-13
- [x] `scripts/02_chicago_data.py` -- Chicago shooting victims + arrests, 2016-01-01 to 2026-06-30, downloaded 2026-11-14
- [x] `scripts/03_simulation.py all 0 99` -- 6 scenarios x 100 worlds, done with corrected profile intervals
- [ ] `scripts/04_models.py shootings`, `shootings_violent`, `carprowl` (car prowls done)
- [ ] paper.qmd: shooting results, abstract from the Chicago results; render pdf/docx/md
- [ ] commit, create private GitHub repo apwheele/ArrestHawkes
- [ ] later (Andrew's plan): a temporal-only version on NIBRS incident + arrest segments

## Ideas not done

- Two temporal components (within-night spree vs. return visits weeks later) instead of one kernel.
- Dampen every linked crime of an offender at the arrest (SPD arrest table has `arrest_number`).
- Network version for shootings (retaliation between groups).
- Parametric bootstrap intervals instead of profile likelihood + process variance.

## Design decisions (and why)

- Car prowls: Seattle. Shootings: Chicago. Andrew first ruled out Chicago and Dallas (overused), then
  asked to check them (2026-11-14) after the alternatives failed, with the rule that arrest rates in the
  open data must match other sources.
- Chicago check: share of shooting incidents with an arrest under the case number vs published arrest
  rates (CBS Chicago from CPD FOIA records: homicides 27% in 2022, 17% in first 10 months of 2023;
  Sun-Times Mar 2025: non-fatal 141 of ~2,300 in 2024, ~6% since 2018; homicides 25% cleared by arrest).
  Open data: fatal 14-28% by year, non-fatal 5-12%. Arrest table has a case number for ~70% of arrests.
- Main Chicago spec = any arrest under the case number; sensitivity = only violent or warrant charges
  (~15% of linked shootings have only gun/drug/other charges, possibly the victim).
- Rejected for shootings, with the reason:
  - Seattle: public file redacts the location of nearly all fatal shootings.
  - Dallas: incident status says 36-60% of gun homicides cleared by arrest, but only half of
    cleared-by-arrest incidents appear in the arrest table (the only arrest date), few homicides.
  - Baltimore: arrests with an incident number link to ~9% of shootings vs >50% reported clearance.
  - Philadelphia: arrests only as daily counts. Arrests are findable on the PA UJS court portal by
    incident number (docket sheets have Arrest Date), but the portal rate limits searches (HTTP 429)
    to about one a minute: ~3,460 searches = ~2.5-3 days. Andrew called that infeasible. Searching by
    filing date does not help: results are silently truncated (oldest days first) at ~500-1,000 rows.
    Code is in the session scratchpad, not the repo.
  - LA (arrest rpt_id is not the crime DR number), NYC (no shooting key on arrests), DC (anonymized
    CCN), Charlotte (no weapon field), Austin (no coordinates), Cincinnati, Rochester (dated arrests only
    for homicides, dates without times).
- Arrest enters as a time-varying multiplier on the crime's own kernel (kappa^A_i(t)), not as a fixed
  mark: the fixed mark has immortal time bias and fails when one arrest clears earlier crimes (sims).
- Ties: an arrest logged at the exact time of another crime counts as after it (strict `<`).
- Background: quarterly steps in time x variable-bandwidth leave-one-out KDE in space (k-th nearest
  neighbor, floor 40 m). k and the temporal kernel (exponential, power law) chosen by likelihood.
- Locations are block-level in both cities: Gaussian jitter, sd 30 m, seed 0.
- Multi-start fitting; profile likelihood intervals for kappa; crimes prevented gets a prediction
  interval (profile interval widened by the branching process variance).
- Fast-arrest comparison group in the follow-on figure = arrest within a day.
- The car prowl compensator once built a 66k x 66k matrix (35 GB) and Claude Code killed the jobs for
  low memory; chunks are now bounded by rows (test in tests/).
- CPU: runs kept to two threads (Andrew's 4-core machine).
