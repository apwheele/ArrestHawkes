# ArrestHawkes

Self-exciting point process models of crime let each crime raise the rate
of further crimes nearby for a while. This repository adds a term that does
the opposite: when a crime has an arrest, whatever triggering it had left is
multiplied by a factor kappa (0 = the arrest ends the chain, 1 = no effect).
The arrest enters at the time it happens, so a fast arrest removes more than
a slow one. The fitted model gives the number of follow-on crimes each
arrest prevented and the total across arrests, including the crimes those
prevented crimes would have triggered.

The paper is `paper.pdf` (also built as `paper.md`), from
`paper.qmd` and `references.bib`. It has the math, simulations where the
true number of crimes prevented is known, and two applications: shootings
in Chicago, Illinois (January 2016 through June 2026) and car prowls
(thefts from motor vehicles) in Seattle, Washington (January 2019 through
June 2026). Both cities publish arrests with a date, time, and the case
number of the crime, which most open data portals do not. The paper
checks that Chicago's linked arrests match published arrest rates for
shootings, and says why the other cities checked (Seattle shootings,
Dallas, Baltimore, Philadelphia and others) could not be used.

It builds on Wheeler, Riddell and Haberman (2021), "Breaking the chain: How
arrests reduce the probability of near repeat crimes," *Criminal Justice
Review* 46(2), and on Mohler et al. (2011), "Self-exciting point process
modeling of crime," *JASA* 106(493).

## Reproducing

This project uses [uv](https://docs.astral.sh/uv/) for the Python environment
and [Quarto](https://quarto.org/) to render the paper.

```bash
uv sync
uv run python -m ipykernel install --user --name arresthawkes --display-name "Python (arresthawkes)"

uv run python scripts/01_seattle_data.py         # SPD car prowls and arrests -> data/
uv run python scripts/02_chicago_data.py         # Chicago shooting victims and arrests -> data/
uv run python scripts/03_simulation.py all 0 99  # Monte Carlo -> results/sim/ (resumable)
uv run python scripts/04_models.py shootings     # -> results/models/
uv run python scripts/04_models.py shootings_violent
uv run python scripts/04_models.py carprowl
quarto render paper.qmd --to all
uv run pytest
```

The processed data in `data/` are committed, so steps 1 and 2 can be
skipped. The simulation takes about an hour on one core (it can be split by
seed range across two terminals), the shooting models about 10 minutes
each, and the car prowl models about 20 minutes; the model script saves
finished steps and resumes if interrupted. The paper only reads
`results/`, so rendering does not refit anything. The city portals update
daily and arrests are added to old cases, so a new download will differ
slightly.

## Using the model

You need one row per crime with a time, projected coordinates, and the time
of the arrest if there was one.

| Input | What it is |
|---|---|
| `t` | Crime times in days from the start of the window |
| `x`, `y` | Coordinates in meters (or feet; distances are in the same units) |
| `arrest_t` | Arrest time in days on the same clock, `np.inf` if no arrest. An arrest logged before the crime is treated as made at the scene |
| `T` | Length of the observation window in days |
| `X` | Optional. Crime characteristics that change how many crimes it triggers |
| `Z` | Optional. Arrest characteristics that change kappa (e.g. booked into jail) |

This example simulates a city and fits the model; it runs in a few seconds.

```python
import numpy as np
from arresthawkes import model, simulate

# A simulated city: 4 years, arrests cut remaining triggering to 30%
w = simulate.hawkes(seed=1, theta=0.35, omega=0.5, sigma=120.0, kappa=0.3)
print(f"{w.n:,} crimes, {np.isfinite(w.arrest_t).sum()} with an arrest, {w.prevented} prevented (truth)")

# Times in days, coordinates in meters, arrest time in days (np.inf if none)
ev = model.build(w.t, w.x, w.y, w.arrest_t, T=w.T, k_nn=15, kernel="exp")
fit = model.fit(ev)
for row in fit.table():
    print(f"{row['name']:10s} {row['exp']:8.3f}  ({row['low']:.3f} to {row['high']:.3f})")

# Test of no arrest effect, profile interval for kappa, and crimes prevented
no_effect = model.fit(ev, start=fit.par, fix={"log kappa": 0.0}, se=False)
print("LR test of kappa = 1: stat %.1f, df %d, p %.2g" % model.lr_test(fit, no_effect))
prof = model.profile(fit)
print("kappa 95%% profile interval: %.2f to %.2f" % (np.exp(prof["low"]), np.exp(prof["high"])))
est = model.prevented(fit.par, ev)
lo, hi = model.prevented(prof["par_high"], ev)["total"], model.prevented(prof["par_low"], ev)["total"]
print(f"crimes prevented: {est['total']:.0f} ({lo:.0f} to {hi:.0f})")
print("per arrest at 0, 1, 7 days:", model.prevented_curve(fit.par, ev, [0, 1, 7]).round(3))
```

```
2,141 crimes, 307 with an arrest, 81 prevented (truth)
log theta     0.339  (0.311 to 0.368)
log kappa     0.353  (0.211 to 0.591)
log omega     0.531  (0.478 to 0.589)
log sigma   120.353  (114.553 to 126.446)
LR test of kappa = 1: stat 24.9, df 1, p 6e-07
kappa 95% profile interval: 0.20 to 0.53
crimes prevented: 65 (47 to 82)
per arrest at 0, 1, 7 days: [0.331 0.195 0.008]
```

`fit.table()` reports each parameter exponentiated (the names keep the
`log` prefix of the scale they are estimated on): `theta` is the number of
crimes a crime triggers with no arrest, `kappa` the arrest multiplier,
`omega` the daily decay rate of the triggering, `sigma` its spatial spread.

Things to know:

- `kernel="exp"` is exponential decay in time; `kernel="lomax"` is a power
  law with a sharper peak and longer tail. Fit both and compare `fit.aic`.
- `k_nn` sets how local the background hot spot surface is (the bandwidth
  at each crime is the distance to its k-th nearest neighbor). Smaller is
  more local and leaves less for triggering to explain. Try several.
- If locations are masked to a block or grid so that many crimes share a
  point, add noise first (`seattle.jitter`). Exact ties break the spatial
  kernel.
- `model.fit(ev, static=True)` is the naive model that treats an eventual
  arrest as a fixed trait of the crime. It is there for comparison; it is
  biased whenever arrests take time (see the paper).
- `max_days` and `max_dist` in `model.build` bound the pairs of crimes
  compared (default 90 days, 1,500 meters). They should be several times
  the fitted kernel scales.

## Layout

- `paper.qmd`, `references.bib` -- paper source and bibliography.
- `src/arresthawkes/` -- the package:
  - `model.py` -- likelihood and gradient, fitting, profile intervals, crimes prevented, residuals.
  - `simulate.py` -- simulated cities: the model itself and an offender spree process, each with the counterfactual count.
  - `seattle.py` -- download and preparation of the Seattle crime and arrest tables.
  - `chicago.py` -- download and preparation of the Chicago shooting victims and arrest tables.
  - `plots.py` -- figures.
- `scripts/` -- the numbered pipeline above.
- `data/` -- Seattle car prowls and Chicago shootings with linked arrests.
- `results/` -- simulation rows and model outputs read by the paper.
- `tests/` -- unit tests (gradient checks, parameter recovery, bookkeeping).
- `filters/` -- pandoc Lua filter for the Markdown build.
- `GoogleReview.md` -- a review of the paper from Google's Deep Research, kept as received.

## License

MIT, see `LICENSE`.
