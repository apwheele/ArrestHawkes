"""Fit the arrest-dampened self-exciting model to Chicago shootings or Seattle car prowls.

    uv run python scripts/04_models.py shootings          # first arrest under the case number
    uv run python scripts/04_models.py shootings_violent  # first arrest with a violent or warrant charge
    uv run python scripts/04_models.py carprowl

Writes tables and a JSON summary to results/models/. The paper only reads
those files. Finished steps are cached, so an interrupted run resumes.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from arresthawkes import chicago, model, seattle

# covariate: a crime trait that may change how many crimes it triggers
# custody:   whether the arrest table says if the person was booked into jail
DATASETS = {
    "shootings": dict(city="Chicago", file="data/chicago_shootings.csv.zip", start=chicago.START, end=chicago.END,
                      pairs=dict(max_days=120.0, max_dist=2000.0), covariate="fatal", custody=False),
    "shootings_violent": dict(city="Chicago", file="data/chicago_shootings_violent.csv.zip", start=chicago.START,
                              end=chicago.END, pairs=dict(max_days=120.0, max_dist=2000.0), covariate="fatal",
                              custody=False),
    "carprowl": dict(city="Seattle", file="data/seattle_carprowl.csv.zip", start=seattle.START, end=seattle.END,
                     pairs=dict(max_days=120.0, max_dist=1000.0), covariate=None, custody=True),
}
NAME = sys.argv[1]
CFG = DATASETS[NAME]
OUT = Path("results/models")
OUT.mkdir(parents=True, exist_ok=True)
JITTER = 30.0  # meters; both departments publish the block, not the address
K_GRID = [5, 10, 15, 30]
KERNELS = ["exp", "lomax"]
PAIRS = CFG["pairs"]
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


df = seattle.load(CFG["file"])  # same file layout for both cities
T = float((pd.Timestamp(CFG["end"]) - pd.Timestamp(CFG["start"])).days)
arrest_t = df["arrest_t"].fillna(np.inf).to_numpy()
booked = df["booked"].to_numpy().astype(float)
X = (df["kind"] == CFG["covariate"]).to_numpy().astype(float) if CFG["covariate"] else None
xnames = [CFG["covariate"]] if CFG["covariate"] else None


def events(kernel, k_nn, seed=0, Z=None):
    x, y = seattle.jitter(df, JITTER, seed=seed)
    return model.build(df["t"].to_numpy(), x, y, arrest_t, T, X=X, Z=Z, k_nn=k_nn, kernel=kernel, **PAIRS)


def natural(f):
    """Named estimates on the natural scale."""
    ev = f.ev
    out = {r["name"]: r for r in f.table()}
    kd = ev.kern.describe(model.unpack(f.par, ev)[3])
    return out, kd


# ---------------------------------------------------------------------------
# 1. Descriptives

delay = (df["arrest_t"] - df["t"])[df["arrested"]]
bins = [-1, 0, 1 / 24, 0.25, 1, 7, 30, 180, 1e5]
labels = ["at the offense time", "within 1 hour", "1 to 6 hours", "6 to 24 hours", "1 to 7 days", "7 to 30 days",
          "30 to 180 days", "over 180 days"]
pd.DataFrame({"delay": labels, "arrests": pd.cut(delay, bins).value_counts(sort=False).to_numpy()}).to_csv(
    OUT / f"{NAME}_delay.csv", index=False)
yr = df.assign(year=df["time"].dt.year).groupby("year").agg(
    n=("t", "size"), arrested=("arrested", "sum"), booked=("booked", "sum")).reset_index()
yr.to_csv(OUT / f"{NAME}_years.csv", index=False)
df.groupby("kind").agg(n=("t", "size"), arrested=("arrested", "sum")).reset_index().to_csv(
    OUT / f"{NAME}_kinds.csv", index=False)
df[df["arrested"]].groupby("arrest_type").size().rename("n").reset_index().sort_values("n", ascending=False).to_csv(
    OUT / f"{NAME}_arrest_types.csv", index=False)

# Follow-on crimes near an earlier crime, by whether that crime had an arrest
# within a day ("fast arrest") or never
# had one. Model-free version of kappa.
ev0 = model.build(df["t"].to_numpy(), df["x"].to_numpy(), df["y"].to_numpy(), arrest_t, T, max_days=120.0, max_dist=200.0)
lag_edges = np.array([0, 0.25, 1, 3, 7, 14, 28, 120])
c0 = ev0.c
fast = ev0.arrested & (c0 <= 1.0)
never = ~ev0.arrested
rows = []
for lab, mask in [("arrest within a day", fast), ("no arrest", never)]:
    sel = mask[ev0.pi]
    cnt = np.histogram(ev0.dt[sel], lag_edges)[0]
    # parents observed for the whole lag bin
    for b in range(len(lag_edges) - 1):
        at_risk = (mask & (ev0.t + lag_edges[b + 1] <= T)).sum()
        rows.append(dict(group=lab, lag_from=lag_edges[b], lag_to=lag_edges[b + 1], pairs=int(cnt[b]),
                         parents=int(at_risk), rate=cnt[b] / max(at_risk, 1) / (lag_edges[b + 1] - lag_edges[b])))
pd.DataFrame(rows).to_csv(OUT / f"{NAME}_followon.csv", index=False)
log(f"{NAME}: n = {len(df):,}, arrests = {int(df['arrested'].sum()):,}")

# ---------------------------------------------------------------------------
# 2. Choose the temporal kernel and background smoothing by likelihood

# Finished steps are kept in a small cache of parameter vectors, so a run
# that is interrupted picks up where it stopped.
CACHE = OUT / f"{NAME}_fits.npz"
cache = dict(np.load(CACHE)) if CACHE.exists() else {}


def keep(**kw):
    cache.update({k: np.asarray(v) for k, v in kw.items()})
    np.savez(CACHE, **cache)


GRID = OUT / f"{NAME}_grid.csv"
grid_done = GRID.exists() and len(pd.read_csv(GRID)) == len(KERNELS) * len(K_GRID)
grid_rows, best = [], None
for kernel in ([] if grid_done else KERNELS):
    ev = events(kernel, K_GRID[0])
    start = None
    for k in K_GRID:
        ev.set_background_knn(k)
        f = model.fit(ev, start=start, se=False, xnames=xnames)
        if start is None:
            start = [f.par] + [model.start_values(ev)]
        f0 = model.fit(ev, start=f.par, se=False, fix={"log kappa": 0.0}, xnames=xnames)
        pv = model.prevented(f.par, ev)
        nat, kd = natural(f)
        grid_rows.append(dict(kernel=kernel, k=k, median_bandwidth=float(np.median(ev.eta)), loglik=f.loglik,
                              loglik_no_arrest=f0.loglik, loglik_poisson=model.poisson_loglik(ev), npar=f.k,
                              theta=nat["log theta"]["exp"], kappa=nat["log kappa"]["exp"], n0=pv["n0"],
                              sigma=nat["log sigma"]["exp"], median_days=kd["median_days"],
                              prevented=pv["total"], lr_p=model.lr_test(f, f0)[2]))
        log(f"  {kernel:5s} k={k:2d} loglik {f.loglik:.1f} theta {nat['log theta']['exp']:.3f} "
            f"kappa {nat['log kappa']['exp']:.3f} prevented {pv['total']:.1f}")
        if best is None or f.aic < best[0]:
            best = (f.aic, kernel, k, f.par)
if grid_done:
    grid = pd.read_csv(GRID)
    top = grid.loc[(2 * grid["npar"] - 2 * grid["loglik"]).idxmin()]
    KERNEL, K_NN, par_best = str(top["kernel"]), int(top["k"]), cache.get("grid_best")
    log("kernel and background grid read from the earlier run")
else:
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(GRID, index=False)
    _, KERNEL, K_NN, par_best = best
    keep(grid_best=par_best)
log(f"chosen: kernel {KERNEL}, k = {K_NN}")

# ---------------------------------------------------------------------------
# 3. Models at the chosen specification

ev = events(KERNEL, K_NN)
main = model.fit(ev, start=cache.get("main", par_best), xnames=xnames)
keep(main=main.par)
no_arrest = model.fit(ev, start=cache.get("no_arrest", main.par), fix={"log kappa": 0.0}, xnames=xnames)
static = model.fit(ev, start=cache.get("static", main.par), static=True, xnames=xnames)
keep(no_arrest=no_arrest.par, static=static.par)
log("main, no-arrest and static models fit")

K = len(ev.widths)
bookfit = None
if CFG["custody"]:
    evb = events(KERNEL, K_NN, Z=booked)
    startb = np.insert(main.par, K + ev.X.shape[1] + 1, 0.0)
    bookfit = model.fit(evb, start=cache.get("booked", startb), xnames=xnames, znames=["booked"])
    keep(booked=bookfit.par)
    log("booked model fit")

# Sensitivity: an exponential kernel held to short lags (median of a day or
# less), the timescale of a spree or a retaliation. The unrestricted fit can
# prefer a slow kernel that describes places with a bad few months instead.
SHORT = [(np.log(np.log(2.0)), np.log(500.0))]
evs = events("exp", K_NN)
short_starts = [model.start_values(evs, kpar=[np.log(2.0)], sigma=sg) for sg in (60.0, 200.0)]
short = model.fit(evs, start=cache.get("short", short_starts), xnames=xnames, kernel_bounds=SHORT)
short0 = model.fit(evs, start=short.par, fix={"log kappa": 0.0}, se=False, xnames=xnames, kernel_bounds=SHORT)
keep(short=short.par)
log(f"short-kernel model fit: theta {np.exp(short.get('log theta')):.4f}, kappa {np.exp(short.get('log kappa')):.2f}")

PROF_KEYS = ["grid", "loglik", "low", "high", "open_low", "open_high", "par_low", "par_high", "max"]
if "prof_grid" in cache:
    prof = {k: cache[f"prof_{k}"] for k in PROF_KEYS}
    prof.update({k: prof[k].item() for k in ["low", "high", "open_low", "open_high", "max"]})
else:
    prof = model.profile(main)
    keep(**{f"prof_{k}": prof[k] for k in PROF_KEYS})
log(f"profile interval for kappa: {np.exp(prof['low']):.3f} to {np.exp(prof['high']):.3f}")

models = {"Poisson background": dict(loglik=model.poisson_loglik(ev), npar=K),
          "Self-exciting, no arrest effect": dict(loglik=no_arrest.loglik, npar=no_arrest.k),
          "Arrest as a fixed mark": dict(loglik=static.loglik, npar=static.k),
          "Arrest dampening (main)": dict(loglik=main.loglik, npar=main.k)}
if bookfit is not None:
    models["Arrest dampening by custody"] = dict(loglik=bookfit.loglik, npar=bookfit.k)
mt = pd.DataFrame([dict(model=k, **v, aic=2 * v["npar"] - 2 * v["loglik"]) for k, v in models.items()])
mt.to_csv(OUT / f"{NAME}_models.csv", index=False)

coef = []
for label, f in [("main", main), ("no arrest", no_arrest), ("static", static), ("booked", bookfit)]:
    for r in (f.table() if f is not None else []):
        coef.append(dict(model=label, **r))
pd.DataFrame(coef).to_csv(OUT / f"{NAME}_coef.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Crimes prevented

pv = model.prevented(main.par, ev, by=np.where(booked[ev.order] > 0, "booked", "not booked"))
# Intervals come from the profile likelihood for kappa: the constrained fits
# at the two ends of its 95% interval (a normal approximation on log kappa
# breaks down when the estimate is near zero).
prof_total = sorted([model.prevented(prof["par_high"], ev)["total"], model.prevented(prof["par_low"], ev)["total"]])
pred_total = model.prediction_interval(pv["total"], prof_total[0], prof_total[1], pv["direct"], pv["n0"])

delays = np.r_[0, np.geomspace(1 / 24, 30, 60)]
curve = model.prevented_curve(main.par, ev, delays)
ends = np.vstack([model.prevented_curve(prof[k], ev, delays) for k in ("par_high", "par_low")])
cl, ch = ends.min(axis=0), ends.max(axis=0)
pd.DataFrame({"delay": delays, "prevented": curve, "low": cl, "high": ch}).to_csv(OUT / f"{NAME}_curve.csv", index=False)

# kernel survival: share of an event's triggering still to come after u days
kpar = model.unpack(main.par, ev)[3]
pd.DataFrame({"delay": delays, "remaining": ev.kern.surv(delays, kpar)[0]}).to_csv(OUT / f"{NAME}_kernel.csv", index=False)

ptrig = model.triggered_prob(main.par, ev)
log(f"prevented: total {pv['total']:.1f} (profile {prof_total[0]:.1f} to {prof_total[1]:.1f}; "
    f"prediction {pred_total[0]:.1f} to {pred_total[1]:.1f})")

# ---------------------------------------------------------------------------
# 5. Residuals (time rescaling) and sensitivity to the location jitter

if "resid" not in cache:
    resid = {"main": np.diff(np.r_[0, model.compensator(main.par, ev)])}
    counts = np.bincount(ev.period, minlength=K)
    edges = np.r_[0, np.cumsum(ev.widths)]
    resid["Poisson background"] = np.diff(np.r_[0, np.interp(ev.t, edges, np.r_[0, np.cumsum(counts)])])
    keep(resid=[[stats.kstest(tau, "expon").statistic, tau.mean()] for tau in resid.values()])
ks = {label: dict(ks=float(row[0]), mean=float(row[1]))
      for label, row in zip(["main", "Poisson background"], cache["resid"])}
tie_share = float((np.diff(ev.t) == 0).mean())  # crimes sharing a timestamp with the one before
on_the_hour = float((df["time"].dt.minute == 0).mean())
log("residuals done")

jit = []
for seed in [1, 2]:
    evj = events(KERNEL, K_NN, seed=seed)
    fj = model.fit(evj, start=main.par, se=False, xnames=xnames)
    jit.append(dict(seed=seed, kappa=float(np.exp(fj.get("log kappa"))), theta=float(np.exp(fj.get("log theta"))),
                    prevented=float(model.prevented(fj.par, evj)["total"])))
log("jitter sensitivity done")

nat, kd = natural(main)
natb = natural(bookfit)[0] if bookfit is not None else {}
summary = dict(
    name=NAME, city=CFG["city"], n=int(ev.n), T=T, start=CFG["start"], end=CFG["end"],
    arrests=int(ev.arrested.sum()),
    booked=int((ev.arrested & (booked[ev.order] > 0)).sum()), jitter=JITTER, kernel=KERNEL, kernel_name=ev.kern.name,
    k_nn=K_NN, median_bandwidth=float(np.median(ev.eta)), pairs=int(len(ev.pi)), kernel_par=kd,
    theta=nat["log theta"], kappa=nat["log kappa"], sigma=nat["log sigma"],
    covariate=CFG["covariate"], covariate_effect=nat.get(f"theta:{CFG['covariate']}"), kappa_static=dict(natural(static)[0]["log kappa"]),
    kappa_profile=[float(np.exp(prof["low"])), float(np.exp(prof["high"]))],
    kappa_profile_open=[bool(prof["open_low"]), bool(prof["open_high"])],
    profile=dict(grid=prof["grid"].tolist(), loglik=prof["loglik"].tolist(), max=float(prof["max"])),
    lr_arrest=dict(zip(["stat", "df", "p"], map(float, model.lr_test(main, no_arrest)))),
    lr_booked=dict(zip(["stat", "df", "p"], map(float, model.lr_test(bookfit, main)))) if bookfit else None,
    kappa_not_booked=natb.get("log kappa"), kappa_booked_ratio=natb.get("kappa:booked"),
    prevented={k: float(v) for k, v in pv.items()},
    prevented_profile=[float(prof_total[0]), float(prof_total[1])],
    prevented_prediction=[float(pred_total[0]), float(pred_total[1])],
    triggered_share=float(ptrig.mean()), residuals=ks, jitter_check=jit, tie_share=tie_share,
    on_the_hour=on_the_hour,
    short_kernel=dict(theta=natural(short)[0]["log theta"], kappa=natural(short)[0]["log kappa"],
                      sigma=natural(short)[0]["log sigma"], median_days=natural(short)[1]["median_days"],
                      loglik=float(short.loglik), lr_p=float(model.lr_test(short, short0)[2]),
                      triggered_share=float(model.triggered_prob(short.par, evs).mean()),
                      prevented=float(model.prevented(short.par, evs)["total"])),
    median_delay_days=float(np.median(ev.c[ev.arrested])), minutes=(time.time() - T0) / 60,
)
json.dump(summary, open(OUT / f"{NAME}_summary.json", "w"), indent=1, default=float)
log("done")
