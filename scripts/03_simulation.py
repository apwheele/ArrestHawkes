"""Monte Carlo check of the arrest-dampened self-exciting model.

    uv run python scripts/03_simulation.py <scenario> <first seed> <last seed>
    uv run python scripts/03_simulation.py all 0 99

Scenarios (see SCENARIOS): data drawn from the model itself, with late
arrests, with a confounder, and from an offender-spree process the model
only approximates. One CSV per scenario and seed in results/sim/rows, so
the run is resumable.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arresthawkes import model, simulate

OUT = Path("results/sim/rows")
OUT.mkdir(parents=True, exist_ok=True)
K_NN = 15

BASE = dict(T=1460.0, size=10_000.0, rate=2.0, theta=0.35, omega=0.5, sigma=120.0, kappa=0.3, p_arrest=0.15)
SPREE = dict(T=1460.0, size=10_000.0, rate=2.0, p_continue=0.45, mean_gap=2.0, step_sd=120.0, p_caught=0.15)
SCENARIOS = {
    # arrests mostly at the scene or within a few days
    "model": ("hawkes", dict(BASE, p_scene=0.4, mean_delay=3.0)),
    # no arrest effect at all (false positive rate of the test)
    "null": ("hawkes", dict(BASE, kappa=1.0, p_scene=0.4, mean_delay=3.0)),
    # every arrest comes after a delay about as long as the triggering lasts
    "late": ("hawkes", dict(BASE, p_scene=0.0, mean_delay=2.0)),
    # a crime trait raises both productivity (x2) and the odds of arrest (x4)
    "confounded": ("hawkes", dict(BASE, p_scene=0.4, mean_delay=3.0, cov_share=0.3, cov_theta=2.0, cov_arrest=4.0)),
    # offender sprees ended by arrest
    "spree": ("sprees", dict(SPREE, p_scene=0.5, mean_delay=2.0)),
    # ... where the arrest also clears most of the offender's earlier crimes
    "spree_clear": ("sprees", dict(SPREE, p_scene=0.5, mean_delay=2.0, p_clear_prior=0.7)),
}


def one(scenario, seed):
    gen, kw = SCENARIOS[scenario]
    w = getattr(simulate, gen)(seed, **kw)
    rows = []
    variants = [("dynamic", False, False), ("static", True, False)]
    if w.cov is not None:
        variants.append(("dynamic + covariate", False, True))
    for label, static, use_cov in variants:
        ev = model.build(w.t, w.x, w.y, w.arrest_t, w.T, X=w.cov if use_cov else None, k_nn=K_NN)
        t0 = time.time()
        f = model.fit(ev, static=static)
        f0 = model.fit(ev, start=f.par, fix={"log kappa": 0.0}, se=False, static=static)
        pv = model.prevented(f.par, ev)
        # Wald interval from draws of the asymptotic normal, and the profile
        # likelihood interval for kappa carried through to crimes prevented
        wlo, whi = model.interval([model.prevented(p, ev)["total"] for p in model.draws(f, lambda p: p, n=400, seed=seed)])
        pr = model.profile(f) if label != "static" else None
        lo = model.prevented(pr["par_high"], ev)["total"] if pr else np.nan
        hi = model.prevented(pr["par_low"], ev)["total"] if pr else np.nan
        rows.append(dict(
            scenario=scenario, seed=seed, method=label, n=w.n, arrests=int(np.isfinite(w.arrest_t).sum()),
            truth=w.prevented, est=pv["total"], low=lo, high=hi, direct=pv["direct"], n0=pv["n0"],
            wald_low=wlo, wald_high=whi, log_kappa=f.get("log kappa"), se_log_kappa=f.get_se("log kappa"),
            kappa_low=np.exp(pr["low"]) if pr else np.nan, kappa_high=np.exp(pr["high"]) if pr else np.nan,
            theta=np.exp(f.get("log theta")), omega=np.exp(f.get("log omega")), sigma=np.exp(f.get("log sigma")),
            lr_p=model.lr_test(f, f0)[2], converged=f.converged, seconds=time.time() - t0,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    names = list(SCENARIOS) if which == "all" else [which]
    todo = [(s, k) for s in names for k in range(first, last + 1) if not (OUT / f"{s}_{k:04d}.csv").exists()]
    print(f"{len(todo)} replications to run", flush=True)
    t0 = time.time()
    for i, (s, k) in enumerate(todo):
        one(s, k).to_csv(OUT / f"{s}_{k:04d}.csv", index=False)
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print(f"{i + 1}/{len(todo)} done ({s} seed {k}), {el / 60:.1f} min elapsed, "
                  f"about {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min left", flush=True)
