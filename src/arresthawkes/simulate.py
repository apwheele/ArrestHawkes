"""Simulated crime data for checking the model.

``hawkes``  draws from the model itself: each crime triggers offspring at
            rate theta g(u) h(r), multiplied by kappa once it has an arrest.
``sprees``  an offender-based process the model does not match exactly:
            an offender commits a run of crimes and an arrest ends the run,
            sometimes clearing the offender's earlier crimes at the same time.

Both can replay the same random draws with the arrest effect switched off,
which gives the number of crimes the arrests prevented in that world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import KERNELS


@dataclass
class World:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    arrest_t: np.ndarray  # np.inf if never arrested
    T: float
    cov: np.ndarray | None  # binary productivity covariate
    parent: np.ndarray  # index of the triggering event, -1 for background
    n_counterfactual: int  # events in the same world with no arrest effect
    truth: dict

    @property
    def n(self):
        return len(self.t)

    @property
    def prevented(self):
        return self.n_counterfactual - self.n


def _background(rng, T, rate, size, hot_share=0.6, n_hot=25, hot_sd=400.0, steps=None):
    """Background events: a mix of Gaussian hot spots and a uniform layer on a square.

    ``steps`` multiplies the rate in equal-length periods (temporal trend).
    """
    steps = np.ones(1) if steps is None else np.asarray(steps, float)
    edges = np.linspace(0, T, len(steps) + 1)
    ts = []
    for k, m in enumerate(steps):
        nk = rng.poisson(rate * m * (edges[k + 1] - edges[k]))
        ts.append(rng.uniform(edges[k], edges[k + 1], nk))
    t = np.sort(np.concatenate(ts))
    n = len(t)
    centers = rng.uniform(0.15 * size, 0.85 * size, (n_hot, 2))
    wts = rng.dirichlet(np.ones(n_hot))
    hot = rng.random(n) < hot_share
    xy = rng.uniform(0, size, (n, 2))
    k = rng.choice(n_hot, n, p=wts)
    xy[hot] = centers[k[hot]] + rng.normal(0, hot_sd, (hot.sum(), 2))
    return t, xy


def _arrest_delay(rng, n, p_arrest, p_scene, mean_delay):
    """Arrest delay in days: at the scene (0) with probability p_scene, else exponential."""
    arrested = rng.random(n) < p_arrest
    delay = np.where(rng.random(n) < p_scene, 0.0, rng.exponential(mean_delay, n))
    return np.where(arrested, delay, np.inf)


def hawkes(seed, T=1460.0, size=10_000.0, rate=1.0, theta=0.35, omega=0.5, sigma=120.0, kappa=0.3,
           kernel="exp", kpar=None,
           p_arrest=0.15, p_scene=0.4, mean_delay=3.0, cov_share=0.0, cov_theta=1.0, cov_arrest=1.0,
           steps=None, max_gen=200) -> World:
    """Simulate the arrest-dampened Hawkes process by branching.

    A binary covariate (share ``cov_share``) multiplies productivity by
    ``cov_theta`` and the odds of arrest by ``cov_arrest``; with both away
    from 1 arrests are confounded unless the covariate is in the model.
    Offspring keep their own random draws when ``kappa`` is set to 1, so
    the counterfactual world differs only by the events arrests removed.
    """
    assert 0 <= kappa <= 1, "simulation by thinning needs kappa in [0, 1]"
    kpar = [np.log(omega)] if kpar is None else kpar  # kernel parameters on the model's log scale
    rng = np.random.default_rng(seed)
    t0, xy0 = _background(rng, T, rate, size, steps=steps)

    def draw_marks(n):
        cov = (rng.random(n) < cov_share).astype(float)
        odds = p_arrest / (1 - p_arrest) * np.where(cov > 0, cov_arrest, 1.0)
        delay = _arrest_delay(rng, n, odds / (1 + odds), p_scene, mean_delay)
        return cov, delay

    # Each event carries: time, xy, cov, arrest delay, parent, and `alive`
    # (present in the factual world; every event exists in the counterfactual).
    cov0, del0 = draw_marks(len(t0))
    T_, XY, COV, DEL, PAR, ALIVE = [t0], [xy0], [cov0], [del0], [np.full(len(t0), -1)], [np.ones(len(t0), bool)]
    gen_t, gen_xy, gen_cov, gen_del, gen_alive = t0, xy0, cov0, del0, np.ones(len(t0), bool)
    offset = 0
    for _ in range(max_gen):
        th = theta * np.where(gen_cov > 0, cov_theta, 1.0)
        k = rng.poisson(th)  # offspring with no arrest effect
        if k.sum() == 0:
            break
        par = np.repeat(np.arange(len(gen_t)), k)
        u = KERNELS[kernel].sample(rng, kpar, len(par))
        ct = gen_t[par] + u
        cxy = gen_xy[par] + rng.normal(0, sigma, (len(par), 2))
        # an offspring due after the parent's arrest survives with probability kappa
        after = u > gen_del[par]
        survive = ~after | (rng.random(len(par)) < kappa)
        keep = (ct < T) & (cxy.min(axis=1) >= 0) & (cxy.max(axis=1) <= size)
        par, ct, cxy, survive = par[keep], ct[keep], cxy[keep], survive[keep]
        ccov, cdel = draw_marks(len(ct))
        calive = gen_alive[par] & survive
        T_.append(ct); XY.append(cxy); COV.append(ccov); DEL.append(cdel)
        PAR.append(par + offset); ALIVE.append(calive)
        offset += len(gen_t)
        gen_t, gen_xy, gen_cov, gen_del, gen_alive = ct, cxy, ccov, cdel, calive
    t, xy, cov, dl, par, alive = map(np.concatenate, (T_, XY, COV, DEL, PAR, ALIVE))
    n_cf = len(t)
    # keep the factual world, re-index parents
    idx = np.flatnonzero(alive)
    remap = -np.ones(n_cf, int)
    remap[idx] = np.arange(len(idx))
    t, xy, cov, dl = t[idx], xy[idx], cov[idx], dl[idx]
    par = np.where(par[idx] >= 0, remap[par[idx]], -1)
    arrest_t = t + dl
    arrest_t = np.where(arrest_t <= T, arrest_t, np.inf)
    o = np.argsort(t, kind="stable")
    inv = np.empty(len(o), int)
    inv[o] = np.arange(len(o))
    par = np.where(par[o] >= 0, inv[par[o]], -1)
    truth = dict(theta=theta, kpar=list(kpar), kernel=kernel, sigma=sigma, kappa=kappa, cov_theta=cov_theta)
    return World(t[o], xy[o, 0], xy[o, 1], arrest_t[o], T, cov[o] if cov_share > 0 else None, par, n_cf, truth)


def sprees(seed, T=1460.0, size=10_000.0, rate=1.0, p_continue=0.5, mean_gap=3.0, step_sd=150.0,
           p_caught=0.15, p_scene=0.5, mean_delay=5.0, p_clear_prior=0.0) -> World:
    """Offender sprees ended by arrest.

    Sprees start as background events. After each crime the offender commits
    another with probability ``p_continue`` after an exponential gap (mean
    ``mean_gap`` days), a short step away. Each crime leads to an arrest with
    probability ``p_caught``; the arrest comes at the scene with probability
    ``p_scene``, otherwise after an exponential delay. An arrest removes
    every later crime in the spree. With probability ``p_clear_prior`` the
    arrest also clears each earlier crime of the spree, logged at the same
    arrest time (a retrospective clearance, which prevents nothing more).
    """
    rng = np.random.default_rng(seed)
    t0, xy0 = _background(rng, T, rate, size)
    t_l, xy_l, arr_l, par_l = [], [], [], []
    n_cf = 0
    for s in range(len(t0)):
        tt, pos = t0[s], xy0[s]
        stop = np.inf  # time the offender is taken off the street
        members = []
        prev = -1
        while True:
            if tt >= T or pos.min() < 0 or pos.max() > size:
                break
            n_cf += 1
            if tt < stop:
                i = len(t_l)
                t_l.append(tt); xy_l.append(pos); arr_l.append(np.inf); par_l.append(prev)
                members.append(i)
                prev = i
                if rng.random() < p_caught:
                    a = tt if rng.random() < p_scene else tt + rng.exponential(mean_delay)
                    if a < stop:
                        stop = a
                    arr_l[i] = min(arr_l[i], a)
            if rng.random() >= p_continue:
                break
            tt = tt + rng.exponential(mean_gap)
            pos = pos + rng.normal(0, step_sd, 2)
        if np.isfinite(stop) and p_clear_prior > 0:
            for i in members:
                if not np.isfinite(arr_l[i]) and t_l[i] < stop and rng.random() < p_clear_prior:
                    arr_l[i] = stop
    t, xy, arrest_t, par = np.array(t_l), np.array(xy_l), np.array(arr_l), np.array(par_l)
    arrest_t = np.where(arrest_t <= T, arrest_t, np.inf)
    o = np.argsort(t, kind="stable")
    inv = np.empty(len(o), int)
    inv[o] = np.arange(len(o))
    par = np.where(par[o] >= 0, inv[par[o]], -1)
    truth = dict(p_continue=p_continue, omega=1 / mean_gap, sigma=step_sd)
    return World(t[o], xy[o, 0], xy[o, 1], arrest_t[o], T, None, par, n_cf, truth)
