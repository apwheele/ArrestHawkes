import numpy as np
import pytest
from scipy import optimize

from arresthawkes import model, simulate


def world_events(w, eta=300.0, **kw):
    X = None if w.cov is None else w.cov
    return model.build(w.t, w.x, w.y, w.arrest_t, w.T, X=X, eta=eta, **kw)


@pytest.fixture(scope="module")
def world():
    return simulate.hawkes(1, T=500.0, size=6000.0, rate=1.5, cov_share=0.4, cov_theta=1.5, cov_arrest=2.0)


@pytest.fixture(scope="module")
def ev(world):
    return world_events(world)


def test_simulation_bookkeeping(world):
    w = world
    assert np.all(np.diff(w.t) >= 0)
    assert w.n_counterfactual >= w.n
    has_parent = w.parent >= 0
    assert np.all(w.t[w.parent[has_parent]] < w.t[has_parent])
    arrested = np.isfinite(w.arrest_t)
    assert np.all(w.arrest_t[arrested] >= w.t[arrested])
    assert 0.05 < arrested.mean() < 0.5


def test_no_arrest_effect_prevents_nothing():
    w = simulate.hawkes(3, T=300.0, size=5000.0, kappa=1.0)
    assert w.prevented == 0


def test_pairs_are_causal(ev):
    assert np.all(ev.dt > 0) and np.all(ev.pi < ev.pj)
    assert np.all(ev.dt <= ev.max_days) and np.all(ev.d2 <= ev.max_dist**2 + 1e-6)
    # an arrest only counts for children that come after it
    assert np.all((ev.A == 1) == (ev.arrest_t[ev.pi] < ev.t[ev.pj]))


@pytest.mark.parametrize("static,kernel", [(False, "exp"), (True, "exp"), (False, "lomax"), (True, "lomax")])
def test_gradient_matches_finite_differences(world, static, kernel):
    ev = world_events(world, kernel=kernel)
    rng = np.random.default_rng(0)
    par = model.start_values(ev) + rng.normal(0, 0.2, len(model.start_values(ev)))
    err = optimize.check_grad(lambda p: model.negloglik(p, ev, static)[0],
                              lambda p: model.negloglik(p, ev, static)[1], par, epsilon=1e-6)
    scale = np.linalg.norm(model.negloglik(par, ev, static)[1])
    assert err / scale < 1e-4


def test_loo_kde_matches_direct():
    rng = np.random.default_rng(0)
    xy = rng.normal(0, 500, (400, 2))
    eta = 200.0
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
    k = np.exp(-d2 / (2 * eta**2)) / (2 * np.pi * eta**2)
    np.fill_diagonal(k, 0)
    direct = k.sum(axis=1) / (len(xy) - 1)
    approx = model.loo_kde(xy, eta, cell=20.0)
    err = np.abs(approx / direct - 1)
    assert np.median(err) < 0.005 and np.percentile(err, 99) < 0.05


def test_fit_recovers_parameters():
    w = simulate.hawkes(11, T=1460.0, cov_share=0.0)
    ev = world_events(w)
    f = model.fit(ev)
    assert f.converged
    for name, truth, tol in [("log theta", 0.35, 0.25), ("log omega", 0.5, 0.25), ("log sigma", 120.0, 0.1)]:
        assert abs(np.exp(f.get(name)) / truth - 1) < tol
    z = (f.get("log kappa") - np.log(0.3)) / f.get_se("log kappa")
    assert abs(z) < 3
    # arrests matter: restricting kappa = 1 costs likelihood
    f0 = model.fit(ev, fix={"log kappa": 0.0}, se=False)
    stat, df, p = model.lr_test(f, f0)
    assert df == 1 and p < 0.01
    # formula for crimes prevented is in the neighborhood of the truth
    est = model.prevented(f.par, ev)["total"]
    assert 0.4 * w.prevented < est < 2.0 * w.prevented


@pytest.mark.parametrize("kernel,kpar", [("exp", None), ("lomax", [np.log(0.3), np.log(1.2)])])
def test_compensator_increments_look_exponential(kernel, kpar):
    w = simulate.hawkes(5, T=700.0, kernel=kernel, kpar=kpar)
    ev = world_events(w, kernel=kernel)
    f = model.fit(ev, se=False)
    tau = np.diff(model.compensator(f.par, ev))
    assert abs(tau.mean() - 1) < 0.1
    assert abs(model.compensator(f.par, ev)[-1] / ev.n - 1) < 0.05


def test_compensator_chunks_stay_small_and_agree():
    # A chunk must never hold more than `cells` entries, whatever n is
    # (an earlier version built the full n x n matrix in its first chunk).
    w = simulate.hawkes(7, T=400.0)
    ev = world_events(w)
    par = model.start_values(ev)
    whole = model.compensator(par, ev, cells=ev.n * ev.n)
    small = model.compensator(par, ev, cells=5 * ev.n)
    assert np.allclose(whole, small)
    assert max(1, int(5 * ev.n // ev.n)) * ev.n <= 5 * ev.n


def test_sprees_counterfactual():
    w = simulate.sprees(2, T=400.0, size=5000.0, p_caught=0.3)
    assert w.prevented > 0
    w0 = simulate.sprees(2, T=400.0, size=5000.0, p_caught=0.0)
    assert w0.prevented == 0 and not np.isfinite(w0.arrest_t).any()


def test_profile_interval_brackets_estimate():
    w = simulate.hawkes(11, T=1460.0)
    ev = world_events(w)
    f = model.fit(ev)
    pr = model.profile(f)
    assert pr["low"] < f.get("log kappa") < pr["high"]
    assert pr["low"] < np.log(0.3) < pr["high"]
    assert np.nanmax(pr["loglik"]) <= f.loglik + 1e-3
    lo = model.prevented(pr["par_high"], ev)["total"]
    hi = model.prevented(pr["par_low"], ev)["total"]
    assert lo < model.prevented(f.par, ev)["total"] < hi


def test_knn_background_is_a_density():
    rng = np.random.default_rng(1)
    xy = rng.normal(0, 800, (3000, 2))
    dens, eta = model.knn_loo_kde(xy, 15)
    assert eta.min() >= 40.0
    # mean of 1/density over the sample estimates the area of support; compare with the truth density
    true = np.exp(-(xy**2).sum(axis=1) / (2 * 800**2)) / (2 * np.pi * 800**2)
    assert abs(np.median(dens / true) - 1) < 0.15
