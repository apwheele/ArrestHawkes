"""Self-exciting point process in which an arrest dampens triggering.

Conditional intensity at time ``t`` (days) and location ``s`` (meters):

    lambda(t, s) = nu(t) b(s)
                 + sum_{i: t_i < t} theta_i kappa_i^{A_i(t)} g(t - t_i) h(s - s_i)

    nu(t)     piecewise-constant background rate (events per day), exp(alpha_k)
    b(s)      leave-one-out Gaussian kernel density of all events, bandwidth eta
    theta_i   exp(x_i' gamma), expected offspring of event i absent an arrest
    kappa_i   exp(z_i' beta), multiplier on the triggering left after the arrest
    A_i(t)    1 if event i has an arrest before t, else 0
    g(u)      temporal kernel, a density on u > 0 with survival function S(u):
              exponential, omega exp(-omega u), or Lomax (power law),
              (p - 1) c^(p-1) (u + c)^(-p)
    h(r)      bivariate Gaussian density with standard deviation sigma

Parameter vector: [alpha (K), gamma (px), beta (pz), kernel parameters, log sigma].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize, stats
from scipy.spatial import cKDTree

TWO_PI = 2 * np.pi


# ---------------------------------------------------------------------------
# Temporal kernels: log density, survival, and derivatives in the parameters


class ExpKernel:
    """g(u) = omega exp(-omega u). Parameter: log omega."""

    name = "exponential"
    names = ["log omega"]
    starts = [[np.log(2.0)], [np.log(0.3)]]
    bounds = [(np.log(1e-3), np.log(500.0))]

    @staticmethod
    def logg(u, k):
        w = np.exp(k[0])
        return k[0] - w * u, [1 - w * u]

    @staticmethod
    def surv(u, k):
        w = np.exp(k[0])
        S = np.exp(-w * u)
        return S, [-w * u * S]

    @staticmethod
    def sample(rng, k, n):
        return rng.exponential(1 / np.exp(k[0]), n)

    @staticmethod
    def describe(k):
        w = float(np.exp(k[0]))
        return {"omega": w, "median_days": float(np.log(2) / w)}


class LomaxKernel:
    """g(u) = (p - 1) c^(p-1) (u + c)^(-p). Parameters: log c, log(p - 1)."""

    name = "power law"
    names = ["log c", "log (p-1)"]
    starts = [[np.log(0.05), np.log(0.5)], [np.log(1.0), np.log(1.5)]]
    bounds = [(np.log(1e-4), np.log(200.0)), (np.log(0.02), np.log(50.0))]

    @staticmethod
    def logg(u, k):
        c, q = np.exp(k[0]), np.exp(k[1])  # q = p - 1
        l1p = np.log1p(u / c)
        return k[1] - k[0] - (q + 1) * l1p, [q - (q + 1) * c / (u + c), 1 - q * l1p]

    @staticmethod
    def surv(u, k):
        c, q = np.exp(k[0]), np.exp(k[1])
        l1p = np.log1p(u / c)
        S = np.exp(-q * l1p)
        return S, [S * q * u / (u + c), -S * q * l1p]

    @staticmethod
    def sample(rng, k, n):
        c, q = np.exp(k[0]), np.exp(k[1])
        return c * ((1 - rng.random(n)) ** (-1 / q) - 1)

    @staticmethod
    def describe(k):
        c, q = float(np.exp(k[0])), float(np.exp(k[1]))
        return {"c": c, "p": q + 1, "median_days": c * (2 ** (1 / q) - 1)}


KERNELS = {"exp": ExpKernel, "lomax": LomaxKernel}


# ---------------------------------------------------------------------------
# Data container


@dataclass
class Events:
    """Events sorted by time, with the pair list used by the likelihood.

    t         event times in days, in [0, T]
    xy        (n, 2) coordinates in meters
    arrest_t  arrest time in days (>= t), np.inf if none or after T
    X         (n, px) productivity covariates, first column ones
    Z         (n, pz) arrest-effect covariates, first column ones
    period    (n,) integer background period of each event
    widths    (K,) length in days of each background period
    """

    t: np.ndarray
    xy: np.ndarray
    arrest_t: np.ndarray
    T: float
    X: np.ndarray
    Z: np.ndarray
    period: np.ndarray
    widths: np.ndarray
    max_days: float = 90.0
    max_dist: float = 1500.0
    kernel: str = "exp"
    b: np.ndarray | None = None
    eta: float | None = None
    pi: np.ndarray = field(init=False, repr=False)
    pj: np.ndarray = field(init=False, repr=False)
    dt: np.ndarray = field(init=False, repr=False)
    d2: np.ndarray = field(init=False, repr=False)
    A: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        assert np.all(np.diff(self.t) >= 0), "events must be sorted by time"
        self.arrest_t = np.where(self.arrest_t <= self.T, np.maximum(self.arrest_t, self.t), np.inf)
        self._pairs()

    @property
    def n(self):
        return len(self.t)

    @property
    def kern(self):
        return KERNELS[self.kernel]

    @property
    def arrested(self):
        return np.isfinite(self.arrest_t)

    @property
    def c(self):
        """Days of full-strength triggering inside the window (to arrest or T)."""
        return np.minimum(self.arrest_t, self.T) - self.t

    def _pairs(self):
        """Parent i, child j with 0 < t_j - t_i <= max_days and distance <= max_dist.

        Built block by block in time so the search never holds pairs that are
        close in space but years apart.
        """
        edges = np.arange(0, self.T + 2 * self.max_days, self.max_days)
        cut = np.searchsorted(self.t, edges)
        blocks = [np.arange(cut[k], cut[k + 1]) for k in range(len(cut) - 1)]
        trees = [cKDTree(self.xy[ix]) if len(ix) else None for ix in blocks]
        PI, PJ, D = [np.empty(0, int)], [np.empty(0, int)], [np.empty(0)]
        for k, ix in enumerate(blocks):
            for prev in ([k - 1] if k > 0 else []) + [k]:
                if trees[prev] is None or trees[k] is None:
                    continue
                sp = trees[prev].sparse_distance_matrix(trees[k], self.max_dist, output_type="ndarray")
                i, j = blocks[prev][sp["i"]], ix[sp["j"]]
                dt = self.t[j] - self.t[i]
                keep = (dt > 0) & (dt <= self.max_days)
                PI.append(i[keep])
                PJ.append(j[keep])
                D.append(sp["v"][keep])
        self.pi, self.pj = np.concatenate(PI), np.concatenate(PJ)
        self.dt = self.t[self.pj] - self.t[self.pi]
        self.d2 = np.concatenate(D) ** 2
        # An arrest logged at the very time of another crime is taken to come
        # after it (the offender was caught at that crime), hence the strict <.
        self.A = (self.arrest_t[self.pi] < self.t[self.pj]).astype(float)

    def set_background(self, eta: float, cell: float = 25.0, weights=None):
        """Leave-one-out Gaussian kernel density at each event (per square meter).

        Computed on a grid of ``cell`` meters with a Gaussian filter, which
        is accurate when ``eta`` is a few cells or more.
        """
        self.b = loo_kde(self.xy, eta, cell=cell, weights=weights)
        self.eta = eta
        return self

    def set_background_knn(self, k: int, floor: float = 40.0):
        """Variable-bandwidth leave-one-out kernel density: each event's
        kernel has bandwidth equal to the distance to its k-th nearest
        neighbor (Mohler et al. 2011), no smaller than ``floor`` meters."""
        self.b, self.eta = knn_loo_kde(self.xy, k, floor=floor)
        self.k_nn = k
        return self


def loo_kde(xy, eta, cell=25.0, weights=None):
    pad = 5 * eta + 2 * cell
    x0, y0 = xy.min(axis=0) - pad
    nx = int(np.ceil((xy[:, 0].max() + pad - x0) / cell)) + 1
    ny = int(np.ceil((xy[:, 1].max() + pad - y0) / cell)) + 1
    fx, fy = (xy[:, 0] - x0) / cell - 0.5, (xy[:, 1] - y0) / cell - 0.5  # position in cell-center units
    ix, iy = np.rint(fx).astype(int), np.rint(fy).astype(int)
    w = np.ones(len(xy)) if weights is None else np.asarray(weights, float)
    grid = np.zeros((nx, ny))
    np.add.at(grid, (ix, iy), w)
    sm = ndimage.gaussian_filter(grid, eta / cell, mode="constant", truncate=5.0)
    val = ndimage.map_coordinates(sm, [fx, fy], order=1, mode="nearest")
    # remove the weight a point contributes to its own location
    r = int(5.0 * eta / cell + 0.5)
    k1 = np.exp(-0.5 * (np.arange(-r, r + 1) / (eta / cell)) ** 2)
    k1 /= k1.sum()
    ox, oy = np.abs(fx - ix), np.abs(fy - iy)
    own = ((1 - ox) * k1[r] + ox * k1[r + 1]) * ((1 - oy) * k1[r] + oy * k1[r + 1])
    dens = (val - w * own) / (w.sum() - w) / cell**2
    return np.maximum(dens, 1e-300)


def knn_loo_kde(xy, k, floor=40.0, cap=3000.0, trunc=4.0, chunk=4000):
    """Leave-one-out density at each point from Gaussian kernels whose
    bandwidth is the distance from the kernel's own point to its k-th
    nearest neighbor. Returns the density and the bandwidths."""
    tree = cKDTree(xy)
    eta = np.clip(tree.query(xy, k=k + 1)[0][:, -1], floor, cap)
    n = len(xy)
    dens = np.zeros(n)
    for part in np.array_split(np.argsort(eta), max(1, n // chunk)):
        nb = tree.query_ball_point(xy[part], r=trunc * eta[part])
        src = np.repeat(part, np.fromiter((len(v) for v in nb), int, len(nb)))
        dst = np.concatenate([np.asarray(v, int) for v in nb])
        keep = src != dst
        src, dst = src[keep], dst[keep]
        d2 = ((xy[src] - xy[dst]) ** 2).sum(axis=1)
        e = eta[src]
        dens += np.bincount(dst, weights=np.exp(-d2 / (2 * e**2)) / (TWO_PI * e**2), minlength=n)
    return np.maximum(dens / (n - 1), 1e-300), eta


def make_periods(t, T, days=91.3125):
    """Split [0, T] into equal background periods of about ``days`` days."""
    K = max(1, int(round(T / days)))
    edges = np.linspace(0, T, K + 1)
    period = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, K - 1)
    return period, np.diff(edges)


def build(t, x, y, arrest_t, T, X=None, Z=None, period_days=91.3125, eta=None, k_nn=None, **kw) -> Events:
    """Convenience constructor: sorts by time and adds intercept columns.

    ``ev.order`` holds the sort order, so row ``k`` of the result is row
    ``ev.order[k]`` of the inputs.
    """
    t = np.asarray(t, float)
    o = np.argsort(t, kind="stable")
    n = len(t)
    X = np.ones((n, 1)) if X is None else np.column_stack([np.ones(n), X])
    Z = np.ones((n, 1)) if Z is None else np.column_stack([np.ones(n), Z])
    period, widths = make_periods(t[o], T, period_days)
    ev = Events(t[o], np.column_stack([x, y])[o].astype(float), np.asarray(arrest_t, float)[o], float(T),
                X[o], Z[o], period, widths, **kw)
    ev.order = o
    if k_nn is not None:
        ev.set_background_knn(k_nn)
    elif eta is not None:
        ev.set_background(eta)
    return ev


# ---------------------------------------------------------------------------
# Likelihood


def unpack(par, ev: Events):
    K, px, pz = len(ev.widths), ev.X.shape[1], ev.Z.shape[1]
    alpha, gamma, beta = par[:K], par[K:K + px], par[K + px:K + px + pz]
    return alpha, gamma, beta, par[K + px + pz:-1], np.exp(par[-1])


def _parts(par, ev: Events, static=False):
    alpha, gamma, beta, kpar, sigma = unpack(par, ev)
    theta = np.exp(ev.X @ gamma)
    kappa = np.exp(ev.Z @ beta)
    if static:
        # arrest treated as a fixed mark of the event: dampened from the start
        A = ev.arrested[ev.pi].astype(float)
        c = np.where(ev.arrested, 0.0, ev.T - ev.t)
    else:
        A, c = ev.A, ev.c
    mu = np.exp(alpha[ev.period]) * ev.b
    logg, dlogg = ev.kern.logg(ev.dt, kpar)
    kern = np.exp(logg - ev.d2 / (2 * sigma**2)) / (TWO_PI * sigma**2)
    contrib = theta[ev.pi] * np.where(A > 0, kappa[ev.pi], 1.0) * kern
    trig = np.bincount(ev.pj, weights=contrib, minlength=ev.n)
    lam = mu + trig
    Sc, dSc = ev.kern.surv(c, kpar)
    ST, dST = ev.kern.surv(ev.T - ev.t, kpar)
    return dict(alpha=alpha, theta=theta, kappa=kappa, kpar=kpar, sigma=sigma, A=A, c=c, mu=mu,
                contrib=contrib, trig=trig, lam=lam, Sc=Sc, ST=ST, dSc=dSc, dST=dST, dlogg=dlogg)


def negloglik(par, ev: Events, static=False, grad=True):
    """Negative log-likelihood and gradient. ``static=True`` gives the naive
    model that treats an eventual arrest as a mark known at the event time."""
    p = _parts(par, ev, static)
    theta, kappa, sigma = p["theta"], p["kappa"], p["sigma"]
    pre = 1 - p["Sc"]  # share of offspring due before the arrest (or T)
    post = kappa * (p["Sc"] - p["ST"])  # share due after it, dampened
    comp = np.exp(p["alpha"]) @ ev.widths + theta @ (pre + post)
    ll = np.log(p["lam"]).sum() - comp
    if not grad:
        return -ll
    w = p["contrib"] / p["lam"][ev.pj]
    wi = np.bincount(ev.pi, weights=w, minlength=ev.n)
    wiA = np.bincount(ev.pi, weights=w * p["A"], minlength=ev.n)
    K = len(ev.widths)
    g_alpha = np.bincount(ev.period, weights=p["mu"] / p["lam"], minlength=K) - np.exp(p["alpha"]) * ev.widths
    g_gamma = ev.X.T @ (wi - theta * (pre + post))
    g_beta = ev.Z.T @ (wiA - theta * post)
    g_kern = [(w * dl).sum() - theta @ (-dc + kappa * (dc - dT))
              for dl, dc, dT in zip(p["dlogg"], p["dSc"], p["dST"])]
    g_ls = (w * (ev.d2 / sigma**2 - 2)).sum()
    return -ll, -np.concatenate([g_alpha, g_gamma, g_beta, g_kern, [g_ls]])


def poisson_loglik(ev: Events):
    """Log-likelihood of the background-only model (theta = 0), closed form."""
    counts = np.bincount(ev.period, minlength=len(ev.widths))
    rate = np.maximum(counts, 1e-12) / ev.widths
    return (counts * np.log(rate)).sum() + np.log(ev.b).sum() - counts.sum()


# ---------------------------------------------------------------------------
# Fitting


@dataclass
class Fit:
    par: np.ndarray
    loglik: float
    cov: np.ndarray | None
    names: list
    ev: Events
    static: bool = False
    converged: bool = True
    fixed: dict = field(default_factory=dict)

    @property
    def se(self):
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))

    @property
    def k(self):
        return len(self.par) - len(self.fixed)

    @property
    def aic(self):
        return 2 * self.k - 2 * self.loglik

    def get(self, name):
        return self.par[self.names.index(name)]

    def get_se(self, name):
        return self.se[self.names.index(name)]

    def table(self):
        """Structural parameters (not the background steps): log-scale
        coefficient and standard error, exponentiated estimate and 95% interval."""
        rows = []
        for i in range(len(self.ev.widths), len(self.par)):
            est = self.par[i]
            se = self.se[i] if self.cov is not None and self.names[i] not in self.fixed else np.nan
            rows.append(dict(name=self.names[i], coef=est, se=se, exp=np.exp(est),
                             low=np.exp(est - 1.96 * se), high=np.exp(est + 1.96 * se)))
        return rows


def param_names(ev: Events, xnames=None, znames=None):
    K, px, pz = len(ev.widths), ev.X.shape[1], ev.Z.shape[1]
    xn = ["theta"] + list(xnames or [f"x{i}" for i in range(1, px)])
    zn = ["kappa"] + list(znames or [f"z{i}" for i in range(1, pz)])
    return [f"alpha{k}" for k in range(K)] + [f"log {v}" if i == 0 else f"theta:{v}" for i, v in enumerate(xn)] \
        + [f"log {v}" if i == 0 else f"kappa:{v}" for i, v in enumerate(zn)] + list(ev.kern.names) + ["log sigma"]


def start_values(ev: Events, theta=0.2, kpar=None, sigma=150.0):
    K, px, pz = len(ev.widths), ev.X.shape[1], ev.Z.shape[1]
    counts = np.bincount(ev.period, minlength=K)
    alpha = np.log(np.maximum(counts, 1) * (1 - theta) / ev.widths)
    gamma = np.r_[np.log(theta), np.zeros(px - 1)]
    kpar = ev.kern.starts[0] if kpar is None else kpar
    return np.concatenate([alpha, gamma, np.zeros(pz), kpar, [np.log(sigma)]])


def multi_starts(ev: Events, sigmas=(60.0, 250.0), thetas=(0.2,)):
    return [start_values(ev, th, kp, sg) for kp in ev.kern.starts for sg in sigmas for th in thetas]


def hessian(par, ev: Events, static=False, free=None, step=1e-4):
    """Central-difference Hessian of the negative log-likelihood from analytic gradients."""
    free = np.arange(len(par)) if free is None else free
    H = np.zeros((len(free), len(free)))
    for a, i in enumerate(free):
        h = step * max(1.0, abs(par[i]))
        up, dn = par.copy(), par.copy()
        up[i] += h
        dn[i] -= h
        H[a] = (negloglik(up, ev, static)[1][free] - negloglik(dn, ev, static)[1][free]) / (2 * h)
    return (H + H.T) / 2


def fit(ev: Events, start=None, static=False, fix=None, se=True, xnames=None, znames=None,
        beta_bounds=(-7.0, 4.0), sigma_min=10.0, maxiter=3000, kernel_bounds=None) -> Fit:
    """Maximum likelihood by L-BFGS-B.

    ``start`` is one parameter vector or a list of them (the best optimum is
    kept); the default tries short and long kernels with small and large
    sigma, since the likelihood has more than one mode. ``fix`` maps
    parameter names to fixed values (``{"log kappa": 0.0}`` is the model
    with no arrest effect). ``kernel_bounds`` replaces the default bounds on
    the temporal kernel's parameters, e.g. to hold the kernel to short lags.
    """
    assert ev.b is not None, "call ev.set_background(eta) first"
    names = param_names(ev, xnames, znames)
    starts = multi_starts(ev) if start is None else [np.array(v, float) for v in np.atleast_2d(start)]
    K, px, pz = len(ev.widths), ev.X.shape[1], ev.Z.shape[1]
    bounds = [(-30, 15)] * K + [(-12, 3)] * px + [beta_bounds] * pz + list(kernel_bounds or ev.kern.bounds) \
        + [(np.log(sigma_min), np.log(ev.max_dist))]
    fix = dict(fix or {})
    for nm, v in fix.items():
        bounds[names.index(nm)] = (v, v)
    lo, hi = np.array(bounds).T
    best = None
    for par0 in starts:
        for nm, v in fix.items():
            par0[names.index(nm)] = v
        res = optimize.minimize(negloglik, np.clip(par0, lo, hi), args=(ev, static), jac=True, method="L-BFGS-B",
                                bounds=bounds, options=dict(maxiter=maxiter, maxfun=5 * maxiter, ftol=1e-13,
                                                            gtol=1e-6, maxcor=30))
        if best is None or res.fun < best.fun:
            best = res
    par = best.x
    cov = None
    if se:
        free = np.array([i for i, nm in enumerate(names) if nm not in fix])
        H = hessian(par, ev, static, free)
        cov = np.zeros((len(par), len(par)))
        try:
            cov[np.ix_(free, free)] = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov[np.ix_(free, free)] = np.linalg.pinv(H)
    return Fit(par, -best.fun, cov, names, ev, static, bool(best.success), fix)


PROFILE_GRID = (-7.0, -5.5, -4.5, -3.75, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)


def profile(fit_: Fit, name="log kappa", grid=PROFILE_GRID, level=0.95):
    """Profile likelihood for one parameter (default the arrest effect).

    Refits with the parameter fixed at each grid value, moving outward from
    the estimate with warm starts. Returns the grid, the profile
    log-likelihoods, the interval where twice the drop in log-likelihood is
    below the chi-square cutoff, and the constrained fits at its two ends
    (``open_low``/``open_high`` flag an interval that runs off the grid).
    """
    ev, est = fit_.ev, fit_.get(name)
    crit = stats.chi2.ppf(level, 1) / 2
    refit = lambda v, start: fit(ev, start=start, static=fit_.static, se=False, fix={**fit_.fixed, name: v})
    grid = np.array(sorted(grid))
    ll = np.full(len(grid), np.nan)
    ends, pars = {}, {}
    for side, idx in (("low", np.flatnonzero(grid < est)[::-1]), ("high", np.flatnonzero(grid >= est))):
        start, prev_v, prev_ll, bound = fit_.par, est, fit_.loglik, None
        for i in idx:
            f = refit(grid[i], start)
            ll[i], start = f.loglik, f.par
            if bound is None and fit_.loglik - f.loglik > crit:
                # interpolate on the root of the drop, which is close to
                # linear in the parameter (exactly so for a quadratic profile)
                r0, r1 = np.sqrt(max(fit_.loglik - prev_ll, 0.0)), np.sqrt(fit_.loglik - f.loglik)
                bound = prev_v + (np.sqrt(crit) - r0) / (r1 - r0) * (grid[i] - prev_v)
            prev_v, prev_ll = grid[i], f.loglik
        ends[f"open_{side}"] = bound is None
        ends[side] = (grid[0] if side == "low" else grid[-1]) if bound is None else bound
        pars[side] = refit(ends[side], fit_.par).par
    return dict(grid=grid, loglik=ll, low=ends["low"], high=ends["high"], open_low=ends["open_low"],
                open_high=ends["open_high"], par_low=pars["low"], par_high=pars["high"], max=fit_.loglik)


def lr_test(full: Fit, restricted: Fit):
    stat = 2 * (full.loglik - restricted.loglik)
    df = full.k - restricted.k
    return stat, df, stats.chi2.sf(max(stat, 0), df)


# ---------------------------------------------------------------------------
# Estimands


def prevented(par, ev: Events, by=None):
    """Crimes prevented by the arrests that were made.

    direct    first-generation offspring removed, sum over arrests of
              theta_i (1 - kappa_i) [S(c_i) - S(T - t_i)]
    n0        branching ratio with no arrest effect (mean theta_i)
    n_actual  branching ratio given the arrests made
    total     direct / (1 - n0): adds the descendants the removed offspring
              would have had
    """
    alpha, gamma, beta, kpar, sigma = unpack(par, ev)
    theta, kappa = np.exp(ev.X @ gamma), np.exp(ev.Z @ beta)
    Sc, ST = ev.kern.surv(ev.c, kpar)[0], ev.kern.surv(ev.T - ev.t, kpar)[0]
    D = np.where(ev.arrested, theta * (1 - kappa) * (Sc - ST), 0.0)
    n0 = theta.mean()
    out = dict(direct=D.sum(), n0=n0, n_actual=n0 - D.mean(), total=D.sum() / (1 - n0),
               per_arrest=D.sum() / (1 - n0) / max(ev.arrested.sum(), 1),
               share=D.sum() / (1 - n0) / ev.n)
    if by is not None:
        for key in np.unique(by[ev.arrested]):
            out[f"total[{key}]"] = D[ev.arrested & (by == key)].sum() / (1 - n0)
    return out


def process_sd(direct, n0):
    """Standard deviation of the realized number of crimes prevented around
    its expectation. The removed first-generation crimes are Poisson with
    mean ``direct``; each would have started a branching family whose size
    has mean 1 / (1 - n0) and variance n0 / (1 - n0)^3.
    """
    second_moment = n0 / (1 - n0) ** 3 + 1 / (1 - n0) ** 2
    return np.sqrt(np.abs(direct) * second_moment)


def prediction_interval(est, low, high, direct, n0, z=1.959964):
    """Interval for the realized number of crimes prevented: the interval
    for its expectation (``low``, ``high``) widened by the process noise,
    combining the two half-widths on each side in quadrature."""
    s = z * process_sd(direct, n0)
    return est - np.sqrt((est - low) ** 2 + s**2), est + np.sqrt((high - est) ** 2 + s**2)


def prevented_curve(par, ev: Events, delays, x=None, z=None):
    """Crimes prevented by one arrest made ``delays`` days after the crime
    (descendants included), for covariate rows ``x`` and ``z`` (default: the
    average event and an intercept-only arrest effect)."""
    alpha, gamma, beta, kpar, sigma = unpack(par, ev)
    theta_all = np.exp(ev.X @ gamma)
    theta = theta_all.mean() if x is None else np.exp(np.asarray(x) @ gamma)
    zz = np.r_[1.0, np.zeros(ev.Z.shape[1] - 1)] if z is None else np.asarray(z)
    kappa = np.exp(zz @ beta)
    return theta * (1 - kappa) * ev.kern.surv(np.asarray(delays, float), kpar)[0] / (1 - theta_all.mean())


def draws(fit_: Fit, func, n=2000, seed=0):
    """Evaluate ``func(par)`` at draws from the asymptotic normal of the MLE."""
    rng = np.random.default_rng(seed)
    free = np.array([i for i, nm in enumerate(fit_.names) if nm not in fit_.fixed])
    cov = fit_.cov[np.ix_(free, free)]
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    L = vecs * np.sqrt(np.clip(vals, 0, None))
    out = []
    for _ in range(n):
        p = fit_.par.copy()
        p[free] += L @ rng.standard_normal(len(free))
        out.append(func(p))
    return out


def interval(vals, level=0.95):
    lo, hi = np.percentile(np.asarray(vals, float), [50 * (1 - level), 100 - 50 * (1 - level)], axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# Diagnostics


def triggered_prob(par, ev: Events, static=False):
    """Probability each event was triggered by an earlier one (vs background)."""
    p = _parts(par, ev, static)
    return p["trig"] / p["lam"]


def compensator(par, ev: Events, cells=3e6):
    """Citywide integrated intensity at each event time (time-rescaling residuals).

    Under a correct model the differences are independent Exponential(1).
    Sums over every earlier event, so the cost is quadratic in n.
    """
    alpha, gamma, beta, kpar, sigma = unpack(par, ev)
    theta, kappa = np.exp(ev.X @ gamma), np.exp(ev.Z @ beta)
    edges = np.r_[0, np.cumsum(ev.widths)]
    cum_bg = np.r_[0, np.cumsum(np.exp(alpha) * ev.widths)]
    bg = np.interp(ev.t, edges, cum_bg)
    Sc = ev.kern.surv(ev.c, kpar)[0]
    out = np.empty(ev.n)
    # Rows a:e are compared with columns 0:e, so a chunk holds (e - a) * e
    # cells. A fixed number of rows keeps that under `cells` for every chunk.
    rows = max(1, int(cells // ev.n))
    a = 0
    while a < ev.n:
        e = min(ev.n, a + rows)
        u = ev.t[a:e, None] - ev.t[None, :e]
        live = u > 0
        Su = ev.kern.surv(np.where(live, u, 0.0), kpar)[0]
        before = u <= ev.c[None, :e]  # evaluation time precedes the parent's arrest
        val = np.where(before, 1 - Su, (1 - Sc[None, :e]) + kappa[None, :e] * (Sc[None, :e] - Su))
        out[a:e] = (val * live) @ theta[:e]
        a = e
    return bg + out
