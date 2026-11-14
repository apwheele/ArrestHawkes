"""Figure helpers for the paper (same theme as apwheele/WDDControl)."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

THEME = {
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "#d9d8d4",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 1,
    "legend.facecolor": "white",
    "legend.shadow": True,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.titlelocation": "left",
    "figure.dpi": 150,
    "lines.linewidth": 1.8,
}

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#8a8985"
FAINT = "#e4e3df"
# Categorical slots in fixed order: blue, orange, aqua.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

LAG_TICKS = [1 / 24, 0.25, 1, 3, 7, 14, 30]
LAG_LABELS = ["1 hour", "6 hours", "1 day", "3 days", "1 week", "2 weeks", "30 days"]


def use_theme():
    matplotlib.rcParams.update(THEME)


def schematic(ax, omega=0.5, theta=0.4, kappa=0.3, delay=1.5, horizon=8.0):
    """One crime's triggering rate over time with and without an arrest."""
    u = np.linspace(0, horizon, 400)
    full = theta * omega * np.exp(-omega * u)
    damp = np.where(u < delay, full, kappa * full)
    ax.plot(u, full, color=MUTED, ls="--", lw=1.4, label="No arrest")
    ax.plot(u, damp, color=BLUE, label="Arrest at day 1.5")
    after = u >= delay
    ax.fill_between(u[after], damp[after], full[after], color=BLUE, alpha=0.18, lw=0)
    ax.axvline(delay, color=INK, lw=0.8, ls=":")
    ax.annotate("arrest", (delay, full[0] * 0.93), xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                color=SECONDARY, va="top")
    ax.annotate("follow-on crimes\nprevented", (delay + 1.1, theta * omega * np.exp(-omega * (delay + 1.1)) * 0.62),
                xytext=(delay + 2.6, full[0] * 0.55), fontsize=8.5, color=SECONDARY,
                arrowprops=dict(arrowstyle="-", color=SECONDARY, lw=0.7))
    ax.set_xlabel("Days since the crime")
    ax.set_ylabel("Follow-on crimes triggered per day")
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")


def kernel_survival(summary, u):
    """Share of a crime's triggering still to come after u days, from a results summary."""
    kp, u = summary["kernel_par"], np.asarray(u, float)
    if summary["kernel"] == "exp":
        return np.exp(-kp["omega"] * u)
    return (1 + u / kp["c"]) ** (-(kp["p"] - 1))


def followon_panel(ax, tab, title, summary=None, radius=200.0):
    """Later crimes within 200 m per earlier crime per day, by lag bin, for
    crimes with a fast arrest and crimes with none. With ``summary``, dashed
    lines add what the fitted model implies: each group's long-run rate (its
    last bin) plus the direct triggering expected in the bin, multiplied by
    kappa for the arrested group.
    """
    fmt = lambda d: f"{d * 24:.0f} h" if d < 1 else f"{d:.0f} d"
    for lab, color in [("no arrest", MUTED), ("arrest within a day", BLUE)]:
        g = tab[tab["group"] == lab].sort_values("lag_from")
        x = np.arange(len(g))
        ax.plot(x, g["rate"], color=color, marker="o", ms=5, mec="white", mew=1, label=lab.capitalize())
        if summary is not None:
            lo, hi = g["lag_from"].to_numpy(), g["lag_to"].to_numpy()
            mult = summary["kappa"]["exp"] if lab != "no arrest" else 1.0
            inside = 1 - np.exp(-radius**2 / (2 * summary["sigma"]["exp"] ** 2))
            excess = (mult * summary["prevented"]["n0"] * inside
                      * (kernel_survival(summary, lo) - kernel_survival(summary, hi)) / (hi - lo))
            base = g["rate"].to_numpy()[-1] - excess[-1]
            ax.plot(x, base + excess, color=color, ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fmt(a).split()[0]}-{fmt(b)}" if fmt(a).split()[1] == fmt(b).split()[1]
                        else f"{fmt(a)}-{fmt(b)}" for a, b in zip(g["lag_from"], g["lag_to"])], rotation=30, ha="right")
    ax.set_xlabel("Time since the earlier crime")
    ax.set_title(title)
    ax.set_ylim(0, None)


def curve_panel(ax, curve, title, color=BLUE):
    c = curve[curve["delay"] > 0]
    ax.fill_between(c["delay"], c["low"], c["high"], color=color, alpha=0.18, lw=0)
    ax.plot(c["delay"], c["prevented"], color=color)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(LAG_TICKS)
    ax.set_xticklabels(LAG_LABELS, rotation=30, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("Time from crime to arrest")
    ax.set_title(title)


def profile_panel(ax, summary, title, color=BLUE):
    pr = summary["profile"]
    grid, ll = np.array(pr["grid"]), np.array(pr["loglik"])
    est = summary["kappa"]["coef"]
    x = np.r_[grid, est]
    y = np.r_[2 * (pr["max"] - ll), 0.0]
    o = np.argsort(x)
    ax.plot(np.exp(x[o]), y[o], color=color)
    ax.axhline(3.84, color=INK, lw=0.8, ls=":")
    ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.scatter([np.exp(est)], [0], color=color, s=28, zorder=3, edgecolor="white", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.1, 1, 4])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "4"])
    ax.minorticks_off()
    ax.set_ylim(0, 12)
    ax.set_xlabel(r"Arrest multiplier $\kappa$")
    ax.set_title(title)


def sim_strip(ax, df, col, truth, methods, labels, colors, ylabel, log=False, seed=0):
    """Jittered points per method with the mean as a tick, truth as a line."""
    rng = np.random.default_rng(seed)
    for i, (m, c) in enumerate(zip(methods, colors)):
        v = df.loc[df["method"] == m, col].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=c, alpha=0.45, linewidth=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=1.6)
    if truth is not None:
        ax.axhline(truth, color=INK, lw=0.8, ls=":")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
