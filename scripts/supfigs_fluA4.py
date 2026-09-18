"""Supplement figures that depend only on already-computed fluA4 JSON.

Regenerates, for the four influenza-A seasons:
  sickleave_term_vs_vacation        policy_posterior_fluA4.json
  byage_rate_vs_number_allseasons   policy_posterior_fluA4.json
  age_redistribution_v2             sens_theta_fluA4.json
  effect_ratio_v2                   sens_theta_fluA4.json

No simulation and no re-fitting: every value is read from the stored posterior
summaries, so this runs anywhere numpy+matplotlib do.
"""
import json, os, sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ["HOME"]) / "mnt" / "kt_epimodel_hira"
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

OUT = REPO / "outputs" / "fluA"
P = json.load(open(OUT / "policy_posterior_fluA4.json"))
T = json.load(open(OUT / "sens_theta_fluA4.json"))
SEASONS = P["seasons"]
AGES = fs.AGES
# figstyle's COL_SEASON predates 2018-19; extend without touching the module.
CS = dict(fs.COL_SEASON)
CS.setdefault("2018-2019", "#D55E00")

def sig(d):
    return d["q05"] * d["q95"] > 0


# ---------------------------------------------------------------- figure 1
def fig_term_vs_vac():
    """Sick-leave d-attack by age, school term vs winter break, 4 seasons."""
    fig, axes = plt.subplots(1, 4, figsize=(fs.W_DOUBLE, 2.6), sharey=True,
                             constrained_layout=True)
    x = np.arange(len(AGES))
    for ax, s in zip(axes, SEASONS):
        for win, off, mk, lab in (("term", -0.16, "o", "school term"),
                                  ("vac", +0.16, "s", "winter break")):
            m = [P[s][f"sick_d_by_age_{win}"][a]["mean"] for a in AGES]
            lo = [P[s][f"sick_d_by_age_{win}"][a]["q05"] for a in AGES]
            hi = [P[s][f"sick_d_by_age_{win}"][a]["q95"] for a in AGES]
            sg = [sig(P[s][f"sick_d_by_age_{win}"][a]) for a in AGES]
            c = CS[s]
            ax.errorbar(x + off, m, yerr=[np.array(m) - lo, np.array(hi) - m],
                        fmt="none", ecolor=c, elinewidth=0.9, capsize=2, alpha=0.8)
            for xi, mi, si in zip(x + off, m, sg):
                ax.plot(xi, mi, marker=mk, ms=5,
                        mfc=c if si else "white", mec=c, mew=1.1, ls="none")
        ax.axhline(0, color=fs.COL_ZERO, lw=0.8, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(AGES, rotation=45, ha="right")
        ax.set_title(f"{s[:4]}–{s[7:9]}")
    axes[0].set_ylabel(r"$\Delta$attack under sick leave (%p)")
    axes[0].legend(handles=[Line2D([], [], marker="o", color=fs.COL_ZERO, ls="none",
                                   ms=5, label="school term"),
                            Line2D([], [], marker="s", color=fs.COL_ZERO, ls="none",
                                   ms=5, label="winter break")],
                   loc="lower left", frameon=False)
    fig.suptitle("Positive = attack rate rises (risk transferred to that band); "
                 "filled = 90% CI excludes 0", fontsize=8, y=1.06)
    return fs.savefig(fig, "sickleave_term_vs_vacation")


# ---------------------------------------------------------------- figure 2
def fig_rate_vs_number():
    """By-age averted effect in rate and in counts, both policies, 4 seasons."""
    fig, axes = plt.subplots(2, 2, figsize=(fs.W_DOUBLE, 5.0),
                             constrained_layout=True)
    x = np.arange(len(AGES))
    for col, (pol, colr, nm) in enumerate(
            ((["sick"], fs.COL_SICK, "Sick leave"),
             (["school"], fs.COL_SCHOOL, "School absence"))):
        pol = pol[0]
        for row, (key, ylab, scale) in enumerate(
                ((f"{pol}_d_by_age_term", "averted attack rate (%p)", -1.0),
                 (f"{pol}_num_by_age_term", "averted infections (thousands)", 1e-3))):
            ax = axes[row][col]
            for j, s in enumerate(SEASONS):
                off = (j - 1.5) * 0.16
                d = P[s][key]
                m = np.array([d[a]["mean"] for a in AGES]) * scale
                lo = np.array([d[a]["q05"] for a in AGES]) * scale
                hi = np.array([d[a]["q95"] for a in AGES]) * scale
                if scale < 0:
                    lo, hi = hi, lo
                sg = [sig(d[a]) for a in AGES]
                c = CS[s]
                ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="none",
                            ecolor=c, elinewidth=0.8, capsize=1.8, alpha=0.8)
                for xi, mi, si in zip(x + off, m, sg):
                    ax.plot(xi, mi, "o", ms=4, mfc=c if si else "white",
                            mec=c, mew=1.0, ls="none")
            ax.axhline(0, color=fs.COL_ZERO, lw=0.8, zorder=0)
            ax.set_xticks(x); ax.set_xticklabels(AGES, rotation=45, ha="right")
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(nm, color=colr)
    fs.panel_label(axes[0][0], "A"); fs.panel_label(axes[1][0], "B")
    axes[0][1].legend(handles=[Line2D([], [], marker="o", color=CS[s], ls="none",
                                      ms=4, label=f"{s[:4]}–{s[7:9]}")
                               for s in SEASONS],
                      loc="lower right", frameon=False, ncol=2)
    fig.suptitle("School-term window. Positive = infection averted; "
                 "filled = 90% CI excludes 0", fontsize=8, y=1.03)
    return fs.savefig(fig, "byage_rate_vs_number_allseasons")


# ---------------------------------------------------------------- figure 3
def fig_age_redistribution():
    """Child d-attack against theta, by season and window."""
    kids = ["0-5", "6-11", "12-17"]
    g = T["theta_grid"]
    fig, axes = plt.subplots(2, 4, figsize=(fs.W_DOUBLE, 4.2), sharex=True,
                             sharey="row", constrained_layout=True)
    styles = {"0-5": ":", "6-11": "--", "12-17": "-"}
    for r, (win, wlab) in enumerate((("sick_term", "school term"),
                                     ("sick_vac", "winter break"))):
        for c, s in enumerate(SEASONS):
            ax = axes[r][c]
            for a in kids:
                arr = T["results"][s][win]["age_grid"][a]
                m = np.array([v["mean"] for v in arr])
                lo = np.array([v["q05"] for v in arr])
                hi = np.array([v["q95"] for v in arr])
                ax.plot(g, m, styles[a], color=CS[s], lw=1.3, label=a)
                if a == "12-17":
                    ax.fill_between(g, lo, hi, color=CS[s], alpha=0.13, lw=0)
            ax.axhline(0, color=fs.COL_ZERO, lw=0.8)
            ax.axvline(0, color=fs.COL_ZERO, lw=0.8, ls=":")
            if r == 0:
                ax.set_title(f"{s[:4]}–{s[7:9]}")
            if c == 0:
                ax.set_ylabel(f"{wlab}\n" + r"child $\Delta$attack (%p)")
            if r == 1:
                ax.set_xlabel(r"$\theta$")
    axes[0][0].legend(frameon=False, loc="upper right", fontsize=7)
    fig.suptitle("Positive = risk transferred to children. Band: 90% CI for 12–17. "
                 "Primary analysis is $\\theta=0$ (dotted line).", fontsize=8, y=1.04)
    return fs.savefig(fig, "age_redistribution_v2")


# ---------------------------------------------------------------- figure 4
def fig_effect_ratio():
    """School-absence / sick-leave averted-infection ratio against theta."""
    g = np.array(T["theta_grid"])
    fig, ax = plt.subplots(figsize=(fs.W_SINGLE * 1.4, 2.8), constrained_layout=True)
    for s in SEASONS:
        sk = np.array([v["mean"] for v in T["results"][s]["sick_term"]["total_grid"]])
        sc = np.array([v["mean"] for v in T["results"][s]["student_term"]["total_grid"]])
        r = np.where(sk > 0, sc / np.where(sk == 0, np.nan, sk), np.nan)
        ax.plot(g, r, "-", color=CS[s], lw=1.5, label=f"{s[:4]}–{s[7:9]}")
    ax.axhline(1, color=fs.COL_ZERO, lw=0.8)
    ax.axvline(0, color=fs.COL_ZERO, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("school absence / sick leave\n(averted infection, school term)")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Ratio exceeds 1 wherever sick leave's mean is positive", fontsize=8)
    return fs.savefig(fig, "effect_ratio_v2")


if __name__ == "__main__":
    for f in (fig_term_vs_vac, fig_rate_vs_number,
              fig_age_redistribution, fig_effect_ratio):
        print(" ", f.__name__, "->", f()[0])
