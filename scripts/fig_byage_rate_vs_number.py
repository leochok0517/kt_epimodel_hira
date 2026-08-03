"""Unified by-age figure: averted effect in rate (percentage points) vs number (counts).

Replaces the separate school_vs_sick_number and policy_posterior_byage figures, and
the sick-leave-only rate_vs_number figure, with one two-panel figure showing both
policies in both metrics for the representative 2019-20 season.

Run:  FIG_OUTDIR_SUFFIX=_seasonpop python scripts/fig_byage_rate_vs_number.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "outputs" / "eda" / "policy_posterior_seasonpop.json"
SEASON = "2019-2020"

d = json.load(open(DATA))[SEASON]
ages = fs.AGES


def series(prefix, kind):
    """kind='rate' -> averted attack rate (%p, = -Δattack); 'num' -> averted infections."""
    key = f"{prefix}_d_by_age_term" if kind == "rate" else f"{prefix}_num_by_age_term"
    mean, lo, hi, sig = [], [], [], []
    for a in ages:
        v = d[key][a]
        if kind == "rate":  # averted rate = -Δattack ; flip CI accordingly
            m, l, h = -v["mean"], -v["q95"], -v["q05"]
        else:
            m, l, h = v["mean"], v["q05"], v["q95"]
        mean.append(m); lo.append(l); hi.append(h)
        sig.append(fs.is_sig(v["q05"], v["q95"]))
    return np.array(mean), np.array(lo), np.array(hi), sig


def plot_panel(ax, kind, ylabel):
    x = np.arange(len(ages))
    for prefix, col, dx in [("sick", fs.COL_SICK, -0.14), ("school", fs.COL_SCHOOL, +0.14)]:
        m, lo, hi, sig = series(prefix, kind)
        yerr = np.vstack([m - lo, hi - m])
        ax.errorbar(x + dx, m, yerr=yerr, fmt="none", ecolor=col, elinewidth=1.2,
                    capsize=2, zorder=2)
        for xi, mi, si in zip(x + dx, m, sig):
            ax.plot(xi, mi, linestyle="none", zorder=3, **fs.marker_style(si, col))
    fs.zero_line(ax, "h")
    ax.set_xticks(x); ax.set_xticklabels(ages, rotation=0)
    ax.set_xlabel("Age group")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(ages) - 0.5)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(fs.W_DOUBLE, 3.1))
plot_panel(axA, "rate", "Averted attack rate (%p)")
plot_panel(axB, "num", "Averted infections (n)")
fs.panel_label(axA, "A"); fs.panel_label(axB, "B")

# single legend (filled = 0∉CI, open = 0∈CI)
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=fs.COL_SICK, marker="o", lw=0, markersize=6, label="Sick leave"),
    Line2D([0], [0], color=fs.COL_SCHOOL, marker="o", lw=0, markersize=6, label="School absence"),
]
axA.legend(handles=handles, loc="upper right", frameon=False)

fig.suptitle("")
fig.text(0.5, -0.06,
         "Sick leave and school absence, 2019-20. Positive = averted. "
         "Filled = $0\\notin$CI, open = $0\\in$CI.",
         ha="center", fontsize=7.5, color="0.35")
fig.tight_layout()
pdf, png = fs.savefig(fig, "byage_rate_vs_number")
print("saved:", pdf, png)
