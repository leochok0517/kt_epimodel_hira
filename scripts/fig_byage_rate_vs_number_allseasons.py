"""Supplement: by-age averted effect in rate (%p) vs number (counts), ALL three seasons.

3 rows (seasons) x 2 cols (rate | number).  Complements the main-text single-season figure.

Run:  FIG_OUTDIR_SUFFIX=_seasonpop python scripts/fig_byage_rate_vs_number_allseasons.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "outputs" / "eda" / "policy_posterior_seasonpop.json"
D = json.load(open(DATA))
ages = fs.AGES
seasons = fs.SEASONS


def series(s, prefix, kind):
    key = f"{prefix}_d_by_age_term" if kind == "rate" else f"{prefix}_num_by_age_term"
    mean, lo, hi, sig = [], [], [], []
    for a in ages:
        v = D[s][key][a]
        if kind == "rate":
            m, l, h = -v["mean"], -v["q95"], -v["q05"]
        else:
            m, l, h = v["mean"], v["q05"], v["q95"]
        mean.append(m); lo.append(l); hi.append(h)
        sig.append(fs.is_sig(v["q05"], v["q95"]))
    return np.array(mean), np.array(lo), np.array(hi), sig


def plot_panel(ax, s, kind):
    x = np.arange(len(ages))
    for prefix, col, dx in [("sick", fs.COL_SICK, -0.14), ("school", fs.COL_SCHOOL, +0.14)]:
        m, lo, hi, sig = series(s, prefix, kind)
        ax.errorbar(x + dx, m, yerr=np.vstack([m - lo, hi - m]), fmt="none",
                    ecolor=col, elinewidth=1.1, capsize=2, zorder=2)
        for xi, mi, si in zip(x + dx, m, sig):
            ax.plot(xi, mi, linestyle="none", zorder=3, **fs.marker_style(si, col))
    fs.zero_line(ax, "h")
    ax.set_xticks(x); ax.set_xticklabels(ages, rotation=0)
    ax.set_xlim(-0.5, len(ages) - 0.5)


fig, axes = plt.subplots(3, 2, figsize=(fs.W_DOUBLE, 7.4), sharex=True)
for r, s in enumerate(seasons):
    plot_panel(axes[r, 0], s, "rate")
    plot_panel(axes[r, 1], s, "num")
    axes[r, 0].set_ylabel(f"{s}\nAverted attack rate (%p)")
    axes[r, 1].set_ylabel("Averted infections (n)")
axes[0, 0].set_title("Per-capita (rate)")
axes[0, 1].set_title("Absolute (number)")
for ax in axes[2, :]:
    ax.set_xlabel("Age group")

handles = [
    Line2D([0], [0], color=fs.COL_SICK, marker="o", lw=0, markersize=6, label="Sick leave"),
    Line2D([0], [0], color=fs.COL_SCHOOL, marker="o", lw=0, markersize=6, label="School absence"),
]
axes[0, 1].legend(handles=handles, loc="upper right", frameon=False)
fig.text(0.5, -0.02,
         "Positive = averted. Filled = $0\\notin$CI, open = $0\\in$CI. "
         "Per-capita, sick leave lowers adults and raises children; in counts, adult bands dominate.",
         ha="center", fontsize=7.5, color="0.35")
fig.tight_layout()
pdf, png = fs.savefig(fig, "byage_rate_vs_number_allseasons")
print("saved:", pdf, png)
