"""Main-text by-age figure (2x1, stacked): averted effect in rate (%p) and number (counts).

Sick leave shown for all three seasons (season-coloured); school absence shown as the
3-season mean with a min--max band (school is consistent across seasons).

Run:  FIG_OUTDIR_SUFFIX=_seasonpop python scripts/fig_byage_2metric_main.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs

REPO = Path(__file__).resolve().parent.parent
D = json.load(open(REPO / "outputs" / "eda" / "policy_posterior_seasonpop.json"))
ages = fs.AGES
seasons = fs.SEASONS


def val(s, prefix, kind, a):
    key = f"{prefix}_d_by_age_term" if kind == "rate" else f"{prefix}_num_by_age_term"
    v = D[s][key][a]
    if kind == "rate":
        return -v["mean"], -v["q95"], -v["q05"], fs.is_sig(v["q05"], v["q95"])
    return v["mean"], v["q05"], v["q95"], fs.is_sig(v["q05"], v["q95"])


def plot_panel(ax, kind, ylabel):
    x = np.arange(len(ages))
    # school absence: 3-season mean line + band = envelope of the per-season 90% CIs
    sv = [[val(s, "school", kind, a) for a in ages] for s in seasons]  # [season][age] -> (m,lo,hi,sig)
    smean = np.array([[sv[i][j][0] for j in range(len(ages))] for i in range(len(seasons))]).mean(0)
    sband_lo = np.array([[sv[i][j][1] for j in range(len(ages))] for i in range(len(seasons))]).min(0)
    sband_hi = np.array([[sv[i][j][2] for j in range(len(ages))] for i in range(len(seasons))]).max(0)
    ax.fill_between(x, sband_lo, sband_hi, color=fs.COL_SCHOOL, alpha=0.15, zorder=1)
    ax.plot(x, smean, "-s", color=fs.COL_SCHOOL, markersize=5, lw=1.6, zorder=3)
    # sick leave: three seasons, dodged
    dodges = [-0.24, 0.0, 0.24]
    for s, dx in zip(seasons, dodges):
        col = fs.COL_SEASON[s]
        m = np.array([val(s, "sick", kind, a) for a in ages], dtype=object)
        mm = np.array([r[0] for r in m]); lo = np.array([r[1] for r in m])
        hi = np.array([r[2] for r in m]); sg = [r[3] for r in m]
        ax.errorbar(x + dx, mm, yerr=np.vstack([mm - lo, hi - mm]), fmt="none",
                    ecolor=col, elinewidth=1.0, capsize=1.5, zorder=2)
        for xi, mi, si in zip(x + dx, mm, sg):
            ax.plot(xi, mi, linestyle="none", zorder=4, **fs.marker_style(si, col))
    fs.zero_line(ax, "h")
    ax.set_xticks(x); ax.set_xticklabels(ages)
    ax.set_xlim(-0.5, len(ages) - 0.5)
    ax.set_ylabel(ylabel)


fig, (axA, axB) = plt.subplots(2, 1, figsize=(fs.W_DOUBLE, 6.4), sharex=True)
plot_panel(axA, "rate", "Averted attack rate (%p)")
plot_panel(axB, "num", "Averted infections (n)")
fs.panel_label(axA, "A", x=-0.07); fs.panel_label(axB, "B", x=-0.07)
axB.set_xlabel("Age group")

handles = [Line2D([0], [0], color=fs.COL_SEASON[s], marker="o", lw=0, markersize=6,
                  label=f"Sick leave, {s}") for s in seasons]
handles.append(Line2D([0], [0], color=fs.COL_SCHOOL, marker="s", lw=1.6, markersize=6,
                      label="School absence (3-season mean, band = 90% CI envelope)"))
axA.legend(handles=handles, loc="upper right", frameon=False, fontsize=7)
fig.tight_layout()
pdf, png = fs.savefig(fig, "byage_2metric_main")
print("saved:", pdf, png)
