"""Regenerate holiday figure (F5) with averted-attack-rate convention.
Standalone: reads policy_posterior_seasonpop.json + figstyle only (no NUTS/JAX).
In-image caption removed (moved to LaTeX); taller figure for a larger plot."""
import os, json, sys
from pathlib import Path
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_seasonpop"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from figstyle import savefig, is_sig, zero_line, COL_SEASON, COL_ZERO, SEASONS, W_DOUBLE

POL = json.load(open(REPO / "outputs/eda/policy_posterior_seasonpop.json"))

def _key(pol_s, win):
    if win == "term":
        return pol_s.get("sick_d_by_age_term") or pol_s.get("sick_by_age", {})
    return pol_s.get("sick_d_by_age_vac", {})

fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.95, 4.2), constrained_layout=True)
child_ages = ["0-5", "6-11", "12-17"]
xs = np.arange(len(child_ages))
dx = 0.10
for j, s in enumerate(SEASONS):
    col = COL_SEASON[s]
    base_off = (j - 1) * 0.28
    for wi, (win, mk) in enumerate([("term", "o"), ("vac", "s")]):
        src = _key(POL[s], win)
        offs = base_off + (wi - 0.5) * dx
        for i, ag in enumerate(child_ages):
            v = src.get(ag, {})
            if not v: continue
            sig = is_sig(v["q05"], v["q95"])
            mean_av = -v["mean"]                       # averted = -Δattack
            yerr = [[v["q95"] - v["mean"]], [v["mean"] - v["q05"]]]
            ax.errorbar([xs[i] + offs], [mean_av], yerr=yerr,
                        color=col, lw=1.0, capsize=2, zorder=3,
                        marker=mk, markersize=6,
                        markerfacecolor=(col if sig else "white"),
                        markeredgecolor=col, markeredgewidth=1.2)
    ax.plot([], [], color=col, marker="o", ls="", label=s, markersize=5)
win_handles = [
    Line2D([0], [0], marker="o", color=COL_ZERO, ls="", markersize=6,
           label="School term (days 70–113)"),
    Line2D([0], [0], marker="s", color=COL_ZERO, ls="", markersize=6,
           label="Winter break (days 113–183)"),
]
zero_line(ax)
ax.set_xticks(xs); ax.set_xticklabels(child_ages)
ax.set_xlabel("Age group (children)")
ax.set_ylabel(r"Averted attack rate (%p), sick leave")
leg1 = ax.legend(title="Season", frameon=False, fontsize=7.5, title_fontsize=8,
                 loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.add_artist(leg1)
ax.legend(handles=win_handles, frameon=False, fontsize=7.5,
          loc="lower left", bbox_to_anchor=(1.01, 0.0))
savefig(fig, "holiday")
print("regenerated holiday (averted convention, no in-image caption)")
