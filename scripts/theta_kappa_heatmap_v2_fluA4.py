"""Figure theta_kappa_heatmap_v2 for fluA4 (4 seasons).

Sick leave's net averted infection over theta (rows) x kappa (columns),
school-term window, one panel per season. The theta=0 row (primary
analysis) is boxed; cells whose 90% CI includes zero are hatched.

Reads outputs/fluA/sens_theta_kappa_grid_fluA4.json
(sens_theta_kappa_grid_fluA4.py).
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import savefig, COL_ZERO, W_DOUBLE, panel_label

REPO = Path(__file__).resolve().parent.parent
COL_SEASON_FLUA4 = {"2016-2017": "#009E73", "2017-2018": "#CC79A7",
                     "2018-2019": "#D55E00", "2019-2020": "#56B4E9"}

d = json.load(open(REPO / "outputs/fluA/sens_theta_kappa_grid_fluA4.json"))
SEASONS = d["seasons"]
THETA = d["theta_grid"]
KAPPA = d["kappa_grid"]
i_theta0 = THETA.index(0.0)

fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 8.6), constrained_layout=True)
axes = axes.flatten()

all_means = []
for s in SEASONS:
    for k in KAPPA:
        for r in d["results"][s][f"kappa_{k}"]:
            all_means.append(r["mean"])
vmax = max(abs(min(all_means)), abs(max(all_means)))
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.get_cmap("RdBu_r")

im = None
for panel_i, s in enumerate(SEASONS):
    ax = axes[panel_i]
    M = np.zeros((len(THETA), len(KAPPA)))
    ci0 = np.zeros((len(THETA), len(KAPPA)), dtype=bool)
    for j, k in enumerate(KAPPA):
        rows = d["results"][s][f"kappa_{k}"]
        for i, r in enumerate(rows):
            M[i, j] = r["mean"]
            ci0[i, j] = r["q05"] * r["q95"] <= 0
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    # hatching on CI-includes-zero cells
    for i in range(len(THETA)):
        for j in range(len(KAPPA)):
            if ci0[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                         hatch="//////", edgecolor="black",
                                         linewidth=0))
    # box the theta=0 row (primary analysis)
    ax.add_patch(Rectangle((-0.5, i_theta0 - 0.5), len(KAPPA), 1, fill=False,
                             edgecolor="black", linewidth=1.8, zorder=5))
    ax.set_xticks(range(len(KAPPA))); ax.set_xticklabels([f"{k:.2f}" for k in KAPPA], fontsize=7)
    ax.set_yticks(range(0, len(THETA), 4))
    ax.set_yticklabels([f"{THETA[i]:+.2f}" for i in range(0, len(THETA), 4)], fontsize=7)
    ax.set_xlabel(r"$\kappa$ (adult)")
    ax.set_ylabel(r"$\theta$")
    ax.set_title(s, fontsize=9, fontweight="bold", color=COL_SEASON_FLUA4[s])
    panel_label(ax, "ABCD"[panel_i])

cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, fraction=0.025, pad=0.01)
cbar.set_label("Sick-leave averted (%), term window", fontsize=8)
fig.text(0.5, -0.01,
          "Boxed row = theta=0 (primary analysis). Hatched cells: 90% CI includes zero.",
          ha="center", fontsize=7, color=COL_ZERO)
savefig(fig, "theta_kappa_heatmap_v2")
print("saved theta_kappa_heatmap_v2 (fluA4)")
