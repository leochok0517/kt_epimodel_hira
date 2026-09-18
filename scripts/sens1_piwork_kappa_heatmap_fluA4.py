"""T2 figure -- pi_work x kappa heatmap for fluA4 (4 seasons).

Copy of sens_phaseA_figs.py's fig_S1, adapted for outputs/fluA/
sens_piwork_kappa_fluA4.json (4 seasons instead of 3) and saved through
figstyle.savefig into figures/paper_fluA4/ (flat layout, matching the rest
of the fluA4 figure set) rather than the original figures/v4.../sensitivity/
nested layout, so the manuscript's flat \\graphicspath can find it. Filename
kept identical ("sens_S1_piwork_kappa_heatmap") per prompts/10's convention.
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import savefig, COL_ZERO, W_DOUBLE, panel_label

REPO = Path(__file__).resolve().parent.parent
SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
COL_SEASON_FLUA4 = {"2016-2017": "#009E73", "2017-2018": "#CC79A7",
                     "2018-2019": "#E69F00", "2019-2020": "#56B4E9"}

d = json.load(open(REPO / "outputs/fluA/sens_piwork_kappa_fluA4.json"))
PIW = d["meta"]["pi_work_grid"]; KAP = d["meta"]["kappa_grid"]
grid = defaultdict(dict)
for r in d["rows"]:
    grid[(r["pi_work"], r["kappa"])][r["season"]] = r

fig, axes = plt.subplots(1, 5, figsize=(W_DOUBLE * 1.7, 3.6),
                          constrained_layout=True,
                          gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.05]))
vmax = 3.0
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.get_cmap("RdBu_r")

im = None
for k, s in enumerate(SEASONS):
    ax = axes[k]
    M = np.zeros((len(PIW), len(KAP)))
    for i, pw in enumerate(PIW):
        for j, ka in enumerate(KAP):
            M[i, j] = grid[(pw, ka)][s]["averted_sick_term"]
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(KAP)))
    ax.set_xticklabels([f"{k:.2f}" for k in KAP], fontsize=6.5, rotation=45)
    ax.set_yticks(range(len(PIW)))
    ax.set_yticklabels([f"{p:.2f}" for p in PIW], fontsize=6.5)
    if k == 0:
        ax.set_ylabel(r"$\pi_{\mathrm{work}}$")
    ax.set_xlabel(r"$\kappa$ (adult)")
    for i in range(len(PIW)):
        for j in range(len(KAP)):
            col = "white" if abs(M[i, j]) > vmax * 0.55 else "black"
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                    fontsize=5.5, color=col)
    panel_label(ax, "ABCD"[k])
    ax.set_title(s, fontsize=8.5, pad=3, color=COL_SEASON_FLUA4[s])

ax = axes[4]
sign_map = np.zeros((len(PIW), len(KAP)), dtype=int)
for i, pw in enumerate(PIW):
    for j, ka in enumerate(KAP):
        signs = [np.sign(grid[(pw, ka)][s]["averted_sick_term"]) for s in SEASONS]
        if all(sg > 0 for sg in signs):
            sign_map[i, j] = 1
        elif all(sg < 0 for sg in signs):
            sign_map[i, j] = -1
        else:
            sign_map[i, j] = 0
ax.imshow(sign_map, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", origin="lower")
ax.set_xticks(range(len(KAP))); ax.set_xticklabels([f"{k:.2f}" for k in KAP], fontsize=6.5, rotation=45)
ax.set_yticks(range(len(PIW))); ax.set_yticklabels([f"{p:.2f}" for p in PIW], fontsize=6.5)
ax.set_xlabel(r"$\kappa$ (adult)")
for i in range(len(PIW)):
    for j in range(len(KAP)):
        lab = {1: "+", -1: "-", 0: "·"}[sign_map[i, j]]
        ax.text(j, i, lab, ha="center", va="center", fontsize=8,
                color="black", fontweight="bold")
panel_label(ax, "E")
ax.set_title("4-season sign", fontsize=8.5, pad=3)

cbar = fig.colorbar(im, ax=axes[:4], location="right", shrink=0.85,
                     fraction=0.03, pad=0.01)
cbar.set_label("Sick-leave averted (%)", fontsize=8)
fig.text(0.5, -0.03,
          "Term window; positive = averted, negative = transferred. "
          "Panel E: '+/-' = 4/4 seasons agree, '·' = mixed.",
          ha="center", fontsize=7, color=COL_ZERO)
savefig(fig, "sens_S1_piwork_kappa_heatmap")
print("saved sens_S1_piwork_kappa_heatmap (fluA4)")
