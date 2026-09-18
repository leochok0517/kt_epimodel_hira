"""Figure sens_S3_R0_transition_heatmap for fluA4 (4-season mean).

Copy of sens_phaseA_figs.py's fig_S3, reading outputs/fluA/
sens_r0_immunity_fluA4.json (sens3_R0_transition_fluA4.py), averaged over
4 seasons instead of 3. Saved through figstyle.savefig into
figures/paper_fluA4/ (flat layout). Filename kept identical.
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

d = json.load(open(REPO / "outputs/fluA/sens_r0_immunity_fluA4.json"))
IMM2 = d["meta"]["imm_20_44_grid"]; IMM4 = d["meta"]["imm_45_64_grid"]
grid = defaultdict(dict)
for r in d["rows"]:
    grid[(r["imm_20_44"], r["imm_45_64"])][r["season"]] = r

fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.4), constrained_layout=True)

ax = axes[0]
M = np.zeros((len(IMM2), len(IMM4)))
for i, imm2 in enumerate(IMM2):
    for j, imm4 in enumerate(IMM4):
        M[i, j] = np.mean([grid[(imm2, imm4)][s]["child_delta_vac_sick"] for s in SEASONS])
vmax = max(abs(M.min()), abs(M.max()))
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
ax.set_xticks(range(len(IMM4))); ax.set_xticklabels([f"{v:.2f}" for v in IMM4])
ax.set_yticks(range(len(IMM2))); ax.set_yticklabels([f"{v:.2f}" for v in IMM2])
ax.set_xlabel(r"$R(0)_{45\text{-}64}$"); ax.set_ylabel(r"$R(0)_{20\text{-}44}$")
for i in range(len(IMM2)):
    for j in range(len(IMM4)):
        ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(M[i, j]) > vmax * 0.55 else "black")
fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02,
              label=r"child $\Delta$ attack, vac (%p, mean of 4 seasons)")
panel_label(ax, "A")
ax.set_title("Vacation window", fontsize=9, pad=3)

ax = axes[1]
Mt = np.zeros((len(IMM2), len(IMM4)))
for i, imm2 in enumerate(IMM2):
    for j, imm4 in enumerate(IMM4):
        Mt[i, j] = np.mean([grid[(imm2, imm4)][s]["child_delta_term_sick"] for s in SEASONS])
im2 = ax.imshow(Mt, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
ax.set_xticks(range(len(IMM4))); ax.set_xticklabels([f"{v:.2f}" for v in IMM4])
ax.set_yticks(range(len(IMM2))); ax.set_yticklabels([f"{v:.2f}" for v in IMM2])
ax.set_xlabel(r"$R(0)_{45\text{-}64}$")
for i in range(len(IMM2)):
    for j in range(len(IMM4)):
        ax.text(j, i, f"{Mt[i, j]:+.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(Mt[i, j]) > vmax * 0.55 else "black")
fig.colorbar(im2, ax=ax, fraction=0.05, pad=0.02,
              label=r"child $\Delta$ attack, term (%p)")
panel_label(ax, "B")
ax.set_title("Term window", fontsize=9, pad=3)

n_pos = int((M > 0).sum()); n_neg = int((M < 0).sum()); n_tot = M.size
fig.text(0.5, -0.02,
          f"fluA4 reference: $R(0)_{{20\\text{{-}}44}}=0.40$, $R(0)_{{45\\text{{-}}64}}=0.60$. "
          f"Vacation panel: {n_pos}/{n_tot} grid cells positive, {n_neg}/{n_tot} negative "
          "(mean of 4 seasons per cell -- season-level sign flips are not shown here).",
          ha="center", fontsize=7, color=COL_ZERO)
savefig(fig, "sens_S3_R0_transition_heatmap")
print(f"saved sens_S3_R0_transition_heatmap (fluA4); vac panel {n_pos}/{n_tot} positive, {n_neg}/{n_tot} negative")
