"""fit_total 재작도 — 모형 곡선을 시즌색 대신 검정으로 통일.

시뮬레이션 재실행 없이 fig_fit_total_fluA4.py 가 남긴
outputs/fluA/fit_total_curves_fluA4.json (시즌별 mean 곡선 + 관측점) 만 쓴다.
"""
import json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(os.environ["HOME"]) / "mnt" / "kt_epimodel_hira"
OUT = Path(os.environ["HOME"]) / "mnt" / "sickleave_InfluenzaA" / "manuscript" / "figures_fluA4"
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
sys.path.insert(0, str(REPO / "scripts"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import savefig, COL_ZERO, W_DOUBLE

SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
INK = "black"

D = json.load(open(REPO / "outputs" / "fluA" / "fit_total_curves_fluA4.json"))

fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.0), constrained_layout=True)
for k, s in enumerate(SEASONS):
    ax = axes[k // 2][k % 2]
    mean_pr = np.asarray(D[s]["mean"]); o_tot = np.asarray(D[s]["obs"])
    wks = np.arange(len(mean_pr))
    ax.plot(wks, mean_pr, color=INK, lw=1.6)
    ax.plot(wks, o_tot, "o", color=COL_ZERO, ms=3, alpha=0.75)
    ax.set_title(s, color=INK, fontsize=9, fontweight="bold")
    ax.set_xlabel("Epidemic week")
    if k % 2 == 0:
        ax.set_ylabel("Weekly incidence (influenza A)")

handles = [
    Line2D([0], [0], color=INK, lw=1.6, label="Posterior mean"),
    Line2D([0], [0], marker="o", color=COL_ZERO, ls="", markersize=4,
           alpha=0.75, label="Observed (influenza A)"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
           ncol=2, frameon=False, fontsize=8)
pdf, png = savefig(fig, "fit_total")
for src in (pdf, png):
    (OUT / Path(src).parent.name / Path(src).name).write_bytes(Path(src).read_bytes())
print("ok ->", png)
