"""fit_total 재생성 (캐시 사용, 재적합 없음).
legend의 'Posterior mean'은 검정 단일 선 하나로 표시."""
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
from figstyle import savefig, COL_SEASON, COL_ZERO, SEASONS, W_DOUBLE

CURVES = json.load(open(REPO / "outputs/eda/fit_total_curves.json"))

fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.8),
                          constrained_layout=True, sharey=False)
for k, s in enumerate(SEASONS):
    ax = axes[k]
    mean_pr = np.asarray(CURVES[s]["mean"])
    o = np.asarray(CURVES[s]["obs"])
    wks = np.arange(len(mean_pr))
    col = COL_SEASON[s]
    ax.plot(wks, mean_pr, color=col, lw=1.6)
    ax.plot(wks[:len(o)], o, "o", color=COL_ZERO, ms=2.5, alpha=0.75)
    ax.set_xlabel("Epidemic week")
    if k == 0:
        ax.set_ylabel("Weekly incidence (NHIS)")
    ax.set_title(s, color=col, fontsize=9, fontweight="bold")

handles = [
    Line2D([0], [0], color="black", lw=1.6, label="Posterior mean"),
    Line2D([0], [0], marker="o", color=COL_ZERO, ls="",
           markersize=4, alpha=0.75, label="Observed (NHIS)"),
]
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
           fontsize=8, handlelength=2.2, columnspacing=2.2)
savefig(fig, "fit_total")
print("regenerated fit_total (black single legend line)")
