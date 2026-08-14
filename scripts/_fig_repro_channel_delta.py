"""정책별 채널 ΔR_e^c(t) = policy - baseline (daily grid). Supplement figure.
좌: sick leave / 우: school absence. 각 정책이 자기 대상 채널을 억제함을 보임."""
import os, json, sys
from pathlib import Path
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_seasonpop"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from figstyle import savefig, COL_SCHOOL, COL_ZERO, W_DOUBLE

CH_COL = {"home": "#555555", "work": "#55A868", "school": COL_SCHOOL, "other": "#8172B3"}
D = json.load(open(REPO / "outputs/eda/reproduction_numbers_2019-2020.json"))
Rc = D["Re"]["Re_channel"]; t = np.array(D["Re"]["t_days"])
meta = D.get("meta", {})
BREAK = tuple(meta.get("break_window", (113, 183)))
# policy onset = last grid point before school scenario diverges from baseline
_d = np.abs(np.array(D["Re"]["Re"]["baseline"]["mean"]) - np.array(D["Re"]["Re"]["school"]["mean"]))
pol_start = float(t[max(int(np.argmax(_d > 1e-4)) - 1, 0)]); pol_end = BREAK[1]

fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.2), constrained_layout=True, sharex=True)
for ax, (pol, title) in zip(axes, [("sick", "Sick leave"), ("school", "School absence")]):
    ax.axhline(0, color="black", lw=0.8, ls=":", zorder=1)
    ax.axvspan(*BREAK, color="#4C72B0", alpha=0.07, lw=0)
    for x in (pol_start, pol_end):
        ax.axvline(x, color="#333333", lw=1.0, ls="--", zorder=2)
    for ch in ["home", "school", "other", "work"]:
        delta = np.array(Rc[pol][ch]["mean"]) - np.array(Rc["baseline"][ch]["mean"])
        ax.plot(t, delta, color=CH_COL[ch], lw=1.8, label=ch)
    ax.set_xlim(0, 250)
    ax.set_xlabel("Season day (0 = Sep 1)")
    ax.set_title(title, fontsize=10, fontweight="bold")
axes[0].set_ylabel(r"$\Delta R_e^{c}(t)$ = policy $-$ baseline")
axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
axes[0].text(pol_start, axes[0].get_ylim()[1], "policy", ha="left", va="top",
             fontsize=7, color="#333333")
savefig(fig, "repro_channel_delta")
print(f"saved repro_channel_delta (policy window {pol_start:.0f}-{pol_end:.0f})")
