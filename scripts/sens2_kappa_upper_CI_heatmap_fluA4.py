"""Figure sens_S2_kappa_upper_CI for fluA4 (4 seasons).

Copy of sens_phaseA_figs.py's fig_S2, reading outputs/fluA/
sens_kappa_upper_fluA4.json (sens2_kappa_upper_fluA4.py) instead of the v5
3-season JSON, saved through figstyle.savefig into figures/paper_fluA4/
(flat layout matching the rest of the fluA4 figure set) rather than the
original figures/v4.../sensitivity/ nested layout. Filename kept identical.
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import savefig, COL_ZERO, AGES, W_DOUBLE, panel_label, zero_line, marker_style

REPO = Path(__file__).resolve().parent.parent
SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
COL_SEASON_FLUA4 = {"2016-2017": "#009E73", "2017-2018": "#CC79A7",
                     "2018-2019": "#D55E00", "2019-2020": "#56B4E9"}

d = json.load(open(REPO / "outputs/fluA/sens_kappa_upper_fluA4.json"))
KAP = d["meta"]["kappa_grid"]
by_ks = defaultdict(list); by_ks_v = defaultdict(list)
by_age_ks = {a: defaultdict(list) for a in AGES}
for r in d["rows"]:
    by_ks[(r["kappa"], r["season"])].append(r["averted_sick_term"])
    by_ks_v[(r["kappa"], r["season"])].append(r["averted_sick_vac"])
    for a in AGES:
        by_age_ks[a][r["kappa"]].append(r["d_attack_sick_by_age_term"][a])

fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE * 1.3, 3.4), constrained_layout=True)

ax = axes[0]
for si, s in enumerate(SEASONS):
    col = COL_SEASON_FLUA4[s]
    xs = np.array(KAP) + (si - 1.5) * 0.015
    means = [np.mean(by_ks[(k, s)]) for k in KAP]
    q05 = [np.quantile(by_ks[(k, s)], 0.05) for k in KAP]
    q95 = [np.quantile(by_ks[(k, s)], 0.95) for k in KAP]
    for i, k in enumerate(KAP):
        sig = q05[i] * q95[i] > 0
        yerr = [[means[i] - q05[i]], [q95[i] - means[i]]]
        ax.errorbar([xs[i]], [means[i]], yerr=yerr, color=col,
                    lw=1.0, capsize=2, **marker_style(sig, col))
    ax.plot([], [], color=col, marker="o", ls="-", markersize=5, label=s)
zero_line(ax)
ax.set_xlabel(r"$\kappa$ (adult)"); ax.set_ylabel("Averted % (term)")
ax.legend(fontsize=7, frameon=False)
panel_label(ax, "A")

ax = axes[1]
for si, s in enumerate(SEASONS):
    col = COL_SEASON_FLUA4[s]
    xs = np.array(KAP) + (si - 1.5) * 0.015
    means = [np.mean(by_ks_v[(k, s)]) for k in KAP]
    q05 = [np.quantile(by_ks_v[(k, s)], 0.05) for k in KAP]
    q95 = [np.quantile(by_ks_v[(k, s)], 0.95) for k in KAP]
    for i, k in enumerate(KAP):
        sig = q05[i] * q95[i] > 0
        yerr = [[means[i] - q05[i]], [q95[i] - means[i]]]
        ax.errorbar([xs[i]], [means[i]], yerr=yerr, color=col,
                    lw=1.0, capsize=2, **marker_style(sig, col))
zero_line(ax)
ax.set_xlabel(r"$\kappa$ (adult)"); ax.set_ylabel("Averted % (vacation)")
panel_label(ax, "B")

ax = axes[2]
ages_col = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#7b3294"]
for a_i, a in enumerate(AGES):
    means = [np.mean(by_age_ks[a][k]) for k in KAP]
    ax.plot(KAP, means, "o-", color=ages_col[a_i], lw=1.3, ms=4, label=a)
zero_line(ax)
ax.set_xlabel(r"$\kappa$ (adult)")
ax.set_ylabel(r"$\Delta$ attack (%p, term)")
ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper left")
panel_label(ax, "C")

fig.text(0.5, -0.02,
          "Posterior N=50 forward sim. Filled: $0\\notin$ 90% CI. "
          "Reference: fitted empirical $\\kappa$ = 0.34 (student) / 0.40 (adult).",
          ha="center", fontsize=7, color=COL_ZERO)
savefig(fig, "sens_S2_kappa_upper_CI")
print("saved sens_S2_kappa_upper_CI (fluA4)")
