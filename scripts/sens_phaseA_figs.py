"""Phase A sensitivity 그림 생성. figures/v4/sensitivity/ 저장.

Fig S1: π_work × κ 2D heatmap × 3 시즌 (부호 3/3 일관 마스크)
Fig S2: κ 상한 posterior CI (term + vac + age 3패널)
Fig S3: R(0) 5×5 grid child vac Δ heatmap
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (savefig, COL_SICK, COL_SCHOOL, COL_SEASON, COL_ZERO,
                       AGES, SEASONS, W_DOUBLE, panel_label, zero_line,
                       marker_style, is_sig)

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "figures" / "v4" / "sensitivity"
FIG.mkdir(parents=True, exist_ok=True)

# custom savefig into sensitivity dir instead of paper dir
def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=150 if ext=="pdf" else 300)
    plt.close(fig)


# ═════════════ Fig S1: π_work × κ ═════════════
def fig_S1():
    d = json.load(open(REPO/"outputs/eda/sens_piwork_kappa_v4.json"))
    PIW = d["meta"]["pi_work_grid"]; KAP = d["meta"]["kappa_grid"]
    grid = defaultdict(dict)
    for r in d["rows"]:
        grid[(r["pi_work"], r["kappa"])][r["season"]] = r

    fig, axes = plt.subplots(1, 4, figsize=(W_DOUBLE * 1.4, 3.6),
                              constrained_layout=True,
                              gridspec_kw=dict(width_ratios=[1,1,1,1.05]))
    vmax = 3.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    # Panel A-C: 3 시즌 heatmap
    for k, s in enumerate(SEASONS):
        ax = axes[k]
        M = np.zeros((len(PIW), len(KAP)))
        for i, pw in enumerate(PIW):
            for j, ka in enumerate(KAP):
                M[i, j] = grid[(pw, ka)][s]["averted_sick_term"]
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(KAP)))
        ax.set_xticklabels([f"{k:.2f}" for k in KAP], fontsize=7)
        ax.set_yticks(range(len(PIW)))
        ax.set_yticklabels([f"{p:.2f}" for p in PIW], fontsize=7)
        if k == 0: ax.set_ylabel(r"$\pi_{\mathrm{work}}$")
        ax.set_xlabel(r"$\kappa$ (adult)")
        # 값 오버레이
        for i in range(len(PIW)):
            for j in range(len(KAP)):
                col = "white" if abs(M[i, j]) > vmax*0.55 else "black"
                ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center",
                        fontsize=6, color=col)
        panel_label(ax, "ABC"[k])
        ax.set_title(s, fontsize=9, pad=3, color=COL_SEASON[s])

    # Panel D: 3-season sign agreement
    ax = axes[3]
    sign_map = np.zeros((len(PIW), len(KAP)), dtype=int)
    for i, pw in enumerate(PIW):
        for j, ka in enumerate(KAP):
            signs = [np.sign(grid[(pw, ka)][s]["averted_sick_term"]) for s in SEASONS]
            if all(s > 0 for s in signs):   sign_map[i, j] = 1
            elif all(s < 0 for s in signs): sign_map[i, j] = -1
            else:                             sign_map[i, j] = 0
    im2 = ax.imshow(sign_map, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
                     origin="lower")
    ax.set_xticks(range(len(KAP))); ax.set_xticklabels([f"{k:.2f}" for k in KAP], fontsize=7)
    ax.set_yticks(range(len(PIW))); ax.set_yticklabels([f"{p:.2f}" for p in PIW], fontsize=7)
    ax.set_xlabel(r"$\kappa$ (adult)")
    for i in range(len(PIW)):
        for j in range(len(KAP)):
            lab = {1: "+", -1: "-", 0: "·"}[sign_map[i, j]]
            ax.text(j, i, lab, ha="center", va="center", fontsize=8,
                    color="black", fontweight="bold")
    panel_label(ax, "D")
    ax.set_title("3-season sign", fontsize=9, pad=3)

    cbar = fig.colorbar(im, ax=axes[:3], location="right", shrink=0.85,
                         fraction=0.03, pad=0.01)
    cbar.set_label("Sick-leave averted (%)", fontsize=8)
    fig.text(0.5, -0.02,
              "Term window; positive = averted, negative = transferred. "
              "Panel D: '+/-' = 3/3 seasons agree, '·' = mixed.",
              ha="center", fontsize=7, color=COL_ZERO)
    _save(fig, "sens_S1_piwork_kappa_heatmap")


# ═════════════ Fig S2: κ 상한 posterior CI ═════════════
def fig_S2():
    d = json.load(open(REPO/"outputs/eda/sens_kappa_upper_v4.json"))
    KAP = d["meta"]["kappa_grid"]
    by_ks = defaultdict(list); by_ks_v = defaultdict(list)
    by_age_ks = {a: defaultdict(list) for a in AGES}
    for r in d["rows"]:
        by_ks[(r["kappa"], r["season"])].append(r["averted_sick_term"])
        by_ks_v[(r["kappa"], r["season"])].append(r["averted_sick_vac"])
        for a in AGES:
            by_age_ks[a][r["kappa"]].append(r["d_attack_sick_by_age_term"][a])

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE * 1.3, 3.4),
                              constrained_layout=True)
    # Panel A: term averted CI
    ax = axes[0]
    for si, s in enumerate(SEASONS):
        col = COL_SEASON[s]
        xs = np.array(KAP) + (si-1)*0.02
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

    # Panel B: vac averted CI
    ax = axes[1]
    for si, s in enumerate(SEASONS):
        col = COL_SEASON[s]
        xs = np.array(KAP) + (si-1)*0.02
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

    # Panel C: 연령별 Δattack term (κ별)
    ax = axes[2]
    ages_col = ["#4575b4","#74add1","#fdae61","#f46d43","#d73027","#7b3294"]
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
              "Reference: v4 default $\\kappa$ = 0.40 (adult).",
              ha="center", fontsize=7, color=COL_ZERO)
    _save(fig, "sens_S2_kappa_upper_CI")


# ═════════════ Fig S3: R(0) 5×5 child vac ═════════════
def fig_S3():
    d = json.load(open(REPO/"outputs/eda/sens_R0_transition_v4.json"))
    IMM2 = d["meta"]["imm_20_44_grid"]; IMM4 = d["meta"]["imm_45_64_grid"]
    grid = defaultdict(dict)
    for r in d["rows"]:
        grid[(r["imm_20_44"], r["imm_45_64"])][r["season"]] = r

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.4),
                              constrained_layout=True)
    # Panel A: vac child Δ (3시즌 평균)
    ax = axes[0]
    M = np.zeros((len(IMM2), len(IMM4)))
    for i, imm2 in enumerate(IMM2):
        for j, imm4 in enumerate(IMM4):
            M[i, j] = np.mean([grid[(imm2, imm4)][s]["child_delta_vac_sick"]
                                for s in SEASONS])
    vmax = max(abs(M.min()), abs(M.max()))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(IMM4))); ax.set_xticklabels([f"{v:.2f}" for v in IMM4])
    ax.set_yticks(range(len(IMM2))); ax.set_yticklabels([f"{v:.2f}" for v in IMM2])
    ax.set_xlabel(r"$R(0)_{45\text{-}64}$"); ax.set_ylabel(r"$R(0)_{20\text{-}44}$")
    for i in range(len(IMM2)):
        for j in range(len(IMM4)):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(M[i,j])>vmax*0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02,
                  label=r"child $\Delta$ attack, vac (%p, mean of 3 seasons)")
    panel_label(ax, "A")
    ax.set_title("Vacation window", fontsize=9, pad=3)

    # Panel B: term (같은 grid)
    ax = axes[1]
    Mt = np.zeros((len(IMM2), len(IMM4)))
    for i, imm2 in enumerate(IMM2):
        for j, imm4 in enumerate(IMM4):
            Mt[i, j] = np.mean([grid[(imm2, imm4)][s]["child_delta_term_sick"]
                                 for s in SEASONS])
    im2 = ax.imshow(Mt, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(IMM4))); ax.set_xticklabels([f"{v:.2f}" for v in IMM4])
    ax.set_yticks(range(len(IMM2))); ax.set_yticklabels([f"{v:.2f}" for v in IMM2])
    ax.set_xlabel(r"$R(0)_{45\text{-}64}$")
    for i in range(len(IMM2)):
        for j in range(len(IMM4)):
            ax.text(j, i, f"{Mt[i,j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(Mt[i,j])>vmax*0.55 else "black")
    fig.colorbar(im2, ax=ax, fraction=0.05, pad=0.02,
                  label=r"child $\Delta$ attack, term (%p)")
    panel_label(ax, "B")
    ax.set_title("Term window", fontsize=9, pad=3)

    fig.text(0.5, -0.02,
              "v4 reference: $R(0)_{20\\text{-}44}=0.40$, $R(0)_{45\\text{-}64}=0.60$. "
              "No sign inversion in this range (all + across grid).",
              ha="center", fontsize=7, color=COL_ZERO)
    _save(fig, "sens_S3_R0_transition_heatmap")


if __name__ == "__main__":
    fig_S1(); print("S1 saved")
    fig_S2(); print("S2 saved")
    fig_S3(); print("S3 saved")
    print(f"→ {FIG}")
