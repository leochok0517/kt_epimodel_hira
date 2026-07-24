"""논문 그림 F1-F6 생성 + 기준값 대조.

Output: figures/paper/{pdf,png}/{school_vs_sick, school_vs_sick_number,
policy_posterior_byage, rate_vs_number, holiday, nuts_posterior}.
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (
    savefig, is_sig, marker_style, panel_label, zero_line,
    COL_SICK, COL_SCHOOL, COL_SEASON, COL_ZERO,
    AGES, SEASONS, CHANNELS,
    W_SINGLE, W_DOUBLE,
)

REPO = Path(__file__).resolve().parent.parent
POL = json.load(open(REPO / "outputs/eda/policy_posterior_v4.json"))
NUTS_RAW = np.load(REPO / "outputs/eda/nuts_v4_full_raw.npz")
NUTS_EXT = np.load(REPO / "outputs/eda/nuts_v4_full_extended.npz")
NUTS_DIAG = json.load(open(REPO / "outputs/eda/nuts_v4_merged_diagnostics.json"))

# ── Merged NUTS posterior (4000 draws) ──
PI_MERGED = np.concatenate([NUTS_RAW["pi"], NUTS_EXT["pi"]], axis=0)          # (4000, 4)
LOG_R0_MERGED = np.concatenate([NUTS_RAW["log_R0"], NUTS_EXT["log_R0"]], axis=0)  # (4000, 3)
# Chain-structured for trace
PI_CHAINED = np.concatenate([
    NUTS_RAW["pi"].reshape(4, 500, 4),
    NUTS_EXT["pi"].reshape(4, 500, 4),
], axis=0)   # (8, 500, 4)


# ═══════════════════════════════════════════════════════════════════════════
# 검증: §1 기준값 대조
# ═══════════════════════════════════════════════════════════════════════════
def verify_reference():
    print("═"*80)
    print("§1 기준값 대조 (JSON 파싱 검증)")
    print("═"*80)
    ok = True
    # 총 averted
    ref = {
        "2016-2017": (0.49, [0.08, 0.96], 1.58, [0.95, 2.38]),
        "2017-2018": (0.22, [-0.33, 0.87], 3.44, [2.15, 5.08]),
        "2019-2020": (0.40, [-0.05, 0.93], 2.14, [1.31, 3.23]),
    }
    for s, (sk_m, sk_ci, sc_m, sc_ci) in ref.items():
        sk = POL[s]["sick_total"]; sc = POL[s]["school_total"]
        if not (abs(sk["mean"] - sk_m) < 0.03 and abs(sc["mean"] - sc_m) < 0.03):
            print(f"  ✗ {s}: sick got mean={sk['mean']:.2f} ref={sk_m} | "
                  f"school got={sc['mean']:.2f} ref={sc_m}")
            ok = False
        else:
            print(f"  ✓ {s}: sick={sk['mean']:+.2f} 90%[{sk['q05']:+.2f},{sk['q95']:+.2f}] "
                  f"| school={sc['mean']:+.2f} 90%[{sc['q05']:+.2f},{sc['q95']:+.2f}]")
    # NUTS π_work
    piw = np.quantile(PI_MERGED[:, 1], [0.025, 0.5, 0.975])
    print(f"\n  π_work merged 95%CI = [{piw[0]:.3f}, {piw[2]:.3f}] "
          f"(ref [0.242, 0.353])")
    if abs(piw[0] - 0.242) > 0.01 or abs(piw[2] - 0.353) > 0.01:
        print("  ✗ π_work CI mismatch")
        ok = False
    else:
        print("  ✓ π_work CI matches")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# F1 — school_vs_sick (총 averted CI, 시즌×정책)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F1():
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.75, 3.0), constrained_layout=True)
    xs = np.arange(len(SEASONS))
    dx = 0.15
    # 병가 & 학교 각 시즌
    for j, s in enumerate(SEASONS):
        sk = POL[s]["sick_total"]; sc = POL[s]["school_total"]
        # 병가 (주황)
        sig_sk = is_sig(sk["q05"], sk["q95"])
        yerr_sk = [[sk["mean"] - sk["q05"]], [sk["q95"] - sk["mean"]]]
        ax.errorbar([j - dx], [sk["mean"]], yerr=yerr_sk,
                    color=COL_SICK, lw=1.2, capsize=3, zorder=3,
                    label="Sick leave" if j == 0 else None,
                    **marker_style(sig_sk, COL_SICK))
        # 학교 (파랑)
        sig_sc = is_sig(sc["q05"], sc["q95"])
        yerr_sc = [[sc["mean"] - sc["q05"]], [sc["q95"] - sc["mean"]]]
        ax.errorbar([j + dx], [sc["mean"]], yerr=yerr_sc,
                    color=COL_SCHOOL, lw=1.2, capsize=3, zorder=3,
                    label="School absence" if j == 0 else None,
                    **marker_style(sig_sc, COL_SCHOOL))
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(SEASONS)
    ax.set_ylabel("Averted infections (%)")
    ax.set_xlabel("Season")
    # legend
    handles = [
        Line2D([0], [0], marker="o", markerfacecolor=COL_SICK,
                 markeredgecolor=COL_SICK, ls="-", color=COL_SICK,
                 label=r"Sick leave (filled: $0\notin$ 90% CI)", markersize=6),
        Line2D([0], [0], marker="o", markerfacecolor="white",
                 markeredgecolor=COL_SICK, ls="-", color=COL_SICK,
                 label=r"Sick leave (open: $0\in$ 90% CI)", markersize=6),
        Line2D([0], [0], marker="o", markerfacecolor=COL_SCHOOL,
                 markeredgecolor=COL_SCHOOL, ls="-", color=COL_SCHOOL,
                 label="School absence", markersize=6),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.margins(x=0.15)
    savefig(fig, "school_vs_sick")


# ═══════════════════════════════════════════════════════════════════════════
# F2 — school_vs_sick_number (연령별 감염 수 감소, 2019-20 대표)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F2():
    s = "2019-2020"
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.85, 3.8), constrained_layout=True)
    ys = np.arange(len(AGES))
    dy = 0.2
    for i, ag in enumerate(AGES):
        sk = POL[s]["sick_num_by_age"][ag]; sc = POL[s]["school_num_by_age"][ag]
        sig_sk = is_sig(sk["q05"], sk["q95"])
        sig_sc = is_sig(sc["q05"], sc["q95"])
        xerr_sk = [[sk["mean"] - sk["q05"]], [sk["q95"] - sk["mean"]]]
        xerr_sc = [[sc["mean"] - sc["q05"]], [sc["q95"] - sc["mean"]]]
        ax.errorbar([sk["mean"]], [i - dy], xerr=xerr_sk,
                    color=COL_SICK, lw=1.2, capsize=3,
                    **marker_style(sig_sk, COL_SICK))
        ax.errorbar([sc["mean"]], [i + dy], xerr=xerr_sc,
                    color=COL_SCHOOL, lw=1.2, capsize=3,
                    **marker_style(sig_sc, COL_SCHOOL))
    zero_line(ax, orient="v")
    ax.set_yticks(ys); ax.set_yticklabels(AGES)
    ax.invert_yaxis()   # 0-5 위쪽
    ax.set_xlabel("Averted infections (n, 2019-20)")
    ax.set_ylabel("Age group")

    # 순합계 (paper 컨벤션: adult 18-64, child 0-17, 65+ 별도)
    CHILD = ["0-5","6-11","12-17"]; ADULT_2 = ["18-44","45-64"]
    tot_sk_adult = sum(POL[s]["sick_num_by_age"][a]["mean"] for a in ADULT_2)
    tot_sk_child = sum(POL[s]["sick_num_by_age"][a]["mean"] for a in CHILD)
    tot_sk_net = tot_sk_adult + tot_sk_child
    tot_sc = sum(POL[s]["school_num_by_age"][a]["mean"] for a in AGES)

    # 범례 (축 밖 하단, 겹침 방지)
    handles = [
        Line2D([0], [0], marker="o", color=COL_SICK, ls="",
                 markersize=6, label="Sick leave"),
        Line2D([0], [0], marker="o", color=COL_SCHOOL, ls="",
                 markersize=6, label="School absence"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=False, fontsize=8)

    # Total averted 박스 — 데이터 밖 오른쪽 여백 (x 축 상단, 데이터 clip 확장)
    # 65+ 마커는 y=5, x≈−400 근처. 65+ 마커와 겹치지 않게 그림 우상단(fig 좌표) 밖.
    ax.text(1.02, 0.98,
            f"Net averted (sick leave, paper convention):\n"
            f"  Adults (18–64): +{tot_sk_adult:,.0f}\n"
            f"  Children (0–17): {tot_sk_child:+,.0f}\n"
            f"  Net: +{tot_sk_net:,.0f}\n\n"
            f"School absence total: +{tot_sc:,.0f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec=COL_ZERO, lw=0.5))
    savefig(fig, "school_vs_sick_number")


# ═══════════════════════════════════════════════════════════════════════════
# F3 — policy_posterior_byage (연령별 Δattack posterior)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F3():
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.6), constrained_layout=True)
    xs = np.arange(len(AGES))
    # 병가 3시즌: 시즌별 3개 계열, x 위치 살짝씩 오프셋
    dx = 0.11
    for j, s in enumerate(SEASONS):
        offs = (j - 1) * dx
        means, q05s, q95s, sigs = [], [], [], []
        for ag in AGES:
            v = POL[s]["sick_by_age"][ag]
            means.append(v["mean"])
            q05s.append(v["q05"]); q95s.append(v["q95"])
            sigs.append(is_sig(v["q05"], v["q95"]))
        means = np.array(means); q05s = np.array(q05s); q95s = np.array(q95s)
        yerr = np.vstack([means - q05s, q95s - means])
        col = COL_SEASON[s]
        # 유의·비유의 분리 산점
        for i, sg in enumerate(sigs):
            ax.errorbar([xs[i] + offs], [means[i]],
                        yerr=[[yerr[0, i]], [yerr[1, i]]],
                        color=col, lw=1.0, capsize=2, zorder=3,
                        **marker_style(sg, col))
        # 시즌 라벨용 dummy line
        ax.plot([], [], color=col, marker="o", markersize=5,
                label=f"Sick leave, {s}")
    # 학교결석 3시즌 range band
    sc_low = np.array([min(POL[s]["school_by_age"][ag]["q05"] for s in SEASONS)
                        for ag in AGES])
    sc_high = np.array([max(POL[s]["school_by_age"][ag]["q95"] for s in SEASONS)
                         for ag in AGES])
    sc_mean = np.array([np.mean([POL[s]["school_by_age"][ag]["mean"]
                                  for s in SEASONS]) for ag in AGES])
    ax.fill_between(xs, sc_low, sc_high, color=COL_SCHOOL, alpha=0.15, zorder=1,
                     label="School absence, 3-season range (90% CI)")
    ax.plot(xs, sc_mean, color=COL_SCHOOL, lw=1.5, marker="s", markersize=5,
             zorder=2, label="School absence, mean of 3 seasons")
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(AGES)
    ax.set_xlabel("Age group")
    ax.set_ylabel(r"$\Delta$ attack rate (percentage points)")
    # 범례: 축 밖 하단 (school band 겹침 방지)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, fontsize=7.5)
    savefig(fig, "policy_posterior_byage")


# ═══════════════════════════════════════════════════════════════════════════
# F4 — rate_vs_number (병가, 2019-20, Panel A=%pt, B=명)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F4():
    s = "2019-2020"
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.2),
                              constrained_layout=True)
    xs = np.arange(len(AGES))

    # Panel A: Averted attack rate (%p) = -Δattack.  양수 = 감염 감소 (averted).
    ax = axes[0]
    for i, ag in enumerate(AGES):
        v = POL[s]["sick_by_age"][ag]
        # 부호 반전: averted = -Δattack
        mean_av = -v["mean"]
        q05_av = -v["q95"]   # q95 of Δ → q05 of -Δ
        q95_av = -v["q05"]
        sig = is_sig(v["q05"], v["q95"])   # 유의성은 부호 무관
        yerr = [[mean_av - q05_av], [q95_av - mean_av]]
        ax.errorbar([i], [mean_av], yerr=yerr,
                    color=COL_SICK, lw=1.2, capsize=3,
                    **marker_style(sig, COL_SICK))
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(AGES, rotation=0, fontsize=7.5)
    ax.set_ylabel("Averted attack rate (%p)")
    ax.set_xlabel("Age group")
    panel_label(ax, "A")

    # Panel B: averted number (명), 이미 positive = averted
    ax = axes[1]
    for i, ag in enumerate(AGES):
        v = POL[s]["sick_num_by_age"][ag]
        sig = is_sig(v["q05"], v["q95"])
        yerr = [[v["mean"] - v["q05"]], [v["q95"] - v["mean"]]]
        ax.errorbar([i], [v["mean"]], yerr=yerr,
                    color=COL_SICK, lw=1.2, capsize=3,
                    **marker_style(sig, COL_SICK))
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(AGES, rotation=0, fontsize=7.5)
    ax.set_ylabel("Averted infections (n)")
    ax.set_xlabel("Age group")
    panel_label(ax, "B")

    # 부호 규약 note (fig 하단)
    fig.text(0.5, -0.04,
              r"Sick leave, 2019-20. Both panels: positive $=$ averted "
              r"(fewer infections); negative $=$ transferred to this age. "
              r"Filled: $0\notin$ 90% CI.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "rate_vs_number")


# ═══════════════════════════════════════════════════════════════════════════
# F5 — holiday (아동 Δattack, 학기 vs 방학 창, 3시즌)
# ═══════════════════════════════════════════════════════════════════════════
# 참고: policy_posterior_v4.json 는 term-window 만 계산. 방학창 데이터는 없음.
# 대신 outputs/eda/kappa_no_eta_presymp_holiday.json (이전 job 5) 있으나
# posterior CI 없음 → point-est 값과 posterior CI 결합.
# 여기서는 term-window (기본 policy_posterior_v4) 만 사용해 아동 3연령 그룹
# Δattack 을 시즌×연령 표시. 방학창 별도 없다면 caption 에 명시.
def fig_F5():
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.85, 3.2), constrained_layout=True)
    child_ages = ["0-5", "6-11", "12-17"]
    xs = np.arange(len(child_ages))
    dx = 0.18
    for j, s in enumerate(SEASONS):
        offs = (j - 1) * dx
        for i, ag in enumerate(child_ages):
            v = POL[s]["sick_by_age"][ag]
            sig = is_sig(v["q05"], v["q95"])
            yerr = [[v["mean"] - v["q05"]], [v["q95"] - v["mean"]]]
            col = COL_SEASON[s]
            ax.errorbar([xs[i] + offs], [v["mean"]], yerr=yerr,
                        color=col, lw=1.0, capsize=2, zorder=3,
                        **marker_style(sig, col))
        ax.plot([], [], color=COL_SEASON[s], marker="o", ls="",
                label=s, markersize=5)
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(child_ages)
    ax.set_xlabel("Age group (children)")
    ax.set_ylabel(r"$\Delta$ attack rate (%p), sick leave in term window")
    ax.legend(title="Season", frameon=False, loc="upper left", fontsize=7.5,
               title_fontsize=8)
    fig.text(0.5, -0.03,
              "Term window (season days 70–113). Positive = infection transferred "
              "to children.  Posterior 90% CI shown; sign consistent across seasons.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "holiday")


# ═══════════════════════════════════════════════════════════════════════════
# F6 — nuts_posterior (π 4채널 + R0 3시즌 + π_work trace)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F6():
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.0),
                              constrained_layout=True,
                              gridspec_kw=dict(width_ratios=[1.2, 1.0, 1.3]))

    # Panel A: π 4채널 95% CI
    ax = axes[0]
    ys = np.arange(len(CHANNELS))
    for i, ch in enumerate(CHANNELS):
        arr = PI_MERGED[:, i]
        m = arr.mean()
        q025, q975 = np.quantile(arr, [0.025, 0.975])
        # π_work 강조
        col = COL_SICK if ch == "work" else COL_SCHOOL
        xerr = [[m - q025], [q975 - m]]
        ax.errorbar([m], [i], xerr=xerr, color=col, lw=1.2, capsize=3,
                    marker="o", markersize=6, markerfacecolor=col,
                    markeredgecolor=col)
    ax.set_yticks(ys); ax.set_yticklabels(CHANNELS)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\pi$ (channel share)")
    ax.set_xlim(0, 0.6)
    panel_label(ax, "A")

    # Panel B: R0 시즌별 95% CI
    ax = axes[1]
    for j, s in enumerate(SEASONS):
        arr = np.exp(LOG_R0_MERGED[:, j])
        m = arr.mean()
        q025, q975 = np.quantile(arr, [0.025, 0.975])
        col = COL_SEASON[s]
        xerr = [[m - q025], [q975 - m]]
        ax.errorbar([m], [j], xerr=xerr, color=col, lw=1.2, capsize=3,
                    marker="o", markersize=6, markerfacecolor=col,
                    markeredgecolor=col)
    ax.set_yticks(range(len(SEASONS))); ax.set_yticklabels(SEASONS)
    ax.invert_yaxis()
    ax.set_xlabel(r"$R_0$")
    ax.set_xlim(1.95, 2.4)
    panel_label(ax, "B")

    # Panel C: π_work trace (8 chains)
    ax = axes[2]
    for ch in range(PI_CHAINED.shape[0]):
        ax.plot(PI_CHAINED[ch, :, 1], lw=0.4, alpha=0.6)
    ax.set_xlabel("Draw"); ax.set_ylabel(r"$\pi_{\mathrm{work}}$")
    panel_label(ax, "C")

    # 진단 수치 요약 (caption 담당이나 그림 안에도 subtle 하게)
    rhat = NUTS_DIAG["merged"]["pi[1]"]["r_hat"]
    ess = NUTS_DIAG["merged"]["pi[1]"]["ess_bulk"]
    ax.text(0.02, 0.98,
             f"$\\hat{{R}}$={rhat:.3f}  ESS={ess:.0f}",
             transform=ax.transAxes, va="top", ha="left", fontsize=7,
             color=COL_ZERO)

    savefig(fig, "nuts_posterior")


def main():
    print("검증 시작...\n")
    ok = verify_reference()
    if not ok:
        print("\n★ 기준값 불일치 발견 — 그림 생성 중단.")
        sys.exit(1)
    print("\n검증 통과 → 그림 생성")
    fig_F1(); print("  F1 school_vs_sick")
    fig_F2(); print("  F2 school_vs_sick_number")
    fig_F3(); print("  F3 policy_posterior_byage")
    fig_F4(); print("  F4 rate_vs_number")
    fig_F5(); print("  F5 holiday")
    fig_F6(); print("  F6 nuts_posterior")
    print("\n완료. figures/paper/pdf/, figures/paper/png/ 확인.")


if __name__ == "__main__":
    main()
