"""T5 -- fluA4 (4-season) versions of paper_figures.py's F1/F4/F5/F6:
school_vs_sick, rate_vs_number, holiday, nuts_posterior.

paper_figures.py already reads its NUTS/policy paths from env vars, but its
season list (``SEASONS``, imported from figstyle) and its schema-adapter's
season loop are hardcoded to v5's 3 seasons, and its posterior npz layout
(separate raw+extended chains) doesn't match fluA4's single merged npz --
hence a proper copy rather than an env-var override.

Output: figures/paper_fluA4/{pdf,png}/{school_vs_sick, rate_vs_number,
holiday, nuts_posterior}.
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (
    savefig, is_sig, marker_style, panel_label, zero_line,
    COL_SICK, COL_SCHOOL, COL_SEASON, COL_ZERO,
    AGES, CHANNELS, W_SINGLE, W_DOUBLE,
)

REPO = Path(__file__).resolve().parent.parent
SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
COL_SEASON_FLUA4 = dict(COL_SEASON, **{"2018-2019": "#E69F00"})

POL = json.loads((REPO / "outputs/fluA/policy_posterior_fluA4.json").read_text())
NUTS = np.load(REPO / "outputs/eda/nuts_fluA4_raw.npz", allow_pickle=True)
DIAG = json.loads((REPO / "outputs/fluA/nuts_fluA4_diagnostics.json").read_text())


def _adapt_schema(pol):
    """sick_total_term -> sick_total (F1/F4 read the un-suffixed legacy key
    for the term-window headline number, matching v5's convention)."""
    for s in SEASONS:
        if s not in pol:
            continue
        d = pol[s]
        d["sick_total"] = d["sick_total_term"]
        d["school_total"] = d["school_total_term"]
        d["sick_by_age"] = d.get("sick_d_by_age_term", {})
        d["school_by_age"] = d.get("school_d_by_age_term", {})
        d["sick_num_by_age"] = d.get("sick_num_by_age_term", {})
        d["school_num_by_age"] = d.get("school_num_by_age_term", {})


_adapt_schema(POL)

PI_MERGED = np.asarray(NUTS["pi"])                 # (8000, 4)
LOG_R0_MERGED = np.asarray(NUTS["log_R0"])          # (8000, 4)
CFG = DIAG["config"]
PI_CHAINED = PI_MERGED.reshape(CFG["chains"], CFG["samples"], 4)   # (4, 2000, 4)


# ═══════════════════════════════════════════════════════════════════════════
# F1 — school_vs_sick (total averted CI, season x policy)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F1():
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.85, 3.0), constrained_layout=True)
    xs = np.arange(len(SEASONS))
    dx = 0.15
    for j, s in enumerate(SEASONS):
        sk = POL[s]["sick_total"]; sc = POL[s]["school_total"]
        sig_sk = is_sig(sk["q05"], sk["q95"])
        yerr_sk = [[sk["mean"] - sk["q05"]], [sk["q95"] - sk["mean"]]]
        ax.errorbar([j - dx], [sk["mean"]], yerr=yerr_sk,
                    color=COL_SICK, lw=1.2, capsize=3, zorder=3,
                    label="Sick leave" if j == 0 else None,
                    **marker_style(sig_sk, COL_SICK))
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
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.5)
    ax.margins(x=0.15)
    savefig(fig, "school_vs_sick")


# ═══════════════════════════════════════════════════════════════════════════
# F4 — rate_vs_number (2019-20 representative, sick leave by age)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F4():
    s = "2019-2020"
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.2), constrained_layout=True)
    xs = np.arange(len(AGES))

    ax = axes[0]
    for i, ag in enumerate(AGES):
        v = POL[s]["sick_by_age"][ag]
        mean_av = -v["mean"]; q05_av = -v["q95"]; q95_av = -v["q05"]
        sig = is_sig(v["q05"], v["q95"])
        yerr = [[mean_av - q05_av], [q95_av - mean_av]]
        ax.errorbar([i], [mean_av], yerr=yerr, color=COL_SICK, lw=1.2, capsize=3,
                    **marker_style(sig, COL_SICK))
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(AGES, rotation=0, fontsize=7.5)
    ax.set_ylabel("Averted attack rate (%p)")
    ax.set_xlabel("Age group")
    panel_label(ax, "A")

    ax = axes[1]
    for i, ag in enumerate(AGES):
        v = POL[s]["sick_num_by_age"][ag]
        sig = is_sig(v["q05"], v["q95"])
        yerr = [[v["mean"] - v["q05"]], [v["q95"] - v["mean"]]]
        ax.errorbar([i], [v["mean"]], yerr=yerr, color=COL_SICK, lw=1.2, capsize=3,
                    **marker_style(sig, COL_SICK))
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(AGES, rotation=0, fontsize=7.5)
    ax.set_ylabel("Averted infections (n)")
    ax.set_xlabel("Age group")
    panel_label(ax, "B")

    fig.text(0.5, -0.04,
              r"Sick leave, 2019-20. Both panels: positive $=$ averted "
              r"(fewer infections); negative $=$ transferred to this age. "
              r"Filled: $0\notin$ 90% CI.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "rate_vs_number")


# ═══════════════════════════════════════════════════════════════════════════
# F5 — holiday (child Delta-attack, school-term vs winter-break, 4 seasons)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F5():
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 1.0, 3.4), constrained_layout=True)
    child_ages = ["0-5", "6-11", "12-17"]
    xs = np.arange(len(child_ages))
    dx = 0.10

    def _key(pol_s, win):
        if win == "term":
            return pol_s.get("sick_d_by_age_term") or pol_s.get("sick_by_age", {})
        return pol_s.get("sick_d_by_age_vac", {})

    n = len(SEASONS)
    for j, s in enumerate(SEASONS):
        col = COL_SEASON_FLUA4[s]
        base_off = (j - (n - 1) / 2.0) * 0.20
        for wi, (win, mk) in enumerate([("term", "o"), ("vac", "s")]):
            src = _key(POL[s], win)
            offs = base_off + (wi - 0.5) * dx
            for i, ag in enumerate(child_ages):
                v = src.get(ag, {})
                if not v:
                    continue
                sig = is_sig(v["q05"], v["q95"])
                mean_av = -v["mean"]
                yerr = [[v["q95"] - v["mean"]], [v["mean"] - v["q05"]]]
                ax.errorbar([xs[i] + offs], [mean_av], yerr=yerr,
                            color=col, lw=1.0, capsize=2, zorder=3,
                            marker=mk, markersize=6,
                            markerfacecolor=(col if sig else "white"),
                            markeredgecolor=col, markeredgewidth=1.2)
        ax.plot([], [], color=col, marker="o", ls="", label=s, markersize=5)
    win_handles = [
        Line2D([0], [0], marker="o", color=COL_ZERO, ls="",
                 markersize=6, label="School term (days 70-113)"),
        Line2D([0], [0], marker="s", color=COL_ZERO, ls="",
                 markersize=6, label="Winter break (days 113-183)"),
    ]
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(child_ages)
    ax.set_xlabel("Age group (children)")
    ax.set_ylabel(r"Averted attack rate (%p), sick leave")
    leg1 = ax.legend(title="Season", frameon=False, fontsize=7.5,
                      title_fontsize=8, loc="upper left",
                      bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=win_handles, frameon=False, fontsize=7.5,
               loc="lower left", bbox_to_anchor=(1.01, 0.0))
    savefig(fig, "holiday")


# ═══════════════════════════════════════════════════════════════════════════
# F6 — nuts_posterior (pi 4-channel + R_peak,s 4 seasons + pi_work trace)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F6():
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.2),
                              constrained_layout=True,
                              gridspec_kw=dict(width_ratios=[1.2, 1.0, 1.3]))

    ax = axes[0]
    ys = np.arange(len(CHANNELS))
    for i, ch in enumerate(CHANNELS):
        arr = PI_MERGED[:, i]
        m = arr.mean()
        q025, q975 = np.quantile(arr, [0.025, 0.975])
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

    ax = axes[1]
    for j, s in enumerate(SEASONS):
        arr = np.exp(LOG_R0_MERGED[:, j])
        m = arr.mean()
        q025, q975 = np.quantile(arr, [0.025, 0.975])
        col = COL_SEASON_FLUA4[s]
        xerr = [[m - q025], [q975 - m]]
        ax.errorbar([m], [j], xerr=xerr, color=col, lw=1.2, capsize=3,
                    marker="o", markersize=6, markerfacecolor=col,
                    markeredgecolor=col)
    ax.set_yticks(range(len(SEASONS))); ax.set_yticklabels(SEASONS)
    ax.invert_yaxis()
    ax.set_xlabel(r"$R_{\mathrm{peak},s}$")
    ax.set_xlim(1.9, 2.5)
    panel_label(ax, "B")

    ax = axes[2]
    for ch in range(PI_CHAINED.shape[0]):
        ax.plot(PI_CHAINED[ch, :, 1], lw=0.4, alpha=0.6)
    ax.set_xlabel("Draw"); ax.set_ylabel(r"$\pi_{\mathrm{work}}$")
    panel_label(ax, "C")

    rhat = DIAG["rhat_max"]; ess = DIAG["ess_bulk_min"]
    ax.text(0.02, 0.98, f"$\\hat{{R}}_{{\\max}}$={rhat:.4f}  ESS$_{{\\min}}$={ess:.0f}",
             transform=ax.transAxes, va="top", ha="left", fontsize=7, color=COL_ZERO)

    savefig(fig, "nuts_posterior")


def main():
    fig_F1(); print("  F1 school_vs_sick")
    fig_F4(); print("  F4 rate_vs_number")
    fig_F5(); print("  F5 holiday")
    fig_F6(); print("  F6 nuts_posterior")


if __name__ == "__main__":
    main()
