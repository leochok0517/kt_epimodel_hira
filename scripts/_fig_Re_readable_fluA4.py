"""R_e(t) readability figure for fluA4, all 4 seasons
(reproduction_numbers_fluA4_<season>.json, no recomputation).
Top: 3-scenario R_e(t) (start/peak/end marked) / Bottom: baseline channel
components. Copy of _fig_Re_readable.py with the fit-curve overlay pointed
at fit_total_curves_fluA4.json and the advisory/lifted/peak-day annotations
left off for 2018-2019, which v5 never had KDCA advisory dates recorded for.
"""
import os, json, sys, glob
from pathlib import Path
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from figstyle import savefig, COL_SICK, COL_SCHOOL, COL_ZERO, W_DOUBLE

FITDATA_PATH = REPO / "outputs/fluA/fit_total_curves_fluA4.json"
FITDATA = json.load(open(FITDATA_PATH)) if FITDATA_PATH.exists() else {}
# Channel colours match Figure 4A (paper_figures_fluA4.COL_CHANNEL), co-author J. Lee 2026-10-28.
CH_COL = {"home": "#0072B2", "work": "#E69F00", "school": "#009E73", "other": "#999999"}
SC = {"baseline": (COL_ZERO, "Baseline"),
      "sick": (COL_SICK, "Sick leave"),
      "school": (COL_SCHOOL, "School absence")}


def make_fig(json_path, out_name):
    D = json.load(open(json_path))
    t = np.array(D["Re"]["t_days"]); Re = D["Re"]["Re"]; Rec = D["Re"]["Re_channel"]
    meta = D.get("meta", {})
    season = meta.get("season", D.get("static", {}).get("season", "season"))
    BREAK = tuple(meta.get("break_window", (113, 183)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_DOUBLE, 5.6),
                                    constrained_layout=True, sharex=True)
    # ---- Panel A ----
    ax1.axhline(1.0, color="black", lw=0.9, ls=":", zorder=3)
    ax1.axvspan(*BREAK, color="#4C72B0", alpha=0.07, lw=0)
    fc = FITDATA.get(season)
    ax1r = ax1.twinx()
    if fc is not None:
        wk = np.arange(len(fc["mean"])) * 7.0
        ym = np.asarray(fc["mean"], float)
        try:
            from scipy.interpolate import PchipInterpolator
            xd = t[(t >= wk[0]) & (t <= wk[-1])]
            yd = PchipInterpolator(wk, ym)(xd)
        except Exception:
            xd, yd = wk, ym
        ax1r.plot(xd, yd, color="#B0B0B0", lw=1.6, zorder=1, label="Model incidence")
        ax1r.set_ylim(0, ym.max() * 1.1)
    ax1r.set_ylabel("Weekly incidence (influenza A)", color="#7A7A7A")
    ax1r.tick_params(axis="y", colors="#7A7A7A")
    for sc, (col, lab) in SC.items():
        m = np.array(Re[sc]["mean"]); lo = np.array(Re[sc]["q05"]); hi = np.array(Re[sc]["q95"])
        ax1.fill_between(t, lo, hi, color=col, alpha=0.15, lw=0)
        ax1.plot(t, m, color=col, lw=2.0, label=lab)
    ax1.set_zorder(ax1r.get_zorder() + 1); ax1.patch.set_visible(False)
    ymax = max(max(Re[sc]["q95"]) for sc in SC)
    ylim_top = ymax * 1.10
    ytop = ylim_top * 0.97
    ylow = ylim_top * 0.42
    _dref = np.abs(np.array(Re["baseline"]["mean"]) - np.array(Re["school"]["mean"]))
    _di = int(np.argmax(_dref > 1e-4)) if bool((_dref > 1e-4).any()) else 0
    pol_start = float(t[max(_di - 1, 0)])
    pol_end = BREAK[1]
    pcol = "#333333"
    ax1.axvline(pol_start, color=pcol, lw=1.1, ls="--", zorder=4)
    ax1.axvline(pol_end, color=pcol, lw=1.1, ls="--", zorder=4)
    ax1.annotate("policy applied", (pol_start, ylow), textcoords="offset points",
                 xytext=(-3, 0), ha="right", va="center", fontsize=6.8,
                 color=pcol, fontweight="bold")
    ax1.annotate("policy ends", (pol_end, ylow), textcoords="offset points",
                 xytext=(4, 0), ha="left", va="center", fontsize=6.8,
                 color=pcol, fontweight="bold")
    ax1.text(np.mean(BREAK), ytop, "winter break", ha="center", va="top",
             fontsize=7.5, color=COL_ZERO)
    ax1.set_ylabel(r"Effective $R_e(t)$")
    ax1.set_ylim(0, ylim_top)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1r.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax1.set_title(rf"$R_e(t)$ and model incidence by scenario ({season})",
                  fontsize=10, fontweight="bold")
    # ---- Panel B ----
    base = Rec["baseline"]
    for ch in ["home", "school", "other", "work"]:
        ax2.plot(t, np.array(base[ch]["mean"]), color=CH_COL[ch], lw=1.8, label=ch)
    ax2.axvspan(*BREAK, color="#4C72B0", alpha=0.07, lw=0)
    ax2.set_xlabel("Season day (0 = Sep 1)")
    ax2.set_ylabel(r"Channel $R_e^{c}(t)$")
    ax2.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    ax2.set_title("Baseline channel components: the school channel falls in the winter-break period",
                  fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 290)
    savefig(fig, out_name)
    print(f"saved {out_name}  ({season}: policy window {pol_start:.0f}-{pol_end:.0f})")


if __name__ == "__main__":
    per_season = sorted(glob.glob(str(REPO / "outputs/fluA/reproduction_numbers_fluA4_*.json")))
    if not per_season:
        raise SystemExit("no outputs/fluA/reproduction_numbers_fluA4_*.json found - run "
                          "reproduction_analysis_fluA4.py first")
    for jp in per_season:
        tag = Path(jp).stem.replace("reproduction_numbers_fluA4_", "")
        make_fig(jp, f"repro_Re_readable_{tag}")
