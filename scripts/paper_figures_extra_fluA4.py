"""fluA4 (4-season) versions of paper_figures_extra.py's F8/F9/F10/F11/F12:
fit_byage, epicurve_byage, baseline_attack_byage, contact_matrices,
policy_intensity. F7 (fit_total) is already covered by fig_fit_total_fluA4.py
(prompts/10) and is not repeated here.

The most important fix relative to a naive port: v5's fig_F8 loaded
observations through load_hira_target_by_age directly (A+B claims), which no
longer matches the influenza-A target the fluA4 posterior was fit to. This
version loads the full-season (52-week) influenza-A-corrected series via
load_fluA_target_by_age(..., first_peak_only=False) instead.

fig_F11 (contact_matrices) has no season/posterior dependency at all -- the
matrices kt_data.load_contact_matrices returns are already the reciprocity-
corrected ones (the correction lives in the data file, not a flag), so this
is a straight re-run to pick up the corrected matrices under the new figure
directory, not an adaptation.

Fixed params (PHI/GAMMA/KAP/BASE/W/NGM_F) come from nuts_seasonpop, per
prompts/99_conventions.md -- not redefined here.
"""
from __future__ import annotations
import os, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (
    savefig, is_sig, marker_style, panel_label, zero_line,
    COL_SICK, COL_SCHOOL, COL_ZERO,
    AGES, W_SINGLE, W_DOUBLE,
)
import fluA_setup as FS
import nuts_seasonpop as NS
from kt_data import SUDOGWON_SIDO_CODES
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.fluA_target import load_fluA_target_by_age
from kt_epimodel_hira.calibration.hira_target import HIRA_AGE_GROUPS
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)

REPO = Path(__file__).resolve().parent.parent
MERGED = REPO / "outputs" / "eda" / "nuts_fluA4_raw.npz"
SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
COL_SEASON_FLUA4 = {"2016-2017": "#009E73", "2017-2018": "#CC79A7",
                     "2018-2019": "#D55E00", "2019-2020": "#56B4E9"}

PHI = np.asarray(NS.PHI); BASE = float(NS.BASE)
KAP = np.asarray(NS.KAP)
W = float(NS.W); NGM_F = float(NS.NGM_F)
TERM = (70.0, 113.0); WH = (-1e9, 1e9)

N_POST = 60
SEED = 0

_D = np.load(MERGED, allow_pickle=True)
PI_MERGED = np.asarray(_D["pi"])
LOG_R0_MERGED = np.asarray(_D["log_R0"])
PHI_NB_MERGED = np.asarray(_D["phi_nb"])


class _DispatchC:
    """Season-dispatching view over fluA_setup.build_fluA_setup's output,
    matching the interface sim_inc/att6/pred_h expect (same pattern as v5's
    paper_figures_extra.py's USE_SEASONPOP=1 path)."""
    def __init__(self, C_all):
        self._C_all = C_all
        self._current = SEASONS[0]

    def set_season(self, s):
        assert s in SEASONS
        self._current = s

    def __getitem__(self, k):
        s = self._current
        if k in ("shared",):
            return self._C_all["shared_by_s"][s]
        if k == "ngm3":
            return self._C_all["ngm3_by_s"][s]
        if k == "st":
            return {s: self._C_all["st_by_s"][s]}
        if k == "H":
            return self._C_all["H"]
        if k == "pop6":
            return self._C_all["pop6_by_s"][s]
        if k == "nw":
            return self._C_all["nw"]
        raise KeyError(k)


def build_setup():
    C_all = FS.build_fluA_setup(SEASONS, verbose=True)
    return _DispatchC(C_all)


def sim_inc(C, s, R0, pi, p_school=BASE, p_work=BASE, sch_win=WH, work_win=WH):
    C.set_season(s)
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI))
    beta = b0 / NGM_F
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                      w_presymp=W, **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st)


def att6(C, inc):
    return np.asarray(C["H"]) @ np.asarray(inc).sum(0)


def pred_h(C, inc, n_weeks):
    return np.asarray(simulation_to_hira_by_age_jax(
        jnp.asarray(inc), jnp.asarray(NS.GAMMA), n_weeks=n_weeks))


def load_obs_fluA():
    """Full-season (52-week) influenza-A-corrected observations, per season.
    The key fix vs. v5: v5's load_obs() called load_hira_target_by_age
    directly (uncorrected A+B claims), which no longer matches the fluA4
    target. This uses load_fluA_target_by_age(first_peak_only=False)."""
    obs = {}
    for s in SEASONS:
        t = load_fluA_target_by_age(s, sido_codes=list(SUDOGWON_SIDO_CODES),
                                      first_peak_only=False)
        o = np.zeros((t["n_weeks"], 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            o[:, i] = t["hira_counts"][ag]
        obs[s] = o
    return obs


def posterior_predictive(C, n_samples=N_POST, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(PI_MERGED.shape[0], size=n_samples, replace=False)
    pi_s = PI_MERGED[idx]; log_R0_s = LOG_R0_MERGED[idx]
    phi_nb_s = PHI_NB_MERGED[idx]
    preds = {s: [] for s in SEASONS}
    incs = {s: [] for s in SEASONS}
    for k in range(n_samples):
        pi_k = pi_s[k]; R0_vec = np.exp(log_R0_s[k])
        for j, s in enumerate(SEASONS):
            inc = np.asarray(sim_inc(C, s, float(R0_vec[j]), pi_k))
            pr = pred_h(C, inc, n_weeks=52)
            preds[s].append(pr); incs[s].append(inc)
        if (k + 1) % 10 == 0:
            print(f"  posterior sim {k + 1}/{n_samples}", flush=True)
    for s in SEASONS:
        preds[s] = np.stack(preds[s])
        incs[s] = np.stack(incs[s])
    return preds, incs, phi_nb_s


# ═══════════════════════════════════════════════════════════════════════════
# F8 — fit_byage (season x age grid, per-100k, influenza-A observations)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F8(C, preds, obs):
    fig, axes = plt.subplots(4, 6, figsize=(W_DOUBLE, 6.2),
                              constrained_layout=True, sharex=True, sharey=True)
    for r, s in enumerate(SEASONS):
        C.set_season(s)
        pop6 = np.asarray(C["pop6"])
        for c, ag in enumerate(AGES):
            ax = axes[r, c]
            pr = preds[s][:, :, c] / pop6[c] * 1e5
            o = obs[s][:, c] / pop6[c] * 1e5
            wks = np.arange(pr.shape[1])
            mean_pr = pr.mean(axis=0)
            lo, hi = np.quantile(pr, [0.05, 0.95], axis=0)
            col = COL_SEASON_FLUA4[s]
            ax.fill_between(wks, lo, hi, color=col, alpha=0.22)
            ax.plot(wks, mean_pr, color=col, lw=1.0)
            ax.plot(wks[:len(o)], o, "o", color=COL_ZERO, ms=1.5, alpha=0.7)
            if r == 0:
                ax.set_title(ag, fontsize=8)
            if c == 0:
                ax.set_ylabel(s, fontsize=7)
            if r == 3:
                ax.set_xlabel("Week", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.text(0.5, -0.015,
              "Per-100,000 weekly incidence, influenza-A-corrected observations. "
              "Model posterior 90% CI (band) + mean (line).",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "fit_byage")


# ═══════════════════════════════════════════════════════════════════════════
# F9 — epicurve_byage (baseline model, per-age curve, 2019-20)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F9(C, incs):
    s = "2019-2020"
    C.set_season(s)
    pop6 = np.asarray(C["pop6"])
    inc_arr = incs[s]
    H = np.asarray(C["H"])
    inc6 = np.einsum("ai,nti->nta", H, inc_arr)
    inc6_p = inc6 / pop6[None, None, :] * 1e5
    mean_c = inc6_p.mean(axis=0)
    t_days = np.arange(mean_c.shape[0])

    age_col = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#7b3294"]
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.9, 3.2), constrained_layout=True)
    for c, ag in enumerate(AGES):
        ax.plot(t_days, mean_c[:, c], color=age_col[c], lw=1.5, label=ag)
    ax.axvspan(70, 113, color=COL_SICK, alpha=0.08, zorder=0, label="Term policy window")
    ax.axvspan(113, 183, color=COL_SCHOOL, alpha=0.10, zorder=0, label="Winter break window")
    ax.set_xlabel("Season day (0 = Sep 1)")
    ax.set_ylabel("Daily incidence per 100,000 (baseline)")
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)
    fig.text(0.5, -0.03,
              f"Season {s}, no-intervention baseline (p_work=p_school={BASE}). "
              "Sequential palette by age.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "epicurve_byage")


# ═══════════════════════════════════════════════════════════════════════════
# F10 — baseline_attack_byage (4-season)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F10(C, incs):
    H = np.asarray(C["H"])
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.8, 3.0), constrained_layout=True)
    xs = np.arange(len(AGES))
    all_att = np.zeros((len(SEASONS), 6))
    for j, s in enumerate(SEASONS):
        C.set_season(s)
        pop6 = np.asarray(C["pop6"])
        inc_arr = incs[s]
        tot_15 = inc_arr.sum(axis=1)
        a6 = np.einsum("ai,ni->na", H, tot_15) / pop6[None, :]
        all_att[j] = a6.mean(axis=0)
    for j, s in enumerate(SEASONS):
        col = COL_SEASON_FLUA4[s]
        ax.plot(xs, all_att[j] * 100, "o-", color=col, lw=1.2, ms=5, label=s)
    ax.set_xticks(xs); ax.set_xticklabels(AGES)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Baseline attack rate (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    fig.text(0.5, -0.04,
              f"No-intervention baseline (p_work=p_school={BASE}), "
              "posterior mean of final attack rate.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "baseline_attack_byage")


# ═══════════════════════════════════════════════════════════════════════════
# F11 — contact_matrices (term vs winter break x 4 channels; no season/
# posterior dependency -- reciprocity correction lives in the data file)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F11():
    D_CM = REPO.parent / "kt_data/data/external/contact_matrices"
    term = load_contact_matrices(path=D_CM / "empirical_matrices_15.npz")
    vac = load_contact_matrices(path=D_CM / "empirical_matrices_15_vacation.npz")
    LAB = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
           "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
    channels = ["C_home", "C_school", "C_work", "C_other"]
    ch_names = ["Home", "School", "Work", "Other"]
    fig, axes = plt.subplots(2, 4, figsize=(W_DOUBLE, 3.6),
                              constrained_layout=True, sharey=True, sharex=True)
    vmax = {c: max(term[c].max(), vac[c].max()) for c in channels}
    for row, (m, title) in enumerate([(term, "School term"), (vac, "Winter break")]):
        for col, ch in enumerate(channels):
            ax = axes[row, col]
            im = ax.imshow(m[ch], origin="lower", cmap="viridis",
                            vmin=0, vmax=vmax[ch], aspect="equal")
            if row == 0:
                ax.set_title(ch_names[col], fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(title, fontsize=9, fontweight="bold")
            ax.set_xticks(range(0, 15, 3))
            ax.set_xticklabels([LAB[k] for k in range(0, 15, 3)], fontsize=6, rotation=45)
            ax.set_yticks(range(0, 15, 3))
            ax.set_yticklabels([LAB[k] for k in range(0, 15, 3)], fontsize=6)
            if row == 1:
                fig.colorbar(im, ax=[axes[0, col], axes[1, col]],
                              fraction=0.04, pad=0.02, shrink=0.9)
    fig.text(0.5, -0.03,
              "Contact matrices [contacted, participant]. School channel "
              "collapses in winter break. Reciprocity-corrected (Method A).",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "contact_matrices")


# ═══════════════════════════════════════════════════════════════════════════
# F12 — policy_intensity sweep (2019-20 representative, term window)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F12(C, n_post=30):
    s = "2019-2020"; j_s = SEASONS.index(s)
    C.set_season(s)
    p_grid = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    rng = np.random.default_rng(SEED + 1)
    idx = rng.choice(PI_MERGED.shape[0], size=n_post, replace=False)
    pi_s = PI_MERGED[idx]; log_R0_s = LOG_R0_MERGED[idx]

    base_tot = np.zeros(n_post)
    for k in range(n_post):
        inc = sim_inc(C, s, float(np.exp(log_R0_s[k, j_s])), pi_s[k])
        base_tot[k] = float(att6(C, inc).sum())
    print("  F12 baseline done", flush=True)

    sk_pct = np.zeros((n_post, len(p_grid)))
    sc_pct = np.zeros((n_post, len(p_grid)))
    for pi_i, p in enumerate(p_grid):
        for k in range(n_post):
            R0 = float(np.exp(log_R0_s[k, j_s])); pi_k = pi_s[k]
            inc_sk = sim_inc(C, s, R0, pi_k, p_work=p, work_win=TERM)
            inc_sc = sim_inc(C, s, R0, pi_k, p_school=p, sch_win=TERM)
            sk_pct[k, pi_i] = 100.0 * (base_tot[k] - float(att6(C, inc_sk).sum())) / max(base_tot[k], 1)
            sc_pct[k, pi_i] = 100.0 * (base_tot[k] - float(att6(C, inc_sc).sum())) / max(base_tot[k], 1)
        print(f"  F12 p={p} done", flush=True)

    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.85, 3.4), constrained_layout=True)
    xs = np.array([BASE - p for p in p_grid])
    for arr, col, label in [(sk_pct, COL_SICK, "Sick leave"),
                              (sc_pct, COL_SCHOOL, "School absence")]:
        mean_ = arr.mean(axis=0)
        lo, hi = np.quantile(arr, [0.05, 0.95], axis=0)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.20)
        ax.plot(xs, mean_, "o-", color=col, lw=1.4, ms=4, label=label)
    zero_line(ax)
    ax.set_xlabel(r"Intensity $= (\mathrm{baseline} - p)$, baseline$= 0.6$")
    ax.set_ylabel("Averted infections (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.text(0.5, -0.04,
              f"Season {s}, term window. Band = 90% CI from N={n_post} "
              "posterior samples.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "policy_intensity")


def main():
    print("=" * 90)
    print(f"fluA4 F8/F9/F10/F11/F12  (posterior subsample N={N_POST})")
    print("=" * 90)
    t0 = time.perf_counter()
    C = build_setup()
    print(f"[setup] {time.perf_counter() - t0:.1f}s")
    obs = load_obs_fluA(); print("  obs loaded (4 seasons x 52 wk x 6 age, influenza-A)")

    print("\n[posterior predictive]")
    preds, incs, phi_nb_s = posterior_predictive(C, n_samples=N_POST)

    fig_F8(C, preds, obs); print("  F8 fit_byage")
    fig_F9(C, incs); print("  F9 epicurve_byage")
    fig_F10(C, incs); print("  F10 baseline_attack_byage")
    fig_F11(); print("  F11 contact_matrices")
    fig_F12(C, n_post=30); print("  F12 policy_intensity")

    print(f"\n[total wall] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
