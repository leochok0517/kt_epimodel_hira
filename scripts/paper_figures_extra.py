"""논문 그림 F7-F12 (권장·선택).

F7 fit_total     : 시즌별 관측 vs 모델 (총계 + posterior CI band)
F8 fit_byage     : 시즌×연령 grid (관측 vs 모델 per-100k)
F9 epicurve_byage: baseline 모델 연령별 유행곡선 (2019-20)
F10 baseline_attack_byage: baseline attack rate per age (3시즌 평균±범위)
F11 contact_matrices: 학기 vs 방학 접촉행렬 heatmap (4채널)
F12 policy_intensity: p_work sweep + CI band (병가 vs 학교)

Forward sim: posterior 4000 draws 중 N=60 subsample 사용 (속도).
figstyle 공통 스타일 재사용.
"""
from __future__ import annotations
import os, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (
    savefig, is_sig, marker_style, panel_label, zero_line,
    COL_SICK, COL_SCHOOL, COL_SEASON, COL_ZERO,
    AGES, SEASONS, CHANNELS,
    W_SINGLE, W_DOUBLE,
)

# ── v4 forward sim primitives (재사용) ──
from kt_data import SUDOGWON_SIDO_CODES
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.calibration.hira_target import (
    HIRA_AGE_GROUPS, HIRA_GROUP_TO_NIMS_WEIGHTED, load_hira_target_by_age,
)
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
import final_pipeline_confirmed as F

REPO = Path(__file__).resolve().parent.parent
NUTS_RAW = np.load(REPO / "outputs/eda/nuts_v4_full_raw.npz")
NUTS_EXT = np.load(REPO / "outputs/eda/nuts_v4_full_extended.npz")
PI_MERGED = np.concatenate([NUTS_RAW["pi"], NUTS_EXT["pi"]], axis=0)
LOG_R0_MERGED = np.concatenate([NUTS_RAW["log_R0"], NUTS_EXT["log_R0"]], axis=0)

PHI = np.array(F.PHI); BASE = 0.6
GAMMA_15 = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM = np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)
KAP = np.array([0.34]*4+[0.40]*10+[0.0])
W = W_PRESYMP; NGM_F = ngm_factor(W)
TERM = (70.0, 113.0); WH = (-1e9, 1e9)

N_POST = 60         # posterior subsample (속도 조절)
SEED = 0


def build_setup():
    C = F.build()
    pf = np.asarray(C["shared"]["pop_15"])
    C["pf"] = pf.sum(1) if pf.ndim == 2 else pf
    M = C["shared"]
    C["ngm3"] = make_ngm_eigvalue_fn(
        pop_15=np.asarray(M["pop_15"]), rho=np.asarray(M["rho"]),
        C_home=np.asarray(M["C_home"]), C_work=np.asarray(M["C_work"]),
        C_school=np.asarray(M["C_school"]), C_other=np.asarray(M["C_other"]),
        R0_immunity=IMM, gamma=float(M["gamma"]), seasonal_factor=1.0+F.S.AMP)
    C["st"] = {}
    for s in SEASONS:
        sd = estimate_initial_infected_from_hira(s, C["pf"],
            sido_codes=list(SUDOGWON_SIDO_CODES), gamma_15_assumed=GAMMA_15)
        C["st"][s] = jnp.asarray(_build_initial_state_with_age_seed(
            C["pf"], sd, seed_e_factor=0.5, initial_immunity=IMM,
            initial_vaccinated_fraction=0.0))
    return C


def sim_inc(C, s, R0, pi, p_school=BASE, p_work=BASE, sch_win=WH, work_win=WH):
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
    return C["H"] @ np.asarray(inc).sum(0)


def pred_h(C, inc, n_weeks):
    return np.asarray(simulation_to_hira_by_age_jax(
        jnp.asarray(inc), jnp.asarray(GAMMA_15), n_weeks=n_weeks))


def load_obs(C):
    """전체 시즌 관측 HIRA (52주, 6연령)."""
    obs = {}
    for s in SEASONS:
        t = load_hira_target_by_age(s, sido_codes=list(SUDOGWON_SIDO_CODES),
                                     first_peak_only=False)
        o = np.zeros((t["n_weeks"], 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            o[:, i] = t["hira_counts"][ag]
        obs[s] = o
    return obs


# ═══════════════════════════════════════════════════════════════════════════
# 사후 예측 (F7, F8, F9)
# ═══════════════════════════════════════════════════════════════════════════
def posterior_predictive(C, n_samples=N_POST, seed=SEED):
    """N 표본에 대해 forward sim → HIRA 시계열.

    Returns dict[s] = ndarray (N, 52, 6) HIRA weekly counts.
    Also inc_15 dict[s] = (N, T-1, 15) daily infections for epicurve.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(PI_MERGED.shape[0], size=n_samples, replace=False)
    pi_s = PI_MERGED[idx]; log_R0_s = LOG_R0_MERGED[idx]
    preds = {s: [] for s in SEASONS}
    incs = {s: [] for s in SEASONS}
    n_weeks_by_s = {}
    for k in range(n_samples):
        pi_k = pi_s[k]; R0_vec = np.exp(log_R0_s[k])
        for j, s in enumerate(SEASONS):
            inc = np.asarray(sim_inc(C, s, float(R0_vec[j]), pi_k))
            # 52 주 pad (전체 시즌)
            pr = pred_h(C, inc, n_weeks=52)
            preds[s].append(pr); incs[s].append(inc)
            n_weeks_by_s[s] = pr.shape[0]
        if (k+1) % 10 == 0:
            print(f"  posterior sim {k+1}/{n_samples}")
    for s in SEASONS:
        preds[s] = np.stack(preds[s])   # (N, 52, 6)
        incs[s] = np.stack(incs[s])     # (N, T-1, 15)
    return preds, incs


# ═══════════════════════════════════════════════════════════════════════════
# F7 — fit_total (시즌별 관측 vs 모델 총계)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F7(preds, obs):
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.6),
                              constrained_layout=True, sharey=False)
    for k, s in enumerate(SEASONS):
        ax = axes[k]
        pr = preds[s].sum(axis=2)          # (N, 52)
        o = obs[s].sum(axis=1)              # (52,)
        wks = np.arange(pr.shape[1])
        mean_pr = pr.mean(axis=0)
        lo, hi = np.quantile(pr, [0.05, 0.95], axis=0)
        col = COL_SEASON[s]
        ax.fill_between(wks, lo, hi, color=col, alpha=0.25,
                         label="Model 90% CI" if k == 0 else None)
        ax.plot(wks, mean_pr, color=col, lw=1.6,
                label="Model mean" if k == 0 else None)
        ax.plot(wks[:len(o)], o, "o", color=COL_ZERO, ms=2.5, alpha=0.75,
                label="Observed" if k == 0 else None)
        ax.set_xlabel("Epidemic week")
        if k == 0:
            ax.set_ylabel("Weekly incidence (HIRA)")
        ax.text(0.03, 0.95, s, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="left", color=col)
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.text(0.5, -0.06,
              "Observation timing = symptom onset (I₁→I₂ influx).",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "fit_total")


# ═══════════════════════════════════════════════════════════════════════════
# F8 — fit_byage (시즌 × 연령 grid, per-100k)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F8(C, preds, obs):
    # pop6 per age (HIRA-6)
    pop6 = np.asarray(C["pop6"])
    fig, axes = plt.subplots(3, 6, figsize=(W_DOUBLE, 4.8),
                              constrained_layout=True, sharex=True, sharey=True)
    for r, s in enumerate(SEASONS):
        for c, ag in enumerate(AGES):
            ax = axes[r, c]
            pr = preds[s][:, :, c] / pop6[c] * 1e5    # per-100k
            o = obs[s][:, c] / pop6[c] * 1e5
            wks = np.arange(pr.shape[1])
            mean_pr = pr.mean(axis=0)
            lo, hi = np.quantile(pr, [0.05, 0.95], axis=0)
            col = COL_SEASON[s]
            ax.fill_between(wks, lo, hi, color=col, alpha=0.22)
            ax.plot(wks, mean_pr, color=col, lw=1.0)
            ax.plot(wks[:len(o)], o, "o", color=COL_ZERO, ms=1.5, alpha=0.7)
            if r == 0:
                ax.set_title(ag, fontsize=8)
            if c == 0:
                ax.set_ylabel(s, fontsize=7)
            if r == 2:
                ax.set_xlabel("Week", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.text(0.5, -0.02,
              "Per-100,000 weekly incidence. Model posterior 90% CI (band) + "
              "mean (line). Observation = symptom onset.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "fit_byage")


# ═══════════════════════════════════════════════════════════════════════════
# F9 — epicurve_byage (baseline 모델, 연령별 curve, 2019-20)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F9(C, incs):
    s = "2019-2020"
    pop6 = np.asarray(C["pop6"])
    inc_arr = incs[s]                          # (N, T-1, 15)
    # H matrix aggregate to HIRA-6 then per-day
    H = np.asarray(C["H"])                      # (6, 15)
    inc6 = np.einsum("ai,nti->nta", H, inc_arr)   # (N, T-1, 6) daily infections
    # per-100k
    inc6_p = inc6 / pop6[None, None, :] * 1e5   # per-100k/day
    mean_c = inc6_p.mean(axis=0)                 # (T-1, 6)
    t_days = np.arange(mean_c.shape[0])

    # Age color palette: sequential (child→adult→elder)
    age_col = ["#4575b4","#74add1","#fdae61","#f46d43","#d73027","#7b3294"]

    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.9, 3.2),
                            constrained_layout=True)
    for c, ag in enumerate(AGES):
        ax.plot(t_days, mean_c[:, c], color=age_col[c], lw=1.5, label=ag)
    # 학기/방학 창 음영
    ax.axvspan(70, 113, color=COL_SICK, alpha=0.08, zorder=0,
                label="Term policy window")
    ax.axvspan(113, 183, color=COL_SCHOOL, alpha=0.10, zorder=0,
                label="Winter break window")
    ax.set_xlabel("Season day (0 = Sep 1)")
    ax.set_ylabel("Daily incidence per 100,000 (baseline)")
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)
    fig.text(0.5, -0.03,
              f"Season {s}, no-intervention baseline (p_work=p_school={BASE}). "
              "Sequential palette by age.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "epicurve_byage")


# ═══════════════════════════════════════════════════════════════════════════
# F10 — baseline_attack_byage (baseline final attack rate, 3시즌 평균±범위)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F10(C, incs):
    pop6 = np.asarray(C["pop6"])
    H = np.asarray(C["H"])
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.75, 3.0),
                            constrained_layout=True)
    xs = np.arange(len(AGES))
    all_att = np.zeros((len(SEASONS), 6))       # (3, 6) attack rate posterior mean
    for j, s in enumerate(SEASONS):
        inc_arr = incs[s]                        # (N, T-1, 15)
        tot_15 = inc_arr.sum(axis=1)              # (N, 15)
        att6 = np.einsum("ai,ni->na", H, tot_15) / pop6[None, :]  # (N, 6)
        all_att[j] = att6.mean(axis=0)
    for j, s in enumerate(SEASONS):
        col = COL_SEASON[s]
        ax.plot(xs, all_att[j] * 100, "o-", color=col, lw=1.2, ms=5,
                label=s)
    ax.set_xticks(xs); ax.set_xticklabels(AGES)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Baseline attack rate (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.text(0.5, -0.04,
              f"No-intervention baseline (p_work=p_school={BASE}), "
              "posterior mean of final attack rate.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "baseline_attack_byage")


# ═══════════════════════════════════════════════════════════════════════════
# F11 — contact_matrices (학기 vs 방학 × 4채널)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F11():
    from matplotlib.colors import Normalize
    D_CM = REPO.parent / "kt_data/data/external/contact_matrices"
    term = load_contact_matrices(path=D_CM / "empirical_matrices_15.npz")
    vac = load_contact_matrices(path=D_CM / "empirical_matrices_15_vacation.npz")
    LAB = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
            "40-44","45-49","50-54","55-59","60-64","65-69","70+"]
    channels = ["C_home", "C_school", "C_work", "C_other"]
    ch_names = ["Home", "School", "Work", "Other"]
    fig, axes = plt.subplots(2, 4, figsize=(W_DOUBLE, 3.6),
                              constrained_layout=True, sharey=True, sharex=True)
    # 공통 vmax per 채널
    vmax = {c: max(term[c].max(), vac[c].max()) for c in channels}
    for row, (m, title) in enumerate([(term, "Term"), (vac, "Vacation")]):
        for col, ch in enumerate(channels):
            ax = axes[row, col]
            im = ax.imshow(m[ch], origin="lower", cmap="viridis",
                            vmin=0, vmax=vmax[ch], aspect="equal")
            if row == 0:
                ax.set_title(ch_names[col], fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(title, fontsize=9, fontweight="bold")
            ax.set_xticks(range(0, 15, 3))
            ax.set_xticklabels([LAB[k] for k in range(0, 15, 3)], fontsize=6,
                                rotation=45)
            ax.set_yticks(range(0, 15, 3))
            ax.set_yticklabels([LAB[k] for k in range(0, 15, 3)], fontsize=6)
            # colorbar per column (아래줄 오른쪽)
            if row == 1:
                fig.colorbar(im, ax=[axes[0, col], axes[1, col]],
                              fraction=0.04, pad=0.02, shrink=0.9)
    fig.text(0.5, -0.03,
              "Contact matrices [contacted, participant]. School channel "
              "collapses in vacation.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "contact_matrices")


# ═══════════════════════════════════════════════════════════════════════════
# F12 — policy_intensity sweep (병가/학교 averted % vs intensity, CI band)
# ═══════════════════════════════════════════════════════════════════════════
def fig_F12(C, n_post=30):
    """대표 시즌 2019-20, p_work·p_school ∈ {0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0},
    posterior N 표본에 대해 averted % 분포 산출."""
    s = "2019-2020"; j_s = SEASONS.index(s)
    p_grid = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    rng = np.random.default_rng(SEED + 1)
    idx = rng.choice(PI_MERGED.shape[0], size=n_post, replace=False)
    pi_s = PI_MERGED[idx]; log_R0_s = LOG_R0_MERGED[idx]

    # Baseline totals per sample
    base_tot = np.zeros(n_post)
    for k in range(n_post):
        inc = sim_inc(C, s, float(np.exp(log_R0_s[k, j_s])), pi_s[k])
        base_tot[k] = float(att6(C, inc).sum())
    print(f"  F12 baseline done")

    sk_pct = np.zeros((n_post, len(p_grid)))
    sc_pct = np.zeros((n_post, len(p_grid)))
    for pi_i, p in enumerate(p_grid):
        for k in range(n_post):
            R0 = float(np.exp(log_R0_s[k, j_s])); pi_k = pi_s[k]
            inc_sk = sim_inc(C, s, R0, pi_k, p_work=p, work_win=TERM)
            inc_sc = sim_inc(C, s, R0, pi_k, p_school=p, sch_win=TERM)
            sk_pct[k, pi_i] = 100.0 * (base_tot[k] - float(att6(C, inc_sk).sum())) / max(base_tot[k], 1)
            sc_pct[k, pi_i] = 100.0 * (base_tot[k] - float(att6(C, inc_sc).sum())) / max(base_tot[k], 1)
        print(f"  F12 p={p} done")

    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.85, 3.4),
                            constrained_layout=True)
    xs = np.array([BASE - p for p in p_grid])  # intensity = 1 - p_eff (BASE - p)
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
    print("="*90)
    print(f"F7-F12 생성  (posterior subsample N={N_POST} for F7-F10)")
    print("="*90)
    t0 = time.perf_counter()
    C = build_setup()
    print(f"[setup] {time.perf_counter()-t0:.1f}s")
    obs = load_obs(C); print("  obs loaded (3 seasons × 52 wk × 6 age)")

    print("\n[posterior predictive]")
    preds, incs = posterior_predictive(C, n_samples=N_POST)

    fig_F7(preds, obs); print("  F7 fit_total")
    fig_F8(C, preds, obs); print("  F8 fit_byage")
    fig_F9(C, incs); print("  F9 epicurve_byage")
    fig_F10(C, incs); print("  F10 baseline_attack_byage")
    fig_F11(); print("  F11 contact_matrices")
    fig_F12(C, n_post=30); print("  F12 policy_intensity")

    print(f"\n[total wall] {time.perf_counter()-t0:.1f}s")
    print("완료. figures/paper/{pdf,png}/ 확인.")


if __name__ == "__main__":
    main()
