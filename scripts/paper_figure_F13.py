"""F13 sickleave_term_vs_vacation — 병가 정책 학기창 vs 방학창 (전 연령).

Panel A: 연령별 Δattack (%p), 학기창 vs 방학창 side-by-side (2019-20 대표).
Panel B: 총 averted %, 학기 vs 방학 (3시즌 겹쳐 방향 일관성).

Data: posterior N 표본 (병합 4000) → 창별 forward sim → CI.
* policy_posterior_v4.json 은 term-only, vacation window 는 재시뮬 필요.
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
    AGES, SEASONS, W_DOUBLE,
)
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax,
)
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
TERM = (70.0, 113.0); VAC = (113.0, 183.0); WH = (-1e9, 1e9)
P_POL = 0.4
N_POST = 50
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


def sim_inc(C, s, R0, pi, work_win=WH):
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI))
    beta = b0 / NGM_F
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = BASE; kw["p_work"] = P_POL
    kw["policy_school_start_day"], kw["policy_school_end_day"] = WH
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                      w_presymp=W, **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st)


def sim_base(C, s, R0, pi):
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI))
    beta = b0 / NGM_F
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = BASE; kw["p_work"] = BASE
    kw["policy_school_start_day"], kw["policy_school_end_day"] = WH
    kw["policy_work_start_day"], kw["policy_work_end_day"] = WH
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                      w_presymp=W, **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st)


def att6(C, inc):
    return C["H"] @ np.asarray(inc).sum(0)


def main():
    print("="*88); print(f"F13 sickleave_term_vs_vacation  N={N_POST}"); print("="*88)
    t0 = time.perf_counter()
    C = build_setup(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
    pop6 = np.asarray(C["pop6"])

    rng = np.random.default_rng(SEED)
    idx = rng.choice(PI_MERGED.shape[0], size=N_POST, replace=False)
    pi_s = PI_MERGED[idx]; log_R0_s = LOG_R0_MERGED[idx]

    # 산출: {s: {"term": (N,6) Δattack, "vac": (N,6), "term_tot": (N,), "vac_tot": (N,)}}
    results = {s: dict(term_d=np.zeros((N_POST, 6)),
                        vac_d=np.zeros((N_POST, 6)),
                        term_av=np.zeros(N_POST),
                        vac_av=np.zeros(N_POST)) for s in SEASONS}
    for k in range(N_POST):
        for j, s in enumerate(SEASONS):
            R0 = float(np.exp(log_R0_s[k, j])); pi_k = pi_s[k]
            b_inc = sim_base(C, s, R0, pi_k); b6 = att6(C, b_inc)
            t_inc = sim_inc(C, s, R0, pi_k, work_win=TERM); t6 = att6(C, t_inc)
            v_inc = sim_inc(C, s, R0, pi_k, work_win=VAC); v6 = att6(C, v_inc)
            tot_b = float(b6.sum())
            results[s]["term_av"][k] = 100.0 * (tot_b - float(t6.sum())) / max(tot_b, 1)
            results[s]["vac_av"][k] = 100.0 * (tot_b - float(v6.sum())) / max(tot_b, 1)
            results[s]["term_d"][k] = (t6 - b6) / pop6 * 100.0
            results[s]["vac_d"][k] = (v6 - b6) / pop6 * 100.0
        if (k+1) % 10 == 0:
            print(f"  sim {k+1}/{N_POST}")
    print(f"  sim done  ({time.perf_counter()-t0:.1f}s)")

    # ── 그림 ──
    S_REP = "2019-2020"
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.4),
                              constrained_layout=True,
                              gridspec_kw=dict(width_ratios=[1.4, 1.0]))

    # Panel A: 연령별 Δattack, term vs vac (2019-20 posterior mean + 90% CI)
    ax = axes[0]
    xs = np.arange(len(AGES)); dx = 0.18
    for kind, arr, col, ls_label in [("term", "term_d", COL_SICK, "Term window"),
                                       ("vac", "vac_d", COL_SICK, "Vacation window")]:
        d = results[S_REP][arr]
        m = d.mean(axis=0)
        lo = np.quantile(d, 0.05, axis=0); hi = np.quantile(d, 0.95, axis=0)
        offs = -dx if kind == "term" else +dx
        # term vs vacation: 창 구분은 marker style 로 (term = 실선 marker, vac = hatched-like)
        marker = "o" if kind == "term" else "s"
        # 유의성 판정 per age
        for i, ag in enumerate(AGES):
            sig = lo[i] * hi[i] > 0
            yerr = [[m[i] - lo[i]], [hi[i] - m[i]]]
            ax.errorbar([xs[i] + offs], [m[i]], yerr=yerr,
                        color=col, lw=1.0, capsize=2, zorder=3,
                        marker=marker, markersize=6,
                        markerfacecolor=col if sig else "white",
                        markeredgecolor=col, markeredgewidth=1.2)
        ax.plot([], [], color=col, marker=marker, ls="",
                label=ls_label, markersize=6)
    zero_line(ax)
    ax.set_xticks(xs); ax.set_xticklabels(AGES, fontsize=8)
    ax.set_xlabel("Age group")
    ax.set_ylabel(r"$\Delta$ attack rate (%p, sick leave)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, fontsize=8)
    panel_label(ax, "A")

    # Panel B: 총 averted %, 학기 vs 방학 (3시즌)
    ax = axes[1]
    x_positions = np.array([0.0, 1.0])  # term, vacation
    for s in SEASONS:
        col = COL_SEASON[s]
        for i, key in enumerate(("term_av", "vac_av")):
            arr = results[s][key]
            m = arr.mean(); lo = np.quantile(arr, 0.05); hi = np.quantile(arr, 0.95)
            sig = lo * hi > 0
            yerr = [[m - lo], [hi - m]]
            ax.errorbar([x_positions[i]], [m], yerr=yerr,
                        color=col, lw=1.0, capsize=2, zorder=3,
                        marker="o", markersize=6,
                        markerfacecolor=col if sig else "white",
                        markeredgecolor=col, markeredgewidth=1.2)
        # 시즌 라인 연결
        m_term = results[s]["term_av"].mean()
        m_vac = results[s]["vac_av"].mean()
        ax.plot(x_positions, [m_term, m_vac], color=col, lw=1.0, alpha=0.6,
                zorder=2, label=s)
    zero_line(ax)
    ax.set_xticks(x_positions); ax.set_xticklabels(["Term", "Vacation"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Total averted infections (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False, fontsize=8)
    panel_label(ax, "B")

    fig.text(0.5, -0.08,
              f"Sick leave (p_work $=$ {P_POL}) in term window [70, 113] "
              "vs vacation window [113, 183]. Filled: $0\\notin$ 90% CI. "
              "Window lengths differ (43 vs 70 d) — do not compare total "
              "absolute magnitude across windows.",
              ha="center", fontsize=7, color=COL_ZERO)
    savefig(fig, "sickleave_term_vs_vacation")
    print(f"  F13 saved  [total wall {time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
