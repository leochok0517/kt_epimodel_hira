"""Work channel identifiability — presymp 전후 자유추정 + NLL profile.

두 구조 병렬 진단:
  A) NO_PRESYMP: kappa_no_eta 와 동일 (erlang.py, daily_new_infection)
  B) PRESYMP   : kappa_no_eta_presymp (erlang_presymp.py, daily_new_onset, NGM factor)

각 시즌 (16-17, 17-18, 19-20) 에 대해:
  [1] free-fit: pin 완전 해제 (5-D free: logR0, logit_pi[4], phi_nb)
      multistart 16, π_work init 0.05~0.50 균등 분산
      → 수렴 π_work·β_work, railing 여부, multistart std
  [2] NLL profile: π_work 를 [0.05, 0.10, ..., 0.50] 훑으며 나머지 fit → NLL(π_work)
      곡률 비교 (평평 → 비식별, 뚜렷한 최소 → 식별)

기존 파일 무수정. 출력만.
"""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np, jax; jax.config.update("jax_enable_x64",True); jax.devices()
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":9})

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)
from kt_epimodel_hira.jax_model.erlang import (
    simulate_jax_erlang, daily_new_infection_by_age_erlang, split_seed_to_erlang,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    ngm_factor, W_PRESYMP,
)
import final_pipeline_confirmed as F

ED = Path(__file__).resolve().parent.parent / "outputs" / "eda"
FIG = Path(__file__).resolve().parent.parent / "presentations" / "figures"
SEAS = ["2016-2017", "2017-2018", "2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEAS]

PHI = np.array(F.PHI); BASE = 0.6
GAMMA = np.array([0.40,0.40,0.25,0.18] + [0.18]*9 + [0.25,0.25])
IMM = np.array([0.10]*4 + [0.40]*5 + [0.60]*4 + [0.65]*2)
KAP = np.array([0.34]*4 + [0.40]*10 + [0.0])
W = W_PRESYMP; NGM_F = ngm_factor(W)

LOG_R0_B = F.LOG_R0_B; PHI_NB_B = F.PHI_NB_B
LOGIT_B = (-10.0, 10.0)

# Railing 판정: logit_pi 값이 경계 근처 (|logit|>7 → π<0.001 or π>0.999) OR π_work<0.005 or >0.85
RAIL_LO = 0.005; RAIL_HI = 0.85

# π_work sweep grid
PI_WORK_GRID = np.array([0.03, 0.05, 0.08, 0.12, 0.17, 0.22, 0.29, 0.36, 0.44, 0.55])


def build():
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
    for s in SEAS:
        sd = estimate_initial_infected_from_hira(s, C["pf"],
            sido_codes=list(SUDOGWON_SIDO_CODES), gamma_15_assumed=GAMMA)
        C["st"][s] = jnp.asarray(_build_initial_state_with_age_seed(
            C["pf"], sd, seed_e_factor=0.5, initial_immunity=IMM,
            initial_vaccinated_fraction=0.0))
    return C


def sim_no_presymp(C, s, R0, pi):
    beta = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                        jnp.asarray(pi), jnp.asarray(PHI))
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = BASE; kw["p_work"] = BASE
    st = simulate_jax_erlang(split_seed_to_erlang(C["st"][s]), **kw,
                              discretize_time=False)
    return daily_new_infection_by_age_erlang(st), beta


def sim_presymp(C, s, R0, pi):
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI))
    beta = b0 / NGM_F
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = BASE; kw["p_work"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                      w_presymp=W, **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st), beta


def build_loss_free(C, s, i, mode):
    """5-D free (no pin): [logR0, logit_pi(4), phi_nb]."""
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i])
    def loss(x):
        R0 = jnp.exp(x[0]); pi = jax.nn.softmax(x[1:5])
        if mode == "presymp":
            b0 = derive_beta_from_R0_simplex(C["ngm3"], R0, pi, jnp.asarray(PHI))
            beta = b0 / NGM_F
        else:
            beta = derive_beta_from_R0_simplex(C["ngm3"], R0, pi, jnp.asarray(PHI))
        kw = dict(C["shared"])
        kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
        kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
        kw["p_school"] = BASE; kw["p_work"] = BASE
        if mode == "presymp":
            st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                              w_presymp=W, **kw,
                                              discretize_time=False)
            inc = daily_new_onset_by_age_erlang_presymp(st)
        else:
            st = simulate_jax_erlang(split_seed_to_erlang(C["st"][s]), **kw,
                                      discretize_time=False)
            inc = daily_new_infection_by_age_erlang(st)
        pred = simulation_to_hira_by_age_jax(inc, jnp.asarray(GAMMA), n_weeks=C["nw"])
        return nb_nll_jax(obsj, pred, wj, concentration=x[5], min_rate=0.01)
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v):
            v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    return fg


def build_loss_fixed_piw(C, s, i, mode, pi_work_fixed):
    """4-D (π_work 고정): [logR0, u_h, u_s, u_o, phi_nb].
    π = [π_h, π_w, π_s, π_o] with π_w=fixed and softmax_3(u)·(1-π_w) for others."""
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i])
    piw = float(pi_work_fixed)
    scale = 1.0 - piw
    def loss(x):
        R0 = jnp.exp(x[0])
        u = x[1:4]
        s3 = jax.nn.softmax(u) * scale
        pi = jnp.array([s3[0], piw, s3[1], s3[2]])
        if mode == "presymp":
            b0 = derive_beta_from_R0_simplex(C["ngm3"], R0, pi, jnp.asarray(PHI))
            beta = b0 / NGM_F
        else:
            beta = derive_beta_from_R0_simplex(C["ngm3"], R0, pi, jnp.asarray(PHI))
        kw = dict(C["shared"])
        kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
        kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
        kw["p_school"] = BASE; kw["p_work"] = BASE
        if mode == "presymp":
            st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                              w_presymp=W, **kw,
                                              discretize_time=False)
            inc = daily_new_onset_by_age_erlang_presymp(st)
        else:
            st = simulate_jax_erlang(split_seed_to_erlang(C["st"][s]), **kw,
                                      discretize_time=False)
            inc = daily_new_infection_by_age_erlang(st)
        pred = simulation_to_hira_by_age_jax(inc, jnp.asarray(GAMMA), n_weeks=C["nw"])
        return nb_nll_jax(obsj, pred, wj, concentration=x[4], min_rate=0.01)
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v):
            v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    return fg


def free_fit_starts(n_starts=16, seed=42):
    """Diverse init: π_work in 0.05..0.50, others uniformly around A ref."""
    rng = np.random.default_rng(seed)
    starts = []
    piw_grid = np.linspace(0.05, 0.50, n_starts)
    for piw in piw_grid:
        others = np.array([0.30, 0.08, 0.62 - piw])   # h, s, o roughly A norm
        pi0 = np.array([others[0], piw, others[1], max(0.05, others[2])])
        pi0 = pi0 / pi0.sum()
        logit0 = np.log(np.clip(pi0, 1e-4, None))
        logit0 = logit0 - logit0.mean() + rng.normal(0, 0.2, 4)
        logR0 = np.log(rng.uniform(1.8, 2.5))
        starts.append(np.concatenate([[logR0], logit0, [10.0]]))
    return starts


def free_fit(C, s, i, mode, n_starts=16):
    fg = build_loss_free(C, s, i, mode)
    bounds = [LOG_R0_B] + [LOGIT_B]*4 + [PHI_NB_B]
    starts = free_fit_starts(n_starts, seed=(101 + i))
    results = []
    for k, x0 in enumerate(starts):
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=500, ftol=1e-9, gtol=1e-6))
            x = r.x
            pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
            piw = float(pi[1])
            R0 = float(np.exp(x[0]))
            # check railing (logit boundary or π extreme)
            logit_rail = any(abs(v - LOGIT_B[0]) < 0.1 or abs(v - LOGIT_B[1]) < 0.1
                              for v in x[1:5])
            piw_rail = piw < RAIL_LO or piw > RAIL_HI
            results.append(dict(
                start=k, pi_work_init=float(np.exp(x0[2] - x0[1:5].mean())
                                              / np.sum(np.exp(x0[1:5] - x0[1:5].mean()))),
                nll=float(r.fun), R0=R0,
                pi=[float(p) for p in pi], pi_work=piw,
                logit_rail=bool(logit_rail), piw_rail=bool(piw_rail),
                rail=bool(logit_rail or piw_rail),
                success=bool(r.success),
            ))
        except Exception as e:
            results.append(dict(start=k, err=str(e), rail=False))
    best = min((r for r in results if "nll" in r), key=lambda r: r["nll"])
    piw_all = [r["pi_work"] for r in results if "pi_work" in r]
    n_rail = sum(1 for r in results if r.get("rail"))
    return dict(
        best=best,
        pi_work_all=piw_all,
        pi_work_std=float(np.std(piw_all)) if piw_all else float("nan"),
        pi_work_range=[float(min(piw_all)), float(max(piw_all))] if piw_all else None,
        n_rail=n_rail, n_starts=len(results),
        starts_detail=results,
    )


def nll_profile(C, s, i, mode, piw_grid=PI_WORK_GRID, n_restarts=3):
    """π_work sweep, 나머지 4-D fit."""
    bounds = [LOG_R0_B] + [LOGIT_B]*3 + [PHI_NB_B]
    prof = []
    rng = np.random.default_rng(31 + i)
    for piw in piw_grid:
        fg = build_loss_fixed_piw(C, s, i, mode, piw)
        best = None
        for k in range(n_restarts):
            # init u=[u_h, u_s, u_o] proportional to A relative shares (excl. work)
            base_pi = np.array([0.408, 0.085, 0.507])
            u0 = np.log(base_pi) + rng.normal(0, 0.3, 3)
            x0 = np.concatenate([[np.log(rng.uniform(1.9, 2.4))], u0, [10.0]])
            try:
                r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                              options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                continue
        prof.append(dict(pi_work=float(piw),
                         nll=float(best.fun) if best else float("nan"),
                         R0=float(np.exp(best.x[0])) if best else float("nan")))
    return prof


def main():
    print("=" * 96)
    print("★ WORK IDENTIFIABILITY — presymp 전후 자유추정 + NLL profile")
    print(f"  free-fit: 5-D no pin, multistart 16 (π_work init 0.05~0.50)")
    print(f"  profile: π_work ∈ {PI_WORK_GRID.tolist()}, 나머지 4-D fit × 3 restarts")
    print("=" * 96)
    t0 = time.perf_counter(); C = build()
    print(f"[setup] {time.perf_counter()-t0:.1f}s")

    output = dict(seasons=SEAS, w_presymp=W, ngm_factor=NGM_F,
                   pi_work_grid=PI_WORK_GRID.tolist(),
                   railing_thresholds=dict(pi_work_lo=RAIL_LO, pi_work_hi=RAIL_HI,
                                            logit_boundary_slack=0.1))

    # ===== Part 1: Free-fit per season, both modes =====
    print("\n[1] Free-fit (pin 해제, multistart 16):")
    print(f"{'season':>10s} {'mode':>10s}  {'best π_w':>8s}  {'best β_w':>9s}  "
          f"{'best R0':>7s}  {'π_w std':>8s}  {'π_w range':>17s}  {'rail':>6s}")
    print("-" * 96)
    free = {}
    for s, i in zip(SEAS, IDX):
        free[s] = {}
        for mode in ["no_presymp", "presymp"]:
            t_ = time.perf_counter()
            r = free_fit(C, s, i, mode)
            wall = time.perf_counter() - t_
            free[s][mode] = r
            # best β
            best = r["best"]
            if mode == "presymp":
                b_all = np.asarray(
                    derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(best["R0"]),
                                                 jnp.asarray(best["pi"]),
                                                 jnp.asarray(PHI))) / NGM_F
            else:
                b_all = np.asarray(
                    derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(best["R0"]),
                                                 jnp.asarray(best["pi"]),
                                                 jnp.asarray(PHI)))
            bw = float(b_all[1])
            best["beta_4"] = [float(b) for b in b_all]
            print(f"{s:>10s} {mode:>10s}  {best['pi_work']:>8.4f}  {bw:>9.5f}  "
                  f"{best['R0']:>7.3f}  {r['pi_work_std']:>8.4f}  "
                  f"[{r['pi_work_range'][0]:.3f},{r['pi_work_range'][1]:.3f}]"
                  f"  {r['n_rail']:>2d}/{r['n_starts']:<2d}  ({wall:.1f}s)")
    output["free_fit"] = free
    (ED/"work_identifiability_presymp.json").write_text(
        json.dumps(output, indent=2, default=float))
    print(f"\n  [partial save] {ED/'work_identifiability_presymp.json'}")

    # Railing 요약
    print("\n[2] Railing 빈도 (start 중 railing 개수 / 총 start):")
    for mode in ["no_presymp", "presymp"]:
        totals = [(s, free[s][mode]["n_rail"], free[s][mode]["n_starts"]) for s in SEAS]
        tot_r = sum(t[1] for t in totals); tot_n = sum(t[2] for t in totals)
        print(f"  {mode:>10s}: " + "  ".join(f"{s}={t[1]}/{t[2]}" for s,t in zip(SEAS,totals))
              + f"   total={tot_r}/{tot_n}")

    # multistart std 요약
    print("\n[3] Multistart π_work std (수렴 산포):")
    for mode in ["no_presymp", "presymp"]:
        stds = [free[s][mode]["pi_work_std"] for s in SEAS]
        print(f"  {mode:>10s}: " + "  ".join(f"{s}={st:.4f}" for s,st in zip(SEAS,stds)))

    # ===== Part 2: NLL profile =====
    print("\n[4] NLL profile (π_work sweep):")
    profiles = {}
    for s, i in zip(SEAS, IDX):
        profiles[s] = {}
        for mode in ["no_presymp", "presymp"]:
            print(f"  {s} / {mode} ...", end=" ", flush=True)
            t_ = time.perf_counter()
            prof = nll_profile(C, s, i, mode)
            profiles[s][mode] = prof
            nlls = [p["nll"] for p in prof]
            min_i = int(np.argmin(nlls))
            print(f"argmin π_w={PI_WORK_GRID[min_i]:.2f} "
                  f"(NLL range {min(nlls):.1f}~{max(nlls):.1f}, "
                  f"Δ={max(nlls)-min(nlls):.2f})  ({time.perf_counter()-t_:.1f}s)")
    output["nll_profile"] = profiles
    (ED/"work_identifiability_presymp.json").write_text(
        json.dumps(output, indent=2, default=float))

    # 곡률 (Δ NLL max-min = 식별성 척도)
    print("\n[5] NLL 곡률 (Δ = maxNLL − minNLL, 클수록 잘 식별):")
    for mode in ["no_presymp", "presymp"]:
        for s in SEAS:
            nlls = [p["nll"] for p in profiles[s][mode]]
            dp = max(nlls) - min(nlls)
            argmin_i = int(np.argmin(nlls))
            print(f"  {mode:>10s}  {s}: Δ={dp:>7.2f}  "
                  f"argmin π_w={PI_WORK_GRID[argmin_i]:.2f}")

    # ===== Part 3: Figure =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for k, s in enumerate(SEAS):
        ax = axes[k]
        for mode, col, ls in [("no_presymp", "#666666", "--"),
                                ("presymp", "#B23A48", "-")]:
            nlls = np.array([p["nll"] for p in profiles[s][mode]])
            nlls_rel = nlls - nlls.min()
            ax.plot(PI_WORK_GRID, nlls_rel, marker="o", color=col, ls=ls,
                    lw=1.8, ms=5, label=mode)
            argmin_i = int(np.argmin(nlls))
            ax.axvline(PI_WORK_GRID[argmin_i], color=col, ls=":", alpha=0.4)
        ax.set_title(f"{s}", fontsize=11, fontweight="bold")
        ax.set_xlabel("π_work"); ax.set_ylabel("NLL − min(NLL)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("π_work NLL profile — presymp 전(회색) vs 후(빨강)   "
                  "곡률 클수록 잘 식별", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = FIG / "work_identifiability_profile.png"
    fig.savefig(fp, bbox_inches="tight"); plt.close(fig)
    print(f"\n[figure] {fp}")

    (ED/"work_identifiability_presymp.json").write_text(
        json.dumps(output, indent=2, default=float))
    print(f"[json] {ED/'work_identifiability_presymp.json'}")


if __name__ == "__main__":
    main()
