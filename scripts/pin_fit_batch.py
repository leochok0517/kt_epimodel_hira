"""Channel-pin fit batch — target A/B × γ_report {a, b} × [point est + NUTS + policy].

4 combos (single season 2019-2020, φ U-shape fixed, R(0) default, A-fix cov):
  A-a : target_A (NIMS)      γ = (0.40, 0.18, 0.25)  ← current center
  A-b : target_A (NIMS)      γ = (0.40, 0.35, 0.45)  ← φ-matched
  B-a : target_B (literature) γ = (0.40, 0.18, 0.25)
  B-b : target_B (literature) γ = (0.40, 0.35, 0.45)

Per combo, three phases with partial JSON save after each:
  1) point estimate — L-BFGS × 12 starts over (log_R0, logit_pi, phi_nb)
       + channel_prior factor SIGMA_PER_CHANNEL=[0.01, 0.01, 0.30, 0.30].
  2) NUTS full run — 4 chains, warmup 1000, sample 1000, max_tree_depth 8,
       target_accept 0.9, init_to_value from point estimate.
       Local model (production hira_model_nb_chprior is NOT modified).
  3) Policy — posterior mean β_4 forward with (a) sick_leave p_work=0.4,
       (b) school-closure-style p_school=0.4. Report averted% total + per age.

Safe defaults: try/except around each phase; JSON is rewritten after every
successful phase so a crash mid-batch keeps prior results. Production code
is NOT modified.
"""
from __future__ import annotations
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from scipy.optimize import minimize
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value
import arviz as az

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "pin_fit_batch.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "pin_fit_batch.png"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

SEASON_LABEL = "2019-2020"
AMP = 0.9
HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)
UNIT_R0 = np.array([8.70, 6.21, 25.40, 9.33])

PHI_USHAPE = np.array(
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64,
)

TARGETS = {
    "A": np.array([23.26, 17.03, 14.24, 31.77]),   # NIMS contact row-sums (h, w, s, o)
    "B": np.array([0.40, 0.10, 0.27, 0.23]),        # Italy 2009 R0 contrib (h, w, s, o)
}
GAMMA_VARIANTS = {
    "a": (0.40, 0.18, 0.25),
    "b": (0.40, 0.35, 0.45),
}
COMBOS = [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")]

SIGMA_PER_CHANNEL = np.array([0.01, 0.01, 0.30, 0.30])
COVERAGE_CAP = 0.99

# Point estimate
N_STARTS = 12
POINT_START_SEED = 23
LOG_R0_BOUNDS = (float(np.log(0.8)), float(np.log(3.0)))
LOGIT_PI_BOUNDS = (-10.0, 10.0)
PHI_NB_BOUNDS = (1e-3, 1e6)

# NUTS
N_CHAINS = 4
N_WARMUP = 1000
N_SAMPLES = 1000
TARGET_ACCEPT = 0.9
MAX_TREE_DEPTH = 8

# Policy
P_WORK_BASELINE = 1.0
P_SCHOOL_BASELINE = 1.0
P_WORK_SICK = 0.4
P_SCHOOL_ABSENCE = 0.4

# Peak window
PEAK_HALF_WIN = 2


def correct_coverage(cov_15: np.ndarray) -> np.ndarray:
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_gamma_15(child: float, adult: float, elder: float) -> np.ndarray:
    return np.concatenate([np.full(4, child), np.full(9, adult),
                            np.full(2, elder)])


def r0contrib_to_pi(r0c: np.ndarray) -> np.ndarray:
    r0c_n = r0c / r0c.sum()
    beta_share = r0c_n / UNIT_R0
    return beta_share / beta_share.sum()


def logit_centered_target(pi_target: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(pi_target, 1e-6, None))
    return lp - lp.mean()


def build_setup():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    policy = ModelParameters().policy

    seed_15 = estimate_initial_infected_from_hira(
        SEASON_LABEL, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    tgt = load_hira_target_by_age(
        SEASON_LABEL, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    n_weeks = tgt["n_weeks"]
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]

    cov_eff = correct_coverage(np.asarray(vax.annual_coverage, dtype=np.float64))
    shared = dict(
        C_home=jnp.asarray(matrices["C_home"]),
        C_school=jnp.asarray(matrices["C_school"]),
        C_work=jnp.asarray(matrices["C_work"]),
        C_other=jnp.asarray(matrices["C_other"]),
        M_home=jnp.asarray(mobility["home"]),
        M_school=jnp.asarray(mobility["school"]),
        M_work=jnp.asarray(mobility["work"]),
        M_other=jnp.asarray(mobility["other"]),
        pop_15=jnp.asarray(pop_15),
        rho=jnp.asarray(rho_emp),
        kappa=jnp.asarray(disease.kappa_array),
        sigma=disease.sigma, gamma=disease.gamma,
        VE=vax.VE,
        annual_coverage=jnp.asarray(cov_eff),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared.update(HOLIDAY)

    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE,
        initial_vaccinated_fraction=0.0,
    ))
    ngm_default = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )

    return dict(
        shared=shared, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
    )


def predict_hira(beta_4, phi_full_j, gamma_15_j, p_school, p_work, setup):
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_j
    kw["p_school"] = p_school; kw["p_work"] = p_work
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    pred_hira = simulation_to_hira_by_age_jax(inc, gamma_15_j,
                                                n_weeks=setup["n_weeks"])
    return inc, pred_hira


def peak_window_sum(arr_1d, peak_w, half):
    lo = max(0, peak_w - half)
    hi = min(arr_1d.shape[0], peak_w + half + 1)
    return float(arr_1d[lo:hi].sum())


def per_age_ratios(pred, setup):
    obs = setup["obs_np"]
    mask = setup["w_np"].sum(axis=1) > 0
    obs_m = np.where(mask[:, None], obs, -1e18)
    pred_m = np.where(mask[:, None], pred, -1e18)
    rows = {}
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        obs_pw = int(np.argmax(obs_m[:, ai]))
        mdl_pw = int(np.argmax(pred_m[:, ai]))
        obs_sum = peak_window_sum(obs[:, ai], obs_pw, PEAK_HALF_WIN)
        mdl_sum = peak_window_sum(pred[:, ai], mdl_pw, PEAK_HALF_WIN)
        rows[ag] = dict(
            obs_peak_week=obs_pw, model_peak_week=mdl_pw,
            phase_offset_weeks=mdl_pw - obs_pw,
            ratio=obs_sum / max(mdl_sum, 1.0),
        )
    return rows


# ─── Point estimate ────────────────────────────────────────
def build_point_loss(logit_target, sigma_per_channel, phi_full_j, gamma_15_j,
                       setup):
    lt = jnp.asarray(logit_target)
    inv_var = 1.0 / (sigma_per_channel ** 2)

    def loss(x):
        log_R0 = x[0]
        logit_pi = x[1:5]
        phi_nb = x[5]
        R0 = jnp.exp(log_R0)
        pi = jax.nn.softmax(logit_pi)
        beta_4 = derive_beta_from_R0_simplex(setup["ngm_default"], R0, pi,
                                               phi_full_j)
        _, pred = predict_hira(beta_4, phi_full_j, gamma_15_j,
                                 P_SCHOOL_BASELINE, P_WORK_BASELINE, setup)
        nll = nb_nll_jax(setup["obs_j"], pred, setup["w_j"],
                         concentration=phi_nb, min_rate=0.01)
        centered = logit_pi - jnp.mean(logit_pi)
        dev = centered - lt
        ch_pen = 0.5 * jnp.sum(dev * dev * inv_var)
        return nll + ch_pen

    loss_j = jax.jit(loss)
    grad_j = jax.jit(jax.grad(loss))

    def fg_np(x_np):
        x = jnp.asarray(x_np)
        v = float(loss_j(x))
        g = np.asarray(grad_j(x))
        if not np.isfinite(v):
            v = 1e15
            g = np.where(np.isfinite(g), g, 0.0)
        return v, g

    return fg_np


def point_estimate(pi_target, sigma_per_channel, phi_full, gamma_15, setup):
    logit_target = logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(phi_full)
    gamma_15_j = jnp.asarray(gamma_15)
    fg = build_point_loss(logit_target, sigma_per_channel, phi_full_j,
                            gamma_15_j, setup)
    bounds = [LOG_R0_BOUNDS] + [LOGIT_PI_BOUNDS] * 4 + [PHI_NB_BOUNDS]
    rng = np.random.default_rng(POINT_START_SEED)
    starts = []
    for delta in [0.0, 0.2, -0.2, 0.4, -0.4]:
        x0 = np.concatenate([[np.log(2.0) + delta], np.asarray(logit_target),
                              [10.0]])
        starts.append(x0)
    while len(starts) < N_STARTS:
        starts.append(np.concatenate([
            [rng.uniform(*LOG_R0_BOUNDS)],
            np.asarray(logit_target) + rng.normal(0, 0.5, 4),
            [rng.uniform(2.0, 20.0)],
        ]))
    starts = starts[:N_STARTS]

    best = None
    per_start_nll = []
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                            options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
            nll = float(res.fun)
        except Exception:
            continue
        per_start_nll.append(nll)
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0

    x = best["x"]
    log_R0 = float(x[0])
    logit_pi = np.array(x[1:5])
    phi_nb = float(x[5])
    R0 = float(np.exp(log_R0))
    pi = np.array(jax.nn.softmax(jnp.asarray(logit_pi)))
    beta_4 = np.array(derive_beta_from_R0_simplex(
        setup["ngm_default"], jnp.asarray(R0), jnp.asarray(pi), phi_full_j,
    ))
    _, pred_j = predict_hira(beta_4, phi_full_j, gamma_15_j,
                               P_SCHOOL_BASELINE, P_WORK_BASELINE, setup)
    pred = np.asarray(pred_j)
    r_per_age = per_age_ratios(pred, setup)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        phi_full_j,
    ))

    nll_arr = np.array(per_start_nll)
    return dict(
        log_R0=log_R0, R0=R0,
        logit_pi=logit_pi.tolist(), pi=[float(p) for p in pi],
        beta_4=[float(b) for b in beta_4], phi_nb=phi_nb,
        nll=best["nll"], best_start_idx=best["start_idx"],
        R0_ngm=r0_ngm,
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=r_per_age,
        wall_sec=float(wall),
    )


# ─── NUTS ──────────────────────────────────────────────────
def build_nuts_model(logit_target, sigma_per_channel, phi_full_j, gamma_15_j,
                       setup):
    lt = jnp.asarray(logit_target)
    inv_var = 1.0 / (jnp.asarray(sigma_per_channel) ** 2)
    ngm_fn = setup["ngm_default"]

    def model():
        log_R0 = numpyro.sample(
            "log_R0",
            dist.TruncatedNormal(
                jnp.log(2.0), 0.4,
                low=jnp.log(1.1), high=jnp.log(3.0),
            ),
        )
        R0 = jnp.exp(log_R0)
        logit_pi = numpyro.sample(
            "logit_pi",
            dist.Normal(0.0, 2.0).expand([4]).to_event(1),
        )
        centered = logit_pi - jnp.mean(logit_pi)
        dev = centered - lt
        numpyro.factor("ch_prior",
                       -0.5 * jnp.sum(dev * dev * inv_var))
        pi = jax.nn.softmax(logit_pi)
        phi_nb = numpyro.sample("phi_nb", dist.HalfNormal(10.0))
        beta_4 = derive_beta_from_R0_simplex(ngm_fn, R0, pi, phi_full_j)

        kw = dict(setup["shared"])
        kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
        kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
        kw["phi_susc"] = phi_full_j
        kw["p_school"] = P_SCHOOL_BASELINE; kw["p_work"] = P_WORK_BASELINE
        st = simulate_jax(setup["state0"], **kw, discretize_time=False)
        inc = daily_new_infection_by_age_jax(st)
        pred = simulation_to_hira_by_age_jax(inc, gamma_15_j,
                                              n_weeks=setup["n_weeks"])
        nll = nb_nll_jax(setup["obs_j"], pred, setup["w_j"],
                         concentration=phi_nb, min_rate=0.01)
        numpyro.factor("likelihood_nb", -nll)
        numpyro.deterministic("R0", R0)
        numpyro.deterministic("pi", pi)
        numpyro.deterministic("beta_4", beta_4)

    return model


def summarize_nuts(mcmc):
    samples = mcmc.get_samples(group_by_chain=True)
    extras = mcmc.get_extra_fields(group_by_chain=True)
    n_div = int(np.asarray(extras["diverging"]).sum())

    idata = az.from_numpyro(mcmc)
    diag_vars = ["log_R0", "logit_pi", "phi_nb", "R0", "pi", "beta_4"]
    rhat_max = float("nan"); ess_min = float("nan")
    rhat_per = {}
    try:
        rh = az.rhat(idata, var_names=diag_vars)
        for v in rh.data_vars:
            arr = np.asarray(rh[v].values)
            rhat_per[v] = arr.tolist() if arr.ndim else float(arr)
        rhat_max = float(np.nanmax([np.nanmax(rh[v].values)
                                     for v in rh.data_vars]))
    except Exception as e:
        print(f"    [warn] rhat failed: {e}", flush=True)
    try:
        es = az.ess(idata, var_names=diag_vars)
        ess_min = float(np.nanmin([np.nanmin(es[v].values)
                                     for v in es.data_vars]))
    except Exception as e:
        print(f"    [warn] ess failed: {e}", flush=True)

    def flat(arr):
        arr = np.asarray(arr)
        return arr.reshape(-1, *arr.shape[2:])

    R0_s = flat(samples["R0"])
    pi_s = flat(samples["pi"])
    phi_nb_s = flat(samples["phi_nb"])
    beta_s = flat(samples["beta_4"])

    def summ(x):
        return dict(mean=float(np.mean(x)),
                    q05=float(np.quantile(x, 0.05)),
                    q95=float(np.quantile(x, 0.95)))

    return dict(
        n_divergent=n_div,
        rhat_max=rhat_max, rhat_per_var=rhat_per, ess_min=ess_min,
        R0=summ(R0_s),
        phi_nb=summ(phi_nb_s),
        pi_summary=[summ(pi_s[:, c]) for c in range(4)],
        beta_summary=[summ(beta_s[:, c]) for c in range(4)],
        R0_mean=float(np.mean(R0_s)),
        beta_mean=[float(np.mean(beta_s[:, c])) for c in range(4)],
        pi_mean=[float(np.mean(pi_s[:, c])) for c in range(4)],
    )


def run_nuts(combo_name, pi_target, sigma_per_channel, phi_full, gamma_15,
              setup, point_result):
    logit_target = logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(phi_full)
    gamma_15_j = jnp.asarray(gamma_15)
    model = build_nuts_model(logit_target, sigma_per_channel, phi_full_j,
                               gamma_15_j, setup)
    init_vals = {
        "log_R0": jnp.asarray(point_result["log_R0"]),
        "logit_pi": jnp.asarray(point_result["logit_pi"]),
        "phi_nb": jnp.asarray(point_result["phi_nb"]),
    }
    kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT,
                   max_tree_depth=MAX_TREE_DEPTH, dense_mass=False,
                   init_strategy=init_to_value(values=init_vals))
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                 num_chains=N_CHAINS, chain_method="sequential",
                 progress_bar=True)
    seed_map = {"A-a": 101, "A-b": 103, "B-a": 107, "B-b": 109}
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(seed_map[combo_name]),
              extra_fields=("diverging",))
    wall = time.perf_counter() - t0
    rec = summarize_nuts(mcmc)
    rec["wall_sec"] = float(wall)
    rec["config"] = dict(n_chains=N_CHAINS, n_warmup=N_WARMUP,
                          n_samples=N_SAMPLES, target_accept=TARGET_ACCEPT,
                          max_tree_depth=MAX_TREE_DEPTH,
                          sigma_per_channel=sigma_per_channel.tolist())
    return rec, mcmc


# ─── Policy analysis (posterior mean β) ──────────────────
def policy_analysis(nuts_rec, phi_full, gamma_15, setup):
    """Use posterior mean β_4 for baseline / sick_leave / school_absence."""
    beta_4 = np.array(nuts_rec["beta_mean"])
    phi_full_j = jnp.asarray(phi_full)
    gamma_15_j = jnp.asarray(gamma_15)

    def run(p_school, p_work):
        inc, pred_hira = predict_hira(beta_4, phi_full_j, gamma_15_j,
                                        p_school, p_work, setup)
        return np.asarray(inc), np.asarray(pred_hira)

    inc_b, pred_b = run(P_SCHOOL_BASELINE, P_WORK_BASELINE)
    inc_s, pred_s = run(P_SCHOOL_BASELINE, P_WORK_SICK)
    inc_c, pred_c = run(P_SCHOOL_ABSENCE, P_WORK_BASELINE)

    tot_b = float(inc_b.sum())
    tot_s = float(inc_s.sum())
    tot_c = float(inc_c.sum())

    def averted_by_age(pred_base, pred_scenario):
        out = {}
        for ai, ag in enumerate(HIRA_AGE_GROUPS):
            b = float(pred_base[:, ai].sum())
            s = float(pred_scenario[:, ai].sum())
            out[ag] = 100.0 * (b - s) / max(b, 1e-9)
        return out

    return dict(
        beta_used=[float(b) for b in beta_4],
        baseline_total_infections=tot_b,
        sick_leave_total=tot_s,
        school_absence_total=tot_c,
        averted_total_sick_leave_pct=100.0 * (tot_b - tot_s) / max(tot_b, 1.0),
        averted_total_school_absence_pct=100.0 * (tot_b - tot_c) / max(tot_b, 1.0),
        averted_by_age_sick_leave_pct=averted_by_age(pred_b, pred_s),
        averted_by_age_school_absence_pct=averted_by_age(pred_b, pred_c),
    )


# ─── Batch orchestration ─────────────────────────────────
def save_json(all_results: dict):
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)


def main():
    print("=" * 78, flush=True)
    print(f"PIN FIT BATCH  —  {SEASON_LABEL}", flush=True)
    print(f"  4 combos = target(A,B) × γ(a,b)", flush=True)
    print(f"  phase 1 = point (L-BFGS × {N_STARTS})", flush=True)
    print(f"  phase 2 = NUTS ({N_CHAINS} chains, {N_WARMUP}+{N_SAMPLES}, "
          f"max_tree_depth {MAX_TREE_DEPTH})", flush=True)
    print(f"  phase 3 = policy (sick_leave p_work={P_WORK_SICK}, "
          f"school p_school={P_SCHOOL_ABSENCE}) on posterior mean β", flush=True)
    print(f"  φ FIXED U-shape; R(0) default; A-fix coverage; HOLIDAY on",
          flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    all_results = dict(
        season=SEASON_LABEL,
        config=dict(
            AMP=AMP, HOLIDAY=HOLIDAY, UNIT_R0=UNIT_R0.tolist(),
            PHI_USHAPE=PHI_USHAPE.tolist(),
            SIGMA_PER_CHANNEL=SIGMA_PER_CHANNEL.tolist(),
            targets={k: v.tolist() for k, v in TARGETS.items()},
            gamma_variants={k: list(v) for k, v in GAMMA_VARIANTS.items()},
            n_starts=N_STARTS, n_chains=N_CHAINS, n_warmup=N_WARMUP,
            n_samples=N_SAMPLES, target_accept=TARGET_ACCEPT,
            max_tree_depth=MAX_TREE_DEPTH,
            p_work_sick=P_WORK_SICK, p_school_absence=P_SCHOOL_ABSENCE,
        ),
        combos={},
    )

    for target_key, gamma_key in COMBOS:
        combo_name = f"{target_key}-{gamma_key}"
        print(f"\n{'#' * 60}\n#  COMBO {combo_name}\n{'#' * 60}", flush=True)
        pi_target = r0contrib_to_pi(TARGETS[target_key])
        gc, ga, ge = GAMMA_VARIANTS[gamma_key]
        gamma_15 = build_gamma_15(gc, ga, ge)
        print(f"  target = {target_key}: R0-contrib {TARGETS[target_key].tolist()}"
              f" → π {[round(float(x),3) for x in pi_target]}", flush=True)
        print(f"  γ = {gamma_key}: ({gc}, {ga}, {ge})", flush=True)
        print(f"  σ_per_channel = {SIGMA_PER_CHANNEL.tolist()}", flush=True)

        combo_rec = dict(
            target=target_key, gamma_variant=gamma_key,
            pi_target=pi_target.tolist(),
            gamma_15=gamma_15.tolist(),
            phase1_point=None, phase2_nuts=None, phase3_policy=None,
            errors=[],
        )
        all_results["combos"][combo_name] = combo_rec
        save_json(all_results)

        # PHASE 1 — point estimate
        try:
            print(f"\n  [phase 1] point estimate ...", flush=True)
            pt = point_estimate(pi_target, SIGMA_PER_CHANNEL,
                                  PHI_USHAPE, gamma_15, setup)
            combo_rec["phase1_point"] = pt
            print(f"    NLL={pt['nll']:.4e}  R0={pt['R0']:.3f}"
                  f"  β=[{', '.join(f'{b:.4f}' for b in pt['beta_4'])}]  "
                  f"phi_nb={pt['phi_nb']:.2f}", flush=True)
            print(f"    π = {[round(x,4) for x in pt['pi']]}", flush=True)
            print(f"    per-age r: " + "  ".join(
                f"{ag}={pt['per_age'][ag]['ratio']:.2f}"
                for ag in HIRA_AGE_GROUPS), flush=True)
            save_json(all_results)
        except Exception as e:
            msg = f"phase 1 failed: {e}"
            print(f"    [ERROR] {msg}", flush=True)
            combo_rec["errors"].append(msg)
            save_json(all_results)
            continue

        # PHASE 2 — NUTS
        try:
            print(f"\n  [phase 2] NUTS ...", flush=True)
            nu, mcmc = run_nuts(combo_name, pi_target, SIGMA_PER_CHANNEL,
                                  PHI_USHAPE, gamma_15, setup, pt)
            combo_rec["phase2_nuts"] = nu
            print(f"    max r_hat={nu['rhat_max']:.3f}  "
                  f"ess_min={nu['ess_min']:.1f}  div={nu['n_divergent']}  "
                  f"wall={nu['wall_sec']:.0f}s", flush=True)
            print(f"    R0: mean={nu['R0']['mean']:.3f}  "
                  f"[90% {nu['R0']['q05']:.3f}, {nu['R0']['q95']:.3f}]",
                  flush=True)
            for c, ch in enumerate(["home", "work", "school", "other"]):
                s = nu["pi_summary"][c]
                print(f"    π_{ch}: mean={s['mean']:.4f}  "
                      f"[{s['q05']:.4f}, {s['q95']:.4f}]  "
                      f"target={pi_target[c]:.4f}", flush=True)
            save_json(all_results)
        except Exception as e:
            msg = f"phase 2 failed: {e}"
            print(f"    [ERROR] {msg}", flush=True)
            combo_rec["errors"].append(msg)
            save_json(all_results)
            continue

        # PHASE 3 — policy
        try:
            print(f"\n  [phase 3] policy averted ...", flush=True)
            pol = policy_analysis(nu, PHI_USHAPE, gamma_15, setup)
            combo_rec["phase3_policy"] = pol
            print(f"    averted (sick_leave, total) = "
                  f"{pol['averted_total_sick_leave_pct']:+.2f}%", flush=True)
            print(f"    averted (school_absence, total) = "
                  f"{pol['averted_total_school_absence_pct']:+.2f}%", flush=True)
            print(f"    per-age (sick_leave): " + "  ".join(
                f"{ag}={pol['averted_by_age_sick_leave_pct'][ag]:+.2f}%"
                for ag in HIRA_AGE_GROUPS), flush=True)
            print(f"    per-age (school): " + "  ".join(
                f"{ag}={pol['averted_by_age_school_absence_pct'][ag]:+.2f}%"
                for ag in HIRA_AGE_GROUPS), flush=True)
            save_json(all_results)
        except Exception as e:
            msg = f"phase 3 failed: {e}"
            print(f"    [ERROR] {msg}", flush=True)
            combo_rec["errors"].append(msg)
            save_json(all_results)

    # Final figure
    try:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        combo_names = [f"{t}-{g}" for t, g in COMBOS]
        ages = HIRA_AGE_GROUPS
        combo_colors = ["#1a5490", "#27ae60", "#c0392b", "#f39c12"]

        # Panel 1: per-age r for all 4 combos (NUTS posterior forward)
        ax = axes[0]
        xa = np.arange(len(ages))
        bw = 0.2
        for i, cn in enumerate(combo_names):
            rec = all_results["combos"].get(cn)
            if rec and rec.get("phase1_point"):
                vals = [rec["phase1_point"]["per_age"][ag]["ratio"]
                        for ag in ages]
                ax.bar(xa + (i - 1.5) * bw, vals, bw,
                        color=combo_colors[i], label=cn, alpha=0.85)
        ax.axhline(1.0, color="grey", ls=":", lw=1)
        ax.set_xticks(xa); ax.set_xticklabels(ages)
        ax.set_ylabel("r = obs / model")
        ax.set_title("Per-age r (point estimate)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

        # Panel 2: R0 posterior mean + CI
        ax = axes[1]
        x = np.arange(len(combo_names))
        for i, cn in enumerate(combo_names):
            rec = all_results["combos"].get(cn)
            if rec and rec.get("phase2_nuts"):
                r0 = rec["phase2_nuts"]["R0"]
                m = r0["mean"]; lo = r0["q05"]; hi = r0["q95"]
                ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]],
                             fmt="o", ms=10, capsize=5,
                             color=combo_colors[i], label=cn)
        ax.axhline(2.0, color="grey", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(combo_names)
        ax.set_ylabel("R0 posterior")
        ax.set_title("R0 posterior mean + 90% CI (NUTS)")
        ax.grid(True, alpha=0.3)

        # Panel 3: averted % (sick_leave vs school_absence)
        ax = axes[2]
        for i, cn in enumerate(combo_names):
            rec = all_results["combos"].get(cn)
            if rec and rec.get("phase3_policy"):
                p = rec["phase3_policy"]
                vals = [p["averted_total_sick_leave_pct"],
                        p["averted_total_school_absence_pct"]]
                ax.bar([2 * i - 0.3, 2 * i + 0.3], vals, 0.6,
                        color=combo_colors[i], label=cn, alpha=0.85)
        ax.axhline(0.0, color="grey", ls=":", lw=1)
        ax.set_xticks([2 * i for i in range(len(combo_names))])
        ax.set_xticklabels([f"{cn}\n(sick / school)" for cn in combo_names])
        ax.set_ylabel("averted % (total)")
        ax.set_title("Policy averted (sick_leave / school_absence)")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Channel-pin batch — {SEASON_LABEL}  "
                      f"(φ U-shape, R(0) default, A-fix cov)")
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
        print(f"\nsaved {OUT_FIG}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] figure failed: {e}", flush=True)

    save_json(all_results)
    print(f"\nsaved {OUT_JSON}", flush=True)
    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
