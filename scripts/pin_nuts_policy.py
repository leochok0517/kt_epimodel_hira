"""Pin NUTS batch for policy-map representative combos (C1, C2, C3).

STEP 0 : chain-parallelism smoke — measure wall of a tiny NUTS (warmup 50
         + sample 50) once in sequential and once in parallel; pick the
         faster method for the full run and predict full wall.

STEP 1 : 3 representative combos from the work_share × κ sensitivity map:
         C1 : work_share 0.03 κ 0.4   (literature realistic, sick 역효과)
         C2 : work_share 0.06 κ 0.4   (literature mid)
         C3 : work_share 0.15 κ 0.2   (positive corner)
         Full NUTS (chains 4, warmup 600, sample 600, max_tree_depth 8),
         init_to_value from the point estimate, log_R0 low = log(1.1).
         Posterior policy averted computed sample-wise; report mean + 90% CI.

Setup: single season 2019-2020, φ U-shape fixed, R(0) default, A-fix cov,
HOLIDAY on, γ_report = (0.40, 0.18, 0.25). σ_per_channel = [0.15, 0.01,
0.05, 0.15] (school tightened per v2). Non-work distribution follows
A-relative [h 0.29 : s 0.06 : o 0.36] → normalised (0.408, 0.085, 0.507).

Safe defaults: per-combo try/except with partial JSON save; a stall in any
combo does not lose earlier results.

Production code is NOT modified. XLA / OMP env vars come from the launcher
wrapper (before Python import); this script does NOT set XLA_FLAGS.
"""
from __future__ import annotations
import os, json, time
# Force 4 parallel host devices for numpyro chain-parallel — must be set
# BEFORE `import jax`. Wrapper env inheritance was unreliable, so pin here.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "2"))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ.get("OMP_NUM_THREADS", "2"))

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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "pin_nuts_policy.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "pin_nuts_policy.png"
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
PHI_USHAPE = np.array(
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64,
)
GAMMA_CENTER = (0.40, 0.18, 0.25)

A_REL = np.array([0.29, 0.06, 0.36])           # (h, s, o)
A_REL_NORM = A_REL / A_REL.sum()                # → (0.408, 0.085, 0.507)

SIGMA_PER_CHANNEL = np.array([0.15, 0.01, 0.05, 0.15])   # v2 tuning

COMBOS = [
    dict(name="C1", work_share=0.03, kappa=0.4),
    dict(name="C2", work_share=0.06, kappa=0.4),
    dict(name="C3", work_share=0.15, kappa=0.2),
]

# Point estimate
N_STARTS = 12
POINT_START_SEED = 23
LOG_R0_BOUNDS = (float(np.log(0.8)), float(np.log(3.0)))
LOGIT_PI_BOUNDS = (-10.0, 10.0)
PHI_NB_BOUNDS = (1e-3, 1e6)

# NUTS full
FULL_WARMUP = 600
FULL_SAMPLES = 600
N_CHAINS = 4
TARGET_ACCEPT = 0.9
MAX_TREE_DEPTH = 8

# Smoke
SMOKE_WARMUP = 50
SMOKE_SAMPLES = 50

# Policy
P_WORK_BASE = 1.0
P_SCHOOL_BASE = 1.0
P_WORK_SICK = 0.4
P_SCHOOL_ABSENCE = 0.4

COVERAGE_CAP = 0.99
PEAK_HALF_WIN = 2


def correct_coverage(cov_15):
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_gamma_15(c, a, e):
    return np.concatenate([np.full(4, c), np.full(9, a), np.full(2, e)])


def build_pi_target(work_share: float) -> np.ndarray:
    r = 1.0 - work_share
    return np.array([r * A_REL_NORM[0], work_share,
                       r * A_REL_NORM[1], r * A_REL_NORM[2]])


def logit_centered_target(pi_target):
    lp = np.log(np.clip(pi_target, 1e-6, None))
    return lp - lp.mean()


def build_setup():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination

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
    shared_base = dict(
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
        sigma=disease.sigma, gamma=disease.gamma,
        VE=vax.VE,
        annual_coverage=jnp.asarray(cov_eff),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared_base.update(HOLIDAY)

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
        shared_base=shared_base, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
    )


def make_shared(kappa_scalar, setup):
    kw = dict(setup["shared_base"])
    kw["kappa"] = jnp.full(15, kappa_scalar, dtype=jnp.float64)
    return kw


def run_forward(beta_4, phi_full_j, gamma_15_j, p_school, p_work, shared, setup):
    kw = dict(shared)
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_j
    kw["p_school"] = p_school; kw["p_work"] = p_work
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    pred = simulation_to_hira_by_age_jax(inc, gamma_15_j,
                                          n_weeks=setup["n_weeks"])
    return inc, pred


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


# ── Point estimate ────────────────────────────────────────
def build_point_loss(logit_target, sigma_per_channel, phi_full_j, gamma_15_j,
                       shared, setup):
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
        _, pred = run_forward(beta_4, phi_full_j, gamma_15_j,
                                P_SCHOOL_BASE, P_WORK_BASE, shared, setup)
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


def make_starts(logit_target, n_starts, seed):
    rng = np.random.default_rng(seed)
    starts = []
    for delta in [0.0, 0.2, -0.2, 0.4, -0.4]:
        starts.append(np.concatenate([[np.log(2.0) + delta],
                                        np.asarray(logit_target),
                                        [10.0]]))
    while len(starts) < n_starts:
        starts.append(np.concatenate([
            [rng.uniform(*LOG_R0_BOUNDS)],
            np.asarray(logit_target) + rng.normal(0, 0.5, 4),
            [rng.uniform(2.0, 20.0)],
        ]))
    return starts[:n_starts]


def point_fit(work_share, kappa_val, setup):
    pi_target = build_pi_target(work_share)
    logit_target = logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15_j = jnp.asarray(build_gamma_15(*GAMMA_CENTER))
    shared = make_shared(kappa_val, setup)

    fg = build_point_loss(logit_target, SIGMA_PER_CHANNEL, phi_full_j,
                            gamma_15_j, shared, setup)
    bounds = [LOG_R0_BOUNDS] + [LOGIT_PI_BOUNDS] * 4 + [PHI_NB_BOUNDS]
    starts = make_starts(logit_target, N_STARTS, POINT_START_SEED)

    best = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                            options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
            nll = float(res.fun)
        except Exception:
            continue
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0

    x = best["x"]
    log_R0 = float(x[0]); logit_pi = np.array(x[1:5])
    phi_nb = float(x[5])
    R0 = float(np.exp(log_R0))
    pi = np.array(jax.nn.softmax(jnp.asarray(logit_pi)))
    beta_4 = np.array(derive_beta_from_R0_simplex(
        setup["ngm_default"], jnp.asarray(R0), jnp.asarray(pi), phi_full_j,
    ))
    _, pred_j = run_forward(beta_4, phi_full_j, gamma_15_j,
                              P_SCHOOL_BASE, P_WORK_BASE, shared, setup)
    ratios = per_age_ratios(np.asarray(pred_j), setup)
    return dict(
        pi_target=pi_target.tolist(),
        log_R0=log_R0, R0=R0,
        logit_pi=logit_pi.tolist(), pi=[float(p) for p in pi],
        beta_4=[float(b) for b in beta_4],
        phi_nb=phi_nb, nll=best["nll"],
        wall_sec=float(wall), per_age=ratios,
    )


# ── NUTS ──────────────────────────────────────────────────
def build_nuts_model(logit_target, phi_full_j, gamma_15_j, shared, setup):
    lt = jnp.asarray(logit_target)
    inv_var = 1.0 / (jnp.asarray(SIGMA_PER_CHANNEL) ** 2)
    ngm_fn = setup["ngm_default"]

    def model():
        log_R0 = numpyro.sample(
            "log_R0",
            dist.TruncatedNormal(jnp.log(2.0), 0.4,
                                  low=jnp.log(1.1), high=jnp.log(3.0)),
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

        kw = dict(shared)
        kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
        kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
        kw["phi_susc"] = phi_full_j
        kw["p_school"] = P_SCHOOL_BASE; kw["p_work"] = P_WORK_BASE
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


def run_nuts(combo, chain_method, warmup, samples, setup, point_result, seed):
    ws = combo["work_share"]; k = combo["kappa"]
    pi_target = build_pi_target(ws)
    logit_target = logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15_j = jnp.asarray(build_gamma_15(*GAMMA_CENTER))
    shared = make_shared(k, setup)

    model = build_nuts_model(logit_target, phi_full_j, gamma_15_j, shared,
                               setup)
    init_vals = {
        "log_R0": jnp.asarray(point_result["log_R0"]),
        "logit_pi": jnp.asarray(point_result["logit_pi"]),
        "phi_nb": jnp.asarray(point_result["phi_nb"]),
    }
    kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT,
                   max_tree_depth=MAX_TREE_DEPTH, dense_mass=False,
                   init_strategy=init_to_value(values=init_vals))
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                 num_chains=N_CHAINS, chain_method=chain_method,
                 progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(seed), extra_fields=("diverging",))
    wall = time.perf_counter() - t0
    return mcmc, wall


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
        R0=summ(R0_s), phi_nb=summ(phi_nb_s),
        pi_summary=[summ(pi_s[:, c]) for c in range(4)],
        beta_summary=[summ(beta_s[:, c]) for c in range(4)],
        # Keep flat samples for downstream policy
        _R0_samples=R0_s.tolist(),
        _pi_samples=pi_s.tolist(),
        _beta_samples=beta_s.tolist(),
        _phi_nb_samples=phi_nb_s.tolist(),
    )


def posterior_policy(nuts_rec, combo, setup, n_draws=200):
    """Compute averted per posterior draw (thin to n_draws) and summarise."""
    ws = combo["work_share"]; k = combo["kappa"]
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15_j = jnp.asarray(build_gamma_15(*GAMMA_CENTER))
    shared = make_shared(k, setup)

    beta_samples = np.array(nuts_rec["_beta_samples"])   # (N, 4)
    n_total = beta_samples.shape[0]
    idx = np.linspace(0, n_total - 1, n_draws, dtype=int)

    averted_sick_totals = []
    averted_school_totals = []
    averted_sick_by_age = {ag: [] for ag in HIRA_AGE_GROUPS}
    averted_school_by_age = {ag: [] for ag in HIRA_AGE_GROUPS}

    for i in idx:
        beta_4 = jnp.asarray(beta_samples[i])
        inc_b, pred_b = run_forward(beta_4, phi_full_j, gamma_15_j,
                                       P_SCHOOL_BASE, P_WORK_BASE, shared,
                                       setup)
        inc_s, pred_s = run_forward(beta_4, phi_full_j, gamma_15_j,
                                       P_SCHOOL_BASE, P_WORK_SICK, shared,
                                       setup)
        inc_c, pred_c = run_forward(beta_4, phi_full_j, gamma_15_j,
                                       P_SCHOOL_ABSENCE, P_WORK_BASE, shared,
                                       setup)
        tb = float(np.asarray(inc_b).sum())
        ts = float(np.asarray(inc_s).sum())
        tc = float(np.asarray(inc_c).sum())
        averted_sick_totals.append(100.0 * (tb - ts) / max(tb, 1e-9))
        averted_school_totals.append(100.0 * (tb - tc) / max(tb, 1e-9))

        pb = np.asarray(pred_b); ps = np.asarray(pred_s); pc = np.asarray(pred_c)
        for ai, ag in enumerate(HIRA_AGE_GROUPS):
            bb = float(pb[:, ai].sum())
            averted_sick_by_age[ag].append(
                100.0 * (bb - float(ps[:, ai].sum())) / max(bb, 1e-9))
            averted_school_by_age[ag].append(
                100.0 * (bb - float(pc[:, ai].sum())) / max(bb, 1e-9))

    def summ(vals):
        v = np.array(vals)
        return dict(mean=float(np.mean(v)),
                    q05=float(np.quantile(v, 0.05)),
                    q95=float(np.quantile(v, 0.95)),
                    samples=vals)

    return dict(
        n_draws_used=int(len(idx)),
        averted_sick_total=summ(averted_sick_totals),
        averted_school_total=summ(averted_school_totals),
        averted_sick_by_age={ag: summ(averted_sick_by_age[ag])
                             for ag in HIRA_AGE_GROUPS},
        averted_school_by_age={ag: summ(averted_school_by_age[ag])
                                for ag in HIRA_AGE_GROUPS},
    )


def save_json(all_results):
    # Drop the internal _samples fields when writing summary (kept only in
    # memory during posterior policy). Keep policy samples for CI plotting.
    def clean(rec):
        # Deep-copy nuts to avoid mutating original; phase 3 needs _samples
        out = dict(rec)
        nu = out.get("nuts")
        if nu is not None:
            out["nuts"] = {k: v for k, v in nu.items() if not k.startswith("_")}
        return out
    payload = dict(all_results)
    payload["combos"] = [clean(r) for r in all_results["combos"]]
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    print("=" * 78, flush=True)
    print(f"PIN NUTS POLICY  —  {SEASON_LABEL}", flush=True)
    print(f"  XLA_FLAGS = {os.environ.get('XLA_FLAGS','(not set)')}", flush=True)
    print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS','?')}",
          flush=True)
    print(f"  jax.device_count() = {jax.device_count()}", flush=True)
    print(f"  jax.devices() = {jax.devices()}", flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()
    all_results = dict(
        season=SEASON_LABEL,
        config=dict(
            AMP=AMP, HOLIDAY=HOLIDAY,
            PHI_USHAPE=PHI_USHAPE.tolist(),
            GAMMA_CENTER=list(GAMMA_CENTER),
            SIGMA_PER_CHANNEL=SIGMA_PER_CHANNEL.tolist(),
            A_REL=A_REL.tolist(), A_REL_NORM=A_REL_NORM.tolist(),
            combos=COMBOS,
            full_warmup=FULL_WARMUP, full_samples=FULL_SAMPLES,
            n_chains=N_CHAINS, target_accept=TARGET_ACCEPT,
            max_tree_depth=MAX_TREE_DEPTH,
            p_work_sick=P_WORK_SICK, p_school_absence=P_SCHOOL_ABSENCE,
        ),
        smoke=None,
        combos=[],
    )

    # ─── STEP 0 smoke ──────────────────────────────────────
    print("\n[STEP 0] smoke on C2 (work_share=0.06, κ=0.4) — pick faster "
          "chain_method", flush=True)
    c2 = COMBOS[1]
    print(f"    point estimate for C2 (needed for init)...", flush=True)
    c2_point = point_fit(c2["work_share"], c2["kappa"], setup)
    print(f"      NLL={c2_point['nll']:.4e}  R0={c2_point['R0']:.3f}",
          flush=True)

    smoke_records = {}
    method_wall = {}
    for method in ["sequential", "parallel"]:
        print(f"    smoke chain_method={method}  ({SMOKE_WARMUP}+{SMOKE_SAMPLES}"
              f" × {N_CHAINS} chains)...", flush=True)
        try:
            mcmc, wall = run_nuts(c2, method, SMOKE_WARMUP, SMOKE_SAMPLES,
                                     setup, c2_point, seed=201)
            smoke_records[method] = dict(wall_sec=float(wall), ok=True)
            method_wall[method] = wall
            print(f"      wall={wall:.1f}s", flush=True)
        except Exception as e:
            smoke_records[method] = dict(error=str(e), ok=False)
            print(f"      [warn] smoke {method} failed: {e}", flush=True)

    # Pick method
    chosen = None
    if all(m in method_wall for m in ["sequential", "parallel"]):
        if method_wall["parallel"] < method_wall["sequential"]:
            chosen = "parallel"
            speedup = method_wall["sequential"] / method_wall["parallel"]
            print(f"    → parallel faster by {speedup:.2f}x → "
                  f"chosen chain_method=parallel", flush=True)
        else:
            chosen = "sequential"
            print(f"    → parallel not faster → chosen chain_method=sequential",
                  flush=True)
    elif "sequential" in method_wall:
        chosen = "sequential"
    elif "parallel" in method_wall:
        chosen = "parallel"
    else:
        chosen = "sequential"
        print(f"    [warn] both smoke runs failed — defaulting to sequential",
              flush=True)

    # Full-run wall prediction
    smoke_iters = SMOKE_WARMUP + SMOKE_SAMPLES
    full_iters = FULL_WARMUP + FULL_SAMPLES
    if chosen in method_wall:
        wall_smoke = method_wall[chosen]
        # linear scale on iters; parallel already accounts for chain overhead
        pred_full = wall_smoke * (full_iters / smoke_iters)
        print(f"    predicted full-run wall for one combo "
              f"({chosen}, {full_iters} iters): {pred_full:.0f}s "
              f"= {pred_full/60:.1f}min = {pred_full/3600:.2f}h", flush=True)
    else:
        pred_full = None

    all_results["smoke"] = dict(
        C2_point={k: v for k, v in c2_point.items() if k != "per_age"},
        methods=smoke_records,
        chosen_method=chosen,
        predicted_full_wall_sec=pred_full,
    )
    save_json(all_results)

    # ─── STEP 1 full ──────────────────────────────────────
    for combo in COMBOS:
        name = combo["name"]
        print(f"\n{'#' * 60}\n#  COMBO {name}   ws={combo['work_share']} "
              f"κ={combo['kappa']}\n{'#' * 60}", flush=True)
        combo_rec = dict(name=name, work_share=combo["work_share"],
                          kappa=combo["kappa"],
                          chain_method=chosen,
                          point=None, nuts=None, policy=None, errors=[])
        all_results["combos"].append(combo_rec)
        save_json(all_results)

        try:
            print(f"  [phase 1] point estimate ...", flush=True)
            pt = point_fit(combo["work_share"], combo["kappa"], setup)
            combo_rec["point"] = pt
            print(f"    NLL={pt['nll']:.4e}  R0={pt['R0']:.3f}  "
                  f"β={[round(b,4) for b in pt['beta_4']]}", flush=True)
            print(f"    π = {[round(x,4) for x in pt['pi']]}  "
                  f"target = {[round(x,4) for x in pt['pi_target']]}",
                  flush=True)
            save_json(all_results)
        except Exception as e:
            combo_rec["errors"].append(f"phase 1 failed: {e}")
            print(f"    [ERROR] {e}", flush=True)
            save_json(all_results)
            continue

        try:
            print(f"  [phase 2] NUTS ({chosen}, "
                  f"{FULL_WARMUP}+{FULL_SAMPLES} × {N_CHAINS}) ...", flush=True)
            seed = {"C1": 301, "C2": 303, "C3": 307}[name]
            mcmc, wall = run_nuts(combo, chosen, FULL_WARMUP, FULL_SAMPLES,
                                     setup, pt, seed=seed)
            nu = summarize_nuts(mcmc)
            nu["wall_sec"] = float(wall)
            combo_rec["nuts"] = nu
            print(f"    max r_hat={nu['rhat_max']:.3f}  "
                  f"ess_min={nu['ess_min']:.1f}  div={nu['n_divergent']}  "
                  f"wall={wall:.0f}s = {wall/60:.1f}min", flush=True)
            print(f"    R0: mean={nu['R0']['mean']:.3f}  "
                  f"[90% {nu['R0']['q05']:.3f}, {nu['R0']['q95']:.3f}]",
                  flush=True)
            for c, ch in enumerate(["home", "work", "school", "other"]):
                s = nu["pi_summary"][c]
                print(f"    π_{ch}: {s['mean']:.4f}  "
                      f"[{s['q05']:.4f}, {s['q95']:.4f}]  "
                      f"target={pt['pi_target'][c]:.4f}", flush=True)
            save_json(all_results)
        except Exception as e:
            combo_rec["errors"].append(f"phase 2 failed: {e}")
            print(f"    [ERROR NUTS] {e}", flush=True)
            save_json(all_results)
            continue

        try:
            print(f"  [phase 3] posterior policy (200 draws) ...", flush=True)
            pol = posterior_policy(nu, combo, setup, n_draws=200)
            combo_rec["policy"] = pol
            s = pol["averted_sick_total"]
            print(f"    ★ averted sick total: {s['mean']:+.2f}%  "
                  f"[90% {s['q05']:+.2f}, {s['q95']:+.2f}]"
                  f"   CI-includes-0 = {s['q05'] < 0 < s['q95']}", flush=True)
            sc = pol["averted_school_total"]
            print(f"    averted school total: {sc['mean']:+.2f}%  "
                  f"[90% {sc['q05']:+.2f}, {sc['q95']:+.2f}]", flush=True)
            print(f"    per-age sick averted [mean (90% CI)]:", flush=True)
            for ag in HIRA_AGE_GROUPS:
                a = pol["averted_sick_by_age"][ag]
                sign_flip = a["q05"] < 0 < a["q95"]
                mark = "★" if sign_flip else " "
                print(f"      {mark} {ag:>6s}: {a['mean']:>+7.2f}%  "
                      f"[{a['q05']:>+7.2f}, {a['q95']:>+7.2f}]", flush=True)
            save_json(all_results)
        except Exception as e:
            combo_rec["errors"].append(f"phase 3 failed: {e}")
            print(f"    [ERROR policy] {e}", flush=True)
            save_json(all_results)

    save_json(all_results)

    # ─── Figure ─────
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        combos_ok = [r for r in all_results["combos"] if r.get("policy")]
        colors = {"C1": "#c0392b", "C2": "#f39c12", "C3": "#27ae60"}

        # Panel 1: averted sick posterior histograms (or violin-ish)
        ax = axes[0]
        for r in combos_ok:
            samples = r["policy"]["averted_sick_total"]["samples"]
            label = (f"{r['name']} (ws={r['work_share']}, κ={r['kappa']})  "
                     f"mean={r['policy']['averted_sick_total']['mean']:+.1f}%")
            ax.hist(samples, bins=30, alpha=0.55,
                     color=colors[r["name"]], label=label, density=True)
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set_xlabel("averted sick_leave total %")
        ax.set_ylabel("density")
        ax.set_title("sick_leave averted posterior")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Panel 2: R0 posterior mean + CI per combo
        ax = axes[1]
        for i, r in enumerate(combos_ok):
            R0 = r["nuts"]["R0"]
            ax.errorbar([i], [R0["mean"]],
                         yerr=[[R0["mean"] - R0["q05"]],
                                [R0["q95"] - R0["mean"]]],
                         fmt="o", ms=10, capsize=5,
                         color=colors[r["name"]], label=r["name"])
        ax.set_xticks(range(len(combos_ok)))
        ax.set_xticklabels([r["name"] for r in combos_ok])
        ax.set_ylabel("R0")
        ax.set_title("R0 posterior mean + 90% CI")
        ax.grid(True, alpha=0.3)

        # Panel 3: π posterior vs target (bars)
        ax = axes[2]
        channels = ["home", "work", "school", "other"]
        xa = np.arange(4)
        bw = 0.2
        for i, r in enumerate(combos_ok):
            m = [r["nuts"]["pi_summary"][c]["mean"] for c in range(4)]
            los = [r["nuts"]["pi_summary"][c]["q05"] for c in range(4)]
            his = [r["nuts"]["pi_summary"][c]["q95"] for c in range(4)]
            yerr = np.array([[m[c] - los[c] for c in range(4)],
                              [his[c] - m[c] for c in range(4)]])
            ax.bar(xa + (i - 1) * bw, m, bw, yerr=yerr, capsize=3,
                    color=colors[r["name"]], alpha=0.75,
                    label=r["name"])
        # Target markers (using first combo's target — but each combo differs; show all)
        for i, r in enumerate(combos_ok):
            tgt = build_pi_target(r["work_share"]).tolist()
            ax.scatter(xa + (i - 1) * bw, tgt, marker="D", s=50,
                        edgecolor="k", facecolor="white",
                        label=f"{r['name']} target")
        ax.set_xticks(xa); ax.set_xticklabels(channels)
        ax.set_ylabel("π")
        ax.set_title("π posterior vs target")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"pin NUTS policy — {SEASON_LABEL}  "
                      f"(φ U-shape, R(0) default, A-fix cov)")
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
        print(f"\nsaved {OUT_FIG}", flush=True)
    except Exception as e:
        print(f"\n[ERROR figure] {e}", flush=True)

    print(f"\nsaved {OUT_JSON}", flush=True)
    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
