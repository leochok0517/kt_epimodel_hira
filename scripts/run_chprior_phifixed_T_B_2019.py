"""NUTS smoke: channel prior pin + φ FIXED at docs/parameter_justification.md A.2
literature U-shape.  Single season 2019-2020.

Motivation: previous smoke (φ sampled) gave max r_hat 3.03, ESS 2, R0=1.05
after 5h33m — φ non-identifiability blocked NUTS. Fix φ to the literature
U-shape and only sample R0 + logit_pi + phi_nb.

φ (fixed, 15-vector, from docs/parameter_justification.md A.2):
    PHI_FIXED = [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
                 1.05, 1.1, 1.2, 1.3, 1.4, 1.5]

Local smoke model — production hira_model_nb_chprior is NOT modified.

Sampled (all scalar / 4-dim):
  log_R0  : TruncatedNormal(log 2, 0.4, [log 0.8, log 3])
  logit_pi (4) : Normal(0, 2.0) + centered ch_prior factor toward logit_target
  phi_nb  : HalfNormal(10.0)
Fixed: φ_full=PHI_FIXED, γ=CDC, R(0) immunity default, κ default, HOLIDAY,
AMP=0.9, σ=0.5, γ=0.25 (disease).

Sweep: 2 targets (T_B, T_lit) × σ_hw=0.01 (pin). Separate JSON per target
so a stuck T_lit does not lose T_B result.

NUTS: 2 chains, warmup 100, sample 100, target_accept=0.9, max_tree_depth=8,
progress_bar=True (mandatory — previous run gave no live feedback).

Decision guide (comments only):
- max r_hat < 1.10, few divergences, R0 ~ 1.3-2.5 → converged with φ fixed.
- R0 stuck low (<1.2) or r_hat large → φ fix alone insufficient; investigate
  channel or other-channel structure.
- T_B ok, T_lit runaway → T_lit work=0.05 is at floor edge; abandon T_lit,
  proceed with T_B.
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
OUT_DIR = REPO_ROOT / "outputs" / "eda"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "run_T_B_2019.png"
# Full-run output paths (single season 2019-2020, T_B only)
V2_JSON_PREFIX = "run_T_B_2019"
NC_PATH = REPO_ROOT / "outputs" / "eda" / "run_T_B_2019_posterior.nc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
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
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])
UNIT_R0 = np.array([8.70, 6.21, 25.40, 9.33])

# ★ φ FIXED — from docs/parameter_justification.md A.2 (literature U-shape)
PHI_FIXED = np.array([
    2.0, 1.9, 1.7, 1.4, 1.1,   # idx 0-4  (0-24y): high susceptibility
    1.0, 1.0, 1.0, 1.0,         # idx 5-8  (25-44y): reference
    1.05, 1.1, 1.2, 1.3,        # idx 9-12 (45-64y): rising
    1.4, 1.5,                    # idx 13-14 (65+):  elderly high
], dtype=np.float64)
assert PHI_FIXED.shape == (15,)

T_LIT_R0CONTRIB = np.array([0.35, 0.05, 0.20, 0.40])   # (h, w, s, o)
T_B_R0CONTRIB = np.array([0.40, 0.10, 0.27, 0.23])

SIGMA_HW = 0.01
SIGMA_SO = 0.30

# NUTS full-run config
N_CHAINS = 4
N_WARMUP = 1000
N_SAMPLES = 1000
TARGET_ACCEPT = 0.9
MAX_TREE_DEPTH = 8


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
        p_school=policy.p_school, p_work=policy.p_work,
        VE=vax.VE,
        annual_coverage=jnp.asarray(vax.annual_coverage),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared.update(HOLIDAY)
    gamma_15 = jnp.asarray(GAMMA_CDC)

    seed_15 = estimate_initial_infected_from_hira(
        SEASON_LABEL, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    tgt = load_hira_target_by_age(
        SEASON_LABEL, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    n_weeks = tgt["n_weeks"]
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]

    ngm_fn = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )
    return dict(
        shared=shared, gamma_15=gamma_15, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_fn=ngm_fn,
    )


def build_smoke_model(logit_target: np.ndarray, sigma_per_channel: np.ndarray,
                       setup):
    """Local smoke model — φ FIXED at PHI_FIXED, no φ sampling."""
    lt = jnp.asarray(logit_target)
    inv_var = 1.0 / (jnp.asarray(sigma_per_channel) ** 2)
    phi_full = jnp.asarray(PHI_FIXED)

    obs_j = setup["obs_j"]
    w_j = setup["w_j"]
    gamma_15 = setup["gamma_15"]
    state0 = setup["state0"]
    shared = setup["shared"]
    ngm_fn = setup["ngm_fn"]
    n_weeks = setup["n_weeks"]

    def model():
        # v2 change #2: R0 lower bound log(0.8) → log(1.1) to escape sub-critical
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

        beta_4 = derive_beta_from_R0_simplex(ngm_fn, R0, pi, phi_full)

        kw = dict(shared)
        kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
        kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
        kw["phi_susc"] = phi_full
        st = simulate_jax(state0, **kw, discretize_time=False)
        inc = daily_new_infection_by_age_jax(st)
        pred = simulation_to_hira_by_age_jax(inc, gamma_15, n_weeks=n_weeks)

        nll = nb_nll_jax(obs_j, pred, w_j,
                         concentration=phi_nb, min_rate=0.01)
        numpyro.factor("likelihood_nb", -nll)

        numpyro.deterministic("R0", R0)
        numpyro.deterministic("pi", pi)
        numpyro.deterministic("beta_4", beta_4)

    return model


def summarize_run(mcmc, label: str, target_pi: np.ndarray) -> dict:
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
        print(f"    [warn] rhat failed: {e}")
    try:
        es = az.ess(idata, var_names=diag_vars)
        ess_min = float(np.nanmin([np.nanmin(es[v].values)
                                     for v in es.data_vars]))
    except Exception as e:
        print(f"    [warn] ess failed: {e}")

    def flat(arr):
        arr = np.asarray(arr)
        return arr.reshape(-1, *arr.shape[2:])

    R0_s = flat(samples["R0"])
    pi_s = flat(samples["pi"])
    phi_nb_s = flat(samples["phi_nb"])
    beta4_s = flat(samples["beta_4"])

    def summary(x):
        return dict(mean=float(np.mean(x)),
                    q05=float(np.quantile(x, 0.05)),
                    q95=float(np.quantile(x, 0.95)))

    pi_summary = [
        {**summary(pi_s[:, c]), "target": float(target_pi[c])}
        for c in range(4)
    ]
    beta_summary = [summary(beta4_s[:, c]) for c in range(4)]

    return dict(
        label=label,
        n_divergent=n_div,
        rhat_max=rhat_max,
        rhat_per_var=rhat_per,
        ess_min=ess_min,
        R0=summary(R0_s),
        phi_nb=summary(phi_nb_s),
        pi_summary=pi_summary,
        beta_summary=beta_summary,
        phi_fixed_15=PHI_FIXED.tolist(),
    )


def run_target(label: str, r0contrib: np.ndarray, setup) -> dict:
    pi_target = r0contrib_to_pi(r0contrib)
    lt = logit_centered_target(pi_target)
    sig = np.array([SIGMA_HW, SIGMA_HW, SIGMA_SO, SIGMA_SO])
    print(f"\n── smoke run [{label}]  (φ FIXED literature U-shape) ──", flush=True)
    print(f"    target R0 contrib (h,w,s,o) = {r0contrib.tolist()}", flush=True)
    print(f"    target π                    = {[round(float(x),4) for x in pi_target]}",
          flush=True)
    print(f"    σ_per_channel               = {sig.tolist()}", flush=True)
    print(f"    φ_fixed (15)                = {[round(x,3) for x in PHI_FIXED]}",
          flush=True)

    model = build_smoke_model(lt, sig, setup)
    seed = 41 if label == "T_B" else 47

    # v2 change #1: init_to_value using point-estimate T_B σ=0.01 result
    # from outputs/eda/chprior_sweep.json (R0=1.970, pi as below, phi_nb=6.66).
    # Same init for both targets (safe: T_lit is skipped in v2 main()).
    pi_init = np.array([0.4760730985417175, 0.16706606247695396,
                         0.23754625946732033, 0.11931457951400816])
    logit_pi_init = np.log(pi_init)   # unnormalized; softmax invariant to shift
    init_vals = {
        "log_R0": jnp.asarray(np.log(1.970152264837932)),
        "logit_pi": jnp.asarray(logit_pi_init),
        "phi_nb": jnp.asarray(6.660951776633371),
    }

    kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT,
                   max_tree_depth=MAX_TREE_DEPTH, dense_mass=False,
                   init_strategy=init_to_value(values=init_vals))
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                 num_chains=N_CHAINS, chain_method="sequential",
                 progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(seed), extra_fields=("diverging",))
    wall = time.perf_counter() - t0
    print(f"    wall: {wall:.0f}s ({wall/60:.1f}min)", flush=True)

    rec = summarize_run(mcmc, label, pi_target)
    rec["wall_sec"] = float(wall)
    rec["pi_target"] = pi_target.tolist()
    rec["R0_contrib_target"] = r0contrib.tolist()
    rec["config"] = dict(
        n_chains=N_CHAINS, n_warmup=N_WARMUP, n_samples=N_SAMPLES,
        target_accept=TARGET_ACCEPT, max_tree_depth=MAX_TREE_DEPTH,
        sigma_per_channel=sig.tolist(),
    )

    print(f"    max r_hat = {rec['rhat_max']:.3f}   ess_min = {rec['ess_min']:.1f}"
          f"   divergences = {rec['n_divergent']}", flush=True)
    print(f"    R0: mean={rec['R0']['mean']:.3f}  [90%: "
          f"{rec['R0']['q05']:.3f}, {rec['R0']['q95']:.3f}]", flush=True)
    print(f"    phi_nb: mean={rec['phi_nb']['mean']:.3f}  [90%: "
          f"{rec['phi_nb']['q05']:.3f}, {rec['phi_nb']['q95']:.3f}]", flush=True)
    for c, ch in enumerate(["home", "work", "school", "other"]):
        s = rec["pi_summary"][c]
        print(f"    π_{ch}: mean={s['mean']:.4f}  target={s['target']:.4f}  "
              f"[90%: {s['q05']:.4f}, {s['q95']:.4f}]", flush=True)
    for c, ch in enumerate(["home", "work", "school", "other"]):
        s = rec["beta_summary"][c]
        print(f"    β_{ch}: mean={s['mean']:.4f}  [90%: "
              f"{s['q05']:.4f}, {s['q95']:.4f}]", flush=True)
    print(f"    r_hat per var: {rec['rhat_per_var']}", flush=True)

    # ★ Save JSON immediately per target so a subsequent stall doesn't lose data
    json_path = OUT_DIR / f"{V2_JSON_PREFIX}_{label}.json"
    with open(json_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"    saved {json_path}", flush=True)

    # Save full arviz posterior netcdf (T_B only)
    try:
        idata = az.from_numpyro(mcmc)
        idata.to_netcdf(str(NC_PATH))
        print(f"    saved {NC_PATH}", flush=True)
    except Exception as e:
        print(f"    [warn] netcdf save failed: {e}", flush=True)

    return rec


def main():
    print("=" * 78, flush=True)
    print(f"NUTS smoke: chprior pin σ_hw={SIGMA_HW} + φ FIXED (literature U)  —  "
          f"{SEASON_LABEL}", flush=True)
    print(f"  chains={N_CHAINS}  warmup={N_WARMUP}  sample={N_SAMPLES}  "
          f"target_accept={TARGET_ACCEPT}  max_tree_depth={MAX_TREE_DEPTH}",
          flush=True)
    print(f"  φ_full FIXED at PHI_FIXED (docs/parameter_justification.md A.2)",
          flush=True)
    print(f"  {'idx':>3s}  {'age':>6s}  {'φ':>6s}", flush=True)
    ages = [f"{5*i}-{5*i+4}" if i < 14 else "70+" for i in range(15)]
    for i, (a, p) in enumerate(zip(ages, PHI_FIXED)):
        print(f"  {i:>3d}  {a:>6s}  {p:>6.2f}", flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()
    print(f"  n_weeks = {setup['n_weeks']}   obs sum = "
          f"{float(np.asarray(setup['obs_j']).sum()):,.0f}", flush=True)

    all_results = []
    # v2: T_B only (T_lit skipped per instructions)
    for label, tgt in [("T_B", T_B_R0CONTRIB)]:
        try:
            rec = run_target(label, tgt, setup)
            all_results.append(rec)
        except Exception as e:
            print(f"    [ERROR] run {label} failed: {e}", flush=True)
            all_results.append(dict(label=label, error=str(e)))

    # ─── Combined figure (only over successfully completed runs) ─────
    ok = [r for r in all_results if "error" not in r]
    if not ok:
        print("no successful runs — skipping figure", flush=True)
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    channels = ["home", "work", "school", "other"]
    colors = {"T_B": "#c0392b", "T_lit": "#1a5490"}

    # Panel 1: π posterior + target
    ax = axes[0]
    x = np.arange(4)
    w = 0.35
    for i, rec in enumerate(ok):
        means = [rec["pi_summary"][c]["mean"] for c in range(4)]
        los = [rec["pi_summary"][c]["q05"] for c in range(4)]
        his = [rec["pi_summary"][c]["q95"] for c in range(4)]
        yerr = np.array([[m - lo for m, lo in zip(means, los)],
                         [hi - m for m, hi in zip(means, his)]])
        offset = (i - (len(ok) - 1) / 2) * w
        ax.bar(x + offset, means, w, yerr=yerr, capsize=4,
                color=colors.get(rec["label"], "#888"), alpha=0.7,
                label=f"{rec['label']} posterior")
        ax.scatter(x + offset, rec["pi_target"], marker="D", s=80,
                    edgecolor="k", facecolor="white",
                    label=f"{rec['label']} target")
    ax.set_xticks(x); ax.set_xticklabels(channels)
    ax.set_ylabel("π (β share)")
    ax.set_title("π posterior mean + 90% CI vs target")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: r_hat / divergences summary
    ax = axes[1]
    labels = [rec["label"] for rec in ok]
    rhat_vals = [rec["rhat_max"] for rec in ok]
    div_vals = [rec["n_divergent"] for rec in ok]
    ess_vals = [rec["ess_min"] for rec in ok]
    x2 = np.arange(len(labels))
    ax.bar(x2 - 0.25, rhat_vals, 0.25, label="max r_hat", color="#1a5490")
    ax.bar(x2, [d / 10 for d in div_vals], 0.25,
            label="divergences / 10", color="#c0392b")
    ax.bar(x2 + 0.25, [e / 100 for e in ess_vals], 0.25,
            label="ess_min / 100", color="#27ae60")
    ax.axhline(1.1, color="grey", ls=":", lw=1, label="r_hat 1.10 line")
    ax.set_xticks(x2); ax.set_xticklabels(labels)
    ax.set_ylabel("value (see rescaling)")
    ax.set_title("Convergence diagnostics")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: R0 posterior mean + CI
    ax = axes[2]
    for i, rec in enumerate(ok):
        m = rec["R0"]["mean"]; lo = rec["R0"]["q05"]; hi = rec["R0"]["q95"]
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]],
                     fmt="o", ms=10, capsize=5,
                     color=colors.get(rec["label"], "#888"),
                     label=f"{rec['label']}")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels([r["label"] for r in ok])
    ax.axhline(2.0, color="grey", ls=":", lw=1, label="typical seasonal ~2")
    ax.set_ylabel("R0 posterior mean + 90% CI")
    ax.set_title("R0 posterior")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"NUTS smoke — chprior + φ FIXED  ({SEASON_LABEL})")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"\nsaved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
