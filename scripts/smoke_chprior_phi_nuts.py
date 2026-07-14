"""NUTS smoke: channel prior pin + φ sampling with 2nd-order smoothing.

Single season 2019-2020. Purpose: verify convergence (r_hat, divergences,
ESS) of a model that couples channel-prior pin (σ_hw=0.01 on home/work) with
φ_14 sampling under a mild 2nd-order smoothing factor. NOT a final estimate;
this is a pipeline + convergence check before any full production run.

Local model definition — production hira_model_nb_chprior is NOT modified.
This script defines a smoke-only NUTS model inline.

Sampled parameters (single season, 20 total):
  log_R0 : TruncatedNormal(log 2, 0.4, [log 0.8, log 3])
  logit_pi (4) : Normal(0, 2.0)  — base + centered ch_prior factor
  phi_14 : TruncatedNormal(1, 0.5, [0.1, 5.0]) per age (idx 5 anchor ≡ 1.0)
  phi_nb : HalfNormal(10.0)  — same as production
Factors (numpyro.factor):
  ch_prior   = -0.5 · Σ (centered_logit_pi − logit_target)² / σ²
  phi_smooth = -λ · Σ (φ_full[i+1] − 2·φ_full[i] + φ_full[i-1])²
  likelihood = -NB_NLL_jax
Fixed: γ=CDC, R(0) immunity default, κ default, HOLIDAY realloc=1 amp=0.7,
AMP=0.9, σ=0.5, γ=0.25, n_admdong=1.

Sweep: 2 targets (T_B, T_lit) × σ_hw = 0.01. σ pin only (strength sweep not
needed for a convergence smoke).

NUTS: 2 chains, warmup 500, sample 500, target_accept_prob=0.9, max_tree_depth=10,
sequential chain method, fixed seed.

Decision guide (comments only — script does not interpret):
- max r_hat < 1.10 + divergences < few dozen → converged, proceed to full run.
- r_hat large or divergences explode → φ sampling blocks convergence;
  tighten φ prior or step back to φ fixed.
- π_work posterior narrow near target → pin working.
- φ adult/15-19 posterior wide → non-id honestly reflected.
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
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "smoke_chprior_phi.png"
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

T_LIT_R0CONTRIB = np.array([0.35, 0.05, 0.20, 0.40])   # (h, w, s, o)
T_B_R0CONTRIB = np.array([0.40, 0.10, 0.27, 0.23])

SIGMA_HW = 0.01
SIGMA_SO = 0.30
LAMBDA_PHI = 0.1
REF_AGE_IDX = 5

# NUTS smoke config
N_CHAINS = 2
N_WARMUP = 500
N_SAMPLES = 500
TARGET_ACCEPT = 0.9
MAX_TREE_DEPTH = 10


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
    """Local smoke model — replicates production hira_model_nb_chprior structure
    but with φ_14 sampled and a 2nd-order φ smoothing factor added.
    Production numpyro_model.py is NOT modified."""
    lt = jnp.asarray(logit_target)
    inv_var = 1.0 / (jnp.asarray(sigma_per_channel) ** 2)

    obs_j = setup["obs_j"]
    w_j = setup["w_j"]
    gamma_15 = setup["gamma_15"]
    state0 = setup["state0"]
    shared = setup["shared"]
    ngm_fn = setup["ngm_fn"]
    n_weeks = setup["n_weeks"]

    def model():
        log_R0 = numpyro.sample(
            "log_R0",
            dist.TruncatedNormal(
                jnp.log(2.0), 0.4,
                low=jnp.log(0.8), high=jnp.log(3.0),
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

        phi_14 = numpyro.sample(
            "phi_14",
            dist.TruncatedNormal(
                1.0, 0.5, low=0.1, high=5.0,
            ).expand([14]).to_event(1),
        )
        phi_full = jnp.concatenate([phi_14[:REF_AGE_IDX], jnp.array([1.0]),
                                     phi_14[REF_AGE_IDX:]])
        # 2nd-order smoothing on FULL φ (interior 13 curvatures)
        curv = phi_full[2:] - 2.0 * phi_full[1:-1] + phi_full[:-2]
        numpyro.factor("phi_smooth", -LAMBDA_PHI * jnp.sum(curv * curv))

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
        numpyro.deterministic("phi_full", phi_full)

    return model


def summarize_run(mcmc, label: str, target_pi: np.ndarray, setup) -> dict:
    samples = mcmc.get_samples(group_by_chain=True)
    extras = mcmc.get_extra_fields(group_by_chain=True)
    n_div = int(np.asarray(extras["diverging"]).sum())

    # Arviz r_hat / ess (skip forward-pred determ arrays that stress arviz)
    idata = az.from_numpyro(mcmc)
    diag_vars = ["log_R0", "logit_pi", "phi_14", "phi_nb", "R0", "pi", "beta_4"]
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
    pi_s = flat(samples["pi"])                # (n_draws, 4)
    phi_full_s = flat(samples["phi_full"])    # (n_draws, 15)
    phi_nb_s = flat(samples["phi_nb"])
    beta4_s = flat(samples["beta_4"])         # (n_draws, 4)

    def summary(x):
        return dict(mean=float(np.mean(x)),
                    q05=float(np.quantile(x, 0.05)),
                    q95=float(np.quantile(x, 0.95)))

    pi_summary = [
        {**summary(pi_s[:, c]), "target": float(target_pi[c])}
        for c in range(4)
    ]
    phi_summary = [summary(phi_full_s[:, a]) for a in range(15)]
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
        phi_full_summary=phi_summary,
    )


def main():
    print("=" * 78)
    print(f"NUTS smoke: chprior pin (σ_hw={SIGMA_HW}) + φ sampling  —  "
          f"{SEASON_LABEL}")
    print(f"  chains={N_CHAINS}  warmup={N_WARMUP}  sample={N_SAMPLES}  "
          f"target_accept={TARGET_ACCEPT}  max_tree_depth={MAX_TREE_DEPTH}")
    print(f"  φ sampled (TruncNormal(1, 0.5, [0.1, 5.0])) + smoothing λ={LAMBDA_PHI}")
    print("=" * 78)

    setup = build_setup()
    print(f"  n_weeks = {setup['n_weeks']}   obs sum = "
          f"{float(np.asarray(setup['obs_j']).sum()):,.0f}")

    combos = [("T_B", T_B_R0CONTRIB), ("T_lit", T_LIT_R0CONTRIB)]
    all_results = []
    for label, tgt in combos:
        pi_target = r0contrib_to_pi(tgt)
        lt = logit_centered_target(pi_target)
        sig = np.array([SIGMA_HW, SIGMA_HW, SIGMA_SO, SIGMA_SO])
        print(f"\n── smoke run [{label}] ──")
        print(f"    target π = {[round(float(x),4) for x in pi_target]}")
        print(f"    σ_per_channel = {sig.tolist()}")
        model = build_smoke_model(lt, sig, setup)
        seed = 41 if label == "T_B" else 47

        kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT,
                       max_tree_depth=MAX_TREE_DEPTH, dense_mass=False)
        mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                     num_chains=N_CHAINS, chain_method="sequential",
                     progress_bar=False)
        t0 = time.perf_counter()
        mcmc.run(random.PRNGKey(seed), extra_fields=("diverging",))
        wall = time.perf_counter() - t0
        print(f"    wall: {wall:.0f}s ({wall/60:.1f}min)")

        rec = summarize_run(mcmc, label, pi_target, setup)
        rec["wall_sec"] = float(wall)
        rec["pi_target"] = pi_target.tolist()
        rec["R0_contrib_target"] = tgt.tolist()
        rec["config"] = dict(
            n_chains=N_CHAINS, n_warmup=N_WARMUP, n_samples=N_SAMPLES,
            target_accept=TARGET_ACCEPT, max_tree_depth=MAX_TREE_DEPTH,
            sigma_per_channel=sig.tolist(), lambda_phi=LAMBDA_PHI,
        )
        all_results.append(rec)

        print(f"    max r_hat = {rec['rhat_max']:.3f}   ess_min = {rec['ess_min']:.0f}"
              f"   divergences = {rec['n_divergent']}")
        print(f"    R0: mean={rec['R0']['mean']:.3f}  [90%: "
              f"{rec['R0']['q05']:.3f}, {rec['R0']['q95']:.3f}]")
        print(f"    phi_nb: mean={rec['phi_nb']['mean']:.3f}  [90%: "
              f"{rec['phi_nb']['q05']:.3f}, {rec['phi_nb']['q95']:.3f}]")
        for c, ch in enumerate(["home", "work", "school", "other"]):
            s = rec["pi_summary"][c]
            print(f"    π_{ch}: mean={s['mean']:.4f}  target={s['target']:.4f}  "
                  f"[90%: {s['q05']:.4f}, {s['q95']:.4f}]")
        for c, ch in enumerate(["home", "work", "school", "other"]):
            s = rec["beta_summary"][c]
            print(f"    β_{ch}: mean={s['mean']:.4f}  [90%: "
                  f"{s['q05']:.4f}, {s['q95']:.4f}]")
        for a, s in enumerate(rec["phi_full_summary"]):
            age_lbl = f"{5*a}-{5*a+4}" if a < 14 else "70+"
            print(f"    φ[{a:>2}] {age_lbl:>6s}: mean={s['mean']:.3f}  "
                  f"[90%: {s['q05']:.3f}, {s['q95']:.3f}]")
        print(f"    r_hat per var: {rec['rhat_per_var']}")

        json_path = OUT_DIR / f"smoke_chprior_phi_{label}.json"
        with open(json_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"    saved {json_path}")

    # ─── Combined figure ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    channels = ["home", "work", "school", "other"]
    colors = {"T_B": "#c0392b", "T_lit": "#1a5490"}

    # Panel 1: π posterior mean + 90% CI, target markers
    ax = axes[0]
    x = np.arange(4)
    w = 0.35
    for i, rec in enumerate(all_results):
        means = [rec["pi_summary"][c]["mean"] for c in range(4)]
        los = [rec["pi_summary"][c]["q05"] for c in range(4)]
        his = [rec["pi_summary"][c]["q95"] for c in range(4)]
        yerr = np.array([[m - lo for m, lo in zip(means, los)],
                         [hi - m for m, hi in zip(means, his)]])
        offset = (i - 0.5) * w
        ax.bar(x + offset, means, w, yerr=yerr, capsize=4,
                color=colors[rec["label"]], alpha=0.7,
                label=f"{rec['label']} posterior")
        ax.scatter(x + offset, rec["pi_target"], marker="D", s=80,
                    edgecolor="k", facecolor="white",
                    label=f"{rec['label']} target")
    ax.set_xticks(x); ax.set_xticklabels(channels)
    ax.set_ylabel("π (β share)")
    ax.set_title("π posterior mean + 90% CI  vs  target")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: φ_full posterior band per run
    ax = axes[1]
    ages_idx = np.arange(15)
    for rec in all_results:
        means = np.array([s["mean"] for s in rec["phi_full_summary"]])
        los = np.array([s["q05"] for s in rec["phi_full_summary"]])
        his = np.array([s["q95"] for s in rec["phi_full_summary"]])
        ax.plot(ages_idx, means, "-o", color=colors[rec["label"]], lw=1.6,
                 label=f"{rec['label']} mean")
        ax.fill_between(ages_idx, los, his, color=colors[rec["label"]],
                          alpha=0.25)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.axvline(REF_AGE_IDX, color="red", ls=":", lw=1)
    ax.axvline(3, color="magenta", ls=":", lw=1)
    ax.set_xticks(ages_idx)
    ax.set_xticklabels([f"{5*i}-{5*i+4}" for i in range(14)] + ["70+"],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("φ_full")
    ax.set_title("φ_full posterior mean + 90% CI")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: r_hat max + divergence summary bar
    ax = axes[2]
    labels = [rec["label"] for rec in all_results]
    rhat_vals = [rec["rhat_max"] for rec in all_results]
    div_vals = [rec["n_divergent"] for rec in all_results]
    ess_vals = [rec["ess_min"] for rec in all_results]
    x2 = np.arange(len(labels))
    ax.bar(x2 - 0.25, rhat_vals, 0.25, label="max r_hat", color="#1a5490")
    ax.bar(x2, [d / 100 for d in div_vals], 0.25,
            label="divergences / 100", color="#c0392b")
    ax.bar(x2 + 0.25, [e / 1000 for e in ess_vals], 0.25,
            label="ess_min / 1000", color="#27ae60")
    ax.axhline(1.1, color="grey", ls=":", lw=1, label="r_hat 1.10 threshold")
    ax.set_xticks(x2); ax.set_xticklabels(labels)
    ax.set_ylabel("value (see rescaling in legend)")
    ax.set_title("Diagnostics summary")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"NUTS smoke — chprior σ_hw={SIGMA_HW} + φ sampling  "
                  f"({SEASON_LABEL})")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"\nsaved {OUT_FIG}")


if __name__ == "__main__":
    main()
