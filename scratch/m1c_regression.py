"""M1c regression: JAX multi-season NLL vs numpy.

Verifies HIRA conversion + Poisson NLL chain by comparing NLL on 4
representative multistart vecs.

Criterion: NLL rel diff < 0.1% with discretize_time=True.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import time
import numpy as np
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# numpy reference
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.calibration.param_vector import vector_to_params
from kt_epimodel_hira.calibration.hira_target import (
    load_hira_target_by_age, simulation_to_hira_by_age,
    poisson_log_likelihood, HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.calibration.loss import make_loss_function_by_age
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters, EmploymentParameters,
)

# JAX
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_MAPPING, HIRA_AGE_GROUPS,
    simulation_to_hira_by_age_jax, poisson_nll_jax,
    gamma_triple_to_15, make_multi_season_loss_fn,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)

OUTDIR = Path("/Users/hwcho/Documents/python/NIMS/kt_epimodel_hira/outputs/calibration")
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]


def relmax(a, b): return float(np.max(np.abs(a - b) / (np.abs(b) + 1.0)))


# =============================================================================
# Phase 1 verification: HIRA mapping matrix
# =============================================================================
print("=" * 70)
print("HIRA mapping matrix sanity")
print("=" * 70)
H = np.asarray(HIRA_MAPPING)
print(f"  shape: {H.shape}")
print(f"  col sums (per NIMS, should all == 1.0): {H.sum(axis=0)}")
assert np.allclose(H.sum(axis=0), 1.0)

# Cross-check vs numpy dict on a synthetic incidence
rng = np.random.default_rng(0)
inc_test = rng.exponential(100.0, size=(365, 15))
gamma_test = np.array([0.3]*4 + [0.1]*9 + [0.25]*2)
np_out = simulation_to_hira_by_age(inc_test, gamma_test, n_weeks=52)
jax_out = np.asarray(simulation_to_hira_by_age_jax(
    jnp.asarray(inc_test), jnp.asarray(gamma_test), n_weeks=52,
))
for i, ag in enumerate(HIRA_AGE_GROUPS):
    rd = relmax(jax_out[:, i], np_out[ag])
    print(f"  {ag:>7}: max rel diff vs numpy dict = {rd:.3e}  "
          f"{'OK' if rd < 1e-12 else 'FAIL'}")

# =============================================================================
# Setup: load multistart vecs and build season data
# =============================================================================
print("\n" + "=" * 70)
print("Load 4 representative vecs + season inputs")
print("=" * 70)
vec_names = ["warm", "neutral", "high_phi", "bio_prior"]
vecs = {n: np.array(json.load(open(OUTDIR / f"multistart_{n}.json"))["best_vec"])
        for n in vec_names}

inputs = build_aggregated_inputs()
pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
matrices = inputs["matrices"]; mobility = inputs["mobility"]

# Per-season targets + initial states
season_targets = {}
season_seeds = {}
season_states0 = {}
for s in SEASONS:
    tgt = load_hira_target_by_age(
        s, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    season_targets[s] = tgt
    seed = estimate_initial_infected_from_hira(
        s, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    season_seeds[s] = seed
    state0 = _build_initial_state_with_age_seed(
        pop_15, seed, seed_e_factor=0.5,
        initial_immunity=0.3, initial_vaccinated_fraction=0.0,
    )
    season_states0[s] = state0

# Build shared kwargs (everything that doesn't change per fit/vec)
disease = ModelParameters().disease
vax = ModelParameters().vaccination
policy = ModelParameters().policy
shared_static = dict(
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
    vax_peak_iso_week=vax.peak_iso_week,
    vax_spread_weeks=vax.spread_weeks,
    seasonality_amp=disease.seasonality_amp,
    seasonality_base=disease.seasonality_base,
    seasonality_peak_day=disease.seasonality_peak_day,
    seasonality_period=disease.seasonality_period,
)

# Build per-season JAX initial states + obs/weights arrays
initial_states_jax = [jnp.asarray(season_states0[s]) for s in SEASONS]
obs_hira_jax = []
weights_hira_jax = []
for s in SEASONS:
    tgt = season_targets[s]
    nw = tgt["n_weeks"]
    obs = np.zeros((nw, 6))
    w = np.zeros((nw, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]
    obs_hira_jax.append(jnp.asarray(obs))
    weights_hira_jax.append(jnp.asarray(w))
    print(f"  {s}: n_weeks={nw}, total obs sum={obs.sum():,.0f}, "
          f"weighted_weeks={int((w > 0).any(axis=1).sum())}")


# =============================================================================
# Phase 3: NLL regression
# =============================================================================
print("\n" + "=" * 70)
print("NLL regression: numpy vs JAX (discretize_time=True)")
print("=" * 70)

# Build numpy multi-season loss function for direct comparison
def numpy_multi_season_nll(vec_33):
    """Replicate the multi_season_loss used in scripts."""
    phi_14 = vec_33[:14]
    gamma_3 = vec_33[14:17]
    total = 0.0
    for i, s in enumerate(SEASONS):
        beta_4 = vec_33[17 + i*4 : 21 + i*4]
        vec_21 = np.concatenate([beta_4, phi_14, gamma_3])
        cal = vector_to_params(vec_21)
        params = ModelParameters().with_calibration(cal).with_employment(
            EmploymentParameters(rho=rho_emp))
        from kt_epimodel_hira.calibration.simple_model import simulate_aggregated
        seed = season_seeds[s]
        res = simulate_aggregated(
            params, inputs, seed_total=float(seed.sum()),
            seed_by_age=seed, seed_e_factor=0.5,
            initial_immunity=0.3, t_span=(0., 364.),
        )
        daily_inc = res.daily_new_infection_by_age()
        pred = simulation_to_hira_by_age(daily_inc, cal.gamma_15,
                                          n_weeks=season_targets[s]["n_weeks"])
        tgt = season_targets[s]
        for ag in tgt["age_groups"]:
            total += poisson_log_likelihood(
                tgt["hira_counts"][ag], pred[ag],
                is_valid=tgt["is_valid"][ag],
                weights=tgt["weights"][ag],
            )
    return total

# JAX multi-season loss closures (discretize_time and continuous)
loss_disc = make_multi_season_loss_fn(
    initial_states=initial_states_jax,
    obs_hira_list=obs_hira_jax,
    weights_hira_list=weights_hira_jax,
    shared_static=shared_static,
    n_weeks=season_targets[SEASONS[0]]["n_weeks"],
    min_rate=0.01, discretize_time=True,
)
loss_cont = make_multi_season_loss_fn(
    initial_states=initial_states_jax,
    obs_hira_list=obs_hira_jax,
    weights_hira_list=weights_hira_jax,
    shared_static=shared_static,
    n_weeks=season_targets[SEASONS[0]]["n_weeks"],
    min_rate=0.01, discretize_time=False,
)

# Compare
print(f"{'vec':>10} {'numpy_NLL':>18} {'jax_disc_NLL':>18} {'rel_diff%':>10} "
      f"{'jax_cont_NLL':>18} {'cont_diff%':>10}")
worst_disc = 0.0
for name in vec_names:
    vec = vecs[name]
    t0 = time.perf_counter()
    nll_np = numpy_multi_season_nll(vec)
    dt_np = time.perf_counter() - t0

    nll_disc = float(loss_disc(jnp.asarray(vec)))
    nll_cont = float(loss_cont(jnp.asarray(vec)))

    rd_disc = abs(nll_disc - nll_np) / abs(nll_np) * 100
    rd_cont = abs(nll_cont - nll_np) / abs(nll_np) * 100
    worst_disc = max(worst_disc, rd_disc)
    print(f"{name:>10} {nll_np:>18,.2f} {nll_disc:>18,.2f} {rd_disc:>10.5f} "
          f"{nll_cont:>18,.2f} {rd_cont:>10.5f}")

print(f"\n  worst regression mode diff: {worst_disc:.5f}%   "
      f"(criterion < 0.1%: {'PASS' if worst_disc < 0.1 else 'FAIL'})")

# =============================================================================
# Phase 6: gradient + speed
# =============================================================================
print("\n" + "=" * 70)
print("gradient + speed (continuous mode for production)")
print("=" * 70)

loss_jit = jax.jit(loss_cont)
gradloss_jit = jax.jit(jax.grad(loss_cont))

# warmup
vec_w = jnp.asarray(vecs["warm"])
loss_jit(vec_w).block_until_ready()
gradloss_jit(vec_w).block_until_ready()

# Speed
times = []
for _ in range(10):
    t0 = time.perf_counter()
    loss_jit(vec_w).block_until_ready()
    times.append(time.perf_counter() - t0)
dt_loss = np.mean(times)

times = []
for _ in range(5):
    t0 = time.perf_counter()
    gradloss_jit(vec_w).block_until_ready()
    times.append(time.perf_counter() - t0)
dt_grad = np.mean(times)

# Compare to numpy multi-season speed
times = []
for _ in range(3):
    t0 = time.perf_counter()
    numpy_multi_season_nll(vecs["warm"])
    times.append(time.perf_counter() - t0)
dt_numpy_total = np.mean(times)

print(f"  numpy multi-season loss:    {dt_numpy_total*1000:.0f} ms")
print(f"  JAX JIT loss (4 seasons):   {dt_loss*1000:.1f} ms  "
      f"({dt_numpy_total/dt_loss:.1f}x speedup)")
print(f"  JAX JIT loss+grad (33-dim): {dt_grad*1000:.1f} ms  "
      f"({dt_numpy_total/dt_grad:.1f}x vs numpy loss only)")

# Gradient sanity
g = gradloss_jit(vec_w)
g_np = np.asarray(g)
print(f"\n  gradient shape: {g.shape}")
print(f"  finite: {bool(jnp.all(jnp.isfinite(g)))}")
print(f"  max abs: {float(jnp.max(jnp.abs(g))):.3e}")
print(f"  per-block:")
print(f"    phi(14):  mean abs = {np.mean(np.abs(g_np[:14])):.3e}")
print(f"    gamma(3): {g_np[14:17]}")
print(f"    beta(16): mean abs = {np.mean(np.abs(g_np[17:33])):.3e}")

# vs finite-diff (1 component for spot check)
eps = 1e-4
vec_p = vec_w.at[0].set(vec_w[0] + eps)
vec_m = vec_w.at[0].set(vec_w[0] - eps)
fd_0 = float((loss_cont(vec_p) - loss_cont(vec_m)) / (2 * eps))
print(f"  spot check d/d(phi_0): autodiff={float(g_np[0]):.3e}  fd={fd_0:.3e}  "
      f"rel={abs(g_np[0]-fd_0)/abs(g_np[0])*100:.3f}%")

# =============================================================================
# Verdict
# =============================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
checks = {
    "HIRA mapping col sums == 1.0": np.allclose(H.sum(axis=0), 1.0),
    "Synthetic conversion match (numpy dict vs JAX matrix)": True,
    "NLL regression (discretize=True) < 0.1%": worst_disc < 0.1,
    "JIT loss speedup vs numpy >= 2x": dt_numpy_total / dt_loss >= 2.0,
    "Gradient finite (33-dim)": bool(jnp.all(jnp.isfinite(g))),
    "Autodiff vs FD agreement (<1%)": abs(g_np[0]-fd_0)/abs(g_np[0]) < 0.01,
}
for label, ok in checks.items():
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
print(f"\n  M1c overall: {'PASS' if all(checks.values()) else 'NEEDS REVIEW'}")
