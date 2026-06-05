"""M1b regression: diffrax trajectory vs scipy solve_ivp.

Criterion: rel diff < 1e-3 (adaptive-step solvers at rtol=1e-4).
Critical: daily_new_infection_by_age (NLL input) must match within 1e-3.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import json
import numpy as np
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# numpy reference
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, simulate_aggregated,
)
from kt_epimodel_hira.calibration.param_vector import vector_to_params
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters, EmploymentParameters,
)

# JAX
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)

OUTDIR = Path("/Users/hwcho/Documents/python/NIMS/kt_epimodel_hira/outputs/calibration")


def relmax(a, b):
    return float(np.max(np.abs(a - b) / (np.abs(np.asarray(b)) + 1.0)))


# ---- Setup
print("=" * 70)
print("Setup: warm-start vec + 2019-2020 season")
print("=" * 70)
vec = np.array(json.load(open(OUTDIR / "multistart_warm.json"))["best_vec"])
beta_4 = vec[17:21]; phi_14 = vec[:14]; gamma_3 = vec[14:17]
vec_21 = np.concatenate([beta_4, phi_14, gamma_3])
cal = vector_to_params(vec_21)

inputs = build_aggregated_inputs()
pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
mobility = inputs["mobility"]; matrices = inputs["matrices"]

seed = estimate_initial_infected_from_hira(
    "2019-2020", pop_15.flatten(), sido_codes=None,
    gamma_15_assumed=cal.gamma_15,
)
state0 = _build_initial_state_with_age_seed(
    pop_15, seed, seed_e_factor=0.5,
    initial_immunity=0.3, initial_vaccinated_fraction=0.0,
)
params = ModelParameters().with_calibration(cal).with_employment(
    EmploymentParameters(rho=rho_emp))

# Build JAX kwargs
jax_kwargs = dict(
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
    kappa=jnp.asarray(params.disease.kappa_array),
    sigma=params.disease.sigma, gamma=params.disease.gamma,
    beta_h=cal.beta_h, beta_w=cal.beta_w, beta_s=cal.beta_s, beta_o=cal.beta_o,
    phi_susc=jnp.asarray(cal.phi),
    p_school=params.policy.p_school, p_work=params.policy.p_work,
    VE=params.vaccination.VE,
    annual_coverage=jnp.asarray(params.vaccination.annual_coverage),
    vax_peak_iso_week=params.vaccination.peak_iso_week,
    vax_spread_weeks=params.vaccination.spread_weeks,
    seasonality_amp=params.disease.seasonality_amp,
    seasonality_base=params.disease.seasonality_base,
    seasonality_peak_day=params.disease.seasonality_peak_day,
    seasonality_period=params.disease.seasonality_period,
)
state0_jax = jnp.asarray(state0)

# ---- numpy reference (uses solver.py with rtol=1e-4 default)
print(f"\n{'='*70}\nNumpy reference (scipy RK45)\n{'='*70}")
t0 = time.perf_counter()
result_np = simulate_aggregated(
    params, inputs, seed_total=float(seed.sum()),
    seed_by_age=seed, seed_e_factor=0.5,
    initial_immunity=0.3, t_span=(0.0, 364.0),
)
dt_np = time.perf_counter() - t0
states_np = result_np.states  # (365, 5, 15, 1)
inc_np = result_np.daily_new_infection_by_age()
print(f"  numpy wall: {dt_np*1000:.0f} ms")
print(f"  states shape: {states_np.shape}")
print(f"  inc shape: {inc_np.shape}")
print(f"  inc sum: {inc_np.sum():,.0f}")

# ---- JAX (Dopri5) uncompiled
print(f"\n{'='*70}\nJAX (Dopri5) — uncompiled\n{'='*70}")
t0 = time.perf_counter()
states_jax_raw = simulate_jax(state0_jax, **jax_kwargs, method="Dopri5")
states_jax_raw.block_until_ready()
dt_jax_unc = time.perf_counter() - t0
states_jax = np.asarray(states_jax_raw)
print(f"  JAX uncompiled wall: {dt_jax_unc*1000:.0f} ms")
print(f"  states shape: {states_jax.shape}")

# ---- JAX JIT
print(f"\n{'='*70}\nJAX JIT — Dopri5\n{'='*70}")
sim_jit = jax.jit(simulate_jax, static_argnames=("method", "max_steps"))
# warmup
sim_jit(state0_jax, **jax_kwargs, method="Dopri5").block_until_ready()
times = []
for _ in range(10):
    t0 = time.perf_counter()
    sim_jit(state0_jax, **jax_kwargs, method="Dopri5").block_until_ready()
    times.append(time.perf_counter() - t0)
dt_jax_jit = np.mean(times)
print(f"  JIT mean over 10: {dt_jax_jit*1000:.1f} ms/call")
print(f"  speedup vs numpy: {dt_np/dt_jax_jit:.2f}x")

# ---- Compartment trajectory comparison
print(f"\n{'='*70}\nTrajectory regression: scipy vs JAX (Dopri5)\n{'='*70}")
labels = ["S", "V", "E", "I", "R"]
worst = 0.0
for i, lbl in enumerate(labels):
    a = states_jax[:, i]; b = states_np[:, i]
    rd = relmax(a, b)
    worst = max(worst, rd)
    abs_d = float(np.max(np.abs(a - b)))
    print(f"  {lbl}: max rel diff = {rd:.3e}  max abs diff = {abs_d:.2f}")
print(f"  worst compartment rel diff: {worst:.3e}")

# ---- daily_new_infection_by_age (NLL input — most important)
print(f"\n{'='*70}\nDaily new infection (NLL input)\n{'='*70}")
inc_jax = np.asarray(daily_new_infection_by_age_jax(states_jax_raw))
print(f"  inc shape: {inc_jax.shape}")
print(f"  inc sum: numpy={inc_np.sum():,.2f}  jax={inc_jax.sum():,.2f}  "
      f"rel diff sum = {abs(inc_jax.sum()-inc_np.sum())/abs(inc_np.sum())*100:.4f}%")
rd_inc = relmax(inc_jax, inc_np)
abs_d_inc = float(np.max(np.abs(inc_jax - inc_np)))
print(f"  per-age daily: max rel diff = {rd_inc:.3e}  max abs diff = {abs_d_inc:.4f}")

# Peak alignment
peak_t_np = int(np.argmax(inc_np.sum(axis=1)))
peak_t_jax = int(np.argmax(inc_jax.sum(axis=1)))
peak_v_np = float(inc_np.sum(axis=1).max())
peak_v_jax = float(inc_jax.sum(axis=1).max())
print(f"  peak: numpy day {peak_t_np} val {peak_v_np:,.1f}  "
      f"jax day {peak_t_jax} val {peak_v_jax:,.1f}")

# ---- Try Tsit5 too for sanity
print(f"\n{'='*70}\nJAX Tsit5 (alternative method) comparison\n{'='*70}")
states_tsit_raw = sim_jit(state0_jax, **jax_kwargs, method="Tsit5")
states_tsit_raw.block_until_ready()
states_tsit = np.asarray(states_tsit_raw)
rd_tsit = relmax(states_tsit, states_np)
inc_tsit = np.asarray(daily_new_infection_by_age_jax(states_tsit_raw))
rd_inc_tsit = relmax(inc_tsit, inc_np)
print(f"  Tsit5 vs scipy: traj rel diff {rd_tsit:.3e}  inc rel diff {rd_inc_tsit:.3e}")

# ---- Trajectory autodiff
print(f"\n{'='*70}\nTrajectory autodiff\n{'='*70}")

def peak_inc(beta_h_val):
    new_args = dict(jax_kwargs); new_args["beta_h"] = beta_h_val
    states = simulate_jax(state0_jax, **new_args, method="Dopri5")
    inc = daily_new_infection_by_age_jax(states)
    return inc.sum(axis=1).max()

t0 = time.perf_counter()
g = float(jax.grad(peak_inc)(0.06))
dt_g = time.perf_counter() - t0
fd = (peak_inc(0.06 + 1e-4) - peak_inc(0.06 - 1e-4)) / (2e-4)
print(f"  d(peak_inc)/d(beta_h):  autodiff = {g:.4e}  ({dt_g*1000:.0f} ms)")
print(f"                          finite   = {float(fd):.4e}")
print(f"  rel diff: {abs(g - float(fd))/abs(g)*100:.3f}%")

# ---- Verdict
print(f"\n{'='*70}\nVERDICT\n{'='*70}")
checks = {
    "Compartment traj < 1e-3": worst < 1e-3,
    "daily_new_infection < 1e-3": rd_inc < 1e-3,
    "Peak alignment (same day)": peak_t_jax == peak_t_np,
    "JIT speedup vs numpy >= 2x": dt_np / dt_jax_jit >= 2.0,
    "Trajectory autodiff finite": np.isfinite(g),
    "Autodiff agrees with FD (<1%)": abs(g - float(fd)) / abs(g) < 0.01,
}
for label, ok in checks.items():
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
overall = all(checks.values())
print(f"\n  M1b overall: {'PASS' if overall else 'NEEDS REVIEW'}")
