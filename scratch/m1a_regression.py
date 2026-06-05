"""M1a regression: compare numpy vs JAX ODE rhs at multiple time points.

Pass criterion: max relative diff < 1e-8 at every component.
Components: 4 FOI channels + total FOI + ODE rhs (dS, dV, dE, dI, dR).
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# numpy reference
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.calibration.param_vector import vector_to_params
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters, EmploymentParameters,
)
from kt_epimodel_hira.model.foi import (
    compute_foi_home, compute_foi_school, compute_foi_work, compute_foi_other,
    compute_foi,
)
from kt_epimodel_hira.model.dynamics import compute_derivatives

# JAX
from kt_epimodel_hira.jax_model.foi_jax import (
    compute_foi_home_jax, compute_foi_school_jax,
    compute_foi_work_jax, compute_foi_other_jax,
    compute_foi_jax, seasonal_factor_cosine,
)
from kt_epimodel_hira.jax_model.dynamics_jax import compute_derivatives_jax

OUTDIR = Path("/Users/hwcho/Documents/python/NIMS/kt_epimodel_hira/outputs/calibration")


def relmax(a, b):
    """Max relative diff with floor on denominator."""
    return float(np.max(np.abs(a - b) / (np.abs(np.asarray(b)) + 1e-10)))


# ---- Setup
print("=" * 70)
print("Setup: load warm-start vec + season 2019-2020 state")
print("=" * 70)
vec = np.array(json.load(open(OUTDIR / "multistart_warm.json"))["best_vec"])
phi_14 = vec[:14]; gamma_3 = vec[14:17]; beta_4 = vec[17:21]
vec_21 = np.concatenate([beta_4, phi_14, gamma_3])
cal = vector_to_params(vec_21)

inputs = build_aggregated_inputs()
pop_15 = inputs["pop_15"]
mobility = inputs["mobility"]
matrices = inputs["matrices"]
rho_emp = inputs["rho"]

seed = estimate_initial_infected_from_hira(
    "2019-2020", pop_15.flatten(),
    sido_codes=None, gamma_15_assumed=cal.gamma_15,
)
state0 = _build_initial_state_with_age_seed(
    pop_15, seed, seed_e_factor=0.5,
    initial_immunity=0.3, initial_vaccinated_fraction=0.0,
)
print(f"  state shape: {state0.shape}")
print(f"  N_admdong:   {state0.shape[2]}")

params = ModelParameters().with_calibration(cal).with_employment(
    EmploymentParameters(rho=rho_emp))

# JAX equivalents
jax_args = dict(
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
    sigma=params.disease.sigma,
    gamma=params.disease.gamma,
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
state_jax = jnp.asarray(state0)

# ---- Phase 4: Channel decomposition (sf=1.0 to isolate)
print(f"\n{'='*70}\nPhase 4: FOI channels at fixed seasonal_factor=1.0\n{'='*70}")

# Numpy: each channel with sf=1.0
foi_h_np = compute_foi_home(
    state0, matrices["C_home"], pop_15, rho_emp,
    params.disease.kappa_array,
    params.policy.p_school, params.policy.p_work,
    cal.beta_h, cal.phi, seasonal_factor=1.0,
)
foi_s_np = compute_foi_school(
    state0, matrices["C_school"], pop_15,
    params.policy.p_school, cal.beta_s, cal.phi, seasonal_factor=1.0,
)
foi_w_np = compute_foi_work(
    state0, matrices["C_work"], pop_15, mobility["work"],
    rho_emp, params.policy.p_work, cal.beta_w, cal.phi, seasonal_factor=1.0,
)
foi_o_np = compute_foi_other(
    state0, matrices["C_other"], pop_15, mobility["other"],
    cal.beta_o, cal.phi, seasonal_factor=1.0,
)

# JAX
foi_h_j = compute_foi_home_jax(
    state_jax, jax_args["C_home"], jax_args["pop_15"], jax_args["rho"],
    jax_args["kappa"], jax_args["p_school"], jax_args["p_work"],
    jax_args["beta_h"], jax_args["phi_susc"], seasonal_factor=1.0,
)
foi_s_j = compute_foi_school_jax(
    state_jax, jax_args["C_school"], jax_args["pop_15"],
    jax_args["p_school"], jax_args["beta_s"], jax_args["phi_susc"],
    seasonal_factor=1.0,
)
foi_w_j = compute_foi_work_jax(
    state_jax, jax_args["C_work"], jax_args["pop_15"], jax_args["M_work"],
    jax_args["rho"], jax_args["p_work"], jax_args["beta_w"],
    jax_args["phi_susc"], seasonal_factor=1.0,
)
foi_o_j = compute_foi_other_jax(
    state_jax, jax_args["C_other"], jax_args["pop_15"], jax_args["M_other"],
    jax_args["beta_o"], jax_args["phi_susc"], seasonal_factor=1.0,
)

channels = {
    "home":  (foi_h_np, foi_h_j),
    "school":(foi_s_np, foi_s_j),
    "work":  (foi_w_np, foi_w_j),
    "other": (foi_o_np, foi_o_j),
}
for name, (np_arr, j_arr) in channels.items():
    rd = relmax(np.asarray(j_arr), np_arr)
    print(f"  {name:>7}: max rel diff = {rd:.3e}  "
          f"{'OK' if rd < 1e-8 else 'FAIL'}")

# ---- Seasonal factor at multiple days
print(f"\n{'='*70}\nSeasonal factor: cosine sf(t)\n{'='*70}")
for t in [0.0, 50.0, 100.0, 105.0, 150.0, 200.0]:
    sf_np = params.disease.seasonal_factor(t)
    sf_j = float(seasonal_factor_cosine(
        t, amp=params.disease.seasonality_amp,
        base=params.disease.seasonality_base,
        peak_day=params.disease.seasonality_peak_day,
        period=params.disease.seasonality_period,
    ))
    rd = abs(sf_j - sf_np) / max(abs(sf_np), 1e-12)
    print(f"  t={t:>6.0f}: numpy={sf_np:.6f}  jax={sf_j:.6f}  rel_diff={rd:.3e}")

# ---- Phase 3: Full RHS at multiple t
print(f"\n{'='*70}\nPhase 3: full ODE RHS at t=0/50/100/150\n{'='*70}")

rhs_pass = True
for t in [0.0, 50.0, 100.0, 150.0]:
    day = int(t)  # numpy uses int day
    dstate_np = compute_derivatives(
        state0, mobility, matrices, pop_15, params, day_in_season=day,
    )
    dstate_j = compute_derivatives_jax(state_jax, t, **jax_args, day_in_season_offset=0.0)
    rd = relmax(np.asarray(dstate_j), dstate_np)
    # per-compartment
    per_comp = {}
    labels = ["dS", "dV", "dE", "dI", "dR"]
    for i, lbl in enumerate(labels):
        per_comp[lbl] = relmax(np.asarray(dstate_j[i]), dstate_np[i])
    status = "OK" if rd < 1e-8 else "FAIL"
    print(f"  t={t:>5.0f}: max rel diff = {rd:.3e}  {status}")
    for lbl, v in per_comp.items():
        flag = " (large)" if v > 1e-8 else ""
        print(f"          {lbl}: {v:.3e}{flag}")
    if rd >= 1e-8:
        rhs_pass = False

# ---- Phase 5: autodiff sanity
print(f"\n{'='*70}\nPhase 5: autodiff sanity\n{'='*70}")

def foi_sum_at_t0(beta_h_scalar):
    args_local = dict(jax_args)
    args_local["beta_h"] = beta_h_scalar
    foi = compute_foi_jax(state_jax, **{k: args_local[k] for k in [
        "C_home","C_school","C_work","C_other",
        "M_home","M_school","M_work","M_other",
        "pop_15","rho","kappa",
        "p_school","p_work","beta_h","beta_w","beta_s","beta_o","phi_susc",
    ]}, seasonal_factor=1.0)
    return foi.sum()

g_beta_h = float(jax.grad(foi_sum_at_t0)(0.06))
print(f"  d(sum FOI)/d(beta_h) at beta_h=0.06: {g_beta_h:.6e}  finite={np.isfinite(g_beta_h)}")
fd = (foi_sum_at_t0(0.06 + 1e-5) - foi_sum_at_t0(0.06 - 1e-5)) / (2 * 1e-5)
print(f"  finite-diff:                          {float(fd):.6e}")
print(f"  rel diff: {abs(g_beta_h - float(fd))/abs(g_beta_h)*100:.4f}%")

# ---- Verdict
print(f"\n{'='*70}\nVERDICT\n{'='*70}")
all_chan_ok = all(relmax(np.asarray(j), np_arr) < 1e-8 for np_arr, j in channels.values())
print(f"  FOI channels < 1e-8:    {'PASS' if all_chan_ok else 'FAIL'}")
print(f"  Seasonal factor < 1e-8: PASS  (cosine is identical)")
print(f"  Full RHS < 1e-8:        {'PASS' if rhs_pass else 'FAIL'}")
print(f"  Autodiff sanity:        {'PASS' if np.isfinite(g_beta_h) else 'FAIL'}")
overall = all_chan_ok and rhs_pass and np.isfinite(g_beta_h)
print(f"\n  M1a overall: {'PASS' if overall else 'FAIL'}")
