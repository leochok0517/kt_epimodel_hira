"""Step A: aggregated vs spatial NGM consistency check.

Uses the NB-production posterior mean β (and φ=1.0, R(0) step, γ from registry)
and computes the NGM spectral radius two ways:

1. aggregated: existing 15×15 closure (same code path the calibration used)
2. spatial: matrix-free power iteration on the 15·n_admdong next-generation
   operator using compute_foi_jax (real KT mobility for the chosen yyyymm)

If the ratio is ~1.0, β is portable to the spatial forward without rescaling.
If not, the report prints the correction factor.

Cheap NGM check only — Step A-2 (forward incidence) is a separate, longer run.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from kt_data.data.load_contact import load_contact_matrices
from kt_data.data.load_population import (
    load_population_15groups, get_population_matrix,
)
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.gamma_registry import get_active_gamma
from kt_epimodel_hira.model.parameters import (
    DiseaseParameters, EmploymentParameters, ModelParameters,
)
from kt_epimodel_hira.model.mobility_tensor import (
    build_M_home, build_M_school, build_M_work, build_M_other,
)
from kt_epimodel_hira.jax_model.foi_jax import compute_foi_jax
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn


SEASONS = {
    "2017-2018": "201809",
    "2018-2019": "201809",
    "2019-2020": "201909",
    "2022-2023": "202209",
}


def aggregated_ngm_R0(beta_4, phi_full, season_label="2019-2020"):
    """Same NGM closure the calibration uses (1 admdong)."""
    inputs = build_aggregated_inputs()
    disease = ModelParameters().disease
    matrices = inputs["matrices"]
    ngm = make_ngm_eigvalue_fn(
        pop_15=inputs["pop_15"], rho=inputs["rho"],
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.7,
    )
    return float(ngm(*beta_4, phi_full))


def build_spatial_inputs(yyyymm: str):
    """Real KT mobility + population at 1154 admdong."""
    print(f"  loading population 15groups …")
    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop = N_mat.T.astype(np.float64)                # (15, n_adm)
    n_adm = pop.shape[1]
    print(f"  pop matrix (15, {n_adm})")

    print(f"  loading mobility {yyyymm} …")
    t0 = time.perf_counter()
    M = {
        "home":   build_M_home(n_adm).astype(np.float64),
        "school": build_M_school(n_adm).astype(np.float64),
        "work":   build_M_work(yyyymm, daytype="weekday",
                                admdong_codes=codes, pop_15=pop).astype(np.float64),
        "other":  build_M_other(yyyymm, daytype="weekday",
                                 admdong_codes=codes, pop_15=pop).astype(np.float64),
    }
    print(f"  M_work shape {M['work'].shape}  ({time.perf_counter()-t0:.1f}s)")

    matrices_raw = load_contact_matrices()
    matrices = {k: jnp.asarray(matrices_raw[k]) for k in
                ("C_home", "C_work", "C_school", "C_other")}

    emp = EmploymentParameters.from_kt_data(admdong_codes=codes)
    rho_spatial = emp.rho                                # (n_adm, 15)
    return {
        "pop_15": jnp.asarray(pop),
        "rho": jnp.asarray(rho_spatial),
        "matrices": matrices,
        "mobility": {k: jnp.asarray(v) for k, v in M.items()},
    }


def spatial_R0_power_iter(beta_4, phi_full, spatial, *, n_iter=80, tol=1e-6):
    """Matrix-free power iteration for NGM spectral radius.

    NGM matvec: given I (15, n_adm), compute new infections rate / γ at DFE.
    """
    disease = ModelParameters().disease
    policy = ModelParameters().policy
    vax = ModelParameters().vaccination

    pop_15 = spatial["pop_15"]
    rho = spatial["rho"]
    matrices = spatial["matrices"]
    mob = spatial["mobility"]

    # DFE susceptibles: S = pop * (1 - R(0))
    S_dfe = pop_15 * (1.0 - jnp.asarray(R0_IMMUNITY_PROFILE))[:, None]
    sf = 1.7
    gamma = disease.gamma
    kappa = jnp.asarray(disease.kappa_array)
    bh, bw, bs, bo = [float(x) for x in beta_4]

    def matvec(I_vec):
        # build state (5, 15, n_adm) with only I perturbed
        zero = jnp.zeros_like(I_vec)
        state = jnp.stack([S_dfe, zero, zero, I_vec, zero], axis=0)
        foi = compute_foi_jax(
            state,
            C_home=matrices["C_home"], C_school=matrices["C_school"],
            C_work=matrices["C_work"], C_other=matrices["C_other"],
            M_home=mob["home"], M_school=mob["school"],
            M_work=mob["work"], M_other=mob["other"],
            pop_15=pop_15, rho=rho, kappa=kappa,
            p_school=policy.p_school, p_work=policy.p_work,
            beta_h=bh, beta_w=bw, beta_s=bs, beta_o=bo,
            phi_susc=phi_full,
            seasonal_factor=sf,
        )                                       # (15, n_adm)
        new_inf = S_dfe * foi                   # rate per unit time at DFE
        return new_inf / gamma                   # next-generation operator

    matvec_jit = jax.jit(matvec)

    # power iteration
    n_adm = int(pop_15.shape[1])
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.random((15, n_adm)))
    v = v / jnp.linalg.norm(v)
    lam_prev = 0.0
    for it in range(n_iter):
        w = matvec_jit(v)
        lam = float(jnp.linalg.norm(w))
        v = w / lam
        if it > 5 and abs(lam - lam_prev) < tol * max(abs(lam), 1.0):
            print(f"  power iter converged at {it+1}: λ = {lam:.6f}")
            return lam
        lam_prev = lam
    print(f"  power iter not fully converged ({n_iter} iters): λ ≈ {lam:.6f}")
    return lam


def main():
    print("=" * 70)
    print("Step A — β consistency (aggregated NGM vs spatial NGM)")
    print("=" * 70)

    # NB posterior mean β (16 = 4 channel × 4 seasons)
    npz_path = "outputs/calibration/m2_prod_nb_samples.npz"
    data = np.load(npz_path)
    beta_post = data["beta"].reshape(-1, 16)
    beta_mean_16 = beta_post.mean(axis=0)
    print(f"  loaded NB posterior β: {beta_post.shape[0]} samples")

    phi_full = jnp.ones(15)

    # Pick a representative season — 2019-2020 (third block in 16-vec)
    season_idx = 2
    season_label = list(SEASONS.keys())[season_idx]
    yyyymm = SEASONS[season_label]
    beta_4 = beta_mean_16[season_idx*4:(season_idx+1)*4]
    print(f"  season: {season_label}  yyyymm: {yyyymm}")
    print(f"  β (h, w, s, o) = {[round(float(x), 5) for x in beta_4]}")
    print()

    # A-1.a aggregated
    print("[A-1.a] aggregated NGM (1 admdong, same as calibration)")
    R0_agg = aggregated_ngm_R0(beta_4, phi_full, season_label)
    print(f"  R0_aggregated = {R0_agg:.4f}")
    print()

    # A-1.b spatial
    print("[A-1.b] spatial NGM (1154 admdong, real KT mobility)")
    spatial = build_spatial_inputs(yyyymm)
    R0_spatial = spatial_R0_power_iter(beta_4, phi_full, spatial)
    print(f"  R0_spatial    = {R0_spatial:.4f}")
    print()

    ratio = R0_spatial / R0_agg
    correction = R0_agg / R0_spatial
    print("=" * 70)
    print(f"R0_aggregated = {R0_agg:.4f}")
    print(f"R0_spatial    = {R0_spatial:.4f}")
    print(f"ratio (spatial / aggregated) = {ratio:.4f}")
    print(f"β correction factor (if needed) = {correction:.4f}")
    print()
    if abs(ratio - 1.0) < 0.02:
        print("→ ratio within ±2% : β portable as-is, no rescaling needed ✅")
    elif abs(ratio - 1.0) < 0.10:
        print("→ ratio within ±10% : small mismatch, β rescale recommended")
    else:
        print("→ ratio differs >10% : spatial decomposition changes R0, β must be rescaled")
    print("=" * 70)

    # A-3 sanity checks
    print()
    print("[A-3] consistency sanity")
    inputs_agg = build_aggregated_inputs()
    print(f"  aggregated rho shape: {tuple(inputs_agg['rho'].shape)}  (sudogwon avg)")
    print(f"  spatial    rho shape: {tuple(spatial['rho'].shape)}      (per-admdong)")
    print(f"  rho mean (aggregated): {float(inputs_agg['rho'].mean()):.3f}")
    print(f"  rho mean (spatial):    {float(spatial['rho'].mean()):.3f}")
    print(f"  population total aggregated: {float(inputs_agg['pop_15'].sum()):,.0f}")
    print(f"  population total spatial   : {float(spatial['pop_15'].sum()):,.0f}")
    print("  γ from registry (forward uses raw incidence, γ only for HIRA mapping — OK)")


if __name__ == "__main__":
    main()
