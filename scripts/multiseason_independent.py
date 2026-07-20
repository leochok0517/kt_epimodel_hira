"""Experiment 1 — independent per-season fits → redistribution reproducibility.

Tests whether the 2019-2020 redistribution finding (adults↓ / school-age↑ under
sick-leave, with low-work_share backfire) recurs across the 6 normal complete
HIRA seasons — i.e. it is not a one-season artifact.

Setup identical to policy_map_stepAB (Step A: C(t) term↔vacation + κ 3-way;
Step B: v(t) finite-season norm; RAW coverage; φ U-shape; γ CDC Reed; pin;
12-start L-BFGS-B). Contact/pop/vaccine shared across seasons (single-value
limitation, acknowledged). Only the HIRA target + seed vary per season.

Per season: independent pin-fit β_4 at work_share ∈ {0.06 (literature), 0.18
(boundary)}, μ=1.0; sick-leave policy (p_work=0.4) redistribution.

Output: outputs/eda/multiseason_independent.json + console table.
"""
from __future__ import annotations
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from kt_data import SUDOGWON_SIDO_CODES
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn

import sens_workshare_kappa_v2 as S
import policy_map_stepAB as P          # reuse fit_beta, policy_at_mu, build_hira_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "multiseason_independent.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
VAC_NPZ = REPO_ROOT.parent / "kt_data/data/external/contact_matrices/empirical_matrices_15_vacation.npz"

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2022-2023"]
WS_POINTS = [0.06, 0.18]
MU = 1.0
ADULT = ["18-44", "45-64"]
SCHOOL = ["6-11", "12-17"]


def build_setup_season(season, shared_static, state_common):
    """Per-season setup: only HIRA target + seed differ; statics reused."""
    tgt = load_hira_target_by_age(season, sido_codes=list(SUDOGWON_SIDO_CODES),
                                  first_peak_only=True, first_peak_end_week=26)
    n_weeks = tgt["n_weeks"]
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]; w[:, i] = tgt["weights"][ag]
    seed = estimate_initial_infected_from_hira(
        season, state_common["pop_15"].flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        state_common["pop_15"], seed, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    return dict(shared_base=shared_static, state0=state0,
                obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
                obs_np=obs, w_np=w, n_weeks=n_weeks,
                ngm_default=state_common["ngm_default"],
                H=state_common["H"], pop_6=state_common["pop_6"])


def build_common():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease; vax = ModelParameters().vaccination
    vac = load_contact_matrices(path=VAC_NPZ)
    shared = dict(
        C_home=jnp.asarray(matrices["C_home"]), C_school=jnp.asarray(matrices["C_school"]),
        C_work=jnp.asarray(matrices["C_work"]), C_other=jnp.asarray(matrices["C_other"]),
        C_home_vac=jnp.asarray(vac["C_home"]), C_school_vac=jnp.asarray(vac["C_school"]),
        C_work_vac=jnp.asarray(vac["C_work"]), C_other_vac=jnp.asarray(vac["C_other"]),
        M_home=jnp.asarray(mobility["home"]), M_school=jnp.asarray(mobility["school"]),
        M_work=jnp.asarray(mobility["work"]), M_other=jnp.asarray(mobility["other"]),
        pop_15=jnp.asarray(pop_15), rho=jnp.asarray(rho_emp),
        sigma=disease.sigma, gamma=disease.gamma, VE=vax.VE,
        annual_coverage=jnp.asarray(np.asarray(vax.annual_coverage, dtype=np.float64)),  # RAW
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=S.AMP, seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp, C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE, gamma=disease.gamma, seasonal_factor=1.0 + S.AMP)
    H = P.build_hira_matrix()
    pop_15_flat = np.asarray(pop_15).sum(axis=1) if np.asarray(pop_15).ndim == 2 else np.asarray(pop_15)
    return dict(pop_15=jnp.asarray(pop_15), ngm_default=ngm, H=H, pop_6=H @ pop_15_flat), shared


def fit_quality(beta_4, setup):
    """baseline forward → per-age obs/model peak-window ratio (fit quality)."""
    phi = jnp.asarray(S.PHI_USHAPE); g15 = jnp.asarray(S.build_gamma_15(*S.GAMMA_CENTER))
    shared = P.shared_with_kappa(setup, P.KAPPA_BASE)
    _, pred = S.run_forward(np.array(beta_4), phi, g15, S.P_SCHOOL_BASE, S.P_WORK_BASE, shared, setup)
    r = S.per_age_ratios(np.asarray(pred), setup)
    return {ag: round(r[ag]["ratio"], 3) for ag in HIRA_AGE_GROUPS}


def main():
    print("=" * 100)
    print("EXPERIMENT 1 — independent per-season fits, redistribution reproducibility")
    print(f"  seasons={SEASONS}")
    print(f"  ws∈{WS_POINTS} μ={MU}  Step A+B | φ U-shape | γ CDC Reed | 12 starts | shared contact/pop/vaccine")
    print("=" * 100)
    t0 = time.perf_counter()
    common, shared = build_common()
    print(f"[common setup] {time.perf_counter()-t0:.1f}s\n")

    results = {}
    for s in SEASONS:
        ts = time.perf_counter()
        setup = build_setup_season(s, shared, common)
        rec = {"work_share_points": {}}
        for ws in WS_POINTS:
            fit = P.fit_beta(ws, setup)
            pol = P.policy_at_mu(fit["beta_4"], MU, setup)
            da = pol["d_attack_pct_by_age"]
            adult_down = all(da[a] < 0 for a in ADULT)
            school_up = any(da[sc] > 0 for sc in SCHOOL)
            rec["work_share_points"][f"{ws:.2f}"] = dict(
                R0=fit["R0"], beta_4=fit["beta_4"], beta_home=fit["beta_4"][0],
                pi=fit["pi"], nll=fit["nll"], nll_std=fit["nll_std"],
                averted_pct=pol["averted_total_pct"],
                d_attack_pct_by_age=da, transfer_ratio=pol["transfer_ratio"],
                age_flip=pol["age_flip"],
                adult_down=bool(adult_down), school_up=bool(school_up),
                redistribution_dir=bool(adult_down and school_up),
                fit_quality_ratio=fit_quality(fit["beta_4"], setup),
            )
        results[s] = rec
        f06 = rec["work_share_points"]["0.06"]; f18 = rec["work_share_points"]["0.18"]
        print(f"── {s} ({time.perf_counter()-ts:.1f}s) ──")
        print(f"   R0={f06['R0']:.3f}  β_home={f06['beta_home']:.4f}  nll(ws06)={f06['nll']:.1f}")
        print(f"   ws=0.06: averted={f06['averted_pct']:+.2f}%  redist_dir(성인↓/학령↑)={f06['redistribution_dir']}  flip={f06['age_flip']}  transfer={f06['transfer_ratio']:.3f}")
        print(f"   ws=0.18: averted={f18['averted_pct']:+.2f}%  redist_dir={f18['redistribution_dir']}  flip={f18['age_flip']}  transfer={f18['transfer_ratio']:.3f}")
        da = f18["d_attack_pct_by_age"]
        print(f"     Δattack(ws0.18) " + "  ".join(f"{ag}:{da[ag]:+.3f}" for ag in HIRA_AGE_GROUPS))

    # ---- reproducibility summary ----
    print("\n" + "=" * 100)
    print("REPRODUCIBILITY SUMMARY")
    for ws in WS_POINTS:
        k = f"{ws:.2f}"
        dir_ok = [s for s in SEASONS if results[s]["work_share_points"][k]["redistribution_dir"]]
        av_signs = {s: results[s]["work_share_points"][k]["averted_pct"] for s in SEASONS}
        n_pos = sum(1 for v in av_signs.values() if v >= 0); n_neg = len(SEASONS) - n_pos
        print(f"  ws={ws:.2f}: 재분배방향(성인↓/학령기↑) {len(dir_ok)}/{len(SEASONS)} 시즌"
              f"   averted 부호: +{n_pos} / −{n_neg}")
    r0s = [results[s]["work_share_points"]["0.06"]["R0"] for s in SEASONS]
    bhs = [results[s]["work_share_points"]["0.06"]["beta_home"] for s in SEASONS]
    print(f"  R0 range: [{min(r0s):.3f}, {max(r0s):.3f}]  β_home range: [{min(bhs):.4f}, {max(bhs):.4f}]"
          f"  (mean±sd R0={np.mean(r0s):.3f}±{np.std(r0s):.3f}, βh={np.mean(bhs):.4f}±{np.std(bhs):.4f})")
    print("=" * 100)

    out = dict(
        meta=dict(seasons=SEASONS, ws_points=WS_POINTS, mu=MU,
                  step="A(C(t)+kappa3)+B(v(t))", coverage="raw", phi="U-shape",
                  gamma_15=list(S.GAMMA_CENTER), n_starts=S.N_STARTS,
                  shared_inputs="contact/pop/vaccine single-value across seasons"),
        results=results,
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
