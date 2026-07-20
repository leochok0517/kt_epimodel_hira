"""Multi-season reproducibility of the holiday sign-reversal in child Δattack.

2019-2020 finding: sick-leave child Δattack flips sign term-window (−, benefit)
→ vacation-window (+, backfire) — spillover is the work→home path, independent
of school closure. This checks whether the flip recurs across the 6 normal
seasons (strengthen or refute the single-season finding).

Setup = experiment 2 (policy_compare): Step A+B, shared π FIXED, per-season R0
(multi-season joint), κ 3-way, φ U-shape, γ CDC Reed. β_{4,s}=derive(R0_s,π,φ),
NO refit. Sick-leave p_work=0.4, μ=1.0.

FIXED windows across ALL seasons (peak_day & holiday calendar are season-invariant):
  term    = [70, 113]  (rise→peak, before break)
  vacation= [113, 183] (winter break)

Output: outputs/eda/holiday_reversal_multiseason.json + console.
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
jax.devices()
import jax.numpy as jnp

from kt_data import SUDOGWON_SIDO_CODES
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import (
    load_hira_target_by_age, HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
from kt_epimodel_hira.jax_model.solver_jax import simulate_jax, daily_new_infection_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn, derive_beta_from_R0_simplex
import sens_workshare_kappa_v2 as S
import policy_compare_school_vs_sickleave as PC   # reuse run / attack6 / policy_eval / windows

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "holiday_reversal_multiseason.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
VAC_NPZ = REPO_ROOT.parent / "kt_data/data/external/contact_matrices/empirical_matrices_15_vacation.npz"

# per-season R0 from multi-season joint (shared π)
R0_BY_SEASON = {
    "2015-2016": 1.8638, "2016-2017": 2.2619, "2017-2018": 2.0423,
    "2018-2019": 2.1678, "2019-2020": 2.1087, "2022-2023": 2.0383,
}
SEASONS = list(R0_BY_SEASON.keys())
PI_SHARED = np.array([0.357, 0.255, 0.067, 0.321])
KAPPA_BASE = np.array([0.29]*4 + [0.30]*10 + [0.0])
PHI_USHAPE = np.array(S.PHI_USHAPE)
TERM_WIN = (70.0, 113.0); VAC_WIN = (113.0, 183.0); WHOLE = (-1.0e9, 1.0e9)
CHILD = ["0-5", "6-11", "12-17"]


def build_common():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho = inputs["rho"]; M = inputs["matrices"]; mob = inputs["mobility"]
    disease = ModelParameters().disease; vax = ModelParameters().vaccination
    vac = load_contact_matrices(path=VAC_NPZ)
    shared = dict(
        C_home=jnp.asarray(M["C_home"]), C_school=jnp.asarray(M["C_school"]),
        C_work=jnp.asarray(M["C_work"]), C_other=jnp.asarray(M["C_other"]),
        C_home_vac=jnp.asarray(vac["C_home"]), C_school_vac=jnp.asarray(vac["C_school"]),
        C_work_vac=jnp.asarray(vac["C_work"]), C_other_vac=jnp.asarray(vac["C_other"]),
        M_home=jnp.asarray(mob["home"]), M_school=jnp.asarray(mob["school"]),
        M_work=jnp.asarray(mob["work"]), M_other=jnp.asarray(mob["other"]),
        pop_15=jnp.asarray(pop_15), rho=jnp.asarray(rho),
        sigma=disease.sigma, gamma=disease.gamma, VE=vax.VE,
        annual_coverage=jnp.asarray(np.asarray(vax.annual_coverage, dtype=np.float64)),  # RAW
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=S.AMP, seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day, seasonality_period=disease.seasonality_period,
    )
    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho, C_home=M["C_home"], C_work=M["C_work"],
        C_school=M["C_school"], C_other=M["C_other"], R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + S.AMP)
    H = PC.build_hira_matrix()
    pop_flat = np.asarray(pop_15).sum(axis=1) if np.asarray(pop_15).ndim == 2 else np.asarray(pop_15)
    return dict(shared=shared, ngm=ngm, H=H, pop_6=H @ pop_flat, pop_15=pop_15)


def season_setup(season, common):
    seed = estimate_initial_infected_from_hira(
        season, common["pop_15"].flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15)
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        common["pop_15"], seed, seed_e_factor=0.5, initial_immunity=R0_IMMUNITY_PROFILE,
        initial_vaccinated_fraction=0.0))
    R0 = R0_BY_SEASON[season]
    beta_4 = np.asarray(derive_beta_from_R0_simplex(
        common["ngm"], jnp.asarray(R0), jnp.asarray(PI_SHARED), jnp.asarray(PHI_USHAPE)))
    return dict(shared=common["shared"], state0=state0, ngm=common["ngm"],
                beta_4=beta_4, H=common["H"], pop_6=common["pop_6"]), R0, beta_4


def main():
    print("=" * 96)
    print("HOLIDAY SIGN-REVERSAL — multi-season (sick-leave p_work=0.4, μ=1.0)")
    print(f"  seasons={SEASONS}  π FIXED {list(np.round(PI_SHARED,3))}  per-season R0(joint)")
    print(f"  FIXED windows: term{TERM_WIN} vacation{VAC_WIN}  (peak105 & holiday calendar season-invariant)")
    print("=" * 96)
    t0 = time.perf_counter(); common = build_common()
    print(f"[common setup] {time.perf_counter()-t0:.1f}s\n")

    kap = 1.0 * KAPPA_BASE
    results = {}
    print(f"  {'season':>11} {'R0':>6} | {'window':>10} | {'0-5':>8} {'6-11':>8} {'12-17':>8} | child_sum  reversal")
    for s in SEASONS:
        ts = time.perf_counter()
        setup, R0, b4 = season_setup(s, common)
        base_inf6 = PC.attack6(PC.run(setup, kap, 1.0, 1.0, WHOLE, WHOLE), setup)
        rec = {"R0": R0, "beta_4": [float(x) for x in b4]}
        cs = {}
        for wname, wwin in (("term", TERM_WIN), ("vacation", VAC_WIN)):
            r = PC.policy_eval(setup, kap, 1.0, 0.4, WHOLE, wwin, base_inf6)
            da = r["d_attack_pct"]
            child_sum = sum(da[c] for c in CHILD)
            rec[wname] = dict(averted_pct=r["averted_pct"],
                              child_d_attack={c: da[c] for c in CHILD},
                              all_d_attack=da, child_sum=child_sum)
            cs[wname] = child_sum
        # reversal: term child<0 (benefit) AND vacation child>0 (backfire)
        reversal = (cs["term"] < 0) and (cs["vacation"] > 0)
        rec["reversal"] = bool(reversal)
        results[s] = rec
        for wname in ("term", "vacation"):
            da = rec[wname]["child_d_attack"]; cS = rec[wname]["child_sum"]
            tag = ("Y" if reversal else "N") if wname == "vacation" else ""
            print(f"  {s if wname=='term' else '':>11} {R0 if wname=='term' else '':>6} | {wname:>10} | "
                  f"{da['0-5']:>+8.3f} {da['6-11']:>+8.3f} {da['12-17']:>+8.3f} | {cS:>+8.3f}   {tag}")
        print(f"  {'':>11} {'':>6} | {'':>10} | {'':>8} {'':>8} {'':>8} | ({time.perf_counter()-ts:.1f}s)")

    n_rev = sum(1 for s in SEASONS if results[s]["reversal"])
    print("\n" + "=" * 96)
    print(f"REVERSAL REPRODUCIBILITY: {n_rev}/{len(SEASONS)} seasons")
    print(f"  (term child Δattack < 0 [benefit]  AND  vacation child Δattack > 0 [backfire])")
    for s in SEASONS:
        r = results[s]
        print(f"   {s}: term_child={r['term']['child_sum']:+.3f}  vac_child={r['vacation']['child_sum']:+.3f}  "
              f"reversal={'Y' if r['reversal'] else 'N'}")
    if n_rev < len(SEASONS):
        exc = [s for s in SEASONS if not results[s]["reversal"]]
        print(f"  exceptions: {exc}  (check R0 / peak-vs-window alignment)")
    print("=" * 96)

    out = dict(
        meta=dict(seasons=SEASONS, pi_shared=PI_SHARED.tolist(), R0_by_season=R0_BY_SEASON,
                  kappa_base=KAPPA_BASE.tolist(), term_window=TERM_WIN, vacation_window=VAC_WIN,
                  sick_p_work=0.4, mu=1.0, note="fixed windows all seasons; π+R0 fixed, no refit"),
        results=results, reversal_count=n_rev, n_seasons=len(SEASONS),
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
