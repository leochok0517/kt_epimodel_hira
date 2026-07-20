"""School-absence vs sick-leave symmetric comparison (exp1) + sick-leave
spillover term-window vs vacation-window (exp2). Point estimate, fixed shared π.

Structure (current confirmed): Step A+B — C(t) term↔vacation, κ 3-way
[0.29,0.30,0], v(t) finite-season norm, RAW coverage. Shared π FIXED
[home0.357, work0.255, school0.067, other0.321] (multi-season joint), R0=2.109
(joint 2019-2020). φ U-shape, γ CDC Reed [0.40,0.18,0.25]. β_4 derived from
(R0, π, φ) — NO refit (deterministic forward sims).

Time-windowed policy via jax_model separate school/work windows: p applies only
inside [start,end], else 1.0. Baseline p=1 → no policy (window irrelevant).

Exp1: two policies (sick=p_work, school=p_school) in the SAME term window
[70,113] (peak day105 inside), grid p×μ. Compare averted sign/size + per-age Δ.
Exp2: sick-leave (p_work=0.4, μ=1) in term-window vs vacation-window vs whole,
under school-absence background p_school_baseline ∈ {1.0, 0.6} (whole-season).

Output: outputs/eda/policy_compare_school_work.json + console.
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
jax.devices()  # early device init (parallel guarantee per task)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "policy_compare_school_work.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
VAC_NPZ = REPO_ROOT.parent / "kt_data/data/external/contact_matrices/empirical_matrices_15_vacation.npz"

if not VAC_NPZ.exists():
    raise SystemExit(f"[ABORT] vacation matrix not found: {VAC_NPZ}")

SEASON = "2019-2020"
PI_SHARED = np.array([0.357, 0.255, 0.067, 0.321])   # multi-season joint
R0_FIXED = 2.109                                       # joint 2019-2020
KAPPA_BASE = np.array([0.29]*4 + [0.30]*10 + [0.0])
PHI_USHAPE = np.array(S.PHI_USHAPE)
GAMMA_15 = np.array(S.build_gamma_15(*S.GAMMA_CENTER))
# term window (rise→peak, before break): peak day105 inside
TERM_WIN = (70.0, 113.0)
VAC_WIN = (113.0, 183.0)
WHOLE_WIN = (-1.0e9, 1.0e9)
ADULT = ["18-44", "45-64"]; SCHOOL_AGE = ["6-11", "12-17"]; CHILD = ["0-5", "6-11", "12-17"]


def build_hira_matrix():
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for idx, w in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, idx] = w
    return H


def build_setup():
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
    seed = estimate_initial_infected_from_hira(
        SEASON, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15)
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed, seed_e_factor=0.5, initial_immunity=R0_IMMUNITY_PROFILE,
        initial_vaccinated_fraction=0.0))
    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho, C_home=M["C_home"], C_work=M["C_work"],
        C_school=M["C_school"], C_other=M["C_other"], R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + S.AMP)
    beta_4 = np.asarray(derive_beta_from_R0_simplex(
        ngm, jnp.asarray(R0_FIXED), jnp.asarray(PI_SHARED), jnp.asarray(PHI_USHAPE)))
    H = build_hira_matrix()
    pop_flat = np.asarray(pop_15).sum(axis=1) if np.asarray(pop_15).ndim == 2 else np.asarray(pop_15)
    return dict(shared=shared, state0=state0, ngm=ngm, beta_4=beta_4, H=H, pop_6=H @ pop_flat)


def run(setup, kappa_vec, p_school, p_work, sch_win=WHOLE_WIN, work_win=WHOLE_WIN):
    kw = dict(setup["shared"])
    b = setup["beta_4"]
    kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = [float(x) for x in b]
    kw["phi_susc"] = jnp.asarray(PHI_USHAPE); kw["kappa"] = jnp.asarray(kappa_vec)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = np.asarray(daily_new_infection_by_age_jax(st))    # (T-1, 15)
    return inc


def attack6(inc, setup):
    return setup["H"] @ inc.sum(axis=0)                      # (6,) infections per HIRA group


def policy_eval(setup, kappa_vec, p_school, p_work, sch_win, work_win, base_inf6):
    inc = run(setup, kappa_vec, p_school, p_work, sch_win, work_win)
    inf6 = attack6(inc, setup); pop6 = setup["pop_6"]
    averted = 100.0 * (base_inf6.sum() - inf6.sum()) / max(base_inf6.sum(), 1.0)
    d_attack = (inf6 - base_inf6) / pop6                     # fraction; + = more infections
    da = {ag: float(100.0*d_attack[i]) for i, ag in enumerate(HIRA_AGE_GROUPS)}
    dc = d_attack * pop6
    pos = float(dc[dc > 0].sum()); neg = float(-dc[dc < 0].sum())
    transfer = (pos/neg) if neg > 1e-9 else float("inf")
    return dict(averted_pct=float(averted), d_attack_pct=da, transfer=transfer,
                total_inf=float(inf6.sum()))


def main():
    print("=" * 100)
    print("POLICY COMPARE — school-absence vs sick-leave + sick spillover term/vacation")
    print(f"  {SEASON} | π FIXED {list(np.round(PI_SHARED,3))} | R0={R0_FIXED} | κ_base=[.29,.30,0] | φ U-shape")
    print(f"  windowed policy: term{TERM_WIN} vacation{VAC_WIN}; baseline p=1 → no policy")
    print("=" * 100)
    t0 = time.perf_counter(); setup = build_setup()
    print(f"[setup] {time.perf_counter()-t0:.1f}s  β_4={[round(float(x),4) for x in setup['beta_4']]}\n")

    # ================= EXPERIMENT 1 =================
    P_GRID = [1.0, 0.8, 0.6, 0.4, 0.2]
    MU_GRID = [0.5, 1.0, 1.5]
    exp1 = {"sick": {}, "school": {}}
    print("── EXP1: averted% grid (policy × p × μ), TERM window [70,113] ──")
    for mu in MU_GRID:
        kap = mu * KAPPA_BASE
        base_inf6 = attack6(run(setup, kap, 1.0, 1.0), setup)   # baseline (no policy)
        for pol in ("sick", "school"):
            for p in P_GRID:
                if pol == "sick":
                    r = policy_eval(setup, kap, 1.0, p, WHOLE_WIN, TERM_WIN, base_inf6)
                else:
                    r = policy_eval(setup, kap, p, 1.0, TERM_WIN, WHOLE_WIN, base_inf6)
                exp1[pol][f"mu{mu}_p{p}"] = r
    # print averted grids
    for pol in ("sick", "school"):
        print(f"\n  [{pol}]  averted%   rows=p(strength 1-p)  cols=μ")
        print("   p\\μ " + "".join(f"{mu:>9.1f}" for mu in MU_GRID))
        for p in P_GRID:
            row = "".join(f"{exp1[pol][f'mu{mu}_p{p}']['averted_pct']:>+9.2f}" for mu in MU_GRID)
            print(f"   {p:.1f} {row}")
    # per-age Δattack contrast at p=0.4, μ=1.0
    print("\n  PER-AGE Δattack (%pt) at p=0.4 (60% strength), μ=1.0  [+ = load received]")
    for pol in ("sick", "school"):
        r = exp1[pol]["mu1.0_p0.4"]; da = r["d_attack_pct"]
        print(f"   {pol:>6}: averted={r['averted_pct']:+.2f}% transfer={r['transfer']:.3f}")
        print("          " + "  ".join(f"{ag}:{da[ag]:+.3f}" for ag in HIRA_AGE_GROUPS))

    # ================= EXPERIMENT 2 =================
    print("\n" + "─" * 100)
    print("── EXP2: sick-leave (p_work=0.4, μ=1.0) spillover — window × school-absence background ──")
    exp2 = {}
    windows = {"term[70,113]": TERM_WIN, "vacation[113,183]": VAC_WIN, "whole": WHOLE_WIN}
    kap = 1.0 * KAPPA_BASE
    for psb in (1.0, 0.6):
        exp2[f"p_school_base={psb}"] = {}
        # baseline: no sick-leave, but school-absence background psb applied whole season
        base_inf6 = attack6(run(setup, kap, psb, 1.0, WHOLE_WIN, WHOLE_WIN), setup)
        for wname, wwin in windows.items():
            r = policy_eval(setup, kap, psb, 0.4, WHOLE_WIN, wwin, base_inf6)
            exp2[f"p_school_base={psb}"][wname] = r
    print(f"\n  {'p_school_base':>14} {'window':>18} | {'averted%':>9} | child Δattack (0-5, 6-11, 12-17) %pt")
    for psb in (1.0, 0.6):
        for wname in windows:
            r = exp2[f"p_school_base={psb}"][wname]; da = r["d_attack_pct"]
            child = "  ".join(f"{da[c]:+.3f}" for c in CHILD)
            print(f"  {psb:>14} {wname:>18} | {r['averted_pct']:>+8.2f}% | {child}")

    # C(t) contact & spill_factor snapshots (counts view)
    print("\n  C(t) channel rowsum: term(day100,h=0) vs vacation(day150,h=1)")
    from kt_epimodel_hira.jax_model.foi_jax import vacation_weight, compute_phi_spillover
    Ct = {c: np.asarray(setup["shared"][f"C_{c}"]) for c in ("home", "school", "work", "other")}
    Cv = {c: np.asarray(setup["shared"][f"C_{c}_vac"]) for c in ("home", "school", "work", "other")}
    for day in (100.0, 150.0):
        h = float(vacation_weight(jnp.asarray(day)))
        sums = {c: float(((1-h)*Ct[c] + h*Cv[c]).sum()) for c in Ct}
        print(f"    day{int(day):>3} h={h:.2f}: " + " ".join(f"{c}={sums[c]:.2f}" for c in ("home","school","work","other")))
    # spill_factor under sick-leave (p_work=0.4): worker slice
    rho = np.asarray(setup["shared"]["rho"])
    phi_spill = np.asarray(compute_phi_spillover(1.0, 0.4, jnp.asarray(rho)))
    sf_worker = 1.0 + (1.0*KAPPA_BASE)[None, :] * phi_spill
    print(f"  spill_factor (p_work=0.4, μ=1) worker idx4-13 mean = {sf_worker[:,4:14].mean():.4f} (home amplification when workers stay home)")

    # sign summary
    print("\n  [SIGN SUMMARY]")
    s04 = exp1["sick"]["mu1.0_p0.4"]["averted_pct"]; c04 = exp1["school"]["mu1.0_p0.4"]["averted_pct"]
    print(f"   exp1 @p=0.4,μ=1: sick={s04:+.2f}% school={c04:+.2f}%")
    for psb in (1.0, 0.6):
        t = exp2[f'p_school_base={psb}']['term[70,113]']; v = exp2[f'p_school_base={psb}']['vacation[113,183]']
        tc = sum(t['d_attack_pct'][c] for c in CHILD); vc = sum(v['d_attack_pct'][c] for c in CHILD)
        print(f"   exp2 p_sch_base={psb}: child Δ term={tc:+.3f} vacation={vc:+.3f} → vacation {'SMALLER' if abs(vc)<abs(tc) else 'NOT smaller'} (hypothesis: smaller)")
    print("=" * 100)

    out = dict(
        meta=dict(season=SEASON, pi_shared=PI_SHARED.tolist(), R0=R0_FIXED,
                  kappa_base=KAPPA_BASE.tolist(), term_window=TERM_WIN, vacation_window=VAC_WIN,
                  beta_4=[float(x) for x in setup["beta_4"]], phi="U-shape", gamma_15=GAMMA_15.tolist(),
                  note="policy time-windowed via jax_model separate school/work windows; pi fixed, no refit"),
        exp1_school_vs_sick=exp1, exp2_spillover_window=exp2,
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
