"""Phase 2: 시즌별 실측 인구로 point-fit 재적합 → 2023-pop 델타.

각 시즌:
  - Ca = build 2023-pop setup (baseline)
  - Cb = build season-pop setup
  - fit_pi_pin(π_work=0.29) 로 R0/π/β 재추정
  - obs/model total ratio 등 적합품질 비교
"""
from __future__ import annotations
import time, json, sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_common import (
    ED, SEASONS, IDX, IMM_DEF, KAP_DEF, PHI_DEF, GAMMA_15,
    fit_pi_pin, evaluate_full_stratified, HIRA_AGE_GROUPS,
)
from season_pop_setup import build_seasonwise_setup

PI_WORK_PIN = 0.29


def make_season_setup(C_all, s):
    """C_all(시즌별) 에서 s 시즌 하나만 뽑아 fit_pi_pin 호환 dict 생성.

    fit_pi_pin/evaluate_full_stratified 는:
      C["shared"]  = shared params
      C["ngm3"]    = NGM fn
      C["st"][s]   = initial state
      C["obs"], C["w"], C["nw"], C["H"], C["pop6"]
    """
    return dict(
        shared=C_all["shared_by_s"][s],
        ngm3=C_all["ngm3_by_s"][s],
        st={s: C_all["st_by_s"][s]},
        obs=C_all["obs"], w=C_all["w"], nw=C_all["nw"],
        H=C_all["H"], pop6=C_all["pop6_by_s"][s],
        full_obs=C_all["full_obs"],
    )


def main():
    t0 = time.time()
    print("="*80)
    print("Phase 2: point-fit re-calibration with season-specific population")
    print("="*80)

    # Build both setups
    C_new = build_seasonwise_setup(use_season_pop=True)
    C_old = build_seasonwise_setup(use_season_pop=False)   # 2023 default
    print(f"[setup] {time.time()-t0:.1f}s")

    results = []
    for i_s, s in enumerate(SEASONS):
        i = IDX[i_s]
        for tag, C_all in [("2023_default", C_old), ("season_pop", C_new)]:
            C_s = make_season_setup(C_all, s)
            t_c = time.time()
            fit = fit_pi_pin(C_s, s, i, PI_WORK_PIN)
            # 적합품질: obs/model total ratio (첫 피크만)
            from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax
            from kt_epimodel_hira.jax_model.erlang_presymp import (
                simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
                split_seed_to_erlang, ngm_factor,
            )
            from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
            b0 = derive_beta_from_R0_simplex(
                C_s["ngm3"], jnp.asarray(fit["R0"]),
                jnp.asarray(fit["pi"]), jnp.asarray(PHI_DEF))
            beta = b0 / ngm_factor(0.22)
            kw = dict(C_s["shared"])
            kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
            kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
            kw["phi_susc"] = jnp.asarray(PHI_DEF); kw["kappa"] = jnp.asarray(KAP_DEF)
            kw["p_school"] = 0.6; kw["p_work"] = 0.6
            kw["policy_school_start_day"], kw["policy_school_end_day"] = (-1e9, 1e9)
            kw["policy_work_start_day"], kw["policy_work_end_day"] = (-1e9, 1e9)
            kw["policy_school_baseline"] = 0.6; kw["policy_work_baseline"] = 0.6
            st = simulate_jax_erlang_presymp(
                split_seed_to_erlang(C_s["st"][s]), w_presymp=0.22, **kw,
                discretize_time=False)
            pred = np.asarray(simulation_to_hira_by_age_jax(
                daily_new_onset_by_age_erlang_presymp(st),
                jnp.asarray(GAMMA_15), n_weeks=C_s["nw"]))
            obs = np.asarray(C_s["obs"][i]); wt = np.asarray(C_s["w"][i])
            mask = wt.sum(1) > 0
            om_total = float(obs[mask].sum() / max(pred[mask].sum(), 1))
            # per-age
            om_by_age = {ag: float(obs[mask, a].sum() / max(pred[mask, a].sum(), 1))
                          for a, ag in enumerate(HIRA_AGE_GROUPS)}
            wall = time.time() - t_c
            results.append(dict(season=s, pop_source=tag,
                                 R0=fit["R0"], pi=fit["pi"], beta_4=fit["beta_4"],
                                 nll=fit["nll"], om_total=om_total,
                                 om_by_age=om_by_age, wall=wall,
                                 total_pop=float(C_all["pop_15_by_s"][s].sum())))
            print(f"  {s} · {tag:14s}: R0={fit['R0']:.4f}  om_total={om_total:.3f}  "
                  f"β_h={fit['beta_4'][0]:.4f} β_w={fit['beta_4'][1]:.4f}  "
                  f"({wall:.1f}s)")

    # Save + 델타 표
    out = dict(results=results, meta=dict(pi_work_pin=PI_WORK_PIN,
                                            n_wall=time.time()-t0))
    (ED/"pop_repop_phase2_fit.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[saved] {ED/'pop_repop_phase2_fit.json'}  wall={time.time()-t0:.1f}s")

    # 델타 표
    print("\n=== 델타 표 (season_pop vs 2023_default) ===")
    print(f"{'season':>10s}  {'R0_old':>7s}  {'R0_new':>7s}  {'ΔR0%':>6s}  "
          f"{'om_old':>6s}  {'om_new':>6s}  {'β_h_new/old':>12s}")
    for s in SEASONS:
        r_old = next(r for r in results if r["season"]==s and r["pop_source"]=="2023_default")
        r_new = next(r for r in results if r["season"]==s and r["pop_source"]=="season_pop")
        dR = 100*(r_new["R0"]-r_old["R0"])/r_old["R0"]
        d_bh = r_new["beta_4"][0]/r_old["beta_4"][0]
        print(f"{s:>10s}  {r_old['R0']:>7.3f}  {r_new['R0']:>7.3f}  {dR:>+5.2f}%  "
              f"{r_old['om_total']:>6.3f}  {r_new['om_total']:>6.3f}  {d_bh:>11.3f}×")


if __name__ == "__main__":
    main()
