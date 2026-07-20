"""Seed γ-fix: recompute initial seed with CONFIRMED γ (12-17=0.18) instead of
old γ (12-17=0.40), then Erlang(3) refit. Check timing improvement.
확정 파라미터·Erlang·baseline 0.6 위에서. seed는 데이터 산출 유지(fit 아님).
Output: outputs/eda/seed_gamma_fix.json
"""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np, jax; jax.config.update("jax_enable_x64",True); jax.devices()
import jax.numpy as jnp
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE)
import final_pipeline_confirmed as F
import erlang_fit as EF

ED = Path(__file__).resolve().parent.parent/"outputs"/"eda"
SEAS = EF.SEAS; IDX = EF.IDX
GAMMA_NEW = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])

def main():
    print("="*88); print("SEED γ-FIX (12-17: 0.40→0.18) + Erlang(3) refit"); print("="*88)
    t0=time.perf_counter(); C=F.build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
    old = json.load(open(ED/"erlang_fit.json"))["results"]   # old-seed Erlang
    pop15 = np.asarray(C["shared"]["pop_15"])
    # recompute seeds with confirmed γ, replace states
    for s,i in zip(SEAS,IDX):
        seed = estimate_initial_infected_from_hira(s, pop15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
                    gamma_15_assumed=GAMMA_NEW)
        C["states"][i] = jnp.asarray(_build_initial_state_with_age_seed(
            pop15, seed, seed_e_factor=0.5, initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0))
    res={}
    for s,i in zip(SEAS,IDX):
        f = EF.fit_erlang(C,i)
        pe = EF.pred_erlang(C,i,f["R0"],f["pi"])
        me = EF.metrics(C,i,pe)
        da,adown = EF.redistribution(C,i,f["R0"],f["pi"])
        o = old[s]["erlang"]
        res[s]=dict(R0=f["R0"], **me, redist_adult_down=bool(adown))
        print(f"\n  [{s}]")
        print(f"    peak주(mdl/obs): 구seed_Erl={o['mdl_pk']}/{o['obs_pk']} → 신seed_Erl={me['mdl_pk']}/{me['obs_pk']}")
        print(f"    width_ratio: {o['width_ratio']:.2f} → {me['width_ratio']:.2f}")
        print(f"    obs/model total: {o['om_total']:.2f} → {me['om_total']:.2f}")
        print(f"    재분배 성인↓={adown}")
    print("\n"+"="*88)
    te = sum(abs(res[s]["mdl_pk"]-res[s]["obs_pk"]) for s in SEAS)
    te_old = sum(abs(old[s]["erlang"]["mdl_pk"]-old[s]["erlang"]["obs_pk"]) for s in SEAS)
    print(f"★ 타이밍오차합: 구seed={te_old}주 → 신seed(확정γ)={te}주")
    print("="*88)
    (ED/"seed_gamma_fix.json").write_text(json.dumps(dict(
        meta=dict(gamma_new=GAMMA_NEW.tolist(), stages=3, baseline=0.6),
        results=res, timing_err=te, timing_err_old=te_old), indent=2, default=float))
    print(f"[json] {ED/'seed_gamma_fix.json'}")

if __name__=="__main__": main()
