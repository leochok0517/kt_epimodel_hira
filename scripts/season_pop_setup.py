"""Season-aware setup builder — pop_15 를 시즌별로 override.

- build_setup_by_season(pop_override=None): sens_common.build_setup 기반이지만,
  전체 시즌 모두 각 시즌 pop 배열을 개별 state0 / ngm 로 재빌드.
- 반환: dict[season] → season-specific C (setup) 병합 구조는 아니고,
  각 시즌마다 별도 state0/ngm 을 저장한 combined dict.

사용:
  C = build_seasonwise_setup()   # 시즌별 pop 반영
  # C["ngm3"]           → dict[season] → ngm_fn  (기존은 단일 ngm)
  # C["st"][season]     → initial state (기존 동일)
  # C["pop_15_by_s"]    → dict[season] → np.ndarray (15,)
  # C["pop6_by_s"]      → dict[season] → np.ndarray (6,) HIRA
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pop_seasons import load_pop_by_season, SEASON_MAP
from sens_common import (
    build_setup as build_setup_default,
    SEASONS, IDX, KAP_DEF, PHI_DEF, IMM_DEF,
)
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.calibration.hira_target import (
    HIRA_AGE_GROUPS, HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn
import final_pipeline_confirmed as F


def build_hira_matrix() -> np.ndarray:
    """HIRA 6군 × NIMS 15 가중 매트릭스. (6, 15)."""
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for j, w in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, j] = w
    return H


def build_seasonwise_setup(imm=None,
                            gamma_15=None,
                            use_season_pop: bool = True):
    """시즌별 인구를 반영한 setup.

    use_season_pop=False 이면 기존 2023 default (regression test 용).
    반환 dict:
      shared_default: 기존 build_setup 반환 (2023 pop 기반, obs/w 등 포함).
      ngm3_by_s:      dict[season] → ngm_eigval_fn (시즌별 pop 반영)
      st_by_s:        dict[season] → initial_state jnp array (시즌별 pop 반영)
      pop_15_by_s:    dict[season] → np.ndarray (15,)
      pop6_by_s:      dict[season] → np.ndarray (6,) HIRA
      H:              (6, 15) HIRA 매핑
      obs, w, nw:     기존 (2023 setup 에서 그대로 - 관측값은 인구와 무관)
    """
    imm_use = np.asarray(imm) if imm is not None else IMM_DEF
    gamma_use = np.asarray(gamma_15) if gamma_15 is not None \
        else np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])

    # 기존 setup — obs, w, nw, 대부분 shared 컴포넌트 확보
    default = build_setup_default(imm=imm_use)

    # 시즌별 pop
    if use_season_pop:
        pop_dict = load_pop_by_season()   # dict[season] → (15,)
    else:
        pop_flat = np.asarray(default["shared"]["pop_15"]).flatten()
        pop_dict = {s: pop_flat.copy() for s in SEASONS}

    M = default["shared"]
    disease = F.MJ.ModelParameters().disease
    rho_arr = np.asarray(M["rho"])
    C_mats = dict(C_home=np.asarray(M["C_home"]), C_work=np.asarray(M["C_work"]),
                   C_school=np.asarray(M["C_school"]), C_other=np.asarray(M["C_other"]))

    ngm3_by_s = {}
    st_by_s = {}
    pop_15_by_s = {}
    for s in SEASONS:
        pop_flat = pop_dict[s]
        pop_2d = pop_flat.reshape(15, 1)
        pop_15_by_s[s] = pop_flat
        # NGM 재빌드
        ngm3_by_s[s] = make_ngm_eigvalue_fn(
            pop_15=pop_2d, rho=rho_arr,
            **C_mats,
            R0_immunity=imm_use, gamma=disease.gamma,
            seasonal_factor=1.0 + F.S.AMP,
        )
        # Initial state 재빌드
        seed = estimate_initial_infected_from_hira(
            s, pop_flat, sido_codes=list(SUDOGWON_SIDO_CODES),
            gamma_15_assumed=gamma_use)
        st_by_s[s] = jnp.asarray(_build_initial_state_with_age_seed(
            pop_2d, seed, seed_e_factor=0.5, initial_immunity=imm_use,
            initial_vaccinated_fraction=0.0))

    # HIRA-6 pop
    H = build_hira_matrix()
    pop6_by_s = {s: H @ pop_15_by_s[s] for s in SEASONS}

    # shared_base 도 season 별 pop 를 채워 넣기 위해 dict of shared 준비
    shared_by_s = {}
    for s in SEASONS:
        sh = dict(M)
        sh["pop_15"] = jnp.asarray(pop_15_by_s[s].reshape(15, 1))
        shared_by_s[s] = sh

    return dict(
        default_setup=default,
        ngm3_by_s=ngm3_by_s,
        st_by_s=st_by_s,
        shared_by_s=shared_by_s,
        pop_15_by_s=pop_15_by_s,
        pop6_by_s=pop6_by_s,
        H=H,
        obs=default["obs"], w=default["w"], nw=default["nw"],
        full_obs=default["full_obs"],
    )


if __name__ == "__main__":
    C = build_seasonwise_setup(use_season_pop=True)
    print("=== season-wise setup ===")
    for s in SEASONS:
        p = C["pop_15_by_s"][s]
        p6 = C["pop6_by_s"][s]
        print(f"  {s}: pop_15 total={p.sum():,.0f}  pop6=[{','.join(f'{v:,.0f}' for v in p6)}]")

    # Regression test: 2023 default 로 돌리면 기존 sens_common.build_setup 과 일치해야 함
    print("\n=== regression: use_season_pop=False vs default ===")
    C_def = build_seasonwise_setup(use_season_pop=False)
    default = C_def["default_setup"]
    for s in SEASONS:
        p_new = C_def["pop_15_by_s"][s]
        p_default = np.asarray(default["shared"]["pop_15"]).flatten()
        diff = np.abs(p_new - p_default).max()
        print(f"  {s}: pop diff max={diff:.2e}")
