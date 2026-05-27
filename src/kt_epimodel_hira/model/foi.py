"""Force of Infection — 4채널 분해 (Step C).

채널별 FOI:
- home   : λ^h = β_h · φ_a · Σ_a' C^h[a,a'] · I[a',i]·(1 + κ_a'·φ_spill[i,a']) / N[a',i]
- school : λ^s = β_s · φ_a · p_school · Σ_a' C^s[a,a'] · I[a',i] / N[a',i] (학생만)
- work   : λ^w = β_w · φ_a · ρ[i,a] · M^w 매개 mixing × p_work (근로자만)
- other  : λ^o = β_o · φ_a · M^o 매개 mixing (모든 연령)

Sick-leave 결과 가구 노출 보정:
- φ_spill[i, a'] = 학생 (1-p_school), 근로자 ρ[i,a']·(1-p_work), 70+ 0
"""

from __future__ import annotations

import numpy as np

from kt_epimodel_hira.model.compartments import IDX_I, N_AGE
from kt_epimodel_hira.model.parameters import ModelParameters

# 연령 그룹
STUDENT_SLICE = slice(0, 4)      # NIMS 0-3 (0-19세)
WORKER_SLICE = slice(4, 14)      # NIMS 4-13 (20-69세)

_EPS: float = 1e-10


# ---------------------------------------------------------------------------
# Sick-leave fraction helpers
# ---------------------------------------------------------------------------

def compute_phi_school(p_school: float) -> np.ndarray:
    """학교 결석 비율 (15,) — 학생 (0-3): 1-p_school, 그 외 0."""
    phi = np.zeros(N_AGE, dtype=np.float64)
    phi[STUDENT_SLICE] = 1.0 - p_school
    return phi


def compute_phi_work(p_work: float, rho: np.ndarray) -> np.ndarray:
    """직장 결근 비율 (n_admdong, 15) — 근로자만 ρ·(1-p_work)."""
    phi = np.zeros_like(rho)
    phi[:, WORKER_SLICE] = rho[:, WORKER_SLICE] * (1.0 - p_work)
    return phi


def compute_phi_spillover(
    p_school: float, p_work: float, rho: np.ndarray,
) -> np.ndarray:
    """가구 spillover 비율 (n_admdong, 15) — 학생 + 근로자 합."""
    phi = np.zeros_like(rho)
    phi[:, STUDENT_SLICE] = 1.0 - p_school
    phi[:, WORKER_SLICE] = rho[:, WORKER_SLICE] * (1.0 - p_work)
    return phi


# ---------------------------------------------------------------------------
# Per-channel FOI
# ---------------------------------------------------------------------------

def compute_foi_home(
    state: np.ndarray,
    C_home: np.ndarray,
    pop_15: np.ndarray,
    rho: np.ndarray,
    kappa: np.ndarray,
    p_school: float,
    p_work: float,
    beta_h: float,
    phi_susc: np.ndarray,
    seasonal_factor: float = 1.0,
) -> np.ndarray:
    """Home (자기 행정동) + spillover. (15, n_admdong)."""
    I = state[IDX_I]                                       # (15, n)
    N = pop_15
    N_safe = np.maximum(N, _EPS)

    phi_spill = compute_phi_spillover(p_school, p_work, rho)   # (n, 15)
    # spill_factor[i, a'] = 1 + κ[a']·φ_spill[i, a']
    spill_factor = 1.0 + kappa[None, :] * phi_spill            # (n, 15)
    I_eff = I * spill_factor.T                                 # (15, n)

    # C[a, a'] · I_eff[a', i] / N[a', i]
    contact_pressure = C_home @ (I_eff / N_safe)               # (15, n)
    foi_h = (beta_h * seasonal_factor) * phi_susc[:, None] * contact_pressure
    return np.where(N > _EPS, foi_h, 0.0)


def compute_foi_school(
    state: np.ndarray,
    C_school: np.ndarray,
    pop_15: np.ndarray,
    p_school: float,
    beta_s: float,
    phi_susc: np.ndarray,
    seasonal_factor: float = 1.0,
) -> np.ndarray:
    """School (학생만, M^s = identity 가정)."""
    I = state[IDX_I]
    N = pop_15
    N_safe = np.maximum(N, _EPS)
    n_adm = N.shape[1]

    I_eff = np.zeros_like(I)
    I_eff[STUDENT_SLICE] = p_school * I[STUDENT_SLICE]

    contact_pressure = C_school @ (I_eff / N_safe)             # (15, n)

    foi_s = np.zeros((N_AGE, n_adm), dtype=np.float64)
    foi_s[STUDENT_SLICE] = (
        (beta_s * seasonal_factor)
        * phi_susc[STUDENT_SLICE, None] * contact_pressure[STUDENT_SLICE]
    )
    return np.where(N > _EPS, foi_s, 0.0)


def compute_foi_work(
    state: np.ndarray,
    C_work: np.ndarray,
    pop_15: np.ndarray,
    M_work: np.ndarray,
    rho: np.ndarray,
    p_work: float,
    beta_w: float,
    phi_susc: np.ndarray,
    seasonal_factor: float = 1.0,
) -> np.ndarray:
    """Work (근로자만, M_work commuting). (15, n)."""
    I = state[IDX_I]
    N = pop_15
    n_adm = N.shape[1]

    rho_T = rho.T                                              # (15, n)
    weighted_I = rho_T * I
    weighted_N = rho_T * N

    # 직장 j 에 있는 근로자
    I_at_j = np.einsum("akj,ak->aj", M_work, weighted_I)       # (15, n)
    N_at_j = np.einsum("akj,ak->aj", M_work, weighted_N)
    N_at_j_safe = np.maximum(N_at_j, _EPS)

    ratio_at_j = p_work * I_at_j / N_at_j_safe                 # (15, n)
    contact_pressure_at_j = C_work @ ratio_at_j                # (15, n)

    # 거주지 i 로 복귀 (residents of i at j)
    pressure_at_i = np.einsum("aij,aj->ai", M_work, contact_pressure_at_j)
    foi_w_all = (beta_w * seasonal_factor) * phi_susc[:, None] * rho_T * pressure_at_i

    foi_w = np.zeros((N_AGE, n_adm), dtype=np.float64)
    foi_w[WORKER_SLICE] = foi_w_all[WORKER_SLICE]
    return np.where(N > _EPS, foi_w, 0.0)


def compute_foi_other(
    state: np.ndarray,
    C_other: np.ndarray,
    pop_15: np.ndarray,
    M_other: np.ndarray,
    beta_o: float,
    phi_susc: np.ndarray,
    seasonal_factor: float = 1.0,
) -> np.ndarray:
    """Other (저녁/여가, 모든 연령, M_other commuting)."""
    I = state[IDX_I]
    N = pop_15

    I_at_j = np.einsum("akj,ak->aj", M_other, I)
    N_at_j = np.einsum("akj,ak->aj", M_other, N)
    N_at_j_safe = np.maximum(N_at_j, _EPS)

    ratio_at_j = I_at_j / N_at_j_safe
    contact_pressure_at_j = C_other @ ratio_at_j
    pressure_at_i = np.einsum("aij,aj->ai", M_other, contact_pressure_at_j)

    foi_o = (beta_o * seasonal_factor) * phi_susc[:, None] * pressure_at_i
    return np.where(N > _EPS, foi_o, 0.0)


# ---------------------------------------------------------------------------
# Combined FOI
# ---------------------------------------------------------------------------

def compute_foi(
    state: np.ndarray,
    mobility: dict[str, np.ndarray],
    matrices: dict[str, np.ndarray],
    pop_15: np.ndarray,
    params: ModelParameters,
    day_in_season: float = 0.0,
) -> np.ndarray:
    """4채널 FOI 합산. day_in_season → seasonal_factor."""
    if params.employment is None:
        raise ValueError("params.employment must be set (call with_employment)")

    rho = params.employment.rho
    if rho.shape[0] != pop_15.shape[1]:
        raise ValueError(
            f"rho admdong dim {rho.shape[0]} != pop_15 admdong {pop_15.shape[1]}"
        )

    kappa = params.disease.kappa_array
    p_school = params.policy.p_school
    p_work = params.policy.p_work
    phi_susc = params.calibration.phi
    betas = params.calibration.betas
    sf = params.disease.seasonal_factor(day_in_season)

    foi_h = compute_foi_home(
        state, matrices["C_home"], pop_15, rho, kappa,
        p_school, p_work, betas["home"], phi_susc, seasonal_factor=sf,
    )
    foi_s = compute_foi_school(
        state, matrices["C_school"], pop_15,
        p_school, betas["school"], phi_susc, seasonal_factor=sf,
    )
    foi_w = compute_foi_work(
        state, matrices["C_work"], pop_15, mobility["work"],
        rho, p_work, betas["work"], phi_susc, seasonal_factor=sf,
    )
    foi_o = compute_foi_other(
        state, matrices["C_other"], pop_15, mobility["other"],
        betas["other"], phi_susc, seasonal_factor=sf,
    )
    return foi_h + foi_s + foi_w + foi_o


def assemble_contact_matrix(
    matrices: dict[str, np.ndarray],
    daytype: str,
    lambdas: dict[str, float],
) -> np.ndarray:
    """Legacy: λ·C 합산 → (15, 15). 새 모델에선 β_ch 가 흡수 — 호환 유지."""
    required = {"home", "work", "school", "other"}
    missing = required - set(lambdas.keys())
    if missing:
        raise ValueError(f"lambdas missing channels for daytype {daytype!r}: {missing}")
    return (
        lambdas["home"] * matrices["C_home"]
        + lambdas["work"] * matrices["C_work"]
        + lambdas["school"] * matrices["C_school"]
        + lambdas["other"] * matrices["C_other"]
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from kt_data import (
        build_rho_matrix,
        get_population_matrix,
        load_contact_matrices,
        load_population_15groups,
    )

    from kt_epimodel_hira.model.compartments import initial_state
    from kt_epimodel_hira.model.mobility_tensor import (
        build_M_home,
        build_M_other,
        build_M_school,
        build_M_work,
    )
    from kt_epimodel_hira.model.parameters import (
        AGE_LABELS_15,
        EmploymentParameters,
        ModelParameters,
        PolicyParameters,
    )

    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop = N_mat.T.astype(np.float64)
    n = pop.shape[1]

    matrices = load_contact_matrices()

    M_home = build_M_home(n)
    M_school = build_M_school(n)
    M_work = build_M_work("202301", admdong_codes=codes, pop_15=pop)
    M_other = build_M_other("202301", admdong_codes=codes, pop_15=pop)
    mobility = {"home": M_home, "school": M_school, "work": M_work, "other": M_other}

    params = ModelParameters().with_employment(EmploymentParameters.from_kt_data(codes))
    state = initial_state(pop, seed_per_admdong=10)

    foi_h = compute_foi_home(
        state, matrices["C_home"], pop, params.employment.rho,
        params.disease.kappa_array, params.policy.p_school, params.policy.p_work,
        params.calibration.beta_h, params.calibration.phi,
    )
    foi_s = compute_foi_school(
        state, matrices["C_school"], pop,
        params.policy.p_school, params.calibration.beta_s, params.calibration.phi,
    )
    foi_w = compute_foi_work(
        state, matrices["C_work"], pop, M_work,
        params.employment.rho, params.policy.p_work,
        params.calibration.beta_w, params.calibration.phi,
    )
    foi_o = compute_foi_other(
        state, matrices["C_other"], pop, M_other,
        params.calibration.beta_o, params.calibration.phi,
    )

    print("=== FOI by channel (mean) ===")
    print(f"  home:   {foi_h.mean():.3e}")
    print(f"  school: {foi_s.mean():.3e}")
    print(f"  work:   {foi_w.mean():.3e}")
    print(f"  other:  {foi_o.mean():.3e}")

    foi_total = compute_foi(state, mobility, matrices, pop, params)
    print(f"\n  total:  {foi_total.mean():.3e}")

    print("\n=== FOI by age (total) ===")
    for a, label in enumerate(AGE_LABELS_15):
        print(f"  [{a:>2}] {label:>5}: {foi_total[a].mean():.3e}")

    print("\n=== Policy scenario effect ===")
    for name, pol in [
        ("baseline", PolicyParameters.baseline()),
        ("school_closure", PolicyParameters.school_closure()),
        ("sick_leave", PolicyParameters.sick_leave_enhanced()),
        ("comprehensive", PolicyParameters.comprehensive()),
    ]:
        p = params.with_policy(pol)
        foi = compute_foi(state, mobility, matrices, pop, p)
        print(
            f"  {name:>16}: total={foi.mean():.3e}, "
            f"students={foi[0:4].mean():.3e}, workers={foi[4:14].mean():.3e}"
        )
