"""Erlang I n-stage + presymptomatic — 격리 신규 모듈 (n 파라미터화).

erlang_presymp.py 의 일반화. compartments: [S, V, E, I₁, ..., I_n, R] (5+n).
  I₁ = presymptomatic (전파력 w<1, 정책 무관)
  I₂..I_n = symptomatic (전파력 1, 정책 p 적용, home spillover 대상)
Observation: daily new symptom onset = d(I₂ + .. + I_n + R)/dt (I₁→I₂ 유입 누적).
Rate per stage = n·γ (mean 1/γ 보존).

NGM factor (평균 실효 전파시간):
  w·(1/n) + 1·((n-1)/n) = (w + n - 1) / n
  ⇒ β_effective = β_ngm_inv / factor(n, w)

기존 erlang_presymp.py (N_STAGES=3, w=0.22 fixed) 무수정 유지.
"""
from __future__ import annotations
import jax.numpy as jnp
from diffrax import (ODETerm, Dopri5, Tsit5, diffeqsolve, SaveAt,
                     PIDController, RESULTS)
from kt_epimodel_hira.jax_model.foi_jax import (
    STUDENT_SLICE, WORKER_SLICE, N_AGE, _EPS,
    seasonal_factor_cosine, school_calendar_mult,
    vacation_weight, policy_window_weight,
    compute_phi_spillover,
)
from kt_epimodel_hira.jax_model.dynamics_jax import vax_rate_vector_jax

# Compartment index helpers (dynamic on n)
# state.shape = (5 + n, 15, n_admdong)
# S=0, V=1, E=2, I₁=3, ..., I_n = 2+n, R = 3+n


def E_idx(n_stages: int) -> int: return 2
def I_start(n_stages: int) -> int: return 3            # I₁ index
def I_end(n_stages: int) -> int: return 3 + n_stages    # exclusive
def R_idx(n_stages: int) -> int: return 3 + n_stages


def ngm_factor(w: float, n_stages: int) -> float:
    """평균 실효 전파시간 factor = (w + n - 1) / n."""
    return (w + n_stages - 1.0) / n_stages


def w_for_presymp_fraction(fraction: float, n_stages: int) -> float:
    """목표 presymp 전파 비율 fraction (예: 0.10) 을 만드는 w 값.
    fraction = w·(1/n) / (w·(1/n) + 1·((n-1)/n)) = w / (w + n - 1)
    ⇒ w = fraction · (n-1) / (1 - fraction)
    n=1 (분리 없음) → w=0 (무의미), 함수에선 1 반환 (I 전체가 하나).
    """
    if n_stages <= 1:
        return 1.0
    return fraction * (n_stages - 1) / (1.0 - fraction)


def split_seed_to_n(state5: jnp.ndarray, n_stages: int) -> jnp.ndarray:
    """(5, 15, n_adm) → (5+n, 15, n_adm).  I → I₁..I_n 각 1/n."""
    S = state5[0]; V = state5[1]; E = state5[2]
    I = state5[3]; R = state5[4]
    piece = I / n_stages
    stages = [piece] * n_stages
    return jnp.stack([S, V, E, *stages, R], axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# FOI (I₁ vs I₂..I_n 분리)
# ═══════════════════════════════════════════════════════════════════════════
def _foi_home_n(I1, Irest, C_home, pop_15, rho, kappa,
                p_school, p_work, beta_h, phi_susc,
                seasonal_factor, w_presymp):
    N = pop_15; N_safe = jnp.maximum(N, _EPS)
    phi_spill = compute_phi_spillover(p_school, p_work, rho)
    spill_factor = 1.0 + kappa[None, :] * phi_spill
    I_eff = w_presymp * I1 + Irest * spill_factor.T
    contact_pressure = C_home @ (I_eff / N_safe)
    foi_h = (beta_h * seasonal_factor) * phi_susc[:, None] * contact_pressure
    return jnp.where(N > _EPS, foi_h, 0.0)


def _foi_school_n(I1, Irest, C_school, pop_15, p_school, beta_s,
                   phi_susc, seasonal_factor, w_presymp):
    N = pop_15; N_safe = jnp.maximum(N, _EPS); n_adm = N.shape[1]
    I_eff = jnp.zeros_like(I1)
    I_eff = I_eff.at[STUDENT_SLICE].set(
        w_presymp * I1[STUDENT_SLICE] + p_school * Irest[STUDENT_SLICE])
    contact_pressure = C_school @ (I_eff / N_safe)
    foi_s_all = (beta_s * seasonal_factor) * phi_susc[:, None] * contact_pressure
    foi_s = jnp.zeros((N_AGE, n_adm), dtype=jnp.float64)
    foi_s = foi_s.at[STUDENT_SLICE].set(foi_s_all[STUDENT_SLICE])
    return jnp.where(N > _EPS, foi_s, 0.0)


def _foi_work_n(I1, Irest, C_work, pop_15, M_work, rho, p_work, beta_w,
                 phi_susc, seasonal_factor, w_presymp):
    N = pop_15; n_adm = N.shape[1]
    rho_T = rho.T
    I_eff = w_presymp * I1 + p_work * Irest
    weighted_I = rho_T * I_eff; weighted_N = rho_T * N
    I_at_j = jnp.einsum("akj,ak->aj", M_work, weighted_I)
    N_at_j = jnp.einsum("akj,ak->aj", M_work, weighted_N)
    N_at_j_safe = jnp.maximum(N_at_j, _EPS)
    ratio_at_j = I_at_j / N_at_j_safe
    contact_pressure_at_j = C_work @ ratio_at_j
    pressure_at_i = jnp.einsum("aij,aj->ai", M_work, contact_pressure_at_j)
    foi_w_all = (beta_w * seasonal_factor) * phi_susc[:, None] * rho_T * pressure_at_i
    foi_w = jnp.zeros((N_AGE, n_adm), dtype=jnp.float64)
    foi_w = foi_w.at[WORKER_SLICE].set(foi_w_all[WORKER_SLICE])
    return jnp.where(N > _EPS, foi_w, 0.0)


def _foi_other_n(I1, Irest, C_other, pop_15, M_other, beta_o, phi_susc,
                  seasonal_factor, w_presymp):
    N = pop_15
    I_eff = w_presymp * I1 + Irest
    I_at_j = jnp.einsum("akj,ak->aj", M_other, I_eff)
    N_at_j = jnp.einsum("akj,ak->aj", M_other, N)
    N_at_j_safe = jnp.maximum(N_at_j, _EPS)
    ratio_at_j = I_at_j / N_at_j_safe
    contact_pressure_at_j = C_other @ ratio_at_j
    pressure_at_i = jnp.einsum("aij,aj->ai", M_other, contact_pressure_at_j)
    foi_o = (beta_o * seasonal_factor) * phi_susc[:, None] * pressure_at_i
    return jnp.where(N > _EPS, foi_o, 0.0)


def compute_foi_n(state, n_stages, *,
    C_home, C_school, C_work, C_other,
    M_home, M_school, M_work, M_other,
    pop_15, rho, kappa,
    p_school, p_work,
    beta_h, beta_w, beta_s, beta_o,
    phi_susc, seasonal_factor, w_presymp,
):
    I1 = state[I_start(n_stages)]
    if n_stages == 1:
        # n=1 → I₁ 만, w=1 (전 I 가 정책 적용) 이라 Irest=0 무의미하지만 formula 일관성 위해
        Irest = jnp.zeros_like(I1)
    else:
        # I₂ + I₃ + ... + I_n
        Irest = state[I_start(n_stages)+1 : I_end(n_stages)].sum(axis=0)
    foi_h = _foi_home_n(I1, Irest, C_home, pop_15, rho, kappa,
                          p_school, p_work, beta_h, phi_susc,
                          seasonal_factor, w_presymp)
    foi_s = _foi_school_n(I1, Irest, C_school, pop_15,
                            p_school, beta_s, phi_susc,
                            seasonal_factor, w_presymp)
    foi_w = _foi_work_n(I1, Irest, C_work, pop_15, M_work, rho,
                          p_work, beta_w, phi_susc,
                          seasonal_factor, w_presymp)
    foi_o = _foi_other_n(I1, Irest, C_other, pop_15, M_other,
                           beta_o, phi_susc, seasonal_factor, w_presymp)
    return foi_h + foi_s + foi_w + foi_o


# ═══════════════════════════════════════════════════════════════════════════
# RHS + solver
# ═══════════════════════════════════════════════════════════════════════════
def compute_derivatives_erlang_n(
    state, t, *, n_stages,
    C_home, C_school, C_work, C_other,
    M_home, M_school, M_work, M_other,
    C_home_vac=None, C_school_vac=None, C_work_vac=None, C_other_vac=None,
    pop_15, rho, kappa, sigma, gamma,
    beta_h, beta_w, beta_s, beta_o, phi_susc,
    p_school, p_work,
    VE, annual_coverage, vax_peak_iso_week=42, vax_spread_weeks=4.0,
    seasonality_amp=0.7, seasonality_base=0.0, seasonality_peak_day=105.0,
    seasonality_period=365.0, day_in_season_offset=0.0,
    school_holiday_amp=0.0, school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0, school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0, school_holiday_realloc=0.0,
    policy_school_start_day=-1.0e9, policy_school_end_day=1.0e9,
    policy_work_start_day=-1.0e9, policy_work_end_day=1.0e9,
    policy_ramp_days=3.0, policy_school_baseline=1.0, policy_work_baseline=1.0,
    w_presymp=0.22,
):
    day_in_season = t + day_in_season_offset
    sf = seasonal_factor_cosine(day_in_season, amp=seasonality_amp,
                                 base=seasonality_base,
                                 peak_day=seasonality_peak_day,
                                 period=seasonality_period)
    w_school = policy_window_weight(day_in_season,
        start_day=policy_school_start_day, end_day=policy_school_end_day,
        ramp_days=policy_ramp_days)
    w_work = policy_window_weight(day_in_season,
        start_day=policy_work_start_day, end_day=policy_work_end_day,
        ramp_days=policy_ramp_days)
    p_school = policy_school_baseline - w_school * (policy_school_baseline - p_school)
    p_work = policy_work_baseline - w_work * (policy_work_baseline - p_work)

    if C_home_vac is not None:
        h = vacation_weight(day_in_season,
            start_day=school_holiday_start_day,
            min_start_day=school_holiday_min_start_day,
            min_end_day=school_holiday_min_end_day, end_day=school_holiday_end_day)
        omh = 1.0 - h
        C_home_t = omh*C_home + h*C_home_vac
        C_school_t = omh*C_school + h*C_school_vac
        C_work_t = omh*C_work + h*C_work_vac
        C_other_t = omh*C_other + h*C_other_vac
        p_school_eff = p_school; beta_s_eff = beta_s
    else:
        school_mult = school_calendar_mult(day_in_season,
            holiday_amp=school_holiday_amp,
            holiday_start_day=school_holiday_start_day,
            holiday_min_start_day=school_holiday_min_start_day,
            holiday_min_end_day=school_holiday_min_end_day,
            holiday_end_day=school_holiday_end_day)
        mult_via_p = school_holiday_realloc*school_mult + (1.0-school_holiday_realloc)*1.0
        mult_via_beta = (1.0-school_holiday_realloc)*school_mult + school_holiday_realloc*1.0
        p_school_eff = p_school*mult_via_p
        beta_s_eff = beta_s*mult_via_beta
        C_home_t, C_school_t, C_work_t, C_other_t = C_home, C_school, C_work, C_other

    S = state[0]; V = state[1]; E = state[E_idx(n_stages)]
    I_all = state[I_start(n_stages):I_end(n_stages)]      # (n_stages, 15, n_adm)
    R = state[R_idx(n_stages)]

    foi = compute_foi_n(state, n_stages,
        C_home=C_home_t, C_school=C_school_t, C_work=C_work_t, C_other=C_other_t,
        M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
        pop_15=pop_15, rho=rho, kappa=kappa,
        p_school=p_school_eff, p_work=p_work,
        beta_h=beta_h, beta_w=beta_w, beta_s=beta_s_eff, beta_o=beta_o,
        phi_susc=phi_susc, seasonal_factor=sf, w_presymp=w_presymp)

    v_rate = vax_rate_vector_jax(day_in_season, annual_coverage,
        peak_iso_week=vax_peak_iso_week, spread_weeks=vax_spread_weeks)[:, None]
    breakthrough = (1.0 - VE) * foi
    rate = n_stages * gamma   # n·γ per stage → mean 1/γ preserved

    dS = -foi*S - v_rate*S
    dV = v_rate*S - breakthrough*V
    dE = foi*S + breakthrough*V - sigma*E
    # dI₁ = σE - rate·I₁; dI_k = rate·I_{k-1} - rate·I_k
    dI_all = []
    prev = sigma * E
    for k in range(n_stages):
        dI_k = prev - rate * I_all[k]
        dI_all.append(dI_k)
        prev = rate * I_all[k]
    dR = prev   # rate·I_n
    return jnp.stack([dS, dV, dE, *dI_all, dR], axis=0)


def simulate_jax_erlang_n(initial_state, *, n_stages, t_span=(0.0, 364.0),
                            rtol=1e-4, atol=1e-6, method="Dopri5",
                            max_steps=200_000, discretize_time=False, **kw):
    t_eval = jnp.arange(t_span[0], t_span[1] + 1.0, 1.0)

    def rhs(t, y, args):
        t_used = jnp.floor(t) if discretize_time else t
        return compute_derivatives_erlang_n(y, t_used, n_stages=n_stages, **kw)

    solver = Dopri5() if method == "Dopri5" else Tsit5()
    sol = diffeqsolve(ODETerm(rhs), solver, t0=t_span[0], t1=t_span[1], dt0=0.1,
                      y0=initial_state, saveat=SaveAt(ts=t_eval),
                      stepsize_controller=PIDController(rtol=rtol, atol=atol),
                      max_steps=max_steps, throw=False)
    ok = sol.result == RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.full_like(sol.ys, 1e10))


def daily_new_onset_by_age_erlang_n(states: jnp.ndarray, n_stages: int) -> jnp.ndarray:
    """★ 증상 발현 신규 = d(I₂+..+I_n+R)/dt.
    n=1 인 경우: I₁ 만 존재 → onset = 감염 시점 = d(E+I₁+R)/dt.
    """
    if n_stages == 1:
        # n=1: presymp 분리 없음 → 관측 시점 = 감염 시점 (E 유입)
        E = states[:, E_idx(1)]
        I1 = states[:, I_start(1)]
        R = states[:, R_idx(1)]
        cum = (E + I1 + R).sum(axis=-1)
    else:
        # I₂..I_n + R
        i_start = I_start(n_stages) + 1
        i_end = I_end(n_stages)
        Isum = states[:, i_start:i_end].sum(axis=(1, -1))
        R = states[:, R_idx(n_stages)].sum(axis=-1)
        cum = Isum + R
    return jnp.diff(cum, axis=0)
