"""Erlang I₃ + presymptomatic (I₁) 상대 전파력 w — 격리 신규 모듈.

7-compartment SVE-I₁I₂I₃-R. Compartment 구조는 erlang.py 와 동일.

FOI 변경:
  I₁ = presymptomatic (전파력 w<1, 정책 무관 attend)
  I₂₃ = I₂ + I₃ = symptomatic (전파력 1, 정책 p 적용, 결근 시 home spillover)
  Channel 별:
    work:   β·C·ρ_T·[(w·I₁ + p·I₂₃) rho-weighted] / N  (WORKER_SLICE 만)
    school: β·C·[(w·I₁ + p·I₂₃) STUDENT] / N          (STUDENT_SLICE 만)
    home:   β·C·[w·I₁ + I₂₃·(1 + κ·φ_spill)] / N       (spillover I₂₃ 만)
    other:  β·C·(w·I₁ + I₂₃) / N                       (정책 없음)

관측:
  daily_new_onset = d(I₂+I₃+R).sum()   # 증상 발현 = I₁→I₂ 유입 누적

NGM 정합:
  실효 전파시간 = w·(1/3γ) + 1·(2/3γ) = (w+2)/(3γ) = 0.741/γ  (w=0.22)
  ⇒ β_derived_from_R0 = R0·π / [(w+2)/3 · ρ_ngm_original]
  ⇒ derive_beta_presymp(...) = derive_beta_from_R0_simplex(...) / ((w+2)/3)

기존 erlang.py / foi_jax.py 는 수정하지 않음.
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

# ---- 상수 ----
N_STAGES = 3                                    # I 3-stage Erlang
E_S, E_V, E_E, E_I1, E_I2, E_I3, E_R = 0, 1, 2, 3, 4, 5, 6
W_PRESYMP: float = 0.22                          # I₁ 상대 전파력 (Nature Hlth 2026 중심)


def split_seed_to_erlang(state5: jnp.ndarray) -> jnp.ndarray:
    """(5, 15, n) → (7, 15, n).  I → I₁·I₂·I₃ 각 1/3."""
    S = state5[0]; V = state5[1]; E = state5[2]; I = state5[3]; R = state5[4]
    third = I / N_STAGES
    return jnp.stack([S, V, E, third, third, third, R], axis=0)


def ngm_factor(w: float = W_PRESYMP) -> float:
    """실효 전파시간 factor.  적용: β_effective = β_ngm_inv / ngm_factor(w)."""
    return (w + 2.0) / 3.0


# ═══════════════════════════════════════════════════════════════════════════
# Per-channel FOI (I₁·I₂₃ 분리)
# ═══════════════════════════════════════════════════════════════════════════
def _foi_home_presymp(I1, I23, C_home, pop_15, rho, kappa,
                       p_school, p_work, beta_h, phi_susc,
                       seasonal_factor, w_presymp):
    N = pop_15
    N_safe = jnp.maximum(N, _EPS)
    # φ_spill (n_admdong, 15) — 학생·근로자 결근 비율
    phi_spill = compute_phi_spillover(p_school, p_work, rho)
    spill_factor = 1.0 + kappa[None, :] * phi_spill              # (n, 15)
    # I₁ 은 결근 안 함(spillover 무관), I₂₃ 만 spillover
    I_eff = w_presymp * I1 + I23 * spill_factor.T                # (15, n)
    contact_pressure = C_home @ (I_eff / N_safe)                 # (15, n)
    foi_h = (beta_h * seasonal_factor) * phi_susc[:, None] * contact_pressure
    return jnp.where(N > _EPS, foi_h, 0.0)


def _foi_school_presymp(I1, I23, C_school, pop_15,
                         p_school, beta_s, phi_susc,
                         seasonal_factor, w_presymp):
    N = pop_15
    N_safe = jnp.maximum(N, _EPS)
    n_adm = N.shape[1]
    I_eff = jnp.zeros_like(I1)
    I_eff = I_eff.at[STUDENT_SLICE].set(
        w_presymp * I1[STUDENT_SLICE] + p_school * I23[STUDENT_SLICE]
    )
    contact_pressure = C_school @ (I_eff / N_safe)               # (15, n)
    foi_s_all = (beta_s * seasonal_factor) * phi_susc[:, None] * contact_pressure
    foi_s = jnp.zeros((N_AGE, n_adm), dtype=jnp.float64)
    foi_s = foi_s.at[STUDENT_SLICE].set(foi_s_all[STUDENT_SLICE])
    return jnp.where(N > _EPS, foi_s, 0.0)


def _foi_work_presymp(I1, I23, C_work, pop_15, M_work, rho,
                       p_work, beta_w, phi_susc,
                       seasonal_factor, w_presymp):
    N = pop_15
    n_adm = N.shape[1]
    rho_T = rho.T                                                # (15, n)
    # I_eff already 정책 반영: w·I₁(항상 출근) + p·I₂₃(결근율 반영)
    I_eff = w_presymp * I1 + p_work * I23                        # (15, n)
    weighted_I = rho_T * I_eff
    weighted_N = rho_T * N
    I_at_j = jnp.einsum("akj,ak->aj", M_work, weighted_I)
    N_at_j = jnp.einsum("akj,ak->aj", M_work, weighted_N)
    N_at_j_safe = jnp.maximum(N_at_j, _EPS)
    ratio_at_j = I_at_j / N_at_j_safe                            # p_work 이미 포함
    contact_pressure_at_j = C_work @ ratio_at_j
    pressure_at_i = jnp.einsum("aij,aj->ai", M_work, contact_pressure_at_j)
    foi_w_all = (beta_w * seasonal_factor) * phi_susc[:, None] * rho_T * pressure_at_i
    foi_w = jnp.zeros((N_AGE, n_adm), dtype=jnp.float64)
    foi_w = foi_w.at[WORKER_SLICE].set(foi_w_all[WORKER_SLICE])
    return jnp.where(N > _EPS, foi_w, 0.0)


def _foi_other_presymp(I1, I23, C_other, pop_15, M_other,
                        beta_o, phi_susc, seasonal_factor, w_presymp):
    N = pop_15
    I_eff = w_presymp * I1 + I23
    I_at_j = jnp.einsum("akj,ak->aj", M_other, I_eff)
    N_at_j = jnp.einsum("akj,ak->aj", M_other, N)
    N_at_j_safe = jnp.maximum(N_at_j, _EPS)
    ratio_at_j = I_at_j / N_at_j_safe
    contact_pressure_at_j = C_other @ ratio_at_j
    pressure_at_i = jnp.einsum("aij,aj->ai", M_other, contact_pressure_at_j)
    foi_o = (beta_o * seasonal_factor) * phi_susc[:, None] * pressure_at_i
    return jnp.where(N > _EPS, foi_o, 0.0)


def compute_foi_presymp(state, *,
    C_home, C_school, C_work, C_other,
    M_home, M_school, M_work, M_other,
    pop_15, rho, kappa,
    p_school, p_work,
    beta_h, beta_w, beta_s, beta_o,
    phi_susc, seasonal_factor,
    w_presymp=W_PRESYMP,
):
    I1 = state[E_I1]
    I23 = state[E_I2] + state[E_I3]
    foi_h = _foi_home_presymp(I1, I23, C_home, pop_15, rho, kappa,
                                p_school, p_work, beta_h, phi_susc,
                                seasonal_factor, w_presymp)
    foi_s = _foi_school_presymp(I1, I23, C_school, pop_15,
                                  p_school, beta_s, phi_susc,
                                  seasonal_factor, w_presymp)
    foi_w = _foi_work_presymp(I1, I23, C_work, pop_15, M_work, rho,
                                p_work, beta_w, phi_susc,
                                seasonal_factor, w_presymp)
    foi_o = _foi_other_presymp(I1, I23, C_other, pop_15, M_other,
                                 beta_o, phi_susc,
                                 seasonal_factor, w_presymp)
    return foi_h + foi_s + foi_w + foi_o


# ═══════════════════════════════════════════════════════════════════════════
# RHS + solver
# ═══════════════════════════════════════════════════════════════════════════
def compute_derivatives_erlang_presymp(
    state, t, *,
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
    w_presymp=W_PRESYMP,
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

    S = state[E_S]; V = state[E_V]; E = state[E_E]
    I1 = state[E_I1]; I2 = state[E_I2]; I3 = state[E_I3]; R = state[E_R]

    foi = compute_foi_presymp(state,
        C_home=C_home_t, C_school=C_school_t, C_work=C_work_t, C_other=C_other_t,
        M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
        pop_15=pop_15, rho=rho, kappa=kappa,
        p_school=p_school_eff, p_work=p_work,
        beta_h=beta_h, beta_w=beta_w, beta_s=beta_s_eff, beta_o=beta_o,
        phi_susc=phi_susc, seasonal_factor=sf, w_presymp=w_presymp)

    v_rate = vax_rate_vector_jax(day_in_season, annual_coverage,
        peak_iso_week=vax_peak_iso_week, spread_weeks=vax_spread_weeks)[:, None]
    breakthrough = (1.0 - VE) * foi
    rate = N_STAGES * gamma

    dS = -foi*S - v_rate*S
    dV = v_rate*S - breakthrough*V
    dE = foi*S + breakthrough*V - sigma*E
    dI1 = sigma*E - rate*I1
    dI2 = rate*I1 - rate*I2
    dI3 = rate*I2 - rate*I3
    dR = rate*I3
    return jnp.stack([dS, dV, dE, dI1, dI2, dI3, dR], axis=0)


def simulate_jax_erlang_presymp(initial_state, *, t_span=(0.0, 364.0),
                                  rtol=1e-4, atol=1e-6, method="Dopri5",
                                  max_steps=200_000, discretize_time=False, **kw):
    """Integrate 7-comp presymp ODE. initial_state (7,15,n) via split_seed_to_erlang."""
    t_eval = jnp.arange(t_span[0], t_span[1] + 1.0, 1.0)

    def rhs(t, y, args):
        t_used = jnp.floor(t) if discretize_time else t
        return compute_derivatives_erlang_presymp(y, t_used, **kw)

    solver = Dopri5() if method == "Dopri5" else Tsit5()
    sol = diffeqsolve(ODETerm(rhs), solver, t0=t_span[0], t1=t_span[1], dt0=0.1,
                      y0=initial_state, saveat=SaveAt(ts=t_eval),
                      stepsize_controller=PIDController(rtol=rtol, atol=atol),
                      max_steps=max_steps, throw=False)
    ok = sol.result == RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.full_like(sol.ys, 1e10))


def daily_new_onset_by_age_erlang_presymp(states: jnp.ndarray) -> jnp.ndarray:
    """★ 증상 발현 신규 = d(I₂+I₃+R)/dt.  (n_t,7,15,n) → (n_t-1,15)."""
    cum_onset = (states[:, E_I2] + states[:, E_I3]
                 + states[:, E_R]).sum(axis=-1)                  # (n_t, 15)
    return jnp.diff(cum_onset, axis=0)


def daily_new_infection_by_age_erlang_presymp(states: jnp.ndarray) -> jnp.ndarray:
    """감염 시점 신규 (참조용, R0 계산 검증).  d(E+I₁+I₂+I₃+R)/dt."""
    cum = (states[:, E_E] + states[:, E_I1] + states[:, E_I2]
           + states[:, E_I3] + states[:, E_R]).sum(axis=-1)
    return jnp.diff(cum, axis=0)
