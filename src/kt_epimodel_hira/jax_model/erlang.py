"""Erlang(3) infectious compartment: I → I₁ → I₂ → I₃ → R.

Single-I gives an exponential infectious period (long tail) → wide generation
interval → over-wide epidemics (width_ratio 0.62–0.87 vs data). Splitting I into
3 stages (each rate 3γ) turns the infectious period into Erlang(3, 3γ): same mean
1/γ, variance ÷3 → sharper generation interval → sharper epidemic (Wearing 2005).

Mean infectious period is preserved (3 × 1/(3γ) = 1/γ), so R0 = NGM(β, 1/γ) is
UNCHANGED — the existing make_ngm_eigvalue_fn is reused verbatim (verified).

7-compartment state: [S, V, E, I₁, I₂, I₃, R]. FOI reuses compute_foi_jax by
passing a pseudo-5 state whose index-3 slot holds I_total = I₁+I₂+I₃ (FOI only
reads that slot). Incidence = d(E+I₁+I₂+I₃+R). All seasonality / C(t) / windowed
policy / v(t) logic is identical to the single-I path (dynamics_jax).
"""
from __future__ import annotations
import jax.numpy as jnp
from diffrax import ODETerm, Dopri5, Tsit5, diffeqsolve, SaveAt, PIDController, RESULTS
from kt_epimodel_hira.jax_model.foi_jax import (
    compute_foi_jax, seasonal_factor_cosine, school_calendar_mult,
    vacation_weight, policy_window_weight,
)
from kt_epimodel_hira.jax_model.dynamics_jax import vax_rate_vector_jax

N_STAGES = 3
# 7-comp indices
E_S, E_V, E_E, E_I1, E_I2, E_I3, E_R = 0, 1, 2, 3, 4, 5, 6


def split_seed_to_erlang(state5: jnp.ndarray) -> jnp.ndarray:
    """(5,15,n) [S,V,E,I,R] → (7,15,n) [S,V,E,I/3,I/3,I/3,R]. I_total preserved."""
    S, V, E, I, R = state5[0], state5[1], state5[2], state5[3], state5[4]
    third = I / N_STAGES
    return jnp.stack([S, V, E, third, third, third, R], axis=0)


def compute_derivatives_erlang(
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
):
    """RHS of the 7-compartment SVE-I₁I₂I₃-R ODE. Same physics as single-I."""
    day_in_season = t + day_in_season_offset
    sf = seasonal_factor_cosine(day_in_season, amp=seasonality_amp, base=seasonality_base,
                                peak_day=seasonality_peak_day, period=seasonality_period)
    w_school = policy_window_weight(day_in_season, start_day=policy_school_start_day,
                                    end_day=policy_school_end_day, ramp_days=policy_ramp_days)
    w_work = policy_window_weight(day_in_season, start_day=policy_work_start_day,
                                  end_day=policy_work_end_day, ramp_days=policy_ramp_days)
    p_school = policy_school_baseline - w_school * (policy_school_baseline - p_school)
    p_work = policy_work_baseline - w_work * (policy_work_baseline - p_work)

    if C_home_vac is not None:
        h = vacation_weight(day_in_season, start_day=school_holiday_start_day,
                            min_start_day=school_holiday_min_start_day,
                            min_end_day=school_holiday_min_end_day, end_day=school_holiday_end_day)
        omh = 1.0 - h
        C_home_t = omh*C_home + h*C_home_vac; C_school_t = omh*C_school + h*C_school_vac
        C_work_t = omh*C_work + h*C_work_vac; C_other_t = omh*C_other + h*C_other_vac
        p_school_eff = p_school; beta_s_eff = beta_s
    else:
        school_mult = school_calendar_mult(day_in_season, holiday_amp=school_holiday_amp,
            holiday_start_day=school_holiday_start_day, holiday_min_start_day=school_holiday_min_start_day,
            holiday_min_end_day=school_holiday_min_end_day, holiday_end_day=school_holiday_end_day)
        mult_via_p = school_holiday_realloc*school_mult + (1.0-school_holiday_realloc)*1.0
        mult_via_beta = (1.0-school_holiday_realloc)*school_mult + school_holiday_realloc*1.0
        p_school_eff = p_school*mult_via_p; beta_s_eff = beta_s*mult_via_beta
        C_home_t, C_school_t, C_work_t, C_other_t = C_home, C_school, C_work, C_other

    S = state[E_S]; V = state[E_V]; E = state[E_E]
    I1 = state[E_I1]; I2 = state[E_I2]; I3 = state[E_I3]; R = state[E_R]
    I_total = I1 + I2 + I3
    # pseudo-5 state so compute_foi_jax reads I_total from its index-3 slot
    pseudo = jnp.stack([S, V, E, I_total, R], axis=0)
    foi = compute_foi_jax(
        pseudo, C_home=C_home_t, C_school=C_school_t, C_work=C_work_t, C_other=C_other_t,
        M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
        pop_15=pop_15, rho=rho, kappa=kappa, p_school=p_school_eff, p_work=p_work,
        beta_h=beta_h, beta_w=beta_w, beta_s=beta_s_eff, beta_o=beta_o,
        phi_susc=phi_susc, seasonal_factor=sf)  # (15, n)

    v_rate = vax_rate_vector_jax(day_in_season, annual_coverage,
                                 peak_iso_week=vax_peak_iso_week, spread_weeks=vax_spread_weeks)[:, None]
    breakthrough = (1.0 - VE) * foi
    rate = N_STAGES * gamma   # 3γ per stage → mean infectious period 1/γ preserved

    dS = -foi*S - v_rate*S
    dV = v_rate*S - breakthrough*V
    dE = foi*S + breakthrough*V - sigma*E
    dI1 = sigma*E - rate*I1
    dI2 = rate*I1 - rate*I2
    dI3 = rate*I2 - rate*I3
    dR = rate*I3
    return jnp.stack([dS, dV, dE, dI1, dI2, dI3, dR], axis=0)


def simulate_jax_erlang(initial_state, *, t_span=(0.0, 364.0), rtol=1e-4, atol=1e-6,
                        method="Dopri5", max_steps=200_000, discretize_time=False, **kw):
    """Integrate the 7-comp Erlang ODE. `initial_state` is (7,15,n) (use
    split_seed_to_erlang). All physics kwargs forwarded to the RHS. Returns
    (n_t, 7, 15, n)."""
    t_eval = jnp.arange(t_span[0], t_span[1] + 1.0, 1.0)

    def rhs(t, y, args):
        t_used = jnp.floor(t) if discretize_time else t
        return compute_derivatives_erlang(y, t_used, **kw)

    solver = Dopri5() if method == "Dopri5" else Tsit5()
    sol = diffeqsolve(ODETerm(rhs), solver, t0=t_span[0], t1=t_span[1], dt0=0.1,
                      y0=initial_state, saveat=SaveAt(ts=t_eval),
                      stepsize_controller=PIDController(rtol=rtol, atol=atol),
                      max_steps=max_steps, throw=False)
    ok = sol.result == RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.full_like(sol.ys, 1e10))


def daily_new_infection_by_age_erlang(states: jnp.ndarray) -> jnp.ndarray:
    """(n_t,7,15,n) → (n_t-1,15) daily new infections = d(E+I₁+I₂+I₃+R)."""
    cum = (states[:, E_E] + states[:, E_I1] + states[:, E_I2]
           + states[:, E_I3] + states[:, E_R]).sum(axis=-1)   # (n_t, 15)
    return jnp.diff(cum, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# E₂I₃ variant: E → E₁ → E₂ → I₁ → I₂ → I₃ → R  (E Erlang(2) + I Erlang(3))
# Latent period Erlang(2, 2σ): mean 1/σ preserved, variance ÷2 → sharper
# generation interval → sharper epidemic. R0 unchanged (mean periods preserved).
# 8-compartment state: [S, V, E₁, E₂, I₁, I₂, I₃, R].
#
# ★ v4 LEGACY (실험용): v4 pipeline 은 basic Erlang I₃ (E 1-stage) 를 사용
#   → 아래 E2I3 함수들(simulate_jax_erlang_E2I3, split_seed_to_erlang_E2I3,
#   daily_new_infection_by_age_erlang_E2I3, compute_derivatives_erlang_E2I3)
#   은 미사용. 재현성 위해 코드 유지.
# ═══════════════════════════════════════════════════════════════════════════
N_E_STAGES = 2
E8_S, E8_V, E8_E1, E8_E2, E8_I1, E8_I2, E8_I3, E8_R = 0, 1, 2, 3, 4, 5, 6, 7


def split_seed_to_erlang_E2I3(state5: jnp.ndarray) -> jnp.ndarray:
    """(5,15,n) [S,V,E,I,R] → (8,15,n) [S,V,E/2,E/2,I/3,I/3,I/3,R]. Totals preserved."""
    S, V, E, I, R = state5[0], state5[1], state5[2], state5[3], state5[4]
    eh = E / N_E_STAGES; it = I / N_STAGES
    return jnp.stack([S, V, eh, eh, it, it, it, R], axis=0)


def compute_derivatives_erlang_E2I3(
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
):
    """RHS of the 8-compartment SV-E₁E₂-I₁I₂I₃-R ODE. Same physics; E Erlang(2)."""
    day_in_season = t + day_in_season_offset
    sf = seasonal_factor_cosine(day_in_season, amp=seasonality_amp, base=seasonality_base,
                                peak_day=seasonality_peak_day, period=seasonality_period)
    w_school = policy_window_weight(day_in_season, start_day=policy_school_start_day,
                                    end_day=policy_school_end_day, ramp_days=policy_ramp_days)
    w_work = policy_window_weight(day_in_season, start_day=policy_work_start_day,
                                  end_day=policy_work_end_day, ramp_days=policy_ramp_days)
    p_school = policy_school_baseline - w_school * (policy_school_baseline - p_school)
    p_work = policy_work_baseline - w_work * (policy_work_baseline - p_work)

    if C_home_vac is not None:
        h = vacation_weight(day_in_season, start_day=school_holiday_start_day,
                            min_start_day=school_holiday_min_start_day,
                            min_end_day=school_holiday_min_end_day, end_day=school_holiday_end_day)
        omh = 1.0 - h
        C_home_t = omh*C_home + h*C_home_vac; C_school_t = omh*C_school + h*C_school_vac
        C_work_t = omh*C_work + h*C_work_vac; C_other_t = omh*C_other + h*C_other_vac
        p_school_eff = p_school; beta_s_eff = beta_s
    else:
        school_mult = school_calendar_mult(day_in_season, holiday_amp=school_holiday_amp,
            holiday_start_day=school_holiday_start_day, holiday_min_start_day=school_holiday_min_start_day,
            holiday_min_end_day=school_holiday_min_end_day, holiday_end_day=school_holiday_end_day)
        mult_via_p = school_holiday_realloc*school_mult + (1.0-school_holiday_realloc)*1.0
        mult_via_beta = (1.0-school_holiday_realloc)*school_mult + school_holiday_realloc*1.0
        p_school_eff = p_school*mult_via_p; beta_s_eff = beta_s*mult_via_beta
        C_home_t, C_school_t, C_work_t, C_other_t = C_home, C_school, C_work, C_other

    S = state[E8_S]; V = state[E8_V]; E1 = state[E8_E1]; E2 = state[E8_E2]
    I1 = state[E8_I1]; I2 = state[E8_I2]; I3 = state[E8_I3]; R = state[E8_R]
    I_total = I1 + I2 + I3
    pseudo = jnp.stack([S, V, E1 + E2, I_total, R], axis=0)   # FOI reads slot-3 = I_total
    foi = compute_foi_jax(
        pseudo, C_home=C_home_t, C_school=C_school_t, C_work=C_work_t, C_other=C_other_t,
        M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
        pop_15=pop_15, rho=rho, kappa=kappa, p_school=p_school_eff, p_work=p_work,
        beta_h=beta_h, beta_w=beta_w, beta_s=beta_s_eff, beta_o=beta_o,
        phi_susc=phi_susc, seasonal_factor=sf)

    v_rate = vax_rate_vector_jax(day_in_season, annual_coverage,
                                 peak_iso_week=vax_peak_iso_week, spread_weeks=vax_spread_weeks)[:, None]
    breakthrough = (1.0 - VE) * foi
    e_rate = N_E_STAGES * sigma   # 2σ per latent stage → mean 1/σ preserved
    i_rate = N_STAGES * gamma     # 3γ per infectious stage → mean 1/γ preserved

    dS = -foi*S - v_rate*S
    dV = v_rate*S - breakthrough*V
    dE1 = foi*S + breakthrough*V - e_rate*E1
    dE2 = e_rate*E1 - e_rate*E2
    dI1 = e_rate*E2 - i_rate*I1
    dI2 = i_rate*I1 - i_rate*I2
    dI3 = i_rate*I2 - i_rate*I3
    dR = i_rate*I3
    return jnp.stack([dS, dV, dE1, dE2, dI1, dI2, dI3, dR], axis=0)


def simulate_jax_erlang_E2I3(initial_state, *, t_span=(0.0, 364.0), rtol=1e-4, atol=1e-6,
                             method="Dopri5", max_steps=200_000, discretize_time=False, **kw):
    """Integrate the 8-comp E₂I₃ ODE. initial_state (8,15,n) via split_seed_to_erlang_E2I3."""
    t_eval = jnp.arange(t_span[0], t_span[1] + 1.0, 1.0)

    def rhs(t, y, args):
        t_used = jnp.floor(t) if discretize_time else t
        return compute_derivatives_erlang_E2I3(y, t_used, **kw)

    solver = Dopri5() if method == "Dopri5" else Tsit5()
    sol = diffeqsolve(ODETerm(rhs), solver, t0=t_span[0], t1=t_span[1], dt0=0.1,
                      y0=initial_state, saveat=SaveAt(ts=t_eval),
                      stepsize_controller=PIDController(rtol=rtol, atol=atol),
                      max_steps=max_steps, throw=False)
    ok = sol.result == RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.full_like(sol.ys, 1e10))


def daily_new_infection_by_age_erlang_E2I3(states: jnp.ndarray) -> jnp.ndarray:
    """(n_t,8,15,n) → (n_t-1,15) new infections = d(E₁+E₂+I₁+I₂+I₃+R)."""
    cum = (states[:, E8_E1] + states[:, E8_E2] + states[:, E8_I1]
           + states[:, E8_I2] + states[:, E8_I3] + states[:, E8_R]).sum(axis=-1)
    return jnp.diff(cum, axis=0)
