"""Erlang(2) latent period on top of erlang_presymp — diagnostic only (prompts/16).

``erlang_presymp.py`` gives the latent period a single exponential compartment
E, which makes the latent distribution maximally dispersed and widens the
epidemic (the point Wearing et al. 2005 make about the infectious period; we
had applied the Erlang treatment only to I). This module splits E into
E1 -> E2, each leaving at rate 2*sigma, so the mean latent period is unchanged
at 1/sigma but its variance halves.

8 compartments: S, V, E1, E2, I1, I2, I3, R.

Nothing in ``erlang_presymp.py`` is modified. The four per-channel FOI
helpers are imported and reused unchanged — they take I1 and I23 as arrays
rather than indexing the state, so they are layout-agnostic. Only the
compartment indices, the seed split, the E block of the RHS and the observation
helpers are new.

Observation is unchanged in meaning: symptom onset is the I1 -> I2 transition,
so ``daily_new_onset_by_age_e2`` is still d(I2 + I3 + R)/dt.

This is a diagnostic variant for deciding whether a v9 refit is worth doing. It
is not part of the published pipeline.
"""
from __future__ import annotations

import jax.numpy as jnp
from diffrax import (ODETerm, Dopri5, Tsit5, diffeqsolve, SaveAt,
                     PIDController, RESULTS)

from kt_epimodel_hira.jax_model.dynamics_jax import vax_rate_vector_jax
from kt_epimodel_hira.jax_model.foi_jax import (
    school_calendar_mult, seasonal_factor_cosine, policy_window_weight,
    vacation_weight,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    N_STAGES, W_PRESYMP, ngm_factor,
    _foi_home_presymp, _foi_school_presymp, _foi_work_presymp,
    _foi_other_presymp,
)

#: 8-compartment layout (E split in two)
E2_S, E2_V, E2_E1, E2_E2, E2_I1, E2_I2, E2_I3, E2_R = 0, 1, 2, 3, 4, 5, 6, 7

N_LATENT_STAGES = 2


def split_seed_to_e2(state5: jnp.ndarray) -> jnp.ndarray:
    """(5, 15, n) -> (8, 15, n). E split evenly over E1/E2, I over I1..I3."""
    S = state5[0]; V = state5[1]; E = state5[2]; I = state5[3]; R = state5[4]
    half = E / N_LATENT_STAGES
    third = I / N_STAGES
    return jnp.stack([S, V, half, half, third, third, third, R], axis=0)


def compute_foi_e2(state, *, C_home, C_school, C_work, C_other,
                   M_home, M_school, M_work, M_other,
                   pop_15, rho, kappa, p_school, p_work,
                   beta_h, beta_w, beta_s, beta_o,
                   phi_susc, seasonal_factor, w_presymp=W_PRESYMP):
    """Same four channels as compute_foi_presymp, new state indices."""
    I1 = state[E2_I1]
    I23 = state[E2_I2] + state[E2_I3]
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
                                beta_o, phi_susc, seasonal_factor, w_presymp)
    return foi_h + foi_s + foi_w + foi_o


def compute_derivatives_e2(
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
    """Identical to compute_derivatives_erlang_presymp except the E block."""
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

    S = state[E2_S]; V = state[E2_V]
    E1 = state[E2_E1]; E2 = state[E2_E2]
    I1 = state[E2_I1]; I2 = state[E2_I2]; I3 = state[E2_I3]

    foi = compute_foi_e2(state,
        C_home=C_home_t, C_school=C_school_t, C_work=C_work_t, C_other=C_other_t,
        M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
        pop_15=pop_15, rho=rho, kappa=kappa,
        p_school=p_school_eff, p_work=p_work,
        beta_h=beta_h, beta_w=beta_w, beta_s=beta_s_eff, beta_o=beta_o,
        phi_susc=phi_susc, seasonal_factor=sf, w_presymp=w_presymp)

    v_rate = vax_rate_vector_jax(day_in_season, annual_coverage,
        peak_iso_week=vax_peak_iso_week, spread_weeks=vax_spread_weeks)[:, None]
    breakthrough = (1.0 - VE) * foi
    rate = N_STAGES * gamma              # infectious stage rate, unchanged
    lat = N_LATENT_STAGES * sigma        # each latent stage leaves at 2*sigma

    dS = -foi*S - v_rate*S
    dV = v_rate*S - breakthrough*V
    dE1 = foi*S + breakthrough*V - lat*E1
    dE2 = lat*E1 - lat*E2
    dI1 = lat*E2 - rate*I1
    dI2 = rate*I1 - rate*I2
    dI3 = rate*I2 - rate*I3
    dR = rate*I3
    return jnp.stack([dS, dV, dE1, dE2, dI1, dI2, dI3, dR], axis=0)


def simulate_jax_e2(initial_state, *, t_span=(0.0, 364.0),
                    rtol=1e-4, atol=1e-6, method="Dopri5",
                    max_steps=200_000, discretize_time=False, **kw):
    """Integrate the 8-compartment ODE. initial_state (8,15,n) via split_seed_to_e2."""
    t_eval = jnp.arange(t_span[0], t_span[1] + 1.0, 1.0)

    def rhs(t, y, args):
        t_used = jnp.floor(t) if discretize_time else t
        return compute_derivatives_e2(y, t_used, **kw)

    solver = Dopri5() if method == "Dopri5" else Tsit5()
    sol = diffeqsolve(ODETerm(rhs), solver, t0=t_span[0], t1=t_span[1], dt0=0.1,
                      y0=initial_state, saveat=SaveAt(ts=t_eval),
                      stepsize_controller=PIDController(rtol=rtol, atol=atol),
                      max_steps=max_steps, throw=False)
    ok = sol.result == RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.full_like(sol.ys, 1e10))


def daily_new_onset_by_age_e2(states: jnp.ndarray) -> jnp.ndarray:
    """Symptom onset = d(I2 + I3 + R)/dt, i.e. the I1 -> I2 transition."""
    cum_onset = (states[:, E2_I2] + states[:, E2_I3]
                 + states[:, E2_R]).sum(axis=-1)
    return jnp.diff(cum_onset, axis=0)


def daily_new_infection_by_age_e2(states: jnp.ndarray) -> jnp.ndarray:
    """Infection time = d(E1 + E2 + I1 + I2 + I3 + R)/dt."""
    cum = (states[:, E2_E1] + states[:, E2_E2] + states[:, E2_I1]
           + states[:, E2_I2] + states[:, E2_I3] + states[:, E2_R]).sum(axis=-1)
    return jnp.diff(cum, axis=0)
