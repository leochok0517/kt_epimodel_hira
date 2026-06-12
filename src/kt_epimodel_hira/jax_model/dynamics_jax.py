"""JAX rewrite of SVEIR dynamics (M1a).

dS/dt = -lambda*S - v(t)*S
dV/dt =  v(t)*S - (1-VE)*lambda*V
dE/dt =  lambda*S + (1-VE)*lambda*V - sigma*E
dI/dt =  sigma*E - gamma*I
dR/dt =  gamma*I

Vaccination v(t) is a Gaussian over day-in-season (precomputed externally
or computed inline). Seasonal factor: cosine (hardcoded; HIRA fixed defaults).
"""
from __future__ import annotations
import jax.numpy as jnp
from kt_epimodel_hira.jax_model.foi_jax import (
    compute_foi_jax, seasonal_factor_cosine, school_calendar_mult,
)

_SEASON_START_ISO_WEEK = 36


def vax_rate_vector_jax(
    day_in_season: float,
    annual_coverage: jnp.ndarray,
    peak_iso_week: int = 42,
    spread_weeks: float = 4.0,
) -> jnp.ndarray:
    """Gaussian density × annual coverage. Returns (15,)."""
    peak_day = (peak_iso_week - _SEASON_START_ISO_WEEK) * 7
    sigma_days = spread_weeks * 7
    z = (day_in_season - peak_day) / sigma_days
    density = jnp.exp(-0.5 * z * z) / (sigma_days * jnp.sqrt(2.0 * jnp.pi))
    return annual_coverage * density


def compute_derivatives_jax(
    state: jnp.ndarray,             # (5, 15, n_admdong)
    t: float,
    *,
    # contact + mobility (static once converted)
    C_home: jnp.ndarray, C_school: jnp.ndarray, C_work: jnp.ndarray, C_other: jnp.ndarray,
    M_home: jnp.ndarray, M_school: jnp.ndarray, M_work: jnp.ndarray, M_other: jnp.ndarray,
    pop_15: jnp.ndarray,
    rho: jnp.ndarray,
    # disease
    kappa: jnp.ndarray, sigma: float, gamma: float,
    # calibration
    beta_h: float, beta_w: float, beta_s: float, beta_o: float,
    phi_susc: jnp.ndarray,
    # policy
    p_school: float, p_work: float,
    # vaccination
    VE: float,
    annual_coverage: jnp.ndarray,
    vax_peak_iso_week: int = 42, vax_spread_weeks: float = 4.0,
    # seasonality (cosine)
    seasonality_amp: float = 0.7, seasonality_base: float = 0.0,
    seasonality_peak_day: float = 105.0, seasonality_period: float = 365.0,
    day_in_season_offset: float = 0.0,
    # school calendar (winter break) — default amp=0 keeps backward compat
    school_holiday_amp: float = 0.0,
    school_holiday_start_day: float = 113.0,
    school_holiday_min_start_day: float = 127.0,
    school_holiday_min_end_day: float = 162.0,
    school_holiday_end_day: float = 183.0,
    # Holiday channel-loss routing. 0 = scale β_s only (school contacts lost),
    # 1 = scale p_school instead (school contacts reallocate to home via the
    # existing spillover κ — no new parameter). Smooth blend in between.
    school_holiday_realloc: float = 0.0,
) -> jnp.ndarray:
    """ODE right-hand side. Returns same shape (5, 15, n_admdong)."""
    day_in_season = t + day_in_season_offset
    sf = seasonal_factor_cosine(
        day_in_season,
        amp=seasonality_amp, base=seasonality_base,
        peak_day=seasonality_peak_day, period=seasonality_period,
    )

    school_mult = school_calendar_mult(
        day_in_season,
        holiday_amp=school_holiday_amp,
        holiday_start_day=school_holiday_start_day,
        holiday_min_start_day=school_holiday_min_start_day,
        holiday_min_end_day=school_holiday_min_end_day,
        holiday_end_day=school_holiday_end_day,
    )
    # Distribute the break "school-channel loss" between β_s (lost) and
    # p_school (reallocated). At realloc=0 the loss is destroyed (β_s × mult).
    # At realloc=1 the loss is routed through p_school, which triggers the
    # existing home spillover (κ × (1-p_school)) — same machinery as policy
    # school closure, no new κ.
    mult_via_p = (school_holiday_realloc * school_mult
                  + (1.0 - school_holiday_realloc) * 1.0)
    mult_via_beta = ((1.0 - school_holiday_realloc) * school_mult
                     + school_holiday_realloc * 1.0)
    p_school_eff = p_school * mult_via_p
    beta_s_eff = beta_s * mult_via_beta

    foi = compute_foi_jax(
        state,
        C_home=C_home, C_school=C_school, C_work=C_work, C_other=C_other,
        M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
        pop_15=pop_15, rho=rho, kappa=kappa,
        p_school=p_school_eff, p_work=p_work,
        beta_h=beta_h, beta_w=beta_w, beta_s=beta_s_eff, beta_o=beta_o,
        phi_susc=phi_susc, seasonal_factor=sf,
    )  # (15, n)

    S = state[0]; V = state[1]; E = state[2]; I = state[3]
    v_rate = vax_rate_vector_jax(
        day_in_season, annual_coverage,
        peak_iso_week=vax_peak_iso_week, spread_weeks=vax_spread_weeks,
    )[:, None]  # (15, 1)
    breakthrough = (1.0 - VE) * foi  # (15, n)

    dS = -foi * S - v_rate * S
    dV = v_rate * S - breakthrough * V
    dE = foi * S + breakthrough * V - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    return jnp.stack([dS, dV, dE, dI, dR], axis=0)
