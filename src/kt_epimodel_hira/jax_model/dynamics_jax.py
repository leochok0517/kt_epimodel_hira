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
from jax.scipy.special import erf
from kt_epimodel_hira.jax_model.foi_jax import (
    compute_foi_jax, seasonal_factor_cosine, school_calendar_mult, vacation_weight,
    policy_window_weight,
)

_SEASON_START_ISO_WEEK = 36
_SQRT2 = 1.4142135623730951
_SEASON_LENGTH_DAYS = 365.0


def vax_rate_vector_jax(
    day_in_season: float,
    annual_coverage: jnp.ndarray,
    peak_iso_week: int = 42,
    spread_weeks: float = 4.0,
) -> jnp.ndarray:
    """Vaccination hazard rate v(t), finite-season normalised. Returns (15,).

    ``annual_coverage`` is the target cumulative coverage fraction C per age.
    Under dS = -v·S the vaccinated fraction over the season is
    ``1 - exp(-∫v dt)``, so reaching C requires ``∫₀^365 v dt = -ln(1-C)``
    (hazard-integral / "A-fix").

    The Gaussian campaign profile is truncated to [0, 365]; its integral is
    ``Z = ∫₀^365 density ds < 1`` (~6.7% mass lost in the pre-season left tail
    with the default peak/spread). Dividing by ``Z`` restores full campaign
    mass so cumulative coverage lands exactly on C. Z is analytic (erf) and,
    for fixed peak/spread, a compile-time constant — JIT/grad safe.
    """
    peak_day = (peak_iso_week - _SEASON_START_ISO_WEEK) * 7
    sigma_days = spread_weeks * 7
    z = (day_in_season - peak_day) / sigma_days
    density = jnp.exp(-0.5 * z * z) / (sigma_days * jnp.sqrt(2.0 * jnp.pi))
    Z = 0.5 * (
        erf((_SEASON_LENGTH_DAYS - peak_day) / (sigma_days * _SQRT2))
        - erf((0.0 - peak_day) / (sigma_days * _SQRT2))
    )
    # -ln(1-C), clipped so C→1 stays finite.
    hazard = -jnp.log1p(-jnp.clip(annual_coverage, 0.0, 1.0 - 1e-6))
    return (hazard / Z) * density


def compute_derivatives_jax(
    state: jnp.ndarray,             # (5, 15, n_admdong)
    t: float,
    *,
    # contact + mobility (static once converted)
    C_home: jnp.ndarray, C_school: jnp.ndarray, C_work: jnp.ndarray, C_other: jnp.ndarray,
    M_home: jnp.ndarray, M_school: jnp.ndarray, M_work: jnp.ndarray, M_other: jnp.ndarray,
    # optional vacation contact matrices → enables time-switching C(t).
    # When provided, C(t) = (1-h(t))·C_term + h(t)·C_vac blends per channel and
    # the school_calendar_mult β-scaling path is bypassed (no double-counting).
    C_home_vac: jnp.ndarray | None = None,
    C_school_vac: jnp.ndarray | None = None,
    C_work_vac: jnp.ndarray | None = None,
    C_other_vac: jnp.ndarray | None = None,
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
    # ★ v4 LEGACY: 아래 school_holiday_* 파라미터 블록은 C_*_vac (vacation
    #   matrix) 옵션 도입 이전 경로. C_*_vac 이 제공되면 이 블록은 완전히
    #   우회됨 (아래 if-branch 로 진입 안 함). v4 pipeline (kappa_no_eta_presymp,
    #   nuts_v4, final_v4_figures) 은 C(t) = (1-h)·C_term + h·C_vac 스위칭
    #   경로 사용 → 이 파라미터들은 미사용. 재현성 위해 코드 삭제하지 않음.
    school_holiday_amp: float = 0.0,
    school_holiday_start_day: float = 113.0,
    school_holiday_min_start_day: float = 127.0,
    school_holiday_min_end_day: float = 162.0,
    school_holiday_end_day: float = 183.0,
    # Holiday channel-loss routing. 0 = scale β_s only (school contacts lost),
    # 1 = scale p_school instead (school contacts reallocate to home via the
    # existing spillover κ — no new parameter). Smooth blend in between.
    school_holiday_realloc: float = 0.0,
    # Time-windowed policy — independent windows for the school and work levers.
    # p applies only inside its [start, end]; outside it reverts to the lever's
    # baseline (default 1.0 = no policy). Defaults (−inf,+inf) window + baseline
    # 1.0 → whole-season, backward compatible.
    policy_school_start_day: float = -1.0e9,
    policy_school_end_day: float = 1.0e9,
    policy_work_start_day: float = -1.0e9,
    policy_work_end_day: float = 1.0e9,
    policy_ramp_days: float = 3.0,
    # baseline attendance-if-sick outside the window (e.g. 0.6 presenteeism)
    policy_school_baseline: float = 1.0,
    policy_work_baseline: float = 1.0,
) -> jnp.ndarray:
    """ODE right-hand side. Returns same shape (5, 15, n_admdong)."""
    day_in_season = t + day_in_season_offset
    sf = seasonal_factor_cosine(
        day_in_season,
        amp=seasonality_amp, base=seasonality_base,
        peak_day=seasonality_peak_day, period=seasonality_period,
    )

    # Windowed policy: p_eff(t) = base − w(t)·(base − p). w≡0 → base; w≡1 → p.
    # Default base=1 → p_eff = 1 − w·(1−p) (whole-season with w≡1 gives p).
    w_school = policy_window_weight(
        day_in_season, start_day=policy_school_start_day,
        end_day=policy_school_end_day, ramp_days=policy_ramp_days,
    )
    w_work = policy_window_weight(
        day_in_season, start_day=policy_work_start_day,
        end_day=policy_work_end_day, ramp_days=policy_ramp_days,
    )
    p_school = policy_school_baseline - w_school * (policy_school_baseline - p_school)
    p_work = policy_work_baseline - w_work * (policy_work_baseline - p_work)

    if C_home_vac is not None:
        # Time-switching contacts: blend term↔vacation per channel by h(t).
        # The winter-break attenuation lives entirely in C(t); β_s and p_school
        # are left untouched, so spillover κ·φ_spill fires from POLICY only
        # (φ_spill = 1-p_school / ρ·(1-p_work)) and is vacation-agnostic.
        h = vacation_weight(
            day_in_season,
            start_day=school_holiday_start_day,
            min_start_day=school_holiday_min_start_day,
            min_end_day=school_holiday_min_end_day,
            end_day=school_holiday_end_day,
        )
        one_minus_h = 1.0 - h
        C_home_t = one_minus_h * C_home + h * C_home_vac
        C_school_t = one_minus_h * C_school + h * C_school_vac
        C_work_t = one_minus_h * C_work + h * C_work_vac
        C_other_t = one_minus_h * C_other + h * C_other_vac
        p_school_eff = p_school
        beta_s_eff = beta_s
    else:
        # Legacy path (no vacation matrices): winter break via β_s / p_school
        # scaling. Distribute the break "school-channel loss" between β_s (lost)
        # and p_school (reallocated); realloc=0 destroys it, realloc=1 routes it
        # through the home spillover κ × (1-p_school).
        school_mult = school_calendar_mult(
            day_in_season,
            holiday_amp=school_holiday_amp,
            holiday_start_day=school_holiday_start_day,
            holiday_min_start_day=school_holiday_min_start_day,
            holiday_min_end_day=school_holiday_min_end_day,
            holiday_end_day=school_holiday_end_day,
        )
        mult_via_p = (school_holiday_realloc * school_mult
                      + (1.0 - school_holiday_realloc) * 1.0)
        mult_via_beta = ((1.0 - school_holiday_realloc) * school_mult
                         + school_holiday_realloc * 1.0)
        p_school_eff = p_school * mult_via_p
        beta_s_eff = beta_s * mult_via_beta
        C_home_t, C_school_t, C_work_t, C_other_t = C_home, C_school, C_work, C_other

    foi = compute_foi_jax(
        state,
        C_home=C_home_t, C_school=C_school_t, C_work=C_work_t, C_other=C_other_t,
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
