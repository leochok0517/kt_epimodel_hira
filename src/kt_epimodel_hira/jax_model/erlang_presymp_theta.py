"""theta (other-channel differential) on top of erlang_presymp.

main_v5 sec:theta_methods, Eq eq:increment (line 153). Not implemented in
``erlang_presymp.py`` - this module adds it without touching that file.

Only the symptomatic block of the *other*-channel FOI is scaled:

    lambda_other,a = phi_a * s_f(t) * beta_other
                      * sum_b C_other,ab(t) * [w*I1,b + (1 - theta*Delta_b)*I23,b] / N_b

Delta_b (policy-attributable increment, age-band b):
    students : p_base_school - p_school
    workers  : rho_b * (p_base_work - p_work)      (rho = employment rate)
    else     : 0

``compute_phi_spillover(p_school, p_work, rho)`` already returns
``(1-p_school)`` on the student slice and ``rho*(1-p_work)`` on the worker
slice (zero elsewhere), so
``compute_phi_spillover(p, p, rho) - compute_phi_spillover(p_base, p_base, rho)``
is exactly Delta_b - no new parameter, no new data.

Everything except the other-channel FOI and the two functions that have to
call it (the FOI assembly, the RHS, the integrator wrapper) is imported
unchanged from ``erlang_presymp``. Redefining those three is unavoidable -
theta has to be threaded through the call chain - but no other function is
copied, and their bodies are otherwise byte-identical to the source so a
diff against ``erlang_presymp.py`` shows only the theta plumbing.

theta defaults to 0.0, so calling with the default reduces exactly to the
untouched pipeline (see tests/test_theta_equivalence.py, T1).

Does not modify erlang_presymp.py or foi_jax.py.
"""
from __future__ import annotations

import jax.numpy as jnp
from diffrax import (ODETerm, Dopri5, Tsit5, diffeqsolve, SaveAt,
                      PIDController, RESULTS)

from kt_epimodel_hira.jax_model.foi_jax import (
    N_AGE, _EPS,
    seasonal_factor_cosine, school_calendar_mult,
    vacation_weight, policy_window_weight,
    compute_phi_spillover,
)
from kt_epimodel_hira.jax_model.dynamics_jax import vax_rate_vector_jax

# Reused unchanged - do not redefine.
from kt_epimodel_hira.jax_model.erlang_presymp import (
    N_STAGES, E_S, E_V, E_E, E_I1, E_I2, E_I3, E_R, W_PRESYMP,
    split_seed_to_erlang, ngm_factor,
    _foi_home_presymp, _foi_school_presymp, _foi_work_presymp,
    daily_new_onset_by_age_erlang_presymp, daily_new_infection_by_age_erlang_presymp,
)


# ═══════════════════════════════════════════════════════════════════════════
# other-channel FOI with the (1 - theta*Delta_b) symptomatic scaling
# ═══════════════════════════════════════════════════════════════════════════
def _foi_other_presymp_theta(I1, I23, C_other, pop_15, M_other,
                              beta_o, phi_susc, seasonal_factor, w_presymp,
                              theta, p_school, p_work,
                              policy_school_baseline, policy_work_baseline, rho):
    N = pop_15
    delta_b = (compute_phi_spillover(p_school, p_work, rho)
               - compute_phi_spillover(policy_school_baseline, policy_work_baseline, rho))  # (n_admdong, 15)
    scale = 1.0 - theta * delta_b                                 # (n_admdong, 15)
    # w*I1 is untouched (presymptomatic attends regardless of policy);
    # only the symptomatic I23 block is scaled - matches eq:increment.
    I_eff = w_presymp * I1 + I23 * scale.T                        # (15, n_admdong)
    I_at_j = jnp.einsum("akj,ak->aj", M_other, I_eff)
    N_at_j = jnp.einsum("akj,ak->aj", M_other, N)
    N_at_j_safe = jnp.maximum(N_at_j, _EPS)
    ratio_at_j = I_at_j / N_at_j_safe
    contact_pressure_at_j = C_other @ ratio_at_j
    pressure_at_i = jnp.einsum("aij,aj->ai", M_other, contact_pressure_at_j)
    foi_o = (beta_o * seasonal_factor) * phi_susc[:, None] * pressure_at_i
    return jnp.where(N > _EPS, foi_o, 0.0)


def compute_foi_presymp_theta(state, *,
    C_home, C_school, C_work, C_other,
    M_home, M_school, M_work, M_other,
    pop_15, rho, kappa,
    p_school, p_work,
    beta_h, beta_w, beta_s, beta_o,
    phi_susc, seasonal_factor,
    theta, policy_school_baseline, policy_work_baseline,
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
    foi_o = _foi_other_presymp_theta(I1, I23, C_other, pop_15, M_other,
                                       beta_o, phi_susc, seasonal_factor, w_presymp,
                                       theta, p_school, p_work,
                                       policy_school_baseline, policy_work_baseline, rho)
    return foi_h + foi_s + foi_w + foi_o


# ═══════════════════════════════════════════════════════════════════════════
# RHS + solver - byte-identical to erlang_presymp.py except for the theta
# kwarg and calling compute_foi_presymp_theta instead of compute_foi_presymp.
# ═══════════════════════════════════════════════════════════════════════════
def compute_derivatives_erlang_presymp_theta(
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
    theta=0.0,
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

    foi = compute_foi_presymp_theta(state,
        C_home=C_home_t, C_school=C_school_t, C_work=C_work_t, C_other=C_other_t,
        M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
        pop_15=pop_15, rho=rho, kappa=kappa,
        p_school=p_school_eff, p_work=p_work,
        beta_h=beta_h, beta_w=beta_w, beta_s=beta_s_eff, beta_o=beta_o,
        phi_susc=phi_susc, seasonal_factor=sf,
        theta=theta, policy_school_baseline=policy_school_baseline,
        policy_work_baseline=policy_work_baseline,
        w_presymp=w_presymp)

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


def simulate_jax_erlang_presymp_theta(initial_state, *, t_span=(0.0, 364.0),
                                        rtol=1e-4, atol=1e-6, method="Dopri5",
                                        max_steps=200_000, discretize_time=False, **kw):
    """Same integrator wrapper as erlang_presymp.py, calling the theta RHS."""
    t_eval = jnp.arange(t_span[0], t_span[1] + 1.0, 1.0)

    def rhs(t, y, args):
        t_used = jnp.floor(t) if discretize_time else t
        return compute_derivatives_erlang_presymp_theta(y, t_used, **kw)

    solver = Dopri5() if method == "Dopri5" else Tsit5()
    sol = diffeqsolve(ODETerm(rhs), solver, t0=t_span[0], t1=t_span[1], dt0=0.1,
                       y0=initial_state, saveat=SaveAt(ts=t_eval),
                       stepsize_controller=PIDController(rtol=rtol, atol=atol),
                       max_steps=max_steps, throw=False)
    ok = sol.result == RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.full_like(sol.ys, 1e10))
