"""JAX/diffrax integration of SVEIR ODE (M1b).

Mirrors src/kt_epimodel_hira/simulation/solver.py::run_simulation.

Replaces scipy.integrate.solve_ivp (RK45) with diffrax.diffeqsolve (Dopri5).
Dopri5 is the same Dormand-Prince(5) tableau as scipy's RK45 — numerical
agreement to within adaptive-step tolerance.

State carried as 3D pytree (5, 15, n_admdong); no flatten needed.
"""
from __future__ import annotations
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from diffrax import ODETerm, Dopri5, Tsit5, diffeqsolve, SaveAt, PIDController, RESULTS

from kt_epimodel_hira.jax_model.dynamics_jax import compute_derivatives_jax

IDX_S, IDX_V, IDX_E, IDX_I, IDX_R = 0, 1, 2, 3, 4


def simulate_jax(
    initial_state: jnp.ndarray,           # (5, 15, n_admdong)
    *,
    # contact + mobility (static converted)
    C_home, C_school, C_work, C_other,
    M_home, M_school, M_work, M_other,
    # optional vacation contact matrices → time-switching C(t) blend (see
    # compute_derivatives_jax). None → legacy time-invariant contacts.
    C_home_vac=None, C_school_vac=None, C_work_vac=None, C_other_vac=None,
    pop_15, rho, kappa,
    # disease
    sigma: float, gamma: float,
    # calibration
    beta_h: float, beta_w: float, beta_s: float, beta_o: float,
    phi_susc: jnp.ndarray,
    # policy
    p_school: float, p_work: float,
    # vaccination
    VE: float, annual_coverage: jnp.ndarray,
    vax_peak_iso_week: int = 42, vax_spread_weeks: float = 4.0,
    # seasonality (cosine fixed)
    seasonality_amp: float = 0.7, seasonality_base: float = 0.0,
    seasonality_peak_day: float = 105.0, seasonality_period: float = 365.0,
    day_in_season_offset: float = 0.0,
    # school calendar (winter break) — optional, defaults preserve old behaviour
    school_holiday_amp: float = 0.0,
    school_holiday_start_day: float = 113.0,
    school_holiday_min_start_day: float = 127.0,
    school_holiday_min_end_day: float = 162.0,
    school_holiday_end_day: float = 183.0,
    school_holiday_realloc: float = 0.0,
    # time-windowed policy — separate school/work windows (defaults → whole-season)
    policy_school_start_day: float = -1.0e9,
    policy_school_end_day: float = 1.0e9,
    policy_work_start_day: float = -1.0e9,
    policy_work_end_day: float = 1.0e9,
    policy_ramp_days: float = 3.0,
    policy_school_baseline: float = 1.0,
    policy_work_baseline: float = 1.0,
    # integration
    t_span: tuple[float, float] = (0.0, 364.0),
    rtol: float = 1e-4,
    atol: float = 1e-6,
    method: str = "Dopri5",
    max_steps: int = 200_000,
    discretize_time: bool = False,
):
    """Integrate SVEIR ODE. Returns states (n_t, 5, 15, n_admdong).

    Args:
        discretize_time: if True, time argument to RHS is floor(t). Matches
            scipy reference (which uses ``int(t + offset)``) bit-by-bit for
            regression. Default False uses continuous t (smoother ODE; more
            physically correct; better for MCMC).
    """
    t_eval = jnp.arange(t_span[0], t_span[1] + 1.0, 1.0)

    def rhs(t, y, args):
        t_used = jnp.floor(t) if discretize_time else t
        return compute_derivatives_jax(
            y, t_used,
            C_home=C_home, C_school=C_school, C_work=C_work, C_other=C_other,
            M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
            C_home_vac=C_home_vac, C_school_vac=C_school_vac,
            C_work_vac=C_work_vac, C_other_vac=C_other_vac,
            pop_15=pop_15, rho=rho, kappa=kappa,
            sigma=sigma, gamma=gamma,
            beta_h=beta_h, beta_w=beta_w, beta_s=beta_s, beta_o=beta_o,
            phi_susc=phi_susc,
            p_school=p_school, p_work=p_work,
            VE=VE, annual_coverage=annual_coverage,
            vax_peak_iso_week=vax_peak_iso_week, vax_spread_weeks=vax_spread_weeks,
            seasonality_amp=seasonality_amp, seasonality_base=seasonality_base,
            seasonality_peak_day=seasonality_peak_day, seasonality_period=seasonality_period,
            day_in_season_offset=day_in_season_offset,
            school_holiday_amp=school_holiday_amp,
            school_holiday_start_day=school_holiday_start_day,
            school_holiday_min_start_day=school_holiday_min_start_day,
            school_holiday_min_end_day=school_holiday_min_end_day,
            school_holiday_end_day=school_holiday_end_day,
            school_holiday_realloc=school_holiday_realloc,
            policy_school_start_day=policy_school_start_day,
            policy_school_end_day=policy_school_end_day,
            policy_work_start_day=policy_work_start_day,
            policy_work_end_day=policy_work_end_day,
            policy_ramp_days=policy_ramp_days,
            policy_school_baseline=policy_school_baseline,
            policy_work_baseline=policy_work_baseline,
        )

    solver = Dopri5() if method == "Dopri5" else Tsit5()
    term = ODETerm(rhs)
    sol = diffeqsolve(
        term, solver,
        t0=t_span[0], t1=t_span[1], dt0=0.1,
        y0=initial_state,
        saveat=SaveAt(ts=t_eval),
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        throw=False,   # do not raise on failure — needed under NUTS where extreme
                       # parameter probes (esp. reparam v7+ with near-degenerate
                       # simplex) can stiffen the ODE past max_steps.
    )
    # Replace NaN/inf with a finite large sentinel so downstream NLL is finite
    # (large) and HMC rejects cleanly. NaN-filled ys would poison the mass-matrix
    # adaptation through autodiff. Prior tightening (logit_pi σ, etc.) is the
    # primary defence; this is the secondary net.
    ok = sol.result == RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.full_like(sol.ys, 1e10))


def daily_new_infection_by_age_jax(states: jnp.ndarray) -> jnp.ndarray:
    """Mirror of SimulationResult.daily_new_infection_by_age.

    Args:
        states: (n_t, 5, 15, n_admdong)
    Returns:
        (n_t-1, 15)
    """
    E = states[:, IDX_E, :, :].sum(axis=-1)
    I = states[:, IDX_I, :, :].sum(axis=-1)
    R = states[:, IDX_R, :, :].sum(axis=-1)
    return jnp.diff(E + I + R, axis=0)


def daily_new_infection_jax(states: jnp.ndarray) -> jnp.ndarray:
    """Mirror of SimulationResult.daily_new_infection. (n_t-1,)."""
    E = states[:, IDX_E].sum(axis=(1, 2))
    I = states[:, IDX_I].sum(axis=(1, 2))
    R = states[:, IDX_R].sum(axis=(1, 2))
    return jnp.diff(E + I + R)
