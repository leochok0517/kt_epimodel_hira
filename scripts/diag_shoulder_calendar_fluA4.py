"""Diagnostic (prompts/16c): is the post-peak decline a calendar problem?

prompts/16 and 16b tested four structural levers (shorter timescales, Erlang
latent period, infectious stage count, and the combination) and none moved the
decline. The remaining candidate is the winter-break calendar.

In 2016-17 (season day 0 = 2016-09-05) the observed collapse runs 12/26-1/8,
i.e. days 112-125 — exactly the model's trapezoid ramp (start 113, full 127),
over which modelled school contact is only about half reduced.

Variants (point fit as in prompts/16: pi at the shared median, only log R_peak
and log phi_NB free, four seasons; latent/infectious timescales and n at their
defaults):

    V0  113 / 127 / 162 / 183   current
    V6  106 / 120 / 155 / 176   whole calendar 7 days earlier
    V7  106 / 107 / 162 / 183   7 days earlier + near-instant transition
    V8   99 / 100 / 162 / 183   14 days earlier + near-instant transition
    V9  current calendar + all four channels cut 30% over days 112-125

V6-V8 are kwargs only. V9 is not a calendar hypothesis at all — it asks whether
the year-end drop is broader than school. Its 30% is arbitrary and **cannot be
quoted in the manuscript**; it exists only to discriminate.

V9 is implemented by wrapping the unmodified
``compute_derivatives_erlang_presymp`` in a local RHS that scales the four beta
values by a time window, and integrating that. No model file is modified.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from diffrax import (ODETerm, Dopri5, diffeqsolve, SaveAt,  # noqa: E402
                     PIDController, RESULTS)

import fluA_setup as FS  # noqa: E402
import nuts_seasonpop as NS  # noqa: E402
from figstyle import savefig, COL_ZERO, W_DOUBLE  # noqa: E402
from kt_epimodel_hira.jax_model.erlang_presymp import (  # noqa: E402
    compute_derivatives_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    simulate_jax_erlang_presymp, split_seed_to_erlang,
)
from kt_epimodel_hira.jax_model.foi_jax import policy_window_weight  # noqa: E402
from kt_epimodel_hira.jax_model.loss_jax import (  # noqa: E402
    nb_nll_jax, simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (  # noqa: E402
    derive_beta_from_R0_simplex, make_ngm_eigvalue_fn,
)

OUTD = REPO / "outputs" / "fluA"
SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
ISO_START_WEEK = 36
PI_SHARED = np.array([0.4224, 0.3143, 0.0730, 0.1894])

#: days 112-125 in 2016-17 = 26 Dec - 8 Jan, the observed collapse window
V9_WINDOW = (112.0, 125.0)
V9_FACTOR = 0.70

VARIANTS = {
    "V6": dict(label="V6 calendar 7 d earlier",
               holiday=dict(start_day=106.0, min_start_day=120.0,
                            min_end_day=155.0, end_day=176.0),
               allcut=False, color="#009E73"),
    "V7": dict(label="V7 7 d earlier + instant switch",
               holiday=dict(start_day=106.0, min_start_day=107.0,
                            min_end_day=162.0, end_day=183.0),
               allcut=False, color="#E69F00"),
    "V8": dict(label="V8 14 d earlier + instant switch",
               holiday=dict(start_day=99.0, min_start_day=100.0,
                            min_end_day=162.0, end_day=183.0),
               allcut=False, color="#CC79A7"),
    "V9": dict(label="V9 current calendar + 30% all-channel cut, d112-125",
               holiday=dict(start_day=113.0, min_start_day=127.0,
                            min_end_day=162.0, end_day=183.0),
               allcut=True, color="#0072B2"),
    "V0-check": dict(label="V0 cross-check (current calendar)",
                     holiday=dict(start_day=113.0, min_start_day=127.0,
                                  min_end_day=162.0, end_day=183.0),
                     allcut=False, color="#999999"),
}


def iso_week_labels(season: str, n_weeks: int, every: int = 4):
    start_year = int(season.split("-")[0])
    d0 = _dt.date.fromisocalendar(start_year, ISO_START_WEEK, 1)
    idx = list(range(0, n_weeks, every))
    return idx, [str((d0 + _dt.timedelta(weeks=i)).isocalendar()[1]) for i in idx]


def decline_ratio(curve: np.ndarray, k: int, peak_idx: int = None) -> float:
    pk = int(np.argmax(curve)) if peak_idx is None else int(peak_idx)
    if pk + k >= len(curve):
        return float("nan")
    return float(curve[pk + k] / max(curve[pk], 1e-9))


def simulate_with_allcut(initial_state, *, t_span=(0.0, 364.0), rtol=1e-4,
                         atol=1e-6, max_steps=200_000, **kw):
    """Integrate the unmodified RHS with the four betas scaled by a window.

    ``compute_derivatives_erlang_presymp`` takes beta as keyword arguments, so
    the time dependence can be applied in this closure without touching the
    model file. A 1-day linear ramp on each edge keeps the step controller from
    tripping on a discontinuity.
    """
    base = {k: kw.pop(k) for k in ("beta_h", "beta_w", "beta_s", "beta_o")}
    t_eval = jnp.arange(t_span[0], t_span[1] + 1.0, 1.0)

    def rhs(t, y, args):
        m = 1.0 - (1.0 - V9_FACTOR) * policy_window_weight(
            t, start_day=V9_WINDOW[0], end_day=V9_WINDOW[1], ramp_days=1.0)
        return compute_derivatives_erlang_presymp(
            y, t, beta_h=base["beta_h"] * m, beta_w=base["beta_w"] * m,
            beta_s=base["beta_s"] * m, beta_o=base["beta_o"] * m, **kw)

    sol = diffeqsolve(ODETerm(rhs), Dopri5(), t0=t_span[0], t1=t_span[1], dt0=0.1,
                      y0=initial_state, saveat=SaveAt(ts=t_eval),
                      stepsize_controller=PIDController(rtol=rtol, atol=atol),
                      max_steps=max_steps, throw=False)
    ok = sol.result == RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.full_like(sol.ys, 1e10))


def main():
    t0 = time.time()
    pi_fixed = PI_SHARED / PI_SHARED.sum()
    print("[diag shoulder: calendar] pi fixed at "
          + str(np.round(pi_fixed, 4).tolist()))
    for vk, spec in VARIANTS.items():
        h = spec["holiday"]
        print(f"    {vk}: holiday {h['start_day']:.0f}/{h['min_start_day']:.0f}/"
              f"{h['min_end_day']:.0f}/{h['end_day']:.0f}"
              + ("  + 30% all-channel cut d112-125" if spec["allcut"] else ""))

    prev = json.loads((OUTD / "diag_shoulder_fluA4.json").read_text())
    v0_prev = prev["results"]["V0"]

    C = FS.build_fluA_setup(SEASONS, verbose=False)
    phi_j = jnp.asarray(NS.PHI)
    gamma_rep = jnp.asarray(NS.GAMMA)
    ngm_f = float(NS.NGM_F)

    results = {}
    ngm_checks = {}
    for vk, spec in VARIANTS.items():
        results[vk] = {}
        for si, s in enumerate(SEASONS):
            sh = dict(C["shared_by_s"][s])
            ngm = make_ngm_eigvalue_fn(
                pop_15=np.asarray(sh["pop_15"]), rho=np.asarray(sh["rho"]),
                C_home=np.asarray(sh["C_home"]), C_work=np.asarray(sh["C_work"]),
                C_school=np.asarray(sh["C_school"]), C_other=np.asarray(sh["C_other"]),
                R0_immunity=np.asarray(NS.IMM), gamma=float(sh["gamma"]),
                seasonal_factor=1.0 + float(NS.F.S.AMP))

            obs = jnp.asarray(C["obs"][si])
            wts = jnp.asarray(C["w"][si])
            nw = C["nw"]
            init = split_seed_to_erlang(C["st_by_s"][s])
            allcut = spec["allcut"]
            hol = spec["holiday"]

            def curve(log_R):
                b0 = derive_beta_from_R0_simplex(
                    ngm, jnp.exp(log_R), jnp.asarray(pi_fixed), phi_j)
                beta = b0 / ngm_f
                kw = dict(sh)
                kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = (
                    beta[0], beta[1], beta[2], beta[3])
                kw["phi_susc"] = phi_j
                kw["kappa"] = jnp.asarray(NS.KAP)
                kw["p_school"] = NS.BASE
                kw["p_work"] = NS.BASE
                kw["school_holiday_start_day"] = hol["start_day"]
                kw["school_holiday_min_start_day"] = hol["min_start_day"]
                kw["school_holiday_min_end_day"] = hol["min_end_day"]
                kw["school_holiday_end_day"] = hol["end_day"]
                if allcut:
                    st = simulate_with_allcut(init, w_presymp=NS.W, **kw)
                else:
                    st = simulate_jax_erlang_presymp(init, w_presymp=NS.W, **kw,
                                                      discretize_time=False)
                inc = daily_new_onset_by_age_erlang_presymp(st)
                return simulation_to_hira_by_age_jax(inc, gamma_rep, n_weeks=nw)

            def loss(x):
                pred = curve(x[0])
                return nb_nll_jax(obs, pred, wts, concentration=jnp.exp(x[1]),
                                  min_rate=0.01)

            vg = jax.jit(jax.value_and_grad(loss))

            def fg(x):
                v, g = vg(jnp.asarray(x))
                v = float(v); g = np.asarray(g, dtype=float)
                if not np.isfinite(v):
                    return 1e15, np.where(np.isfinite(g), g, 0.0)
                return v, g

            best = None
            for r0 in (2.2, 1.6, 2.8):
                for p0 in (1.38, 0.8):
                    r = minimize(fg, np.array([np.log(r0), np.log(p0)]), jac=True,
                                 method="L-BFGS-B",
                                 bounds=[(np.log(1.05), np.log(5.0)),
                                         (np.log(0.05), np.log(50.0))],
                                 options=dict(maxiter=400, ftol=1e-12, gtol=1e-8))
                    if best is None or r.fun < best.fun:
                        best = r
            R_hat = float(np.exp(best.x[0])); phi_hat = float(np.exp(best.x[1]))

            b0 = np.asarray(derive_beta_from_R0_simplex(
                ngm, jnp.asarray(R_hat), jnp.asarray(pi_fixed), phi_j))
            eig = float(ngm(*[jnp.asarray(x) for x in b0], phi_j))
            ngm_checks[f"{vk}|{s}"] = dict(R_peak=R_hat, eigenvalue=eig,
                                            rel_err=abs(eig - R_hat) / R_hat)

            pred = np.asarray(curve(jnp.asarray(np.log(R_hat))))
            mod_tot = pred.sum(axis=1)
            obs_tot = np.asarray(obs).sum(axis=1)
            mask = np.asarray(wts)[:, 0] > 0
            pk_o = int(np.argmax(obs_tot)); pk_m = int(np.argmax(mod_tot))

            results[vk][s] = dict(
                holiday=hol, allcut=allcut,
                R_peak=R_hat, phi_nb=phi_hat, nll=float(best.fun),
                week_obs=pk_o, week_model=pk_m,
                peak_obs=float(obs_tot[pk_o]), peak_model=float(mod_tot[pk_m]),
                peak_ratio=float(obs_tot[pk_o] / max(mod_tot[pk_m], 1.0)),
                total_ratio=float(obs_tot[mask].sum() / max(mod_tot[mask].sum(), 1.0)),
                model_over_obs_total=float(mod_tot[mask].sum() / max(obs_tot[mask].sum(), 1.0)),
                decline_2wk_obs=decline_ratio(obs_tot, 2),
                decline_4wk_obs=decline_ratio(obs_tot, 4),
                decline_2wk_model=decline_ratio(mod_tot, 2),
                decline_4wk_model=decline_ratio(mod_tot, 4),
                decline_2wk_model_at_obs_peak=decline_ratio(mod_tot, 2, pk_o),
                decline_4wk_model_at_obs_peak=decline_ratio(mod_tot, 4, pk_o),
                model_curve=mod_tot.tolist(), obs_curve=obs_tot.tolist(),
            )
            r = results[vk][s]
            print(f"  [{vk}][{s}] R_peak={R_hat:.3f} phi={phi_hat:.3f} "
                  f"NLL={best.fun:,.1f}  decline2wk model={r['decline_2wk_model']:.3f} "
                  f"obs={r['decline_2wk_obs']:.3f}", flush=True)

    cross = {}
    for s in SEASONS:
        a = results["V0-check"][s]; b = v0_prev[s]
        cross[s] = dict(R_peak_new=a["R_peak"], R_peak_stored=b["R_peak"],
                        R_peak_rel_diff=abs(a["R_peak"] - b["R_peak"]) / b["R_peak"],
                        nll_abs_diff=abs(a["nll"] - b["nll"]),
                        decline_2wk_new=a["decline_2wk_model"],
                        decline_2wk_stored=b["decline_2wk_model"])
    print(f"\n[V0 cross-check] max |dR_peak|/R = "
          f"{max(v['R_peak_rel_diff'] for v in cross.values()):.2e}, "
          f"max |dNLL| = {max(v['nll_abs_diff'] for v in cross.values()):.4f}")
    max_rel = max(v["rel_err"] for v in ngm_checks.values())
    print(f"[NGM check] max relative error {max_rel:.2e}")

    out = dict(
        purpose=("diagnostic for prompts/16c — does the winter-break calendar's "
                 "timing/speed explain the post-peak decline? Not a manuscript "
                 "calculation. V9's 30% is arbitrary and must not be quoted."),
        meta=dict(
            seasons=SEASONS, pi_fixed=pi_fixed.tolist(),
            free_parameters=["log R_peak", "log phi_nb"],
            v9_window=list(V9_WINDOW), v9_factor=V9_FACTOR,
            v9_note=("all four betas scaled by 0.70 over days 112-125 with 1-day "
                     "ramps, applied by wrapping the unmodified RHS; the value is "
                     "arbitrary and diagnostic only"),
            season_day_zero="ISO week 36 Monday of the first year",
            variants={vk: dict(holiday=spec["holiday"], allcut=spec["allcut"])
                      for vk, spec in VARIANTS.items()},
            v0_source="outputs/fluA/diag_shoulder_fluA4.json (read, not refitted)",
            ngm_check=dict(max_rel_err=max_rel),
        ),
        v0_reference=v0_prev, v0_cross_check=cross,
        results=results, ngm_checks=ngm_checks,
        wall_sec=time.time() - t0,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    outp = OUTD / "diag_shoulder_calendar_fluA4.json"
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {outp}")

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.0), constrained_layout=True)
    for k, s in enumerate(SEASONS):
        ax = axes[k // 2][k % 2]
        obs_tot = np.array(results["V6"][s]["obs_curve"])
        nw = len(obs_tot)
        ax.plot(np.arange(nw), obs_tot, "o", color=COL_ZERO, ms=2.6, alpha=0.85,
                zorder=5, label="Observed" if k == 0 else None)
        ax.plot(np.arange(nw), v0_prev[s]["model_curve"], color="black", lw=1.4,
                label="V0 current" if k == 0 else None)
        for vk in ("V6", "V7", "V8", "V9"):
            ax.plot(np.arange(nw), results[vk][s]["model_curve"],
                    color=VARIANTS[vk]["color"], lw=1.3,
                    label=VARIANTS[vk]["label"] if k == 0 else None)
        ticks, labels = iso_week_labels(s, nw)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlim(-0.5, nw - 0.5)
        ax.set_title(s, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if k // 2 == 1:
            ax.set_xlabel("Week (ISO, season starts at week 36)", fontsize=8.5)
        if k % 2 == 0:
            ax.set_ylabel("Weekly incidence (influenza A)", fontsize=8.5)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2,
               frameon=False, fontsize=7.5)
    savefig(fig, "diag_shoulder_calendar")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
