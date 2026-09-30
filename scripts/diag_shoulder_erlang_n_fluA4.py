"""Diagnostic (prompts/16b): does raising the infectious-stage count fix the decline?

Companion to ``diag_shoulder_fluA4.py``, which tested the *latent* period
(Erlang(2) made no difference) and the timescales. prompts/15 swept the
infectious stage count n, but only as a forward simulation at the existing
posterior's R_peak — it never refitted, so it could not speak to fit quality or
to the shape of the post-peak decline. This does.

Same point-fit setup as prompts/16: pi held at the 4-season shared posterior
median, only log R_peak and log phi_NB free, four seasons.

    V0  n=3, w=0.22    GI 5.14 d   (read from diag_shoulder_fluA4.json, not refitted)
    V4  n=5, w=0.4444  GI 4.60 d
    V5  n=8, w=0.7778  GI 4.30 d

w follows ``erlang_n.w_for_presymp_fraction``, i.e. w = f(n-1)/(1-f) with
f = 0.10, which reproduces prompts/16b's own table (0.44, 0.78). The model code
is ``erlang_n`` used exactly as ``sens6_erlang_n_fluA4.py`` uses it; nothing
existing is modified.

A V0 cross-check is included: n=3 at w=0.22 run through the erlang_n path,
point-fitted the same way, should reproduce the stored erlang_presymp V0.
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

import fluA_setup as FS  # noqa: E402
import nuts_seasonpop as NS  # noqa: E402
from figstyle import savefig, COL_ZERO, W_DOUBLE  # noqa: E402
from kt_epimodel_hira.jax_model.erlang_n import (  # noqa: E402
    daily_new_onset_by_age_erlang_n, ngm_factor, simulate_jax_erlang_n,
    split_seed_to_n, w_for_presymp_fraction,
)
from kt_epimodel_hira.jax_model.loss_jax import (  # noqa: E402
    nb_nll_jax, simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (  # noqa: E402
    derive_beta_from_R0_simplex, make_ngm_eigvalue_fn,
)

OUTD = REPO / "outputs" / "fluA"
SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
ISO_START_WEEK = 36
PRESYMP_FRACTION = 0.10
INV_SIGMA = 2.0
INV_GAMMA = 4.0

PI_SHARED = np.array([0.4224, 0.3143, 0.0730, 0.1894])

VARIANTS = {
    "V4": dict(label="V4 infectious Erlang(5)", n_stages=5,
               w=w_for_presymp_fraction(PRESYMP_FRACTION, 5), color="#009E73"),
    "V5": dict(label="V5 infectious Erlang(8)", n_stages=8,
               w=w_for_presymp_fraction(PRESYMP_FRACTION, 8), color="#CC79A7"),
    # cross-check: the current model re-run through the erlang_n code path
    "V0-check": dict(label="V0 cross-check (n=3, w=0.22, erlang_n path)",
                     n_stages=3, w=float(NS.W), color="#999999"),
}


def implied_gi(n_stages: int, w: float,
               inv_sigma: float = INV_SIGMA, inv_gamma: float = INV_GAMMA) -> float:
    weights = [w] + [1.0] * (n_stages - 1)
    times = [(k + 1) * (inv_gamma / n_stages) for k in range(n_stages)]
    return inv_sigma + sum(a * b for a, b in zip(weights, times)) / sum(weights)


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


def main():
    t0 = time.time()
    pi_fixed = PI_SHARED / PI_SHARED.sum()
    print("[diag shoulder: infectious n] pi fixed at "
          + str(np.round(pi_fixed, 4).tolist()))
    for vk, spec in VARIANTS.items():
        print(f"    {vk}: n={spec['n_stages']} w={spec['w']:.4f}  "
              f"GI={implied_gi(spec['n_stages'], spec['w']):.3f} d  "
              f"ngm_factor={ngm_factor(spec['w'], spec['n_stages']):.4f}  "
              f"presymptomatic window={INV_GAMMA/spec['n_stages']:.3f} d")

    prev = json.loads((OUTD / "diag_shoulder_fluA4.json").read_text())
    v0_prev = prev["results"]["V0"]

    C = FS.build_fluA_setup(SEASONS, verbose=False)
    phi_j = jnp.asarray(NS.PHI)
    gamma_rep = jnp.asarray(NS.GAMMA)

    results = {}
    ngm_checks = {}
    for vk, spec in VARIANTS.items():
        n_stages = spec["n_stages"]; w_val = spec["w"]
        nf = ngm_factor(w_val, n_stages)
        results[vk] = {}
        for si, s in enumerate(SEASONS):
            sh = dict(C["shared_by_s"][s])
            sh["sigma"] = 1.0 / INV_SIGMA
            sh["gamma"] = 1.0 / INV_GAMMA
            # gamma is unchanged from V0, so this is V0's eigenvalue function;
            # rebuilt here anyway so the script does not depend on that fact.
            ngm = make_ngm_eigvalue_fn(
                pop_15=np.asarray(sh["pop_15"]), rho=np.asarray(sh["rho"]),
                C_home=np.asarray(sh["C_home"]), C_work=np.asarray(sh["C_work"]),
                C_school=np.asarray(sh["C_school"]), C_other=np.asarray(sh["C_other"]),
                R0_immunity=np.asarray(NS.IMM), gamma=float(sh["gamma"]),
                seasonal_factor=1.0 + float(NS.F.S.AMP))

            obs = jnp.asarray(C["obs"][si])
            wts = jnp.asarray(C["w"][si])
            nw = C["nw"]
            init = split_seed_to_n(C["st_by_s"][s], n_stages)

            def curve(log_R):
                b0 = derive_beta_from_R0_simplex(
                    ngm, jnp.exp(log_R), jnp.asarray(pi_fixed), phi_j)
                beta = b0 / nf
                kw = dict(sh)
                kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = (
                    beta[0], beta[1], beta[2], beta[3])
                kw["phi_susc"] = phi_j
                kw["kappa"] = jnp.asarray(NS.KAP)
                kw["p_school"] = NS.BASE
                kw["p_work"] = NS.BASE
                st = simulate_jax_erlang_n(init, n_stages=n_stages,
                                            w_presymp=w_val, **kw,
                                            discretize_time=False)
                inc = daily_new_onset_by_age_erlang_n(st, n_stages)
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
                n_stages=n_stages, w=w_val,
                implied_generation_interval_d=implied_gi(n_stages, w_val),
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

    # V0 cross-check against the stored erlang_presymp fit
    cross = {}
    for s in SEASONS:
        a = results["V0-check"][s]; b = v0_prev[s]
        cross[s] = dict(
            R_peak_erlang_n=a["R_peak"], R_peak_stored=b["R_peak"],
            R_peak_rel_diff=abs(a["R_peak"] - b["R_peak"]) / b["R_peak"],
            nll_erlang_n=a["nll"], nll_stored=b["nll"],
            nll_abs_diff=abs(a["nll"] - b["nll"]),
            decline_2wk_erlang_n=a["decline_2wk_model"],
            decline_2wk_stored=b["decline_2wk_model"])
    max_R_diff = max(v["R_peak_rel_diff"] for v in cross.values())
    max_nll_diff = max(v["nll_abs_diff"] for v in cross.values())
    print(f"\n[V0 cross-check] erlang_n(n=3, w=0.22) vs stored erlang_presymp V0: "
          f"max |dR_peak|/R = {max_R_diff:.2e}, max |dNLL| = {max_nll_diff:.3f}")

    max_rel = max(v["rel_err"] for v in ngm_checks.values())
    print(f"[NGM check] max relative error {max_rel:.2e}")

    out = dict(
        purpose=("diagnostic for prompts/16b — does raising the infectious "
                 "Erlang stage count fix the post-peak decline? Not a "
                 "manuscript calculation."),
        meta=dict(
            seasons=SEASONS, pi_fixed=pi_fixed.tolist(),
            pi_source="per_season_pi_fluA4.json shared.pi.*.median (held fixed)",
            free_parameters=["log R_peak", "log phi_nb"],
            inv_sigma=INV_SIGMA, inv_gamma=INV_GAMMA,
            presymp_fraction=PRESYMP_FRACTION,
            w_rule=("erlang_n.w_for_presymp_fraction: w = f*(n-1)/(1-f). "
                    "prompts/16b's parenthetical f*n/(1-f) would give 0.556 / "
                    "0.889 and does not match that prompt's own table."),
            variants={vk: dict(n_stages=spec["n_stages"], w=spec["w"],
                               implied_generation_interval_d=implied_gi(
                                   spec["n_stages"], spec["w"]),
                               ngm_factor=ngm_factor(spec["w"], spec["n_stages"]),
                               presymptomatic_window_d=INV_GAMMA / spec["n_stages"])
                      for vk, spec in VARIANTS.items()},
            v0_source="outputs/fluA/diag_shoulder_fluA4.json (read, not refitted)",
            ngm_check=dict(max_rel_err=max_rel),
        ),
        v0_reference=v0_prev,
        v0_cross_check=cross,
        results=results, ngm_checks=ngm_checks,
        wall_sec=time.time() - t0,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    outp = OUTD / "diag_shoulder_erlang_n_fluA4.json"
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {outp}")

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.0), constrained_layout=True)
    for k, s in enumerate(SEASONS):
        ax = axes[k // 2][k % 2]
        obs_tot = np.array(results["V4"][s]["obs_curve"])
        nw = len(obs_tot)
        ax.plot(np.arange(nw), obs_tot, "o", color=COL_ZERO, ms=2.6, alpha=0.85,
                zorder=5, label="Observed" if k == 0 else None)
        ax.plot(np.arange(nw), v0_prev[s]["model_curve"], color="black", lw=1.4,
                label="V0 current (n=3)" if k == 0 else None)
        for vk in ("V4", "V5"):
            ax.plot(np.arange(nw), results[vk][s]["model_curve"],
                    color=VARIANTS[vk]["color"], lw=1.4,
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
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4,
               frameon=False, fontsize=7.5)
    savefig(fig, "diag_shoulder_erlang_n")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
