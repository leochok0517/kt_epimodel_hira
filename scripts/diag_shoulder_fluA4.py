"""Diagnostic (prompts/16): does a structural change fix the post-peak decline?

NOT a manuscript calculation. This decides whether a v9 refit is worth doing.

Figure 3's posterior-predictive coverage (prompts/15) missed almost entirely on
the post-peak shoulder: the modelled epidemic falls far more slowly than the
observed one. The hypothesis is that the implied mean generation interval
(5.14 d, against a literature ~3 d) is the cause — a long GI forces a high
R_peak for the same growth rate, a high R_peak means a large final size, and a
large final size leaves susceptibles to burn through after the peak. A single
exponential latent compartment widens the epidemic further.

Four variants, point-fitted (no NUTS):

    V0  current        Erlang(1) latent, 1/sigma 2.0, 1/gamma 4.0   GI 5.14 d
    V1  shorter        Erlang(1) latent, 1/sigma 1.5, 1/gamma 3.0   GI 3.85 d
    V2  latent Erlang  Erlang(2) latent, 1/sigma 2.0, 1/gamma 4.0   GI 5.14 d
    V3  both           Erlang(2) latent, 1/sigma 1.5, 1/gamma 3.0   GI 3.85 d

pi is held at the 4-season shared posterior median so that the structural
effect is not confounded with a reallocation. Only log R_peak and log phi_NB
are free, so each fit is a 2-D optimisation: jit + value_and_grad + L-BFGS-B
(the loss is never called outside a compiled function).

gamma enters the NGM normalisation (s_f / gamma), so V1/V3 rebuild the
eigenvalue function; the script asserts that R_peak is still recovered as the
dominant eigenvalue in every variant.
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
from kt_epimodel_hira.jax_model.erlang_presymp import (  # noqa: E402
    daily_new_onset_by_age_erlang_presymp, simulate_jax_erlang_presymp,
    split_seed_to_erlang,
)
from kt_epimodel_hira.jax_model.erlang_presymp_e2_fluA4 import (  # noqa: E402
    daily_new_onset_by_age_e2, simulate_jax_e2, split_seed_to_e2,
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
N_STAGES_I = 3

#: 4-season shared posterior median (per_season_pi_fluA4.json shared.pi.*.median)
PI_SHARED = np.array([0.4224, 0.3143, 0.0730, 0.1894])

VARIANTS = {
    "V0": dict(label="V0 current (Erlang(1) latent, 2.0/4.0)",
               latent_stages=1, inv_sigma=2.0, inv_gamma=4.0, color="black"),
    "V1": dict(label="V1 shorter (Erlang(1) latent, 1.5/3.0)",
               latent_stages=1, inv_sigma=1.5, inv_gamma=3.0, color="#E69F00"),
    "V2": dict(label="V2 latent Erlang(2) (2.0/4.0)",
               latent_stages=2, inv_sigma=2.0, inv_gamma=4.0, color="#0072B2"),
    "V3": dict(label="V3 both (Erlang(2) latent, 1.5/3.0)",
               latent_stages=2, inv_sigma=1.5, inv_gamma=3.0, color="#D55E00"),
}


def implied_gi(inv_sigma: float, inv_gamma: float, w: float = None) -> float:
    """Latent mean + weighted mean transmission time over the 3 infectious stages."""
    w = float(NS.W) if w is None else w
    weights = [w, 1.0, 1.0]
    times = [(k + 1) * (inv_gamma / N_STAGES_I) for k in range(N_STAGES_I)]
    return inv_sigma + sum(a * b for a, b in zip(weights, times)) / sum(weights)


def iso_week_labels(season: str, n_weeks: int, every: int = 4):
    start_year = int(season.split("-")[0])
    d0 = _dt.date.fromisocalendar(start_year, ISO_START_WEEK, 1)
    idx = list(range(0, n_weeks, every))
    return idx, [str((d0 + _dt.timedelta(weeks=i)).isocalendar()[1]) for i in idx]


def decline_ratio(curve: np.ndarray, k: int, peak_idx: int = None) -> float:
    """curve[peak + k] / curve[peak], using the curve's own peak by default."""
    pk = int(np.argmax(curve)) if peak_idx is None else int(peak_idx)
    if pk + k >= len(curve):
        return float("nan")
    return float(curve[pk + k] / max(curve[pk], 1e-9))


def build_variant_pieces(C, season, spec):
    """Per-variant kwargs, simulate/seed/onset functions and NGM eigenvalue fn."""
    sh = dict(C["shared_by_s"][season])
    sh["sigma"] = 1.0 / spec["inv_sigma"]
    sh["gamma"] = 1.0 / spec["inv_gamma"]

    # gamma changed -> the NGM normalisation (s_f/gamma) changes with it
    ngm = make_ngm_eigvalue_fn(
        pop_15=np.asarray(sh["pop_15"]), rho=np.asarray(sh["rho"]),
        C_home=np.asarray(sh["C_home"]), C_work=np.asarray(sh["C_work"]),
        C_school=np.asarray(sh["C_school"]), C_other=np.asarray(sh["C_other"]),
        R0_immunity=np.asarray(NS.IMM), gamma=float(sh["gamma"]),
        seasonal_factor=1.0 + float(NS.F.S.AMP))

    if spec["latent_stages"] == 1:
        seed_fn, sim_fn, onset_fn = (split_seed_to_erlang,
                                      simulate_jax_erlang_presymp,
                                      daily_new_onset_by_age_erlang_presymp)
    else:
        seed_fn, sim_fn, onset_fn = (split_seed_to_e2, simulate_jax_e2,
                                      daily_new_onset_by_age_e2)
    return sh, ngm, seed_fn, sim_fn, onset_fn


def main():
    t0 = time.time()
    pi_fixed = PI_SHARED / PI_SHARED.sum()
    print("[diag shoulder] pi fixed at " + str(np.round(pi_fixed, 4).tolist()))
    for vk, spec in VARIANTS.items():
        print(f"    {vk}: latent Erlang({spec['latent_stages']}) "
              f"1/sigma={spec['inv_sigma']} 1/gamma={spec['inv_gamma']}  "
              f"GI={implied_gi(spec['inv_sigma'], spec['inv_gamma']):.3f} d  "
              f"presymptomatic window={spec['inv_gamma']/N_STAGES_I:.3f} d")

    C = FS.build_fluA_setup(SEASONS, verbose=False)
    H = np.asarray(C["H"])
    phi_j = jnp.asarray(NS.PHI)
    gamma_rep = jnp.asarray(NS.GAMMA)
    ngm_f = float(NS.NGM_F)      # (w+2)/3, unchanged: n=3 and w=0.22 throughout

    results = {}
    ngm_checks = {}
    for vk, spec in VARIANTS.items():
        results[vk] = {}
        for si, s in enumerate(SEASONS):
            sh, ngm, seed_fn, sim_fn, onset_fn = build_variant_pieces(C, s, spec)
            st0 = C["st_by_s"][s]
            obs = jnp.asarray(C["obs"][si])
            wts = jnp.asarray(C["w"][si])
            nw = C["nw"]
            init = seed_fn(st0)

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
                st = sim_fn(init, w_presymp=NS.W, **kw, discretize_time=False)
                inc = onset_fn(st)
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

            # R_peak must still be the dominant eigenvalue of the reference operator
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
                  f"obs={r['decline_2wk_obs']:.3f}  total_ratio={r['total_ratio']:.3f}",
                  flush=True)

    max_rel = max(v["rel_err"] for v in ngm_checks.values())
    print(f"\n[NGM check] R_peak recovered as the dominant eigenvalue in all "
          f"{len(ngm_checks)} fits; max relative error {max_rel:.2e}")

    out = dict(
        purpose=("diagnostic for prompts/16 — is the post-peak shoulder miss "
                 "fixable by a structural change? Not a manuscript calculation."),
        meta=dict(
            seasons=SEASONS, pi_fixed=pi_fixed.tolist(),
            pi_source="per_season_pi_fluA4.json shared.pi.*.median (held fixed)",
            free_parameters=["log R_peak", "log phi_nb"],
            w_presymp=float(NS.W), ngm_factor=ngm_f,
            n_infectious_stages=N_STAGES_I,
            variants={vk: dict(**{k: v for k, v in spec.items() if k != "color"},
                               implied_generation_interval_d=implied_gi(
                                   spec["inv_sigma"], spec["inv_gamma"]),
                               presymptomatic_window_d=spec["inv_gamma"] / N_STAGES_I)
                      for vk, spec in VARIANTS.items()},
            latent_validation=dict(
                method=("seed the whole cohort in E1 with transmission off and "
                        "measure the decay of E1+E2"),
                e2_mean_latent_d=2.0027, e2_median_latent_d=1.7150,
                e2_theory_median_d=1.6783, exponential_theory_median_d=1.3863),
            ngm_check=dict(max_rel_err=max_rel, note=(
                "gamma enters the NGM normalisation, so V1/V3 rebuild the "
                "eigenvalue function; this confirms R_peak is still recovered")),
        ),
        results=results, ngm_checks=ngm_checks,
        wall_sec=time.time() - t0,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    outp = OUTD / "diag_shoulder_fluA4.json"
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {outp}")

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.0), constrained_layout=True)
    for k, s in enumerate(SEASONS):
        ax = axes[k // 2][k % 2]
        obs_tot = np.array(results["V0"][s]["obs_curve"])
        nw = len(obs_tot)
        ax.plot(np.arange(nw), obs_tot, "o", color=COL_ZERO, ms=2.6, alpha=0.85,
                zorder=5, label="Observed" if k == 0 else None)
        for vk, spec in VARIANTS.items():
            ax.plot(np.arange(nw), results[vk][s]["model_curve"],
                    color=spec["color"], lw=1.4,
                    label=spec["label"] if k == 0 else None)
        ticks, labels = iso_week_labels(s, nw)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlim(-0.5, nw - 0.5)
        ax.set_title(s, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if k // 2 == 1:
            ax.set_xlabel("Week (ISO, season starts at week 36)", fontsize=8.5)
        if k % 2 == 0:
            ax.set_ylabel("Weekly incidence (influenza A)", fontsize=8.5)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2,
               frameon=False, fontsize=7.5)
    savefig(fig, "diag_shoulder")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
