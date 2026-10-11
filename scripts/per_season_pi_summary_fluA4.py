"""Summary JSON + figure for the per-season pi fits (prompts/13 §3).

Reads the per-season posteriors written by ``nuts_per_season_pi_fluA4.py`` and
the existing 4-season joint posterior (``outputs/eda/nuts_fluA4_raw.npz`` — read,
never refitted), and writes:

  outputs/fluA/per_season_pi_fluA4.json
  figures/paper_fluA4/{pdf,png}/per_season_pi.{pdf,png}

Summaries are median with q05/q95 (the manuscript's 90% CI convention). The
estimated reproduction number is named ``R_peak``, never ``R_0`` — R_0 is the
derived quantity (see outputs/fluA/reproduction_numbers_fluA4.json).

Fit metrics per season, all at that season's own posterior median unless
stated:

``nll_own_pi``
    joint-loss NLL with the season's own pi / R_peak / phi_nb.
``nll_shared_pi``
    same, with pi swapped for the shared (4-season joint) median pi. The
    difference isolates the cost of sharing the allocation, holding everything
    else at the per-season value.
``peak_ratio`` / ``total_ratio``
    obs/model, computed exactly as ``_fit_quality_fluA.py`` does (unmasked max
    for the peak, the fitted-week mask for the total), reported under both the
    own and the shared allocation. ``_fit_quality_fluA.py`` uses the posterior
    *mean*; the per-season values here are at the median per prompts/13 §3.1,
    and the mean-based variants are included alongside so they stay comparable
    to fit_quality_fluA4.json.

Runs in a couple of minutes: 2 forward simulations per season, no sampling.
"""
from __future__ import annotations

import argparse
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
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import fluA_setup as FS  # noqa: E402
import nuts_fluA as FA  # noqa: E402
import nuts_seasonpop as NS  # noqa: E402
from nuts_per_season_pi_fluA4 import SEASONS_ALL, slice_setup  # noqa: E402
from figstyle import savefig, W_DOUBLE, COL_ZERO  # noqa: E402
from kt_epimodel_hira.jax_model.erlang_presymp import (  # noqa: E402
    daily_new_onset_by_age_erlang_presymp, simulate_jax_erlang_presymp,
    split_seed_to_erlang,
)
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax  # noqa: E402
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex  # noqa: E402

ED = REPO / "outputs" / "eda"
FLUA_DIR = REPO / "outputs" / "fluA"
JOINT_NPZ = ED / "nuts_fluA4_raw.npz"
JOINT_SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
#: Secondary reference only. The 5-season joint fit is the one shared-allocation
#: run that *includes* 2015-16, so it supplies a shared pi whose R_peak was
#: co-fitted for that season — the fair baseline for "does a season-specific pi
#: fix 2015-16?". The manuscript's shared allocation remains the 4-season fit.
JOINT5_NPZ = ED / "nuts_fluA_raw.npz"
CHANNELS = ["home", "work", "school", "other"]

#: Okabe-Ito; 2018-19 is absent from figstyle.COL_SEASON (prompts/11 §공통).
COL_SEASON_5 = {"2015-2016": "#666666", "2016-2017": "#009E73",
                 "2017-2018": "#CC79A7", "2018-2019": "#D55E00",
                 "2019-2020": "#56B4E9"}
EXCLUDED = "2015-2016"

#: Set in main() from the 4-season joint posterior; every cross-arm NLL is
#: evaluated here as well as at each fit's own phi.
COMMON_PHI = float("nan")


def refit_R_peak(loss, pi_vec: np.ndarray, phi: float, R_init: float) -> tuple[float, float]:
    """Minimize the NLL over log R_peak alone, with pi and phi held fixed.

    One free scalar, so no sampling is needed — but the loss must never be
    called outside a compiled function (an eager loop over the ODE is what ate
    54 GB on 2026-08-28), hence jit + value_and_grad + L-BFGS-B, the same
    pattern as ``final_pipeline_confirmed.fit_season_pi``.

    Returns ``(R_peak_hat, nll_at_R_peak_hat)``.
    """
    logit_pi = jnp.asarray(np.log(pi_vec))
    phi_j = jnp.asarray(phi)

    def f(log_R_scalar):
        return loss(jnp.reshape(log_R_scalar, (1,)), logit_pi, phi_j)

    vg = jax.jit(jax.value_and_grad(lambda x: f(x[0])))

    def fg(x):
        v, g = vg(jnp.asarray(x))
        v = float(v); g = np.asarray(g, dtype=float)
        if not np.isfinite(v):
            return 1e15, np.where(np.isfinite(g), g, 0.0)
        return v, g

    best = None
    for start in (np.log(R_init), np.log(max(R_init * 0.8, 1.25)),
                   np.log(min(R_init * 1.2, 3.8))):
        r = minimize(fg, np.array([start]), jac=True, method="L-BFGS-B",
                     bounds=[(np.log(1.2), np.log(4.0))],
                     options=dict(maxiter=200, ftol=1e-12, gtol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(best.fun)


def refit_R_phi(loss, pi_vec: np.ndarray, R_init: float,
                 phi_init: float) -> tuple[float, float, float]:
    """Minimize the NLL over (log R_peak, log phi) with pi held fixed.

    Fixing phi at any single value tilts a pi-vs-pi comparison, because pi and
    phi were co-fitted: the allocation from a fit at phi=0.78 is not optimal
    when scored at phi=1.38, and vice versa. Profiling both nuisance parameters
    out removes that tilt, so the remaining difference between two allocations
    is attributable to the allocation alone.

    Returns ``(R_peak_hat, phi_hat, nll_hat)``.
    """
    logit_pi = jnp.asarray(np.log(pi_vec))

    def f(x):
        return loss(jnp.reshape(x[0], (1,)), logit_pi, jnp.exp(x[1]))

    vg = jax.jit(jax.value_and_grad(f))

    def fg(x):
        v, g = vg(jnp.asarray(x))
        v = float(v); g = np.asarray(g, dtype=float)
        if not np.isfinite(v):
            return 1e15, np.where(np.isfinite(g), g, 0.0)
        return v, g

    best = None
    for r0 in (R_init, max(R_init * 0.85, 1.3), min(R_init * 1.15, 3.7)):
        for p0 in (phi_init, 1.38):
            r = minimize(fg, np.array([np.log(r0), np.log(p0)]), jac=True,
                         method="L-BFGS-B",
                         bounds=[(np.log(1.2), np.log(4.0)),
                                 (np.log(0.05), np.log(50.0))],
                         options=dict(maxiter=400, ftol=1e-12, gtol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
    return (float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(best.fun))


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return dict(median=float(np.median(x)), mean=float(x.mean()),
                q05=float(np.quantile(x, 0.05)), q95=float(np.quantile(x, 0.95)),
                sd=float(x.std()))


def forward_ratios(C_s: dict, season: str, pi: np.ndarray, R_peak: float) -> dict:
    """obs/model peak and total ratios, following _fit_quality_fluA.py."""
    phi_j = jnp.asarray(NS.PHI)
    gamma_j = jnp.asarray(NS.GAMMA)
    beta = derive_beta_from_R0_simplex(
        C_s["ngm3_by_s"][season], jnp.asarray(R_peak), jnp.asarray(pi), phi_j
    ) / NS.NGM_F
    kw = dict(C_s["shared_by_s"][season])
    kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = (
        beta[0], beta[1], beta[2], beta[3])
    kw["phi_susc"] = phi_j
    kw["kappa"] = jnp.asarray(NS.KAP)
    kw["p_school"] = NS.BASE
    kw["p_work"] = NS.BASE
    st = simulate_jax_erlang_presymp(
        split_seed_to_erlang(C_s["st_by_s"][season]), w_presymp=NS.W, **kw,
        discretize_time=False)
    pred = np.asarray(simulation_to_hira_by_age_jax(
        daily_new_onset_by_age_erlang_presymp(st), gamma_j, n_weeks=C_s["nw"]))
    obs = np.asarray(C_s["obs"][0])
    mask = np.asarray(C_s["w"][0])[:, 0] > 0
    o1, p1 = obs.sum(1), pred.sum(1)
    return dict(total_ratio=float(o1[mask].sum() / max(p1[mask].sum(), 1.0)),
                peak_obs=float(o1.max()), peak_model=float(p1.max()),
                peak_ratio=float(o1.max() / max(p1.max(), 1.0)),
                week_obs=int(o1.argmax()), week_model=int(p1.argmax()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=None)
    ap.add_argument("--figure-only", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    # ---- shared (4-season joint) posterior: read only, never refit ----------
    dj = np.load(JOINT_NPZ, allow_pickle=True)
    joint_seasons = [str(s) for s in dj["seasons"]]
    if joint_seasons != JOINT_SEASONS:
        raise SystemExit(f"{JOINT_NPZ} holds {joint_seasons}, expected {JOINT_SEASONS}")
    pi_shared_draws = np.asarray(dj["pi"])
    shared = dict(
        source=str(JOINT_NPZ.relative_to(REPO)), seasons=joint_seasons,
        pi={c: summarize(pi_shared_draws[:, i]) for i, c in enumerate(CHANNELS)},
        phi_nb=summarize(np.asarray(dj["phi_nb"])),
        R_peak={s: summarize(np.exp(np.asarray(dj["log_R0"])[:, j]))
                for j, s in enumerate(joint_seasons)},
    )
    pi_shared_median = np.array([shared["pi"][c]["median"] for c in CHANNELS])
    # softmax-derived draws need not sum to 1 after taking marginal medians
    pi_shared_median = pi_shared_median / pi_shared_median.sum()
    # Common phi for every cross-arm NLL comparison: the 4-season joint median
    # (the allocation the manuscript reports). Read from the posterior, not
    # hardcoded.
    global COMMON_PHI
    COMMON_PHI = float(shared["phi_nb"]["median"])
    print(f"[common phi for NLL comparisons] {COMMON_PHI:.4f} "
          f"(4-season joint median)")

    # ---- secondary reference: the 5-season joint fit (includes 2015-16) -----
    shared5 = None
    pi_shared5_median = None
    R_shared5 = {}
    if JOINT5_NPZ.exists():
        d5 = np.load(JOINT5_NPZ, allow_pickle=True)
        s5_seasons = [str(s) for s in d5["seasons"]]
        pi5 = np.asarray(d5["pi"])
        pi_shared5_median = np.array([float(np.median(pi5[:, i])) for i in range(4)])
        pi_shared5_median = pi_shared5_median / pi_shared5_median.sum()
        R_shared5 = {s: float(np.median(np.exp(np.asarray(d5["log_R0"])[:, j])))
                     for j, s in enumerate(s5_seasons)}
        shared5 = dict(
            source=str(JOINT5_NPZ.relative_to(REPO)), seasons=s5_seasons,
            role=("secondary reference only — the shared-allocation run that "
                  "includes 2015-16, so its R_peak for that season is co-fitted "
                  "with the shared pi. The manuscript's shared allocation is the "
                  "4-season fit above."),
            pi={c: summarize(pi5[:, i]) for i, c in enumerate(CHANNELS)},
            phi_nb=summarize(np.asarray(d5["phi_nb"])),
            R_peak={s: summarize(np.exp(np.asarray(d5["log_R0"])[:, j]))
                    for j, s in enumerate(s5_seasons)},
        )

    # ---- per-season posteriors ---------------------------------------------
    seasons = args.seasons if args.seasons else list(SEASONS_ALL)
    available, missing = [], []
    for s in seasons:
        (available if (ED / f"nuts_per_season_{s}_raw.npz").exists() else missing).append(s)
    if not available:
        raise SystemExit("no per-season posteriors found — run "
                          "scripts/nuts_per_season_pi_fluA4.py first")
    print(f"[per-season posteriors] {len(available)} found: {available}")
    if missing:
        print(f"[per-season posteriors] MISSING (not yet fitted): {missing}")

    diag_all = {}
    partial = FLUA_DIR / "per_season_pi_partial.json"
    if partial.exists():
        diag_all = json.loads(partial.read_text())

    C = FS.build_fluA_setup(available, verbose=False)
    per_season: dict[str, dict] = {}
    for s in available:
        d = np.load(ED / f"nuts_per_season_{s}_raw.npz", allow_pickle=True)
        pi_draws = np.asarray(d["pi"])
        R_draws = np.exp(np.asarray(d["log_R0"])[:, 0])
        phi_draws = np.asarray(d["phi_nb"])
        pi_med = np.array([float(np.median(pi_draws[:, i])) for i in range(4)])
        pi_med = pi_med / pi_med.sum()
        pi_mean = pi_draws.mean(0)
        R_med = float(np.median(R_draws))
        R_mean = float(R_draws.mean())
        phi_med = float(np.median(phi_draws))

        C_s = slice_setup(C, s)
        loss = jax.jit(FA.make_joint_loss_fluA(C_s))

        def nll(pi_vec, R, phi):
            # logit_pi enters the loss through softmax, so pass log(pi):
            # softmax(log pi) == pi for any positive pi summing to 1.
            return float(loss(jnp.asarray([np.log(R)]),
                               jnp.asarray(np.log(pi_vec)), jnp.asarray(phi)))

        nll_own = nll(pi_med, R_med, phi_med)
        nll_shared = nll(pi_shared_median, R_med, phi_med)
        # phi is a fitted parameter and ranges 0.78-2.41 across seasons, so an
        # NLL at each season's own phi is not comparable across arms. Every arm
        # is therefore repeated at the common phi (4-season joint median).
        nll_own_cp = nll(pi_med, R_med, COMMON_PHI)
        nll_shared_cp = nll(pi_shared_median, R_med, COMMON_PHI)
        # Swapping pi moves the optimal R_peak too, so holding R at the own-pi
        # value overstates the cost of sharing. Refit R_peak (1-D) under each
        # allocation at the common phi for the like-for-like number.
        R_own_refit, nll_own_refit = refit_R_peak(loss, pi_med, COMMON_PHI, R_med)
        R_shared_refit, nll_shared_refit = refit_R_peak(
            loss, pi_shared_median, COMMON_PHI, R_med)
        # Both nuisance parameters profiled out — the tilt-free comparison.
        R_own_rp, phi_own_rp, nll_own_rp = refit_R_phi(loss, pi_med, R_med, phi_med)
        R_sh_rp, phi_sh_rp, nll_sh_rp = refit_R_phi(
            loss, pi_shared_median, R_med, phi_med)

        entry = dict(
            n_draws=int(pi_draws.shape[0]),
            excluded_from_main_fit=bool(s == EXCLUDED),
            **{f"pi_{c}": summarize(pi_draws[:, i]) for i, c in enumerate(CHANNELS)},
            R_peak=summarize(R_draws),
            phi_nb=summarize(phi_draws),
            fit=dict(
                nll_own_pi=nll_own,
                nll_shared_pi=nll_shared,
                nll_penalty_of_sharing=nll_shared - nll_own,
                phi_used_own=phi_med,
                # --- common-phi arms (the comparable ones) -------------------
                common_phi=COMMON_PHI,
                nll_own_pi_at_common_phi=nll_own_cp,
                nll_shared_pi_at_common_phi=nll_shared_cp,
                nll_penalty_of_sharing_at_common_phi=nll_shared_cp - nll_own_cp,
                # --- R_peak refitted under each allocation, common phi ------
                at_own_pi_refit_R=dict(
                    R_peak=R_own_refit, nll=nll_own_refit,
                    **forward_ratios(C_s, s, pi_med, R_own_refit)),
                at_shared_pi_refit_R=dict(
                    R_peak=R_shared_refit, nll=nll_shared_refit,
                    **forward_ratios(C_s, s, pi_shared_median, R_shared_refit)),
                penalty_of_sharing_refit_R_at_common_phi=(
                    nll_shared_refit - nll_own_refit),
                # --- R_peak AND phi profiled out (primary) -------------------
                at_own_pi_refit_R_phi=dict(
                    R_peak=R_own_rp, phi_nb=phi_own_rp, nll=nll_own_rp,
                    **forward_ratios(C_s, s, pi_med, R_own_rp)),
                at_shared_pi_refit_R_phi=dict(
                    R_peak=R_sh_rp, phi_nb=phi_sh_rp, nll=nll_sh_rp,
                    **forward_ratios(C_s, s, pi_shared_median, R_sh_rp)),
                penalty_of_sharing_refit_R_phi=nll_sh_rp - nll_own_rp,
                # --- fixed-R forward diagnostics -----------------------------
                at_median_own_pi=forward_ratios(C_s, s, pi_med, R_med),
                at_median_shared_pi=forward_ratios(C_s, s, pi_shared_median, R_med),
                at_mean_own_pi=forward_ratios(C_s, s, pi_mean / pi_mean.sum(), R_mean),
            ),
        )
        # Shared allocation with its own co-fitted R_peak (5-season joint fit).
        # This is the comparison that answers "does a season-specific pi fix
        # 2015-16?" without confounding pi against a mismatched R_peak.
        if pi_shared5_median is not None and s in R_shared5:
            entry["fit"]["at_shared5_pi_and_its_R_peak"] = forward_ratios(
                C_s, s, pi_shared5_median, R_shared5[s])
            entry["fit"]["nll_shared5_pi_and_its_R_peak"] = nll(
                pi_shared5_median, R_shared5[s],
                float(shared5["phi_nb"]["median"]))
            entry["fit"]["nll_shared5_pi_and_its_R_peak_at_common_phi"] = nll(
                pi_shared5_median, R_shared5[s], COMMON_PHI)
        if s in diag_all:
            dg = diag_all[s]
            entry["diagnostics"] = dict(
                r_hat_max=dg["rhat_max"], ess_bulk_min=dg["ess_bulk_min"],
                ess_tail_min=dg["ess_tail_min"],
                n_divergent_sampling=dg["sampling_stats"].get("n_divergent"),
                n_divergent_warmup=dg["warmup_stats"].get("n_divergent"),
                saturation_pct_sampling=dg["sampling_stats"].get("saturation_pct"),
                warmup=dg["config"]["warmup"], samples=dg["config"]["samples"],
                chains=dg["config"]["chains"],
                goal_check=dg["goal_check"],
                wall_warmup_sec=dg["wall_warmup_sec"],
                wall_sampling_sec=dg["wall_sampling_sec"],
            )
        per_season[s] = entry
        print(f"  [{s}] pi_med={np.round(pi_med, 4).tolist()} "
              f"R_peak={R_med:.3f} phi={phi_med:.3f}", flush=True)
        print(f"        common-phi NLL: own={nll_own_cp:,.1f} "
              f"shared={nll_shared_cp:,.1f} "
              f"(penalty {nll_shared_cp - nll_own_cp:+,.1f})", flush=True)
        print(f"        refit R_peak: own={R_own_refit:.3f} "
              f"(NLL {nll_own_refit:,.1f})  shared={R_shared_refit:.3f} "
              f"(NLL {nll_shared_refit:,.1f})  "
              f"penalty {nll_shared_refit - nll_own_refit:+,.1f}", flush=True)
        print(f"        refit R+phi:  own R={R_own_rp:.3f} phi={phi_own_rp:.3f} "
              f"(NLL {nll_own_rp:,.1f})  shared R={R_sh_rp:.3f} "
              f"phi={phi_sh_rp:.3f} (NLL {nll_sh_rp:,.1f})  "
              f"penalty {nll_sh_rp - nll_own_rp:+,.1f}", flush=True)

    # ---- spread block ------------------------------------------------------
    main_fit_seasons = [s for s in available if s != EXCLUDED]
    spread = dict(channels={}, inside_shared_ci={})
    for i, c in enumerate(CHANNELS):
        meds_all = {s: per_season[s][f"pi_{c}"]["median"] for s in available}
        meds_main = {s: v for s, v in meds_all.items() if s != EXCLUDED}
        lo, hi = shared["pi"][c]["q05"], shared["pi"][c]["q95"]
        spread["channels"][c] = dict(
            per_season_median=meds_all,
            min=min(meds_all.values()), max=max(meds_all.values()),
            range=max(meds_all.values()) - min(meds_all.values()),
            min_main_fit_seasons=(min(meds_main.values()) if meds_main else None),
            max_main_fit_seasons=(max(meds_main.values()) if meds_main else None),
            range_main_fit_seasons=((max(meds_main.values()) - min(meds_main.values()))
                                     if meds_main else None),
            shared_median=shared["pi"][c]["median"],
            shared_q05=lo, shared_q95=hi,
        )
        spread["inside_shared_ci"][c] = {
            s: bool(lo <= meds_all[s] <= hi) for s in available
        }

    # ---- delta_fit block ---------------------------------------------------
    main4 = [s for s in JOINT_SEASONS if s in per_season]
    delta_fit = dict(
        note=("NLL at the posterior median of each fit. phi is a fitted "
              "parameter and spans 0.78-2.41 across seasons, so arms are only "
              "comparable at a common phi; the *_at_common_phi fields (phi = "
              "the 4-season joint median) are the comparable ones. Free "
              "parameters: joint = 4 R_peak + 3 logit_pi_free + 1 phi_nb = 8; "
              "per-season over the four main seasons = 4 x (1 + 3 + 1) = 20, so "
              "12 extra. Not a likelihood-ratio test — these are posterior "
              "medians, not maximum-likelihood fits — so the chi-square "
              "yardstick is a rough reference only."),
        common_phi=COMMON_PHI,
        headline=("sum over the four main-fit seasons of "
                  "penalty_of_sharing_refit_R_phi — within-season, with BOTH "
                  "R_peak and phi profiled out under each allocation, so pi is "
                  "the only thing that differs. Fixing phi at any single value "
                  "instead tilts the comparison, because pi and phi were "
                  "co-fitted; the common-phi and own-phi variants are kept as "
                  "sensitivity bounds on either side."),
        # --- representative numbers (within-season, common phi) -------------
        penalty_of_sharing_at_common_phi={
            s: per_season[s]["fit"]["nll_penalty_of_sharing_at_common_phi"]
            for s in available},
        penalty_of_sharing_refit_R_at_common_phi={
            s: per_season[s]["fit"]["penalty_of_sharing_refit_R_at_common_phi"]
            for s in available},
        sum_penalty_4season_at_common_phi=float(sum(
            per_season[s]["fit"]["nll_penalty_of_sharing_at_common_phi"]
            for s in main4)),
        sum_penalty_4season_refit_R_at_common_phi=float(sum(
            per_season[s]["fit"]["penalty_of_sharing_refit_R_at_common_phi"]
            for s in main4)),
        penalty_of_sharing_refit_R_phi={
            s: per_season[s]["fit"]["penalty_of_sharing_refit_R_phi"]
            for s in available},
        sum_penalty_4season_refit_R_phi=float(sum(
            per_season[s]["fit"]["penalty_of_sharing_refit_R_phi"]
            for s in main4)),
        # --- kept for transparency: own-phi arms, not cross-comparable ------
        per_season_nll_own_pi={s: per_season[s]["fit"]["nll_own_pi"] for s in available},
        per_season_nll_shared_pi={s: per_season[s]["fit"]["nll_shared_pi"]
                                    for s in available},
        penalty_of_sharing_pi_own_phi={
            s: per_season[s]["fit"]["nll_penalty_of_sharing"] for s in available},
    )
    if all(s in per_season for s in JOINT_SEASONS):
        C_joint = FS.build_fluA_setup(JOINT_SEASONS, verbose=False)
        loss_joint = jax.jit(FA.make_joint_loss_fluA(C_joint))
        log_R0_joint = np.asarray(dj["log_R0"])
        # Evaluated at the same kind of point as the per-season NLLs: log of the
        # normalized marginal-median pi (so the NLL corresponds to the pi values
        # reported in the table), and the marginal median of R_peak / phi_nb.
        joint_nll = float(loss_joint(
            jnp.asarray(np.median(log_R0_joint, axis=0)),
            jnp.asarray(np.log(pi_shared_median)),
            jnp.asarray(float(np.median(np.asarray(dj["phi_nb"])))),
        ))
        sum_own = float(sum(per_season[s]["fit"]["nll_own_pi"] for s in JOINT_SEASONS))
        # The joint fit already sits at the common phi, so the comparable
        # per-season sum is the common-phi one.
        sum_own_cp = float(sum(
            per_season[s]["fit"]["nll_own_pi_at_common_phi"] for s in JOINT_SEASONS))
        sum_own_refit_cp = float(sum(
            per_season[s]["fit"]["at_own_pi_refit_R"]["nll"] for s in JOINT_SEASONS))
        delta_fit.update(
            joint_nll_4season=joint_nll,
            per_season_nll_sum_4season_own_phi=sum_own,
            per_season_nll_sum_4season_at_common_phi=sum_own_cp,
            per_season_nll_sum_4season_refit_R_at_common_phi=sum_own_refit_cp,
            improvement_from_per_season_pi_own_phi=joint_nll - sum_own,
            improvement_from_per_season_pi_at_common_phi=joint_nll - sum_own_cp,
            improvement_from_per_season_pi_refit_R_at_common_phi=(
                joint_nll - sum_own_refit_cp),
            n_params_joint=8, n_params_per_season_4=4 * 5,
            extra_params=4 * 5 - 8,
            chi2_95_reference_for_12_df=21.03,
            half_chi2_95_reference=10.5,
        )
        print(f"\n[delta_fit] joint NLL (4 seasons, phi={COMMON_PHI:.3f}) = "
              f"{joint_nll:,.1f}")
        print(f"            per-season sum, own phi      = {sum_own:,.1f}  "
              f"(improvement {joint_nll - sum_own:+,.1f})  <- NOT comparable")
        print(f"            per-season sum, common phi   = {sum_own_cp:,.1f}  "
              f"(improvement {joint_nll - sum_own_cp:+,.1f})")
        print(f"            per-season sum, common phi + refit R = "
              f"{sum_own_refit_cp:,.1f}  "
              f"(improvement {joint_nll - sum_own_refit_cp:+,.1f})")
    else:
        print("\n[delta_fit] joint-vs-sum comparison skipped — not all four "
              "main-fit seasons are available yet")

    out = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        convention=dict(
            summary="median with q05/q95 (90% CI)",
            estimated_reproduction_number="R_peak (R_0 is derived)",
            nll_comparability=(
                "phi_NB is a fitted parameter and spans 0.783-2.412 across the "
                "per-season fits, so an NLL evaluated at each fit's own phi is "
                "NOT comparable across arms: by construction the own-phi arm is "
                "lower than the same mean curve scored at any other phi. Compare "
                "only the *_at_common_phi fields, evaluated at phi = "
                f"{COMMON_PHI:.4f} (the 4-season joint median). Within a single "
                "season, nll_penalty_of_sharing holds phi and R_peak fixed and "
                "varies only pi, so it is internally consistent even at own phi; "
                "penalty_of_sharing_refit_R_at_common_phi additionally refits "
                "R_peak under each allocation and is the like-for-like cost of "
                "sharing."),
            common_phi=COMMON_PHI),
        seasons_available=available, seasons_missing=missing,
        excluded_from_main_fit=EXCLUDED,
        shared=shared, shared_5season_reference=shared5,
        per_season=per_season, spread=spread, delta_fit=delta_fit,
    )
    outp = FLUA_DIR / "per_season_pi_fluA4.json"
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[json] {outp}")

    make_figure(out)
    print(f"[total] {time.perf_counter() - t0:.1f}s")


def make_figure(out: dict) -> None:
    """Four channel panels; seasons as coloured markers with 90% CI bars, the
    shared allocation as a grey band (its 90% CI) plus a line at its median."""
    available = out["seasons_available"]
    per_season = out["per_season"]
    shared = out["shared"]

    fig, axes = plt.subplots(1, 4, figsize=(W_DOUBLE, 3.0),
                             constrained_layout=True)
    for i, (c, ax) in enumerate(zip(CHANNELS, axes)):
        lo = shared["pi"][c]["q05"]; hi = shared["pi"][c]["q95"]
        med = shared["pi"][c]["median"]
        ax.axhspan(lo, hi, color="#BBBBBB", alpha=0.45, lw=0, zorder=0)
        ax.axhline(med, color="#555555", lw=1.1, ls="--", zorder=1)
        for j, s in enumerate(available):
            v = per_season[s][f"pi_{c}"]
            excl = per_season[s]["excluded_from_main_fit"]
            col = COL_SEASON_5[s]
            ax.errorbar([j], [v["median"]],
                        yerr=[[v["median"] - v["q05"]], [v["q95"] - v["median"]]],
                        color=col, lw=1.2, capsize=3, zorder=3,
                        marker=("s" if excl else "o"), markersize=6,
                        markerfacecolor=("white" if excl else col),
                        markeredgecolor=col, markeredgewidth=1.4)
        ax.set_xticks(range(len(available)))
        ax.set_xticklabels([s[2:4] + "-" + s[7:9] for s in available],
                           fontsize=7, rotation=45, ha="right")
        ax.set_title(rf"$\pi_{{\mathrm{{{c}}}}}$", fontsize=10)
        ax.set_xlim(-0.6, len(available) - 0.4)
        if i == 0:
            ax.set_ylabel("Channel share (per-season fit)")
    handles = [
        Line2D([0], [0], marker="o", color=COL_ZERO, ls="", markersize=6,
               label="Season fitted on its own (90% CI)"),
        Line2D([0], [0], marker="s", color=COL_ZERO, ls="", markersize=6,
               markerfacecolor="white",
               label=f"{EXCLUDED[:4]}-{EXCLUDED[7:9]}: excluded from the main fit"),
        Line2D([0], [0], color="#555555", lw=1.1, ls="--",
               label="Shared 4-season allocation (median)"),
        Line2D([0], [0], color="#BBBBBB", lw=6, alpha=0.6,
               label="Shared allocation, 90% CI"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
               ncol=2, frameon=False, fontsize=7.5)
    savefig(fig, "per_season_pi")
    print("[fig] per_season_pi")


if __name__ == "__main__":
    main()
