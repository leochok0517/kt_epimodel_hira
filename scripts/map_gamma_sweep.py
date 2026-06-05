"""MAP-Bayesian gamma_elder prior sigma sweep.

Phase 1 of strategy roadmap: test whether informative gamma_elder prior
breaks the Corner B mode discovered by 8-start verification.

Design (see docs/PRIOR_SPECIFICATION.md §4):
  MAP_loss = NLL_data
           + (gamma_child - 0.40)^2 / (2 * 0.10^2)
           + (gamma_adult - 0.18)^2 / (2 * 0.05^2)
           + (gamma_elder - 0.25)^2 / (2 * sigma_elder^2)   <-- swept
           + lambda_phi * sum((phi[i+1] - phi[i])^2)
           + sum(beta^2) / (2 * sigma_beta^2)

Sweep: sigma_elder in {0.03, 0.05, 0.07, 0.10, inf} (5)
Starts: from_cornerA (warm), from_cornerB (neutral) (2)
Total: 10 fits, parallel n_jobs=8.

Scale check first: corner B vec -> NLL_data vs penalty ratio. If penalty
contribution < 0.1% of |NLL|, normalize NLL per observation.

Resume: skip if outputs/calibration/map_sigma{sig}_{start}.json exists.

Run detached:
    nohup caffeinate -i -s uv run python scripts/map_gamma_sweep.py \\
        > outputs/calibration/map_sweep.log 2>&1 &
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import math
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)
MLFLOW_URI = "sqlite:///" + str((REPO_ROOT / "outputs" / "mlruns" / "mlflow.db").resolve())

SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
N_TOTAL = 33
MAX_ITER = 1000

# Fixed prior hyperparams
GAMMA_CHILD_MU = 0.40
GAMMA_CHILD_SIGMA = 0.10
GAMMA_ADULT_MU = 0.18
GAMMA_ADULT_SIGMA = 0.05
GAMMA_ELDER_MU = 0.25
LAMBDA_PHI = 0.1
SIGMA_BETA = 1.0  # genuinely weak (per PRIOR_SPECIFICATION.md fix)

# Sweep
SIGMA_ELDER_VALUES = [0.03, 0.05, 0.07, 0.10, float("inf")]  # inf = no prior
START_NAMES = ["from_cornerA", "from_cornerB"]
START_SOURCE = {
    "from_cornerA": "multistart_warm.json",
    "from_cornerB": "multistart_neutral.json",
}

BOUNDS = (
    [(0.05, 5.0)] * 14 +
    [(0.10, 0.60), (0.05, 0.40), (0.03, 0.50)] +
    [(0.001, 0.30)] * 16
)


def load_start_vec(start_name):
    src = OUTDIR / START_SOURCE[start_name]
    with open(src) as f:
        d = json.load(f)
    return np.array(d["best_vec"])


def gamma_penalty(gamma_3, sigma_elder):
    """Sum of Gaussian neg-log-priors for gamma triple."""
    pen = (gamma_3[0] - GAMMA_CHILD_MU) ** 2 / (2 * GAMMA_CHILD_SIGMA ** 2)
    pen += (gamma_3[1] - GAMMA_ADULT_MU) ** 2 / (2 * GAMMA_ADULT_SIGMA ** 2)
    if math.isinf(sigma_elder):
        pass  # no prior
    else:
        pen += (gamma_3[2] - GAMMA_ELDER_MU) ** 2 / (2 * sigma_elder ** 2)
    return float(pen)


def phi_penalty(phi_14, lam=LAMBDA_PHI):
    if lam <= 0:
        return 0.0
    return float(lam * np.sum(np.diff(phi_14) ** 2))


def beta_penalty(beta_16, sigma=SIGMA_BETA):
    if math.isinf(sigma):
        return 0.0
    return float(np.sum(beta_16 ** 2) / (2 * sigma ** 2))


def scale_check(sigma_elder):
    """Pre-sweep diagnostic: NLL vs prior penalty magnitudes for corner B vec."""
    print("\n" + "=" * 70)
    print("SCALE CHECK (corner B representative)")
    print("=" * 70)

    from kt_data import SUDOGWON_SIDO_CODES
    from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
    from kt_epimodel_hira.calibration.simple_model import (
        build_aggregated_inputs, estimate_initial_infected_from_hira,
    )
    from kt_epimodel_hira.calibration.loss import make_loss_function_by_age
    from kt_epimodel_hira.model.parameters import (
        CalibrationParameters, ModelParameters,
    )

    vec = load_start_vec("from_cornerB")
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"].flatten()

    # Build per-season losses
    losses = {}
    n_data_points = 0
    for s in SEASONS:
        tgt = load_hira_target_by_age(
            s, sido_codes=list(SUDOGWON_SIDO_CODES),
            first_peak_only=True, first_peak_end_week=26,
        )
        seed = estimate_initial_infected_from_hira(
            s, pop_15, sido_codes=list(SUDOGWON_SIDO_CODES),
            gamma_15_assumed=CalibrationParameters().gamma_15,
        )
        losses[s] = make_loss_function_by_age(
            tgt, inputs, ModelParameters(),
            seed_total=float(seed.sum()), seed_by_age=seed, seed_e_factor=0.5,
            initial_immunity=0.3, t_span=(0.0, 364.0), verbose=False,
        )
        # Count observations: weights > 0 for each age
        for ag in tgt["age_groups"]:
            n_data_points += int((tgt["weights"][ag] > 0).sum())

    # Compute NLL_data for corner B vec
    phi_14 = vec[:14]
    gamma_3 = vec[14:17]
    nll_data = 0.0
    for i, s in enumerate(SEASONS):
        beta_4 = vec[17 + i*4 : 21 + i*4]
        vec_21 = np.concatenate([beta_4, phi_14, gamma_3])
        nll_data += losses[s](vec_21)

    gp = gamma_penalty(gamma_3, sigma_elder)
    pp = phi_penalty(phi_14)
    bp = beta_penalty(vec[17:33])
    total_pen = gp + pp + bp
    abs_nll = abs(nll_data)
    ratio_gamma = gp / abs_nll * 100 if abs_nll > 0 else 0
    ratio_total = total_pen / abs_nll * 100 if abs_nll > 0 else 0

    print(f"NLL_data:                {nll_data:>20,.0f}")
    print(f"|NLL_data|:              {abs_nll:>20,.0f}")
    print(f"n_data_points:           {n_data_points:>20d}")
    print(f"NLL per observation:     {nll_data / max(n_data_points, 1):>20,.2f}")
    print()
    print(f"gamma_penalty (sigma_e={sigma_elder}): {gp:>20,.4f}")
    print(f"phi_penalty (lambda={LAMBDA_PHI}):     {pp:>20,.4f}")
    print(f"beta_penalty (sigma_b={SIGMA_BETA}):   {bp:>20,.4f}")
    print(f"total_penalty:           {total_pen:>20,.4f}")
    print()
    print(f"gamma_pen / |NLL_data|:  {ratio_gamma:>20.6f} %")
    print(f"total_pen / |NLL_data|:  {ratio_total:>20.6f} %")
    print()

    # Decision: normalize NLL if penalty is < 0.1% of |NLL|
    normalize = ratio_total < 0.1
    if normalize:
        print("[SCALE WARNING] Total penalty < 0.1% of |NLL|.")
        print("  Applying normalization: MAP_loss = NLL_data / n_obs + penalty.")
        print(f"  After normalization, NLL becomes ~{nll_data / n_data_points:.2f}")
        print(f"  New gamma_pen contribution: {gp / abs(nll_data / n_data_points) * 100:.2f}%")
    else:
        print("[SCALE OK] Penalty has non-trivial weight. No normalization needed.")
    print("=" * 70 + "\n")
    return {
        "nll_data": nll_data,
        "n_data_points": n_data_points,
        "gamma_pen": gp, "phi_pen": pp, "beta_pen": bp,
        "ratio_total_pct": ratio_total,
        "normalize": normalize,
    }


def fit_map(sigma_elder, start_name, start_vec, mlflow_uri, normalize_nll,
            n_data_points, max_iter=MAX_ITER):
    """Single MAP fit (parallel-safe)."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    import time, json, math, traceback
    from pathlib import Path
    import numpy as np
    from scipy.optimize import minimize
    import mlflow

    from kt_data import SUDOGWON_SIDO_CODES
    from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
    from kt_epimodel_hira.calibration.simple_model import (
        build_aggregated_inputs, estimate_initial_infected_from_hira,
    )
    from kt_epimodel_hira.calibration.loss import make_loss_function_by_age
    from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters

    sigma_tag = "inf" if math.isinf(sigma_elder) else f"{sigma_elder:.2f}"
    out_path = OUTDIR / f"map_sigma{sigma_tag}_{start_name}.json"

    if out_path.exists():
        return {"sigma_elder": sigma_elder, "start": start_name,
                "status": "skipped_resume", "path": str(out_path)}

    try:
        # Build losses
        inputs = build_aggregated_inputs()
        pop_15 = inputs["pop_15"].flatten()
        losses = {}
        for s in SEASONS:
            tgt = load_hira_target_by_age(
                s, sido_codes=list(SUDOGWON_SIDO_CODES),
                first_peak_only=True, first_peak_end_week=26,
            )
            seed = estimate_initial_infected_from_hira(
                s, pop_15, sido_codes=list(SUDOGWON_SIDO_CODES),
                gamma_15_assumed=CalibrationParameters().gamma_15,
            )
            losses[s] = make_loss_function_by_age(
                tgt, inputs, ModelParameters(),
                seed_total=float(seed.sum()), seed_by_age=seed, seed_e_factor=0.5,
                initial_immunity=0.3, t_span=(0.0, 364.0), verbose=False,
            )

        n_evals = [0]
        nll_history = []

        def map_loss(vec):
            n_evals[0] += 1
            phi = vec[:14]; gamma = vec[14:17]; betas = vec[17:33]
            nll = 0.0
            for i, s in enumerate(SEASONS):
                beta_4 = vec[17 + i*4 : 21 + i*4]
                vec_21 = np.concatenate([beta_4, phi, gamma])
                nll += losses[s](vec_21)

            if normalize_nll:
                nll_term = nll / max(n_data_points, 1)
            else:
                nll_term = nll

            gp = (gamma[0] - 0.40)**2 / (2 * 0.10**2)
            gp += (gamma[1] - 0.18)**2 / (2 * 0.05**2)
            if not math.isinf(sigma_elder):
                gp += (gamma[2] - 0.25)**2 / (2 * sigma_elder**2)
            pp = LAMBDA_PHI * float(np.sum(np.diff(phi)**2))
            bp = float(np.sum(betas**2) / (2 * SIGMA_BETA**2))

            total = nll_term + gp + pp + bp
            nll_history.append((float(total), float(nll), float(gp), float(pp), float(bp)))
            return total

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("hira_calibration_map_gamma_sweep")

        with mlflow.start_run(run_name=f"map_sigma{sigma_tag}_{start_name}"):
            mlflow.log_params({
                "method": "L-BFGS-B MAP",
                "sigma_elder": sigma_tag,
                "start": start_name,
                "lambda_phi": LAMBDA_PHI,
                "sigma_beta": SIGMA_BETA,
                "normalize_nll": normalize_nll,
                "n_data_points": n_data_points,
                "maxiter": max_iter,
            })

            t0 = time.perf_counter()
            sol = minimize(
                map_loss, start_vec.copy(),
                method="L-BFGS-B", bounds=BOUNDS,
                options={"maxiter": max_iter, "disp": False,
                         "ftol": 1e-5, "gtol": 1e-3},
            )
            wall = time.perf_counter() - t0

            # Decompose final
            phi_f = sol.x[:14]; gamma_f = sol.x[14:17]; beta_f = sol.x[17:33]
            nll_data_f = 0.0
            for i, s in enumerate(SEASONS):
                vec_21 = np.concatenate([beta_f[i*4:(i+1)*4], phi_f, gamma_f])
                nll_data_f += losses[s](vec_21)
            gp_f = (gamma_f[0] - 0.40)**2 / (2 * 0.10**2)
            gp_f += (gamma_f[1] - 0.18)**2 / (2 * 0.05**2)
            if not math.isinf(sigma_elder):
                gp_f += (gamma_f[2] - 0.25)**2 / (2 * sigma_elder**2)
            pp_f = LAMBDA_PHI * float(np.sum(np.diff(phi_f)**2))
            bp_f = float(np.sum(beta_f**2) / (2 * SIGMA_BETA**2))

            mlflow.log_metric("map_loss_final", sol.fun)
            mlflow.log_metric("nll_data_final", nll_data_f)
            mlflow.log_metric("gamma_pen_final", gp_f)
            mlflow.log_metric("phi_pen_final", pp_f)
            mlflow.log_metric("beta_pen_final", bp_f)
            mlflow.log_metric("gamma_elder_final", float(gamma_f[2]))
            mlflow.log_metric("gamma_child_final", float(gamma_f[0]))
            mlflow.log_metric("gamma_adult_final", float(gamma_f[1]))
            mlflow.log_metric("n_evals", n_evals[0])
            mlflow.log_metric("wall_time_min", wall / 60)
            mlflow.log_metric("success", int(sol.success))

            out = {
                "sigma_elder": sigma_tag,
                "sigma_elder_value": ("inf" if math.isinf(sigma_elder) else float(sigma_elder)),
                "start": start_name,
                "normalize_nll": normalize_nll,
                "n_data_points": n_data_points,
                "lambda_phi": LAMBDA_PHI,
                "sigma_beta": SIGMA_BETA,
                "method": "L-BFGS-B",
                "maxiter": max_iter,
                "success": bool(sol.success),
                "message": str(sol.message),
                "n_evals": int(n_evals[0]),
                "wall_time_sec": float(wall),
                "map_loss_final": float(sol.fun),
                "nll_data_final": float(nll_data_f),
                "gamma_pen_final": float(gp_f),
                "phi_pen_final": float(pp_f),
                "beta_pen_final": float(bp_f),
                "best_vec": sol.x.tolist(),
                "phi": phi_f.tolist(),
                "gamma": {"child": float(gamma_f[0]), "adult": float(gamma_f[1]),
                          "elder": float(gamma_f[2])},
                "betas": {SEASONS[i]: beta_f[i*4:(i+1)*4].tolist()
                          for i in range(len(SEASONS))},
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            mlflow.log_artifact(str(out_path))

        return {"sigma_elder": sigma_elder, "start": start_name,
                "status": "complete", "gamma_elder": float(gamma_f[2]),
                "nll_data": float(nll_data_f), "wall_min": wall / 60}

    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        with open(out_path.parent / f"map_sigma{sigma_tag}_{start_name}_ERROR.json", "w") as f:
            json.dump({"sigma": sigma_tag, "start": start_name, "error": err}, f, indent=2)
        return {"sigma_elder": sigma_elder, "start": start_name,
                "status": "error", "error": err[:500]}


def main():
    print("=" * 70)
    print("MAP gamma_elder prior sigma sweep")
    print(f"sigmas: {SIGMA_ELDER_VALUES}")
    print(f"starts: {START_NAMES}")
    print(f"total fits: {len(SIGMA_ELDER_VALUES) * len(START_NAMES)}")
    print("=" * 70)

    # Scale check (with default sigma=0.07)
    sc = scale_check(0.07)

    # Configs
    starts = {name: load_start_vec(name) for name in START_NAMES}
    print("Start vectors loaded:")
    for n, v in starts.items():
        print(f"  {n}: phi[0]={v[0]:.3f}, gamma_elder={v[16]:.4f}")

    configs = [
        (sig, sname, svec)
        for sig in SIGMA_ELDER_VALUES
        for sname, svec in starts.items()
    ]

    print(f"\nLaunching {len(configs)} parallel fits (n_jobs=8)...")
    t0 = time.perf_counter()

    results = Parallel(n_jobs=8, backend="loky", verbose=10)(
        delayed(fit_map)(
            sig, sname, svec, MLFLOW_URI,
            sc["normalize"], sc["n_data_points"],
        )
        for sig, sname, svec in configs
    )

    wall_total = time.perf_counter() - t0
    print(f"\n=== Total wall time: {wall_total/60:.1f} min ===\n")

    # Summary
    summary = {}
    for r in results:
        key = f"sigma_{r['sigma_elder']}_{r['start']}"
        summary[key] = r
    with open(OUTDIR / "map_sweep_summary.json", "w") as f:
        json.dump({"scale_check": sc, "results": summary}, f, indent=2, default=str)

    print("=== Results ===")
    for r in results:
        sig = r["sigma_elder"]
        sig_str = "inf" if math.isinf(sig) else f"{sig:.2f}"
        if r["status"] == "complete":
            print(f"  sigma={sig_str:>5} start={r['start']:>13}: "
                  f"gamma_elder={r['gamma_elder']:.4f}  "
                  f"NLL_data={r['nll_data']:,.0f}  ({r['wall_min']:.0f}min)")
        elif r["status"] == "skipped_resume":
            print(f"  sigma={sig_str:>5} start={r['start']:>13}: SKIPPED (resume)")
        else:
            print(f"  sigma={sig_str:>5} start={r['start']:>13}: ERROR ({r.get('error', '?')[:80]})")


if __name__ == "__main__":
    main()
