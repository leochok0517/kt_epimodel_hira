"""Multi-start verification for multi-season joint fit.

Goal: check whether current best (v2 lambda=0.1) is unique global minimum or
one of multiple corners. 8 starting points, parallel L-BFGS-B with lambda=0.1.

Run with sleep blocking:
    caffeinate -i -s uv run python scripts/multistart_verification.py \\
        2>&1 | tee outputs/calibration/multistart.log
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
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)
MLFLOW_URI = "sqlite:///" + str((REPO_ROOT / "outputs" / "mlruns" / "mlflow.db").resolve())

SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
N_TOTAL = 33
LAMBDA = 0.1
MAX_ITER = 1000

# Bounds (same as v2)
BOUNDS = (
    [(0.05, 5.0)] * 14 +
    [(0.10, 0.60), (0.05, 0.40), (0.03, 0.50)] +
    [(0.001, 0.30)] * 16
)


def build_vec(phi, gamma, beta_per_season):
    """Build 33-dim vector from components."""
    vec = np.zeros(N_TOTAL)
    vec[:14] = phi
    vec[14:17] = gamma
    for i, beta_4 in enumerate(beta_per_season):
        vec[17 + i*4 : 21 + i*4] = beta_4
    return vec


def make_start_points(seed=42):
    """Return dict of name -> 33-dim starting vector."""
    rng = np.random.default_rng(seed)
    starts = {}

    # 0. warm: v2 lambda=0.1 result
    v2_path = OUTDIR / "multi_season_v2_lambda0.1.json"
    with open(v2_path) as f:
        v2 = json.load(f)
    starts["warm"] = np.array(v2["vector_33"])

    # 1. neutral: phi=1.0, beta=0.06 (R0 sweep target)
    starts["neutral"] = build_vec(
        np.ones(14), [0.40, 0.18, 0.25], [[0.06]*4]*4,
    )

    # 2. low_phi high beta
    starts["low_phi"] = build_vec(
        np.full(14, 0.5), [0.40, 0.18, 0.25], [[0.15]*4]*4,
    )

    # 3. high_phi low beta (opposite corner, ILI-style)
    starts["high_phi"] = build_vec(
        np.full(14, 2.0), [0.40, 0.18, 0.25], [[0.02]*4]*4,
    )

    # 4. bio_prior: children high, elderly low
    phi_bio = np.array([1.5, 1.5, 1.3, 1.2, 1.0,
                         0.9, 0.9, 0.8, 0.8, 0.7,
                         0.7, 0.6, 0.6, 0.5])
    starts["bio_prior"] = build_vec(
        phi_bio, [0.40, 0.18, 0.25], [[0.06]*4]*4,
    )

    # 5. home_dominant: beta_h large, others small (ILI corner)
    starts["home_dominant"] = build_vec(
        np.ones(14), [0.40, 0.18, 0.25],
        [[0.20, 0.01, 0.01, 0.01]]*4,
    )

    # 6. distributed: equal 4 channels
    starts["distributed"] = build_vec(
        np.ones(14), [0.40, 0.18, 0.25], [[0.06]*4]*4,
    )

    # 7. random: bounds-clipped uniform (constrained to plausible)
    phi_rand = 0.3 + rng.random(14) * 1.7              # 0.3-2.0
    gamma_rand = np.array([0.20, 0.10, 0.20]) + rng.random(3) * np.array([0.30, 0.20, 0.20])
    beta_rand = 0.02 + rng.random((4, 4)) * 0.15        # 0.02-0.17
    starts["random"] = build_vec(phi_rand, gamma_rand, beta_rand)

    # Clip everything to bounds
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    for name in starts:
        starts[name] = np.clip(starts[name], lo, hi)

    return starts


def compute_R0_ngm(vec_21, inputs_dict, disease):
    """R0 at peak for season-mapped 21-dim cal vector."""
    from kt_epimodel_hira.calibration.param_vector import vector_to_params
    cal = vector_to_params(vec_21)
    pop = inputs_dict["pop_15"].flatten()
    N_safe = np.maximum(pop, 1e-10)
    matrices = inputs_dict["matrices"]
    rho = inputs_dict["rho"].flatten()
    sf = disease.seasonal_factor(disease.seasonality_peak_day)
    C_eff = np.zeros((15, 15))
    C_eff += cal.beta_h * matrices["C_home"]
    C_eff[:4, :4] += cal.beta_s * matrices["C_school"][:4, :4]
    rho_ok = (rho > 0).astype(float)
    for a in range(4, 14):
        C_eff[a, :] += cal.beta_w * matrices["C_work"][a, :] * rho[a] * rho_ok
    C_eff += cal.beta_o * matrices["C_other"]
    K = (0.7 * sf / disease.gamma) * np.diag(pop) @ np.diag(cal.phi) @ C_eff @ np.diag(1.0/N_safe)
    return float(np.max(np.real(np.linalg.eigvals(K))))


def fit_from_start(start_name, start_vec, seasons, outdir_str, mlflow_uri, lam, max_iter):
    """Run multi-season smoothed L-BFGS-B from a given start. Parallel-safe."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    import time, json, traceback
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
    from kt_epimodel_hira.model.parameters import (
        CalibrationParameters, DiseaseParameters, ModelParameters,
    )
    from kt_epimodel_hira.calibration.param_vector import vector_to_params

    out_path = Path(outdir_str) / f"multistart_{start_name}.json"
    try:
        # Build per-worker loss closures
        inputs = build_aggregated_inputs()
        pop_15 = inputs["pop_15"].flatten()
        losses = {}
        for s in seasons:
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

        dis = DiseaseParameters()

        # Initial R0 (use season 0 beta for representative R0)
        phi_init = start_vec[:14]; gamma_init = start_vec[14:17]; beta_s0 = start_vec[17:21]
        vec_21_init = np.concatenate([beta_s0, phi_init, gamma_init])
        try:
            R0_init = compute_R0_ngm(vec_21_init, inputs, dis)
        except Exception:
            R0_init = float("nan")

        # Smoothed loss
        n_evals = [0]

        def loss_smoothed(vec):
            n_evals[0] += 1
            phi = vec[:14]; gamma = vec[14:17]
            base = 0.0
            for i, s in enumerate(seasons):
                beta = vec[17 + i*4 : 21 + i*4]
                vec_21 = np.concatenate([beta, phi, gamma])
                base += losses[s](vec_21)
            smooth = lam * float(np.sum(np.diff(phi)**2)) if lam > 0 else 0.0
            return base + smooth

        # Initial NLL
        nll_initial = loss_smoothed(start_vec.copy())

        # mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("hira_calibration_multistart")

        with mlflow.start_run(run_name=f"multistart_{start_name}"):
            mlflow.log_params({
                "start_point": start_name, "method": "L-BFGS-B",
                "lambda_smooth": lam, "maxiter": max_iter,
                "dimension": 33, "n_seasons": len(seasons),
                "phi_bound_lower": 0.05,
            })
            mlflow.log_metric("R0_initial", R0_init)
            mlflow.log_metric("nll_initial", nll_initial)

            t0 = time.perf_counter()
            bounds = (
                [(0.05, 5.0)] * 14 +
                [(0.10, 0.60), (0.05, 0.40), (0.03, 0.50)] +
                [(0.001, 0.30)] * 16
            )
            sol = minimize(
                loss_smoothed, start_vec.copy(),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": max_iter, "disp": False},
            )
            wall = time.perf_counter() - t0

            phi_f = sol.x[:14]
            smooth_f = lam * float(np.sum(np.diff(phi_f)**2))
            data_nll_f = sol.fun - smooth_f

            # Final R0 (season 0)
            vec_21_f = np.concatenate([sol.x[17:21], sol.x[:14], sol.x[14:17]])
            try:
                R0_final = compute_R0_ngm(vec_21_f, inputs, dis)
            except Exception:
                R0_final = float("nan")

            mlflow.log_metric("nll_total_final", sol.fun)
            mlflow.log_metric("nll_data_final", data_nll_f)
            mlflow.log_metric("smoothing_final", smooth_f)
            mlflow.log_metric("R0_final", R0_final)
            mlflow.log_metric("n_evals", n_evals[0])
            mlflow.log_metric("wall_time_min", wall / 60)
            mlflow.log_metric("success", int(sol.success))

            out = {
                "start_name": start_name, "method": "L-BFGS-B",
                "lambda_smooth": lam, "maxiter": max_iter,
                "success": bool(sol.success), "message": str(sol.message),
                "n_evals": int(n_evals[0]),
                "wall_time_sec": float(wall),
                "nll_initial": float(nll_initial),
                "nll_total_final": float(sol.fun),
                "nll_data_final": float(data_nll_f),
                "smoothing_final": float(smooth_f),
                "R0_initial": float(R0_init),
                "R0_final": float(R0_final),
                "start_vec": start_vec.tolist(),
                "best_vec": sol.x.tolist(),
                "phi": sol.x[:14].tolist(),
                "gamma": {"child": float(sol.x[14]), "adult": float(sol.x[15]),
                          "elder": float(sol.x[16])},
                "betas": {seasons[i]: sol.x[17+i*4:21+i*4].tolist()
                          for i in range(len(seasons))},
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            mlflow.log_artifact(str(out_path))

        return {"start": start_name, "result": out, "error": None}

    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        # Save partial error info
        with open(out_path.parent / f"multistart_{start_name}_ERROR.json", "w") as f:
            json.dump({"start_name": start_name, "error": err}, f, indent=2)
        return {"start": start_name, "result": None, "error": err}


def main():
    starts = make_start_points(seed=42)
    print(f"Built {len(starts)} starting points: {list(starts.keys())}")
    print(f"Lambda={LAMBDA}, maxiter={MAX_ITER}")

    # Pre-flight: R0 for each start
    print("\n=== Pre-flight R0 (initial, peak sf) ===")
    from kt_epimodel_hira.calibration.simple_model import build_aggregated_inputs
    from kt_epimodel_hira.model.parameters import DiseaseParameters
    inputs = build_aggregated_inputs()
    dis = DiseaseParameters()
    for name, vec in starts.items():
        vec_21 = np.concatenate([vec[17:21], vec[:14], vec[14:17]])
        try:
            R0 = compute_R0_ngm(vec_21, inputs, dis)
            print(f"  {name:>15}: R0 = {R0:.3f}")
        except Exception as e:
            print(f"  {name:>15}: R0 error: {e}")

    print(f"\n=== Launching {len(starts)} parallel fits ===")
    t0 = time.perf_counter()

    results = Parallel(n_jobs=len(starts), backend="loky", verbose=10)(
        delayed(fit_from_start)(
            name, vec, SEASONS, str(OUTDIR), MLFLOW_URI, LAMBDA, MAX_ITER,
        )
        for name, vec in starts.items()
    )

    wall_total = time.perf_counter() - t0
    print(f"\n=== Total wall time: {wall_total/60:.1f} min ===")

    # Summary
    summary = {}
    for r in results:
        if r["error"]:
            summary[r["start"]] = {"error": r["error"][:200]}
        else:
            res = r["result"]
            summary[r["start"]] = {
                "nll_total_final": res["nll_total_final"],
                "nll_data_final": res["nll_data_final"],
                "n_evals": res["n_evals"],
                "wall_time_min": round(res["wall_time_sec"] / 60, 1),
                "success": res["success"],
                "R0_final": res["R0_final"],
            }
    with open(OUTDIR / "multistart_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {OUTDIR}/multistart_summary.json")
    for name, s in summary.items():
        if "error" in s:
            print(f"  {name}: FAILED — {s['error'][:80]}")
        else:
            print(f"  {name}: NLL={s['nll_total_final']:,.0f}  "
                  f"R0={s['R0_final']:.3f}  {s['wall_time_min']:.0f}min  "
                  f"success={s['success']}")


if __name__ == "__main__":
    main()
