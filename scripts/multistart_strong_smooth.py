"""Strong phi smoothing (lambda=10) multi-start verification.

Hypothesis: Corner B (gamma_elder ~ 0.05) is driven by oscillating phi profile.
Strong phi smoothing (lambda=10, 100x previous) should eliminate phi
oscillation and thereby remove Corner B if the hypothesis holds.

Loss = NLL_data + lambda_phi * sum((phi[i+1] - phi[i])^2)
(no gamma prior: smoothing effect in isolation)

Auto-applies new ODE/optimizer tolerances (rtol=1e-4, atol=1e-6, ftol=1e-5,
gtol=1e-3) via updated defaults in solver.py and optimizer.

Run detached:
    nohup caffeinate -i -s uv run python scripts/multistart_strong_smooth.py \\
        > outputs/calibration/strong_smooth.log 2>&1 & disown
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
LAMBDA_PHI = 10.0  # 100x previous (0.1)
MAX_ITER = 1000

BOUNDS = (
    [(0.05, 5.0)] * 14 +
    [(0.10, 0.60), (0.05, 0.40), (0.03, 0.50)] +
    [(0.001, 0.30)] * 16
)


def build_vec(phi, gamma, beta_per_season):
    vec = np.zeros(N_TOTAL)
    vec[:14] = phi
    vec[14:17] = gamma
    for i, beta_4 in enumerate(beta_per_season):
        vec[17 + i*4 : 21 + i*4] = beta_4
    return vec


def make_start_points(seed=42):
    """Same 8 starts as multistart_verification.py."""
    rng = np.random.default_rng(seed)
    starts = {}

    v2_path = OUTDIR / "multi_season_v2_lambda0.1.json"
    with open(v2_path) as f:
        v2 = json.load(f)
    starts["warm"] = np.array(v2["vector_33"])

    starts["neutral"] = build_vec(
        np.ones(14), [0.40, 0.18, 0.25], [[0.06]*4]*4,
    )
    starts["low_phi"] = build_vec(
        np.full(14, 0.5), [0.40, 0.18, 0.25], [[0.15]*4]*4,
    )
    starts["high_phi"] = build_vec(
        np.full(14, 2.0), [0.40, 0.18, 0.25], [[0.02]*4]*4,
    )
    phi_bio = np.array([1.5, 1.5, 1.3, 1.2, 1.0,
                         0.9, 0.9, 0.8, 0.8, 0.7,
                         0.7, 0.6, 0.6, 0.5])
    starts["bio_prior"] = build_vec(
        phi_bio, [0.40, 0.18, 0.25], [[0.06]*4]*4,
    )
    starts["home_dominant"] = build_vec(
        np.ones(14), [0.40, 0.18, 0.25],
        [[0.20, 0.01, 0.01, 0.01]]*4,
    )
    starts["distributed"] = build_vec(
        np.ones(14), [0.40, 0.18, 0.25], [[0.06]*4]*4,
    )
    phi_rand = 0.3 + rng.random(14) * 1.7
    gamma_rand = np.array([0.20, 0.10, 0.20]) + rng.random(3) * np.array([0.30, 0.20, 0.20])
    beta_rand = 0.02 + rng.random((4, 4)) * 0.15
    starts["random"] = build_vec(phi_rand, gamma_rand, beta_rand)

    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    for name in starts:
        starts[name] = np.clip(starts[name], lo, hi)
    return starts


def scale_check_smoothing(starts, lambda_phi):
    """Pre-sweep diagnostic: NLL_data vs phi smoothing penalty magnitudes."""
    print("=" * 70)
    print("SCALE CHECK (smoothing penalty)")
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

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"].flatten()

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
        for ag in tgt["age_groups"]:
            n_data_points += int((tgt["weights"][ag] > 0).sum())

    # Sample 3 starts
    sample = ["warm", "neutral", "random"]
    print(f"NLL_data per start (sample, rtol=1e-4 new default):")
    nll_vals = []
    for name in sample:
        vec = starts[name]
        nll = 0.0
        for i, s in enumerate(SEASONS):
            beta_4 = vec[17 + i*4 : 21 + i*4]
            vec_21 = np.concatenate([beta_4, vec[:14], vec[14:17]])
            nll += losses[s](vec_21)
        phi_pen = lambda_phi * float(np.sum(np.diff(vec[:14])**2))
        nll_vals.append(nll)
        print(f"  {name:>12}: NLL_data={nll:>15,.0f}  phi_pen={phi_pen:>10.2f}  "
              f"ratio={abs(phi_pen/nll)*100:.6f}%")

    mean_nll = float(np.mean(nll_vals))
    # Decision rule (same as MAP sweep): normalize if total penalty < 0.1% of |NLL|
    sample_phi_pen = lambda_phi * float(np.sum(np.diff(starts["random"][:14])**2))
    ratio = abs(sample_phi_pen / mean_nll) * 100
    normalize = ratio < 0.1
    print(f"\nn_data_points: {n_data_points}")
    print(f"After normalization NLL/n_obs ~ {mean_nll/n_data_points:.2f}")
    if normalize:
        print("[SCALE WARNING] phi penalty < 0.1% of |NLL|.")
        print("  Applying normalization: loss = NLL_data / n_obs + phi_pen.")
    else:
        print("[SCALE OK] phi penalty has non-trivial weight.")
    print("=" * 70 + "\n")
    return {"n_data_points": n_data_points, "normalize": normalize}


def fit_one_start(start_name, start_vec, seasons, outdir_str, mlflow_uri,
                  lambda_phi, max_iter, normalize_nll, n_data_points):
    """Single start fit with strong phi smoothing."""
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
    from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters

    out_path = Path(outdir_str) / f"strongsmooth_{start_name}.json"
    if out_path.exists():
        return {"start": start_name, "status": "skipped_resume"}

    try:
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

        n_evals = [0]

        def loss_smoothed(vec):
            n_evals[0] += 1
            phi = vec[:14]
            nll = 0.0
            for i, s in enumerate(seasons):
                beta = vec[17 + i*4 : 21 + i*4]
                vec_21 = np.concatenate([beta, phi, vec[14:17]])
                nll += losses[s](vec_21)
            if normalize_nll:
                nll_term = nll / max(n_data_points, 1)
            else:
                nll_term = nll
            smooth = lambda_phi * float(np.sum(np.diff(phi)**2))
            return nll_term + smooth

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("hira_calibration_strong_smooth")

        with mlflow.start_run(run_name=f"strongsmooth_{start_name}"):
            mlflow.log_params({
                "method": "L-BFGS-B strong smoothing",
                "lambda_phi": lambda_phi,
                "start": start_name,
                "dimension": 33, "maxiter": max_iter,
                "normalize_nll": normalize_nll, "n_data_points": n_data_points,
                "gamma_prior": "none",
            })

            t0 = time.perf_counter()
            sol = minimize(
                loss_smoothed, start_vec.copy(),
                method="L-BFGS-B", bounds=BOUNDS,
                options={"maxiter": max_iter, "disp": False,
                         "ftol": 1e-5, "gtol": 1e-3},
            )
            wall = time.perf_counter() - t0

            # Decompose final
            phi_f = sol.x[:14]; gamma_f = sol.x[14:17]; beta_f = sol.x[17:33]
            nll_data_f = 0.0
            for i, s in enumerate(seasons):
                vec_21 = np.concatenate([beta_f[i*4:(i+1)*4], phi_f, gamma_f])
                nll_data_f += losses[s](vec_21)
            phi_pen_f = lambda_phi * float(np.sum(np.diff(phi_f)**2))
            phi_jump_max = float(np.max(np.abs(np.diff(phi_f))))

            mlflow.log_metric("loss_final", sol.fun)
            mlflow.log_metric("nll_data_final", nll_data_f)
            mlflow.log_metric("phi_pen_final", phi_pen_f)
            mlflow.log_metric("phi_jump_max", phi_jump_max)
            mlflow.log_metric("gamma_elder_final", float(gamma_f[2]))
            mlflow.log_metric("gamma_child_final", float(gamma_f[0]))
            mlflow.log_metric("gamma_adult_final", float(gamma_f[1]))
            mlflow.log_metric("n_evals", n_evals[0])
            mlflow.log_metric("wall_time_min", wall / 60)
            mlflow.log_metric("success", int(sol.success))

            out = {
                "start": start_name,
                "lambda_phi": lambda_phi,
                "normalize_nll": normalize_nll,
                "n_data_points": n_data_points,
                "method": "L-BFGS-B",
                "success": bool(sol.success),
                "message": str(sol.message),
                "n_evals": int(n_evals[0]),
                "wall_time_sec": float(wall),
                "loss_final": float(sol.fun),
                "nll_data_final": float(nll_data_f),
                "phi_pen_final": float(phi_pen_f),
                "phi_jump_max": float(phi_jump_max),
                "start_vec": start_vec.tolist(),
                "best_vec": sol.x.tolist(),
                "phi": phi_f.tolist(),
                "gamma": {"child": float(gamma_f[0]), "adult": float(gamma_f[1]),
                          "elder": float(gamma_f[2])},
                "betas": {seasons[i]: beta_f[i*4:(i+1)*4].tolist() for i in range(len(seasons))},
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            mlflow.log_artifact(str(out_path))

        return {"start": start_name, "status": "complete",
                "gamma_elder": float(gamma_f[2]), "phi_jump_max": phi_jump_max,
                "nll_data": float(nll_data_f), "wall_min": wall / 60}

    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        with open(out_path.parent / f"strongsmooth_{start_name}_ERROR.json", "w") as f:
            json.dump({"start": start_name, "error": err}, f, indent=2)
        return {"start": start_name, "status": "error", "error": err[:300]}


def main():
    starts = make_start_points(seed=42)
    print(f"Built {len(starts)} starting points: {list(starts.keys())}")
    print(f"lambda_phi={LAMBDA_PHI} (100x previous 0.1)")
    print(f"gamma prior: NONE (smoothing in isolation)")
    print(f"new tol: ODE rtol=1e-4 atol=1e-6, opt ftol=1e-5 gtol=1e-3 "
          f"(auto via solver/optimizer defaults)")

    sc = scale_check_smoothing(starts, LAMBDA_PHI)

    print(f"=== Launching {len(starts)} parallel fits ===")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=len(starts), backend="loky", verbose=10)(
        delayed(fit_one_start)(
            name, vec, SEASONS, str(OUTDIR), MLFLOW_URI,
            LAMBDA_PHI, MAX_ITER, sc["normalize"], sc["n_data_points"],
        )
        for name, vec in starts.items()
    )
    wall_total = time.perf_counter() - t0
    print(f"\n=== Total wall time: {wall_total/60:.1f} min ===")

    summary = {r["start"]: r for r in results}
    with open(OUTDIR / "strongsmooth_summary.json", "w") as f:
        json.dump({"scale_check": sc, "results": summary}, f, indent=2, default=str)
    print(f"\nSummary: {OUTDIR}/strongsmooth_summary.json")
    for r in results:
        if r["status"] == "complete":
            print(f"  {r['start']:>14}: gamma_elder={r['gamma_elder']:.4f}  "
                  f"phi_jump_max={r['phi_jump_max']:.3f}  NLL={r['nll_data']:,.0f}  "
                  f"({r['wall_min']:.0f}min)")
        else:
            print(f"  {r['start']:>14}: {r['status']}")


if __name__ == "__main__":
    main()
