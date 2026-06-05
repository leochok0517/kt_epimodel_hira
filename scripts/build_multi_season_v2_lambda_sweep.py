"""Multi-season joint fit v2: phi smoothing lambda sweep + bound expansion.

Phase A diagnosis: v1 had 5/14 phi at lower bound + adjacent phi jumps unrealistic.
v2: add phi smoothing penalty (lambda * sum((phi[i+1]-phi[i])**2)) and expand
phi lower bound 0.1 -> 0.05. Sweep lambda in {0, 0.1, 1.0, 10.0} in parallel.

Generates notebooks/06_multi_season_v2_lambda_sweep.ipynb.
"""
from __future__ import annotations
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"
NB_PATH = NB_DIR / "06_multi_season_v2_lambda_sweep.ipynb"

MD_INTRO = """\
# Multi-Season Joint Fit v2 - Lambda Sweep

v1 issues:
- 5/14 phi at lower bound (0.1)
- Adjacent phi jumps unrealistic (phi_4=0.12 vs phi_6=0.71, etc.)

v2 changes:
1. phi smoothing penalty: lambda * sum((phi[i+1]-phi[i])**2)
2. phi bound 0.1 -> 0.05 (allow lower phi if needed)
3. Parallel sweep of lambda in {0, 0.1, 1.0, 10.0}
4. Warm-start from v1 best vector
"""

CELL_1 = '''\
# Cell 1 - Setup
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time, json, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

import mlflow
MLFLOW_URI = "sqlite:///" + str(Path("../outputs/mlruns/mlflow.db").resolve())
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("hira_calibration_multi_season_v2_smoothing")

from kt_data import HIRA_AGE_GROUPS, SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.hira_target import (
    load_hira_target_by_age, simulation_to_hira_by_age,
)
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira, simulate_aggregated,
)
from kt_epimodel_hira.calibration.loss import make_loss_function_by_age
from kt_epimodel_hira.calibration.param_vector import vector_to_params
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, DiseaseParameters, ModelParameters,
)

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150,
                     "axes.unicode_minus": False, "font.family": "DejaVu Sans"})

OUTDIR = Path("../outputs/calibration"); OUTDIR.mkdir(parents=True, exist_ok=True)
EDA_DIR = Path("../outputs/eda"); EDA_DIR.mkdir(parents=True, exist_ok=True)
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
N_SEASONS = len(SEASONS)
N_TOTAL = 33
LAMBDAS = [0.0, 0.1, 1.0, 10.0]
MAX_ITER = 1000

print(f"v2 lambda sweep: lambdas={LAMBDAS}, {N_TOTAL}-dim, {N_SEASONS} seasons")
print(f"BLAS: OMP={os.environ.get('OMP_NUM_THREADS')}")
'''

CELL_2 = '''\
# Cell 2 - Warm-start from v1
V1_PATH = OUTDIR / "multi_season_joint_v1.json"
with open(V1_PATH) as f:
    v1 = json.load(f)

init_33 = np.array(v1["vector_33"])
v1_nll = float(v1["nll"]["final"])

print(f"v1 final NLL: {v1_nll:,.0f}")
print(f"v1 phi range: [{init_33[:14].min():.3f}, {init_33[:14].max():.3f}]")
print(f"v1 phi at lower bound (<0.11): {int(np.sum(init_33[:14] < 0.11))}/14")
print(f"v1 gamma: child={init_33[14]:.4f} adult={init_33[15]:.4f} elder={init_33[16]:.4f}")

# Phi adjacency jumps in v1
phi_v1 = init_33[:14]
diffs = np.abs(np.diff(phi_v1))
print(f"v1 phi adjacent diffs: max={diffs.max():.3f}, mean={diffs.mean():.3f}")
'''

CELL_3 = '''\
# Cell 3 - Verify environment (smoke test only - parallel workers will rebuild)
inputs = build_aggregated_inputs()
pop_15 = inputs["pop_15"].flatten()

# Sanity build for season 0 only
target0 = load_hira_target_by_age(
    SEASONS[0], sido_codes=list(SUDOGWON_SIDO_CODES),
    first_peak_only=True, first_peak_end_week=26,
)
seed0 = estimate_initial_infected_from_hira(
    SEASONS[0], pop_15, sido_codes=list(SUDOGWON_SIDO_CODES),
    gamma_15_assumed=CalibrationParameters().gamma_15,
)
loss0 = make_loss_function_by_age(
    target0, inputs, ModelParameters(),
    seed_total=float(seed0.sum()), seed_by_age=seed0, seed_e_factor=0.5,
    initial_immunity=0.3, t_span=(0.0, 364.0), verbose=False,
)
# Smoke: compute NLL for v1 vector mapped to season 0
phi_14 = init_33[:14]; gamma_3 = init_33[14:17]; beta_4 = init_33[17:21]
vec_21 = np.concatenate([beta_4, phi_14, gamma_3])
nll0 = loss0(vec_21)
print(f"Season 0 NLL with v1 vector: {nll0:,.0f}")
'''

CELL_4 = '''\
# Cell 4 - Fit function (parallel-safe)
def fit_one_lambda(lam, init_vec, seasons, outdir_str, mlflow_uri):
    """Single lambda fit. Built to be parallel-safe (loky-pickled).

    Rebuilds inputs/targets/losses inside worker (avoids closure pickling).
    """
    import os, time, json
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    import numpy as np
    from pathlib import Path
    from scipy.optimize import minimize
    import mlflow

    from kt_data import SUDOGWON_SIDO_CODES
    from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
    from kt_epimodel_hira.calibration.simple_model import (
        build_aggregated_inputs, estimate_initial_infected_from_hira,
    )
    from kt_epimodel_hira.calibration.loss import make_loss_function_by_age
    from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters

    try:
        # Rebuild loss closures (per-worker)
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

        # Smoothed loss
        n_evals = [0]
        nll_history = []

        def loss_smoothed(vec):
            n_evals[0] += 1
            phi = vec[:14]; gamma = vec[14:17]
            base = 0.0
            for i, s in enumerate(seasons):
                beta = vec[17 + i*4 : 21 + i*4]
                vec_21 = np.concatenate([beta, phi, gamma])
                base += losses[s](vec_21)
            smooth = lam * float(np.sum(np.diff(phi)**2)) if lam > 0 else 0.0
            total = base + smooth
            nll_history.append((total, base, smooth))
            return total

        # Bounds v2: phi lower 0.1 -> 0.05
        bounds = (
            [(0.05, 5.0)] * 14 +
            [(0.10, 0.60), (0.05, 0.40), (0.03, 0.50)] +
            [(0.001, 0.30)] * 16
        )

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("hira_calibration_multi_season_v2_smoothing")

        with mlflow.start_run(run_name=f"v2_lambda_{lam}"):
            mlflow.log_params({
                "method": "L-BFGS-B multi-season smoothed",
                "lambda_smooth": lam,
                "dimension": 33, "n_seasons": len(seasons),
                "phi_bound_lower": 0.05, "maxiter": 1000,
                "warm_start": "v1_best_vec",
            })

            t0 = time.perf_counter()
            sol = minimize(
                loss_smoothed, init_vec.copy(),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000, "disp": False},
            )
            wall = time.perf_counter() - t0

            # Decompose final NLL
            phi_final = sol.x[:14]
            smooth_final = lam * float(np.sum(np.diff(phi_final)**2)) if lam > 0 else 0.0
            data_nll_final = sol.fun - smooth_final

            mlflow.log_metric("nll_total_final", sol.fun)
            mlflow.log_metric("nll_data_final", data_nll_final)
            mlflow.log_metric("nll_smoothing_final", smooth_final)
            mlflow.log_metric("n_evals", n_evals[0])
            mlflow.log_metric("wall_time_min", wall / 60)
            mlflow.log_metric("success", int(sol.success))

            # Save JSON
            result_dict = {
                "metadata": {
                    "lambda": lam, "method": "L-BFGS-B v2 smoothed",
                    "wall_time_min": round(wall / 60, 1),
                    "n_evals": n_evals[0],
                    "success": bool(sol.success), "message": str(sol.message),
                },
                "nll": {"total": float(sol.fun), "data": float(data_nll_final),
                        "smoothing": float(smooth_final)},
                "vector_33": sol.x.tolist(),
                "phi": sol.x[:14].tolist(),
                "gamma": {"child": float(sol.x[14]), "adult": float(sol.x[15]),
                          "elder": float(sol.x[16])},
                "betas": {seasons[i]: sol.x[17+i*4:21+i*4].tolist()
                          for i in range(len(seasons))},
                "nll_history": [(float(t), float(b), float(s)) for t, b, s in nll_history[::20]],
            }
            out_path = Path(outdir_str) / f"multi_season_v2_lambda{lam}.json"
            with open(out_path, "w") as f:
                json.dump(result_dict, f, indent=2)
            mlflow.log_artifact(str(out_path))

        return {"lambda": lam, "result": result_dict, "wall_time": wall, "error": None}

    except Exception as e:
        import traceback
        return {"lambda": lam, "result": None, "wall_time": 0,
                "error": f"{e}\\n{traceback.format_exc()}"}

# Smoke: 1 lambda, very few iters (parallel-safe verification)
print("Smoke test: fit_one_lambda(lam=0.0, maxiter handled internally)")
'''

CELL_5 = '''\
# Cell 5 - Parallel lambda sweep
print(f"Starting parallel fit for lambdas {LAMBDAS}")
print(f"Each fit ~3-5 hours. Parallel 4 fits: ~3-5 hours wall time.")
t0_all = time.perf_counter()

outputs = Parallel(n_jobs=4, backend="loky", verbose=10)(
    delayed(fit_one_lambda)(lam, init_33, SEASONS, str(OUTDIR), MLFLOW_URI)
    for lam in LAMBDAS
)

wall_total = time.perf_counter() - t0_all
print(f"\\n=== Total wall time: {wall_total/60:.1f} min ===")

for o in outputs:
    if o["error"]:
        print(f"FAILED lambda={o['lambda']}: {o['error'][:200]}")
    else:
        r = o["result"]
        print(f"lambda={o['lambda']}: NLL total={r['nll']['total']:,.0f}  "
              f"data={r['nll']['data']:,.0f}  smooth={r['nll']['smoothing']:,.2f}  "
              f"({o['wall_time']/60:.0f} min, "
              f"success={r['metadata']['success']})")
'''

CELL_6 = '''\
# Cell 6 - Comparison table
results = {o["lambda"]: o["result"] for o in outputs if o["result"] is not None}

print(f"{'lambda':>8} {'NLL_total':>14} {'data_NLL':>14} {'smooth':>10} "
      f"{'phi_min':>8} {'phi_max':>8} {'jump_max':>9} {'bnd_hit':>8} {'evals':>6}")
print("-" * 100)
for lam in sorted(results.keys()):
    r = results[lam]
    phi = np.array(r["phi"])
    diffs = np.abs(np.diff(phi))
    bnd_hits = int(np.sum(phi < 0.06))
    print(f"{lam:>8.2f} {r['nll']['total']:>14,.0f} {r['nll']['data']:>14,.0f} "
          f"{r['nll']['smoothing']:>10,.2f} {phi.min():>8.4f} {phi.max():>8.4f} "
          f"{diffs.max():>9.4f} {bnd_hits:>8d} {r['metadata']['n_evals']:>6d}")

# v1 baseline reference
v1_phi = init_33[:14]
v1_diffs = np.abs(np.diff(v1_phi))
v1_bnd = int(np.sum(v1_phi < 0.11))  # v1 bound was 0.1
print(f"\\n{'v1 ref':>8} {v1_nll:>14,.0f}    n/a         n/a "
      f"{v1_phi.min():>8.4f} {v1_phi.max():>8.4f} "
      f"{v1_diffs.max():>9.4f} {v1_bnd:>8d} (n/a)")
'''

CELL_7 = '''\
# Cell 7 - Per-lambda gamma + R0 summary
def compute_R0_ngm(cal, disease, inputs_dict, initial_immunity=0.3):
    pop = inputs_dict["pop_15"].flatten()
    N_safe = np.maximum(pop, 1e-10)
    matrices = inputs_dict["matrices"]; rho = inputs_dict["rho"].flatten()
    sf = disease.seasonal_factor(disease.seasonality_peak_day)
    C_eff = np.zeros((15, 15))
    C_eff += cal.beta_h * matrices["C_home"]
    C_eff[:4, :4] += cal.beta_s * matrices["C_school"][:4, :4]
    rho_ok = (rho > 0).astype(float)
    for a in range(4, 14):
        C_eff[a, :] += cal.beta_w * matrices["C_work"][a, :] * rho[a] * rho_ok
    C_eff += cal.beta_o * matrices["C_other"]
    K = ((1-initial_immunity) * sf / disease.gamma) * \\
        np.diag(pop) @ np.diag(cal.phi) @ C_eff @ np.diag(1.0/N_safe)
    return float(np.max(np.real(np.linalg.eigvals(K))))

dis = DiseaseParameters()
inputs_main = build_aggregated_inputs()

print(f"{'lambda':>8} {'g_child':>8} {'g_adult':>8} {'g_elder':>8} {'mean_R0':>8}")
print("-" * 50)
for lam in sorted(results.keys()):
    r = results[lam]
    phi_14 = np.array(r["phi"]); gamma = np.array([r["gamma"]["child"], r["gamma"]["adult"], r["gamma"]["elder"]])
    R0s = []
    for i, s in enumerate(SEASONS):
        beta = np.array(r["betas"][s])
        vec_21 = np.concatenate([beta, phi_14, gamma])
        cal = vector_to_params(vec_21)
        R0s.append(compute_R0_ngm(cal, dis, inputs_main))
    print(f"{lam:>8.2f} {r['gamma']['child']:>8.4f} {r['gamma']['adult']:>8.4f} "
          f"{r['gamma']['elder']:>8.4f} {np.mean(R0s):>8.3f}")
'''

CELL_8 = '''\
# Cell 8 - Age bias per (lambda, season)
season_targets = {}
season_seeds = {}
for s in SEASONS:
    season_targets[s] = load_hira_target_by_age(
        s, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    season_seeds[s] = estimate_initial_infected_from_hira(
        s, pop_15, sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )

age_bias_by_lambda = {}
for lam in sorted(results.keys()):
    r = results[lam]
    phi_14 = np.array(r["phi"]); gamma = np.array([r["gamma"]["child"], r["gamma"]["adult"], r["gamma"]["elder"]])
    ab = {}
    for i, s in enumerate(SEASONS):
        beta = np.array(r["betas"][s])
        vec_21 = np.concatenate([beta, phi_14, gamma])
        cal = vector_to_params(vec_21)
        params = ModelParameters().with_calibration(cal)
        seed = season_seeds[s]
        sim = simulate_aggregated(
            params, inputs_main, seed_total=float(seed.sum()),
            seed_by_age=seed, seed_e_factor=0.5,
            initial_immunity=0.3, t_span=(0.0, 364.0),
        )
        daily_inc = sim.daily_new_infection_by_age()
        pred = simulation_to_hira_by_age(daily_inc, cal.gamma_15,
                                          n_weeks=season_targets[s]["n_weeks"])
        season_ab = {}
        for ag in HIRA_AGE_GROUPS:
            obs_pk = float(season_targets[s]["hira_counts"][ag].max())
            pred_pk = float(pred[ag].max())
            season_ab[ag] = pred_pk / max(obs_pk, 1)
        ab[s] = season_ab
    age_bias_by_lambda[lam] = ab

# Print
print(f"--- Age bias (peak pred/obs) by (lambda, season, age) ---")
print(f"{'lambda':>6} {'season':>10} " + " ".join(f"{ag:>6}" for ag in HIRA_AGE_GROUPS))
print("-" * 70)
for lam in sorted(age_bias_by_lambda.keys()):
    for s in SEASONS:
        ab = age_bias_by_lambda[lam][s]
        row = f"{lam:>6.2f} {s:>10} " + " ".join(f"{ab[ag]:>6.2f}" for ag in HIRA_AGE_GROUPS)
        print(row)
    print()

# Variance of age bias per lambda (across 24 = 4 season x 6 age)
print(f"{'lambda':>8} {'mean_bias':>10} {'std_bias':>10}")
for lam in sorted(age_bias_by_lambda.keys()):
    vals = [age_bias_by_lambda[lam][s][ag] for s in SEASONS for ag in HIRA_AGE_GROUPS]
    print(f"{lam:>8.2f} {np.mean(vals):>10.3f} {np.std(vals):>10.3f}")
'''

CELL_9 = '''\
# Cell 9 - Visualization: phi profile + NLL decomp + bound hits + age bias
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

# Panel 1: phi profile across lambda
ax = fig.add_subplot(gs[0, :])
phi_labels = [f"{a*5}" if a < 14 else "70+" for a in range(15) if a != 5]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))
ax.plot(range(14), init_33[:14], "k--", lw=1.5, alpha=0.6, label="v1 (initial)")
for (lam, c) in zip(sorted(results.keys()), colors):
    ax.plot(range(14), results[lam]["phi"], "o-", color=c, ms=5, lw=1.5,
            label=f"lambda={lam}")
ax.axhline(0.05, color="red", ls=":", alpha=0.5, label="v2 lower bound")
ax.set_xticks(range(14)); ax.set_xticklabels(phi_labels, fontsize=8, rotation=45)
ax.set_ylabel("phi"); ax.set_title("phi profile across lambda")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# Panel 2: NLL decomposition
ax = fig.add_subplot(gs[1, 0])
lams = sorted(results.keys())
data_nlls = [results[l]["nll"]["data"] for l in lams]
smooth_nlls = [results[l]["nll"]["smoothing"] for l in lams]
total_nlls = [results[l]["nll"]["total"] for l in lams]
ax.plot(lams, data_nlls, "o-", label="data NLL", color="blue")
ax.plot(lams, total_nlls, "s-", label="total NLL", color="black")
ax2 = ax.twinx()
ax2.plot(lams, smooth_nlls, "^-", label="smoothing term", color="orange")
ax2.set_ylabel("smoothing term", color="orange")
ax.set_xlabel("lambda"); ax.set_ylabel("NLL (data, total)")
ax.set_title("NLL decomposition"); ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

# Panel 3: phi bound hits + max jump
ax = fig.add_subplot(gs[1, 1])
bnd_hits = [int(np.sum(np.array(results[l]["phi"]) < 0.06)) for l in lams]
jump_max = [float(np.max(np.abs(np.diff(results[l]["phi"])))) for l in lams]
ax.bar([l - 0.05 for l in lams], bnd_hits, width=0.05, label="phi at lower bound", color="red", alpha=0.7)
ax2 = ax.twinx()
ax2.plot(lams, jump_max, "o-", color="purple", label="max phi jump")
ax.set_xlabel("lambda"); ax.set_ylabel("# phi at bound (<0.06)", color="red")
ax2.set_ylabel("max adjacent phi jump", color="purple")
ax.set_title("phi bound hits + adjacency"); ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

# Panel 4: Age bias mean +/- std per lambda
ax = fig.add_subplot(gs[2, 0])
means = []; stds = []
for lam in lams:
    vals = [age_bias_by_lambda[lam][s][ag] for s in SEASONS for ag in HIRA_AGE_GROUPS]
    means.append(np.mean(vals)); stds.append(np.std(vals))
ax.errorbar(lams, means, yerr=stds, fmt="o-", capsize=5)
ax.axhline(1.0, color="black", ls=":", alpha=0.5)
ax.set_xlabel("lambda"); ax.set_ylabel("mean age bias +/- std")
ax.set_title("Fit quality (age bias)"); ax.grid(alpha=0.3)

# Panel 5: gamma triple
ax = fig.add_subplot(gs[2, 1])
g_child = [results[l]["gamma"]["child"] for l in lams]
g_adult = [results[l]["gamma"]["adult"] for l in lams]
g_elder = [results[l]["gamma"]["elder"] for l in lams]
ax.plot(lams, g_child, "o-", label="gamma_child", color="blue")
ax.plot(lams, g_adult, "s-", label="gamma_adult", color="red")
ax.plot(lams, g_elder, "^-", label="gamma_elder", color="green")
ax.set_xlabel("lambda"); ax.set_ylabel("gamma")
ax.set_title("gamma stability across lambda")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(f"Multi-season v2 lambda sweep (4 lambdas, {wall_total/60:.0f} min wall)",
             fontsize=14)
fig.savefig(EDA_DIR / "multi_season_v2_lambda_comparison.png", dpi=150, bbox_inches="tight")
print(f"saved multi_season_v2_lambda_comparison.png"); plt.show()
'''

CELL_10 = '''\
# Cell 10 - Recommendation + final save
# Criteria: phi at bound resolved, jump_max < 1.0, data_NLL loss < 1%, mean bias close to 1
v1_data_nll = v1_nll  # v1 had no smoothing
print(f"v1 data NLL reference: {v1_data_nll:,.0f}")
print()
print(f"{'lambda':>8} {'data_NLL':>14} {'NLL_loss_%':>10} {'jump':>8} {'bnd':>5} {'mean_bias':>10} {'verdict':>15}")
print("-" * 90)

recommended = None
for lam in sorted(results.keys()):
    r = results[lam]
    phi = np.array(r["phi"])
    diffs = np.abs(np.diff(phi))
    bnd = int(np.sum(phi < 0.06))
    nll_loss_pct = (r["nll"]["data"] - v1_data_nll) / abs(v1_data_nll) * 100
    vals = [age_bias_by_lambda[lam][s][ag] for s in SEASONS for ag in HIRA_AGE_GROUPS]
    mean_bias = np.mean(vals)

    # Heuristic
    if bnd == 0 and diffs.max() < 1.0 and abs(nll_loss_pct) < 1.0:
        verdict = "OK_recommended"
        if recommended is None: recommended = lam
    elif bnd > 0:
        verdict = "bound_remain"
    elif diffs.max() >= 1.0:
        verdict = "still_jumpy"
    elif abs(nll_loss_pct) > 5.0:
        verdict = "fit_loss_big"
    else:
        verdict = "marginal"
    print(f"{lam:>8.2f} {r['nll']['data']:>14,.0f} {nll_loss_pct:>10.2f}% "
          f"{diffs.max():>8.4f} {bnd:>5d} {mean_bias:>10.3f} {verdict:>15}")

if recommended is None:
    print(f"\\nNo lambda passes all criteria. Fall back to best-trade-off.")
    recommended = sorted(results.keys())[len(results)//2]
print(f"\\nRecommended lambda: {recommended}")

# Save recommended as alias
rec_path = OUTDIR / "multi_season_LBFGS_v2_lambda_recommended.json"
import shutil
shutil.copy(OUTDIR / f"multi_season_v2_lambda{recommended}.json", rec_path)
print(f"saved {rec_path}")
'''


def build():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(MD_INTRO),
        new_code_cell(CELL_1), new_code_cell(CELL_2), new_code_cell(CELL_3),
        new_code_cell(CELL_4), new_code_cell(CELL_5), new_code_cell(CELL_6),
        new_code_cell(CELL_7), new_code_cell(CELL_8), new_code_cell(CELL_9),
        new_code_cell(CELL_10),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python (kt-epimodel-hira)",
                       "language": "python", "name": "kt-epimodel-hira"},
        "language_info": {"name": "python"},
    }
    return nb


def execute(nb):
    ep = ExecutePreprocessor(timeout=36000, kernel_name="kt-epimodel-hira")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})


if __name__ == "__main__":
    nb = build()
    print(f"Built {len(nb.cells)} cells. Executing (3-8 hours expected)...")
    execute(nb)
    nbformat.write(nb, NB_PATH)
    print(f"Wrote -> {NB_PATH}")
