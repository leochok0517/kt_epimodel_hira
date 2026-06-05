"""L-BFGS-B × 4 seasons parallel fit (joblib + mlflow).

Generates notebooks/04_calibration_lbfgsb_multi_season.ipynb.
"""
from __future__ import annotations
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"
NB_PATH = NB_DIR / "04_calibration_lbfgsb_multi_season.ipynb"

MD_INTRO = """\
# L-BFGS-B Multi-Season Fit (21-dim, age-dependent gamma)

First production fit. 4 seasons parallel via joblib, tracked with mlflow.
- Vector: 21-dim (beta(4) + phi(14) + gamma_child/adult/elder(3))
- Seasonality: cosine fixed (amp=0.7, peak=105)
- Method: L-BFGS-B, maxiter=500
- Seasons: 2017-2018, 2018-2019, 2019-2020, 2022-2023
"""

CELL_1 = '''\
# Cell 1 — Setup + BLAS safety
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time, json, warnings
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

import mlflow
mlflow.set_tracking_uri("sqlite:///" + str(Path("../outputs/mlruns/mlflow.db").resolve()))
mlflow.set_experiment("hira_calibration_first_fit")

from kt_data import HIRA_AGE_GROUPS, SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.hira_target import (
    HIRA_GROUP_TO_NIMS_WEIGHTED, load_hira_target_by_age,
    simulation_to_hira_by_age, poisson_log_likelihood,
)
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira, simulate_aggregated,
)
from kt_epimodel_hira.calibration.loss import make_loss_function_by_age
from kt_epimodel_hira.calibration.param_vector import (
    get_bounds_vector, get_param_names, initial_guess, vector_to_params,
    params_to_vector, ParameterBounds, N_VECTOR,
)
from kt_epimodel_hira.calibration.optimizer import (
    optimize_calibration_by_age, CalibrationResult, save_result,
)
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, DiseaseParameters, ModelParameters, GAMMA_AGE_GROUPS,
)

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150,
                     "axes.unicode_minus": False, "font.family": "DejaVu Sans"})

OUTDIR = Path("../outputs/calibration"); OUTDIR.mkdir(parents=True, exist_ok=True)
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
MAX_ITER = 500
BETA_INIT = 0.06
NAMES = get_param_names()

print(f"L-BFGS-B x {len(SEASONS)} seasons, maxiter={MAX_ITER}, {N_VECTOR}-dim")
print(f"BLAS threads: OMP={os.environ.get('OMP_NUM_THREADS')}")
'''

CELL_2 = '''\
# Cell 2 — Initial vector + custom bounds
cal_init = CalibrationParameters(
    beta_h=BETA_INIT, beta_w=BETA_INIT, beta_s=BETA_INIT, beta_o=BETA_INIT,
)
init_vec = params_to_vector(cal_init)
print(f"Initial vector ({len(init_vec)}-dim):")
for n, v in zip(NAMES, init_vec):
    print(f"  {n:>18}: {v:.4f}")

custom_bounds = ParameterBounds(
    beta_h=(0.01, 0.30), beta_w=(0.01, 0.30),
    beta_s=(0.01, 0.30), beta_o=(0.01, 0.30),
    phi=(0.1, 5.0),
    gamma_child=(0.10, 0.60), gamma_adult=(0.05, 0.40), gamma_elder=(0.10, 0.50),
)
bounds_vec = get_bounds_vector(custom_bounds)
print(f"\\nBounds ({len(bounds_vec)} entries):")
for n, (lo, hi) in zip(NAMES, bounds_vec):
    print(f"  {n:>18}: [{lo:.3f}, {hi:.3f}]")
'''

CELL_3 = '''\
# Cell 3 — fit function (single season, mlflow-tracked)
def fit_single_season(season, init_vec, bounds_vec, max_iter=500):
    """Single season L-BFGS-B fit. Safe for parallel (loky)."""
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///" + str(Path("../outputs/mlruns/mlflow.db").resolve()))
        mlflow.set_experiment("hira_calibration_first_fit")

        with mlflow.start_run(run_name=f"LBFGS_{season}_age_gamma"):
            mlflow.log_params({
                "method": "L-BFGS-B", "maxiter": max_iter, "season": season,
                "dimension": len(init_vec), "seasonality": "cosine_amp0.7_peak105",
                "first_peak_only": True, "initial_immunity": 0.3,
            })

            t0 = time.perf_counter()
            result = optimize_calibration_by_age(
                season=season, method="L-BFGS-B",
                max_iterations=max_iter, initial_vec=init_vec.copy(),
                initial_immunity=0.3, verbose=False,
                first_peak_only=True, first_peak_end_week=26,
            )
            wall = time.perf_counter() - t0

            mlflow.log_metric("nll_initial", result.nll_initial)
            mlflow.log_metric("nll_final", result.nll)
            mlflow.log_metric("n_evaluations", result.n_evaluations)
            mlflow.log_metric("wall_time_sec", wall)

            cal = result.calibration
            mlflow.log_metric("beta_h", cal.beta_h)
            mlflow.log_metric("beta_w", cal.beta_w)
            mlflow.log_metric("beta_s", cal.beta_s)
            mlflow.log_metric("beta_o", cal.beta_o)
            mlflow.log_metric("gamma_child", cal.gamma_child)
            mlflow.log_metric("gamma_adult", cal.gamma_adult)
            mlflow.log_metric("gamma_elder", cal.gamma_elder)

            out_path = OUTDIR / f"calib_{season}_LBFGS_age_gamma.json"
            save_result(result, out_path)
            mlflow.log_artifact(str(out_path))

        return {"season": season, "result": result, "wall_time": wall, "error": None}

    except Exception as e:
        return {"season": season, "result": None, "wall_time": 0, "error": str(e)}

# Quick serial test: 1 season, 2 iter (verify plumbing)
test = fit_single_season(SEASONS[0], init_vec, bounds_vec, max_iter=2)
if test["error"]:
    print(f"SMOKE TEST FAILED: {test['error']}")
    raise RuntimeError(test["error"])
print(f"Smoke test OK: {test['season']} NLL={test['result'].nll:.2f} in {test['wall_time']:.1f}s")
'''

CELL_4 = '''\
# Cell 4 — Parallel L-BFGS-B x 4 seasons
print(f"Starting parallel fit: {SEASONS}, maxiter={MAX_ITER}")
t0_all = time.perf_counter()

outputs = Parallel(n_jobs=4, backend="loky", verbose=10)(
    delayed(fit_single_season)(s, init_vec, bounds_vec, MAX_ITER) for s in SEASONS
)

wall_total = time.perf_counter() - t0_all
print(f"\\n=== Total wall time: {wall_total:.1f}s ({wall_total/60:.1f} min) ===")

# Check for failures
for o in outputs:
    if o["error"]:
        print(f"FAILED: {o['season']}: {o['error']}")
    else:
        r = o["result"]
        print(f"{o['season']}: NLL {r.nll_initial:,.0f} -> {r.nll:,.0f}  "
              f"({o['wall_time']:.0f}s, {r.n_evaluations} evals, success={r.success})")
'''

CELL_5 = '''\
# Cell 5 — Results summary table
results = {o["season"]: o["result"] for o in outputs if o["result"] is not None}

print(f"{'season':>12} {'NLL_init':>12} {'NLL_final':>12} {'evals':>6} {'time_s':>7} {'success':>8}")
print("-" * 70)
for o in outputs:
    if o["result"] is None: continue
    r = o["result"]
    print(f"{o['season']:>12} {r.nll_initial:>12,.0f} {r.nll:>12,.0f} {r.n_evaluations:>6} "
          f"{o['wall_time']:>7.0f} {str(r.success):>8}")

print(f"\\n--- Fitted parameters ---")
print(f"{'season':>12} {'beta_h':>7} {'beta_w':>7} {'beta_s':>7} {'beta_o':>7} "
      f"{'g_child':>7} {'g_adult':>7} {'g_elder':>7}")
print("-" * 75)
for s, r in results.items():
    c = r.calibration
    print(f"{s:>12} {c.beta_h:>7.4f} {c.beta_w:>7.4f} {c.beta_s:>7.4f} {c.beta_o:>7.4f} "
          f"{c.gamma_child:>7.4f} {c.gamma_adult:>7.4f} {c.gamma_elder:>7.4f}")
'''

CELL_6 = '''\
# Cell 6 — R0 + age bias per season
def compute_R0_ngm(cal, disease, inputs_dict, initial_immunity=0.3):
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
    K = ((1-initial_immunity) * sf / disease.gamma) * np.diag(pop) @ np.diag(cal.phi) @ C_eff @ np.diag(1.0/N_safe)
    return float(np.max(np.real(np.linalg.eigvals(K))))

agg_inputs = build_aggregated_inputs()
dis = DiseaseParameters()

print(f"{'season':>12} {'R0':>6}")
for s, r in results.items():
    R0 = compute_R0_ngm(r.calibration, dis, agg_inputs)
    print(f"{s:>12} {R0:>6.3f}")

# Age bias per season
print(f"\\n--- Peak ratio (pred/obs) per age group ---")
print(f"{'season':>12} {'0-5':>7} {'6-11':>7} {'12-17':>7} {'18-44':>7} {'45-64':>7} {'65+':>7}")
print("-" * 65)
age_bias_all = {}
for s, r in results.items():
    target = load_hira_target_by_age(s, sido_codes=list(SUDOGWON_SIDO_CODES),
                                      first_peak_only=True, first_peak_end_week=26)
    params = ModelParameters().with_calibration(r.calibration)
    seed = estimate_initial_infected_from_hira(
        s, agg_inputs["pop_15"].flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=r.calibration.gamma_15,
    )
    sim = simulate_aggregated(
        params, agg_inputs, seed_total=float(seed.sum()),
        seed_by_age=seed, seed_e_factor=0.5,
        initial_immunity=0.3, initial_vaccinated_fraction=0.0, t_span=(0.0, 364.0),
    )
    daily_inc = sim.daily_new_infection_by_age()
    pred = simulation_to_hira_by_age(daily_inc, r.calibration.gamma_15,
                                      n_weeks=target["n_weeks"])
    ratios = {}
    row = f"{s:>12}"
    for ag in HIRA_AGE_GROUPS:
        obs_pk = float(target["hira_counts"][ag].max())
        pred_pk = float(pred[ag].max())
        ratio = pred_pk / max(obs_pk, 1)
        ratios[ag] = ratio
        row += f" {ratio:>7.2f}"
    age_bias_all[s] = ratios
    print(row)
'''

CELL_7 = '''\
# Cell 7 — Bound hits per season
print("--- Bound hits (within 2% of range) ---")
for s, r in results.items():
    vec = r.vector
    lo_arr = np.array([b[0] for b in bounds_vec])
    hi_arr = np.array([b[1] for b in bounds_vec])
    span = hi_arr - lo_arr
    hits = []
    for i, n in enumerate(NAMES):
        if (vec[i] - lo_arr[i]) / span[i] < 0.02:
            hits.append(f"{n}=LO({vec[i]:.4f})")
        elif (hi_arr[i] - vec[i]) / span[i] < 0.02:
            hits.append(f"{n}=HI({vec[i]:.4f})")
    print(f"  {s}: {', '.join(hits) if hits else '(none)'}")
'''

CELL_8 = '''\
# Cell 8 — Stability plot (phi, gamma, beta across seasons)
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
season_list = list(results.keys())

# Panel 1: phi across seasons
ax = axes[0]
phi_labels = [f"{a*5}-{a*5+4}" if a < 14 else "70+" for a in range(15) if a != 5]
for s, r in results.items():
    phi = np.ones(15)
    idx = 4
    for a in range(15):
        if a == 5: continue
        phi[a] = r.vector[idx]; idx += 1
    ax.plot(range(14), [phi[a] for a in range(15) if a != 5], "o-", label=s, ms=4)
ax.axhline(1.0, color="black", ls=":", alpha=0.5)
ax.set_xticks(range(14))
ax.set_xticklabels(phi_labels, fontsize=7, rotation=45)
ax.set_ylabel("phi_a"); ax.set_title("phi across seasons (biological -> should be stable)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: gamma
ax = axes[1]
x = np.arange(len(season_list))
w = 0.2
for i, (gname, gi) in enumerate([("child", 18), ("adult", 19), ("elder", 20)]):
    vals = [results[s].vector[gi] for s in season_list]
    ax.bar(x + (i-1)*w, vals, width=w, label=f"gamma_{gname}")
ax.set_xticks(x); ax.set_xticklabels(season_list)
ax.set_ylabel("gamma"); ax.set_title("gamma across seasons (reporting behavior -> should be stable)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# Panel 3: beta channels
ax = axes[2]
for i, bn in enumerate(["beta_h", "beta_w", "beta_s", "beta_o"]):
    vals = [results[s].vector[i] for s in season_list]
    ax.plot(season_list, vals, "o-", label=bn, ms=6)
ax.set_ylabel("beta"); ax.set_title("beta channels across seasons (may vary by strain/immunity)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTDIR / "multi_season_stability.png", dpi=150, bbox_inches="tight")
print("saved multi_season_stability.png"); plt.show()
'''

CELL_9 = '''\
# Cell 9 — Per-season fit plots (6 age panels each)
for s, r in results.items():
    target = load_hira_target_by_age(s, sido_codes=list(SUDOGWON_SIDO_CODES),
                                      first_peak_only=True, first_peak_end_week=26)
    params = ModelParameters().with_calibration(r.calibration)
    seed = estimate_initial_infected_from_hira(
        s, agg_inputs["pop_15"].flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=r.calibration.gamma_15,
    )
    sim = simulate_aggregated(
        params, agg_inputs, seed_total=float(seed.sum()),
        seed_by_age=seed, seed_e_factor=0.5,
        initial_immunity=0.3, initial_vaccinated_fraction=0.0, t_span=(0.0, 364.0),
    )
    daily_inc = sim.daily_new_infection_by_age()
    pred = simulation_to_hira_by_age(daily_inc, r.calibration.gamma_15,
                                      n_weeks=target["n_weeks"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, ag in zip(axes.flat, HIRA_AGE_GROUPS):
        obs = target["hira_counts"][ag]
        pf = pred[ag]
        ax.plot(target["week_in_season"], obs, "o", color="black", ms=3, label="Observed")
        ax.plot(np.arange(len(pf)), pf, color="#1f77b4", lw=2, label="L-BFGS-B fit")
        ax.axvspan(26, len(pf), alpha=0.08, color="gray")
        ax.set_title(f"Age {ag}"); ax.grid(alpha=0.3)
        if ax is axes.flat[0]: ax.legend(fontsize=7)
    c = r.calibration
    fig.suptitle(f"{s} L-BFGS-B (NLL {r.nll:,.0f}, {r.n_evaluations} evals)\\n"
                 f"beta=({c.beta_h:.3f},{c.beta_w:.3f},{c.beta_s:.3f},{c.beta_o:.3f}) "
                 f"g=({c.gamma_child:.3f},{c.gamma_adult:.3f},{c.gamma_elder:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"calib_{s}_LBFGS_age_gamma.png", dpi=150, bbox_inches="tight")
    print(f"saved calib_{s}_LBFGS_age_gamma.png")
    plt.show()
'''

CELL_10 = '''\
# Cell 10 — Summary + CV analysis
print("\\n" + "="*70)
print("SUMMARY")
print("="*70)

# Parameter CV across seasons
param_arrays = {n: [] for n in NAMES}
for s, r in results.items():
    for i, n in enumerate(NAMES):
        param_arrays[n].append(r.vector[i])

print(f"\\n--- Parameter CV across {len(results)} seasons ---")
print(f"{'param':>18} {'mean':>8} {'std':>8} {'CV%':>6}")
for n in NAMES:
    vals = np.array(param_arrays[n])
    mean = vals.mean(); std = vals.std()
    cv = std / max(abs(mean), 1e-10) * 100
    if cv > 20:
        flag = " <-- HIGH"
    elif cv > 10:
        flag = " <-- moderate"
    else:
        flag = ""
    print(f"{n:>18} {mean:>8.4f} {std:>8.4f} {cv:>5.1f}%{flag}")

# v4 sanity comparison (2019-2020 only)
if "2019-2020" in results:
    r19 = results["2019-2020"]
    c19 = r19.calibration
    print(f"\\n--- v4 NM sanity vs L-BFGS-B (2019-2020) ---")
    print(f"  NLL: (sanity) -> {r19.nll:,.0f}")
    if "2019-2020" in age_bias_all:
        ab = age_bias_all["2019-2020"]
        print(f"  Age bias (peak pred/obs):")
        for ag in HIRA_AGE_GROUPS:
            v4_val = {
                "0-5": 0.37, "6-11": 0.80, "12-17": 1.60,
                "18-44": 1.13, "45-64": 1.60, "65+": 2.43,
            }.get(ag, "?")
            print(f"    {ag:>6}: v4_NM={v4_val}  LBFGS={ab[ag]:.2f}")

print(f"\\nTotal parallel wall time: {wall_total:.0f}s ({wall_total/60:.1f} min)")
print(f"Sequential estimate: {sum(o['wall_time'] for o in outputs if o['result']):.0f}s")
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
    ep = ExecutePreprocessor(timeout=7200, kernel_name="kt-epimodel-hira")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})


if __name__ == "__main__":
    nb = build()
    print(f"Built {len(nb.cells)} cells. Executing (may take 30-60 min)...")
    execute(nb)
    nbformat.write(nb, NB_PATH)
    print(f"Wrote -> {NB_PATH}")
