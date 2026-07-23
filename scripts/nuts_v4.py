"""v4 확정 파라미터 NUTS — 3시즌 joint (16-17, 17-18, 19-20).

파라미터 (final_v4_figures 와 동일):
  φ 선형, γ CDC Reed, Erlang I₃, R(0)신규, κ={0.34,0.40,0} (η제거),
  presymp w=0.22 + NGM factor (w+2)/3, 관측=발현시점, baseline p=0.6,
  C(t) term↔vacation.

파라미터 블록 (10-D):
  log_R0 (3,)   season별 R0
  logit_pi (4,) shared channel mix
  log_phi_nb    NB concentration

Priors:
  log_R0[i]  ~ N(log 2.0, 0.5)
  centered logit_pi ~ N(logit_ref(π_work=0.29), σ=[0.15, 0.10, 0.05, 0.15])
  log_phi_nb ~ N(log 10, 1.5)

Stages:
  --stage smoke: 2ch seq + 4ch par × (50+50) → 벽시간 + full 예상
  --stage full : warmup 500 + sample 500 × 4 chains parallel

Immediate raw dump 이후 posterior summary.
"""
from __future__ import annotations
import os, sys, json, time, argparse, pickle, subprocess
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS","") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","2")

from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
_DEVS = jax.devices()
print(f"[early init] jax.devices() = {_DEVS}", flush=True)

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value
import arviz as az

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
import final_pipeline_confirmed as F


REPO = Path(__file__).resolve().parent.parent
ED = REPO / "outputs" / "eda"; ED.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO / "outputs" / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEAS]
N_SEA = len(SEAS)

# ★ v4 대칭 baseline: p_work = p_school = 0.6 (논문 표기; PolicyParameters
# default 1.0 대신 스크립트에서 직접 전달).  φ 선형 (v3 U-shape 아님).
PHI = np.array(F.PHI); BASE = 0.6
GAMMA = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM = np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)
KAP = np.array([0.34]*4+[0.40]*10+[0.0])
W = W_PRESYMP; NGM_F = ngm_factor(W)
SIGMA_PIN = np.array(F.SIGMA_PIN)
LOGIT_REF = np.array(F.LOGIT_REF)


def build():
    C = F.build()
    pf = np.asarray(C["shared"]["pop_15"])
    C["pf"] = pf.sum(1) if pf.ndim == 2 else pf
    M = C["shared"]
    C["ngm3"] = make_ngm_eigvalue_fn(
        pop_15=np.asarray(M["pop_15"]), rho=np.asarray(M["rho"]),
        C_home=np.asarray(M["C_home"]), C_work=np.asarray(M["C_work"]),
        C_school=np.asarray(M["C_school"]), C_other=np.asarray(M["C_other"]),
        R0_immunity=IMM, gamma=float(M["gamma"]), seasonal_factor=1.0+F.S.AMP)
    C["st"] = {}
    for s in SEAS:
        sd = estimate_initial_infected_from_hira(s, C["pf"],
            sido_codes=list(SUDOGWON_SIDO_CODES), gamma_15_assumed=GAMMA)
        C["st"][s] = jnp.asarray(_build_initial_state_with_age_seed(
            C["pf"], sd, seed_e_factor=0.5, initial_immunity=IMM,
            initial_vaccinated_fraction=0.0))
    return C


def make_joint_loss(C):
    phi_j = jnp.asarray(PHI); gamma_j = jnp.asarray(GAMMA)
    kap_j = jnp.asarray(KAP); shared = C["shared"]
    obs_l = [jnp.asarray(C["obs"][i]) for i in IDX]
    w_l = [jnp.asarray(C["w"][i]) for i in IDX]
    nw = C["nw"]
    init_states = [split_seed_to_erlang(C["st"][s]) for s in SEAS]

    def loss_joint(log_R0_vec, logit_pi, phi_nb):
        pi = jax.nn.softmax(logit_pi)
        beta_common = derive_beta_from_R0_simplex(C["ngm3"],
                        jnp.exp(log_R0_vec[0]), pi, phi_j) / NGM_F
        # per season simulate → nb_nll → sum
        total = 0.0
        for i in range(N_SEA):
            R0_s = jnp.exp(log_R0_vec[i])
            beta_s = derive_beta_from_R0_simplex(C["ngm3"], R0_s, pi, phi_j) / NGM_F
            kw = dict(shared)
            kw["beta_h"] = beta_s[0]; kw["beta_w"] = beta_s[1]
            kw["beta_s"] = beta_s[2]; kw["beta_o"] = beta_s[3]
            kw["phi_susc"] = phi_j; kw["kappa"] = kap_j
            kw["p_school"] = BASE; kw["p_work"] = BASE
            st = simulate_jax_erlang_presymp(init_states[i], w_presymp=W, **kw,
                                              discretize_time=False)
            inc = daily_new_onset_by_age_erlang_presymp(st)
            pred = simulation_to_hira_by_age_jax(inc, gamma_j, n_weeks=nw)
            total = total + nb_nll_jax(obs_l[i], pred, w_l[i],
                                        concentration=phi_nb, min_rate=0.01)
        return total

    return loss_joint


def build_model(C):
    loss_fn = make_joint_loss(C)
    logit_ref_j = jnp.asarray(LOGIT_REF); sigma_pin_j = jnp.asarray(SIGMA_PIN)
    def model():
        log_R0 = numpyro.sample("log_R0",
            dist.Normal(jnp.log(2.0)*jnp.ones(N_SEA), 0.5))
        logit_pi = numpyro.sample("logit_pi",
            dist.Uniform(jnp.full(4, -10.0), jnp.full(4, 10.0)))
        centered = logit_pi - jnp.mean(logit_pi)
        numpyro.factor("pin_prior",
            -0.5 * jnp.sum(((centered - logit_ref_j)/sigma_pin_j)**2))
        log_phi_nb = numpyro.sample("log_phi_nb", dist.Normal(jnp.log(10.0), 1.5))
        phi_nb = jnp.exp(log_phi_nb)
        numpyro.deterministic("phi_nb", phi_nb)
        numpyro.deterministic("pi", jax.nn.softmax(logit_pi))
        nll = loss_fn(log_R0, logit_pi, phi_nb)
        numpyro.factor("data_loglik", -nll)
    return model


def run_mcmc(model, chain_method, n_chains, warmup, samples, seed):
    numpyro.set_host_device_count(4)
    init = init_to_value(values=dict(
        log_R0=jnp.log(2.2)*jnp.ones(N_SEA),
        logit_pi=jnp.asarray(LOGIT_REF),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                num_chains=n_chains, chain_method=chain_method, progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(seed))
    wall = time.perf_counter() - t0
    return mcmc, wall


def quick_summary(mcmc, warmup, samples, n_chains, wall):
    samples_dict = mcmc.get_samples(group_by_chain=False)
    pi = np.asarray(samples_dict["pi"])
    log_R0 = np.asarray(samples_dict["log_R0"])
    try:
        ss = az.summary(az.from_numpyro(mcmc), round_to=4)
        rhat_max = float(ss["r_hat"].max()); ess_min = float(ss["ess_bulk"].min())
    except Exception:
        rhat_max = float("nan"); ess_min = float("nan")
    try:
        ndiv = int(mcmc.get_extra_fields()["diverging"].sum())
    except Exception:
        ndiv = -1
    total_iter = (warmup + samples) * n_chains
    return dict(
        wall_sec=wall, warmup=warmup, samples=samples, n_chains=n_chains,
        total_iter=total_iter, step_per_sec=total_iter/wall if wall > 0 else float("nan"),
        rhat_max=rhat_max, ess_min=ess_min, n_div=ndiv,
        pi_mean=[float(pi[:, k].mean()) for k in range(4)],
        pi_work_mean=float(pi[:,1].mean()),
        pi_work_q05=float(np.quantile(pi[:,1], 0.05)),
        pi_work_q95=float(np.quantile(pi[:,1], 0.95)),
        R0_mean_by_season=[float(np.exp(log_R0[:, i]).mean()) for i in range(N_SEA)],
    )


def do_smoke():
    print("="*90, flush=True)
    print(f"NUTS v4 SMOKE  seasons={SEAS}  10-D  (log_R0×3 + logit_pi×4 + log_phi_nb)",
          flush=True)
    print(f"  jax devices: {jax.device_count()}   κ={KAP.tolist()[0]:.2f}/{KAP[4]:.2f}/0  "
          f"w={W}  NGM_F={NGM_F:.4f}", flush=True)
    print(f"  σ_pin={SIGMA_PIN.tolist()}  logit_ref(π_w=0.29)={LOGIT_REF.tolist()}",
          flush=True)
    print("="*90, flush=True)
    t_ = time.perf_counter(); C = build()
    print(f"[setup] {time.perf_counter()-t_:.1f}s", flush=True)
    model = build_model(C)

    W_ = 50; S_ = 50
    print(f"\n[SEQ] 2ch × ({W_}+{S_})...", flush=True)
    mcmc_s, wall_s = run_mcmc(model, "sequential", 2, W_, S_, 0)
    seq = quick_summary(mcmc_s, W_, S_, 2, wall_s)
    print(f"  wall={seq['wall_sec']:.1f}s  step/sec={seq['step_per_sec']:.3f}  "
          f"rhat={seq['rhat_max']:.3f}  ess_min={seq['ess_min']:.1f}  div={seq['n_div']}",
          flush=True)
    print(f"  π_work mean={seq['pi_work_mean']:.4f}  [90% {seq['pi_work_q05']:.4f},{seq['pi_work_q95']:.4f}]",
          flush=True)

    print(f"\n[PAR] 4ch × ({W_}+{S_})...", flush=True)
    mcmc_p, wall_p = run_mcmc(model, "parallel", 4, W_, S_, 1)
    par = quick_summary(mcmc_p, W_, S_, 4, wall_p)
    print(f"  wall={par['wall_sec']:.1f}s  step/sec={par['step_per_sec']:.3f}  "
          f"rhat={par['rhat_max']:.3f}  ess_min={par['ess_min']:.1f}  div={par['n_div']}",
          flush=True)
    print(f"  π_work mean={par['pi_work_mean']:.4f}  [90% {par['pi_work_q05']:.4f},{par['pi_work_q95']:.4f}]",
          flush=True)

    full_iter = (500+500)*4
    full_wall = full_iter / par["step_per_sec"] if par["step_per_sec"] > 0 else float("nan")
    print(f"\n[FULL 예상] {full_iter} iter @ parallel step/sec {par['step_per_sec']:.3f}  "
          f"→ {full_wall:.0f}s = {full_wall/60:.1f} min = {full_wall/3600:.2f} h",
          flush=True)

    payload = dict(seasons=SEAS, kap=KAP.tolist(), w_presymp=W, ngm_factor=NGM_F,
                    sigma_pin=SIGMA_PIN.tolist(), pi_ref_work=0.29,
                    smoke=dict(sequential=seq, parallel=par),
                    full_estimate=dict(warmup=500, samples=500, chains=4,
                                        total_iter=full_iter,
                                        predicted_wall_sec=full_wall))
    (ED/"nuts_v4_smoke.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"[json] {ED/'nuts_v4_smoke.json'}", flush=True)
    print("[SMOKE DONE]", flush=True)


def _notify(msg):
    try:
        subprocess.run(["curl", "-s", "-d", msg, "ntfy.sh/hwcho-nuts"],
                        timeout=5, capture_output=True)
    except Exception:
        pass


WARMUP_PKL = None  # 초기화; do_full/do_resume 에서 set


def do_full(warmup, samples, n_chains, resume=False):
    global WARMUP_PKL
    WARMUP_PKL = ED / "nuts_v4_warmup_state.pkl"
    print("="*90, flush=True)
    print(f"NUTS v4 FULL  warmup={warmup} samples={samples} chains={n_chains}  "
          f"resume={resume}", flush=True)
    print(f"  jax devices: {jax.device_count()}   κ={KAP.tolist()[0]:.2f}/{KAP[4]:.2f}/0  "
          f"w={W}  NGM_F={NGM_F:.4f}", flush=True)
    print(f"  warmup_pkl: {WARMUP_PKL}", flush=True)
    print("="*90, flush=True)
    t_ = time.perf_counter(); C = build()
    print(f"[setup] {time.perf_counter()-t_:.1f}s", flush=True)
    model = build_model(C)

    # ── 2-phase: WARMUP ──
    numpyro.set_host_device_count(4)
    init = init_to_value(values=dict(
        log_R0=jnp.log(2.2)*jnp.ones(N_SEA),
        logit_pi=jnp.asarray(LOGIT_REF),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                num_chains=n_chains, chain_method="parallel", progress_bar=True)

    if resume and WARMUP_PKL.exists():
        print(f"\n[RESUME] loading warmup state from {WARMUP_PKL}", flush=True)
        with open(WARMUP_PKL, "rb") as f:
            mcmc.post_warmup_state = pickle.load(f)
        print(f"[RESUME] state loaded, skipping warmup → sampling only", flush=True)
        rng_sample = random.PRNGKey(43)
    else:
        print(f"\n[WARMUP] {n_chains}ch × {warmup} iter parallel...", flush=True)
        t_w = time.perf_counter()
        mcmc.warmup(random.PRNGKey(42), extra_fields=("diverging",))
        wall_w = time.perf_counter() - t_w
        print(f"  warmup wall={wall_w:.1f}s = {wall_w/60:.1f} min", flush=True)
        # ★ post_warmup_state 즉시 pickle 저장
        try:
            with open(WARMUP_PKL, "wb") as f:
                pickle.dump(mcmc.post_warmup_state, f)
            print(f"[warmup state saved] {WARMUP_PKL}", flush=True)
            _notify(f"NUTS v4 warmup 완료 ({wall_w/60:.1f}min), sampling 시작")
        except Exception as e:
            print(f"[warmup state save WARN] {e}", flush=True)
        rng_sample = random.PRNGKey(43)

    # ── 2-phase: SAMPLING ──
    print(f"\n[SAMPLING] {n_chains}ch × {samples} iter parallel...", flush=True)
    t_s = time.perf_counter()
    mcmc.run(rng_sample, extra_fields=("diverging",))
    wall = time.perf_counter() - t_s
    print(f"  sampling wall={wall:.1f}s = {wall/60:.1f} min", flush=True)

    # ★ 즉시 raw dump
    samples_dict = mcmc.get_samples(group_by_chain=False)
    raw_path = ED / "nuts_v4_full_raw.npz"
    np.savez(raw_path,
             log_R0=np.asarray(samples_dict["log_R0"]),
             logit_pi=np.asarray(samples_dict["logit_pi"]),
             log_phi_nb=np.asarray(samples_dict["log_phi_nb"]),
             pi=np.asarray(samples_dict["pi"]),
             phi_nb=np.asarray(samples_dict["phi_nb"]))
    print(f"[raw npz saved] {raw_path}", flush=True)

    # summary
    try:
        ss = az.summary(az.from_numpyro(mcmc), round_to=4)
        rhat_max = float(ss["r_hat"].max()); ess_min = float(ss["ess_bulk"].min())
        rhat_by = ss["r_hat"].to_dict(); ess_by = ss["ess_bulk"].to_dict()
    except Exception as e:
        print(f"[diag warn] {e}", flush=True)
        rhat_max = float("nan"); ess_min = float("nan")
        rhat_by = {}; ess_by = {}
    try:
        ndiv = int(mcmc.get_extra_fields()["diverging"].sum())
    except Exception:
        ndiv = -1

    pi = np.asarray(samples_dict["pi"])
    log_R0 = np.asarray(samples_dict["log_R0"])
    phi_nb = np.asarray(samples_dict["phi_nb"])

    def sm(x):
        return dict(mean=float(x.mean()),
                     q025=float(np.quantile(x,0.025)),
                     q05=float(np.quantile(x,0.05)),
                     q50=float(np.quantile(x,0.5)),
                     q95=float(np.quantile(x,0.95)),
                     q975=float(np.quantile(x,0.975)))

    summary = dict(
        wall_sec=wall, warmup=warmup, samples=samples, n_chains=n_chains,
        rhat_max=rhat_max, ess_min=ess_min, n_div=ndiv,
        rhat_by_param=rhat_by, ess_by_param=ess_by,
        pi=[sm(pi[:, k]) for k in range(4)],
        pi_channels=["home","work","school","other"],
        R0_by_season={SEAS[i]: sm(np.exp(log_R0[:, i])) for i in range(N_SEA)},
        phi_nb=sm(phi_nb),
    )
    (ED/"nuts_v4_full.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[summary saved] {ED/'nuts_v4_full.json'}", flush=True)

    # console short
    print("\n=== FULL POSTERIOR SUMMARY ===", flush=True)
    for k, ch in enumerate(summary["pi_channels"]):
        p = summary["pi"][k]
        print(f"  π_{ch:6s}: mean={p['mean']:.4f}  95%CI=[{p['q025']:.4f},{p['q975']:.4f}]",
              flush=True)
    for s, r in summary["R0_by_season"].items():
        print(f"  R0[{s}]: mean={r['mean']:.3f}  95%CI=[{r['q025']:.3f},{r['q975']:.3f}]",
              flush=True)
    print(f"  phi_nb: mean={summary['phi_nb']['mean']:.2f}  "
          f"95%CI=[{summary['phi_nb']['q025']:.2f},{summary['phi_nb']['q975']:.2f}]",
          flush=True)
    print(f"  r_hat max={rhat_max:.4f}  ess_min={ess_min:.1f}  div={ndiv}",
          flush=True)
    print("[FULL DONE]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke","full"], required=True)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from saved warmup state (skip warmup phase).")
    args = ap.parse_args()
    if args.stage == "smoke":
        do_smoke()
    else:
        do_full(args.warmup, args.samples, args.chains, resume=args.resume)


if __name__ == "__main__":
    main()
