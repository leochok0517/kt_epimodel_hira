"""NUTS v4 + season-specific population.

nuts_v4.py 와 사양 100% 동일하되 build() 만 시즌별 pop 반영:
  - ngm3_by_s: dict[season] -> ngm_fn (season pop 로 NGM 재빌드)
  - st_by_s : dict[season] -> initial state (season pop seed)
  - shared_by_s: dict[season] -> shared dict (pop_15 교체본)
  - loss_joint: 시즌 루프에서 시즌별 ngm/shared/init 사용

파일:
  outputs/eda/nuts_seasonpop_raw.npz (or _extended.npz)
  outputs/eda/nuts_seasonpop_full.json
  outputs/eda/nuts_seasonpop_warmup_state.pkl
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
from kt_epimodel_hira.jax_model.loss_jax import (
    simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    derive_beta_from_R0_simplex,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
import final_pipeline_confirmed as F
from season_pop_setup import build_seasonwise_setup


REPO = Path(__file__).resolve().parent.parent
ED = REPO / "outputs" / "eda"; ED.mkdir(parents=True, exist_ok=True)

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
N_SEA = len(SEAS)

PHI = np.array(F.PHI); BASE = 0.6
GAMMA = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM = np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)
KAP = np.array([0.34]*4+[0.40]*10+[0.0])
W = W_PRESYMP; NGM_F = ngm_factor(W)
SIGMA_PIN = np.array(F.SIGMA_PIN)
LOGIT_REF = np.array(F.LOGIT_REF)


def build():
    """시즌별 pop 반영 setup."""
    C_all = build_seasonwise_setup(imm=IMM, gamma_15=GAMMA, use_season_pop=True)
    return C_all


def make_joint_loss(C_all):
    phi_j = jnp.asarray(PHI); gamma_j = jnp.asarray(GAMMA)
    kap_j = jnp.asarray(KAP)
    obs_l = [jnp.asarray(C_all["obs"][F.SEASONS.index(s)]) for s in SEAS]
    w_l   = [jnp.asarray(C_all["w"][F.SEASONS.index(s)]) for s in SEAS]
    nw = C_all["nw"]
    init_states = [split_seed_to_erlang(C_all["st_by_s"][s]) for s in SEAS]
    ngm_by_s = [C_all["ngm3_by_s"][s] for s in SEAS]
    shared_by_s = [C_all["shared_by_s"][s] for s in SEAS]

    def loss_joint(log_R0_vec, logit_pi, phi_nb):
        pi = jax.nn.softmax(logit_pi)
        total = 0.0
        for i in range(N_SEA):
            R0_s = jnp.exp(log_R0_vec[i])
            beta_s = derive_beta_from_R0_simplex(ngm_by_s[i], R0_s, pi, phi_j) / NGM_F
            kw = dict(shared_by_s[i])
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


def build_model(C_all):
    loss_fn = make_joint_loss(C_all)
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


def _notify(msg):
    try:
        subprocess.run(["curl", "-s", "-d", msg, "ntfy.sh/hwcho-nuts"],
                        timeout=5, capture_output=True)
    except Exception:
        pass


def do_full(warmup, samples, n_chains, resume_from=None, seed=42, tag=""):
    """resume_from: 기존 warmup_state.pkl 경로 (v4 warmup 재사용 시 지정)."""
    print("="*90, flush=True)
    print(f"NUTS seasonpop  warmup={warmup} samples={samples} chains={n_chains}  "
          f"resume_from={resume_from}  tag={tag}", flush=True)
    print(f"  jax devices: {jax.device_count()}   κ={KAP.tolist()[0]:.2f}/{KAP[4]:.2f}/0  "
          f"w={W}  NGM_F={NGM_F:.4f}", flush=True)
    print("="*90, flush=True)
    t_ = time.perf_counter(); C_all = build()
    print(f"[setup] {time.perf_counter()-t_:.1f}s", flush=True)
    for s in SEAS:
        print(f"  {s}: pop_total={float(C_all['pop_15_by_s'][s].sum()):,.0f}", flush=True)
    model = build_model(C_all)

    numpyro.set_host_device_count(4)
    init = init_to_value(values=dict(
        log_R0=jnp.log(2.2)*jnp.ones(N_SEA),
        logit_pi=jnp.asarray(LOGIT_REF),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                num_chains=n_chains, chain_method="parallel", progress_bar=True)

    if resume_from is not None and Path(resume_from).exists():
        print(f"\n[RESUME] loading warmup state from {resume_from}", flush=True)
        with open(resume_from, "rb") as f:
            mcmc.post_warmup_state = pickle.load(f)
        print(f"[RESUME] state loaded, skipping warmup → sampling only", flush=True)
        rng_sample = random.PRNGKey(seed+1)
    else:
        WARMUP_PKL = ED / "nuts_seasonpop_warmup_state.pkl"
        print(f"\n[WARMUP] {n_chains}ch × {warmup} iter parallel...", flush=True)
        t_w = time.perf_counter()
        mcmc.warmup(random.PRNGKey(seed), extra_fields=("diverging",))
        wall_w = time.perf_counter() - t_w
        print(f"  warmup wall={wall_w:.1f}s = {wall_w/60:.1f} min", flush=True)
        try:
            with open(WARMUP_PKL, "wb") as f:
                pickle.dump(mcmc.post_warmup_state, f)
            print(f"[warmup state saved] {WARMUP_PKL}", flush=True)
            _notify(f"NUTS seasonpop warmup 완료 ({wall_w/60:.1f}min), sampling 시작")
        except Exception as e:
            print(f"[warmup state save WARN] {e}", flush=True)
        rng_sample = random.PRNGKey(seed+1)

    print(f"\n[SAMPLING] {n_chains}ch × {samples} iter parallel...", flush=True)
    t_s = time.perf_counter()
    mcmc.run(rng_sample, extra_fields=("diverging",))
    wall = time.perf_counter() - t_s
    print(f"  sampling wall={wall:.1f}s = {wall/60:.1f} min", flush=True)

    samples_dict = mcmc.get_samples(group_by_chain=False)
    suffix = f"_{tag}" if tag else ""
    raw_path = ED / f"nuts_seasonpop_raw{suffix}.npz"
    np.savez(raw_path,
             log_R0=np.asarray(samples_dict["log_R0"]),
             logit_pi=np.asarray(samples_dict["logit_pi"]),
             log_phi_nb=np.asarray(samples_dict["log_phi_nb"]),
             pi=np.asarray(samples_dict["pi"]),
             phi_nb=np.asarray(samples_dict["phi_nb"]))
    print(f"[raw npz saved] {raw_path}", flush=True)

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
        tag=tag, resumed_from=str(resume_from) if resume_from else None,
        rhat_max=rhat_max, ess_min=ess_min, n_div=ndiv,
        rhat_by_param=rhat_by, ess_by_param=ess_by,
        pi=[sm(pi[:, k]) for k in range(4)],
        pi_channels=["home","work","school","other"],
        R0_by_season={SEAS[i]: sm(np.exp(log_R0[:, i])) for i in range(N_SEA)},
        phi_nb=sm(phi_nb),
    )
    sum_path = ED / f"nuts_seasonpop_full{suffix}.json"
    sum_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[summary saved] {sum_path}", flush=True)

    print("\n=== POSTERIOR SUMMARY ===", flush=True)
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
    print(f"  r_hat max={rhat_max:.4f}  ess_min={ess_min:.1f}  div={ndiv}", flush=True)

    ok = (rhat_max <= 1.06) and (ess_min >= 90) and (ndiv == 0)
    print(f"\n[ACCEPT_GATE] rhat<=1.06 & ess>=90 & div==0 → {'PASS' if ok else 'FAIL'}",
          flush=True)
    _notify(f"NUTS seasonpop{suffix} done rhat={rhat_max:.3f} ess={ess_min:.0f} div={ndiv}")
    print("[DONE]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Path to existing warmup_state.pkl to reuse.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix appended to output files (e.g. 'reuse', 'extended').")
    args = ap.parse_args()
    do_full(args.warmup, args.samples, args.chains,
            resume_from=args.resume_from, seed=args.seed, tag=args.tag)


if __name__ == "__main__":
    main()
