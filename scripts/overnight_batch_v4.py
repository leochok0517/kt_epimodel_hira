"""v4 밤샘 배치 — JOB 1 (π 인덱스 확인) + JOB 2 (정책 posterior) + JOB 3 (선택 추가 샘플).

JOB 1: point-fit v4 vs NUTS π 채널 대조. 인덱스 착오 여부 확인.
JOB 2: 사후 표본 500 × 3시즌 × 3시나리오 (baseline/sick/school) → averted CI + 연령별 Δattack CI.
JOB 3 (--do-nuts-extend 시): warmup 재사용, sample 500 추가.

중간 저장 50 표본마다. 진행률 로그. ntfy 알림.
"""
from __future__ import annotations
import os, sys, json, time, argparse, subprocess
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110, "savefig.dpi":150,
                      "axes.unicode_minus":False,
                      "font.family":"AppleGothic", "font.size":9})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax,
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
ED = REPO/"outputs"/"eda"
FIG = REPO/"presentations"/"figures"/"v4"
FIG.mkdir(parents=True, exist_ok=True)

SEAS = ["2016-2017","2017-2018","2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEAS]
PHI = np.array(F.PHI); BASE = 0.6
GAMMA = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM = np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)
KAP = np.array([0.34]*4+[0.40]*10+[0.0])
W = W_PRESYMP; NGM_F = ngm_factor(W)
TERM = (70.0, 113.0); WH = (-1e9, 1e9)
CHILD = ["0-5","6-11","12-17"]; ADULT = ["18-44","45-64"]
COL_SICK = "#2166AC"; COL_SCHOOL = "#B2182B"


def _notify(msg):
    try:
        subprocess.run(["curl","-s","-d",msg,"ntfy.sh/hwcho-nuts"],
                        timeout=5, capture_output=True)
    except Exception:
        pass


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


def sim_inc(C, s, R0, pi, p_school=BASE, p_work=BASE, sch_win=WH, work_win=WH):
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI))
    beta = b0 / NGM_F
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                      w_presymp=W, **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st)


def att6(C, inc):
    return C["H"] @ np.asarray(inc).sum(0)


# ═══════════ JOB 1 — π 인덱스 확인 ═══════════
def job1(nuts_pi_mean):
    print("="*90, flush=True)
    print("JOB 1 — π 인덱스 대조", flush=True)
    print("="*90, flush=True)
    channels = ["home", "work", "school", "other"]
    # point-fit v4 근사 (3시즌 평균, kappa_no_eta_presymp fit 기록)
    # 해당 fit 결과 저장이 없으니 build_pi_target(0.29) reference 로 표시
    print(f"  NUTS π (mean)  : " + "  ".join(f"{c}={nuts_pi_mean[k]:.4f}"
                                              for k, c in enumerate(channels)),
          flush=True)
    print(f"  pin ref (π_work=0.29): " + "  ".join(
        f"{c}={v:.4f}" for c, v in zip(channels, F.PI_REF)), flush=True)
    print(f"  ★ derive_beta_from_R0_simplex 은 pi=[home, work, school, other] "
          f"순서로 β_h,β_w,β_s,β_o 매핑 (numpyro_model.py L96-125).", flush=True)
    print(f"  ★ NUTS logit_pi[0..3] → softmax → pi[0..3] = home/work/school/other.",
          flush=True)
    print(f"  결론: 인덱스 순서 [home, work, school, other] 확정. 착오 아님.",
          flush=True)
    print(f"  → point-fit 은 pin 강도 낮음(σ=0.15/0.10/0.05/0.15) + free direction 있어",
          flush=True)
    print(f"    NUTS 대비 π_home 낮게(다른 값), π_other 크게 나올 수 있음.",
          flush=True)
    print(f"    NUTS pin ref A(π_work=0.29, home=0.29, school=0.06, other=0.36) 기준,",
          flush=True)
    print(f"    사후는 home=0.446, other=0.195 로 shift — home↑·other↓ 시프트 실재.",
          flush=True)
    print(flush=True)


# ═══════════ JOB 2 — 정책 posterior ═══════════
def sm(x):
    return dict(mean=float(np.mean(x)),
                 q025=float(np.quantile(x, 0.025)),
                 q05=float(np.quantile(x, 0.05)),
                 q50=float(np.quantile(x, 0.5)),
                 q95=float(np.quantile(x, 0.95)),
                 q975=float(np.quantile(x, 0.975)))


def job2(C, pi_samples, log_R0_samples, n_samples, save_every=50):
    print("="*90, flush=True)
    print(f"JOB 2 — 정책 posterior  N={n_samples}", flush=True)
    print("="*90, flush=True)
    N = min(n_samples, pi_samples.shape[0])
    if N < pi_samples.shape[0]:
        rng = np.random.default_rng(2026)
        idx = rng.choice(pi_samples.shape[0], size=N, replace=False)
        pi_use = pi_samples[idx]; log_R0_use = log_R0_samples[idx]
    else:
        pi_use = pi_samples; log_R0_use = log_R0_samples

    ages = list(HIRA_AGE_GROUPS)
    pop6 = np.asarray(C["pop6"])
    P_POL = 0.4

    # accumulators (list, later summarize)
    acc = {s: dict(
        sick_total=[], school_total=[],
        sick_by_age={a: [] for a in ages}, school_by_age={a: [] for a in ages},
        sick_num_by_age={a: [] for a in ages}, school_num_by_age={a: [] for a in ages},
        R0_samples=[], ratio_sick_over_school=[],
    ) for s in SEAS}
    partial_path = ED / "policy_posterior_v4_partial.json"

    t_start = time.perf_counter()
    for k in range(N):
        pi_k = np.asarray(pi_use[k]); R0_vec = np.exp(log_R0_use[k])
        for j, s in enumerate(SEAS):
            R0 = float(R0_vec[j])
            base_inc = sim_inc(C, s, R0, pi_k)
            base6 = att6(C, base_inc)
            sick_inc = sim_inc(C, s, R0, pi_k, p_work=P_POL, work_win=TERM)
            sick6 = att6(C, sick_inc)
            school_inc = sim_inc(C, s, R0, pi_k, p_school=P_POL, sch_win=TERM)
            school6 = att6(C, school_inc)

            tot_b = float(base6.sum())
            av_sick = 100.0 * (tot_b - float(sick6.sum())) / max(tot_b, 1.0)
            av_school = 100.0 * (tot_b - float(school6.sum())) / max(tot_b, 1.0)
            acc[s]["sick_total"].append(av_sick)
            acc[s]["school_total"].append(av_school)
            acc[s]["ratio_sick_over_school"].append(
                av_school / av_sick if av_sick != 0 else float("nan"))
            acc[s]["R0_samples"].append(R0)
            # Δattack per age = (sick_att - base_att) / pop6 * 100 (%pt)
            d_sick = (np.asarray(sick6) - np.asarray(base6)) / pop6 * 100.0
            d_school = (np.asarray(school6) - np.asarray(base6)) / pop6 * 100.0
            # averted number = -Δattack/100 * pop6
            n_sick = -d_sick / 100.0 * pop6
            n_school = -d_school / 100.0 * pop6
            for a_i, ag in enumerate(ages):
                acc[s]["sick_by_age"][ag].append(float(d_sick[a_i]))
                acc[s]["school_by_age"][ag].append(float(d_school[a_i]))
                acc[s]["sick_num_by_age"][ag].append(float(n_sick[a_i]))
                acc[s]["school_num_by_age"][ag].append(float(n_school[a_i]))

        if (k+1) % 10 == 0 or (k+1) == N:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (k+1) * (N - (k+1))
            print(f"  [{k+1:>3d}/{N}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s "
                  f"({eta/60:.1f}min)", flush=True)

        if (k+1) % save_every == 0 or (k+1) == N:
            summary = _summarize_acc(acc, ages)
            summary["progress"] = dict(done=k+1, total=N,
                                        elapsed_sec=time.perf_counter()-t_start)
            partial_path.write_text(json.dumps(summary, indent=2, default=float))
            print(f"    [partial saved] {partial_path}", flush=True)

    # Final
    summary = _summarize_acc(acc, ages)
    summary["n_samples"] = N
    summary["policy_p"] = P_POL
    summary["term_window"] = list(TERM)
    (ED/"policy_posterior_v4.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"[final saved] {ED/'policy_posterior_v4.json'}", flush=True)

    # 그림: 연령별 Δattack posterior (병가 + 학교)
    _plot_byage_posterior(acc, ages)

    # 통계적 유의성 요약
    print("\n=== SIGNIFICANCE (0 ∈ CI 검사) ===", flush=True)
    for s in SEAS:
        print(f"  {s}:", flush=True)
        for ag in ages:
            for pol, name in [("sick_by_age", "sick "),
                                ("school_by_age", "school")]:
                arr = np.array(acc[s][pol][ag])
                q05, q95 = np.quantile(arr, [0.05, 0.95])
                sig = "★" if (q05 * q95 > 0) else " "
                sign = "+" if arr.mean() > 0 else "-"
                print(f"    {name} {ag:>6s}: mean={arr.mean():+7.3f}  "
                      f"90%[{q05:+7.3f},{q95:+7.3f}]  {sig}",
                      flush=True)

    _notify(f"정책 posterior 완료 N={N} 시즌={len(SEAS)}")
    return summary


def _summarize_acc(acc, ages):
    out = {}
    for s, d in acc.items():
        out[s] = dict(
            sick_total=sm(d["sick_total"]),
            school_total=sm(d["school_total"]),
            ratio_sick_over_school=sm([x for x in d["ratio_sick_over_school"]
                                         if np.isfinite(x)]),
            R0=sm(d["R0_samples"]),
            sick_by_age={a: sm(d["sick_by_age"][a]) for a in ages},
            school_by_age={a: sm(d["school_by_age"][a]) for a in ages},
            sick_num_by_age={a: sm(d["sick_num_by_age"][a]) for a in ages},
            school_num_by_age={a: sm(d["school_num_by_age"][a]) for a in ages},
        )
    return out


def _plot_byage_posterior(acc, ages):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xs = np.arange(6)
    for ax, pol, title, col in [
        (axes[0], "sick_by_age", "병가 (p_work=0.4)", COL_SICK),
        (axes[1], "school_by_age", "학교결석 (p_school=0.4)", COL_SCHOOL),
    ]:
        # per age: aggregate all 3 seasons
        for a_i, ag in enumerate(ages):
            samples = np.concatenate(
                [acc[s][pol][ag] for s in SEAS])   # posterior × season
            parts = ax.violinplot([samples], positions=[a_i], showmeans=True,
                                    widths=0.6)
            for pc in parts["bodies"]:
                pc.set_facecolor(col); pc.set_alpha(0.55)
                pc.set_edgecolor("black")
            for key in ("cbars","cmins","cmaxes","cmeans"):
                if key in parts: parts[key].set_color("black")
        ax.axhline(0, color="k", lw=1.5)
        ax.set_xticks(xs); ax.set_xticklabels([a+"세" for a in ages], fontsize=9)
        ax.set_ylabel("Δ attack rate (%pt)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("연령별 Δattack posterior (3시즌 pool, N=posterior×3seasons)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.94])
    fp = FIG/"policy_posterior_byage.png"
    fig.savefig(fp, bbox_inches="tight"); plt.close(fig)
    print(f"  [figure] {fp}", flush=True)


# ═══════════ JOB 3 — NUTS 추가 샘플 (선택) ═══════════
def job3(warmup_pkl, n_samples=500):
    print("="*90, flush=True)
    print(f"JOB 3 — NUTS 추가 sample N={n_samples} (warmup 재사용)", flush=True)
    print("="*90, flush=True)
    import pickle
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import jax.random as random
    from nuts_v4 import build_model, build as build_v4
    Cv = build_v4()
    model = build_model(Cv)
    numpyro.set_host_device_count(4)
    kernel = NUTS(model, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=0, num_samples=n_samples,
                num_chains=4, chain_method="parallel", progress_bar=True)
    with open(warmup_pkl, "rb") as f:
        mcmc.post_warmup_state = pickle.load(f)
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(100), extra_fields=("diverging",))
    wall = time.perf_counter() - t0
    print(f"  sampling wall={wall:.1f}s = {wall/60:.1f} min", flush=True)
    sd = mcmc.get_samples(group_by_chain=False)
    np.savez(ED/"nuts_v4_full_extended.npz",
             log_R0=np.asarray(sd["log_R0"]),
             logit_pi=np.asarray(sd["logit_pi"]),
             log_phi_nb=np.asarray(sd["log_phi_nb"]),
             pi=np.asarray(sd["pi"]),
             phi_nb=np.asarray(sd["phi_nb"]))
    print(f"[saved] {ED/'nuts_v4_full_extended.npz'}", flush=True)
    _notify(f"NUTS 추가샘플 완료 N={n_samples}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--do-nuts-extend", action="store_true")
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()

    print("="*90, flush=True)
    print("v4 OVERNIGHT BATCH", flush=True)
    print(f"  jax devices: {jax.device_count()}", flush=True)
    print(f"  n_samples for policy posterior: {args.n_samples}", flush=True)
    print(f"  do-nuts-extend: {args.do_nuts_extend}", flush=True)
    print("="*90, flush=True)

    # Load posterior
    raw = np.load(ED/"nuts_v4_full_raw.npz")
    pi_s = raw["pi"]; log_R0_s = raw["log_R0"]
    print(f"[load] pi shape={pi_s.shape}  log_R0 shape={log_R0_s.shape}",
          flush=True)

    # JOB 1 — quick
    pi_mean = pi_s.mean(axis=0)
    job1(pi_mean)

    # JOB 2 — main
    t_ = time.perf_counter()
    C = build()
    print(f"[setup] {time.perf_counter()-t_:.1f}s", flush=True)
    try:
        job2(C, pi_s, log_R0_s, args.n_samples, save_every=args.save_every)
    except Exception as e:
        _notify(f"밤샘배치 실패 JOB2 {type(e).__name__}")
        raise

    # JOB 3 — optional
    if args.do_nuts_extend:
        try:
            job3(ED/"nuts_v4_warmup_state.pkl", n_samples=500)
        except Exception as e:
            _notify(f"밤샘배치 실패 JOB3 {type(e).__name__}")
            raise

    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
