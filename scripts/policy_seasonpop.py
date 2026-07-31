"""S3: Phase 4 policy posterior forward (season pop).

New posterior (merged 4000) × 3 시즌 × {병가, 학교} forward with season-specific
population.  Age × window × season 층화 산출:
  - 시즌별 총 averted% (90/95 CI)
  - 연령별 Δattack (%p) + CI + 유의성 (0∉CI)
  - 절대 감염 수 (성인/아동/순/학교)
  - 아동 (0-17, 인구가중) 집계 term / vac

저장:
  outputs/eda/policy_posterior_seasonpop.json
  outputs/eda/policy_posterior_seasonpop_partial.jsonl   (resume-aware)
"""
from __future__ import annotations
import os, json, time, sys, subprocess
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"; os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.devices()
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_common import (
    ED, SEASONS, HIRA_AGE_GROUPS, PHI_DEF, KAP_DEF, GAMMA_15, IMM_DEF,
    BASE, TERM, VAC, WH,
)
from season_pop_setup import build_seasonwise_setup
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax

MERGED_NPZ = ED / "nuts_seasonpop_merged.npz"
OUT_JSON = ED / "policy_posterior_seasonpop.json"
PARTIAL = ED / "policy_posterior_seasonpop_partial.jsonl"

N_POST = 500
SEED = 2027
P_POL = 0.4
W = W_PRESYMP
NGM_F = ngm_factor(W)

ADULT_AGE = ["18-44", "45-64"]
CHILD_AGE = ["0-5", "6-11", "12-17"]
SCHOOL_AGE = ["6-11", "12-17"]


def ntfy(msg):
    try: subprocess.run(["curl","-s","-d",msg,"ntfy.sh/hwcho-nuts"],
                         timeout=5, capture_output=True)
    except Exception: pass


def load_partial():
    done = {}
    if not PARTIAL.exists(): return done
    with open(PARTIAL) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line); done[r["_key"]] = r
            except Exception: pass
    return done


def append_partial(row):
    with open(PARTIAL, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def sim_inc_season(C_s, s, R0, pi, p_school=BASE, p_work=BASE,
                    sch_win=WH, work_win=WH):
    b0 = derive_beta_from_R0_simplex(C_s["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI_DEF))
    beta = b0 / NGM_F
    kw = dict(C_s["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI_DEF); kw["kappa"] = jnp.asarray(KAP_DEF)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C_s["st"][s]),
                                      w_presymp=W, **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st)


def att6(C_s, inc):
    return C_s["H"] @ np.asarray(inc).sum(0)


def sm(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), sd=float(x.std()),
                q025=float(np.quantile(x,0.025)),
                q05=float(np.quantile(x,0.05)),
                q50=float(np.quantile(x,0.5)),
                q95=float(np.quantile(x,0.95)),
                q975=float(np.quantile(x,0.975)))


def child_agg_weighted(vals_by_age, pop6):
    """0-17 인구가중 평균 (0-4, 5-17)."""
    idx = [HIRA_AGE_GROUPS.index(a) for a in CHILD_AGE]
    ws = np.array([pop6[i] for i in idx]); ws = ws / ws.sum()
    return sum(vals_by_age[HIRA_AGE_GROUPS[i]] * w for i, w in zip(idx, ws))


def main():
    t0 = time.time()
    print("="*90); print("Phase 4: policy_seasonpop"); print("="*90)
    if not MERGED_NPZ.exists():
        raise FileNotFoundError(f"merged npz missing: {MERGED_NPZ}")

    # ── Setup ──
    C_all = build_seasonwise_setup(imm=IMM_DEF, gamma_15=GAMMA_15, use_season_pop=True)
    print(f"[setup] {time.time()-t0:.1f}s")
    C_BY_S = {}
    for s in SEASONS:
        C_BY_S[s] = dict(shared=C_all["shared_by_s"][s],
                          ngm3=C_all["ngm3_by_s"][s],
                          st={s: C_all["st_by_s"][s]},
                          H=C_all["H"], pop6=np.asarray(C_all["pop6_by_s"][s]))

    d = np.load(MERGED_NPZ)
    pi_all = np.asarray(d["pi"]); log_R0_all = np.asarray(d["log_R0"])
    N_TOT = pi_all.shape[0]
    print(f"[posterior] N_TOT={N_TOT}, using N={N_POST}")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(N_TOT, size=N_POST, replace=False)
    pi_use = pi_all[idx]; log_R0_use = log_R0_all[idx]

    done = load_partial()
    print(f"[partial] existing rows: {len(done)}")

    ages = list(HIRA_AGE_GROUPS)
    acc = {s: dict(
        sick_total_term=[], sick_total_vac=[],
        school_total_term=[], school_total_vac=[],
        sick_d_by_age_term={a: [] for a in ages},
        sick_d_by_age_vac={a: [] for a in ages},
        school_d_by_age_term={a: [] for a in ages},
        school_d_by_age_vac={a: [] for a in ages},
        sick_num_by_age_term={a: [] for a in ages},
        sick_num_by_age_vac={a: [] for a in ages},
        school_num_by_age_term={a: [] for a in ages},
        school_num_by_age_vac={a: [] for a in ages},
        baseline_total=[], baseline_by_age={a: [] for a in ages},
        R0=[], child_baseline_attack=[],
    ) for s in SEASONS}

    total = N_POST * len(SEASONS); done_cnt = 0
    ntfy(f"policy_seasonpop 시작 N={N_POST}")
    for k in range(N_POST):
        pi_k = np.asarray(pi_use[k]); R0_vec = np.exp(log_R0_use[k])
        for j, s in enumerate(SEASONS):
            key = f"post={k}|season={s}"
            done_cnt += 1
            if key in done:
                r = done[key]
                acc[s]["baseline_total"].append(r["baseline_total"])
                acc[s]["R0"].append(r["R0"])
                for a in ages:
                    acc[s]["baseline_by_age"][a].append(r["baseline_by_age"][a])
                acc[s]["sick_total_term"].append(r["sick_total_term"])
                acc[s]["sick_total_vac"].append(r["sick_total_vac"])
                acc[s]["school_total_term"].append(r["school_total_term"])
                acc[s]["school_total_vac"].append(r["school_total_vac"])
                for a in ages:
                    acc[s]["sick_d_by_age_term"][a].append(r["sick_d_term"][a])
                    acc[s]["sick_d_by_age_vac"][a].append(r["sick_d_vac"][a])
                    acc[s]["school_d_by_age_term"][a].append(r["school_d_term"][a])
                    acc[s]["school_d_by_age_vac"][a].append(r["school_d_vac"][a])
                    acc[s]["sick_num_by_age_term"][a].append(r["sick_num_term"][a])
                    acc[s]["sick_num_by_age_vac"][a].append(r["sick_num_vac"][a])
                    acc[s]["school_num_by_age_term"][a].append(r["school_num_term"][a])
                    acc[s]["school_num_by_age_vac"][a].append(r["school_num_vac"][a])
                acc[s]["child_baseline_attack"].append(r["child_baseline_attack"])
                continue

            C_s = C_BY_S[s]; pop6 = C_s["pop6"]
            R0 = float(R0_vec[j])
            base_inc = sim_inc_season(C_s, s, R0, pi_k)
            base6 = att6(C_s, base_inc)
            tot_b = float(base6.sum())

            def _run(kind, win_name):
                win = TERM if win_name == "term" else VAC
                if kind == "sick":
                    inc = sim_inc_season(C_s, s, R0, pi_k,
                                          p_work=P_POL, work_win=win)
                else:
                    inc = sim_inc_season(C_s, s, R0, pi_k,
                                          p_school=P_POL, sch_win=win)
                s6 = att6(C_s, inc)
                av = 100.0 * (tot_b - float(s6.sum())) / max(tot_b, 1.0)
                d_att = (np.asarray(s6) - np.asarray(base6)) / pop6 * 100.0
                n_av = -d_att / 100.0 * pop6
                return av, d_att, n_av

            av_s_t, d_s_t, n_s_t = _run("sick", "term")
            av_s_v, d_s_v, n_s_v = _run("sick", "vac")
            av_sc_t, d_sc_t, n_sc_t = _run("school", "term")
            av_sc_v, d_sc_v, n_sc_v = _run("school", "vac")

            base_by_age = {a: float(base6[i]) for i, a in enumerate(ages)}
            child_ba = float(sum(base6[ages.index(a)] for a in CHILD_AGE)
                             / sum(pop6[ages.index(a)] for a in CHILD_AGE) * 100.0)
            row = dict(_key=key, post=k, season=s, R0=R0,
                        baseline_total=tot_b,
                        baseline_by_age=base_by_age,
                        sick_total_term=av_s_t, sick_total_vac=av_s_v,
                        school_total_term=av_sc_t, school_total_vac=av_sc_v,
                        sick_d_term={a: float(d_s_t[i]) for i, a in enumerate(ages)},
                        sick_d_vac={a: float(d_s_v[i]) for i, a in enumerate(ages)},
                        school_d_term={a: float(d_sc_t[i]) for i, a in enumerate(ages)},
                        school_d_vac={a: float(d_sc_v[i]) for i, a in enumerate(ages)},
                        sick_num_term={a: float(n_s_t[i]) for i, a in enumerate(ages)},
                        sick_num_vac={a: float(n_s_v[i]) for i, a in enumerate(ages)},
                        school_num_term={a: float(n_sc_t[i]) for i, a in enumerate(ages)},
                        school_num_vac={a: float(n_sc_v[i]) for i, a in enumerate(ages)},
                        child_baseline_attack=child_ba)
            append_partial(row); done[key] = row

            acc[s]["baseline_total"].append(tot_b)
            for i, a in enumerate(ages):
                acc[s]["baseline_by_age"][a].append(float(base6[i]))
            acc[s]["R0"].append(R0)
            acc[s]["sick_total_term"].append(av_s_t)
            acc[s]["sick_total_vac"].append(av_s_v)
            acc[s]["school_total_term"].append(av_sc_t)
            acc[s]["school_total_vac"].append(av_sc_v)
            for i, a in enumerate(ages):
                acc[s]["sick_d_by_age_term"][a].append(float(d_s_t[i]))
                acc[s]["sick_d_by_age_vac"][a].append(float(d_s_v[i]))
                acc[s]["school_d_by_age_term"][a].append(float(d_sc_t[i]))
                acc[s]["school_d_by_age_vac"][a].append(float(d_sc_v[i]))
                acc[s]["sick_num_by_age_term"][a].append(float(n_s_t[i]))
                acc[s]["sick_num_by_age_vac"][a].append(float(n_s_v[i]))
                acc[s]["school_num_by_age_term"][a].append(float(n_sc_t[i]))
                acc[s]["school_num_by_age_vac"][a].append(float(n_sc_v[i]))
            acc[s]["child_baseline_attack"].append(child_ba)

        if (k+1) % 25 == 0 or (k+1) == N_POST:
            el = time.time() - t0
            eta = el / max(done_cnt, 1) * (total - done_cnt)
            print(f"  [{k+1}/{N_POST}] elapsed={el/60:.1f}min "
                  f"ETA={eta/60:.1f}min", flush=True)

    # Summarize
    print("\n=== SUMMARY (season × policy × window) ===")
    out = {"n_samples": N_POST, "policy_p": P_POL,
           "term_window": list(TERM), "vac_window": list(VAC),
           "seasons": SEASONS}
    for s in SEASONS:
        d = acc[s]
        pop6 = np.asarray(C_BY_S[s]["pop6"])
        # aggregate
        summary_s = dict(R0=sm(d["R0"]),
                          baseline_total=sm(d["baseline_total"]),
                          baseline_by_age={a: sm(d["baseline_by_age"][a]) for a in ages},
                          child_baseline_attack=sm(d["child_baseline_attack"]),
                          sick_total_term=sm(d["sick_total_term"]),
                          sick_total_vac=sm(d["sick_total_vac"]),
                          school_total_term=sm(d["school_total_term"]),
                          school_total_vac=sm(d["school_total_vac"]),
                          sick_d_by_age_term={a: sm(d["sick_d_by_age_term"][a]) for a in ages},
                          sick_d_by_age_vac={a: sm(d["sick_d_by_age_vac"][a]) for a in ages},
                          school_d_by_age_term={a: sm(d["school_d_by_age_term"][a]) for a in ages},
                          school_d_by_age_vac={a: sm(d["school_d_by_age_vac"][a]) for a in ages},
                          sick_num_by_age_term={a: sm(d["sick_num_by_age_term"][a]) for a in ages},
                          sick_num_by_age_vac={a: sm(d["sick_num_by_age_vac"][a]) for a in ages},
                          school_num_by_age_term={a: sm(d["school_num_by_age_term"][a]) for a in ages},
                          school_num_by_age_vac={a: sm(d["school_num_by_age_vac"][a]) for a in ages},
                          pop6={a: float(pop6[i]) for i, a in enumerate(ages)},
                          total_pop=float(pop6.sum()))
        # significance markers (0 ∉ 90% CI)
        for scen in ("sick", "school"):
            for win in ("term", "vac"):
                key = f"{scen}_d_by_age_{win}"
                sig = {a: (summary_s[key][a]["q05"] * summary_s[key][a]["q95"] > 0)
                        for a in ages}
                summary_s[f"{scen}_sig_by_age_{win}"] = sig
        # child weighted d attack
        for scen in ("sick", "school"):
            for win in ("term", "vac"):
                arr = np.array([child_agg_weighted(
                    {a: d[f"{scen}_d_by_age_{win}"][a][i] for a in ages}, pop6)
                    for i in range(len(d["R0"]))])
                summary_s[f"{scen}_child_d_{win}"] = sm(arr)
        # net absolute counts (adult sick / child school / net etc.)
        adult_idx = [ages.index(a) for a in ADULT_AGE]
        child_idx = [ages.index(a) for a in CHILD_AGE]
        for scen in ("sick", "school"):
            for win in ("term", "vac"):
                by_a = d[f"{scen}_num_by_age_{win}"]
                adult_arr = np.array([sum(by_a[ages[i]][k] for i in adult_idx)
                                       for k in range(len(d["R0"]))])
                child_arr = np.array([sum(by_a[ages[i]][k] for i in child_idx)
                                       for k in range(len(d["R0"]))])
                net_arr = np.array([sum(by_a[a][k] for a in ages)
                                     for k in range(len(d["R0"]))])
                summary_s[f"{scen}_num_adult_{win}"] = sm(adult_arr)
                summary_s[f"{scen}_num_child_{win}"] = sm(child_arr)
                summary_s[f"{scen}_num_net_{win}"] = sm(net_arr)
        out[s] = summary_s
        print(f"  {s}: sick_term={summary_s['sick_total_term']['mean']:+.2f}% "
              f"[{summary_s['sick_total_term']['q05']:+.2f},{summary_s['sick_total_term']['q95']:+.2f}]"
              f"  school_term={summary_s['school_total_term']['mean']:+.2f}% "
              f"[{summary_s['school_total_term']['q05']:+.2f},{summary_s['school_total_term']['q95']:+.2f}]")

    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[saved] {OUT_JSON}")
    ntfy(f"policy_seasonpop 완료 N={N_POST}")


if __name__ == "__main__":
    main()
