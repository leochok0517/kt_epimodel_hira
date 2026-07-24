"""Sens 6: Erlang n ∈ {1, 2, 3, 5} sensitivity.

각 n 마다 w 재조정 (presymp fraction 10% 유지):
  n=1: w=1 (분리 없음, 관측=감염시점)
  n=2: w=0.111
  n=3: w=0.222 (v4 default)
  n=5: w=0.444
Posterior N=50 × 3 시즌 forward sim.
"""
from __future__ import annotations
import time, json, sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_common import (ED, SEASONS, IDX, HIRA_AGE_GROUPS,
    build_setup, PI_POST, LOG_R0_POST, KAP_DEF, PHI_DEF,
    load_partial, append_partial, key_str, ntfy, BASE, GAMMA_15,
    TERM, VAC, WH,
)
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang_n import (
    split_seed_to_n, simulate_jax_erlang_n, daily_new_onset_by_age_erlang_n,
    ngm_factor, w_for_presymp_fraction,
)
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax

NAME = "sens_erlang_n_v4"
PARTIAL = ED / f"{NAME}.partial.jsonl"; FINAL = ED / f"{NAME}.json"

N_STAGES_LIST = [1, 2, 3, 5]
PRESYMP_FRACTION = 0.10   # 목표 (n=3 → w≈0.111·2=0.222 v4)
N_POST = 50; SEED = 0


def sim_inc_n(C, s, R0, pi, n_stages, w_val,
               p_school=BASE, p_work=BASE, sch_win=WH, work_win=WH):
    """Erlang n forward + onset observation."""
    ngm_f = ngm_factor(w_val, n_stages)
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI_DEF))
    beta = b0 / ngm_f
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI_DEF); kw["kappa"] = jnp.asarray(KAP_DEF)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    initial = split_seed_to_n(C["st"][s], n_stages)
    st = simulate_jax_erlang_n(initial, n_stages=n_stages, w_presymp=w_val,
                                 **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_n(st, n_stages)


def att6(C, inc):
    return C["H"] @ np.asarray(inc).sum(0)


def evaluate_n(C, s, R0, pi, n_stages, w_val):
    """baseline + sick + school × term + vac."""
    pop6 = np.asarray(C["pop6"])
    def _sim(**pol):
        return sim_inc_n(C, s, R0, pi, n_stages, w_val, **pol)
    base_inc = _sim(p_school=BASE, p_work=BASE)
    base6 = att6(C, base_inc); tot_b = float(base6.sum())
    out = {"total_baseline": tot_b}
    for win_name, win in [("term", TERM), ("vac", VAC)]:
        for kind, kwargs in [("sick", dict(p_school=BASE, p_work=0.4, work_win=win)),
                              ("school", dict(p_school=0.4, p_work=BASE, sch_win=win))]:
            inc = _sim(**kwargs); s6 = att6(C, inc)
            av = 100.0 * (tot_b - float(s6.sum())) / max(tot_b, 1)
            d = (np.asarray(s6) - np.asarray(base6)) / pop6 * 100.0
            out[f"averted_{kind}_{win_name}"] = av
            out[f"d_attack_{kind}_by_age_{win_name}"] = {
                ag: float(d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return out


def main():
    t0 = time.time(); done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows")
    C = build_setup()
    print(f"[{NAME}] setup {time.time()-t0:.1f}s")

    # w per n
    w_by_n = {n: w_for_presymp_fraction(PRESYMP_FRACTION, n) for n in N_STAGES_LIST}
    print(f"[{NAME}] w by n: " + "  ".join(
        f"n={n}: w={w:.3f}" for n, w in w_by_n.items()))

    rng = np.random.default_rng(SEED)
    idx = rng.choice(PI_POST.shape[0], size=N_POST, replace=False)
    pi_s = PI_POST[idx]; log_R0_s = LOG_R0_POST[idx]

    ntfy(f"sens6 Erlang n 시작 ({len(N_STAGES_LIST)} n × 3시즌 × N={N_POST})")

    total = len(N_STAGES_LIST) * len(SEASONS) * N_POST; n_count = 0
    for n_stages in N_STAGES_LIST:
        w_val = w_by_n[n_stages]
        for j, s in enumerate(SEASONS):
            for k_i in range(N_POST):
                n_count += 1
                k = key_str(n_stages=n_stages, season=s, post=k_i)
                if k in done: continue
                R0 = float(np.exp(log_R0_s[k_i, j])); pi = pi_s[k_i].tolist()
                t_c = time.time()
                res = evaluate_n(C, s, R0, pi, n_stages, w_val)
                row = dict(_key=k, n_stages=n_stages, w=w_val,
                            season=s, post=k_i, R0=R0, pi=pi,
                            **{k2: v for k2, v in res.items()},
                            wall=time.time()-t_c)
                append_partial(PARTIAL, row); done[k] = row
                if n_count % 30 == 0 or n_count == total:
                    print(f"  [{n_count}/{total}] n={n_stages} w={w_val:.3f} {s} post={k_i} "
                          f"sick_t={res['averted_sick_term']:+.2f}% "
                          f"| elapsed {(time.time()-t0)/60:.1f}min", flush=True)

    rows = list(done.values())
    FINAL.write_text(json.dumps(dict(
        meta=dict(n_stages_list=N_STAGES_LIST, w_by_n=w_by_n,
                    presymp_fraction=PRESYMP_FRACTION,
                    seasons=SEASONS, n_post=N_POST, n_rows=len(rows)),
        rows=rows), indent=2, default=float))
    print(f"[{NAME}] final saved  wall={time.time()-t0:.1f}s")
    ntfy(f"sens6 Erlang n 완료 ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try: main()
    except Exception as e: ntfy(f"sens6 실패: {type(e).__name__}"); raise
