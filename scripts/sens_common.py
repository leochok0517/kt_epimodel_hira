"""Sensitivity 공통 유틸: setup, checkpoint, 층화 forward sim primitives."""
from __future__ import annotations
import os, json, time, sys
from pathlib import Path
import numpy as np
import jax
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

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
ED = REPO / "outputs" / "eda"
FIG = REPO / "figures" / "v4" / "sensitivity"
LOG_DIR = REPO / "logs"
for d in (ED, FIG, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# v4 defaults
PHI_DEF = np.array(F.PHI)
BASE = 0.6
GAMMA_15 = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM_DEF = np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)
KAP_DEF = np.array([0.34]*4+[0.40]*10+[0.0])
W_DEF = W_PRESYMP
NGM_F_DEF = ngm_factor(W_DEF)
TERM = (70.0, 113.0); VAC = (113.0, 183.0); WH = (-1e9, 1e9)

SEASONS = ["2016-2017","2017-2018","2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEASONS]

# NUTS posterior (병합 4000 draws)
NUTS_RAW = np.load(ED / "nuts_v4_full_raw.npz")
NUTS_EXT = np.load(ED / "nuts_v4_full_extended.npz")
PI_POST = np.concatenate([NUTS_RAW["pi"], NUTS_EXT["pi"]], axis=0)
LOG_R0_POST = np.concatenate([NUTS_RAW["log_R0"], NUTS_EXT["log_R0"]], axis=0)


def build_setup(imm=None):
    """3시즌 setup. imm 지정시 R(0) 오버라이드 (초기상태 재계산)."""
    imm_use = np.asarray(imm) if imm is not None else IMM_DEF
    C = F.build()
    pf = np.asarray(C["shared"]["pop_15"])
    C["pf"] = pf.sum(1) if pf.ndim == 2 else pf
    M = C["shared"]
    C["ngm3"] = make_ngm_eigvalue_fn(
        pop_15=np.asarray(M["pop_15"]), rho=np.asarray(M["rho"]),
        C_home=np.asarray(M["C_home"]), C_work=np.asarray(M["C_work"]),
        C_school=np.asarray(M["C_school"]), C_other=np.asarray(M["C_other"]),
        R0_immunity=imm_use, gamma=float(M["gamma"]),
        seasonal_factor=1.0+F.S.AMP)
    C["st"] = {}
    for s in SEASONS:
        sd = estimate_initial_infected_from_hira(s, C["pf"],
            sido_codes=list(SUDOGWON_SIDO_CODES), gamma_15_assumed=GAMMA_15)
        C["st"][s] = jnp.asarray(_build_initial_state_with_age_seed(
            C["pf"], sd, seed_e_factor=0.5, initial_immunity=imm_use,
            initial_vaccinated_fraction=0.0))
    return C


def sim_inc(C, s, R0, pi, kap=None, phi=None, w=None,
             p_school=BASE, p_work=BASE, sch_win=WH, work_win=WH):
    """v4 forward sim.  kap/phi/w override 옵션."""
    kap_use = np.asarray(kap) if kap is not None else KAP_DEF
    phi_use = np.asarray(phi) if phi is not None else PHI_DEF
    w_use = float(w) if w is not None else W_DEF
    ngm_f = ngm_factor(w_use)
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(phi_use))
    beta = b0 / ngm_f
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(phi_use); kw["kappa"] = jnp.asarray(kap_use)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                      w_presymp=w_use, **kw,
                                      discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st)


def att6(C, inc):
    return C["H"] @ np.asarray(inc).sum(0)


def evaluate_full_stratified(C, s, R0, pi, kap=None, phi=None, w=None,
                              p_school_pol=0.4, p_work_pol=0.4,
                              p_school_base=BASE, p_work_base=BASE):
    """한 (시즌, R0, π) 조합에서 baseline + sick + school × term + vac 시뮬 → 층화 결과.

    반환: dict with keys:
      averted_sick_total_term, averted_sick_total_vac,
      averted_school_total_term, averted_school_total_vac,
      d_attack_sick_by_age_term/vac (dict age→%p),
      d_attack_school_by_age_term/vac (dict age→%p).
    """
    pop6 = np.asarray(C["pop6"])
    common = dict(C=C, s=s, R0=R0, pi=pi, kap=kap, phi=phi, w=w)
    base_inc = sim_inc(**common, p_school=p_school_base, p_work=p_work_base)
    base6 = att6(C, base_inc)
    tot_b = float(base6.sum())

    def _win_scenario(win_kind, kind):
        win = TERM if win_kind == "term" else VAC
        if kind == "sick":
            inc = sim_inc(**common, p_school=p_school_base,
                           p_work=p_work_pol, work_win=win)
        else:
            inc = sim_inc(**common, p_work=p_work_base,
                           p_school=p_school_pol, sch_win=win)
        s6 = att6(C, inc)
        av = 100.0 * (tot_b - float(s6.sum())) / max(tot_b, 1)
        d = (np.asarray(s6) - np.asarray(base6)) / pop6 * 100.0
        by_age = {ag: float(d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)}
        return av, by_age

    out = {"total_baseline": tot_b}
    for win in ("term", "vac"):
        for kind in ("sick", "school"):
            av, by_age = _win_scenario(win, kind)
            out[f"averted_{kind}_{win}"] = av
            out[f"d_attack_{kind}_by_age_{win}"] = by_age
    return out


# ────── point-fit (π_work pin) ──────
def fit_pi_pin(C, s, i, pi_work_pin, kap=None, phi=None, w=None,
                sigma_pin=None, n_starts=12, seed_offset=0):
    """β_4 point-fit with π_work pinned to given value.
    나머지 3 채널은 non-work 상대비 (A ref) 로 초기 + free logit-pi + softmax.

    반환: dict(R0, pi(4), beta(4), nll)
    """
    if sigma_pin is None:
        sigma_pin = np.array([0.15, 0.01, 0.05, 0.15])   # work strong pin
    phi_j = jnp.asarray(phi if phi is not None else PHI_DEF)
    kap_j = jnp.asarray(kap if kap is not None else KAP_DEF)
    w_use = float(w) if w is not None else W_DEF
    ngm_f = ngm_factor(w_use)
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i])

    # pin ref: 주어진 π_work + non-work A 상대비 (0.408, 0.085, 0.507)
    remain = 1.0 - pi_work_pin
    pi_ref = np.array([remain*0.408, pi_work_pin, remain*0.085, remain*0.507])
    logit_ref = np.log(np.clip(pi_ref, 1e-6, None))
    logit_ref = logit_ref - logit_ref.mean()

    def loss(x):
        R0 = jnp.exp(x[0]); pi = jax.nn.softmax(x[1:5])
        beta = derive_beta_from_R0_simplex(C["ngm3"], R0, pi, phi_j) / ngm_f
        kw = dict(C["shared"])
        kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
        kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
        kw["phi_susc"] = phi_j; kw["kappa"] = kap_j
        kw["p_school"] = BASE; kw["p_work"] = BASE
        st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                          w_presymp=w_use, **kw,
                                          discretize_time=False)
        pred = simulation_to_hira_by_age_jax(
            daily_new_onset_by_age_erlang_presymp(st),
            jnp.asarray(GAMMA_15), n_weeks=C["nw"])
        c = x[1:5] - jnp.mean(x[1:5])
        pin_pen = 0.5 * jnp.sum((c - jnp.asarray(logit_ref))**2
                                 / jnp.asarray(sigma_pin)**2)
        return nb_nll_jax(obsj, pred, wj, concentration=x[5], min_rate=0.01) + pin_pen

    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v):
            v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g

    rng = np.random.default_rng(200 + i + seed_offset)
    bounds = [F.LOG_R0_B] + [(-10,10)]*4 + [F.PHI_NB_B]
    best = None
    for k in range(n_starts):
        x0 = np.concatenate([[np.log(rng.uniform(1.8, 2.5))],
                              logit_ref + rng.normal(0, 0.5, 4), [10.0]])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    R0 = float(np.exp(x[0]))
    pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    beta = np.asarray(
        derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                     jnp.asarray(pi), phi_j)) / ngm_f
    return dict(R0=R0, pi=[float(p) for p in pi],
                 beta_4=[float(b) for b in beta], nll=float(best.fun))


# ────── Checkpoint utilities ──────
def load_partial(path):
    """jsonl 파일 → dict[key_str → row]."""
    done = {}
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                done[r["_key"]] = r
            except Exception:
                pass
    return done


def append_partial(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def key_str(**kw):
    return "|".join(f"{k}={kw[k]}" for k in sorted(kw))


def ntfy(msg):
    import subprocess
    try:
        subprocess.run(["curl", "-s", "-d", msg, "ntfy.sh/hwcho-nuts"],
                        timeout=5, capture_output=True)
    except Exception:
        pass


if __name__ == "__main__":
    print("sens_common loaded.")
    print(f"  SEASONS = {SEASONS}")
    print(f"  posterior draws: pi={PI_POST.shape}, log_R0={LOG_R0_POST.shape}")
