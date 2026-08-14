"""Reproduction number analysis — (A) channel R_c, (B) age R_b, (C) R_e(t).

기존 posterior draws (nuts_seasonpop_merged) 을 사용해 재fit 없이 forward
계산.  대표 시즌 2019-2020.  season pop.

Model (Diekmann NGM):
  K = (sf/γ) · diag(pop · φ · S/N) · C_eff · diag(1/N)
  C_eff = Σ_c β_c · C_c(t) · [c-specific mask]
  (mask: school → [0:4]×[0:4], work → rho·work_mask row × rho_ok col,
   home/other → all)

(A) 채널별 스펙트럴 기여:  좌/우 dominant eigenvectors u, v
    R_c = uᵀ K_c v / (uᵀ v),   Σ R_c = R0.
(B) 연령별 기여 (열별 감염 생성):  R_b = (uᵀ K[:, b]) v_b / (uᵀ v).
(C) 시점별 R_e(t):  baseline / sick / school 3 시나리오 forward → S_a(t)
    → 매주 K_eff(t) rebuild → dominant eigenvalue.

Saves:
  outputs/eda/reproduction_numbers.json
  figures/paper_seasonpop/{pdf,png}/repro_channel.{pdf,png}
  figures/paper_seasonpop/{pdf,png}/repro_byage.{pdf,png}
  figures/paper_seasonpop/{pdf,png}/repro_Re_time.{pdf,png}
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (savefig, COL_SICK, COL_SCHOOL, COL_SEASON, COL_ZERO,
                       AGES, W_DOUBLE)
from season_pop_setup import build_seasonwise_setup
from kt_epimodel_hira.jax_model.numpyro_model import (
    derive_beta_from_R0_simplex,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, split_seed_to_erlang, ngm_factor, W_PRESYMP,
    E_S, E_V, E_E, E_I1, E_I2, E_I3,
)
from kt_epimodel_hira.jax_model.foi_jax import (
    seasonal_factor_cosine, vacation_weight,
)
import final_pipeline_confirmed as F

REPO = Path(__file__).resolve().parent.parent
ED = REPO / "outputs" / "eda"
MERGED = ED / "nuts_seasonpop_merged.npz"
OUT_JSON = ED / "reproduction_numbers.json"

# Constants
S_REP = "2019-2020"
PHI = np.array(F.PHI); BASE = 0.6
GAMMA_15 = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM = np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)
KAP = np.array([0.34]*4+[0.40]*10+[0.0])
W = W_PRESYMP; NGM_F = ngm_factor(W)
TERM = (70.0, 113.0); VAC = (113.0, 183.0); WH = (-1e9, 1e9)

N_POST = 100         # posterior subsample
SEED = 1234
SEASONAL_AMP = F.S.AMP
SEASONAL_BASE = 1.0       # matches disease.seasonality_base (ODE default)
SEASONAL_PEAK = 105.0
SEASONAL_PERIOD = 365.0
SF_REF = SEASONAL_BASE + SEASONAL_AMP   # 1.9 — reference used by static NGM

# NIMS 15 → HIRA 6 age aggregation weights (population-weighted, but here just
# structural index bins).  For display grouping only:
AGE6_BINS = {"0-5":  [0,1],           # 0-4, 5-9 → 0-5
             "6-11": [2],             # 10-14 (approx; matches HIRA overlap)
             "12-17":[3],
             "18-44":[4,5,6,7,8],
             "45-64":[9,10,11,12],
             "65+":  [13,14]}
AGE15_LAB = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
              "40-44","45-49","50-54","55-59","60-64","65-69","70+"]


# ═══════════════════════════════════════════════════════════════════════════
# NGM builder — per channel + total, on numpy for eigen convenience
# ═══════════════════════════════════════════════════════════════════════════
def build_K_channels(*, pop_15, rho_flat, C_home, C_school, C_work, C_other,
                      beta_h, beta_s, beta_w, beta_o, phi_full, S_frac_15,
                      gamma, sf):
    """K_c (15×15) per channel and total K.  Returns dict."""
    pop = np.asarray(pop_15).reshape(-1)
    N_safe = np.maximum(pop, 1e-10)
    rho = np.asarray(rho_flat).reshape(-1)
    rho_ok = (rho > 0).astype(float)
    work_mask = np.zeros(15); work_mask[4:14] = 1.0
    school_mask = np.zeros((15,15)); school_mask[:4,:4] = 1.0

    sf_gamma = sf / gamma
    row = pop * phi_full * S_frac_15                            # (15,)
    col = 1.0 / N_safe                                          # (15,)

    def _K(C_eff):
        return sf_gamma * row[:, None] * C_eff * col[None, :]

    K_h  = _K(beta_h * np.asarray(C_home))
    K_s  = _K(beta_s * np.asarray(C_school) * school_mask)
    K_w  = _K(beta_w * np.asarray(C_work) *
              (rho * work_mask)[:, None] * rho_ok[None, :])
    K_o  = _K(beta_o * np.asarray(C_other))
    K_total = K_h + K_s + K_w + K_o
    return dict(home=K_h, school=K_s, work=K_w, other=K_o, total=K_total)


def dominant_eigen(K):
    """Return (rho, right v, left u) with v, u normalized so uᵀv = 1.
    All real (taking real parts).  K is 15×15."""
    vals_r, vecs_r = np.linalg.eig(K)
    idx = np.argmax(np.real(vals_r))
    rho = float(np.real(vals_r[idx]))
    v = np.real(vecs_r[:, idx])
    # left = right eigvec of K.T
    vals_l, vecs_l = np.linalg.eig(K.T)
    idxl = np.argmax(np.real(vals_l))
    u = np.real(vecs_l[:, idxl])
    # Sign convention: v components positive on average
    if v.sum() < 0: v = -v
    if u.sum() < 0: u = -u
    # Normalize uᵀ v = 1
    s = u @ v
    if abs(s) < 1e-15:
        raise RuntimeError("uᵀv ≈ 0")
    u = u / s
    return rho, v, u


# ═══════════════════════════════════════════════════════════════════════════
# Load posterior + setup
# ═══════════════════════════════════════════════════════════════════════════
def _load_setup(season=S_REP, C_all=None):
    if C_all is None:
        print(f"[setup:{season}]", flush=True)
        C_all = build_seasonwise_setup(imm=IMM, gamma_15=GAMMA_15,
                                         use_season_pop=True)
    sh = C_all["shared_by_s"][season]
    pop_15 = np.asarray(sh["pop_15"]).reshape(-1)
    rho = np.asarray(sh["rho"])
    if rho.ndim > 1:
        rho = rho.mean(axis=0) if rho.shape[-1] == 15 else rho.reshape(-1)[:15]
    return dict(
        C_all=C_all,
        pop_15=pop_15,
        rho=rho,
        C_home=np.asarray(sh["C_home"]),
        C_school=np.asarray(sh["C_school"]),
        C_work=np.asarray(sh["C_work"]),
        C_other=np.asarray(sh["C_other"]),
        C_home_vac=np.asarray(sh["C_home_vac"]),
        C_school_vac=np.asarray(sh["C_school_vac"]),
        C_work_vac=np.asarray(sh["C_work_vac"]),
        C_other_vac=np.asarray(sh["C_other_vac"]),
        gamma=float(sh["gamma"]),
        H=C_all["H"],
        pop6=np.asarray(C_all["pop6_by_s"][season]),
        shared=sh,
        st0=C_all["st_by_s"][season],
        ngm3=C_all["ngm3_by_s"][season],
        season=season,
    )


def _load_posterior(n_samples, seed):
    d = np.load(MERGED)
    pi_all = np.asarray(d["pi"]); log_R0_all = np.asarray(d["log_R0"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(pi_all.shape[0], size=n_samples, replace=False)
    return pi_all[idx], log_R0_all[idx]


def _sm(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), sd=float(x.std()),
                q05=float(np.quantile(x, 0.05)),
                q50=float(np.quantile(x, 0.5)),
                q95=float(np.quantile(x, 0.95)))


# ═══════════════════════════════════════════════════════════════════════════
# (A) + (B): channel and age decomposition at t=0 (initial S)
# ═══════════════════════════════════════════════════════════════════════════
def analyze_static(setup, pi_samples, log_R0_samples):
    print("[A+B] channel + age decomposition (t=0)", flush=True)
    channels = ["home", "work", "school", "other"]   # match π order (h,w,s,o)
    N = pi_samples.shape[0]
    S_frac_15 = 1.0 - IMM
    season = setup["season"]
    j_s = ["2016-2017","2017-2018","2019-2020"].index(season)

    R_c_samples = {c: np.zeros(N) for c in channels}
    R_b_samples = np.zeros((N, 15))     # per NIMS 15
    R0_check = np.zeros(N)
    beta_c_samples = {c: np.zeros(N) for c in channels}
    rho_pi_samples = np.zeros(N)

    for k in range(N):
        pi_k = pi_samples[k]; R0 = float(np.exp(log_R0_samples[k, j_s]))
        # β_c = π_c · R0 / rho(π) — same as derive_beta_from_R0_simplex
        b0 = np.asarray(derive_beta_from_R0_simplex(setup["ngm3"],
                                                     jnp.asarray(R0),
                                                     jnp.asarray(pi_k),
                                                     jnp.asarray(PHI)))
        # b0 is R0-scaled β (before NGM_F correction for simulator).
        # For static NGM analysis we use b0 directly (matches R0).
        beta_h, beta_w, beta_s, beta_o = float(b0[0]), float(b0[1]), float(b0[2]), float(b0[3])
        Ks = build_K_channels(
            pop_15=setup["pop_15"], rho_flat=setup["rho"],
            C_home=setup["C_home"], C_school=setup["C_school"],
            C_work=setup["C_work"], C_other=setup["C_other"],
            beta_h=beta_h, beta_s=beta_s, beta_w=beta_w, beta_o=beta_o,
            phi_full=PHI, S_frac_15=S_frac_15, gamma=setup["gamma"],
            sf=1.0 + SEASONAL_AMP)
        rho, v, u = dominant_eigen(Ks["total"])
        R0_check[k] = rho
        rho_pi_samples[k] = float(np.real(rho / R0)) if R0 else 1.0
        # channel contribution
        for c in channels:
            R_c_samples[c][k] = float(u @ Ks[c] @ v)
        # age (column) contribution
        R_b_samples[k] = u * (Ks["total"] @ v)   # broadcasting per column
        # Actually: R_b = (uᵀ K)[b] * v[b]  = column-b contribution
        # We want R_b such that Σ_b R_b = uᵀK v = R0.  Correct formula:
        # (uᵀ K)_b * v_b = element-wise.
        R_b_samples[k] = (u @ Ks["total"]) * v
        beta_c_samples["home"][k]   = beta_h
        beta_c_samples["work"][k]   = beta_w
        beta_c_samples["school"][k] = beta_s
        beta_c_samples["other"][k]  = beta_o
        if (k+1) % 20 == 0:
            print(f"  static {k+1}/{N}", flush=True)

    # summaries
    out = dict(
        n_post=N, season=season,
        R0_check=_sm(R0_check),
        rho_pi_over_R0=_sm(rho_pi_samples),
        channels=channels,
        R_c={c: _sm(R_c_samples[c]) for c in channels},
        R_c_share_pct={c: _sm(100.0 * R_c_samples[c] / R0_check) for c in channels},
        beta_c={c: _sm(beta_c_samples[c]) for c in channels},
        R_b_15={AGE15_LAB[b]: _sm(R_b_samples[:, b]) for b in range(15)},
        pop_15=setup["pop_15"].tolist(),
    )
    # aggregate to 6-group
    R_b6 = np.zeros((N, 6))
    for i, (name, bins) in enumerate(AGE6_BINS.items()):
        R_b6[:, i] = R_b_samples[:, bins].sum(axis=1)
    out["age_labels_6"] = list(AGE6_BINS.keys())
    out["R_b_6"] = {list(AGE6_BINS.keys())[i]: _sm(R_b6[:, i]) for i in range(6)}
    out["R_b_6_share_pct"] = {list(AGE6_BINS.keys())[i]:
                                _sm(100.0 * R_b6[:, i] / R0_check) for i in range(6)}
    return out, R_c_samples, R_b6, R0_check, beta_c_samples


# ═══════════════════════════════════════════════════════════════════════════
# (C) time-varying R_e(t)
# ═══════════════════════════════════════════════════════════════════════════
def _simulate_one(setup, R0, pi, p_work=BASE, p_school=BASE,
                    work_win=WH, sch_win=WH, return_inc=False):
    b0 = derive_beta_from_R0_simplex(setup["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI))
    beta = b0 / NGM_F
    kw = dict(setup["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(setup["st0"]),
                                      w_presymp=W, **kw, discretize_time=False)
    st = np.asarray(st)
    # Effective susceptible = S + (1-VE) V   (V people still susceptible to breakthrough)
    S = st[:, E_S, :, :].sum(axis=-1)          # (n_t, 15)
    V = st[:, E_V, :, :].sum(axis=-1)
    VE = float(setup["shared"]["VE"])
    S_eff = S + (1.0 - VE) * V                  # (n_t, 15)
    if return_inc:
        # daily new infections (per age, summed) = d(E+I1+I2+I3+R)/dt
        # simpler: d(E+I_total+R)/dt = new infections into E from S+V
        cum = (st[:, E_E, :, :] + st[:, E_I1, :, :] + st[:, E_I2, :, :]
                + st[:, E_I3, :, :] + st[:, 6, :, :]).sum(axis=(-1,-2))   # (n_t,)
        inc = np.diff(cum)                       # (n_t-1,) daily new infections all ages
        return S_eff, inc
    return S_eff   # (n_t, 15) — effective susceptible count


def analyze_time_varying(setup, pi_samples, log_R0_samples,
                          t_grid_days=None):
    print("[C] R_e(t) — baseline / sick / school", flush=True)
    if t_grid_days is None:
        t_grid_days = np.arange(0, 365, 1)   # daily grid (matches ODE)
    season = setup["season"]
    j_s = ["2016-2017","2017-2018","2019-2020"].index(season)
    channels = ["home", "work", "school", "other"]
    scenarios = [("baseline", dict()),
                 ("sick",     dict(p_work=0.4, work_win=TERM)),
                 ("school",   dict(p_school=0.4, sch_win=TERM))]
    # Policy β multipliers per scenario per day (0 outside TERM window)
    def _policy_scale(name, day):
        """Returns (m_h, m_w, m_s, m_o) — multiplicative scale on β_c."""
        m = [1.0, 1.0, 1.0, 1.0]
        if not (TERM[0] <= day <= TERM[1]):
            return m
        if name == "sick":
            m[1] = 0.4                     # β_w scaled
        elif name == "school":
            m[2] = 0.4                     # β_s scaled
        return m
    N = pi_samples.shape[0]
    T = len(t_grid_days)
    Re_by = {name: np.zeros((N, T)) for name, _ in scenarios}
    Re_channel_by = {name: {c: np.zeros((N, T)) for c in channels}
                       for name, _ in scenarios}
    peak_by = {name: np.zeros(N) for name, _ in scenarios}
    peak_day = {name: np.zeros(N) for name, _ in scenarios}
    below1_day = {name: np.zeros(N) for name, _ in scenarios}
    above1_day = {name: np.zeros(N) for name, _ in scenarios}
    Re_t0 = {name: np.zeros(N) for name, _ in scenarios}

    pop = setup["pop_15"]
    pop_safe = np.maximum(pop, 1e-10)

    # Cross-check (b) growth-rate → R_e via SEIR renewal (approximate):
    # R_e ≈ (σ+r)(γ+r)/(σγ). Uses r(t) from baseline incidence.
    sigma_d = float(setup["shared"]["sigma"])
    gamma_d = float(setup["gamma"])
    daily_inc_all_baseline = None   # populated during loop from baseline scenario

    Re_from_r_samples = np.zeros((N, T))    # per-sample R_e_from_r at t_grid_days
    r_samples = np.zeros((N, T))
    inc_baseline_all = np.zeros((N, 364))   # daily incidence, baseline

    for k in range(N):
        pi_k = pi_samples[k]; R0 = float(np.exp(log_R0_samples[k, j_s]))
        b0 = np.asarray(derive_beta_from_R0_simplex(setup["ngm3"],
                                                     jnp.asarray(R0),
                                                     jnp.asarray(pi_k),
                                                     jnp.asarray(PHI)))
        # β_NGM (matches static R0 formulation exactly — user spec:
        # "정적 NGM 코드에서 S=N, s_f=ref 만 바꾸면 R_e(t)").  At t with
        # sf(t)=SF_REF and S=initial: eig = R0.
        beta_h, beta_w, beta_s, beta_o = float(b0[0]), float(b0[1]), float(b0[2]), float(b0[3])

        for name, kw in scenarios:
            if name == "baseline":
                S_at, inc_daily = _simulate_one(setup, R0, pi_k, **kw, return_inc=True)
                inc_baseline_all[k] = inc_daily
            else:
                S_at = _simulate_one(setup, R0, pi_k, **kw)   # (365, 15)
            for ti, day in enumerate(t_grid_days):
                di = int(day)
                S = S_at[di]                  # effective susceptible count per age (15,)
                S_frac = S / pop_safe
                sf = float(seasonal_factor_cosine(day, amp=SEASONAL_AMP,
                                                     base=SEASONAL_BASE,
                                                     peak_day=SEASONAL_PEAK,
                                                     period=SEASONAL_PERIOD))
                h = float(vacation_weight(day))
                Ch = (1-h) * setup["C_home"] + h * setup["C_home_vac"]
                Cs = (1-h) * setup["C_school"] + h * setup["C_school_vac"]
                Cw = (1-h) * setup["C_work"] + h * setup["C_work_vac"]
                Co = (1-h) * setup["C_other"] + h * setup["C_other_vac"]
                mh, mw, ms, mo = _policy_scale(name, day)
                Ks = build_K_channels(
                    pop_15=pop, rho_flat=setup["rho"],
                    C_home=Ch, C_school=Cs, C_work=Cw, C_other=Co,
                    beta_h=beta_h*mh, beta_s=beta_s*ms,
                    beta_w=beta_w*mw, beta_o=beta_o*mo,
                    phi_full=PHI, S_frac_15=S_frac, gamma=setup["gamma"], sf=sf)
                rho_t, v_t, u_t = dominant_eigen(Ks["total"])
                Re_by[name][k, ti] = rho_t
                for c in channels:
                    Re_channel_by[name][c][k, ti] = float(u_t @ Ks[c] @ v_t)

            # per-sample summaries
            arr = Re_by[name][k]
            peak_by[name][k] = float(arr.max())
            peak_day[name][k] = float(t_grid_days[np.argmax(arr)])
            # first crossing 1 downward:  Re >= 1 then Re < 1
            crossed_up = False
            below_day = -1.0
            above_day = -1.0
            for ti in range(len(arr)):
                if not crossed_up and arr[ti] >= 1.0:
                    crossed_up = True
                    above_day = float(t_grid_days[ti])
                if crossed_up and arr[ti] < 1.0:
                    below_day = float(t_grid_days[ti])
                    break
            below1_day[name][k] = below_day
            above1_day[name][k] = above_day
            Re_t0[name][k] = float(arr[0])

        # (b) growth-rate cross-check on baseline daily incidence
        inc = inc_baseline_all[k]
        # smooth via log-domain centered diff over ±7d window
        log_inc = np.log(np.maximum(inc, 1e-6))
        r_daily = np.zeros_like(log_inc)
        half = 7
        for di_ in range(len(log_inc)):
            lo = max(0, di_-half); hi = min(len(log_inc), di_+half+1)
            xs = np.arange(lo, hi); ys = log_inc[lo:hi]
            if len(xs) >= 3:
                # linear regression slope
                slope, _ = np.polyfit(xs, ys, 1)
                r_daily[di_] = slope
            else:
                r_daily[di_] = 0.0
        for ti, day in enumerate(t_grid_days):
            di = int(day) if int(day) < len(r_daily) else len(r_daily)-1
            r_val = float(r_daily[di])
            r_samples[k, ti] = r_val
            # SEIR approximate: (σ+r)(γ+r)/(σγ)
            Re_from_r_samples[k, ti] = (sigma_d + r_val) * (gamma_d + r_val) / (sigma_d * gamma_d)
        if (k+1) % 10 == 0:
            print(f"  Re {k+1}/{N}", flush=True)

    # Summarize per time point (mean + CI across posterior)
    def _grid_sm(arr2d):
        m = arr2d.mean(axis=0)
        lo = np.quantile(arr2d, 0.05, axis=0)
        hi = np.quantile(arr2d, 0.95, axis=0)
        return dict(mean=m.tolist(), q05=lo.tolist(), q95=hi.tolist())

    # cross-check residual (NGM vs growth-rate implied)
    Re_ngm_base = Re_by["baseline"]              # (N, T)
    resid = Re_ngm_base - Re_from_r_samples
    max_resid = float(np.abs(resid).max())
    # peak of R_e_from_r
    peak_from_r_val = float(Re_from_r_samples.mean(0).max())
    peak_from_r_day = float(t_grid_days[int(np.argmax(Re_from_r_samples.mean(0)))])

    out = dict(t_days=[int(x) for x in t_grid_days],
                scenarios=[n for n, _ in scenarios],
                Re={name: _grid_sm(Re_by[name]) for name, _ in scenarios},
                Re_channel={name: {c: _grid_sm(Re_channel_by[name][c])
                                     for c in channels}
                              for name, _ in scenarios},
                Re_from_growth_rate=_grid_sm(Re_from_r_samples),
                growth_rate_r=_grid_sm(r_samples),
                peak_Re={name: _sm(peak_by[name]) for name, _ in scenarios},
                peak_day={name: _sm(peak_day[name]) for name, _ in scenarios},
                Re_at_t0={name: _sm(Re_t0[name]) for name, _ in scenarios},
                first_above1_day={name: _sm(above1_day[name]) for name, _ in scenarios},
                first_below1_day={name: _sm(below1_day[name]) for name, _ in scenarios},
                cross_check=dict(
                    max_abs_resid_ngm_minus_growth=max_resid,
                    peak_Re_from_growth_mean=peak_from_r_val,
                    peak_day_from_growth=peak_from_r_day,
                    method_growth="SEIR renewal approx: R_e=(σ+r)(γ+r)/(σγ)",
                    sigma=sigma_d, gamma=gamma_d))
    return out, Re_by, Re_channel_by


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════
CHANNEL_COLORS = {"home": "#009E73", "work": "#E69F00",
                    "school": "#0072B2", "other": "#CC79A7"}

def plot_channel(out, R_c_samples, R0_check, pi_samples):
    channels = out["channels"]
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.9, 3.0),
                              constrained_layout=True)
    ax = axes[0]
    xs = np.arange(len(channels))
    means = [out["R_c"][c]["mean"] for c in channels]
    q05 = [out["R_c"][c]["q05"] for c in channels]
    q95 = [out["R_c"][c]["q95"] for c in channels]
    yerr = [np.array(means) - np.array(q05), np.array(q95) - np.array(means)]
    cols = [CHANNEL_COLORS[c] for c in channels]
    ax.bar(xs, means, yerr=yerr, color=cols, alpha=0.85, capsize=3,
           edgecolor="black", linewidth=0.4)
    ax.axhline(np.mean(R0_check), color=COL_ZERO, ls="--", lw=0.8,
                label=fr"$R_0$ mean = {np.mean(R0_check):.2f}")
    ax.set_xticks(xs); ax.set_xticklabels(channels)
    ax.set_ylabel(r"$R_c$ contribution to $R_0$")
    ax.set_title("(A) Channel-wise contribution")
    ax.legend(loc="upper right", fontsize=7, frameon=False)

    # Panel B: R_c share vs π share (bar side-by-side)
    ax = axes[1]
    R_share = np.array([out["R_c_share_pct"][c]["mean"] for c in channels])
    pi_share = 100.0 * pi_samples.mean(axis=0)   # (4,)
    # π index order in npz is (h, w, s, o) matching channels
    dx = 0.35
    ax.bar(xs - dx/2, pi_share, dx, color=cols, alpha=0.5,
            edgecolor="black", linewidth=0.3,
            label=r"$\pi_c$ share (%)")
    ax.bar(xs + dx/2, R_share, dx, color=cols, alpha=0.95,
            edgecolor="black", linewidth=0.3,
            label=r"$R_c$ share (%)")
    ax.set_xticks(xs); ax.set_xticklabels(channels)
    ax.set_ylabel("Share of total (%)")
    ax.set_title("(B) π share vs R contribution")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    savefig(fig, "repro_channel")


def plot_byage(out, R_b6):
    labs = list(AGE6_BINS.keys())
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.7, 3.0), constrained_layout=True)
    xs = np.arange(len(labs))
    means = R_b6.mean(axis=0)
    q05 = np.quantile(R_b6, 0.05, axis=0)
    q95 = np.quantile(R_b6, 0.95, axis=0)
    yerr = [means - q05, q95 - means]
    # child ages orange, adult blue, elder gray
    cols = ["#E69F00","#E69F00","#E69F00","#0072B2","#0072B2","#666666"]
    ax.bar(xs, means, yerr=yerr, color=cols, alpha=0.85, capsize=3,
            edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel(r"$R_b$ contribution to $R_0$")
    ax.set_title("Age-wise contribution to $R_0$ (2019-2020)")
    savefig(fig, "repro_byage")


def plot_Re_time(out_re):
    t = np.asarray(out_re["t_days"])
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.9, 3.2), constrained_layout=True)
    scenario_colors = {"baseline": "#333333",
                        "sick":     COL_SICK,
                        "school":   COL_SCHOOL}
    for name in ("baseline", "sick", "school"):
        m = np.asarray(out_re["Re"][name]["mean"])
        lo = np.asarray(out_re["Re"][name]["q05"])
        hi = np.asarray(out_re["Re"][name]["q95"])
        col = scenario_colors[name]
        ax.fill_between(t, lo, hi, color=col, alpha=0.18)
        label = {"baseline":"Baseline","sick":"Sick leave (term)",
                  "school":"School absence (term)"}[name]
        ax.plot(t, m, color=col, lw=1.6, label=label)
    ax.axhline(1.0, color=COL_ZERO, ls="--", lw=0.8, alpha=0.7)
    # winter break window shading (days 113-183)
    ax.axvspan(113, 183, color="#4a90e2", alpha=0.10,
                label="Winter break")
    # term window (70-113) light orange
    ax.axvspan(70, 113, color="#E69F00", alpha=0.08,
                label="School-term window")
    ax.set_xlabel("Season day (0 = Sep 1)")
    ax.set_ylabel(r"$R_e(t)$")
    ax.set_title(r"Effective reproduction number over time")
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=1)
    savefig(fig, "repro_Re_time")


# ═══════════════════════════════════════════════════════════════════════════
# Per-season pop source label (MOIS June of peak year of season)
# ═══════════════════════════════════════════════════════════════════════════
POP_SOURCE = {
    "2016-2017": "MOIS 2016-06 season-specific",
    "2017-2018": "MOIS 2017-06 season-specific",
    "2019-2020": "MOIS 2019-06 season-specific",
}
# Winter break / school-term windows per season (season t=0 = Sep 1 of first
# year). ODE hardcodes 2019-20 calendar; per-season shifts ≤ ~2 days for
# Korean elementary school breaks, but we keep the same TERM/BREAK for
# consistency with ODE dynamics and record them here.
WINDOWS = {
    "2016-2017": dict(term_window=[70.0, 113.0], break_window=[113.0, 183.0]),
    "2017-2018": dict(term_window=[70.0, 113.0], break_window=[113.0, 183.0]),
    "2019-2020": dict(term_window=[70.0, 113.0], break_window=[113.0, 183.0]),
}


def run_one_season(season, C_all, pi_s, log_R0_s):
    print("\n" + "="*90)
    print(f"SEASON {season}")
    print("="*90, flush=True)
    t0 = time.time()
    setup = _load_setup(season=season, C_all=C_all)
    print(f"[setup:{season}] pop total={setup['pop_15'].sum():,.0f}", flush=True)

    # (A) + (B)
    t_a = time.time()
    static, R_c_samples, R_b6, R0_check, beta_c = analyze_static(setup, pi_s, log_R0_s)
    print(f"[A+B done:{season}] {time.time()-t_a:.1f}s", flush=True)

    # (C)
    t_c = time.time()
    re_out, Re_by, Re_ch = analyze_time_varying(setup, pi_s, log_R0_s)
    print(f"[C done:{season}] {time.time()-t_c:.1f}s", flush=True)

    out = dict(static=static, Re=re_out,
                meta=dict(n_post=N_POST, seed=SEED, season=season,
                          seasonal_amp=SEASONAL_AMP,
                          seasonal_base=SEASONAL_BASE,
                          sf_ref=SF_REF,
                          nuts=str(MERGED.name),
                          pop_source=POP_SOURCE[season],
                          term_window=WINDOWS[season]["term_window"],
                          break_window=WINDOWS[season]["break_window"],
                          wall_sec=time.time()-t0))
    out_path = ED / f"reproduction_numbers_{season}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json:{season}] {out_path}", flush=True)

    # Console summary
    print(f"\n=== {season} SUMMARY ===")
    print(f"R0 (posterior mean, K eigen check): {static['R0_check']['mean']:.3f}"
          f"  [{static['R0_check']['q05']:.3f}, {static['R0_check']['q95']:.3f}]")
    for c in static["channels"]:
        r = static["R_c"][c]; s = static["R_c_share_pct"][c]
        pi_share = 100.0 * pi_s[:, ["home","work","school","other"].index(c)].mean()
        print(f"  {c:6s}: R_c={r['mean']:.3f} ({s['mean']:.1f}%)  π={pi_share:.1f}%")
    for a in static["R_b_6"]:
        r = static["R_b_6"][a]; s = static["R_b_6_share_pct"][a]
        print(f"  age {a:>6s}: R_b={r['mean']:.3f} ({s['mean']:.1f}%)")
    print(f"  SF_REF={SF_REF:.3f}  sf(0)={SEASONAL_BASE+SEASONAL_AMP*np.cos(2*np.pi*(0-SEASONAL_PEAK)/SEASONAL_PERIOD):.3f}")
    for name in ("baseline","sick","school"):
        p = re_out["peak_Re"][name]; d = re_out["peak_day"][name]
        b = re_out["first_below1_day"][name]
        a_ = re_out["first_above1_day"][name]
        t0_ = re_out["Re_at_t0"][name]
        print(f"  {name:8s} Re(0)={t0_['mean']:.3f}  peak={p['mean']:.3f} at day {d['mean']:.0f}"
              f"  |  Re↑1 at day {a_['mean']:.0f}  Re↓1 at day {b['mean']:.0f}")
    cc = re_out["cross_check"]
    print(f"  [cross-check] Re_from_growth peak={cc['peak_Re_from_growth_mean']:.3f}"
          f" at day {cc['peak_day_from_growth']:.0f}  |  max|resid|={cc['max_abs_resid_ngm_minus_growth']:.3f}")
    return out


def main():
    t0 = time.time()
    print(f"[loading season-pop setup for all 3 seasons] …", flush=True)
    C_all = build_seasonwise_setup(imm=IMM, gamma_15=GAMMA_15,
                                     use_season_pop=True)
    print(f"[C_all loaded] {time.time()-t0:.1f}s", flush=True)

    pi_s, log_R0_s = _load_posterior(N_POST, SEED)
    print(f"[posterior] N={N_POST} (from {MERGED.name})", flush=True)

    for season in ("2016-2017", "2017-2018", "2019-2020"):
        run_one_season(season, C_all, pi_s, log_R0_s)

    print(f"\n[all seasons total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
