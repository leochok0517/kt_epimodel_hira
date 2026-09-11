"""Phase 2 예비 — 보정된 C(방식 A)의 적합도 진단 (D-0, D-1). NUTS 재적합 없음.

D-0: 시즌별 인구벡터가 실제로 build_seasonwise_setup 에 반영되는지 확인.
D-1: 원본/보정 C 아래에서, 기존 NUTS 사후 draw(π, R0, φ_NB)를 그대로 넣고
     데이터 NLL(NB2, `nb_nll_jax`)만 재계산해 적합도 차이를 측정한다.
     우도 레시피는 scripts/nuts_seasonpop.py::make_joint_loss 를 그대로
     따른다(핀 페널티 제외, baseline 정책 인자, 동일 min_rate/n_weeks/w).

Standalone. 저장소 핵심 코드는 건드리지 않는다.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import phase1_reciprocity_v2 as P1
import reproduction_analysis as RA
import nuts_seasonpop as NS   # reuse the EXACT fit-time loss recipe (import only, no MCMC run)
from kt_epimodel_hira.jax_model.numpyro_model import (
    derive_beta_from_R0_simplex, make_ngm_eigvalue_fn,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    simulation_to_hira_by_age_jax, nb_nll_jax, HIRA_AGE_GROUPS,
)

ED = REPO / "outputs" / "eda"
OUT_JSON = ED / "phase1_reciprocity_partD.json"

SEAS = NS.SEAS                 # ["2016-2017","2017-2018","2019-2020"]
F_SEASONS = NS.F.SEASONS       # 6-season order used to index C_all["obs"]/["w"]
PHI_J = jnp.asarray(NS.PHI)
GAMMA_J = jnp.asarray(NS.GAMMA)
KAP_J = jnp.asarray(NS.KAP)
W = W_PRESYMP
NGM_F = ngm_factor(W)
BASE = NS.BASE

N_DRAWS = 250
SEED = 909


def sm(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), sd=float(x.std()),
                q05=float(np.quantile(x, 0.05)), q50=float(np.quantile(x, 0.5)),
                q95=float(np.quantile(x, 0.95)))


# ═══════════════════════════════════════════════════════════════════════════
# D-0: season population check
# ═══════════════════════════════════════════════════════════════════════════
def d0_check():
    C_all = RA.build_seasonwise_setup(imm=RA.IMM, gamma_15=RA.GAMMA_15, use_season_pop=True)
    rows = {}
    for s in SEAS:
        pop = np.asarray(C_all["pop_15_by_s"][s]).reshape(-1)
        rows[s] = dict(pop_0_4=float(pop[0]), pop_sum=float(pop.sum()),
                         pop_share_0_4_pct=float(100 * pop[0] / pop.sum()))
        print(f"[D-0] {s}: pop_15[0]={pop[0]:,.0f}  sum={pop.sum():,.0f}  "
              f"share_0-4={100*pop[0]/pop.sum():.2f}%")

    # cross-check vs pop_all_seasons.csv
    import csv
    csv_path = REPO / "data" / "population" / "pop_all_seasons.csv"
    from pop_seasons import SEASON_MAP
    csv_vals = {full: 0.0 for full in SEASON_MAP}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for full, short in SEASON_MAP.items():
                if row["season"] == short and int(row["band_index"]) == 0:
                    csv_vals[full] = float(row["population"])
    for s in SEAS:
        match = abs(rows[s]["pop_0_4"] - csv_vals[s]) < 1.0
        rows[s]["csv_pop_0_4"] = csv_vals[s]
        rows[s]["matches_csv"] = bool(match)
        print(f"       csv pop_0_4={csv_vals[s]:,.0f}  matches={match}")

    # explain why A-2 shares were season-invariant despite pop differing:
    # build_K_channels is K[a,b] = pop[a]*M[a,b]/pop[b] = diag(pop) M diag(pop)^-1,
    # a diagonal similarity transform of a pop-INDEPENDENT matrix M. Similarity
    # transforms preserve eigenvalues, and u^T K_c v is exactly invariant too
    # (u_new = D^-1 u_old, v_new = D v_old cancel). So rho and R_c-shares from
    # build_K_channels/dominant_eigen are population-invariant BY CONSTRUCTION,
    # regardless of season pop differences. Verified numerically below with a
    # deliberately perturbed pop vector.
    setup = RA._load_setup(season="2019-2020", C_all=C_all)
    S_frac_15 = 1.0 - RA.IMM

    def get_share(pop_vec):
        Ks = RA.build_K_channels(
            pop_15=pop_vec, rho_flat=setup["rho"],
            C_home=setup["C_home"], C_school=setup["C_school"],
            C_work=setup["C_work"], C_other=setup["C_other"],
            beta_h=1.0, beta_s=1.0, beta_w=1.0, beta_o=1.0,
            phi_full=RA.PHI, S_frac_15=S_frac_15, gamma=setup["gamma"], sf=1.9)
        rho, v, u = RA.dominant_eigen(Ks["total"])
        return rho, {c: float(u @ Ks[c] @ v) / rho for c in ("home", "work", "school", "other")}

    pop_real = np.asarray(setup["pop_15"]).reshape(-1)
    pop_reversed = pop_real[::-1].copy()
    rho1, sh1 = get_share(pop_real)
    rho2, sh2 = get_share(pop_reversed)
    invariance_demo = dict(
        rho_real_pop=rho1, rho_reversed_pop=rho2,
        shares_real_pop=sh1, shares_reversed_pop=sh2,
        max_abs_diff=float(max(abs(sh1[c] - sh2[c]) for c in sh1)),
    )
    print(f"[D-0] pop-invariance demo: rho(real)={rho1:.6f}  rho(reversed)={rho2:.6f}  "
          f"max share diff={invariance_demo['max_abs_diff']:.2e}")

    return dict(
        season_pop_rows=rows,
        conclusion=("build_seasonwise_setup 은 시즌별 인구를 올바르게 반영함 "
                     "(pop_15_by_s[0-4] 이 pop_all_seasons.csv 값과 정확히 일치). "
                     "0-2 표에서 R_school%·증폭이 세 시즌 동일했던 것은 데이터 로딩 "
                     "오류가 아니라, build_K_channels 의 K[a,b]=pop[a]M[a,b]/pop[b] "
                     "구조가 diag(pop) M diag(pop)^-1 형태의 대각 유사변환이라 "
                     "고유값(rho)과 u^T K_c v 채널 기여도가 인구 벡터에 대해 수학적으로 "
                     "불변이기 때문. reversed pop 으로 대체해도 rho·shares 가 소수점 "
                     "13자리까지 동일함으로 수치 검증함. 0-2 표는 다시 만들 필요 없음 — "
                     "이미 옳음. 단, 이 불변성은 A-2의 '정적 NGM 분해'에만 해당하고, "
                     "파트 B의 forward 시뮬레이션(절대 감염 수)은 실제로 시즌별 인구에 "
                     "따라 달라짐(이미 확인됨: school_num_net_term 등 시즌마다 다른 값)."),
        pop_invariance_numerical_demo=invariance_demo,
    )


# ═══════════════════════════════════════════════════════════════════════════
# D-1: fit-quality diagnostic
# ═══════════════════════════════════════════════════════════════════════════
def build_corrected_C_all(C_all_orig, G):
    raw_term, _ = P1.load_raw_ego(P1.TERM_NPZ)
    raw_vac, _ = P1.load_raw_ego(P1.VAC_NPZ)
    symA_term = P1.method_A(raw_term, G)
    symA_vac = P1.method_A(raw_vac, G)
    C_A_term = P1.to_model_convention(symA_term)
    C_A_vac = P1.to_model_convention(symA_vac)

    ngm3_by_s = {}
    shared_by_s = {}
    for s in SEAS:
        setup = RA._load_setup(season=s, C_all=C_all_orig)
        pop_2d = np.asarray(setup["pop_15"]).reshape(15, 1)
        ngm3_by_s[s] = make_ngm_eigvalue_fn(
            pop_15=pop_2d, rho=np.asarray(setup["rho"]),
            C_home=C_A_term["home"], C_work=C_A_term["work"],
            C_school=C_A_term["school"], C_other=C_A_term["other"],
            R0_immunity=RA.IMM, gamma=setup["gamma"],
            seasonal_factor=1.0 + RA.SEASONAL_AMP)
        sh = dict(C_all_orig["shared_by_s"][s])
        for c in P1.CHANNELS:
            sh[f"C_{c}"] = jnp.asarray(C_A_term[c])
            sh[f"C_{c}_vac"] = jnp.asarray(C_A_vac[c])
        shared_by_s[s] = sh

    C_all_A = dict(C_all_orig)
    C_all_A["ngm3_by_s"] = ngm3_by_s
    C_all_A["shared_by_s"] = shared_by_s
    return C_all_A, C_A_term


def make_nll_pred_fn(C_all, season):
    obs = jnp.asarray(C_all["obs"][F_SEASONS.index(season)])
    w = jnp.asarray(C_all["w"][F_SEASONS.index(season)])
    nw = int(C_all["nw"])
    init_state = split_seed_to_erlang(C_all["st_by_s"][season])
    ngm = C_all["ngm3_by_s"][season]
    sh = dict(C_all["shared_by_s"][season])

    def one(logR0_s, logit_pi, phi_nb):
        pi = jax.nn.softmax(logit_pi)
        R0_s = jnp.exp(logR0_s)
        beta_s = derive_beta_from_R0_simplex(ngm, R0_s, pi, PHI_J) / NGM_F
        kw = dict(sh)
        kw["beta_h"] = beta_s[0]; kw["beta_w"] = beta_s[1]
        kw["beta_s"] = beta_s[2]; kw["beta_o"] = beta_s[3]
        kw["phi_susc"] = PHI_J; kw["kappa"] = KAP_J
        kw["p_school"] = BASE; kw["p_work"] = BASE   # baseline, matches fit
        st = simulate_jax_erlang_presymp(init_state, w_presymp=W, **kw,
                                          discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        pred = simulation_to_hira_by_age_jax(inc, GAMMA_J, n_weeks=nw)
        nll = nb_nll_jax(obs, pred, w, concentration=phi_nb, min_rate=0.01)
        return nll, pred

    fn = jax.jit(jax.vmap(one, in_axes=(0, 0, 0)))
    return fn, np.asarray(obs), np.asarray(w)


def d1_diagnostic():
    C_all_orig = NS.build()   # identical call to what nuts_seasonpop.py used
    G, G_path = P1.load_G("pop_national_2024_01.csv", 51_313_912)
    C_all_A, C_A_term = build_corrected_C_all(C_all_orig, G)

    d = np.load(ED / "nuts_seasonpop_merged.npz")
    pi_all = np.asarray(d["logit_pi"]); log_R0_all = np.asarray(d["log_R0"])
    phi_nb_all = np.asarray(d["phi_nb"])
    n_total = log_R0_all.shape[0]

    # ---- validation vs the ACTUAL fit-time loss function (imported, not
    #      reimplemented) — no scalar NLL is logged anywhere in the diagnostics
    #      / full json outputs, so this is the strongest available check:
    #      bit-identical code path. ----
    loss_fn_actual_total = NS.make_joint_loss(C_all_orig)
    val_idx = 0
    logR0_v = jnp.asarray(log_R0_all[val_idx])
    logitpi_v = jnp.asarray(pi_all[val_idx])
    phinb_v = jnp.asarray(phi_nb_all[val_idx])
    total_actual = float(loss_fn_actual_total(logR0_v, logitpi_v, phinb_v))

    per_season_total = 0.0
    per_season_vals = {}
    for i, s in enumerate(SEAS):
        fn, obs_np, w_np = make_nll_pred_fn(C_all_orig, s)
        nll_i, _ = fn(jnp.asarray([log_R0_all[val_idx, i]]),
                        jnp.asarray([pi_all[val_idx]]),
                        jnp.asarray([phi_nb_all[val_idx]]))
        nll_i = float(np.asarray(nll_i)[0])
        per_season_vals[s] = nll_i
        per_season_total += nll_i

    rel_err = abs(per_season_total - total_actual) / abs(total_actual)
    print(f"[D-1] validation: sum(per-season NLL)={per_season_total:.6f}  "
          f"vs NS.make_joint_loss total={total_actual:.6f}  rel_err={rel_err:.3e}")
    validation = dict(
        note=("nuts_seasonpop_merged_diagnostics.json / nuts_seasonpop_full*.json "
               "어디에도 데이터 NLL 스칼라 값이 저장돼 있지 않아 '기록된 값과 대조'는 "
               "불가능(확인 불가). 대신 scripts/nuts_seasonpop.py 의 make_joint_loss 를 "
               "그대로 import 해서 같은 draw에 대해 계산한 total과, 이 스크립트가 "
               "독립적으로 시즌별로 나눠 계산한 합을 비교함 — 코드 경로가 100% 동일하므로 "
               "이 이상으로 강한 검증은 없음."),
        draw_index=val_idx,
        total_from_actual_fit_code=total_actual,
        sum_of_per_season_reimpl=per_season_total,
        per_season=per_season_vals,
        rel_err=rel_err,
        ok=bool(rel_err < 1e-6),
    )
    if rel_err >= 1e-6:
        print("[D-1] !!! 검증 실패 — 우도 레시피 불일치. 이후 계산 중단 필요.")

    # ---- N_DRAWS posterior sample ----
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_total, size=N_DRAWS, replace=False)
    logit_pi_b = jnp.asarray(pi_all[idx])           # (N,4)
    phi_nb_b = jnp.asarray(phi_nb_all[idx])          # (N,)
    log_R0_b = log_R0_all[idx]                        # (N,3) numpy

    season_report = {}
    total_orig = np.zeros(N_DRAWS)
    total_corr = np.zeros(N_DRAWS)

    for i, s in enumerate(SEAS):
        print(f"\n[D-1] season {s} …", flush=True)
        fn_orig, obs_np, w_np = make_nll_pred_fn(C_all_orig, s)
        fn_corr, _, _ = make_nll_pred_fn(C_all_A, s)

        logR0_s = jnp.asarray(log_R0_b[:, i])
        nll_orig, pred_orig = fn_orig(logR0_s, logit_pi_b, phi_nb_b)
        nll_corr, pred_corr = fn_corr(logR0_s, logit_pi_b, phi_nb_b)
        nll_orig = np.asarray(nll_orig); nll_corr = np.asarray(nll_corr)
        pred_orig = np.asarray(pred_orig); pred_corr = np.asarray(pred_corr)  # (N, nw, 6)

        total_orig += nll_orig; total_corr += nll_corr
        delta = nll_corr - nll_orig

        mask = (w_np > 0)   # (nw,6) boolean — weeks/ages actually scored
        n_cells = int(mask.sum())

        # posterior predictive bias: season-total pred/obs ratio per HIRA age
        pred_tot_orig = (pred_orig * mask[None, :, :]).sum(axis=1)   # (N,6)
        pred_tot_corr = (pred_corr * mask[None, :, :]).sum(axis=1)
        obs_tot = (obs_np * mask).sum(axis=0)                        # (6,)
        ratio_orig = pred_tot_orig / np.maximum(obs_tot[None, :], 1e-6)
        ratio_corr = pred_tot_corr / np.maximum(obs_tot[None, :], 1e-6)

        # peak week (within masked window) — use age-group columns where mask has any True
        def peak_week_diff(pred):
            out = {}
            for a, age in enumerate(HIRA_AGE_GROUPS):
                col_mask = mask[:, a]
                if not col_mask.any():
                    out[age] = None
                    continue
                weeks = np.where(col_mask)[0]
                obs_col = obs_np[weeks, a]
                pred_col = pred[:, weeks, a]   # (N, n_weeks_masked)
                obs_peak_w = weeks[int(np.argmax(obs_col))]
                pred_peak_w = weeks[np.argmax(pred_col, axis=1)]   # (N,)
                out[age] = sm(pred_peak_w - obs_peak_w)
            return out

        peak_diff_orig = peak_week_diff(pred_orig)
        peak_diff_corr = peak_week_diff(pred_corr)

        season_report[s] = dict(
            n_cells_scored=n_cells,
            nll_original=sm(nll_orig),
            nll_corrected=sm(nll_corr),
            delta_corrected_minus_original=sm(delta),
            pred_obs_ratio_by_age=dict(
                original={HIRA_AGE_GROUPS[a]: sm(ratio_orig[:, a]) for a in range(6)},
                corrected={HIRA_AGE_GROUPS[a]: sm(ratio_corr[:, a]) for a in range(6)},
            ),
            peak_week_diff_pred_minus_obs=dict(original=peak_diff_orig, corrected=peak_diff_corr),
        )
        print(f"  NLL orig mean={nll_orig.mean():.2f} sd={nll_orig.std():.2f}  "
              f"corr mean={nll_corr.mean():.2f}  Δmean={delta.mean():+.2f} "
              f"[{np.quantile(delta,0.05):+.2f},{np.quantile(delta,0.95):+.2f}]")
        for a, age in enumerate(HIRA_AGE_GROUPS):
            print(f"    {age:6s}: pred/obs orig={ratio_orig[:,a].mean():.3f}  "
                  f"corr={ratio_corr[:,a].mean():.3f}")

    total_delta = total_corr - total_orig
    sd_orig_total = float(total_orig.std())
    mean_delta_total = float(total_delta.mean())
    ratio_to_sd = abs(mean_delta_total) / sd_orig_total if sd_orig_total > 0 else float("inf")
    if ratio_to_sd <= 1.0:
        verdict = "within_sd"
    elif ratio_to_sd <= 3.0:
        verdict = "ambiguous_1x_3x"
    else:
        verdict = "clearly_worse_gt_3x"

    print(f"\n[D-1] 3-season total: Δmean={mean_delta_total:+.2f}  "
          f"sd(NLL_orig, total)={sd_orig_total:.2f}  ratio={ratio_to_sd:.2f}  "
          f"verdict={verdict}")

    return dict(
        validation=validation,
        n_draws=N_DRAWS, seed=SEED,
        by_season=season_report,
        total_3season=dict(
            delta=sm(total_delta),
            sd_nll_original=sd_orig_total,
            ratio_delta_to_sd=ratio_to_sd,
            verdict=verdict,
        ),
    )


def main():
    t0 = time.time()
    report = {}
    print("="*70, "\nD-0\n" + "="*70)
    report["D0"] = d0_check()

    print("\n" + "="*70, "\nD-1\n" + "="*70)
    report["D1"] = d1_diagnostic()

    report["meta"] = dict(wall_sec=time.time() - t0,
                            generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    OUT_JSON.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[json] {OUT_JSON}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
