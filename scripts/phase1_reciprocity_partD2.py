"""Phase 2 예비 D-2/D-3 — 보정된 C(방식 A)로 3시즌 결합 MAP 재적합 (NUTS 없음).

D-1이 "애매(1-3배 SD)"로 나와, 사후 π를 다시 최적화하면 어디로 가는지 확인한다.
`multiseason_joint_sharedpi.py`의 make_shared_pi_joint_loss_nb 는 내부적으로
simulate_jax(비-erlang, κ 없음) 를 하드코딩하고 있어 nuts_seasonpop.py(그리고
D-1)가 쓴 simulate_jax_erlang_presymp(erlang presymptomatic + κ 3-way, 시즌별
인구)와 동역학이 다르다 — 그대로 쓰면 "C 보정 효과"와 "모형 계열 불일치"가
뒤섞여 판정이 무효화된다. 그래서 이 스크립트는 make_shared_pi_joint_loss_nb를
문자 그대로 재사용하는 대신, D-1에서 이미 검증된 것과 동일한 우도 레시피
(nuts_seasonpop.py::make_joint_loss, erlang, 3시즌, κ, 시즌 인구)에 대해
make_shared_pi_joint_loss_nb와 동일한 구조(공유 π + 시즌별 R0 + φ_NB,
L-BFGS-B multistart, 동일 pin 페널티/σ/bounds/시작점 생성 규칙)로 MAP을 짠다.
이 판단 근거는 보고서에 명시한다.

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
from scipy.optimize import minimize

import phase1_reciprocity_v2 as P1
import reproduction_analysis as RA
import nuts_seasonpop as NS
import phase1_reciprocity_partD as PD   # reuse validated build_corrected_C_all / setup
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    simulation_to_hira_by_age_jax, nb_nll_jax, HIRA_AGE_GROUPS,
)
import sens_workshare_kappa_v2 as S   # bounds only (LOG_R0_BOUNDS/LOGIT_PI_BOUNDS/PHI_NB_BOUNDS)

ED = REPO / "outputs" / "eda"
OUT_JSON = ED / "phase1_reciprocity_partD2.json"

SEAS = NS.SEAS
F_SEASONS = NS.F.SEASONS
PHI_J = jnp.asarray(NS.PHI)
GAMMA_J = jnp.asarray(NS.GAMMA)
KAP_J = jnp.asarray(NS.KAP)
W = W_PRESYMP
NGM_F = ngm_factor(W)
BASE = NS.BASE

# ---- multistart machinery: reused verbatim from multiseason_joint_sharedpi.py ----
N_STARTS = 12
SEED = 23
LOG_R0_B = S.LOG_R0_BOUNDS
LOGIT_B = S.LOGIT_PI_BOUNDS
PHI_NB_B = S.PHI_NB_BOUNDS
LOGIT_REF = np.array(NS.F.LOGIT_REF)
SIGMA_PIN = np.array(NS.F.SIGMA_PIN)


def pin_pen(logit_pi_np, sigma_pin):
    centered = logit_pi_np - np.mean(logit_pi_np)
    return 0.5 * float(np.sum((centered - LOGIT_REF) ** 2 / sigma_pin ** 2))


def pin_grad(logit_pi_np, sigma_pin):
    return (logit_pi_np - np.mean(logit_pi_np) - LOGIT_REF) / sigma_pin ** 2


# ═══════════════════════════════════════════════════════════════════════════
# joint loss builder — erlang presymp, 3 seasons, κ, season pop (matches D-1)
# ═══════════════════════════════════════════════════════════════════════════
def make_total_loss(C_all):
    packs = []
    for s in SEAS:
        obs = jnp.asarray(C_all["obs"][F_SEASONS.index(s)])
        w = jnp.asarray(C_all["w"][F_SEASONS.index(s)])
        nw = int(C_all["nw"])
        init_state = split_seed_to_erlang(C_all["st_by_s"][s])
        ngm = C_all["ngm3_by_s"][s]
        sh = dict(C_all["shared_by_s"][s])
        packs.append((obs, w, nw, init_state, ngm, sh))

    def per_season_nll(i, logR0_s, pi, phi_nb):
        obs, w, nw, init_state, ngm, sh = packs[i]
        beta_s = derive_beta_from_R0_simplex(ngm, jnp.exp(logR0_s), pi, PHI_J) / NGM_F
        kw = dict(sh)
        kw["beta_h"] = beta_s[0]; kw["beta_w"] = beta_s[1]
        kw["beta_s"] = beta_s[2]; kw["beta_o"] = beta_s[3]
        kw["phi_susc"] = PHI_J; kw["kappa"] = KAP_J
        kw["p_school"] = BASE; kw["p_work"] = BASE
        st = simulate_jax_erlang_presymp(init_state, w_presymp=W, **kw,
                                          discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        pred = simulation_to_hira_by_age_jax(inc, GAMMA_J, n_weeks=nw)
        return nb_nll_jax(obs, pred, w, concentration=phi_nb, min_rate=0.01)

    def total_loss(log_R0_vec, logit_pi, phi_nb):
        pi = jax.nn.softmax(logit_pi)
        total = 0.0
        for i in range(3):
            total = total + per_season_nll(i, log_R0_vec[i], pi, phi_nb)
        return total

    def per_season_loss(log_R0_vec, logit_pi, phi_nb):
        pi = jax.nn.softmax(logit_pi)
        return [float(per_season_nll(i, log_R0_vec[i], pi, phi_nb)) for i in range(3)]

    return total_loss, per_season_loss


def run_map(C_all, x0_list, sigma_pin, label):
    total_loss, per_season_loss = make_total_loss(C_all)

    def obj(x):
        lr0 = x[0:3]; lpi = x[3:7]; pnb = x[7]
        return total_loss(lr0, lpi, pnb)

    ov = jax.jit(obj); og = jax.jit(jax.grad(obj))

    def fg(xn):
        x = jnp.asarray(xn)
        v = float(ov(x)) + pin_pen(xn[3:7], sigma_pin)
        g = np.array(og(x))
        g[3:7] += pin_grad(xn[3:7], sigma_pin)
        if not np.isfinite(v):
            v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g

    bounds = [LOG_R0_B] * 3 + [LOGIT_B] * 4 + [PHI_NB_B]
    best = None
    starts_log = []
    t0 = time.time()
    for k, x0 in enumerate(x0_list):
        r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=600, ftol=1e-9, gtol=1e-6))
        starts_log.append(dict(k=k, fun=float(r.fun), nit=int(r.nit)))
        if best is None or r.fun < best.fun:
            best = r
        print(f"  [{label}] start {k+1}/{len(x0_list)} fun={r.fun:.3f} nit={r.nit} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    x = best.x
    pi = np.asarray(jax.nn.softmax(jnp.asarray(x[3:7])))
    R0 = [float(np.exp(x[i])) for i in range(3)]
    phi_nb = float(x[7])
    data_nll_by_season = per_season_loss(jnp.asarray(x[0:3]), jnp.asarray(x[3:7]),
                                           jnp.asarray(phi_nb))
    return dict(pi=pi.tolist(), R0=dict(zip(SEAS, R0)), phi_nb=phi_nb,
                data_nll_by_season=dict(zip(SEAS, data_nll_by_season)),
                data_nll_total=float(sum(data_nll_by_season)),
                obj_at_map=float(best.fun), starts=starts_log,
                wall_sec=time.time() - t0)


def gen_starts(n_starts, seed):
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(n_starts):
        x0 = np.concatenate([np.log(2.0) + rng.normal(0, 0.2, 3),
                               LOGIT_REF + rng.normal(0, 0.5, 4), [10.0]])
        starts.append(x0)
    return starts


# ═══════════════════════════════════════════════════════════════════════════
# D-3: adult over-prediction scale + directional check (home column sums)
# ═══════════════════════════════════════════════════════════════════════════
def d3_checks(G):
    raw_term, _ = P1.load_raw_ego(P1.TERM_NPZ)
    symA_term = P1.method_A(raw_term, G)
    labels = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
              "45-49","50-54","55-59","60-64","65-69","70+"]
    home_col = {}
    for j in range(15):
        rc = float(raw_term["home"][:, j].sum())
        sc = float(symA_term["home"][:, j].sum())
        home_col[labels[j]] = dict(raw=rc, symA=sc, pct_change=100 * (sc - rc) / rc)

    adult_overest_status = dict(
        required_input_exists=(ED / "symmetric_baseline.json").exists(),
        output_exists=(ED / "adult_overest.json").exists(),
        conclusion=("scripts/adult_overest_diagnosis.py 는 필수 입력 "
                     "outputs/eda/symmetric_baseline.json 과 출력 "
                     "outputs/eda/adult_overest.json 이 모두 존재하지 않아 "
                     "실행/완료 여부와 결론을 확인할 수 없음 — 확인 불가. "
                     "docs/*.md, paper/*.tex 어디에도 이 진단에 대한 언급 없음."),
    )

    return dict(
        scale_note=("simulation_to_hira_by_age_jax 는 daily_inc(15) 를 주 단위로 합산한 "
                     "뒤 gamma_15(보고분율)를 곱하고 HIRA 매핑을 적용함 — 즉 obs 와 같은 "
                     "'보고된 건수' 척도로 이미 변환된 값. D-1의 pred/obs 비는 감염 수를 "
                     "관측 건수로 직접 나눈 것이 아니라 동일 척도(보고 건수) 위에서의 "
                     "비이므로, 3~4배로 크게 나오는 것은 척도 문제가 아니라 실제 과대예측."),
        adult_overest_diagnosis_status=adult_overest_status,
        home_channel_column_sums_toward_age=home_col,
        directional_check_conclusion=(
            "직접 계산 결과: 방식 A 보정 후 home 채널의 '연령 b를 향한' 열합(=b에게 "
            "돌아오는 접촉량, FOI 관련량)이 아동(0-4~25-29, NIMS idx0-5)에서는 "
            "+6.9%~+50.8%로 뚜렷이 증가하고, 핵심 근로연령(30-34,35-39,40-44, "
            "HIRA 18-44군의 주요 구성)에서는 -14.3%~-21.8%로 뚜렷이 감소함. "
            "45-64군(NIMS 45-49~60-64)은 혼재(−12.6%~+9.1%)로 순효과가 작음. "
            "이는 D-1에서 18-44군의 pred/obs 비가 세 시즌 모두 하락한 방향과 "
            "부호가 일치하고, 45-64군의 하락폭이 18-44군보다 작았던 것과도 일치함 — "
            "우연이 아니라 home 채널 상호성 보정이 근로연령층 내부(특히 30대 후반)로 "
            "향하던 접촉을 아동 쪽으로 재배분한 직접적 결과로 보임."),
    )


def main():
    t0 = time.time()
    report = {}

    print("="*70, "\nD-2 MAP refit\n" + "="*70)
    C_all_orig = NS.build()
    G, G_path = P1.load_G("pop_national_2024_01.csv", 51_313_912)
    C_all_A, _ = PD.build_corrected_C_all(C_all_orig, G)

    x0_list = gen_starts(N_STARTS, SEED)
    print(f"[D-2] N_STARTS={N_STARTS} SEED={SEED} (multiseason_joint_sharedpi.py 기존 설정)")

    print("\n[D-2] fitting UNCORRECTED C (with pin) …", flush=True)
    map_orig = run_map(C_all_orig, x0_list, SIGMA_PIN, "orig/pinned")
    print("\n[D-2] fitting CORRECTED C, method A (with pin) …", flush=True)
    map_A = run_map(C_all_A, x0_list, SIGMA_PIN, "A/pinned")

    # posterior sd from NUTS merged
    d = np.load(ED / "nuts_seasonpop_merged.npz")
    pi_post = np.asarray(d["pi"]); log_R0_post = np.asarray(d["log_R0"])
    pi_sd = pi_post.std(axis=0)   # (4,) home,work,school,other
    R0_sd = {SEAS[i]: float(np.exp(log_R0_post[:, i]).std()) for i in range(3)}
    channels = ["home", "work", "school", "other"]

    comparison = {}
    max_ratio = 0.0
    for k, c in enumerate(channels):
        shift = map_A["pi"][k] - map_orig["pi"][k]
        ratio = abs(shift) / pi_sd[k]
        max_ratio = max(max_ratio, ratio)
        comparison[f"pi_{c}"] = dict(uncorrected=map_orig["pi"][k], corrected=map_A["pi"][k],
                                       shift=shift, posterior_sd=float(pi_sd[k]),
                                       shift_over_sd=ratio)
    for s in SEAS:
        shift = map_A["R0"][s] - map_orig["R0"][s]
        comparison[f"R0_{s}"] = dict(uncorrected=map_orig["R0"][s], corrected=map_A["R0"][s],
                                       shift=shift, posterior_sd=R0_sd[s],
                                       shift_over_sd=abs(shift) / R0_sd[s])
    comparison["phi_nb"] = dict(uncorrected=map_orig["phi_nb"], corrected=map_A["phi_nb"],
                                  shift=map_A["phi_nb"] - map_orig["phi_nb"])
    comparison["data_nll_total"] = dict(uncorrected=map_orig["data_nll_total"],
                                          corrected=map_A["data_nll_total"],
                                          shift=map_A["data_nll_total"] - map_orig["data_nll_total"])

    if max_ratio <= 0.5:
        verdict = "pi_essentially_unchanged_le_0.5"
    elif max_ratio <= 1.0:
        verdict = "borderline_0.5_1.0"
    else:
        verdict = "pi_moved_gt_1.0_nuts_refit_needed"

    print(f"\n[D-2] max |Δπ_c| / posterior_sd = {max_ratio:.3f}  verdict={verdict}")
    for c in channels:
        r = comparison[f"pi_{c}"]
        print(f"  π_{c:6s}: {r['uncorrected']:.4f} -> {r['corrected']:.4f}  "
              f"Δ={r['shift']:+.4f}  sd={r['posterior_sd']:.4f}  ratio={r['shift_over_sd']:.3f}")

    report["D2_MAP_pinned"] = dict(uncorrected=map_orig, corrected=map_A,
                                     comparison=comparison, max_pi_shift_over_sd=max_ratio,
                                     verdict=verdict,
                                     note="MAP 대 MAP 비교이며 사후분포 자체를 재적합한 것이 아님.")

    # ---- no-pin variant (reference only) ----
    print("\n" + "="*70, "\nD-2 no-pin (참고용)\n" + "="*70)
    SIGMA_PIN_LOOSE = SIGMA_PIN * 1000.0   # effectively removes the pin
    print("[D-2] fitting UNCORRECTED C (no pin) …", flush=True)
    map_orig_np_ = run_map(C_all_orig, x0_list, SIGMA_PIN_LOOSE, "orig/nopin")
    print("\n[D-2] fitting CORRECTED C, method A (no pin) …", flush=True)
    map_A_np_ = run_map(C_all_A, x0_list, SIGMA_PIN_LOOSE, "A/nopin")
    nopin_comparison = {}
    max_ratio_nopin = 0.0
    for k, c in enumerate(channels):
        shift = map_A_np_["pi"][k] - map_orig_np_["pi"][k]
        ratio = abs(shift) / pi_sd[k]
        max_ratio_nopin = max(max_ratio_nopin, ratio)
        nopin_comparison[f"pi_{c}"] = dict(uncorrected=map_orig_np_["pi"][k],
                                             corrected=map_A_np_["pi"][k], shift=shift,
                                             shift_over_sd=ratio)
    print(f"[D-2 no-pin] max |Δπ_c|/sd = {max_ratio_nopin:.3f} (참고용, 판정에 미사용)")
    report["D2_MAP_nopin_reference"] = dict(uncorrected=map_orig_np_, corrected=map_A_np_,
                                              comparison=nopin_comparison,
                                              max_pi_shift_over_sd=max_ratio_nopin,
                                              note="σ_pin×1000 (사실상 무핀). 판정에는 사용하지 않음 — 참고용.")

    print("\n" + "="*70, "\nD-3\n" + "="*70)
    report["D3"] = d3_checks(G)

    report["meta"] = dict(n_starts=N_STARTS, seed=SEED, seasons=SEAS,
                            log_r0_bounds=list(LOG_R0_B), logit_bounds=list(LOGIT_B),
                            phi_nb_bounds=list(PHI_NB_B), logit_ref=LOGIT_REF.tolist(),
                            sigma_pin=SIGMA_PIN.tolist(),
                            model_note=("make_shared_pi_joint_loss_nb 는 simulate_jax(비-erlang, "
                                         "κ 없음)를 하드코딩하고 있어 nuts_seasonpop.py/D-1과 동역학이 "
                                         "달라 문자 그대로 재사용하지 않음. 대신 D-1과 동일한 erlang "
                                         "presymp + κ + 시즌 인구 우도 레시피로 같은 구조(공유 π, "
                                         "시즌별 R0, φ_NB, L-BFGS-B multistart, 동일 pin/σ/bounds)의 "
                                         "MAP을 직접 구성함."),
                            wall_sec=time.time() - t0,
                            generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    OUT_JSON.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[json] {OUT_JSON}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
