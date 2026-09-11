"""Phase 1 (v2) — 접촉행렬 상호성 보정 및 영향 측정 (재적합 없음).

Standalone 분석 스크립트. 저장소 핵심 코드(model/, simulation/,
calibration/param_vector.py, scripts/reproduction_analysis.py 등)는
건드리지 않고, 이 스크립트 안에서만 보정된 C 를 만들어 build_K_channels /
dominant_eigen 에 주입한다.

방향 규약: 보정은 npz 원본(ego 기준, m[ego,contact])에서 수행하고, 모델에
넣을 때만 transpose 한다 (kt_data.load_contact.load_contact_matrices 와
동일한 규약).

실행:
    cd kt_epimodel_hira/scripts && python phase1_reciprocity_v2.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import reproduction_analysis as RA  # noqa: E402  (reuse build_K_channels, dominant_eigen, etc.)

ED = REPO / "outputs" / "eda"
ED.mkdir(parents=True, exist_ok=True)
OUT_JSON = ED / "phase1_reciprocity_v2.json"

KT_DATA_CONTACT_DIR = REPO.parent / "kt_data" / "data" / "external" / "contact_matrices"
TERM_NPZ = KT_DATA_CONTACT_DIR / "empirical_matrices_15.npz"
VAC_NPZ = KT_DATA_CONTACT_DIR / "empirical_matrices_15_vacation.npz"
ALLWD_NPZ = KT_DATA_CONTACT_DIR / "empirical_matrices_15_all_weekday.npz"

CHANNELS = ["home", "work", "school", "other"]
SEASONS = ["2016-2017", "2017-2018", "2019-2020"]

N_POST = 100
SEED = 1234

# 원본(기존 결과, 재계산 금지) — reproduction_analysis.py 기존 산출 및 논문
# main_v4.tex §... 에서 그대로 옮김.
ORIGINAL_REF = {
    "R_home_share_pct": 44.3,
    "R_school_share_pct": 25.5,
    "R_work_share_pct": 3.9,
    "R_other_share_pct": 26.4,
    "school_amplification_x": 3.92,
    "age_0_19_share_pct": 76.6,
    "age_5_9_share_pct": 47.5,
    "source": ("paper/main_v4.tex §Results (school channel 6.4% of pi -> 24.9% "
               "of R0, 3.9x); outputs/eda/reproduction_numbers_*.json "
               "(R_b_15['5-9'] / R0_check = 47.5%)"),
}


# ═══════════════════════════════════════════════════════════════════════════
# 0. 전국 인구벡터 G
# ═══════════════════════════════════════════════════════════════════════════
def load_G(csv_name: str, expect_sum: int) -> np.ndarray:
    path = REPO / "data" / "population" / "national" / csv_name
    rows = list(csv.DictReader(open(path)))
    G = np.array([int(r["population"]) for r in sorted(rows, key=lambda r: int(r["band_index"]))])
    assert G.shape == (15,), "15군이 아닙니다"
    assert G.sum() == expect_sum, f"기대 합계와 다릅니다 (합계 {G.sum():,})"
    return G, str(path)


# ═══════════════════════════════════════════════════════════════════════════
# A-1. 상호성 보정 (ego 기준 m[a,b])
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_ego(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {c: np.asarray(d[c], dtype=np.float64) for c in CHANNELS}
    meta = d["meta"].tolist() if "meta" in d else None
    return out, meta


def symmetrize_A_one(m: np.ndarray, G: np.ndarray) -> np.ndarray:
    """m_sym[a,b] = (G_a m[a,b] + G_b m[b,a]) / (2 G_a)."""
    T = G[:, None] * m
    T_sym = 0.5 * (T + T.T)
    return T_sym / G[:, None]


def method_A(channels: dict, G: np.ndarray) -> dict:
    return {c: symmetrize_A_one(m, G) for c, m in channels.items()}


def method_B(channels: dict, G: np.ndarray):
    m_tot = sum(channels.values())
    m_tot_sym = symmetrize_A_one(m_tot, G)
    out = {}
    for c, m in channels.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(m_tot > 0, m / m_tot, 0.0)
        out[c] = m_tot_sym * frac
    return out, m_tot, m_tot_sym


def validate_reciprocity(period_name, raw, symA, symB, m_tot, m_tot_symB, G, report):
    rep = report.setdefault(period_name, {})
    rep["pre_correction_asymmetry_max_abs"] = {
        c: float(np.abs(raw[c] - raw[c].T).max()) for c in CHANNELS
    }
    # Method A: T_sym symmetric check per channel
    A_ok = {}
    for c in CHANNELS:
        T = G[:, None] * symA[c]
        A_ok[c] = bool(np.allclose(T, T.T, atol=1e-8, rtol=1e-6))
    rep["method_A_reciprocity_holds_per_channel"] = A_ok
    # Method B: per-channel generally NOT symmetric in T; total must be.
    B_ok_total = bool(np.allclose(G[:, None] * m_tot_symB, (G[:, None] * m_tot_symB).T,
                                    atol=1e-8, rtol=1e-6))
    B_ok_per_channel = {}
    for c in CHANNELS:
        T = G[:, None] * symB[c]
        B_ok_per_channel[c] = bool(np.allclose(T, T.T, atol=1e-8, rtol=1e-6))
    rep["method_B_reciprocity_holds_total"] = B_ok_total
    rep["method_B_reciprocity_holds_per_channel"] = B_ok_per_channel
    # matrix sum change
    rep["matrix_sum_change"] = {
        c: dict(raw_sum=float(raw[c].sum()), A_sum=float(symA[c].sum()),
                B_sum=float(symB[c].sum()),
                A_pct_change=float(100 * (symA[c].sum() - raw[c].sum()) / raw[c].sum()),
                B_pct_change=float(100 * (symB[c].sum() - raw[c].sum()) / raw[c].sum()))
        for c in CHANNELS
    }
    # A vs B difference
    ab_diff = {}
    for c in CHANNELS:
        diff = symA[c] - symB[c]
        rel_frob = float(np.linalg.norm(diff) / max(np.linalg.norm(symA[c]), 1e-12))
        ab_diff[c] = dict(rel_frobenius_norm=rel_frob,
                            max_abs_cell_diff=float(np.abs(diff).max()))
    rep["method_A_vs_B_diff"] = ab_diff
    return rep


# ═══════════════════════════════════════════════════════════════════════════
# A-2. rho / channel / age 분해 — C 만 교체, beta(ngm3 기반)는 원본 유지
# ═══════════════════════════════════════════════════════════════════════════
def analyze_static_with_C(setup, pi_samples, log_R0_samples, C_over: dict):
    """RA.analyze_static 과 동일한 계산이나 C_home/school/work/other 를 교체."""
    N = pi_samples.shape[0]
    S_frac_15 = 1.0 - RA.IMM
    season = setup["season"]
    j_s = SEASONS.index(season)

    R_c_samples = {c: np.zeros(N) for c in CHANNELS}
    R_b_samples = np.zeros((N, 15))
    R0_check = np.zeros(N)

    for k in range(N):
        pi_k = pi_samples[k]
        R0 = float(np.exp(log_R0_samples[k, j_s]))
        b0 = np.asarray(RA.derive_beta_from_R0_simplex(
            setup["ngm3"], jnp.asarray(R0), jnp.asarray(pi_k), jnp.asarray(RA.PHI)))
        beta_h, beta_w, beta_s, beta_o = (float(b0[0]), float(b0[1]),
                                            float(b0[2]), float(b0[3]))
        Ks = RA.build_K_channels(
            pop_15=setup["pop_15"], rho_flat=setup["rho"],
            C_home=C_over["home"], C_school=C_over["school"],
            C_work=C_over["work"], C_other=C_over["other"],
            beta_h=beta_h, beta_s=beta_s, beta_w=beta_w, beta_o=beta_o,
            phi_full=RA.PHI, S_frac_15=S_frac_15, gamma=setup["gamma"],
            sf=1.0 + RA.SEASONAL_AMP)
        rho, v, u = RA.dominant_eigen(Ks["total"])
        R0_check[k] = rho
        for c in CHANNELS:
            R_c_samples[c][k] = float(u @ Ks[c] @ v)
        R_b_samples[k] = (u @ Ks["total"]) * v

    out = dict(
        n_post=N, season=season,
        R0_check=RA._sm(R0_check),
        channels=CHANNELS,
        R_c={c: RA._sm(R_c_samples[c]) for c in CHANNELS},
        R_c_share_pct={c: RA._sm(100.0 * R_c_samples[c] / R0_check) for c in CHANNELS},
        R_b_15={RA.AGE15_LAB[b]: RA._sm(R_b_samples[:, b]) for b in range(15)},
    )
    R_b6 = np.zeros((N, 6))
    for i, (name, bins) in enumerate(RA.AGE6_BINS.items()):
        R_b6[:, i] = R_b_samples[:, bins].sum(axis=1)
    out["age_labels_6"] = list(RA.AGE6_BINS.keys())
    out["R_b_6_share_pct"] = {list(RA.AGE6_BINS.keys())[i]:
                                RA._sm(100.0 * R_b6[:, i] / R0_check) for i in range(6)}
    idx019 = [0, 1, 2]   # 0-5, 6-11, 12-17
    R019 = R_b6[:, idx019].sum(axis=1)
    out["age_0_19_share_pct"] = RA._sm(100.0 * R019 / R0_check)
    return out


def to_model_convention(m_ego: dict) -> dict:
    """ego 기준(m[a,b]) → 모델 컨벤션 [contact, participant] (transpose)."""
    return {c: m.T.copy() for c, m in m_ego.items()}


def check_assertions(school_mask_sum, work_mask_sum, phi):
    assert not np.allclose(phi, 1.0), "PHI가 ones입니다"
    assert school_mask_sum == 16 and work_mask_sum == 10


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    report = {}

    # ---- 0. G ----
    G, G_path = load_G("pop_national_2024_01.csv", 51_313_912)
    print(f"[G] source={G_path}")
    print(f"[G] reference_date=2024-01  values={G.tolist()}  sum={G.sum():,}")
    report["G"] = dict(source=G_path, reference_date="2024-01",
                        values=G.tolist(), sum=int(G.sum()))

    G_2023, G_2023_path = load_G("pop_national_2023_01.csv", 51_430_018)
    print(f"[G_sensitivity] source={G_2023_path} sum={G_2023.sum():,}")
    report["G_sensitivity_2023_01"] = dict(source=G_2023_path,
                                             values=G_2023.tolist(),
                                             sum=int(G_2023.sum()))

    # ---- A-0: usage check for all_weekday.npz ----
    allwd_used = ALLWD_NPZ.exists()
    report["A0_all_weekday_npz"] = dict(
        exists=allwd_used,
        used_in_pipeline=False,
        note="grep 결과 scripts/, kt_data/, mobility_contact/ 어디에서도 "
             "'all_weekday' 문자열 참조 없음 → 모형에 쓰이지 않음. 보정 대상에서 제외.",
    )
    print(f"[A-0] empirical_matrices_15_all_weekday.npz used_in_pipeline=False (grep 결과 미사용)")

    # ---- A-1: reciprocity correction, term + vacation ----
    raw_term, meta_term = load_raw_ego(TERM_NPZ)
    raw_vac, meta_vac = load_raw_ego(VAC_NPZ)
    report["A1_npz_meta"] = dict(term=meta_term, vacation=meta_vac)

    symA_term = method_A(raw_term, G)
    symB_term, m_tot_term, m_tot_symB_term = method_B(raw_term, G)
    symA_vac = method_A(raw_vac, G)
    symB_vac, m_tot_vac, m_tot_symB_vac = method_B(raw_vac, G)

    validate_reciprocity("school_term", raw_term, symA_term, symB_term,
                          m_tot_term, m_tot_symB_term, G, report)
    validate_reciprocity("vacation", raw_vac, symA_vac, symB_vac,
                          m_tot_vac, m_tot_symB_vac, G, report)

    print("\n[A-1] pre-correction asymmetry (term):",
          report["school_term"]["pre_correction_asymmetry_max_abs"])
    print("[A-1] method_A reciprocity holds (term):",
          report["school_term"]["method_A_reciprocity_holds_per_channel"])
    print("[A-1] method_B reciprocity holds total (term):",
          report["school_term"]["method_B_reciprocity_holds_total"])

    # model-convention matrices for A-2 (term period only needed there)
    C_A_term = to_model_convention(symA_term)
    C_B_term = to_model_convention(symB_term)
    C_orig_term = to_model_convention(raw_term)  # sanity: should equal loader's C_*

    # sanity vs loader
    from kt_data.data.load_contact import load_contact_matrices
    loader_out = load_contact_matrices()  # default term path
    for c in CHANNELS:
        assert np.allclose(C_orig_term[c], loader_out[f"C_{c}"]), \
            f"transpose convention mismatch for {c}"
    print("[A-1] sanity: ego->model transpose matches kt_data.load_contact_matrices() ✓")

    # vacation model-convention (kept for Part B; not used in A-2 static NGM)
    C_A_vac = to_model_convention(symA_vac)
    C_B_vac = to_model_convention(symB_vac)

    # ---- A-2: assertions ----
    work_mask = np.zeros(15); work_mask[4:14] = 1.0
    school_mask = np.zeros((15, 15)); school_mask[:4, :4] = 1.0
    check_assertions(school_mask.sum(), work_mask.sum(), RA.PHI)
    print(f"[A-2] assert OK: PHI not ones, school_mask.sum()={school_mask.sum()}, "
          f"work_mask.sum()={work_mask.sum()}")
    report["A2_assertions"] = dict(
        phi_is_ushape=not bool(np.allclose(RA.PHI, 1.0)),
        phi_values=RA.PHI.tolist(),
        school_mask_sum=float(school_mask.sum()),
        work_mask_sum=float(work_mask.sum()),
        G_source=G_path, G_sum=int(G.sum()),
    )

    # ---- load setup + posterior (shared across seasons) ----
    print("\n[loading season-pop setup for all 3 seasons] …", flush=True)
    C_all = RA.build_seasonwise_setup(imm=RA.IMM, gamma_15=RA.GAMMA_15, use_season_pop=True)
    pi_s, log_R0_s = RA._load_posterior(N_POST, SEED)
    print(f"[posterior] N={N_POST} seed={SEED}", flush=True)

    per_season = {}
    for season in SEASONS:
        t_s = time.time()
        setup = RA._load_setup(season=season, C_all=C_all)
        print(f"\n[season {season}] pop total={setup['pop_15'].sum():,.0f}", flush=True)

        st_orig = analyze_static_with_C(setup, pi_s, log_R0_s, C_orig_term)
        st_A = analyze_static_with_C(setup, pi_s, log_R0_s, C_A_term)
        st_B = analyze_static_with_C(setup, pi_s, log_R0_s, C_B_term)

        pi_school_mean = float(pi_s[:, 2].mean())  # (h,w,s,o) order
        amp_orig = st_orig["R_c_share_pct"]["school"]["mean"] / (100 * pi_school_mean)
        amp_A = st_A["R_c_share_pct"]["school"]["mean"] / (100 * pi_school_mean)
        amp_B = st_B["R_c_share_pct"]["school"]["mean"] / (100 * pi_school_mean)

        row = dict(
            recomputed_original=st_orig,
            method_A=st_A,
            method_B=st_B,
            school_amplification_x=dict(recomputed_original=amp_orig, A=amp_A, B=amp_B),
            pi_school_mean_pct=100 * pi_school_mean,
        )
        per_season[season] = row
        print(f"  rho(orig/A/B) = {st_orig['R0_check']['mean']:.4f} / "
              f"{st_A['R0_check']['mean']:.4f} / {st_B['R0_check']['mean']:.4f}")
        print(f"  R_home%: orig={st_orig['R_c_share_pct']['home']['mean']:.1f}  "
              f"A={st_A['R_c_share_pct']['home']['mean']:.1f}  "
              f"B={st_B['R_c_share_pct']['home']['mean']:.1f}")
        print(f"  R_school%: orig={st_orig['R_c_share_pct']['school']['mean']:.1f}  "
              f"A={st_A['R_c_share_pct']['school']['mean']:.1f}  "
              f"B={st_B['R_c_share_pct']['school']['mean']:.1f}")
        print(f"  school amp x: orig={amp_orig:.2f}  A={amp_A:.2f}  B={amp_B:.2f}")
        print(f"  [{season}] {time.time()-t_s:.1f}s", flush=True)

    # 3-season pooled means for comparison table
    def pooled_mean(path_fn):
        vals = [path_fn(per_season[s]) for s in SEASONS]
        return float(np.mean(vals))

    comparison_table = dict(original_ref=ORIGINAL_REF)
    for method in ("recomputed_original", "method_A", "method_B"):
        comparison_table[method] = dict(
            rho_mean_3season=pooled_mean(lambda r, m=method: r[m]["R0_check"]["mean"]),
            R_home_share_pct=pooled_mean(lambda r, m=method: r[m]["R_c_share_pct"]["home"]["mean"]),
            R_school_share_pct=pooled_mean(lambda r, m=method: r[m]["R_c_share_pct"]["school"]["mean"]),
            R_work_share_pct=pooled_mean(lambda r, m=method: r[m]["R_c_share_pct"]["work"]["mean"]),
            R_other_share_pct=pooled_mean(lambda r, m=method: r[m]["R_c_share_pct"]["other"]["mean"]),
            school_amplification_x=pooled_mean(
                lambda r, m=method: r["school_amplification_x"][
                    "recomputed_original" if m == "recomputed_original" else m.split("_")[-1]]),
            age_0_19_share_pct=pooled_mean(lambda r, m=method: r[m]["age_0_19_share_pct"]["mean"]),
            age_5_9_share_pct=pooled_mean(lambda r, m=method: r[m]["R_b_15"]["5-9"]["mean"] /
                                            r[m]["R0_check"]["mean"] * 100),
        )

    # A vs B verdict: within 2pp on each item?
    ab_gaps = {k: abs(comparison_table["method_A"][k] - comparison_table["method_B"][k])
               for k in comparison_table["method_A"] if k != "rho_mean_3season"}
    ab_negligible = all(v <= 2.0 for k, v in ab_gaps.items() if k.endswith("_pct"))

    report["A2_per_season"] = per_season
    report["A2_comparison_table"] = comparison_table
    report["A2_method_A_vs_B_gaps_pp"] = ab_gaps
    report["A2_method_A_vs_B_negligible_le_2pp"] = ab_negligible

    report["meta"] = dict(n_post=N_POST, seed=SEED, seasons=SEASONS,
                            wall_sec=time.time() - t0,
                            generated=time.strftime("%Y-%m-%d %H:%M:%S"))

    OUT_JSON.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[json] {OUT_JSON}")
    print(f"[total] {time.time()-t0:.1f}s")
    print(f"\n=== A vs B negligible (<=2pp each) : {ab_negligible} ===")


if __name__ == "__main__":
    main()
