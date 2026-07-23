"""v4 파라미터 인벤토리 — 코드 실측 → JSON + 콘솔.

실행 경로: kappa_no_eta_presymp.py / nuts_v4.py (v4 확정 구성).
모듈: erlang_presymp, foi_jax, dynamics_jax, solver_jax, numpyro_model,
      simple_model, model/parameters.

문서 대조:
  docs/parameter_justification.md, docs/PROJECT_REFERENCE.md,
  docs/PRIOR_SPECIFICATION.md, docs/GAMMA_STRATEGY.md, docs/CONVENTIONS.md.

출력: outputs/eda/parameter_inventory.json + 콘솔.
"""
import os, json
os.environ["JAX_PLATFORMS"] = "cpu"
from pathlib import Path
import numpy as np

# 모델 상수
from kt_epimodel_hira.model.parameters import (
    DiseaseParameters, VaccinationParameters, CalibrationParameters,
    PolicyParameters, EmploymentParameters, ModelParameters,
    GAMMA_AGE_GROUPS, N_AGE, _SEASON_START_ISO_WEEK,
)
from kt_epimodel_hira.calibration.simple_model import (
    R0_IMMUNITY_PROFILE, build_aggregated_inputs,
)
from kt_epimodel_hira.calibration.hira_target import (
    HIRA_AGE_GROUPS, HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.jax_model.foi_jax import (
    STUDENT_SLICE, WORKER_SLICE, N_AGE as N_AGE_FOI,
)
from kt_epimodel_hira.jax_model.erlang import N_STAGES
from kt_epimodel_hira.jax_model.erlang_presymp import W_PRESYMP, ngm_factor

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "eda" / "parameter_inventory.json"

AGE_LABELS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
               "40-44","45-49","50-54","55-59","60-64","65-69","70+"]

# ─── v4 확정 값 (스크립트 상수) ───
V4 = dict(
    # φ 선형 (kappa_no_eta_presymp / nuts_v4 → final_pipeline_confirmed PHI)
    PHI = [2.0, 1.75, 1.5, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0+0.5*1/6, 1.0+0.5*2/6, 1.0+0.5*3/6, 1.0+0.5*4/6,
            1.0+0.5*5/6, 1.5],
    # γ_report (연령별)
    GAMMA_15 = [0.40, 0.40, 0.25, 0.18] + [0.18]*9 + [0.25, 0.25],
    # R(0) 초기면역
    IMM = [0.10]*4 + [0.40]*5 + [0.60]*4 + [0.65]*2,
    # κ (v4: η 제거)
    KAP = [0.34]*4 + [0.40]*10 + [0.0],
    # baseline p_work=p_school
    BASE = 0.6,
    # presymp
    W_PRESYMP = W_PRESYMP,
    NGM_FACTOR = ngm_factor(W_PRESYMP),
    # 정책 시간창 (season day)
    TERM_WIN = [70.0, 113.0],
    VAC_WIN = [113.0, 183.0],
    # 3-fit 시즌
    SEASONS = ["2016-2017", "2017-2018", "2019-2020"],
    # first_peak_only fit
    FIRST_PEAK_END_WEEK = 26,
    # pin
    SIGMA_PIN = [0.15, 0.10, 0.05, 0.15],   # h, w, s, o
    PI_REF_WORK = 0.29,
    # optimizer bounds
    LOG_R0_B = [float(np.log(0.8)), float(np.log(3.5))],
    PHI_NB_B = [1e-3, 1e6],
    LOGIT_PI_B = [-10.0, 10.0],
    # NUTS
    NUTS_WARMUP = 500, NUTS_SAMPLES = 500, NUTS_CHAINS = 4,
    NUTS_TREE_DEPTH = 8, NUTS_TARGET_ACCEPT = 0.9,
    NUTS_PRIOR_log_R0 = "Normal(log(2.0), 0.5)",
    NUTS_PRIOR_logit_pi = "Uniform(-10, 10) + factor: N(centered logit_ref, sigma_pin)",
    NUTS_PRIOR_log_phi_nb = "Normal(log(10), 1.5)",
)

# 디스크에서 실측 (population, rho)
inputs = build_aggregated_inputs()
pop_15 = np.asarray(inputs["pop_15"])
pop_15_flat = pop_15.sum(1) if pop_15.ndim == 2 else pop_15
RHO = np.asarray(inputs["rho"])[0].tolist()   # (15,)

# DiseaseParameters defaults
dp = DiseaseParameters()
vp = VaccinationParameters()
cal = CalibrationParameters()

# ─── 카테고리별 인벤토리 ───
INV = dict(meta=dict(
    version="v4",
    entry_points=["scripts/kappa_no_eta_presymp.py", "scripts/nuts_v4.py"],
    model_module="src/kt_epimodel_hira/jax_model/erlang_presymp.py",
    seasons=V4["SEASONS"],
))

# 1. 역학
INV["epi"] = dict(
    sigma=dict(desc="E→I 전환율", value=dp.sigma, unit="day⁻¹",
                source="model/parameters.py DiseaseParameters.sigma", fixed=True,
                literature="1/σ=2일 잠복기 (표준 문헌)"),
    gamma=dict(desc="회복률 (총 감염기간⁻¹)", value=dp.gamma, unit="day⁻¹",
                source="model/parameters.py DiseaseParameters.gamma", fixed=True,
                literature="1/γ=4일 감염기간"),
    N_STAGES=dict(desc="Erlang I 단계수 (I₁,I₂,I₃)",
                   value=N_STAGES, unit="—",
                   source="jax_model/erlang.py N_STAGES=3", fixed=True,
                   literature="분포 좁혀 세대 간격 분산 감소"),
    stage_rate=dict(desc="stage 전이율 3γ (평균 감염기간 1/γ 보존)",
                     value=3 * dp.gamma, unit="day⁻¹",
                     source="jax_model/erlang.py L98 rate=N_STAGES*gamma", fixed=True),
    w_presymp=dict(desc="I₁(presymp) 상대 전파력", value=V4["W_PRESYMP"], unit="—",
                    source="jax_model/erlang_presymp.py W_PRESYMP=0.22", fixed=True,
                    literature="Nature Health 2026 (홍콩 748가구 인플루엔자A 9.6% presymp) w=0.22"),
    ngm_factor=dict(desc="실효 전파시간 factor (w+2)/3",
                     value=V4["NGM_FACTOR"], unit="—",
                     source="jax_model/erlang_presymp.py ngm_factor(W)",
                     fixed=True,
                     literature="derive_beta_from_R0_simplex 결과 / factor 로 R0 정합"),
    VE=dict(desc="백신효과", value=vp.VE, unit="—",
             source="model/parameters.py VaccinationParameters.VE=0.5",
             fixed=True, literature="계절인플루엔자 표준 assumption"),
    vax_kernel=dict(desc="v(t) Gaussian density × annual_coverage",
                     peak_iso_week=vp.peak_iso_week,
                     spread_weeks=vp.spread_weeks,
                     unit="peak: ISO week; sigma: weeks",
                     source="model/parameters.py L282-292; dynamics_jax.py L22-32",
                     fixed=True,
                     note="주의: ∫dens dt over [0,365] = 0.9332 (tail 6.7% 손실). "
                          "A-fix: annual_coverage = -ln(1-C) 로 correction 하지만 유한적분 잔여 오차 있음"),
    annual_coverage=dict(desc="15군 연간 접종률",
                          value=vp.annual_coverage.tolist(),
                          unit="[0,1]",
                          source="model/parameters.py _default_annual_coverage L245-256",
                          fixed=True, literature="한국 시즌 인플루엔자 잠정치"),
)

# 2. 연령별 벡터
INV["age_vectors"] = dict(
    phi_susc=dict(desc="상대 감수성 (선형)", value=V4["PHI"],
                   labels=AGE_LABELS,
                   source="scripts/final_pipeline_confirmed.py L50-51 (PHI)",
                   fixed=True,
                   note="0-19 감소 (2.0→1.25), 20-44 = 1.0, 45+ 선형 증가 (1.083→1.5)"),
    gamma_report_15=dict(desc="연령별 보고율",
                          value=V4["GAMMA_15"], labels=AGE_LABELS,
                          source="scripts/kappa_no_eta_presymp.py GAMMA=... , "
                                 "카테고리: child 0-11 = 0.40 / 12-17 = 0.25 / "
                                 "adult 18-64 = 0.18 / elder 65+ = 0.25",
                          fixed=True,
                          literature="Reed 2012 (CDC) 원값, docs/GAMMA_STRATEGY.md"),
    R0_immunity=dict(desc="시즌 초 R compartment 배치 (기존 노출+백신)",
                      value=V4["IMM"], labels=AGE_LABELS,
                      source="scripts/kappa_no_eta_presymp.py IMM=... "
                             "(final v3+에서 성인 상향 반영)",
                      fixed=True,
                      literature="POLYMOD + Vandegrift 2014 elderly cross-immunity, "
                                 "docs/PRIOR_SPECIFICATION.md Appendix A"),
    kappa=dict(desc="home spillover 계수 (η 제거, 체류시간 증가율만)",
                value=V4["KAP"], labels=AGE_LABELS,
                source="scripts/kappa_no_eta_presymp.py KAP=... "
                       "(학생 0.34, 성인 0.40, 70+ 0.0)",
                fixed=True,
                literature="Ferguson 2006 (재택시 가족접촉 +50~100%), "
                           "prev 문서 v3까지 η 곱 (0.29/0.30) 사용, v4에서 η 제거"),
    rho_employment=dict(desc="연령별 취업률 (수도권 실측)",
                         value=RHO, labels=AGE_LABELS,
                         source="calibration/simple_model.py L79-85, "
                                "kt_data mois_population + 통계청 employment",
                         fixed=True, note="worker slice 4-13 (idx) 에만 nonzero"),
    student_slice=dict(desc="학생 슬라이스", value=[0, 4],
                        source="jax_model/foi_jax.py STUDENT_SLICE=slice(0,4)"),
    worker_slice=dict(desc="근로자 슬라이스", value=[4, 14],
                       source="jax_model/foi_jax.py WORKER_SLICE=slice(4,14)"),
)

# 3. 접촉·공간
INV["contact_space"] = dict(
    C_term=dict(desc="학기 접촉행렬 (home/work/school/other 4채널)",
                 path="../kt_data/data/external/contact_matrices/empirical_matrices_15.npz",
                 shape=[15, 15],
                 source="load_contact_matrices(); transpose 후 [contact, participant]",
                 fixed=True,
                 literature="NIMS 접촉 조사 (한국 표준)"),
    C_vacation=dict(desc="방학 접촉행렬 (4채널)",
                     path="../kt_data/data/external/contact_matrices/empirical_matrices_15_vacation.npz",
                     shape=[15, 15],
                     source="load_contact_matrices(path=VAC_NPZ)",
                     fixed=True),
    Ct_switch=dict(desc="C(t) = (1-h)·C_term + h·C_vacation, 사다리꼴 h(t)",
                    h_start_day=113.0,   # ≈ Dec 23
                    h_min_start_day=127.0,   # ≈ Jan 6
                    h_min_end_day=162.0,     # ≈ Feb 10
                    h_end_day=183.0,         # ≈ Mar 2
                    source="jax_model/foi_jax.py vacation_weight L64-88; "
                           "dynamics_jax.py L66-88",
                    fixed=True,
                    note="season day 0 = ISO week 36 (Mon), 2019: 9월 2일"),
    population=dict(desc="인구 N_a (수도권 합산)",
                     path="../kt_data/data/mapping/mois_population_202301.parquet",
                     ref_year="2023-01",
                     total=int(pop_15_flat.sum()),
                     by_age_15={AGE_LABELS[i]: int(pop_15_flat[i]) for i in range(15)},
                     source="calibration/simple_model.py build_aggregated_inputs "
                            "→ pop_aggregated = N_mat.sum(axis=0).reshape(15,1)",
                     fixed=True,
                     note="시즌별 인구 미반영 (2023 단일). sanity check: max 상대변화 5.4%, "
                          "재분배 방향 부호 유지 (age_structure_sanity.py)"),
    n_admdong=dict(desc="공간 단위 (수도권 3시도 합산)",
                    value=1,
                    source="simple_model.py build_aggregated_inputs pop_15 shape=(15,1)",
                    fixed=True,
                    note="metapop 아님, 서울+경기+인천 합산"),
    sudogwon_sido=dict(desc="수도권 시도코드",
                        value=[11, 28, 41],
                        source="kt_data.SUDOGWON_SIDO_CODES",
                        note="11=서울, 28=인천, 41=경기"),
)

# 4. 정책
INV["policy"] = dict(
    baseline_p_work=dict(desc="baseline 근무 잔존율",
                          value=V4["BASE"], source="scripts/*.py BASE=0.6",
                          fixed=True,
                          note="1.0 = 완전 근무, 0.6 = 40% 결근이 baseline"),
    baseline_p_school=dict(desc="baseline 등교 잔존율",
                            value=V4["BASE"], source="scripts/*.py BASE=0.6",
                            fixed=True),
    p_policy=dict(desc="정책 강도 (병가/학교결석 grid)",
                   value=[0.6, 0.4, 0.2, 0.0], source="scripts/*.py PL=[...]",
                   note="0.6 = 무정책 (baseline과 동일 → averted 0), "
                        "0.4 = 대표 강도 (본문 결과), "
                        "0.0 = 완전 결근"),
    term_window=dict(desc="학기 창 (병가/학교 정책 발효 기간)",
                      value=V4["TERM_WIN"], unit="season day",
                      source="scripts/*.py TERM=(70.0, 113.0)",
                      note="approx Nov 10 ~ Dec 23 (2019 시즌)"),
    vacation_window=dict(desc="방학 창",
                          value=V4["VAC_WIN"], unit="season day",
                          source="scripts/*.py VAC=(113.0, 183.0)"),
    policy_window_ramp=dict(desc="정책 창 사다리꼴 램프",
                             value=3.0, unit="days",
                             source="dynamics_jax.py policy_ramp_days=3.0"),
    mu=dict(desc="κ 스케일 (mu=1 → κ 그대로)",
             value=1.0,
             source="암시적: κ 벡터에 별도 스케일 없음",
             note="향후 mu 옵션 도입 시 스케일링 매개변수"),
    school_holiday_realloc=dict(desc="방학→spillover 라우팅 (v4에선 C(t) 스위칭 사용)",
                                 value=0.0,
                                 source="dynamics_jax.py 기본값. C_vac 지정시 이 옵션 무시",
                                 note="v4는 C_home_vac 등 활성화 → term↔vac 스위칭 경로 사용, "
                                      "school_holiday_realloc 은 legacy path (사용 안 함)"),
)

# 5. 계절성
INV["seasonality"] = dict(
    mode=dict(value="cosine", source="model/parameters.py DiseaseParameters.seasonality_mode"),
    amp=dict(value=dp.seasonality_amp, source="DiseaseParameters.seasonality_amp",
              fixed=True, note="0.7 = 최대/최소 진폭 ± 70% (기본); scripts는 S.AMP=0.9 대체 사용 여부 확인 필요"),
    amp_script=dict(desc="script(final_pipeline_confirmed → sens_workshare_kappa_v2 S.AMP) 실제 사용값",
                     value=0.9,
                     source="scripts/sens_workshare_kappa_v2.py AMP=0.9",
                     note="★ ModelParameters 기본(0.7) vs script(0.9) 불일치"),
    base=dict(value=dp.seasonality_base,
               source="DiseaseParameters.seasonality_base=1.0",
               note="factor(t) = base + amp·cos(...) = 1.0 + 0.9·cos"),
    peak_day=dict(value=dp.seasonality_peak_day, unit="season day (0=Sep 1 근사)",
                   source="DiseaseParameters.seasonality_peak_day=105.0",
                   note="≈ Dec 15 (season 원점 = ISO week 36 = Sep 2 (2019))"),
    period=dict(value=dp.seasonality_period, unit="days",
                 source="DiseaseParameters.seasonality_period=365.0"),
)

# 6. 추정 파라미터 (fit / NUTS 대상)
INV["estimated"] = dict(
    pi_4=dict(desc="채널 mix (simplex, 4채널 shared)",
               fit_via="fit_season() point / NUTS logit_pi + softmax",
               pin_ref_work=V4["PI_REF_WORK"],
               sigma_pin=V4["SIGMA_PIN"],
               channels=["home","work","school","other"],
               source="scripts/kappa_no_eta_presymp.py fit L118; nuts_v4.py"),
    R0=dict(desc="basic reproduction number (시즌별)",
             fit_via="point / NUTS log_R0[i]",
             bounds=V4["LOG_R0_B"],
             prior="Normal(log(2.0), 0.5)",
             source="derive_beta_from_R0_simplex; NGM 1-homogeneous inversion"),
    phi_nb=dict(desc="NB 관측 concentration (전 시즌 shared)",
                 fit_via="point / NUTS log_phi_nb",
                 bounds=V4["PHI_NB_B"],
                 prior="Normal(log(10), 1.5)"),
    NUTS_config=dict(
        warmup=V4["NUTS_WARMUP"], samples=V4["NUTS_SAMPLES"],
        chains=V4["NUTS_CHAINS"], max_tree_depth=V4["NUTS_TREE_DEPTH"],
        target_accept_prob=V4["NUTS_TARGET_ACCEPT"],
        source="scripts/nuts_v4.py run_mcmc / do_full",
    ),
    NUTS_priors=dict(
        log_R0=V4["NUTS_PRIOR_log_R0"],
        logit_pi=V4["NUTS_PRIOR_logit_pi"],
        log_phi_nb=V4["NUTS_PRIOR_log_phi_nb"],
        note="logit_pi centered_logit_pi = logit_pi - mean(logit_pi), "
             "then factor = -0.5·Σ((centered - logit_ref)/σ_pin)²",
    ),
)

# 7. 초기조건
INV["initial_conditions"] = dict(
    seed=dict(desc="시즌 초 감염 seed (I₀ 총량)",
               method="estimate_initial_infected_from_hira",
               source="calibration/simple_model.py L98-190",
               note="HIRA 첫 3주 발생을 gamma_report_15 로 나눠 감염자 추정, "
                    "15군에 인구비 분배"),
    seed_e_factor=dict(desc="E₀ = seed_e_factor × I₀",
                        value=0.5,
                        source="scripts/*.py _build_initial_state_with_age_seed"
                               "(seed_e_factor=0.5)"),
    initial_vaccinated_fraction=dict(desc="V(0) fraction (0 = R(0) 만 사용)",
                                       value=0.0,
                                       source="scripts/*.py "
                                              "initial_vaccinated_fraction=0.0"),
    erlang_split=dict(desc="I₀ → I₁, I₂, I₃ 균등 분배",
                       value="I/3 per stage",
                       source="jax_model/erlang.py split_seed_to_erlang L30-33"),
    R_init=dict(desc="R₀ = R0_IMMUNITY_PROFILE × pop_a",
                 source="calibration/simple_model.py "
                        "_build_initial_state_with_age_seed L263-285"),
)

# 8. 수치·솔버
INV["numeric_solver"] = dict(
    solver=dict(value="Dopri5 (기본)", alt="Tsit5 옵션",
                 source="jax_model/erlang.py simulate_jax_erlang, method='Dopri5'"),
    rtol=dict(value=1e-4, source="simulate_jax_erlang rtol=1e-4"),
    atol=dict(value=1e-6, source="simulate_jax_erlang atol=1e-6"),
    t_span_days=dict(value=[0.0, 364.0], unit="season day",
                       source="simulate_jax_erlang t_span=(0.0, 364.0)"),
    dt0=dict(value=0.1, source="simulate_jax_erlang dt0=0.1"),
    max_steps=dict(value=200_000, source="simulate_jax_erlang max_steps=200_000"),
    stepsize_controller=dict(value="PIDController (adaptive)",
                              source="diffrax PIDController(rtol,atol)"),
    saveat=dict(value="1-day grid (0..364)",
                 source="t_eval = jnp.arange(t_span[0], t_span[1]+1.0, 1.0)"),
    first_peak_only=dict(value=True, end_week=V4["FIRST_PEAK_END_WEEK"],
                          desc="fit 영역 첫 봉우리만 (week < 26)",
                          source="load_hira_target_by_age(first_peak_only=True, "
                                 "first_peak_end_week=26)"),
    obs_min_rate=dict(value=0.01, source="nb_nll_jax min_rate=0.01"),
    n_admdong=dict(value=1, source="simple_model.py 수도권 합산 (n=1)"),
    NGM_seasonal_factor=dict(value=f"1.0 + AMP = {1.0 + 0.9}",
                              source="make_ngm_eigvalue_fn(seasonal_factor=1.0+S.AMP)",
                              note="NGM 계산시 peak season factor 사용"),
)

# ─── 문서 대조 (불일치·누락·legacy) ───
INV["discrepancies"] = [
    dict(item="seasonality_amp",
         doc="DiseaseParameters.seasonality_amp = 0.7 (기본)",
         code_used="scripts에서 S.AMP=0.9 로 override",
         severity="high (0.9로 fit·NUTS 진행)",
         resolution="논문 표기 값은 0.9 로 기재"),
    dict(item="seasonality_base",
         doc="DiseaseParameters.seasonality_base = 1.0",
         code_used="1.0 (default 유지)",
         severity="none"),
    dict(item="school_holiday_realloc",
         doc="v3 pipeline에서 사용된 legacy 파라미터",
         code_used="v4는 C_vac 활성 → 이 옵션 미사용 (0.0 default)",
         severity="none (legacy)",
         resolution="논문에는 C(t) term↔vacation 스위칭만 언급, "
                    "school_holiday_realloc 는 supplement에서만 배경 서술"),
    dict(item="mu (κ 스케일)",
         doc="planning 단계에서 언급됨",
         code_used="현재 구현 없음 (κ 벡터에 스케일 미적용, 사실상 μ=1)",
         severity="none",
         resolution="논문에는 명시적 언급 불필요"),
    dict(item="κ_home_spill",
         doc="planning 명칭",
         code_used="현재 명칭 = kappa (KAP)",
         severity="none (naming)",
         resolution="논문 표에서 명명 통일 필요"),
    dict(item="phi_susc PHI",
         doc="U-shape (v3까지) vs 선형 (v4)",
         code_used="v4: 선형 [2.0, 1.75, 1.5, 1.25, 1.0×5, ↗1.5]",
         severity="version 표기",
         resolution="v4 선형 명시"),
    dict(item="baseline p_work·p_school",
         doc="일부 문서 p=1.0",
         code_used="v4: p=0.6 (대칭 baseline)",
         severity="high — 결과 해석 차이 큼",
         resolution="baseline 0.6 통일, 정책 강도 = 0.6 → 0.4/0.2/0.0"),
    dict(item="κ (η 제거)",
         doc="v3까지 κ_체류 × η = 0.29/0.30",
         code_used="v4: η 제거, κ = 0.34/0.40",
         severity="version 표기",
         resolution="v4 값 (0.34/0.40/0) 명시, η 제거 이유 노트"),
    dict(item="presymptomatic w",
         doc="이전 문서 없음 (신규 도입)",
         code_used="w=0.22 (Nature Hlth 2026)",
         severity="문서 추가 필요",
         resolution="Methods 신규 문단 + 인용 추가"),
    dict(item="관측 시점",
         doc="이전 문서: 감염 시점 (E 유입)",
         code_used="v4: 발현 시점 (I₁→I₂ 유입, daily_new_onset)",
         severity="문서 추가 필요",
         resolution="관측 정의 발현 시점으로 명시"),
    dict(item="C(t) term↔vacation",
         doc="이전 문서 없거나 realloc으로만 서술",
         code_used="v4: 사다리꼴 h(t) 로 실제 두 행렬 blend",
         severity="문서 추가 필요",
         resolution="공식 C(t)=(1-h)C_term + h·C_vac 명시"),
    dict(item="시즌별 인구",
         doc="특별 언급 없음",
         code_used="2023-01 단일 인구, 3시즌 공유",
         severity="sanity 확인 완료 (변화 <5.4%, 부호 유지)",
         resolution="supplement에 sensitivity 결과 첨부"),
]

INV["legacy_unused"] = [
    dict(item="school_holiday_amp / _start_day / _min_start_day / ... / _realloc",
         file="jax_model/dynamics_jax.py L60-68",
         reason="C(t) term↔vac 스위칭 도입 후 legacy. C_vac 지정시 무시.",
         v4_active=False),
    dict(item="model/foi.py (numpy 참조 구현)",
         file="src/kt_epimodel_hira/model/foi.py",
         reason="jax_model/foi_jax.py 이 실 사용. numpy는 regression 참조용.",
         v4_active=False),
    dict(item="erlang_E2I3 (E 2-stage + I 3-stage)",
         file="jax_model/erlang.py L142+",
         reason="v4는 basic Erlang I₃ 사용 (E 1-stage). E2I3 는 실험만.",
         v4_active=False),
]

# ─── 콘솔 print ───
def _fmt(v):
    if isinstance(v, (list, tuple)):
        if len(v) <= 5 or all(isinstance(x, (int, float)) for x in v):
            if len(v) > 8:
                return f"[{v[0]}, {v[1]}, ..., {v[-2]}, {v[-1]}] (n={len(v)})"
            return str(v)
        return f"list(len={len(v)})"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)

def print_section(name, section):
    print("\n" + "═"*90)
    print(f" {name}")
    print("═"*90)
    for k, entry in section.items():
        if isinstance(entry, dict) and "desc" in entry:
            val = entry.get("value", "—")
            print(f"  {k:>30s} = {_fmt(val):<35s}  # {entry.get('desc','')}")
            for sub in ("unit","source","literature","note","fixed"):
                if sub in entry and entry[sub]:
                    v = entry[sub] if isinstance(entry[sub], str) else str(entry[sub])
                    print(f"  {'':>30s}   {sub}: {v}")
        elif isinstance(entry, dict):
            print(f"  {k}:  " + "  ".join(f"{kk}={_fmt(vv)}" for kk, vv in entry.items()
                                          if not isinstance(vv, (dict, list))
                                          or len(str(vv))<80))

print_section("1. 역학 파라미터", INV["epi"])
print_section("2. 연령별 벡터 (15군)", INV["age_vectors"])
print_section("3. 접촉·공간", INV["contact_space"])
print_section("4. 정책 파라미터", INV["policy"])
print_section("5. 계절성", INV["seasonality"])
print_section("6. 추정 파라미터", INV["estimated"])
print_section("7. 초기조건", INV["initial_conditions"])
print_section("8. 수치·솔버", INV["numeric_solver"])

print("\n" + "═"*90)
print(" 문서-코드 불일치·주의사항 (Δ)")
print("═"*90)
for d in INV["discrepancies"]:
    sev = d["severity"]
    marker = "★" if "high" in sev else " "
    print(f"{marker} {d['item']}")
    print(f"    문서: {d['doc']}")
    print(f"    코드: {d['code_used']}")
    print(f"    심각도: {sev}")
    if "resolution" in d:
        print(f"    해결: {d['resolution']}")

print("\n" + "═"*90)
print(" Legacy / 미사용 (문서엔 있으나 v4 코드 미사용)")
print("═"*90)
for l in INV["legacy_unused"]:
    print(f"  · {l['item']}  ({l['file']})")
    print(f"    사유: {l['reason']}")

# JSON 저장
OUT.write_text(json.dumps(INV, indent=2, default=str, ensure_ascii=False))
print(f"\n[json] {OUT}")
