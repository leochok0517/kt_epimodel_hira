# 11 — Supplement 잔여 그림 재생성 (4시즌)

`prompts/99_conventions.md` 를 함께 적용할 것. `prompts/10` 이 선행이다.

## 상황

`prompts/10` 으로 본문(main_v6)은 v5 그림·v5 숫자를 **전부** 털어냈다.
Supplement 도 대부분 끝났고, 아래 **8개 그림만** 아직 v5 (3시즌, all-influenza) 다.
컴파일 로그에서 `figures/` 로 fallback 되는 것이 이 8개다.

이미 4시즌으로 재생성되어 `figures/paper_fluA4/` 와 `manuscript/figures_fluA4/` 에
들어가 있는 것 (**다시 만들지 말 것**):
`sickleave_term_vs_vacation`, `byage_rate_vs_number_allseasons`,
`age_redistribution_v2`, `effect_ratio_v2` — 이 넷은 저장된 posterior 요약만으로
그려지므로 Cowork 세션에서 이미 처리했다. 생성 스크립트는
`scripts/supfigs_fluA4.py` 로 저장해 두었다 (없으면 이 프롬프트와 함께 전달된 것을 쓸 것).

### 절대 하지 말 것

- **`main_v6.tex` · `supplement_v6.tex` 편집 금지.** 캡션은 이미 새 그림에 맞춰
  고쳐져 있다. 그림 파일만 만들면 된다.
- 그림 **파일명 변경 금지**. `\includegraphics` 가 그 이름을 참조한다.
- v5 재현 경로 수정 금지. 전부 `_fluA4` 접미사 신규 파일로.
- `figures/paper_seasonpop/` 와 `manuscript/figures/` 는 v5 보존용이므로 건드리지 말 것.

## 공통

- 사후분포: `outputs/eda/nuts_fluA4_raw.npz`
- 셋업: `scripts/fluA_setup.py` → `build_fluA_setup(seasons)`
- 시즌: `2016-2017`, `2017-2018`, `2018-2019`, `2019-2020`
- 실행: `FIG_OUTDIR_SUFFIX=_fluA4` → `figures/paper_fluA4/{pdf,png}/`
- `figstyle.COL_SEASON` 에 **2018-2019 가 없다.** 모듈을 고치지 말고 호출부에서
  `#D55E00` 를 추가해 쓸 것 (Okabe–Ito 계열 유지).

---

## A군 — 시뮬레이션이 필요한 것 (4개)

기존 스크립트를 `_fluA4` 복사본으로 만들고 posterior·셋업만 갈아끼운다.

| 그림 | 무엇 | 주의 |
|---|---|---|
| `fit_byage` | 연령대별 관측 vs 모델 | **관측이 A 보정된 값이어야 한다.** v5 그림은 A+B 라 지금 캡션과 어긋난다. 이게 8개 중 가장 중요하다. |
| `epicurve_byage` | baseline 연령별 유행곡선 | 4시즌 |
| `baseline_attack_byage` | baseline 최종 발병률 | 4시즌 |
| `policy_intensity` | 정책 강도별 dose–response | 4시즌. baseline p=0.6 기준 유지 |

## B군 — 새 스윕이 필요한 것 (3개)

| 그림 | 필요한 계산 | 산출 JSON |
|---|---|---|
| `sens_S2_kappa_upper_CI` | κ 스윕 + CI. `sens_kappa_fluA4.json` 은 아동 Δattack 만 있고 averted infection 의 CI 가 없다. `sens2_kappa_upper.py` 를 fluA4 로 재실행 | `outputs/fluA/sens_kappa_upper_fluA4.json` |
| `sens_S3_R0_transition_heatmap` | 성인 초기면역 × 재생산수 격자에서 방학 아동 전이 부호 | `outputs/fluA/sens_r0_immunity_fluA4.json` |
| `theta_kappa_heatmap_v2` | **θ × κ 전체 격자.** 현재 경험적 κ 열(`sens_theta_fluA4.json`)과 κ=1.00 열(`sens_theta_kappa1_fluA4.json`) 둘뿐이다. κ ∈ {0.20, 0.34, 0.40, 0.50, 0.70, 1.00} × θ 21점으로 채울 것 | `outputs/fluA/sens_theta_kappa_grid_fluA4.json` |

θ×κ 격자는 재적합 없이 posterior draw 재사용이므로 비싸지 않다. `n_post=200` 이면 충분.

## C군 — 데이터만 바뀐 것 (1개)

| 그림 | 무엇 |
|---|---|
| `contact_matrices` | 채널 4개 × (학기중/방학) 접촉행렬. **상호성 보정이 데이터 파일에 반영된 뒤로 v5 그림은 보정 전 행렬이다.** `kt_data.data.load_contact.load_contact_matrices` 가 보정된 것을 읽으므로 다시 그리기만 하면 된다. 방학 행렬 경로는 `scripts/check_reciprocity_fluA.py` 가 어떻게 집는지 참고할 것 |

`repro_channel_delta` 도 fallback 목록에 있는데, 이건 `reproduction_numbers_fluA4_<season>.json`
의 `Re.Re_channel` 배열로 이미 계산 가능하다 (prompts/10 T1 이 raw 배열을 저장했다).
정책별 채널 ΔR_e 를 그리면 된다.

---

## 완료 기준

1. `cd manuscript && latexmk -pdf supplement_v6.tex` 로그에 `<./figures/...>` fallback 이
   **0건**일 것. (현재 8건.)
2. 각 그림의 시즌 패널이 4개일 것.
3. `figures/paper_fluA4/{pdf,png}/` → `manuscript/figures_fluA4/{pdf,png}/` 복사까지.

## 보고

`outputs/fluA/REPORT_v6_supfigs.md` 에:

1. 그림별 상태와 산출 경로
2. B군 스윕에서 **v5 결론과 달라진 것**이 있으면 별도 절로. 특히
   `sens_S3` (방학 아동 전이 부호) 는 v5 가 "격자 전체에서 부호 불변" 이라고 적었는데,
   fluA4 에서는 κ=0.20 과 θ≳0.2 에서 부호가 뒤집히는 것이 이미 확인됐다.
   면역×재생산수 격자에서도 뒤집히는 구간이 있는지 그대로 보고할 것.
3. 재생성해 보니 supplement 본문 문장과 어긋나는 것이 있으면 그것도.

숨기지 말 것. 원고는 그 보고를 받아 Cowork 세션에서 고친다.
