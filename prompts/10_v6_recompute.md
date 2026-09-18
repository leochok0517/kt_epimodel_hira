# 10 — v6 원고에 남은 v5 숫자·그림 재계산

`prompts/99_conventions.md` 를 함께 적용할 것.

## 상황

`sickleave_InfluenzaA/manuscript/main_v6.tex` 의 Results·Discussion 초고는
influenza A 4시즌 (`2016-2017`, `2017-2018`, `2018-2019`, `2019-2020`) 결과로
이미 다시 썼다. 총 averted, 연령별 Δattack, θ 스윕, κ 스윕, 채널 배분,
R_peak / R_0 는 전부 `outputs/fluA/*.json` 에서 인용했고 각 자리에
`% outputs/fluA/...` 출처 주석이 달려 있다.

**아직 v5 3시즌 사후분포 숫자로 남아 있는 곳이 6군데**이고, 원고에
`% TODO(v6)` 주석으로 표시해 두었다. 이 프롬프트는 그 6개를 채울
**숫자와 그림을 생산**하는 작업이다.

### 절대 하지 말 것

- **`main_v6.tex` · `supplement_v6.tex` 를 편집하지 말 것.** 원고 문장은
  Cowork 세션에서 쓴다. 여기서는 JSON·PNG·PDF 만 만들고, 마지막에
  "원고에 넣을 값" 을 표로 보고하면 된다.
- `*_v5.tex` 는 어떤 경우에도 건드리지 않는다.
- v5 재현 경로 (`nuts_seasonpop*.py`, `season_pop_setup.py`, `sens_common.py`,
  `policy_seasonpop.py`, `model/`, `simulation/` 등) 수정 금지.
  기존 스크립트를 재사용할 때는 **복사해서 `_fluA4` 접미사 신규 파일**로 만든다.
- `outputs/eda/nuts_seasonpop_*`, `outputs/eda/policy_posterior_seasonpop*`,
  `outputs/eda_v5_uncorrected/*` 덮어쓰기 금지.

### 공통 입력

| 무엇 | 경로 |
|---|---|
| 사후분포 (8000 draws) | `outputs/eda/nuts_fluA4_raw.npz` |
| 정책 forward (N=500) | `outputs/fluA/policy_posterior_fluA4.json` |
| θ 스윕 | `outputs/fluA/sens_theta_fluA4.json` |
| κ 스윕 | `outputs/fluA/sens_kappa_fluA4.json` |
| 재생산수 | `outputs/fluA/reproduction_numbers_fluA4.json` |
| 적합도 | `outputs/fluA/fit_quality_fluA4.json` |
| 셋업 | `scripts/fluA_setup.py` → `build_fluA_setup(seasons)` |

시즌 집합은 항상 위 4개. 2015-16 은 제외 (사유는 원고 Limitations 에 기록됨).

### 공통 산출 규칙

- JSON 은 `outputs/fluA/` 에, 파일명 끝에 `_fluA4`.
- 그림은 `FIG_OUTDIR_SUFFIX=_fluA4` 로 실행 →
  `figures/paper_fluA4/{pdf,png}/` 에 떨어진다. `figstyle.savefig` 를 쓸 것.
- 그림 이름은 v5 와 **동일하게** 유지한다. 원고의 `\includegraphics` 가
  그 이름을 참조하므로 바꾸면 원고도 고쳐야 한다.

---

## T1 — 재생산수 채널·연령 분해 + R_e(t)  (원고 §5.5 전체)

원고 `\subsection{Reproduction-number decomposition by channel and age}` 의
**모든 숫자가 v5 것**이다: 학교 채널 24.9%, 증폭 3.9배, 직장 3.8% (0.13배),
0–19 세 76%, 5–9 세 단독 47%, R_e 피크 1.5 @ day 82, 정책별 피크 1.52→1.49,
피크 시점 82→69.

1. `scripts/reproduction_analysis.py` 를 `scripts/reproduction_analysis_fluA4.py`
   로 복사. 입력 사후분포를 `outputs/eda/nuts_fluA4_raw.npz` 로,
   셋업을 `fluA_setup.build_fluA_setup` 으로 바꾼다. 대표 시즌은 v5 와 같이
   **2019-2020**, 나머지 3시즌도 같이 돌려 Supplement 용으로 남긴다.
2. 산출: `outputs/fluA/reproduction_numbers_fluA4_<season>.json`
   (v5 의 `outputs/eda/reproduction_numbers_2019-2020.json` 과 같은 스키마:
   `static.channel`, `static.age`, `Re_t` 세 블록).
3. 그림: `_fig_repro_bars.py` · `_fig_Re_readable.py` 도 `_fluA4` 복사본으로
   만들어 입력 JSON 경로만 바꿔 실행. 출력 이름은 `repro_bars`,
   `repro_Re_readable_<season>` 유지.

**완료 기준**

- 채널 기여 합 `Σ R_c` 가 해당 시즌 `R_peak` 사후평균
  (2019-20 = 2.193) 과 0.5% 이내로 일치.
- `reproduction_numbers_fluA4.json` 의 `R_peak` 와 T1 이 쓰는 정규화 지점이
  동일한지 (초기 면역 + 계절성 피크 + 학기중 행렬) 명시적으로 확인 후 보고.
  v5 본문은 이 값을 "2.17" 로 적었는데 v6 은 `\Rpeaks` 로 이름이 바뀌었다.
- 증폭배수 = (채널 재생산 점유율) / (π_c). π 는 4시즌 joint 값
  `[home 0.423, work 0.315, school 0.073, other 0.189]`.

**보고 형식** — 아래 값들을 표로:
채널별 R_c 와 점유율(95% CI) · 증폭배수, 연령대별 점유율(0–19 합, 5–9 단독),
R_e(t) 피크 값과 피크 day, 시나리오별 (baseline/sick/school) 피크 값·day,
other 채널 점유율과 그중 0–19 세 비중(원고 §5.7 에서 인용).

---

## T2 — π_work × κ 민감도 맵  (원고 §5.6 두 번째 결과 + Figure sens)

원고의 "κ ≈ 0.46–0.58 임계" 와 `sens_S1_piwork_kappa_heatmap` 그림이 v5 값이다.

1. `scripts/sens1_piwork_kappa.py` → `sens1_piwork_kappa_fluA4.py` 복사.
   `sens_common.SEASONS` 대신 4시즌, 셋업은 `build_fluA_setup`.
   **`scripts/policy_fluA.py` 가 쓰는 몽키패치 패턴을 그대로 따를 것**
   (`sens_common.SEASONS` 와 `season_pop_setup.build_seasonwise_setup` 을
   import 전에 갈아끼운다).
2. 그리드는 v5 와 동일하게 유지 — 바뀐 것이 타깃·시즌이지 그리드가 아니어야
   비교가 된다. 체크포인트(jsonl) 재개 기능 유지.
3. 산출 `outputs/fluA/sens_piwork_kappa_fluA4.json` +
   그림 `sens_S1_piwork_kappa_heatmap` (`FIG_OUTDIR_SUFFIX=_fluA4`).

**완료 기준**

- 시즌별 부호 전환 κ 임계값(π_work 를 사후평균 0.315 에 고정했을 때) 을 보고.
- 경험적 κ ≈ 0.34–0.40 이 여전히 net-beneficial 영역에 있는지 명시.
  **있으면 있다고, 없으면 없다고 그대로 보고할 것.** v5 결론을 지키려 하지 말 것.
- Ajelli 저 π_work ≈ 0.03 지점, Ferguson κ = 1.0 지점의 부호도 함께.

---

## T3 — θ × κ 코너  (원고 §5.7 마지막 문단)

`sens_theta_fluA4.json` 은 **경험적 κ 에서만** 돌았다. 원고는 "κ = 1.00 에서는
θ 전 구간에서 음수" 라고 적고 있는데 이건 v5 결과다.

`scripts/sens_theta_fluA4*.py` (09 에서 만든 것) 를 κ = 1.00 에서 한 번 더 돌린다.
θ 그리드는 동일하게 `[-0.25, 0.75]` 21점, `n_post` 는 200 이면 충분.

산출: `outputs/fluA/sens_theta_kappa1_fluA4.json`.

**완료 기준** — 4시즌 × θ 21점에서 sick leave 총 averted 의 부호와,
음수가 아닌 점이 하나라도 있으면 그 (시즌, θ) 를 전부 나열.

---

## T4 — 사망률 오버레이  (원고 Discussion §5.4 마지막 문단)

v5 는 "school absence 의 averted deaths 중 약 2/3 가 65+" 라고 적었다.
이 계산을 한 스크립트가 저장소에 없다. **새로 만든다** —
`scripts/mortality_overlay_fluA4.py`.

- 입력: `outputs/fluA/policy_posterior_fluA4.json` 의
  `{sick,school}_num_by_age_{term,vac}` (연령별 averted **수**).
- 가중: 연령대별 IFR 을 자릿수 수준으로. v5 본문이 쓴 범위는
  아동 ~1e-5 → 노인 ~1e-2. **정확한 v5 벡터가 저장소에 없으므로
  새로 정하고, 그 값을 JSON 에 그대로 기록할 것.** 근거는 `McDonald2023`.
- 산출 `outputs/fluA/mortality_overlay_fluA4.json`:
  정책 × 시즌 × 윈도우별 averted deaths 와 **65+ 가 차지하는 비중**.

**완료 기준** — 이건 원고가 "illustrative only" 라고 명시한 계산이다.
정밀도를 주장하지 말고, IFR 벡터에 대한 민감도(예: 10배 범위)를 같이 보고할 것.
sick leave 의 65+ averted deaths 부호가 v5(≈0 또는 약간 음)에서
v6(양수)으로 바뀌는지가 핵심 확인 사항이다.

---

## T5 — 본문 그림 4개 재생성  (4시즌)

아래 넷은 3시즌 데이터로 그려져 있어 4시즌 결과와 캡션이 어긋난다.

| 그림 | 스크립트 | 고칠 것 |
|---|---|---|
| `fit_total` | 04/05 단계에서 쓴 것 | 4패널 (2×2) |
| `school_vs_sick` | `policy_compare_school_vs_sickleave.py` | 4시즌 |
| `byage_2metric_main` | `fig_byage_2metric_main.py` | 입력 JSON → `policy_posterior_fluA4.json`, "3-season mean" → 4 |
| `holiday` | `regen_holiday_fig.py` / `viz_holiday_reversal.py` | 4시즌 |
| `nuts_posterior` | 03 단계 산출 | 4시즌 R_peak 패널, 축 라벨 `R_peak,s` |

전부 `_fluA4` 복사본으로 만들고 `FIG_OUTDIR_SUFFIX=_fluA4` 로 실행.
**그림 파일명은 유지**.

`byage_2metric_main` 주의: v6 원고는 이 그림을 "per-capita 로는 등교중단이
아동에게, 병가가 성인에게 이득" 으로 설명한다. v5 의 "병가가 아동을 올린다" 가
**학기중 창에서는 더 이상 나타나지 않기 때문**이다. 그림이 실제로 그렇게
나오는지 확인하고, 다르게 나오면 그대로 보고할 것.

---

## T6 — 원고 그림 폴더 동기화

T1·T2·T5 가 끝나면:

```
figures/paper_fluA4/{pdf,png}/*  →  ~/Documents/cowork/sickleave_InfluenzaA/manuscript/figures_fluA4/
```

원고의 `\graphicspath{{figures/}}` 를 `{{figures_fluA4/}}` 로 바꾸는 것은
**Cowork 세션에서 한다.** 여기서는 폴더만 채우고, 채운 파일 목록을 보고할 것.
v5 그림은 `figures/` 에 그대로 두어 되돌릴 수 있게 한다.

`flow_diagram` 은 데이터와 무관한 모식도이므로 v5 파일을 그대로 복사한다.

---

## 실행 순서

T1 → T2 는 각각 길다. 병렬로 돌리지 말고 순차로, 둘 다 detach:

```
nohup caffeinate -i .venv/bin/python scripts/reproduction_analysis_fluA4.py \
  > logs/t1_repro_fluA4.log 2>&1 & disown
```

T3·T4·T5 는 사후분포 재사용이라 짧다 (수 분~수십 분).
T6 는 마지막.

## 보고

작업이 끝나면 `outputs/fluA/REPORT_v6_recompute.md` 에:

1. T1–T6 별 상태 (완료 / 부분 / 실패) 와 실제 산출 경로
2. **원고에 넣을 값 표** — 항목, v5 값, v6 값, 출처 JSON 키
3. v5 결론이 뒤집힌 항목이 있으면 별도 절로 분리해 명시
4. 재계산해 보니 v6 초고에 이미 쓴 문장과 어긋나는 것이 있으면 그것도 별도로

3·4 는 숨기지 말 것. 원고는 그 보고를 받아 Cowork 세션에서 고친다.
