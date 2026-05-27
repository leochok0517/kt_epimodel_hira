# HIRA vs ILI — kt_epimodel_hira ↔ kt_epimodel 차이점 요약

`kt_epimodel_hira` 는 `kt_epimodel` (ILI fit) 의 자매 프로젝트로,
**모델 코어는 동일**하고 calibration target / 단위 / bound 만 다르다.

상세 설계 근거: [`SKELETON_ANALYSIS.md`](SKELETON_ANALYSIS.md) 섹션 6.

---

## 1. 데이터 소스

| 항목 | ILI (kt_epimodel) | HIRA (kt_epimodel_hira) |
|---|---|---|
| 원본 | 질병관리청 ILI 통계 | 국민건강보험공단 진료에피소드 |
| 파일 | `external/ili/*.csv` (CP949) | `external/hira/*.xlsx` (openpyxl) |
| 시계열 | 시즌별 주별 분율 | 일별 long-form |
| 단위 | per 1,000 outpatients | absolute count |
| 인구 분모 | 외래환자 1,000명 | 없음 (절대 카운트) |
| 시즌 정의 | ISO 36 ~ 35 (1년) | 호출측에서 결정 (default ISO 36 + 52w) |
| 시도별 | ❌ (전국만) | ✅ (17 시도, 수도권 = [11, 28, 41]) |
| 성별 | ❌ | ✅ ("M"/"F", `load_hira_episodes(sex=...)`) |

---

## 2. 연령 그룹

| | ILI | HIRA |
|---|---|---|
| 그룹 수 | 7 | **6** |
| 라벨 | `"0"`, `"1-6"`, `"7-12"`, `"13-18"`, `"19-49"`, `"50-64"`, `"65+"` | `"0-5"`, `"6-11"`, `"12-17"`, `"18-44"`, `"45-64"`, `"65+"` |
| → NIMS 15 매핑 | `ILI_GROUP_TO_NIMS_WEIGHTED` | `HIRA_GROUP_TO_NIMS_WEIGHTED` |

`HIRA_GROUP_TO_NIMS_WEIGHTED` (NIMS 인덱스 합 = 1.0, 단위 테스트로 검증):

```python
{
    "0-5":   {0: 1.0, 1: 0.2},
    "6-11":  {1: 0.8, 2: 0.4},
    "12-17": {2: 0.6, 3: 0.6},
    "18-44": {3: 0.4, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0},
    "45-64": {9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
    "65+":   {13: 1.0, 14: 1.0},
}
```

5세 균등 분포 가정으로 도출. 예: HIRA "0-5" (6 ages) → NIMS 0 (0-4 전부) + NIMS 1 (5세 → 1/5).

---

## 3. 모델 → 관측 변환 (`simulation_to_*`)

| 함수 | ILI (kt_epimodel) | HIRA (kt_epimodel_hira) |
|---|---|---|
| 시그니처 | `simulation_to_ili(daily_inc, population, gamma_report, n_weeks)` | `simulation_to_hira(daily_inc, gamma_report, n_weeks)` |
| 수식 | `ili = γ_r · weekly / population · 1000.0` | `hira = γ_r · weekly` |
| 차이 | per-1000 ×1000 + 인구 분모 ÷ | **둘 다 제거** (카운트 단위) |
| `_by_age` 시그니처 | `(daily_inc_by_age, pop_15, gamma_report, n_weeks, use_weighted)` | `(daily_inc_by_age, gamma_report, n_weeks)` |
| `_by_age` 매핑 | 7 그룹 (`ILI_GROUP_TO_NIMS_WEIGHTED`) | 6 그룹 (`HIRA_GROUP_TO_NIMS_WEIGHTED`) |

---

## 4. Seed 추정 (`estimate_initial_infected_from_*`)

| | ILI | HIRA |
|---|---|---|
| 함수 | `estimate_initial_infected_from_ili` | `estimate_initial_infected_from_hira` |
| 수식 | `I = ili_baseline · pop / 1000 / γ_assumed` | `I = hira_baseline_count / γ_assumed` |
| 차이 | per-1000 ×1/1000 + 인구 분모 × | **둘 다 제거** |
| `gamma_report_assumed` default | 2.0 (노트북: 200.0 corner 탈출용) | **0.2** (도메인 산수 기반) |

### `gamma_report_assumed = 0.2` 근거 (수도권 2019-2020 도메인 산수)

| 항목 | 값 |
|---|---|
| 수도권 인구 (NIMS 15군 합산) | ≈ 2,600만 |
| 시즌 HIRA 에피소드 (수도권, 외래입원) | ≈ 82만 |
| 가정 attack rate | 10–20% |
| → 실제 감염자 | 2,600만 × (0.10–0.20) = 260만–520만 |
| → reporting fraction | 82만 / (260만–520만) = **0.158–0.315** |

중간값 **0.2** 채택. ILI 200 과 자릿수 차이는 ILI 의 `gamma_report_assumed` 가
per-1000 단위환산 + 외래/인구 비율 + 보고율 곱셈 누적이라는 점에서 비롯.

---

## 5. `gamma_report` (fit 대상) bound

| | ILI | HIRA |
|---|---|---|
| 의미 | 외래 분모 흡수 + scaling 항 | reporting fraction (직접) |
| 현재 bound | (0.01, 1.0) | (0.01, 1.0) — ILI 와 동일 유지 (TODO HIRA-B) |
| **권장 1차 bound** | — | **(0.05, 0.5)** |
| 권장 근거 | — | 상한 0.5: 청구 ≥ 50% 감염은 비현실 / 하한 0.05: 그 이하면 다른 시즌 sanity check |

**적용 위치**: `src/kt_epimodel_hira/calibration/param_vector.py:47`.
첫 fit 후 corner 진단 결과 보고 좁힐지 결정.

---

## 6. Loss function (Poisson NLL)

| | ILI | HIRA |
|---|---|---|
| 함수 본체 | `make_loss_function{_by_age}` | 동일 (이름만 같음, HIRA 패키지 안) |
| target dict 키 | `ili_rate`, `ili_rates` | **`hira_count`, `hira_counts`** |
| `pop_total` / `pop_15_flat` 인자 | 사용 | **제거** (인구 분모 없음) |
| `simulation_to_*` 호출 | ILI 변환 | HIRA 변환 |
| `daily_new_infection_by_age()` 사용 | ✅ (vax flux 제외) | ✅ (동일) |
| `poisson_log_likelihood(min_rate=...)` | 0.1 (per-1000 단위) | **1.0** (TODO HIRA-D, 카운트 스케일 잠정) |

---

## 7. Optimizer + Result

| | ILI | HIRA |
|---|---|---|
| `optimize_calibration{_by_age}` 본체 | 23-D minimize | 동일 |
| 추가 인자 | (없음) | `sido_codes`, `setting` |
| `CalibrationResult` dataclass | ILI schema | **동일 schema** (비교 분석 용이성) |
| JSON save/load | ILI schema | **동일** — 같은 `load_result` 로 호환 |
| `_resolve_initial_vec` (warm start) | ✅ | ✅ (동일) |
| 출력 파일 권장 suffix | `*_LBFGS.json` | **`*_LBFGS_HIRA.json`** |

ILI fit 결과를 HIRA optimizer 의 `initial_from_result` 로 넘기는 것도 가능
(23-D 벡터 layout 동일). 단 `gamma_report` 값의 의미가 달라 clip 발생 가능.

---

## 8. 23-D 파라미터 벡터 + bounds

**동일** (둘 다 23-D, layout 동일):

```
[0..3]   β_h, β_w, β_s, β_o          (4)
[4..17]  φ_a (a ∈ 0..14 \ {5})        (14)
[18]     γ_report                      (1)
[19..22] amp, base, σ, peak_day        (4)
```

`ParameterBounds` 도 거의 동일. HIRA 에서 `gamma_report` 만 좁힐 권장 (위 5번).

---

## 9. 1차 fit 후 결정 사항 (TBD)

`SKELETON_ANALYSIS.md` 6.4 의 B/C/D/E. 코드에 `# TODO HIRA-X:` 로 marking.

| ID | 항목 | 위치 | 잠정값 | 결정 트리거 |
|---|---|---|---|---|
| **B** | `ParameterBounds.gamma_report` 좁히기 (0.05, 0.5) | `param_vector.py:47` | ILI 동일 (0.01, 1.0) | 1차 fit 후 박힘 여부 보고 |
| **C** | `gamma_report_assumed` default 조정 | `simple_model.py:estimate_initial_infected_from_hira` | 0.2 (도메인 산수) | 1차 fit 후 corner solution 진단 |
| **D** | `poisson_log_likelihood(min_rate=?)` 재튜닝 | `hira_target.py:poisson_log_likelihood` | 1.0 (잠정) | 1차 fit 후 실측 카운트 분포 |
| **E** | 인구 표준화 옵션 (per 100K) | `hira_target.py:simulation_to_hira{_by_age}` | 없음 (절대 카운트) | 시도별 비교 필요 시 |

---

## 10. 출력 파일 / 비교

| 경로 | ILI | HIRA |
|---|---|---|
| 디렉토리 | `kt_epimodel/outputs/calibration/` | `kt_epimodel_hira/outputs/calibration/` |
| 파일 예시 | `2019-2020_by_age_LBFGS.json` | `2019-2020_by_age_LBFGS_HIRA.json` |
| Schema | 동일 (`CalibrationResult`) | **호환 가능** |
| 비교 분석 | ad-hoc 노트북에서 양쪽 JSON 로드 | 동일 코드로 처리 가능 |

ILI vs HIRA 결과 비교 노트북은 스켈레톤 범위 밖 — 첫 fit 후 별도 작성.

---

## 11. 노트북 (parity 매칭)

`notebooks/` 의 ILI 노트북 7개 → HIRA 버전 일괄 복제 후 함수 호출/import만 교체.
실행은 안 함 — 다음 작업 단계.

| 파일 | 비고 |
|---|---|
| `calibration_01_fit.ipynb` | 연령 합산 single target fit |
| `calibration_02_full_fit.ipynb` | full season fit |
| `calibration_03_by_age.ipynb` | 6-그룹 by_age fit (HIRA 7→6) |
| `calibration_04_1_nelder_mead.ipynb` | NM (LBFGS warm start 패턴 유지) |
| `calibration_04_2_lbfgsb.ipynb` | LBFGS primary fit |
| `calibration_04_3_compare.ipynb` | NM vs LBFGS 비교 |
| `demo_single_season.ipynb` | runner 데모 (ILI 무관, 그대로 적용) |

노트북의 markdown 셀에 ILI 시기 corner-escape 서사 일부 잔존 (vax flux fix 출처
명시용). HIRA fit 진행하며 점진 갱신 권장.

---

## 12. 첫 fit 실행 권장 순서

1. `kt_data` HIRA xlsx 가 `~/Documents/python/NIMS/kt_data/data/external/hira/` 에 있는지 확인.
2. `notebooks/calibration_04_2_lbfgsb.ipynb` 실행 (LBFGS, 60-90분 예상).
3. fit 결과로 B/C/D/E 4 가지 TBD 항목 의사결정.
4. `calibration_04_1_nelder_mead.ipynb` (NM chained from LBFGS) — local refinement.
5. `calibration_04_3_compare.ipynb` — 양 method 비교.
6. ILI vs HIRA 비교 노트북 별도 작성 (양쪽 JSON 로드 + R0, β, φ 비교 plot).
