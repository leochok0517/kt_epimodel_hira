# `kt_epimodel` 스켈레톤 분석 — HIRA 자매 프로젝트 설계 문서

> **본 사본**: 이 파일은 `../kt_epimodel/docs/SKELETON_ANALYSIS.md` 의 동기화
> 사본. 본 프로젝트 `kt_epimodel_hira` 는 이 문서의 섹션 6 권장안을 그대로
> 구현한 결과물 (Phase 1-7 셋업 commit 참조).
>
> 수정 시 두 프로젝트 동시 갱신 권장 (또는 원본만 갱신 후 사본 재복사).
> 짧은 ILI ↔ HIRA 차이 요약은 [`HIRA_VS_ILI_DIFF.md`](HIRA_VS_ILI_DIFF.md) 참조.

목적: `kt_epimodel`을 그대로 두고 HIRA(국민건강보험 인플루엔자 진료 에피소드 카운트)
데이터로 fit하는 자매 프로젝트 `kt_epimodel_hira`를 만들 때 무엇을 복제·교체·
파라미터화할지 판단하기 위한 코드 스켈레톤 분석. **이 문서 작성 시 코드는
수정하지 않았다.**

분석 시점 커밋: `a56d4b8` (kt_epimodel main, "Add chained-start option …").

---

## 1. 패키지 구조 맵

`src/kt_epimodel/` 트리. 각 모듈에 대해 (a) 주요 public 심볼, (b) 한 줄 책임.

```
src/kt_epimodel/
├── __init__.py                          (빈 패키지 init)
│
├── model/                               [데이터 비의존 — 도메인 ODE 코어]
│   ├── __init__.py                      mobility_tensor 빌더 re-export
│   ├── compartments.py                  IDX_S/V/E/I/R, N_AGE=15, N_COMPARTMENTS=5,
│   │                                    flatten_state, unflatten_state, initial_state
│   │                                    → SEIRV 5-compartment × 15 age × n_admdong 텐서 정의
│   ├── parameters.py                    DiseaseParameters, CalibrationParameters,
│   │                                    VaccinationParameters, EmploymentParameters,
│   │                                    ModelParameters (composite + .with_*())
│   │                                    → β/φ/γ_report, σ/γ/VE, ρ, 시즌성 함수 전부
│   ├── foi.py                           compute_foi, channel-wise λ
│   │                                    → 4채널 (home/work/school/other) FOI 계산
│   ├── dynamics.py                      compute_derivatives, make_ode_rhs,
│   │                                    make_time_varying_ode_rhs
│   │                                    → ODE 우변 (S/V→E/I/R 흐름 + 백신 + κ spillover)
│   └── mobility_tensor.py               build_M_home, build_M_school, build_M_work,
│                                        build_M_other, build_M_from_kt_array
│                                        → KT mobility (1154×1154×7×24) → 채널별 M 텐서
│
├── simulation/                          [데이터 비의존 — solver 래퍼]
│   ├── __init__.py
│   ├── solver.py                        SimulationResult (dataclass), run_simulation,
│   │                                    run_simulation_time_varying,
│   │                                    SimulationResult.daily_new_infection(),
│   │                                    SimulationResult.daily_new_infection_by_age()
│   │                                    → scipy solve_ivp 래퍼 + incidence 헬퍼
│   └── runner.py                        Stage 2 single-season demo runner (notebook용)
│                                        → kt_data 로더 + simulate 묶어 호출하는 high-level
│
├── calibration/                         [★ ILI 의존 거점 — 분기 대상]
│   ├── __init__.py                      ili_target / loss / optimizer / simple_model
│   │                                    public symbol re-export
│   ├── ili_target.py                    N_WEEKS, load_ili_target, load_ili_target_by_age,
│   │                                    simulation_to_ili, simulation_to_ili_by_age,
│   │                                    poisson_log_likelihood, season_start_date
│   │                                    → ILI raw → fit target dict / sim → ILI 변환 /
│   │                                      Poisson NLL
│   ├── simple_model.py                  build_aggregated_inputs,
│   │                                    estimate_initial_infected_from_ili,
│   │                                    simulate_aggregated,
│   │                                    _build_initial_state_with_age_seed
│   │                                    → 행정동 합산 (1-admdong) 단순 모델 + ILI seed 추정
│   ├── param_vector.py                  ParameterBounds, params_to_vector,
│   │                                    vector_to_params, get_bounds_vector,
│   │                                    initial_guess, get_param_names
│   │                                    → 23-D fit 벡터 layout 정의 (β 4 + φ 14 + γ_r + 시즌성 4)
│   ├── loss.py                          make_loss_function (single ILI 시계열),
│   │                                    make_loss_function_by_age (7 그룹 동시)
│   │                                    → scipy.minimize 호환 NLL closure
│   └── optimizer.py                     CalibrationResult, optimize_calibration,
│                                        optimize_calibration_by_age, save_result,
│                                        load_result, _resolve_initial_vec
│                                        → fit 진입점 + 결과 dataclass + JSON 직렬화 +
│                                          warm-start helper
│
├── scenarios/__init__.py                (현재 빈 모듈 — 정책 시뮬 스텁)
└── viz/__init__.py                      (현재 빈 모듈 — 시각화 스텁)
```

### `kt_data` import 인벤토리

`grep -rE "from kt_data|import kt_data" src/` 결과 (18개 hit). 모듈 단위로 정리:

| `kt_epimodel` 모듈 | `kt_data` 심볼 | 용도 |
|---|---|---|
| `model/foi.py:248` (__main__) | `kt_data` 일반 import | demo |
| `model/mobility_tensor.py:155, 171, 222` | `load_mobility` | mobility 텐서 빌드 |
| `model/parameters.py:294` | `build_rho_matrix, load_population_15groups` | demo (`EmploymentParameters.from_kt_data`) |
| `model/dynamics.py:118, 159` | `load_calendar.classify_date`, 일반 | time-varying mobility + demo |
| `model/compartments.py:220` (__main__) | `load_population` | demo |
| `simulation/solver.py:177-178` | `load_contact_matrices`, `load_population` | demo |
| `simulation/runner.py:15-16` | `load_contact_matrices`, `load_population` | runner |
| `calibration/simple_model.py:15-20` | `load_contact_matrices`, `load_employment_rate`, `load_population_15groups`, `get_population_matrix` | `build_aggregated_inputs` |
| `calibration/simple_model.py:112-115` | **`ILI_AGE_GROUPS`, `ILI_GROUP_TO_NIMS_WEIGHTED`, `load_ili_by_age`** | `estimate_initial_infected_from_ili` |
| `calibration/ili_target.py:14` | **`load_ili_seasons`** | `load_ili_target` |
| `calibration/ili_target.py:126` | **`ILI_AGE_GROUPS, load_ili_by_age`** | `load_ili_target_by_age` |
| `calibration/ili_target.py:189-192` | **`ILI_GROUP_TO_NIMS, ILI_GROUP_TO_NIMS_WEIGHTED`** | `simulation_to_ili_by_age` |

**굵게 표시한 4건이 ILI-specific 심볼 (HIRA에선 대응 심볼 필요)**.
나머지 `kt_data` import는 ILI와 무관 (인구·접촉·이동·경제활동·달력).

---

## 2. ILI 데이터 의존 지점 ★

키워드별 hit 위치 + 컨텍스트 + HIRA 분기 시 분류
(**변경 필요 / 불필요 / 재검토**).

### 2.1 직접 데이터 로더 호출 — `load_ili*`

#### `src/kt_epimodel/calibration/ili_target.py:14`
```python
from kt_data.data.load_ili import load_ili_seasons
```
- **변경 필요**: HIRA 등가 (예: `load_hira_seasons`) 로 교체.

#### `src/kt_epimodel/calibration/ili_target.py:126`
```python
def load_ili_target_by_age(season: str, ...) -> dict:
    ...
    from kt_data.data.load_ili import ILI_AGE_GROUPS, load_ili_by_age
    result: dict = {
        "season": season,
        "age_groups": list(ILI_AGE_GROUPS),
        ...
    }
    for ag in ILI_AGE_GROUPS:
        df = load_ili_by_age(ag)
```
- **변경 필요**: `HIRA_AGE_GROUPS`, `load_hira_by_age` 로 교체. HIRA가 다른 연령
  segmentation을 쓰면 함수 본체도 약간 조정 가능.

#### `src/kt_epimodel/calibration/ili_target.py:189-192`
```python
from kt_data.data.load_ili import (
    ILI_GROUP_TO_NIMS,
    ILI_GROUP_TO_NIMS_WEIGHTED,
)
```
- **변경 필요**: `HIRA_GROUP_TO_NIMS_WEIGHTED` (HIRA→NIMS 15군 인구비례 분배 매핑) 필요.

#### `src/kt_epimodel/calibration/simple_model.py:112-115`
```python
from kt_data.data.load_ili import (
    ILI_AGE_GROUPS,
    ILI_GROUP_TO_NIMS_WEIGHTED,
    load_ili_by_age,
)
```
- **변경 필요**: `estimate_initial_infected_from_ili()` 가 ILI baseline에서 초기 I 추정.
  HIRA용 등가 함수 신규 작성 (수식 자체가 분모 단위 때문에 달라짐 — 아래 2.2 참조).

### 2.2 단위/분모 변환 — per-1000 / 외래환자 흡수항

#### `src/kt_epimodel/calibration/ili_target.py:106` (`simulation_to_ili`)
```python
ili = gamma_report * weekly / population * 1000.0
```
- ILI = 외래환자 1,000명당 의사환자 수. `* 1000.0`은 비율 단위 변환.
- HIRA는 episode 카운트 (분자만, 인구 분모 없음, 단위 unitless count).
- **변경 필요**: HIRA 버전 변환식:
  `predicted_count = gamma_report * weekly_simulated_incidence`
  (per-1000 곱셈 없음). 단, 인구 표준화 비교를 위해 인구 분모를 옵션으로 둘 수 있음.

#### `src/kt_epimodel/calibration/ili_target.py:224` (`simulation_to_ili_by_age`, weighted)
```python
ili = gamma_report * group_inc / group_pop * 1000.0
```
- 같은 패턴, 연령 그룹별 분모.
- **변경 필요**: 인구 분모 + per-1000 제거.

#### `src/kt_epimodel/calibration/ili_target.py:233` (`simulation_to_ili_by_age`, legacy)
```python
ili = gamma_report * group_inc / group_pop * 1000.0
```
- 동일.
- **변경 필요**.

#### `src/kt_epimodel/calibration/simple_model.py:144`
```python
I_group = (
    ili_baseline[ag] * total_weighted_pop / 1000.0 / gamma_report_assumed
)
```
- ILI rate × pop / 1000 / γ_assumed → 초기 I 수.
- HIRA: `I_group = hira_baseline_count[ag] / gamma_report_assumed` (단순 카운트).
- **변경 필요**.

### 2.3 `gamma_report` — 의미가 분모에 따라 달라지는 파라미터

#### `src/kt_epimodel/model/parameters.py:131-154`
```python
class CalibrationParameters:
    """4 채널 β + 연령별 susceptibility φ + 보고율 γ_report.
    ...
    """
    ...
    gamma_report: float = 0.5

    def __post_init__(self) -> None:
        ...
        if not (0 < self.gamma_report <= 1):
            raise ValueError(f"gamma_report must be in (0, 1], got {self.gamma_report}")
```
- ILI 맥락: `γ_report`는 "ILI rate scaling" — 외래환자 분모를 흡수해서 (0, 1] 범위
  의미가 불분명 (`CLAUDE.md` 미해결 사항 #1).
- HIRA 맥락: `γ_report`는 진짜 reporting fraction 가능성 (감염자 중 진료받은 비율).
- **재검토**: dataclass는 그대로 둘 수 있지만 **bound 의미가 달라짐**. 코드 변경
  불필요하지만 HIRA 쪽 `ParameterBounds.gamma_report`를 좁힐지 (예: [0.05, 0.5]) 결정 필요.

#### `src/kt_epimodel/calibration/param_vector.py:47`
```python
gamma_report: tuple[float, float] = (0.01, 1.0)
```
- **재검토**: HIRA에서 bound 좁혀야 할 가능성. Phase A rollback 이후 (0.01, 1.0).

#### `src/kt_epimodel/calibration/optimizer.py:224, 256-258`
```python
gamma_report_assumed: float = 2.0,
...
seed_by_age_arr = estimate_initial_infected_from_ili(
    season, inputs["pop_15"].flatten(),
    gamma_report_assumed=gamma_report_assumed,
)
```
- `gamma_report_assumed`는 seed 추정에 들어가는 가정값. HIRA에선 의미 다르나
  default=200.0 (현재 노트북) 같은 값은 유지하지 못함.
- **변경 필요**: default 값 + seed 추정 경로 바뀜.

### 2.4 7 그룹 매핑 — ILI의 연령 segmentation

`ILI_AGE_GROUPS = ["0", "1-6", "7-12", "13-18", "19-49", "50-64", "65+"]` (7개).

매핑은 모두 `kt_data/src/kt_data/data/load_ili.py` 안에 있고, `kt_epimodel` 코드는
이름만 import해서 사용. → kt_epimodel 자체엔 7그룹 매핑 코드는 없음.

`load_ili_target_by_age`, `simulation_to_ili_by_age`, `estimate_initial_infected_from_ili`
는 본질적으로 "7개 그룹 → NIMS 15군 weighted 분배" 로직만 들어 있으므로 **HIRA의
그룹 수가 6/7/다른 수더라도 같은 패턴으로 새로 쓰면 됨** (그룹 수에 hard-coded
가정 없음, dict 반복).

### 2.5 Likelihood / weights — 단위에 둔감

#### `src/kt_epimodel/calibration/ili_target.py:240-282` (`poisson_log_likelihood`)
```python
def poisson_log_likelihood(
    observed, predicted, is_valid=None, weights=None, min_rate: float = 0.1,
) -> float:
    """Weighted Poisson NLL = Σ w_i [y_pred − y_obs · log(y_pred)]."""
    ...
    pred = np.maximum(predicted[mask], min_rate)
```
- 수학적으로는 단위 무관 (Poisson NLL은 rate든 count든 동일 공식).
- 단, `min_rate=0.1`은 **ILI 단위(per-1000)에 맞춰 튜닝**된 값 (docstring 명시).
- **재검토**: HIRA 카운트 스케일(예: 주당 수백 건)에선 `min_rate` 재조정 필요
  (예: 1.0~10.0). 함수 자체는 재사용 가능, 호출 시 인자만 다르게.

#### `loss.py`의 `weights` / `first_peak_only`
ILI specific 아님 — 시즌 첫 봉만 fit하는 일반 기능.
- **불필요 변경**.

### 2.6 명시적 문자열 "ILI" 주석/docstring

- `src/kt_epimodel/calibration/__init__.py:1` — `"""kt_epimodel calibration — ILI fitting, optimizer scaffolding."""`
- `src/kt_epimodel/calibration/ili_target.py:1-5`, `:50, :90, :106, :119`, etc. — docstring
- `src/kt_epimodel/calibration/loss.py:1, :4, :38, :49, :51` — docstring
- `src/kt_epimodel/calibration/simple_model.py:88-104` — docstring
- `src/kt_epimodel/calibration/param_vector.py:146` — `# ILI peak ~ week 16-18 day 112-126.`
- `src/kt_epimodel/calibration/optimizer.py:239` — `시즌 시작 ILI baseline 으로 ...`

**불필요 변경 (코드 동작 영향 없음)**, 단 HIRA 프로젝트 작성 시 문서 일치성 위해 재작성 권장.

### 2.7 "7 그룹" / "Poisson" / "NLL" / "weighted" 키워드

| 키워드 | 발견 위치 | 분류 |
|---|---|---|
| `7 그룹` | `loss.py:133`, `optimizer.py:238, 287`, `simple_model.py` (주석) | docstring, **불필요** |
| `Poisson` | `loss.py:1`, `ili_target.py` (poisson_log_likelihood), `optimizer.py:238` | docstring, **불필요** |
| `NLL` | 다수 (verbose 로그, docstring) | **불필요** |
| `weighted` | `ili_target.py:217` (`use_weighted=True`), `simple_model.py:139` | logic — **변경 필요** (HIRA 매핑 dict 이름) |

---

## 3. 데이터 비의존 코어

다음 모듈은 ILI/HIRA fit target과 **완전 무관** — 그대로 복제하면 된다.

| 모듈 | 확인 결과 |
|---|---|
| `src/kt_epimodel/model/compartments.py` | ✅ ILI 언급 없음. kt_data import는 `__main__` demo에서만 (population) |
| `src/kt_epimodel/model/foi.py` | ✅ FOI 수식만. demo `__main__`에서 contact + mobility 로드 |
| `src/kt_epimodel/model/dynamics.py` | ✅ ODE 우변. demo에서 calendar 사용 |
| `src/kt_epimodel/model/mobility_tensor.py` | ✅ KT mobility 텐서 빌더 |
| `src/kt_epimodel/model/parameters.py` | △ **CalibrationParameters에 `gamma_report`** — dataclass 정의는 generic, **HIRA에선 의미 재정의 + bound 좁히기 권장** (코드는 그대로) |
| `src/kt_epimodel/simulation/solver.py` | ✅ scipy solve_ivp 래퍼 + `daily_new_infection_by_age()` 헬퍼 (vax flux 분리) |
| `src/kt_epimodel/simulation/runner.py` | ✅ high-level demo runner |
| `src/kt_epimodel/scenarios/__init__.py` | ✅ 빈 모듈 |
| `src/kt_epimodel/viz/__init__.py` | ✅ 빈 모듈 |

### Subtle 의존성 — `model/parameters.py`

- 라인 8 (CalibrationParameters docstring header): `"- CalibrationParameters: β_h, β_w, β_s, β_o, φ, γ_report"`
- 라인 131 (class docstring): `"4 채널 β + 연령별 susceptibility φ + 보고율 γ_report."`
- 라인 153-154 (validation): `if not (0 < self.gamma_report <= 1)` — HIRA에선 더 좁힐지 결정 필요.

→ **HIRA 프로젝트는 코드 동일 유지 가능, bound는 `ParameterBounds` 쪽에서만 조정**.

---

## 4. Calibration 모듈 상세 분해

각 파일의 public 함수 단위 ILI 의존도 (○ 강한 의존 / △ 호출 의존 / ×
무관) + HIRA 분기 권장.

### `calibration/ili_target.py`

| 함수 | 의존도 | HIRA 권장 |
|---|---|---|
| `load_ili_target(season, ...)` | ● | **새로 작성** (`load_hira_target`). 단위 변환 + interpolate_nan + first_peak weights 패턴은 동일하게 가져가도 됨 |
| `season_start_date(season)` | × | **그대로 재사용** (ISO 36주 → yyyymmdd; 도메인 독립) |
| `simulation_to_ili(daily_incidence, population, gamma_report, n_weeks)` | ● | **새로 작성** (`simulation_to_hira`). 핵심 차이: per-1000/인구 분모 제거 |
| `load_ili_target_by_age(...)` | ● | **새로 작성** (그룹 수 / 매핑 변경) |
| `simulation_to_ili_by_age(...)` | ● | **새로 작성** (HIRA 매핑) |
| `poisson_log_likelihood(observed, predicted, is_valid, weights, min_rate)` | △ | **재사용** (단위 무관). 단 `min_rate` default 0.1은 ILI per-1000 튜닝 → HIRA용 호출 시 인자 명시 |
| `N_WEEKS = 52` | × | **재사용** (시즌 길이) |

### `calibration/simple_model.py`

| 함수 | 의존도 | HIRA 권장 |
|---|---|---|
| `build_aggregated_inputs()` | × | **그대로 재사용** (인구·접촉·이동·고용만 사용) |
| `estimate_initial_infected_from_ili(...)` | ● | **새로 작성** (`estimate_initial_infected_from_hira`). 식: `I = hira_baseline_count / γ_assumed` (per-1000 제거). 7→15군 가중 분배 패턴은 동일 |
| `_build_initial_state_with_age_seed(...)` | × | **그대로 재사용** (seed 벡터 받아 SEIRV 초기 상태 구성) |
| `simulate_aggregated(...)` | × | **그대로 재사용** |

### `calibration/param_vector.py`

| 항목 | 의존도 | HIRA 권장 |
|---|---|---|
| 23-D 벡터 layout (β 4 + φ 14 + γ_r + 시즌성 4) | × | **그대로 재사용** — fit 대상 파라미터는 모델 구조에 종속, target 단위와 무관 |
| `ParameterBounds.gamma_report = (0.01, 1.0)` | △ | **재검토** — HIRA에서 의미가 reporting fraction이면 (0.05, 0.5) 정도로 좁힐지 결정 |
| `initial_guess()` (peak_day=110 hardcode) | × | **재사용** — 110 day는 ILI peak 가정인데 한국 인플루엔자 transmission peak는 ILI peak보다 약간 앞 → HIRA에서도 큰 차이 없을 가능성 |
| 그 외 (params_to_vector / vector_to_params / get_bounds_vector) | × | **재사용** |

### `calibration/loss.py`

| 함수 | 의존도 | HIRA 권장 |
|---|---|---|
| `make_loss_function(target, inputs, base_params, ...)` | ● | **새로 작성** (`make_loss_function_hira`). 본체 구조 동일, `simulation_to_ili` → `simulation_to_hira`, `target["ili_rate"]` → `target["hira_count"]` 같은 키 이름 |
| `make_loss_function_by_age(target_by_age, inputs, base_params, ...)` | ● | **새로 작성** (`make_loss_function_by_age_hira`). 본체 구조 동일, `simulation_to_ili_by_age` → HIRA 등가 호출 |

두 함수의 **로직 자체는 거의 동일**: target dict 받아서 vec → sim → 변환 → NLL.
공통화 후 target-converter 함수를 인자로 받게 만들 수도 있지만, 그러면 ILI 쪽 코드도
변경 필요. **분리 작성이 안전**.

### `calibration/optimizer.py`

| 항목 | 의존도 | HIRA 권장 |
|---|---|---|
| `CalibrationResult` dataclass | △ | **그대로 재사용** — 필드 (β, φ, γ_report, NLL, 시즌성, …) 전부 HIRA에도 그대로 의미 있음. `gamma_report_assumed`는 의미 다르지만 필드명은 OK |
| `_resolve_initial_vec(initial_vec, initial_from_result)` | × | **그대로 재사용** |
| `optimize_calibration(season, ...)` | △ | **새로 작성** (`optimize_calibration_hira`). 본체 동일, `load_ili_target` → `load_hira_target`, `make_loss_function` → `make_loss_function_hira` |
| `optimize_calibration_by_age(...)` | △ | **새로 작성** (`optimize_calibration_by_age_hira`). 본체 동일, ILI 로더 + seed 함수 모두 HIRA 버전으로 |
| `save_result(result, path)` | △ | **그대로 재사용** — JSON schema 변화 없음 |
| `load_result(path)` | △ | **그대로 재사용** |
| `VALID_METHODS = ("Nelder-Mead", "L-BFGS-B")` | × | **재사용** |

`optimizer.py`의 두 fit 진입점은 "어떤 target loader + 어떤 loss" 두 결정만
ILI/HIRA 따라 달라짐. 나머지 90%는 동일 → **파라미터화 가능성 있음** (`target_loader`,
`loss_factory` callable을 인자로). 단 ILI 쪽 코드를 건드려야 함.

---

## 5. 테스트 / 노트북 / 산출물

### 5.1 테스트

| 테스트 파일 | ILI 의존 | 설명 |
|---|---|---|
| `tests/test_compartments.py` | ✕ | SEIRV 텐서 |
| `tests/test_dynamics.py` | ✕ | ODE 우변 |
| `tests/test_foi.py` | ✕ | FOI 채널 |
| `tests/test_mobility_tensor.py` | ✕ | KT mobility 빌더 |
| `tests/test_parameters.py` | ✕ | dataclass validators (단 `gamma_report` validation 포함) |
| `tests/test_runner.py` | ✕ | runner (현재 mobility 경로 mismatch로 1건 fail) |
| `tests/test_setup.py` | ✕ | setup smoke |
| `tests/test_solver.py` | ✕ | solver + incidence helpers |
| `tests/test_param_vector.py` | ✕ | 23-D 벡터 layout (★ ILI specific 아님 — 단위 추상화됨) |
| `tests/test_ili_target.py` | ● | `load_ili_target` 직접 호출 |
| `tests/test_loss.py` | ● | `load_ili_target` 호출, `make_loss_function*` 검증 |
| `tests/test_optimizer.py` | ● | `optimize_calibration{_by_age}` 통합 테스트 |
| `tests/test_simple_model.py` | ● | `estimate_initial_infected_from_ili` 호출 |

**HIRA 프로젝트 테스트 전략**:
- ✕ 표시 9개: **그대로 복제**
- ● 표시 4개: **새로 작성** (HIRA loader / loss / optimizer / seed 함수에 맞춰)

### 5.2 노트북

`notebooks/` 7개 파일 모두 `kt_data.load_ili*` 또는 `kt_epimodel.calibration.ili_target.*`
import. 각 노트북이 의존하는 ILI 경로:

| 노트북 | ILI 의존 |
|---|---|
| `calibration_01_fit.ipynb` | `load_ili_target`, `simulation_to_ili` |
| `calibration_02_full_fit.ipynb` | (확인 안 됨, 같은 패턴 가정) |
| `calibration_03_by_age.ipynb` | `load_ili_target_by_age`, `simulation_to_ili_by_age` |
| `calibration_04_1_nelder_mead.ipynb` | 동일 + LBFGS warm start |
| `calibration_04_2_lbfgsb.ipynb` | 동일 |
| `calibration_04_3_compare.ipynb` | `ILI_AGE_GROUPS` 직접 import + 모든 ILI 함수 |
| `demo_single_season.ipynb` | (확인 안 됨, runner 데모 중심일 가능성 — ILI 무관 추정) |

**HIRA 프로젝트 노트북**: 노트북은 **새로 작성** (loader 이름 + 시각화 단위 모두
바뀜). 다만 cell 골격(데이터 시각화 → fit → fit 진단 → compare)은 그대로
사용 가능.

### 5.3 출력 파일 명명 규약

`outputs/calibration/` 현재 파일:
```
2019-2020_by_age_LBFGS.json
2019-2020_by_age_LBFGS_fit.png
2019-2020_by_age_LBFGS_phi_beta.png
```

- 시즌 + `by_age` (= 7-group ILI) + method 약자.
- HIRA 결과를 같은 디렉토리에 저장하면 **충돌 위험** (예: `2019-2020_by_age_LBFGS.json`이
  ILI 결과인지 HIRA 결과인지 구분 불가).
- HIRA 프로젝트 출력은 별도 폴더 (`outputs/calibration_hira/`) 또는 파일명에 `_hira`
  suffix 부여 권장.

---

## 7. 미해결 의문점

분석하다 떠오른 도메인/구현 질문 + 사용자 답변 상태.

### A. HIRA 데이터 형태 — ✅ 결정됨

- **연령 segmentation**: 6 그룹
  `"0-5", "6-11", "12-17", "18-44", "45-64", "65+"`
  (raw 라벨엔 `"1. 0-5세"` 식 prefix가 있어 로더에서 정리).
- **시간 단위**: raw는 일별 (`요양개시일자` 컬럼). 주별 집계는 calibration 쪽에서 ISO week 변환.
- **공간**: 17 시도. fit target 기본은 수도권 3 시도 합산 (서울 11, 인천 28, 경기 41).
- **6→NIMS 15 가중 매핑**: 섹션 6에 상수 정의 (인구 비례 분배, 각 NIMS idx 가중치 합 = 1.0 확인).

### B. `gamma_report` 의미 — TBD (구현 단계 결정)

- ILI에선 외래환자 분모 흡수항. HIRA에선 reporting fraction 직접 해석 가능.
- `ParameterBounds.gamma_report` 좁힐지 (예: 0.05–0.5)는 실제 fit 1회 후 결정.
- 권장안에 "결정 필요" 코멘트만 남김.

### C. Seed 추정 — TBD

- HIRA seed: `I_baseline = hira_baseline_count / γ_assumed` (per-1000 곱셈 없음).
- `gamma_report_assumed` default 값은 1회 fit 후 corner solution 진단으로 결정.

### D. Loss / weights — TBD

- `poisson_log_likelihood(min_rate=0.1)` 는 ILI per-1000 튜닝값. HIRA count 스케일에서
  실측값 확인 후 재조정.
- `first_peak_only`은 ILI/HIRA 무관한 일반 기능 → 그대로 사용.

### E. 인구 분모와 단위 — TBD

- HIRA는 카운트 unitless. 인구 표준화 옵션 둘지는 fit 안정성 확인 후 결정.

### F. 출력 경로 — ✅ 결정됨

- 새 프로젝트가 통째로 분리되므로 (`kt_epimodel_hira/`) 디렉토리 충돌 없음.
- 각 프로젝트가 자기 `outputs/calibration/` 사용.
- 파일명 suffix: `2019-2020_by_age_LBFGS_HIRA.json` 식.
- ILI vs HIRA 결과 비교는 ad-hoc 노트북에서 처리 (스켈레톤 범위 밖).

### G. `kt_data` 측 HIRA 로더 — ⚠ 선행 작업 필요

- `kt_data` 에 `load_hira.py` 아직 없음. 섹션 6 권장안의 **0번 prerequisite**로 명시.

---

## 6. 복제 vs 교체 권장안

새 프로젝트 디렉토리: `kt_epimodel_hira/` (kt_epimodel sibling).

### 6.0 선행 작업 — `kt_data` 측 HIRA 로더 (★ prerequisite)

`kt_epimodel_hira` 작업 시작 전에 **`kt_data` 에 HIRA 로더 추가 필수**.

| 항목 | 명세 |
|---|---|
| 신규 파일 | `kt_data/src/kt_data/data/load_hira.py` |
| 원본 데이터 | `kt_data/data/external/hira/국민건강보험공단_감염성질환_인플루엔자__의료이용정보_20241231.xlsx` |
| 시트 | `외래입원주부상병`, `입원주부상병`, `시도코드` |
| 노출 심볼 (기존 `load_ili.py` 컨벤션) | `load_hira` 일별 long-form DataFrame (외래입원/입원 setting 선택)<br>주별 집계 헬퍼 (시도/연령 필터)<br>`HIRA_AGE_GROUPS = ["0-5", "6-11", "12-17", "18-44", "45-64", "65+"]`<br>`HIRA_SIDO_CODES` (17 시도 코드↔이름)<br>`SUDOGWON_SIDO_CODES = [11, 28, 41]` |
| kt_data `__init__.py` 재노출 | 위 심볼 모두 |

이 로더가 갖춰진 후에야 `kt_epimodel_hira` 작업 시작 가능.

---

### 6.1 [그대로 복제] — 코드 동일 (ILI/HIRA 무관)

다음 파일은 `kt_epimodel`에서 그대로 가져와 `kt_epimodel_hira/` 같은 경로에 두면 됨.
파일 내용 손댈 필요 없음 (단 패키지 이름 import 경로만 `kt_epimodel` → `kt_epimodel_hira`).

```
[그대로 복제 — 모델 코어]
- src/kt_epimodel_hira/__init__.py
- src/kt_epimodel_hira/model/__init__.py
- src/kt_epimodel_hira/model/compartments.py
- src/kt_epimodel_hira/model/foi.py
- src/kt_epimodel_hira/model/dynamics.py
- src/kt_epimodel_hira/model/mobility_tensor.py
- src/kt_epimodel_hira/model/parameters.py
    └─ NOTE: gamma_report validation (0 < x ≤ 1) 동일 유지 OK.
       Bound 좁히기는 ParameterBounds 쪽 (6.4-B 참조)
- src/kt_epimodel_hira/simulation/__init__.py
- src/kt_epimodel_hira/simulation/solver.py
- src/kt_epimodel_hira/simulation/runner.py
- src/kt_epimodel_hira/scenarios/__init__.py    (현재 빈 모듈)
- src/kt_epimodel_hira/viz/__init__.py          (현재 빈 모듈)

[그대로 복제 — calibration 내 ILI 무관 부분]
- src/kt_epimodel_hira/calibration/param_vector.py
    └─ 23-D vector layout, ParameterBounds, initial_guess 전부 재사용.
       γ_report bound 조정은 fit 1회 후 결정 (6.4-B)

[그대로 복제 — 테스트]
- tests/test_compartments.py
- tests/test_dynamics.py
- tests/test_foi.py
- tests/test_mobility_tensor.py
- tests/test_parameters.py
- tests/test_runner.py
- tests/test_solver.py
- tests/test_param_vector.py
- tests/test_setup.py
```

총 **17개 파일** 그대로 복제. 검증: `kt_epimodel_hira/`에 복제 후
`from kt_data.data.load_hira import ...` import는 **이 경로에 등장하지 않음** 확인.

### 6.2 [새로 작성] — HIRA용 대체 구현

ILI-specific 4개 파일 + 그에 대응하는 테스트 4개 + 노트북.

```
[새로 작성 — calibration]
- src/kt_epimodel_hira/calibration/__init__.py
    └─ docstring: "kt_epimodel_hira calibration — HIRA episode count fitting."
       hira_target / loss / optimizer / simple_model public symbol re-export

- src/kt_epimodel_hira/calibration/hira_target.py     ← ili_target.py 대체
    핵심 변경:
    (1) load_hira_target(season, ...): kt_data.load_hira 호출 (시도 필터 = 수도권 3개 합산)
    (2) load_hira_target_by_age(season, ...): 6 그룹 HIRA_AGE_GROUPS 순회
    (3) simulation_to_hira(daily_incidence, gamma_report, n_weeks):
        ─ 기존 `gamma_report * weekly / population * 1000.0` →
        ─ HIRA: `gamma_report * weekly_simulated_incidence`
          (per-1000 곱셈 + 인구 분모 제거, count 단위 그대로)
    (4) simulation_to_hira_by_age(daily_incidence_by_age, pop_15, gamma_report, ...):
        HIRA_GROUP_TO_NIMS_WEIGHTED 사용
    (5) season_start_date(season): ILI 버전 그대로 (도메인 독립)
    (6) poisson_log_likelihood(...): 함수 본체 그대로 import
        (calibration/hira_target.py에서 from ili_target import poisson_log_likelihood
         가 아니라, 분리 작성이 안전하므로 **본체 복사 + min_rate default 재튜닝 TBD**)
    (7) N_WEEKS = 52: 동일

    상수 (이 파일 또는 별도 mapping.py 두기 — A 답변 기반):
    ```python
    HIRA_GROUP_TO_NIMS_WEIGHTED = {
        "0-5":   {0: 1.0, 1: 0.2},
        "6-11":  {1: 0.8, 2: 0.4},
        "12-17": {2: 0.6, 3: 0.6},
        "18-44": {3: 0.4, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0},
        "45-64": {9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
        "65+":   {13: 1.0, 14: 1.0},
    }
    # 검증: 각 NIMS idx 가중치 합 = 1.0 (단위 테스트로)
    ```

- src/kt_epimodel_hira/calibration/simple_model.py     ← simple_model.py 대체
    (1) build_aggregated_inputs(): ILI 버전 그대로 복제 (인구·접촉·이동·고용만 사용)
    (2) estimate_initial_infected_from_hira(season, pop_15, week_zero_average_n, gamma_report_assumed):
        수식 변경:
        ─ 기존: I_group = ili_baseline[ag] * total_weighted_pop / 1000.0 / γ_assumed
        ─ HIRA: I_group = hira_baseline_count[ag] / γ_assumed
          (per-1000 + pop 분모 제거)
        그룹 매핑은 HIRA_GROUP_TO_NIMS_WEIGHTED (위 hira_target.py 상수)
    (3) _build_initial_state_with_age_seed(...): ILI 버전 그대로 (seed 벡터만 받음)
    (4) simulate_aggregated(...): ILI 버전 그대로

- src/kt_epimodel_hira/calibration/loss.py     ← loss.py 대체
    (1) make_loss_function(target, inputs, base_params, ...):
        target["hira_count"] (또는 target["ili_rate"]과 같은 키 이름 재사용 시
        키만 통일) 사용. simulation_to_hira(daily_inc, gamma_report, n_weeks) 호출
    (2) make_loss_function_by_age(...):
        simulation_to_hira_by_age 호출. 나머지 로직 동일
        (daily_inc_by_age = result.daily_new_infection_by_age() — vax flux 분리 동일)

- src/kt_epimodel_hira/calibration/optimizer.py     ← optimizer.py 대체
    (1) CalibrationResult dataclass: ILI 버전 그대로 (gamma_report_assumed 의미만 다름)
    (2) optimize_calibration / optimize_calibration_by_age:
        ─ load_ili_target → load_hira_target
        ─ make_loss_function → kt_epimodel_hira.calibration.loss.make_loss_function
        ─ estimate_initial_infected_from_ili → estimate_initial_infected_from_hira
        ─ verbose 로그 메시지 'ILI' → 'HIRA' (코사메틱)
    (3) save_result / load_result: 그대로 (JSON schema 동일)
    (4) _resolve_initial_vec: 그대로

[새로 작성 — 테스트]
- tests/test_hira_target.py       ← test_ili_target.py 대체
    포함:
    - HIRA_GROUP_TO_NIMS_WEIGHTED 가중치 합 = 1.0 검증
    - load_hira_target / load_hira_target_by_age smoke
    - simulation_to_hira count-scale 검증 (per-1000 곱셈 없음 확인)

- tests/test_loss.py
    ILI 버전 패턴 따라 작성, make_loss_function/by_age 호출 vec → finite NLL

- tests/test_optimizer.py
    ILI 버전 패턴 따라 작성. warm-start (initial_from_result) 테스트 포함

- tests/test_simple_model.py
    estimate_initial_infected_from_hira smoke

[새로 작성 — 노트북]
새 노트북 cell 골격은 ILI 노트북 그대로. 변경:
- ILI 함수 호출 → HIRA 등가 호출
- 시각화 y축 라벨 "ILI / 1000" → "HIRA count" 
- target_age['ili_rates'] → target_age['hira_counts'] (또는 통일 키 사용 결정)

- notebooks/calibration_01_fit.ipynb
- notebooks/calibration_02_full_fit.ipynb
- notebooks/calibration_03_by_age.ipynb
- notebooks/calibration_04_1_nelder_mead.ipynb
- notebooks/calibration_04_2_lbfgsb.ipynb
- notebooks/calibration_04_3_compare.ipynb
- notebooks/demo_single_season.ipynb     (mostly reusable, 데이터 import만 교체)
```

### 6.3 [파라미터화로 공용 가능] — 추천 안 함 (현재 단계)

다음 함수는 코드 차원에서 공용화 가능하나, **분리 작성 권장**:

| 함수 | 공용화 가능성 | 추천 |
|---|---|---|
| `season_start_date(season)` | 매우 높음 (date math only) | **분리 복제** |
| `poisson_log_likelihood(...)` | 높음 (단위 무관 NLL) | **분리 복제** (min_rate default 다르게 튜닝) |
| `_build_initial_state_with_age_seed(...)` | 매우 높음 | **분리 복제** |
| `optimize_calibration(_by_age)` 본체 | 중간 (target_loader / loss_factory callable 인자화 시) | **분리 복제** |

근거:
- 두 프로젝트가 독립적으로 진화하길 권장 (paper 별도 작성, citation 별도).
- 공용 베이스 패키지 (`kt_epimodel_common` 등) 추출은 한쪽이 안정화된 후 사후
  refactor가 안전.
- 현재 단계에서 양쪽이 다 변경 중이라 공용 코드를 함께 손대면 conflict 비용 큼.

추후 공용화 후보 (필요해지면 그때 고려):
- `poisson_log_likelihood` → `kt_modeling_common.likelihood.poisson_nll`
- `season_start_date`, `N_WEEKS` → `kt_modeling_common.season`
- `_build_initial_state_with_age_seed`, `simulate_aggregated` → 모델 코어가 같으므로 이미
  `kt_epimodel` 측에 두고 `kt_epimodel_hira`가 import해도 됨 (단, 이 경우 두 프로젝트가
  `kt_epimodel`을 dependency로 가짐 → 결합도 ↑, 권장 안 함).

### 6.4 [검토 필요 — 의사결정 보류] — 구현 단계 결정 사항

각 항목은 새 프로젝트 작성 시 해당 파일에 `# TODO: HIRA 결정 필요 ...` 코멘트로 남기고,
첫 fit 1회 후 결정.

**[6.4-B] `ParameterBounds.gamma_report` 좁히기**
- 파일: `src/kt_epimodel_hira/calibration/param_vector.py`
- 현재 ILI: `(0.01, 1.0)` (Phase A rollback)
- HIRA에선 의미가 **"reporting fraction"** (감염자 중 진료 청구된 비율) 로
  단순화 → ILI의 (0.01, 1.0)은 너무 넓음.
- **권장 1차 bound**: `gamma_report: tuple[float, float] = (0.05, 0.5)`
  - **상한 0.5**: 진료 에피소드가 실제 감염의 절반 이상이라는 가정은 비현실적
    (사람들 대부분이 감기약/자가관리). 0.5 위는 차단.
  - **하한 0.05**: 극단적으로 낮은 보고율은 다른 시즌 데이터에서 점검 필요.
    0.05 미만이면 모델 어딘가의 다른 문제 가능성 신호.
- 1차 fit 결과가 bound 끝에 박히면 재조정 (위쪽이면 [0.5, 0.7]로 확장 등).
- 코멘트 예시:
  ```python
  # HIRA reporting fraction 직접 해석 → ILI의 (0.01, 1.0) 보다 좁힘.
  # 상한 0.5: 청구가 실제 감염 ≥50%는 비현실 (대부분 자가관리).
  # 하한 0.05: 그 이하면 다른 시즌으로 sanity check 필요.
  # TODO HIRA-B: 1차 fit 후 bound 끝에 박히면 재조정.
  gamma_report: tuple[float, float] = (0.05, 0.5)
  ```

**[6.4-C] `estimate_initial_infected_from_hira()` default `gamma_report_assumed`**
- 파일: `src/kt_epimodel_hira/calibration/simple_model.py`
- 현재 ILI: `gamma_report_assumed=2.0` (default), 노트북에선 200.0 (corner 탈출용,
  단위환산 + 보고율 + 외래비율 흡수항 모두 포함).
- HIRA는 분모가 인구로 깨끗해져 reporting fraction 으로 **직접 해석 가능** →
  도메인 계산으로 1차값 추정 가능.

**도메인 계산 (2019-2020 시즌)**

| 항목 | 값 |
|---|---|
| 수도권 인구 (NIMS 15군 합산) | ≈ 2,600만 |
| 수도권 HIRA 인플루엔자 에피소드 (시즌 합) | ≈ 82만 |
| 가정 attack rate | 10–20% (한국 인플루엔자 시즌 통상치) |
| → 실제 감염자 추정 | 2,600만 × (0.10–0.20) = **260만–520만** |
| → reporting fraction = HIRA 카운트 / 실제 감염자 | 82만 / (260만–520만) = **0.158–0.315** |

- **권장 1차 default**: `gamma_report_assumed=0.2` (range `0.1–0.3` 중간값).
- ILI 200과 자릿수 차이: ILI의 200은 (per-1000 단위환산 ×1000) + (외래환자 / 인구)
  분모 비율 + 보고율 → 곱셈으로 누적된 큰 값. HIRA는 분모가 인구로 정리되면서
  순수 reporting fraction 값(<1)이 됨.
- 코멘트:
  ```python
  # HIRA reporting fraction 직접 해석 default.
  # 도메인 산수 (2019-2020 수도권):
  #   인구 ≈ 26M, HIRA 시즌 합 ≈ 820K, AR 가정 10-20% → 실제 감염 2.6M-5.2M
  #   reporting fraction = 820K / (2.6M-5.2M) = 0.16-0.32
  # → range 0.1-0.3 중간값 0.2 채택.
  # TODO HIRA-C: 첫 fit 후 corner solution 진단으로 조정.
  gamma_report_assumed: float = 0.2,
  ```

**[6.4-D] `poisson_log_likelihood(min_rate=?)` 재튜닝**
- 파일: `src/kt_epimodel_hira/calibration/hira_target.py`
- 현재 ILI: `min_rate=0.1` (per-1000 단위에 맞춤).
- HIRA: count 단위. 주당 평균 카운트의 1% 또는 1.0 등 실측 후 결정.
- 코멘트:
  ```python
  def poisson_log_likelihood(..., min_rate: float = 1.0):  # TODO HIRA: 첫 fit 후 재튜닝
      """...
      min_rate: count 단위. ILI 버전(0.1, per-1000)과 의미 다름.
      """
  ```

**[6.4-E] 인구 표준화 옵션**
- 파일: `src/kt_epimodel_hira/calibration/hira_target.py`
- 절대 카운트 fit vs 인구 표준화 (per 100K 등) fit 중 선택.
- 현재 default는 절대 카운트 (가장 단순). 시도 비교 시 필요해지면 옵션 추가.
- 코멘트:
  ```python
  # TODO HIRA-E: 인구 표준화 옵션 (수도권 합산 외에 시도별 비교 시 필요).
  # 현재 절대 카운트만 지원. simulation_to_hira(..., population_normalize=False) 인자 추가 검토.
  ```

### 6.5 새 프로젝트 디렉토리 트리 (최종 형태)

```
kt_epimodel_hira/                         (kt_epimodel sibling)
├── pyproject.toml                        (kt_epimodel의 그것 복제 → name=kt-epimodel-hira)
├── CLAUDE.md                             (HIRA 컨벤션으로 재작성)
├── README.md
├── .vscode/settings.json                 (kt_epimodel과 동일 패턴, ../kt_data/src extraPath)
├── docs/                                 (필요 시 새로 작성)
├── notebooks/
│   ├── demo_single_season.ipynb
│   └── calibration_01..04_3_*.ipynb      (6.2 참조 — 본체 복제 + 함수 호출 교체)
├── outputs/calibration/
│   └── 2019-2020_by_age_LBFGS_HIRA.json  (suffix _HIRA)
├── src/kt_epimodel_hira/
│   ├── __init__.py                       (그대로 복제, pkg 이름만)
│   ├── model/                            (전부 그대로 복제, 6.1 참조)
│   ├── simulation/                       (전부 그대로 복제, 6.1 참조)
│   ├── scenarios/__init__.py             (빈 모듈)
│   ├── viz/__init__.py                   (빈 모듈)
│   └── calibration/
│       ├── __init__.py                   (재export, 새로 작성)
│       ├── param_vector.py               (그대로 복제, 6.4-B 코멘트만 추가)
│       ├── hira_target.py                (새로 작성, 6.2)
│       ├── simple_model.py               (새로 작성, 6.2)
│       ├── loss.py                       (새로 작성, 6.2)
│       └── optimizer.py                  (새로 작성, 6.2)
└── tests/
    ├── test_compartments.py              \
    ├── test_dynamics.py                  |
    ├── test_foi.py                       |
    ├── test_mobility_tensor.py           |  6.1 참조 — 9개 그대로 복제
    ├── test_parameters.py                |
    ├── test_runner.py                    |
    ├── test_solver.py                    |
    ├── test_param_vector.py              |
    ├── test_setup.py                     /
    ├── test_hira_target.py               (새로 작성)
    ├── test_loss.py                      (새로 작성)
    ├── test_optimizer.py                 (새로 작성)
    └── test_simple_model.py              (새로 작성)
```

### 6.6 작업 순서 권장

1. **prerequisite**: `kt_data/src/kt_data/data/load_hira.py` 작성 + 테스트 + 커밋 (kt_data 쪽)
2. `kt_epimodel_hira/` 디렉토리 생성, `pyproject.toml` / `.vscode/settings.json` /
   `CLAUDE.md` skeleton 작성
3. 6.1의 17개 파일 그대로 복제 + import 경로 `kt_epimodel` → `kt_epimodel_hira` 일괄 치환
4. 6.1 복제 확인: `pytest tests/test_compartments.py tests/test_dynamics.py
   tests/test_foi.py tests/test_mobility_tensor.py tests/test_parameters.py
   tests/test_solver.py tests/test_param_vector.py tests/test_setup.py` 통과 확인
5. 6.2의 새 파일 작성 — 순서: `hira_target.py` → `simple_model.py` → `loss.py` →
   `optimizer.py` (의존성 순)
6. 새 테스트 4개 작성, 통과 확인
7. 노트북 새로 작성 (01 → 02 → 03 → 04_2 → 04_1 → 04_3 → demo)
8. 첫 fit 1회 (04_2 LBFGS) → 6.4의 B/C/D/E 의사결정 → 코멘트 자리에 실제 값 채워 넣기
9. Commit, push

---

## 부록 — 분석 도구 명령 (재현용)

```bash
# 패키지 트리
find src -type f -name "*.py" | sort

# kt_data import 인벤토리
grep -rnE "from kt_data|import kt_data" src

# ILI 키워드 hit
grep -rnE "ILI|ili|ILI_GROUP_TO_NIMS|load_ili|ili_target|simulation_to_ili|외래환자|의사환자|per_1000|gamma_report|γ_report|Poisson|NLL|weighted" src

# 노트북 ILI 의존
grep -nE "ili|ILI|load_ili|estimate_initial" notebooks/*.ipynb

# 테스트 ILI 의존
grep -rlE "ili|ILI|load_ili" tests
```
