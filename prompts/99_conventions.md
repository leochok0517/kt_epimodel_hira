# 매 세션 공통 주의사항

이 절은 다른 프롬프트를 실행할 때 항상 함께 적용된다.

## 재현성 원칙 (가장 중요)

- 이번 작업과 v5 논문 피팅의 차이는 **target · 시즌 집합 · 인구 셋뿐**이어야 한다.
- `nuts_seasonpop.py`, `nuts_seasonpop_v7_*.py`, `season_pop_setup.py`,
  `sens_common.py`, `final_pipeline_confirmed.py`, `multiseason_joint_sharedpi.py`,
  `src/kt_epimodel_hira/calibration/hira_target.py`, `model/`, `simulation/`,
  `calibration/param_vector.py` — **수정 금지**. v5 재현 경로다.
- 새 작업은 전부 `_fluA` 접미사 신규 파일로.
- 고정 파라미터 (`PHI` `GAMMA` `KAP` `BASE` `W` `NGM_F` `LOGIT_REF` `SIGMA_PIN`) 는
  `nuts_seasonpop` 에서 **import**. 값을 다시 적지 말 것.
- 기존 출력 파일 (`outputs/eda/nuts_seasonpop_v7_*`, `outputs/calibration/*`) 덮어쓰기 금지.
  새 산출물은 `outputs/fluA/` 또는 `*_fluA*` 이름으로.

## 코드 스타일

- Polars 우선 (Pandas 회피), NumPy 행렬 연산, type hint 사용.
- 그래프 라벨·변수명은 영어. 한글 라벨 금지.
- 테스트는 pytest.

## 실행 환경

- 가상환경: `.venv/` (uv). `.venv/bin/python` 으로 실행.
- JAX 는 CPU. 긴 실행은 반드시 detach:
  `nohup caffeinate -i .venv/bin/python <script> > logs/<name>.log 2>&1 & disown`
- HIRA xlsx (30MB) 로드는 약 90초, 프로세스 내 `lru_cache` 로 1회만.

## 수치 안정성

- 극단 영역 진입은 **prior 로 사전 차단**. NaN 발생 후 보정하지 않는다.
- 페널티는 유한값으로. ODE `max_steps` 는 타이트하게 유지.

## 보고 방식

- 결과는 표로. 추정 대신 실제 실행 로그·JSON 값을 인용한다.
- 수렴 실패·발산은 숨기지 말고 그대로 보고하고 원인 가설을 붙인다.
