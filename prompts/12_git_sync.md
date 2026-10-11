# 12. Git 상태 확인 · 커밋 · 푸시

작성 2026-09-21. 대상 저장소는 두 개다.

- `kt_epimodel_hira` — https://github.com/leochok0517/kt_epimodel_hira.git
- `KT_data` — https://github.com/leochok0517/kt_data.git

원고 폴더 `sickleave_InfluenzaA/` 는 **git 저장소가 아니다.** 건드리지 말 것.

---

## 0. 먼저 할 것

Cowork 의 원격 셸이 `.git/index.lock` 을 남겨 놓았을 수 있다. 지우고 시작한다.

```bash
rm -f ~/Documents/python/NIMS/kt_epimodel_hira/.git/index.lock
rm -f ~/Documents/python/NIMS/KT_data/.git/index.lock
```

`index.lock` 이 있으면 `git status` 는 돌아가지만 커밋이 막힌다.

---

## 1. 상태 확인

두 저장소 각각에서:

```bash
git status -sb
git log --oneline -3
git fetch origin && git status -sb
```

**확인할 것**

| 항목 | 기대값 |
|---|---|
| 브랜치 | `main` |
| 원격 | 위 URL 두 개 |
| `KT_data` 작업 트리 | 깨끗함. 변경이 있으면 멈추고 내용을 먼저 보고할 것 |
| `kt_epimodel_hira` 변경 파일 | 아래 8개. 더 있거나 빠지면 멈추고 보고 |

```
 M figures/paper_fluA4/pdf/holiday.pdf
 M figures/paper_fluA4/pdf/nuts_posterior.pdf
 M figures/paper_fluA4/pdf/sens_S1_piwork_kappa_heatmap.pdf
 M figures/paper_fluA4/png/holiday.png
 M figures/paper_fluA4/png/nuts_posterior.png
 M figures/paper_fluA4/png/sens_S1_piwork_kappa_heatmap.png
 M scripts/paper_figures_fluA4.py
 M scripts/sens1_piwork_kappa_heatmap_fluA4.py
```

`git fetch` 후 `[behind N]` 이 뜨면 **커밋하지 말고 멈춰라.** 다른 기기에서 푸시한 것이
있다는 뜻이다. 먼저 보고하고 지시를 받을 것.

이 변경분의 출처: 2026-09-18 노트북에서 한 그림 수정 3건이고, 2026-09-21 에
`_handoff/bundle_20260921.tar.gz` 로 이 Mac 에 반영했다. 계산 결과가 아니라 그림 코드와
그 산출물이다.

---

## 2. 스크립트 diff 검토

커밋 전에 파이썬 두 파일의 diff 를 읽고, 의도한 변경만 들어 있는지 확인한다.

```bash
git diff -- scripts/paper_figures_fluA4.py scripts/sens1_piwork_kappa_heatmap_fluA4.py
```

**들어 있어야 하는 것**

- `paper_figures_fluA4.py::fig_F5` (holiday) — 연령대 3패널 분할
- `paper_figures_fluA4.py::fig_F6` (nuts_posterior) — 패널 A 를 home·work·school·other
  4색으로 분리, 패널 높이 축소
- `sens1_piwork_kappa_heatmap_fluA4.py` — 패널 타이틀 색 제거, 하단 설명문을 캡션으로 이동

**들어 있으면 안 되는 것**

- v5 재현 경로 수정: `nuts_seasonpop*.py`, `season_pop_setup.py`, `sens_common.py`,
  `policy_seasonpop.py`, `model/`, `simulation/`, `calibration/hira_target.py`,
  `foi_jax.py`, `erlang_presymp.py`
- 출력 경로 변경, 하드코딩된 수치, 한글 라벨

하나라도 걸리면 커밋하지 말고 보고할 것.

---

## 3. 커밋

`kt_epimodel_hira` 에서 8개를 한 커밋으로 묶는다. 다른 파일이 섞이지 않게 경로를 명시한다.

```bash
git add scripts/paper_figures_fluA4.py scripts/sens1_piwork_kappa_heatmap_fluA4.py \
        figures/paper_fluA4/pdf figures/paper_fluA4/png
git status --short          # staged 8개인지 다시 확인
git commit -m "Figures: split holiday by age band, color nuts_posterior panel A by channel, clean sens_S1 heatmap"
```

`KT_data` 는 변경이 없으면 커밋하지 않는다.

---

## 4. 푸시

```bash
git push origin main
```

`--force` 는 어떤 경우에도 쓰지 않는다. 거부당하면 멈추고 이유를 보고할 것.

---

## 5. 검증

```bash
git status -sb              # "## main...origin/main" 만, ahead/behind 표시 없음
git log --oneline -2
git diff origin/main --stat # 빈 출력
```

세 줄이 다 통과하면 끝이다. 결과를 이렇게 보고한다.

- 커밋 해시와 메시지
- 푸시된 파일 수
- `KT_data` 상태 (변경 없음 / 있었다면 무엇)
- 중간에 멈춘 항목이 있으면 그 이유

---

## 참고 — 하지 말 것

- 원고 폴더(`~/Documents/cowork/sickleave_InfluenzaA/`)를 git 에 넣지 말 것.
  번들 핸드오프로 관리한다.
- `outputs/` 아래 대용량 산출물을 새로 add 하지 말 것. `.gitignore` 를 확인하고,
  ignore 되어 있으면 그대로 둔다.
- `*.bak_*`, `_overwritten_*`, `_handoff/*.tar.gz` 는 커밋 대상이 아니다.
