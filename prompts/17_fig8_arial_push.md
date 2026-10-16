# 17. Figure 8 Arial 재생성 · 원고 폴더 복사 · 커밋 · 푸시

작성 2026-10-16. 회사 Mac 에서 실행. 계산 없음, 5분.

## 배경

2026-10-15 Cowork 세션에서 공저자(이종걸) v7 코멘트 p.22 를 반영해
`scripts/_fig_repro_bars_fluA4.py` 를 고쳤다 — 패널 제목 제거, 왼쪽 위 A/B,
1 미만 배율은 소수 둘째 자리(0.12×). 그림은 Cowork 의 Linux VM 에서 그려서
**Arial 이 없어 Liberation Sans 로 들어가 있다.** 다른 그림은 전부 Arial 이므로
Mac 에서 다시 그려 통일한다. 코드·수치는 이미 커밋 `6c6156b` 에 들어 있고,
이번 작업은 **그림 파일만** 다시 만드는 것이다.

`6c6156b` 는 아직 push 되지 않았다(origin/main 보다 1 커밋 앞섬).

## 하지 말 것

- `scripts/_fig_repro_bars_fluA4.py` 와 `scripts/figstyle.py` 수정 금지. 그대로 실행만.
- 원고 폴더 `~/Documents/cowork/sickleave_InfluenzaA/` 는 git 저장소가 아니다.
  그림 2개를 **복사만** 하고, `.tex` 등 다른 파일은 건드리지 않는다.
- `sens_common.py` 등 v5 경로 파일은 커밋 대상이 아니다. 스테이징은 4번에 적힌 네 파일만.

## 0. 먼저 할 것

Cowork 원격 셸이 잠금 파일을 남겼을 수 있다. **git 이 실행 중이 아닌지 확인하고** 지운다.

```bash
cd ~/Documents/python/NIMS/kt_epimodel_hira
ls .git/*.lock .git/objects/*.lock 2>/dev/null
rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock
git status -sb
git log --oneline -3
```

기대값: 브랜치 `main`, `[ahead 1]`, HEAD 가 `6c6156b`. 작업 트리에는 이 프롬프트 자체
(`?? prompts/17_fig8_arial_push.md`)와 `M prompts/README.md` 만 있어야 한다.
그 밖의 변경이 있으면 멈추고 보고한다.

## 1. 그림 재생성

```bash
.venv/bin/python scripts/_fig_repro_bars_fluA4.py
```

출력이 정확히 `saved repro_bars (fluA4); children share = 77%, school R/pi = 3.8x`
여야 한다. 다르면 멈추고 보고한다(입력 JSON 이 바뀌었다는 뜻).

## 2. 폰트 확인

```bash
pdffonts figures/paper_fluA4/pdf/repro_bars.pdf
```

`ArialMT`, `Arial-BoldMT` 가 있어야 하고 `LiberationSans` 는 없어야 한다.
수식 기호용 `DejaVuSans`(-Oblique) 는 다른 그림에도 있으므로 정상.
`pdffonts` 가 없으면 `python -c` 로 PDF 바이트에서 `/BaseFont` 를 찾아 확인한다.

PNG 를 열어 눈으로도 확인한다: 패널 제목 없음, 왼쪽 위 굵은 A·B, work 막대
위 라벨 `0.12×`, school `3.8×`, other `1.4×`, B 패널 퍼센트 53/18/6/17/6/0.

## 3. 원고 폴더로 복사

```bash
M=~/Documents/cowork/sickleave_InfluenzaA/manuscript/figures_fluA4
cp figures/paper_fluA4/pdf/repro_bars.pdf $M/pdf/
cp figures/paper_fluA4/png/repro_bars.png $M/png/
cmp figures/paper_fluA4/pdf/repro_bars.pdf $M/pdf/repro_bars.pdf && echo copied
```

`latexmk` 가 있으면 원고를 다시 컴파일해 23쪽 Figure 8 을 확인한다
(`cd ~/Documents/cowork/sickleave_InfluenzaA/manuscript && latexmk -pdf -interaction=nonstopmode main_v8.tex`).
없으면 건너뛴다 — Cowork 세션에서 다시 컴파일한다.

## 4. 커밋 · 푸시

```bash
git add figures/paper_fluA4/pdf/repro_bars.pdf figures/paper_fluA4/png/repro_bars.png \
        prompts/17_fig8_arial_push.md prompts/README.md
git status -sb      # 스테이징된 것이 이 네 파일뿐인지 확인
git commit -m "Figure 8: regenerate on macOS with Arial, add prompts/17"
git push origin main
git status -sb      # 기대값: ## main...origin/main (ahead/behind 없음)
```

push 가 non-fast-forward 로 거절되면 **강제 푸시하지 말고** 멈추고 보고한다
(노트북에서 새 커밋이 올라왔다는 뜻 — `git pull` 후 충돌 여부를 사람이 판단).

## 5. 보고

다음만 짧게:

- 1번 출력 한 줄
- 2번 폰트 목록
- 커밋 해시 두 개(`6c6156b` 와 새 커밋)와 push 결과
- 원고 컴파일을 했다면 쪽수와 경고 유무
