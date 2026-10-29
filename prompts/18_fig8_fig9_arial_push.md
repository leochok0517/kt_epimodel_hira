# 18. Figure 8·9 Arial 재생성 · 원고 폴더 복사 · 커밋 · 푸시

작성 2026-10-29. 회사 Mac 에서 실행. 계산 없음, 5분.

## 배경

2026-10-29 Cowork 세션에서 이종걸 박사 검토(10/28)를 반영해 그림 스크립트 두 개를 고쳤다.

- `scripts/_fig_repro_bars_fluA4.py` (Figure 8): π 점유율을 JSON 의 100-draw 평균
  (41.7/31.7/7.4/19.2%) 대신 `outputs/eda/nuts_fluA4_raw.npz` 전체 8000 draw 평균
  (42.3/31.5/7.3/18.9%, 본문 표 3)으로. 범례 "transmissibility share" → "channel share".
  학교 배율 3.8× → 3.9×.
- `scripts/_fig_Re_readable_fluA4.py` (Figure 9, 보충 R_e 그림 3개): 아래 패널 채널 색을
  Figure 4A(`paper_figures_fluA4.COL_CHANNEL`)와 같게, 패널 제목 문구 수정.
- `scripts/figstyle.py`: 폰트 후보에 "Liberation Sans" 추가(Arial 이 있는 Mac 에서는 영향 없음).

그림은 Cowork Linux VM 에서 그려 **Liberation Sans 로 들어가 있다.** Mac 에서 Arial 로 다시 그린다.
코드 수정은 이미 작업 트리에 있다(미커밋).

## 하지 말 것

- 스크립트 수정 금지. 실행만.
- 원고 폴더 `~/Documents/cowork/sickleave_InfluenzaA/` 는 git 저장소가 아니다. 그림 파일 **복사만** 한다.
- 4번에 적힌 파일 외에는 스테이징하지 않는다.

## 0. 먼저 할 것

```bash
cd ~/Documents/python/NIMS/kt_epimodel_hira
ls .git/*.lock .git/objects/*.lock 2>/dev/null
rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock
git status -sb
```

기대값: `main`, origin 과 같음. 변경은 `scripts/` 3개, `figures/paper_fluA4/{pdf,png}/` 의
`repro_bars`·`repro_Re_readable_*` 10개, `?? prompts/18_fig8_fig9_arial_push.md`.
그 밖의 변경이 있으면 멈추고 보고한다.

## 1. 그림 재생성

```bash
.venv/bin/python scripts/_fig_repro_bars_fluA4.py
.venv/bin/python scripts/_fig_Re_readable_fluA4.py
```

첫 출력: `saved repro_bars (fluA4); children share = 77%, school R/pi = 3.9x`.
둘째: `saved repro_Re_readable_<시즌>` 네 줄. 다르면 멈추고 보고한다.

## 2. 폰트 확인

```bash
pdffonts figures/paper_fluA4/pdf/repro_bars.pdf
pdffonts figures/paper_fluA4/pdf/repro_Re_readable_2019-2020.pdf
```

`ArialMT`(+`Arial-BoldMT`) 있고 `LiberationSans` 없어야 한다. DejaVuSans(수식)는 정상.

## 3. 원고 폴더로 복사

```bash
M=~/Documents/cowork/sickleave_InfluenzaA/manuscript/figures_fluA4
for f in repro_bars repro_Re_readable_2016-2017 repro_Re_readable_2017-2018 repro_Re_readable_2018-2019 repro_Re_readable_2019-2020; do
  cp figures/paper_fluA4/pdf/$f.pdf $M/pdf/; cp figures/paper_fluA4/png/$f.png $M/png/; done
cd ~/Documents/cowork/sickleave_InfluenzaA/manuscript && latexmk -pdf -interaction=nonstopmode main_bmc.tex supplement_bmc.tex; cd -
```

latexmk 가 없으면 컴파일은 건너뛴다(Cowork 에서 한다).

## 4. 커밋 · 푸시

```bash
git add scripts/_fig_repro_bars_fluA4.py scripts/_fig_Re_readable_fluA4.py scripts/figstyle.py \
  figures/paper_fluA4/pdf/repro_bars.pdf figures/paper_fluA4/png/repro_bars.png \
  figures/paper_fluA4/pdf/repro_Re_readable_20*.pdf figures/paper_fluA4/png/repro_Re_readable_20*.png \
  prompts/18_fig8_fig9_arial_push.md
git status -sb
git commit -m "Figures 8-9: full-posterior channel shares, Fig 4A channel colours (co-author review 2026-10-28)"
git push origin main
```

non-fast-forward 면 강제 푸시하지 말고 멈추고 보고한다.

## 5. 보고

1번 출력, 2번 폰트 목록, 새 커밋 해시와 push 결과, 컴파일했다면 쪽수·경고.
