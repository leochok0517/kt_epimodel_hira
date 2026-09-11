"""S1 — 상호성 보정(방식 A) 접촉행렬을 kt_data 표준 경로에 배치.

원본을 명시적 이름으로 백업하고, 방식 A 보정본을 원래 파일명으로 교체한다.
empirical_matrices_15_all_weekday.npz 는 파이프라인에서 참조되지 않으므로
건드리지 않는다.

핵심 코드(kt_data 소스, kt_epimodel_hira 소스)는 건드리지 않는다 — 데이터
파일 교체만 수행한다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import phase1_reciprocity_v2 as P1

CONTACT_DIR = SCRIPTS_DIR.parent.parent / "kt_data" / "data" / "external" / "contact_matrices"
TERM = CONTACT_DIR / "empirical_matrices_15.npz"
VAC = CONTACT_DIR / "empirical_matrices_15_vacation.npz"
TERM_BACKUP = CONTACT_DIR / "empirical_matrices_15_raw_asymmetric.npz"
VAC_BACKUP = CONTACT_DIR / "empirical_matrices_15_vacation_raw_asymmetric.npz"

G_SOURCE = "data/population/national/pop_national_2024_01.csv"
G_SUM = 51_313_912


def build_replacement(orig_path: Path, period_label: str):
    d = np.load(orig_path, allow_pickle=True)
    raw = {c: np.asarray(d[c], dtype=np.float64) for c in P1.CHANNELS}
    G, G_path = P1.load_G("pop_national_2024_01.csv", G_SUM)
    symA = P1.method_A(raw, G)

    orig_meta = list(d["meta"]) if "meta" in d else []
    new_meta = [
        m.replace("symmetrize=none", "symmetrize=reciprocity_channelwise")
        for m in orig_meta
    ]
    new_meta += [
        f"G=national_2024_01_sum_{G_SUM}",
        "correction_method=A_channelwise_reciprocity",
        "source_prompt=PROMPT_claudecode_Phase1_상호성보정_v2.md",
        f"deployed={time.strftime('%Y-%m-%d')}",
        "original_backup=" + orig_path.stem + "_raw_asymmetric.npz",
    ]

    out = dict(d)
    for c in P1.CHANNELS:
        out[c] = symA[c]
    out["meta"] = np.array(new_meta)
    return out, symA, raw, G


def main():
    assert not TERM_BACKUP.exists(), f"백업 파일이 이미 존재: {TERM_BACKUP}"
    assert not VAC_BACKUP.exists(), f"백업 파일이 이미 존재: {VAC_BACKUP}"

    # 1) backup originals
    import shutil
    shutil.copy2(TERM, TERM_BACKUP)
    shutil.copy2(VAC, VAC_BACKUP)
    print(f"[backup] {TERM.name} -> {TERM_BACKUP.name}")
    print(f"[backup] {VAC.name} -> {VAC_BACKUP.name}")

    # 2) build corrected replacements
    out_term, symA_term, raw_term, G = build_replacement(TERM_BACKUP, "term")
    out_vac, symA_vac, raw_vac, _ = build_replacement(VAC_BACKUP, "vacation")

    print(f"[G] source=data/population/national/pop_national_2024_01.csv sum={G.sum():,}")
    assert G.sum() == G_SUM

    # 3) reciprocity check before writing (ego basis)
    for label, raw, symA in (("term", raw_term, symA_term), ("vacation", raw_vac, symA_vac)):
        for c in P1.CHANNELS:
            T = G[:, None] * symA[c]
            ok = np.allclose(T, T.T, atol=1e-8, rtol=1e-6)
            assert ok, f"reciprocity FAILED {label}/{c}"
        print(f"[check] {label}: reciprocity OK all 4 channels (ego basis, G-weighted)")

    # 4) write replacements to the ORIGINAL filenames
    np.savez(TERM, **out_term)
    np.savez(VAC, **out_vac)
    print(f"[deploy] wrote corrected matrices to {TERM}")
    print(f"[deploy] wrote corrected matrices to {VAC}")

    # 5) sums before/after
    for label, raw, out in (("term", raw_term, out_term), ("vacation", raw_vac, out_vac)):
        for c in P1.CHANNELS:
            print(f"  {label}/{c}: raw_sum={raw[c].sum():.3f}  "
                  f"symA_sum={out[c].sum():.3f}")

    # 6) README
    readme = CONTACT_DIR / "README_reciprocity.md"
    readme.write_text(f"""# 상호성 보정 (방식 A) — 배치 기록

작성일 {time.strftime('%Y-%m-%d')}. Phase 2 전면 재적합을 위해 다음 파일을
방식 A(채널별 개별 보정) 접촉행렬로 교체했습니다.

| 파일 | 상태 |
|---|---|
| `empirical_matrices_15.npz` | **교체됨** — 방식 A 보정본 |
| `empirical_matrices_15_vacation.npz` | **교체됨** — 방식 A 보정본 |
| `empirical_matrices_15_raw_asymmetric.npz` | 원본(미보정) 백업 — 학기 |
| `empirical_matrices_15_vacation_raw_asymmetric.npz` | 원본(미보정) 백업 — 방학 |
| `empirical_matrices_15_all_weekday.npz` | 건드리지 않음(파이프라인 미사용) |

## 보정 방법

방식 A(채널별 개별 보정), $G$ = 전국 2024-01 (합계 {G_SUM:,}),
`kt_epimodel_hira/data/population/national/pop_national_2024_01.csv`.

$$m^{{c,\\mathrm{{sym}}}}[a,b] = \\frac{{G_a\\, m^c[a,b] + G_b\\, m^c[b,a]}}{{2 G_a}}$$

보정은 npz 원본(ego 기준, `m[ego,contact]`)에서 수행했고, 모델에 넣을 때
`load_contact_matrices()`의 기본 transpose(`transpose_to_model_convention=True`)가
그대로 적용됩니다 — 이 파일들은 여전히 ego 기준(`rows=ego_age(i), cols=contact_age(j)`)
입니다.

## kt_data는 kt_epimodel(ILI)과 공유됩니다

**주의**: 이 경로(`kt_data/data/external/contact_matrices/`)는
`kt_epimodel`(ILI 전용 자매 프로젝트)과 **공유**합니다. 이 교체는
`kt_epimodel`에도 영향을 줍니다 — 되돌리기 전까지 두 프로젝트 모두 보정된
행렬을 씁니다.

## 되돌리는 방법

```bash
cd kt_data/data/external/contact_matrices
cp empirical_matrices_15_raw_asymmetric.npz empirical_matrices_15.npz
cp empirical_matrices_15_vacation_raw_asymmetric.npz empirical_matrices_15_vacation.npz
```

## 출처

`kt_epimodel_hira/scripts/phase1_reciprocity_v2.py`
(`PROMPT_claudecode_Phase1_상호성보정_v2.md` 실행 결과),
`kt_epimodel_hira/scripts/deploy_reciprocity_matrices.py` (이 배치 스크립트).
""")
    print(f"[readme] {readme}")


if __name__ == "__main__":
    main()
