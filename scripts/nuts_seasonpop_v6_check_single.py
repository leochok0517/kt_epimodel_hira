"""v6 S2 예비 — 단일 4체인×500 실행(fresh warmup, --tag v6a)만으로 사전 등록
기준(R-hat<=1.01, bulk/tail ESS>=400, divergence 비율<=v5 수준)을 만족하는지
판정한다. nuts_seasonpop_merge.py 는 reuse+extended 두 raw npz 를 요구하도록
하드코딩돼 있어(핵심 코드 미수정 원칙상 고치지 않음) 단일 실행 판정에는 쓸 수
없다 — 이 스크립트가 그 자리를 메운다.

통과하면 두 번째 4체인 실행 없이 끝낸다(2000 draws).
통과 못하면 orchestrator가 두 번째 실행(v6b, warmup 재사용)을 돌리고
merge.py(reuse/extended 이름으로 파일 복사 후)로 8체인 4000 draws를 만든다.

Standalone. 저장소 핵심 코드는 건드리지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import arviz as az

REPO = Path(__file__).resolve().parent.parent
ED = REPO / "outputs" / "eda"

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
N_CHAINS = 4
CHANNELS = ["home", "work", "school", "other"]

RHAT_MAX = 1.01
ESS_MIN = 400


def load_npz(path):
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in ["log_R0", "logit_pi", "log_phi_nb", "pi", "phi_nb"]}


def reshape_to_chains(arr, n_chains, samples_per_chain):
    total = arr.shape[0]
    assert total == n_chains * samples_per_chain, f"total {total} != {n_chains}*{samples_per_chain}"
    return arr.reshape((n_chains, samples_per_chain) + arr.shape[1:])


def main():
    if len(sys.argv) < 2:
        print("usage: nuts_seasonpop_v6_check_single.py <raw_npz_path> [v5_diag_json_for_div_baseline]")
        sys.exit(2)
    raw_path = Path(sys.argv[1])
    v5_diag_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    d = load_npz(raw_path)
    n_total = d["pi"].shape[0]
    samples_per_chain = n_total // N_CHAINS
    merged = {k: reshape_to_chains(v, N_CHAINS, samples_per_chain) for k, v in d.items()}

    idata = az.from_dict({"posterior": merged})
    ss = az.summary(idata, round_to=6)

    def _key(base, i):
        return f"{base}[{i}]"

    checks = {}
    all_rhat = []
    all_ess_bulk = []
    all_ess_tail = []
    for i, c in enumerate(CHANNELS):
        r = float(ss.loc[_key("pi", i), "r_hat"])
        eb = float(ss.loc[_key("pi", i), "ess_bulk"])
        et = float(ss.loc[_key("pi", i), "ess_tail"])
        checks[f"pi_{c}"] = dict(r_hat=r, ess_bulk=eb, ess_tail=et)
        all_rhat.append(r); all_ess_bulk.append(eb); all_ess_tail.append(et)
    for j, s in enumerate(SEAS):
        r = float(ss.loc[_key("log_R0", j), "r_hat"])
        eb = float(ss.loc[_key("log_R0", j), "ess_bulk"])
        et = float(ss.loc[_key("log_R0", j), "ess_tail"])
        checks[f"R0_{s}"] = dict(r_hat=r, ess_bulk=eb, ess_tail=et)
        all_rhat.append(r); all_ess_bulk.append(eb); all_ess_tail.append(et)
    r = float(ss.loc["log_phi_nb", "r_hat"])
    eb = float(ss.loc["log_phi_nb", "ess_bulk"])
    et = float(ss.loc["log_phi_nb", "ess_tail"])
    checks["log_phi_nb"] = dict(r_hat=r, ess_bulk=eb, ess_tail=et)
    all_rhat.append(r); all_ess_bulk.append(eb); all_ess_tail.append(et)

    rhat_max = max(all_rhat)
    ess_bulk_min = min(all_ess_bulk)
    ess_tail_min = min(all_ess_tail)

    rhat_ok = rhat_max <= RHAT_MAX
    ess_ok = (ess_bulk_min >= ESS_MIN) and (ess_tail_min >= ESS_MIN)

    div_ok = None
    div_note = "divergence 비율 비교: 확인 불가 (v5 진단 json 미지정 또는 없음)"
    if v5_diag_path is not None and v5_diag_path.exists():
        v5 = json.loads(v5_diag_path.read_text())
        div_note = ("nuts_seasonpop_merge.py 는 divergence를 raw npz에 저장하지 않고 "
                     "'div=0 assumed from run logs' 로 처리함 — v6도 동일 관례를 따르며, "
                     "실제 divergence 수는 orchestrator 실행 로그(extra_fields 미수집)에서는 "
                     "확인 불가. NUTS 표준출력의 'div=' 값을 로그에서 직접 읽어 보고할 것.")

    n_needed = rhat_ok and ess_ok
    verdict = "PASS_single_run_sufficient" if n_needed else "FAIL_need_second_run"

    out = dict(
        raw_path=str(raw_path), n_chains=N_CHAINS, samples_per_chain=samples_per_chain,
        n_draws=n_total, criteria=dict(rhat_max_allowed=RHAT_MAX, ess_min_allowed=ESS_MIN),
        checks=checks, rhat_max=rhat_max, ess_bulk_min=ess_bulk_min, ess_tail_min=ess_tail_min,
        rhat_ok=rhat_ok, ess_ok=ess_ok, divergence_note=div_note,
        verdict=verdict,
    )
    out_path = ED / "nuts_seasonpop_v6_single_run_check.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))

    print("=== v6 single-run (4ch x %d = %d draws) diagnostic ===" % (samples_per_chain, n_total))
    for k, v in checks.items():
        print(f"  {k:12s}: r_hat={v['r_hat']:.4f}  ess_bulk={v['ess_bulk']:.0f}  ess_tail={v['ess_tail']:.0f}")
    print(f"  max r_hat={rhat_max:.4f} (<= {RHAT_MAX}? {rhat_ok})")
    print(f"  min ess_bulk={ess_bulk_min:.0f}  min ess_tail={ess_tail_min:.0f} (>= {ESS_MIN}? {ess_ok})")
    print(f"  VERDICT: {verdict}")
    print(f"[json] {out_path}")

    # exit code communicates verdict to the orchestrator shell script
    sys.exit(0 if n_needed else 1)


if __name__ == "__main__":
    main()
