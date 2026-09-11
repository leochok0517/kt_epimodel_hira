"""v6 S2 — 단일 4체인×500 실행이 사전 등록 기준을 통과했을 때, 그 결과를
nuts_seasonpop_merge.py 와 동일한 스키마로 outputs/eda/nuts_seasonpop_merged.npz
+ _diagnostics.json 에 저장한다 (하위 스크립트들이 경로를 하드코딩하고 있어
파일명은 그대로 맞춰야 함). 원본 merge.py 는 reuse+extended 두 raw npz 를
요구하도록 하드코딩돼 있어 단일 실행에는 쓸 수 없으므로 이 스크립트로 대신한다.

두 번째 실행이 필요했던 경우에는 이 스크립트를 쓰지 않고, v6a/v6b 를
reuse/extended 이름으로 복사한 뒤 원본 nuts_seasonpop_merge.py 를 그대로 쓴다.

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
OUT_NPZ = ED / "nuts_seasonpop_merged.npz"
OUT_JSON = ED / "nuts_seasonpop_merged_diagnostics.json"

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
N_CHAINS = 4
CHANNELS = ["home", "work", "school", "other"]

RHAT_PI_MAX = 1.05
RHAT_R0_MAX = 1.06
ESS_SOFT_TARGET = 120


def load_npz(path):
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in ["log_R0", "logit_pi", "log_phi_nb", "pi", "phi_nb"]}


def reshape_to_chains(arr, n_chains, samples_per_chain):
    total = arr.shape[0]
    assert total == n_chains * samples_per_chain
    return arr.reshape((n_chains, samples_per_chain) + arr.shape[1:])


def sm(x):
    return dict(mean=float(x.mean()), q025=float(np.quantile(x, 0.025)),
                q05=float(np.quantile(x, 0.05)), q50=float(np.quantile(x, 0.5)),
                q95=float(np.quantile(x, 0.95)), q975=float(np.quantile(x, 0.975)))


def main():
    if len(sys.argv) < 2:
        print("usage: nuts_seasonpop_v6_finalize_single.py <raw_npz_path>")
        sys.exit(2)
    raw_path = Path(sys.argv[1])

    d = load_npz(raw_path)
    n_total = d["pi"].shape[0]
    samples_per_chain = n_total // N_CHAINS
    merged = {k: reshape_to_chains(v, N_CHAINS, samples_per_chain) for k, v in d.items()}

    np.savez(OUT_NPZ,
              log_R0=d["log_R0"], logit_pi=d["logit_pi"], log_phi_nb=d["log_phi_nb"],
              pi=d["pi"], phi_nb=d["phi_nb"])
    print(f"[saved] {OUT_NPZ}  (n_draws={n_total}, single-run 4ch x {samples_per_chain})")

    idata = az.from_dict({"posterior": merged})
    ss = az.summary(idata, round_to=4)

    def _key(base, i):
        return f"{base}[{i}]"

    pi_flat = d["pi"]; R0_flat = np.exp(d["log_R0"])
    pi_rhat = {c: float(ss.loc[_key("pi", i), "r_hat"]) for i, c in enumerate(CHANNELS)}
    pi_ess = {c: float(ss.loc[_key("pi", i), "ess_bulk"]) for i, c in enumerate(CHANNELS)}
    R0_rhat = {s: float(ss.loc[_key("log_R0", j), "r_hat"]) for j, s in enumerate(SEAS)}
    R0_ess = {s: float(ss.loc[_key("log_R0", j), "ess_bulk"]) for j, s in enumerate(SEAS)}

    pi_rhat_max = max(pi_rhat.values()); R0_rhat_max = max(R0_rhat.values())
    gate_pass = (pi_rhat_max <= RHAT_PI_MAX) and (R0_rhat_max <= RHAT_R0_MAX)
    pi_ess_low = [c for c, e in pi_ess.items() if e < ESS_SOFT_TARGET]

    def _ss_to_dict(ss):
        return {p: {c: float(ss.loc[p, c]) for c in ss.columns} for p in ss.index}

    out = dict(
        n_draws=n_total, n_chains_merged=N_CHAINS, samples_per_chain=samples_per_chain,
        note="single 4-chain run (fresh warmup, v6, 방식 A 보정 C) — nuts_seasonpop_merge.py 의 2-run merge 아님",
        source=dict(raw=str(raw_path), merged=str(OUT_NPZ)),
        gate=dict(pi_rhat_max=pi_rhat_max, R0_rhat_max=R0_rhat_max, pass_hard=bool(gate_pass),
                    pi_ess_below_soft=pi_ess_low, soft_target_ess=ESS_SOFT_TARGET),
        pi_rhat=pi_rhat, pi_ess=pi_ess, R0_rhat=R0_rhat, R0_ess=R0_ess,
        pi={CHANNELS[i]: sm(pi_flat[:, i]) for i in range(4)},
        R0={SEAS[j]: sm(R0_flat[:, j]) for j in range(3)},
        phi_nb=sm(d["phi_nb"]),
        summary_all=_ss_to_dict(ss),
    )
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
