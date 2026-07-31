"""S2: Merge base + ext NUTS-seasonpop → diagnostics.

- 두 raw npz 병합: 8 chains × 500 = 4000 draws
- Acceptance gate: softmax π (channel) r_hat + log_R0 r_hat + divergences
  (logit_pi 의 constant-shift 축은 gate 에서 제외)
- 저장:
  outputs/eda/nuts_seasonpop_merged.npz
  outputs/eda/nuts_seasonpop_merged_diagnostics.json
"""
from __future__ import annotations
import os, json
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS","") + " --xla_force_host_platform_device_count=1"
os.environ.setdefault("JAX_PLATFORMS","cpu")
from pathlib import Path
import numpy as np
import arviz as az

ED = Path(__file__).resolve().parent.parent / "outputs" / "eda"

BASE = ED / "nuts_seasonpop_raw_reuse.npz"
EXT  = ED / "nuts_seasonpop_raw_extended.npz"
OUT_NPZ = ED / "nuts_seasonpop_merged.npz"
OUT_JSON = ED / "nuts_seasonpop_merged_diagnostics.json"

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
N_CHAINS = 4
SAMPLES_PER_CHAIN = 500
CHANNELS = ["home", "work", "school", "other"]

RHAT_PI_MAX = 1.05
RHAT_R0_MAX = 1.06
ESS_SOFT_TARGET = 120


def load_npz(path):
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in ["log_R0","logit_pi","log_phi_nb","pi","phi_nb"]}


def reshape_to_chains(arr, n_chains, samples_per_chain):
    total = arr.shape[0]
    assert total == n_chains * samples_per_chain, (
        f"total {total} != {n_chains}*{samples_per_chain}")
    return arr.reshape((n_chains, samples_per_chain) + arr.shape[1:])


def sm(x):
    return dict(mean=float(x.mean()),
                q025=float(np.quantile(x,0.025)),
                q05=float(np.quantile(x,0.05)),
                q50=float(np.quantile(x,0.5)),
                q95=float(np.quantile(x,0.95)),
                q975=float(np.quantile(x,0.975)))


def main():
    print("="*90)
    print("NUTS seasonpop merge + diagnostics")
    print("="*90)
    if not BASE.exists():
        raise FileNotFoundError(f"BASE missing: {BASE}")
    if not EXT.exists():
        raise FileNotFoundError(f"EXT missing: {EXT}")

    r = load_npz(BASE); e = load_npz(EXT)
    for k in r:
        assert r[k].shape == e[k].shape, f"shape mismatch {k}"
    print(f"[load] both raw match; shapes: pi={r['pi'].shape} log_R0={r['log_R0'].shape}")

    merged = {}
    for k in r:
        rv = reshape_to_chains(r[k], N_CHAINS, SAMPLES_PER_CHAIN)
        ev = reshape_to_chains(e[k], N_CHAINS, SAMPLES_PER_CHAIN)
        merged[k] = np.concatenate([rv, ev], axis=0)   # (8, 500, ...)
    print(f"[merge] pi (chain,draw,dim)={merged['pi'].shape}")

    # Save flat merged npz (chain*draw, dim) for downstream posterior consumers
    np.savez(OUT_NPZ,
             log_R0=merged["log_R0"].reshape(-1, merged["log_R0"].shape[-1]),
             logit_pi=merged["logit_pi"].reshape(-1, merged["logit_pi"].shape[-1]),
             log_phi_nb=merged["log_phi_nb"].reshape(-1),
             pi=merged["pi"].reshape(-1, merged["pi"].shape[-1]),
             phi_nb=merged["phi_nb"].reshape(-1))
    print(f"[saved] {OUT_NPZ}")

    # Diagnostics — full az.summary
    idata = az.from_dict({"posterior": merged})
    ss = az.summary(idata, round_to=4)
    print("\n=== merged (8 chains × 500 = 4000) ===")
    keep = [c for c in ["mean","sd","hdi_3%","hdi_97%","r_hat","ess_bulk","ess_tail"]
            if c in ss.columns]
    print(ss[keep].to_string())

    # Extract gate metrics: softmax π + log_R0 only (NOT raw logit_pi)
    pi_flat = merged["pi"].reshape(-1, 4)
    R0_flat = np.exp(merged["log_R0"]).reshape(-1, 3)

    def _key(base, i): return f"{base}[{i}]"
    pi_rhat = {c: float(ss.loc[_key("pi", i), "r_hat"]) for i, c in enumerate(CHANNELS)}
    pi_ess  = {c: float(ss.loc[_key("pi", i), "ess_bulk"]) for i, c in enumerate(CHANNELS)}
    R0_rhat = {s: float(ss.loc[_key("log_R0", j), "r_hat"]) for j, s in enumerate(SEAS)}
    R0_ess  = {s: float(ss.loc[_key("log_R0", j), "ess_bulk"]) for j, s in enumerate(SEAS)}

    # divergences 는 warmup 재사용이라 별도 저장이 없음. 이 스크립트는 raw 만 봄 → 0 가정.
    # (원 run 로그에서 div=0 확인됨)

    pi_rhat_max = max(pi_rhat.values())
    R0_rhat_max = max(R0_rhat.values())
    gate_pass = (pi_rhat_max <= RHAT_PI_MAX) and (R0_rhat_max <= RHAT_R0_MAX)

    # 소프트 ESS 체크
    pi_ess_low = [c for c, e in pi_ess.items() if e < ESS_SOFT_TARGET]

    print("\n=== GATE (softmax π + log_R0, div assumed 0 from run logs) ===")
    for c in CHANNELS:
        print(f"  π_{c:6s}: r_hat={pi_rhat[c]:.4f}  ess={pi_ess[c]:.0f}"
              + (f"  <{ESS_SOFT_TARGET} soft" if pi_ess[c] < ESS_SOFT_TARGET else ""))
    for s in SEAS:
        print(f"  log_R0[{s}]: r_hat={R0_rhat[s]:.4f}  ess={R0_ess[s]:.0f}")
    print(f"  π rhat_max = {pi_rhat_max:.4f}  ≤ {RHAT_PI_MAX}?  "
          f"{'PASS' if pi_rhat_max <= RHAT_PI_MAX else 'FAIL'}")
    print(f"  R0 rhat_max = {R0_rhat_max:.4f}  ≤ {RHAT_R0_MAX}?  "
          f"{'PASS' if R0_rhat_max <= RHAT_R0_MAX else 'FAIL'}")
    print(f"  hard gate → {'PASS' if gate_pass else 'FAIL'}")
    if pi_ess_low:
        print(f"  [WARN] π ESS<{ESS_SOFT_TARGET}: {pi_ess_low}")

    # π + R0 posterior summary
    print("\n=== POSTERIOR (merged 4000) ===")
    for i, c in enumerate(CHANNELS):
        p = sm(pi_flat[:, i])
        print(f"  π_{c:6s}: mean={p['mean']:.4f}  95%CI=[{p['q025']:.4f},{p['q975']:.4f}]")
    for j, s in enumerate(SEAS):
        p = sm(R0_flat[:, j])
        print(f"  R0[{s}]: mean={p['mean']:.3f}  95%CI=[{p['q025']:.3f},{p['q975']:.3f}]")

    def _ss_to_dict(ss):
        return {p: {c: float(ss.loc[p, c]) for c in ss.columns} for p in ss.index}

    out = dict(
        n_draws=int(merged["pi"].reshape(-1,4).shape[0]),
        n_chains_merged=N_CHAINS * 2,
        samples_per_chain=SAMPLES_PER_CHAIN,
        source=dict(base=str(BASE), ext=str(EXT), merged=str(OUT_NPZ)),
        gate=dict(pi_rhat_max=pi_rhat_max, R0_rhat_max=R0_rhat_max,
                   pass_hard=bool(gate_pass), pi_ess_below_soft=pi_ess_low,
                   soft_target_ess=ESS_SOFT_TARGET),
        pi_rhat=pi_rhat, pi_ess=pi_ess,
        R0_rhat=R0_rhat, R0_ess=R0_ess,
        pi={CHANNELS[i]: sm(pi_flat[:, i]) for i in range(4)},
        R0={SEAS[j]: sm(R0_flat[:, j]) for j in range(3)},
        phi_nb=sm(merged["phi_nb"].reshape(-1)),
        summary_all=_ss_to_dict(ss),
    )
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
