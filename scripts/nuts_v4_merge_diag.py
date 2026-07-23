"""NUTS v4 merged diagnostics — 기존(2000) + 확장(2000) = 4000 표본.

병합 타당성:
  두 run 모두 동일 warmup_state (post_warmup_state) 로 sampling 시작 →
  같은 mass matrix + step size 하에서 독립 rng(43 vs 100) 로 chain 진행.
  → 같은 posterior 로 수렴하는 8 chains 로 취급 (4+4).

계산: r_hat, ESS(bulk/tail), CI 병합 vs 원본 비교. trace plot.
"""
from __future__ import annotations
import os, json
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS","") + " --xla_force_host_platform_device_count=1"
os.environ.setdefault("JAX_PLATFORMS","cpu")
from pathlib import Path
import numpy as np
import arviz as az
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110, "savefig.dpi":150,
                      "axes.unicode_minus":False,
                      "font.family":"AppleGothic", "font.size":9})

ED = Path(__file__).resolve().parent.parent / "outputs" / "eda"
FIG = Path(__file__).resolve().parent.parent / "presentations" / "figures" / "v4"
FIG.mkdir(parents=True, exist_ok=True)

RAW_PATH = ED / "nuts_v4_full_raw.npz"
EXT_PATH = ED / "nuts_v4_full_extended.npz"
OUT_JSON = ED / "nuts_v4_merged_diagnostics.json"
FIG_TRACE = FIG / "nuts_v4_merged_trace.png"

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
N_CHAINS = 4                      # 각 run 4 chains
SAMPLES_PER_CHAIN = 500           # 각 run 500 samples/chain


def load_npz(path):
    d = np.load(path)
    keys = ["log_R0", "logit_pi", "log_phi_nb", "pi", "phi_nb"]
    return {k: np.asarray(d[k]) for k in keys}


def to_idata(samples_dict, n_chains, samples_per_chain):
    """Reshape (chains*samples, dim) → (chains, samples, dim) → arviz InferenceData."""
    post = {}
    for k, v in samples_dict.items():
        total = v.shape[0]
        assert total == n_chains * samples_per_chain, \
            f"{k}: total {total} != {n_chains}*{samples_per_chain}"
        # (chain, draw, [dim...])
        new_shape = (n_chains, samples_per_chain) + v.shape[1:]
        post[k] = v.reshape(new_shape)
    return az.from_dict({"posterior": post})


def summarise(idata, tag):
    ss = az.summary(idata, round_to=4)
    keep = [c for c in ["mean", "sd", "eti89_lb", "eti89_ub",
                          "r_hat", "ess_bulk", "ess_tail"] if c in ss.columns]
    ss = ss[keep]
    return ss


def print_summary(ss, tag):
    print(f"\n=== {tag} ===")
    print(ss.to_string())


def main():
    print("="*94)
    print("NUTS v4 병합 진단 — 원본(2000) + 확장(2000) = 4000")
    print("="*94)

    if not RAW_PATH.exists() or not EXT_PATH.exists():
        raise FileNotFoundError(f"필요 파일: {RAW_PATH}, {EXT_PATH}")

    raw = load_npz(RAW_PATH)
    ext = load_npz(EXT_PATH)
    for k in raw:
        assert raw[k].shape == ext[k].shape, f"shape mismatch: {k}"
    print(f"[load] raw/ext shapes match")
    for k in raw:
        print(f"  {k}: shape={raw[k].shape}")

    # ── 원본만 (2000 = 4 chains × 500) ──
    idata_raw = to_idata(raw, N_CHAINS, SAMPLES_PER_CHAIN)
    ss_raw = summarise(idata_raw, "raw (4 chains, 500 draws)")
    print_summary(ss_raw, "raw (원본 4 chains × 500 = 2000)")

    # ── 확장만 (2000) ──
    idata_ext = to_idata(ext, N_CHAINS, SAMPLES_PER_CHAIN)
    ss_ext = summarise(idata_ext, "ext")
    print_summary(ss_ext, "extended (추가 4 chains × 500 = 2000)")

    # ── 병합: 8 chains × 500 (동일 warmup, 다른 rng) ──
    merged = {}
    for k in raw:
        v_r = raw[k].reshape((N_CHAINS, SAMPLES_PER_CHAIN) + raw[k].shape[1:])
        v_e = ext[k].reshape((N_CHAINS, SAMPLES_PER_CHAIN) + ext[k].shape[1:])
        merged[k] = np.concatenate([v_r, v_e], axis=0)   # (8, 500, ...)
    idata_merged = az.from_dict({"posterior": merged})
    ss_merged = summarise(idata_merged, "merged")
    print_summary(ss_merged, "merged (8 chains × 500 = 4000)")

    # 병합 정합성 체크: 두 run의 posterior mean이 같은가
    print("\n=== 병합 정합성 (raw vs ext posterior mean) ===")
    for k in ["pi", "log_R0", "phi_nb"]:
        r_m = raw[k].mean(axis=0); e_m = ext[k].mean(axis=0)
        diff = np.abs(r_m - e_m) / (np.abs(r_m) + 1e-9)
        print(f"  {k}: raw_mean={np.round(r_m,4).tolist()}  "
              f"ext_mean={np.round(e_m,4).tolist()}  "
              f"rel_diff_max={diff.max():.4f}")

    # 병합 전후 비교 표
    print("\n=== 병합 전후 진단 (r_hat, ESS) ===")
    print(f"{'param':>18s}  {'raw rhat':>9s} {'ess':>7s}  {'ext rhat':>9s} {'ess':>7s}  "
          f"{'merged rhat':>11s} {'ess':>7s}  {'ess 증가배':>10s}")
    for pname in ss_raw.index:
        r_rh = float(ss_raw.loc[pname, "r_hat"])
        r_es = float(ss_raw.loc[pname, "ess_bulk"])
        e_rh = float(ss_ext.loc[pname, "r_hat"])
        e_es = float(ss_ext.loc[pname, "ess_bulk"])
        m_rh = float(ss_merged.loc[pname, "r_hat"])
        m_es = float(ss_merged.loc[pname, "ess_bulk"])
        print(f"{pname:>18s}  {r_rh:>9.4f} {r_es:>7.0f}  "
              f"{e_rh:>9.4f} {e_es:>7.0f}  "
              f"{m_rh:>11.4f} {m_es:>7.0f}  {m_es/max(r_es,1):>9.2f}×")

    # π_work 95% CI 병합 전후
    print("\n=== π_work (channel index 1) 95% CI 병합 전후 ===")
    for tag, arr in [("raw", raw["pi"][:,1]), ("ext", ext["pi"][:,1]),
                       ("merged", merged["pi"].reshape(-1,4)[:,1])]:
        m = float(arr.mean())
        q025, q05, q95, q975 = np.quantile(arr, [0.025, 0.05, 0.95, 0.975])
        print(f"  {tag:>7s}: mean={m:.4f}  90%[{q05:.4f},{q95:.4f}]  95%[{q025:.4f},{q975:.4f}]")

    # log_R0 시즌별 CI
    print("\n=== log_R0 시즌별 R0 95% CI ===")
    for j, s in enumerate(SEAS):
        for tag, arr in [("raw", np.exp(raw["log_R0"][:,j])),
                         ("ext", np.exp(ext["log_R0"][:,j])),
                         ("merged", np.exp(merged["log_R0"].reshape(-1,3)[:,j]))]:
            m = float(arr.mean())
            q025, q05, q95, q975 = np.quantile(arr, [0.025, 0.05, 0.95, 0.975])
            print(f"  {s} {tag:>7s}: mean={m:.3f}  90%[{q05:.3f},{q95:.3f}]  "
                  f"95%[{q025:.3f},{q975:.3f}]")

    # Trace plot (chain별 시계열, 수동 plot)
    print("\n[trace plot]")
    # merged shape: (8, 500, dim)
    channels = ["home", "work", "school", "other"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    # 1행: pi (4 채널)
    for k in range(4):
        r, c = 0, k if k < 3 else (1, k-3)   # first 3 in row 0, 4th moved to (1,0)
        if k < 3:
            ax = axes[0, k]
        else:
            ax = axes[1, 0]
        for ch in range(merged["pi"].shape[0]):
            ax.plot(merged["pi"][ch, :, k], lw=0.5, alpha=0.6)
        ax.set_title(f"π_{channels[k]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("draw"); ax.grid(alpha=0.3)
    # 2행 (1,1)~(1,2): log_R0[0..1]
    axes[1, 1].set_visible(False)
    axes[1, 2].set_visible(False)
    # 3행: log_R0[0,1,2]
    for j, s in enumerate(SEAS):
        ax = axes[2, j]
        for ch in range(merged["log_R0"].shape[0]):
            ax.plot(np.exp(merged["log_R0"][ch, :, j]), lw=0.5, alpha=0.6)
        ax.set_title(f"R0 {s}", fontsize=10, fontweight="bold")
        ax.set_xlabel("draw"); ax.grid(alpha=0.3)
    fig.suptitle("NUTS v4 merged (8 chains × 500 draws) — trace per chain",
                  fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(FIG_TRACE, bbox_inches="tight"); plt.close(fig)
    print(f"[trace] {FIG_TRACE}")

    # JSON dump
    def ss_to_dict(ss):
        return {p: {c: float(ss.loc[p, c]) for c in ss.columns}
                for p in ss.index}
    out = dict(
        counts=dict(raw_total=raw["pi"].shape[0], ext_total=ext["pi"].shape[0],
                     merged_total=merged["pi"].reshape(-1,4).shape[0],
                     merged_chains=N_CHAINS * 2, samples_per_chain=SAMPLES_PER_CHAIN),
        merge_note="8 chains (original 4 + extended 4). Same post_warmup_state, different rng keys.",
        raw=ss_to_dict(ss_raw),
        extended=ss_to_dict(ss_ext),
        merged=ss_to_dict(ss_merged),
        pi_work_ci=dict(
            raw=dict(mean=float(raw["pi"][:,1].mean()),
                       q025=float(np.quantile(raw["pi"][:,1], 0.025)),
                       q05=float(np.quantile(raw["pi"][:,1], 0.05)),
                       q95=float(np.quantile(raw["pi"][:,1], 0.95)),
                       q975=float(np.quantile(raw["pi"][:,1], 0.975))),
            merged=dict(mean=float(merged["pi"].reshape(-1,4)[:,1].mean()),
                          q025=float(np.quantile(merged["pi"].reshape(-1,4)[:,1], 0.025)),
                          q05=float(np.quantile(merged["pi"].reshape(-1,4)[:,1], 0.05)),
                          q95=float(np.quantile(merged["pi"].reshape(-1,4)[:,1], 0.95)),
                          q975=float(np.quantile(merged["pi"].reshape(-1,4)[:,1], 0.975))),
        ),
        R0_ci={s: dict(
            raw=dict(mean=float(np.exp(raw["log_R0"][:,j]).mean()),
                       q025=float(np.quantile(np.exp(raw["log_R0"][:,j]), 0.025)),
                       q975=float(np.quantile(np.exp(raw["log_R0"][:,j]), 0.975))),
            merged=dict(mean=float(np.exp(merged["log_R0"].reshape(-1,3)[:,j]).mean()),
                          q025=float(np.quantile(np.exp(merged["log_R0"].reshape(-1,3)[:,j]), 0.025)),
                          q975=float(np.quantile(np.exp(merged["log_R0"].reshape(-1,3)[:,j]), 0.975))),
        ) for j, s in enumerate(SEAS)},
    )
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
