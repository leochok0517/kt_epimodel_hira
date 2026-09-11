"""W-0 1단계 — 재모수화가 pi 에 유도하는 사전분포를 바꾸는지 확인 (우도 없음).

원본 4성분 Uniform(-10,10)+pin 좌표계와, 3성분 표집+4번째 0 고정 좌표계에서
pin 만 켠 채(우도 없음) NUTS로 표집해 pi 의 사전 주변분포를 비교한다.
몇 초 안에 끝난다 — 시뮬레이션이 전혀 들어가지 않는다.

Standalone. 저장소 핵심 코드 미수정.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import jax
import jax.random as random
from numpyro.infer import MCMC, NUTS

import nuts_seasonpop_v7_reparam as V7

REPO = SCRIPTS_DIR.parent
ED = REPO / "outputs" / "eda" / "timing"
ED.mkdir(parents=True, exist_ok=True)
OUT_JSON = ED / "w0_prior_check.json"

CHANNELS = ["home", "work", "school", "other"]
N_WARMUP = 1000
N_SAMPLES = 20000
N_CHAINS = 4


def sm(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), sd=float(x.std()),
                q05=float(np.quantile(x, 0.05)), q50=float(np.quantile(x, 0.5)),
                q95=float(np.quantile(x, 0.95)))


def run_prior(reparam: bool, seed: int):
    model = V7.build_model_v7(C_all=None, reparam=reparam, with_likelihood=False)
    kernel = NUTS(model, max_tree_depth=10, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                num_chains=N_CHAINS, chain_method="sequential", progress_bar=False)
    t0 = time.time()
    mcmc.run(random.PRNGKey(seed))
    wall = time.time() - t0
    samples = mcmc.get_samples(group_by_chain=False)
    pi = np.asarray(samples["pi"])
    return pi, wall


def main():
    print("[W-0 prior check] sampling prior(+pin) only, no likelihood, no simulation …")
    pi_orig, wall_orig = run_prior(reparam=False, seed=1)
    print(f"  original 4-comp coord: {pi_orig.shape[0]} draws, wall={wall_orig:.1f}s")
    pi_reparam, wall_reparam = run_prior(reparam=True, seed=2)
    print(f"  sum-fixed 3-comp coord: {pi_reparam.shape[0]} draws, wall={wall_reparam:.1f}s")

    report = {}
    print(f"\n{'channel':10s} {'orig mean':>10s} {'orig sd':>8s} {'reparam mean':>13s} "
          f"{'reparam sd':>11s} {'mean diff':>10s}")
    for i, c in enumerate(CHANNELS):
        so = sm(pi_orig[:, i]); sr = sm(pi_reparam[:, i])
        diff = sr["mean"] - so["mean"]
        report[c] = dict(original=so, reparam=sr, mean_diff=diff)
        print(f"{c:10s} {so['mean']:10.4f} {so['sd']:8.4f} {sr['mean']:13.4f} "
              f"{sr['sd']:11.4f} {diff:+10.4f}")
        print(f"{'':10s} q05={so['q05']:.4f} q50={so['q50']:.4f} q95={so['q95']:.4f}"
              f"   |   q05={sr['q05']:.4f} q50={sr['q50']:.4f} q95={sr['q95']:.4f}")

    max_mean_diff = max(abs(report[c]["mean_diff"]) for c in CHANNELS)
    max_sd_ratio = max(report[c]["reparam"]["sd"] / max(report[c]["original"]["sd"], 1e-9)
                        for c in CHANNELS)
    print(f"\nmax |mean diff| across channels: {max_mean_diff:.5f}")
    print(f"max (reparam sd / original sd) across channels: {max_sd_ratio:.4f}")

    verdict = "MATCH" if (max_mean_diff < 0.005 and 0.8 < max_sd_ratio < 1.25) else "MISMATCH"
    print(f"VERDICT: {verdict}")

    out = dict(n_warmup=N_WARMUP, n_samples=N_SAMPLES, n_chains=N_CHAINS,
                per_channel=report, max_mean_diff=max_mean_diff,
                max_sd_ratio=max_sd_ratio, verdict=verdict,
                wall_original=wall_orig, wall_reparam=wall_reparam)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
