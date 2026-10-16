"""Prior-to-posterior contraction of the channel allocation pi.

Answers one question: which parts of pi does the claims data actually
determine, and which are fixed by the pinning prior?

The model places the prior on ``logit_pi`` through ``numpyro.factor("pin_prior",
...)``, a Gaussian ``N(LOGIT_REF, SIGMA_PIN)`` restricted to the sum-zero
subspace (``centered = logit_pi - mean(logit_pi)``). Because softmax is
shift-invariant, sampling that restricted Gaussian and pushing it through
softmax gives the exact marginal prior on ``pi``. Comparing its spread with the
posterior's gives the contraction ``1 - Var_post / Var_prior``: 0 means the data
added nothing, 1 means the prior was irrelevant.

Numpy only - no JAX, no model evaluation, no refitting. Reads the saved
posteriors and writes ``outputs/fluA/prior_contraction_fluA4.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ED = ROOT / "outputs" / "eda"
OUT = ROOT / "outputs" / "fluA" / "prior_contraction_fluA4.json"

CHANNELS = ["home", "work", "school", "other"]
# Reference allocation and pinning widths, as stated in the manuscript
# (Section "Estimated parameters") and set in scripts/final_pipeline_confirmed.py.
PI_REF = np.array([0.290, 0.290, 0.060, 0.360])
SIGMA_PIN = np.array([0.15, 0.10, 0.05, 0.15])

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020"]
N_PRIOR = 600_000
SEED = 20260921

CONTRASTS = {
    "home_plus_work": np.array([1.0, 1.0, 0.0, 0.0]),
    "home_minus_work": np.array([-1.0, 1.0, 0.0, 0.0]),
    "home_minus_other": np.array([1.0, 0.0, 0.0, -1.0]),
    "work_minus_other": np.array([0.0, 1.0, 0.0, -1.0]),
}


def sample_prior_pi(n: int, seed: int) -> np.ndarray:
    """Exact marginal prior on pi induced by the pin_prior factor."""
    rng = np.random.default_rng(seed)
    lp = np.log(PI_REF)
    ref = lp - lp.mean()                      # LOGIT_REF, sums to zero
    cov_full = np.diag(SIGMA_PIN ** 2)
    one = np.ones(4)
    # N(ref, diag(sigma^2)) conditioned on 1'c = 0. 1'ref is already 0, so the
    # conditional mean stays at ref and only the covariance changes.
    cov = cov_full - np.outer(cov_full @ one, one @ cov_full) / (one @ cov_full @ one)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(4))
    c = ref + (L @ rng.normal(size=(4, n))).T
    e = np.exp(c - c.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def summarize(x: np.ndarray) -> dict:
    return dict(
        median=float(np.median(x)),
        q05=float(np.quantile(x, 0.05)),
        q95=float(np.quantile(x, 0.95)),
        q025=float(np.quantile(x, 0.025)),
        q975=float(np.quantile(x, 0.975)),
        sd=float(x.std()),
        width90=float(np.quantile(x, 0.95) - np.quantile(x, 0.05)),
    )


def compare(prior: np.ndarray, post: np.ndarray) -> dict:
    p, q = summarize(prior), summarize(post)
    return dict(
        prior=p,
        posterior=q,
        width90_ratio=q["width90"] / p["width90"],
        contraction=float(1.0 - post.var() / prior.var()),
        shift_in_prior_sd=float((q["median"] - p["median"]) / p["sd"]),
    )


def main() -> None:
    pri = sample_prior_pi(N_PRIOR, SEED)

    joint = np.load(ED / "nuts_fluA4_raw.npz", allow_pickle=True)["pi"]
    res: dict = {
        "note": (
            "contraction = 1 - Var_post/Var_prior. 0 means the data added nothing "
            "and the posterior is the prior. Negative means the posterior is wider "
            "than the prior. The prior is the exact marginal on pi induced by the "
            "pin_prior factor, drawn by Monte Carlo."
        ),
        "pi_ref": PI_REF.tolist(),
        "sigma_pin": SIGMA_PIN.tolist(),
        "n_prior_draws": N_PRIOR,
        "seed": SEED,
        "joint_4season": {"channels": {}, "contrasts": {}},
        "per_season": {},
    }

    for i, ch in enumerate(CHANNELS):
        res["joint_4season"]["channels"][ch] = compare(pri[:, i], joint[:, i])
    for name, v in CONTRASTS.items():
        res["joint_4season"]["contrasts"][name] = compare(pri @ v, joint @ v)

    for s in SEASONS:
        f = ED / f"nuts_per_season_{s}_raw.npz"
        if not f.exists():
            continue
        q = np.load(f, allow_pickle=True)["pi"]
        res["per_season"][s] = {
            ch: compare(pri[:, i], q[:, i]) for i, ch in enumerate(CHANNELS)
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    print(f"{'':<12}" + "".join(f"{c:>12}" for c in CHANNELS))
    print("contraction, joint 4-season fit")
    print(f"{'joint':<12}" + "".join(
        f"{res['joint_4season']['channels'][c]['contraction']:>+12.3f}" for c in CHANNELS))
    print("\ncontraction, single-season fits")
    for s, d in res["per_season"].items():
        print(f"{s:<12}" + "".join(f"{d[c]['contraction']:>+12.3f}" for c in CHANNELS))
    print("\ncontrasts, joint 4-season fit")
    for name, d in res["joint_4season"]["contrasts"].items():
        print(f"  {name:<18} {d['contraction']:>+7.3f}   "
              f"width90 {d['prior']['width90']:.3f} -> {d['posterior']['width90']:.3f}")
    print(f"\n[json] {OUT}")


if __name__ == "__main__":
    main()
