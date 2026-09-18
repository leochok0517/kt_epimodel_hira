"""Report both reproduction numbers the model defines, and the factor between them.

The quantity the sampler estimates is NOT the basic reproduction number. The
next-generation operator it inverts is

    K = (s_f / gamma) . diag(pop . phi . S) . C_eff . diag(1/N)

evaluated with S = 1 - R(0)_a (the season's initial immunity), s_f at the PEAK
of the seasonal forcing (1 + amp), and the school-term contact matrices. All
three depart from the R0 definition. It is not R_eff(t) either, since by the
seasonal peak the susceptible pool has already been drawn down. It is the
reproduction number the season would reach at peak forcing if susceptibility
still stood at its start-of-season level — the season's maximum potential
reproduction number.

This script reports it under its own name and derives the conventional R0
alongside:

  R_peak : initial immunity, seasonal forcing at peak, school-term matrices.
           The estimated parameter; what `log_R0` in the posterior npz carries.
  R_0    : full susceptibility, seasonal forcing at its annual mean (= base,
           since the forcing is base + amp*cos), school-term matrices.
           Derived, for comparison with the literature.

  R_0 = R_peak * rho(full S, mean forcing) / rho(initial immunity, peak forcing)

The ratio is taken on the same operator, so the (w+2)/3 Erlang factor and every
other common scaling cancels. rho depends on pi, so the factor is computed per
posterior draw, which also propagates the uncertainty.

Two definitional choices are deliberate and belong in the Methods text: the
annual mean rather than the peak as the R0 reference forcing, and the
school-term matrices rather than a term/break time average.

NOTE ON main_v5. Its Table tab:param_est reports 1.813 / 1.647 / 1.719 for the
three v5 seasons and calls them R0. Those values cannot be reproduced under the
convention its line 215 describes: with full susceptibility and mean forcing the
spectral radius is at its ceiling, and it falls about 9.8% short of what those
numbers require. Since susceptibility cannot exceed 1, no immunity profile
closes the gap. Their provenance is unknown — they appear in no script or output
in this repository — so they are not used as a validation target here. main_v5
line 439 already notes the two quantities differ; the fix is to name them.

Usage:
    python scripts/r0_conventional_fluA.py [--tag fluA4] [--seasons ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn  # noqa: E402

OUT = REPO / "outputs" / "fluA"
ED = REPO / "outputs" / "eda"

#: main_v5 Table tab:param_est. Reported for reference only — see the module
#: docstring for why these are not a validation target.
V5_PUBLISHED = {
    "2016-2017": 1.813, "2017-2018": 1.647, "2019-2020": 1.719,
}
V5_SEASONS = ["2016-2017", "2017-2018", "2019-2020"]
V5_NPZ = REPO / "outputs" / "eda_v5_uncorrected" / "nuts_seasonpop_merged.npz"


def conversion_factors(C, seasons, pi_draws, phi, imm, amp):
    """Per-season, per-draw rho_conventional / rho_normalization.

    Both operators are built from the same inputs; only the immunity profile and
    the seasonal factor differ. Everything else — matrices, population, rho,
    gamma — is held identical so the ratio isolates the convention.
    """
    n_draw = pi_draws.shape[0]
    out = np.zeros((n_draw, len(seasons)))
    phi_j = jnp.asarray(phi)

    for k, s in enumerate(seasons):
        pop_2d = np.asarray(C["pop_15_by_s"][s]).reshape(15, 1)
        sh = C["shared_by_s"][s]
        common = dict(
            pop_15=pop_2d, rho=np.asarray(sh["rho"]),
            C_home=np.asarray(sh["C_home"]), C_work=np.asarray(sh["C_work"]),
            C_school=np.asarray(sh["C_school"]), C_other=np.asarray(sh["C_other"]),
            gamma=float(sh["gamma"]),
        )
        ngm_norm = make_ngm_eigvalue_fn(
            **common, R0_immunity=np.asarray(imm), seasonal_factor=1.0 + float(amp))
        ngm_conv = make_ngm_eigvalue_fn(
            **common, R0_immunity=np.zeros(15), seasonal_factor=1.0)

        f_norm = jax.jit(lambda p: ngm_norm(p[0], p[1], p[2], p[3], phi_j))
        f_conv = jax.jit(lambda p: ngm_conv(p[0], p[1], p[2], p[3], phi_j))
        rn = np.asarray(jax.vmap(f_norm)(jnp.asarray(pi_draws)))
        rc = np.asarray(jax.vmap(f_conv)(jnp.asarray(pi_draws)))
        out[:, k] = rc / rn
        print(f"  {s}: factor mean {out[:, k].mean():.4f} "
              f"(rho_norm {rn.mean():.3f}, rho_conv {rc.mean():.3f})", flush=True)
    return out


def summarize(draws, seasons):
    return {
        s: dict(mean=float(draws[:, k].mean()),
                q025=float(np.percentile(draws[:, k], 2.5)),
                q975=float(np.percentile(draws[:, k], 97.5)))
        for k, s in enumerate(seasons)
    }


def v5_reference(NS, FS):
    """Compute R_peak and R_0 for the v5 seasons, beside what main_v5 printed.

    Not a gate. It documents, in the output, how far the published table sits
    from every configuration this code can produce, so the gap is on record
    rather than rediscovered later.
    """
    if not V5_NPZ.exists():
        print(f"[v5 reference] skipped — {V5_NPZ} not found\n")
        return None
    print("=" * 78)
    print("v5 REFERENCE (not a gate)")
    print(f"  posterior: {V5_NPZ}")
    print("=" * 78, flush=True)

    d = np.load(V5_NPZ)
    C = FS.build_fluA_setup(V5_SEASONS, verbose=False)
    fac = conversion_factors(C, V5_SEASONS, d["pi"], NS.PHI, NS.IMM, NS.F.S.AMP)
    peak = np.exp(d["log_R0"])
    r0 = peak * fac

    print(f"\n{'season':>11} {'R_peak':>9} {'R_0 (derived)':>15} "
          f"{'main_v5 table':>14} {'table/R_0':>10}")
    out = {}
    for k, s_ in enumerate(V5_SEASONS):
        a, b, p_ = peak[:, k].mean(), r0[:, k].mean(), V5_PUBLISHED[s_]
        out[s_] = dict(R_peak=float(a), R0=float(b), published=p_,
                       published_over_R0=float(p_ / b))
        print(f"{s_:>11} {a:>9.3f} {b:>15.3f} {p_:>14.3f} {p_ / b:>10.3f}")
    print("\n  The published column is not reproducible from this operator; see")
    print("  the module docstring. Reported so the discrepancy stays visible.\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fluA4")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--seasons", nargs="*", default=None)
    ap.add_argument("--no-v5-reference", action="store_true")
    args = ap.parse_args()

    import fluA_setup as FS
    import nuts_seasonpop as NS

    # NOTE: build_fluA_setup carries the seasons' own populations, matrices and
    # rho, which is exactly what the conversion needs. The immunity profile,
    # phi and the seasonal amplitude come from nuts_seasonpop so this script
    # cannot drift from the fit.
    v5_ref = None if args.no_v5_reference else v5_reference(NS, FS)

    npz = Path(args.npz or ED / f"nuts_{args.tag}_raw.npz")
    d = np.load(npz, allow_pickle=True)
    seasons = args.seasons or (
        [str(s) for s in d["seasons"]] if "seasons" in d.files else None)
    if seasons is None:
        raise SystemExit(f"{npz} has no season list — pass --seasons")

    print("=" * 78)
    print(f"CONVENTIONAL R0 — {args.tag}   seasons={seasons}")
    print(f"  posterior: {npz}  ({d['pi'].shape[0]} draws)")
    print("=" * 78, flush=True)

    C = FS.build_fluA_setup(seasons, verbose=False)
    fac = conversion_factors(C, seasons, d["pi"], NS.PHI, NS.IMM, NS.F.S.AMP)
    R_peak = np.exp(d["log_R0"])          # the estimated parameter
    R_0 = R_peak * fac                    # derived, conventional definition

    samp = summarize(R_peak, seasons)
    conv = summarize(R_0, seasons)

    print(f"\n{'season':>11} {'R_peak (estimated)':>26} "
          f"{'R_0 (derived)':>30} {'factor':>7}")
    for k, s in enumerate(seasons):
        a, b = samp[s], conv[s]
        print(f"{s:>11} {a['mean']:>8.3f} [{a['q025']:.3f}, {a['q975']:.3f}] "
              f"{b['mean']:>12.3f} [{b['q025']:.3f}, {b['q975']:.3f}] "
              f"{fac[:, k].mean():>7.4f}")

    print("\n  R_peak  initial immunity, seasonal forcing at peak, school-term C")
    print("          — the estimated parameter, what the fit moves")
    print("  R_0     full susceptibility, seasonal forcing at annual mean, school-term C")
    print("          — derived; the conventional definition")

    print("\nLaTeX rows for tab:param_est —")
    for s in seasons:
        a, b = samp[s], conv[s]
        lab = f"{s[:4]}--{s[7:9]}"
        print(f"& {lab} & ${a['mean']:.3f}\\ [{a['q025']:.3f},\\ {a['q975']:.3f}]$ "
              f"& ${b['mean']:.3f}\\ [{b['q025']:.3f},\\ {b['q975']:.3f}]$ \\\\")

    payload = dict(
        tag=args.tag, npz=str(npz), seasons=seasons,
        definitions=dict(
            R_peak="initial immunity R(0)_a, seasonal forcing at its peak (base+amp), "
                   "school-term contact matrices — the estimated parameter",
            R_0="full susceptibility, seasonal forcing at its annual mean (base), "
                "school-term contact matrices — derived, conventional definition",
            note="R_peak is not R_0 and not R_eff(t); see the module docstring"),
        amp=float(NS.F.S.AMP),
        factor_mean={s: float(fac[:, k].mean()) for k, s in enumerate(seasons)},
        R_peak=samp, R_0=conv,
        v5_reference=v5_ref,
    )
    outp = OUT / f"reproduction_numbers_{args.tag}.json"
    outp.write_text(json.dumps(payload, indent=2))
    print(f"\n[json] {outp}")


if __name__ == "__main__":
    main()
