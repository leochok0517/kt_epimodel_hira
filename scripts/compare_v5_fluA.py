"""Compare the influenza-A posterior against the published v5 posterior.

Produces:
  outputs/fluA/compare_v5_fluA.json   numbers behind everything below
  outputs/fluA/compare_v5_fluA.png    forest plots, pi and R0
  outputs/fluA/fit_total_fluA.png     observed A counts vs posterior-mean model

The v5 baseline is ``outputs/eda_v5_uncorrected/nuts_seasonpop_merged.npz``
(4000 draws) — the pre-reciprocity run the manuscript actually reports.
``outputs/eda/nuts_seasonpop_merged.npz`` holds the later corrected-C rerun and
is NOT what main_v5 quotes. ``nuts_seasonpop_v7reparam_raw.npz`` is the later
reparametrised run and is reported alongside when present, but the headline
comparison is against merged.

Sanity check that matters: 2016-17 and 2019-20 have an influenza-A share near
1.0, so their targets barely move. If their R0 moves a lot, something other
than the A correction changed - the script flags it rather than leaving it for
a reader to notice.
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / "outputs" / "fluA"
ED = REPO / "outputs" / "eda"
CHANNELS = ["home", "work", "school", "other"]
V5_SEASONS = ["2016-2017", "2017-2018", "2019-2020"]
#: seasons whose A share is near 1.0 — their R0 should barely move
STABLE_SEASONS = {"2016-2017", "2019-2020"}
R0_STABLE_TOL = 0.10          # relative shift beyond this is flagged

BLUE, RED, GREY = "#2166AC", "#B23A48", "#888888"


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    return float(x.mean()), float(np.percentile(x, lo)), float(np.percentile(x, hi))


def overlap(a, b):
    """Fraction of the shorter interval covered by the intersection."""
    lo = max(a[1], b[1]); hi = min(a[2], b[2])
    if hi <= lo:
        return 0.0
    return float((hi - lo) / min(a[2] - a[1], b[2] - b[1]))


def fig_forest(v5, fa, seasons, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw=dict(width_ratios=[1, 1.25]))

    ax = axes[0]
    for i, c in enumerate(CHANNELS):
        m5, l5, h5 = v5["pi"][c]
        ma, la, ha = fa["pi"][c]
        ax.plot([l5, h5], [i + 0.14] * 2, color=BLUE, lw=2)
        ax.plot(m5, i + 0.14, "o", color=BLUE, ms=6,
                label="v5 (A+B, 3 seasons)" if i == 0 else None)
        ax.plot([la, ha], [i - 0.14] * 2, color=RED, lw=2)
        ax.plot(ma, i - 0.14, "o", color=RED, ms=6,
                label="influenza A" if i == 0 else None)
    ax.set_yticks(range(len(CHANNELS)))
    ax.set_yticklabels(CHANNELS)
    ax.invert_yaxis()
    ax.set_xlabel("channel allocation $\\pi$")
    ax.set_title("Channel allocation", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.25)

    ax = axes[1]
    for i, s in enumerate(seasons):
        ma, la, ha = fa["R0"][s]
        new = s not in v5["R0"]
        col = RED if not new else "#8c6bb1"
        ax.plot([la, ha], [i - 0.14] * 2, color=col, lw=2)
        ax.plot(ma, i - 0.14, "o", color=col, ms=6,
                label=("influenza A" if i == 0 else
                       ("influenza A (new season)" if new and
                        all(x in v5["R0"] for x in seasons[:i]) else None)))
        if not new:
            m5, l5, h5 = v5["R0"][s]
            ax.plot([l5, h5], [i + 0.14] * 2, color=BLUE, lw=2)
            ax.plot(m5, i + 0.14, "o", color=BLUE, ms=6,
                    label="v5" if i == 0 else None)
    ax.set_yticks(range(len(seasons)))
    ax.set_yticklabels(seasons)
    ax.invert_yaxis()
    ax.set_xlabel("$R_0$")
    ax.set_title("Season reproduction number", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.25)

    fig.suptitle("Influenza-A refit versus the published v5 posterior",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_fit(C, seasons, pi_mean, R0_mean, path):
    """Observed influenza-A counts against the posterior-mean model."""
    import jax.numpy as jnp
    from kt_epimodel_hira.jax_model.erlang_presymp import (
        daily_new_onset_by_age_erlang_presymp, simulate_jax_erlang_presymp,
        split_seed_to_erlang,
    )
    from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax
    from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
    import nuts_seasonpop as NS

    n = len(seasons)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow),
                             squeeze=False)
    phi_j = jnp.asarray(NS.PHI)
    gamma_j = jnp.asarray(NS.GAMMA)

    for k, s in enumerate(seasons):
        ax = axes[k // ncol][k % ncol]
        beta = derive_beta_from_R0_simplex(
            C["ngm3_by_s"][s], jnp.asarray(R0_mean[s]),
            jnp.asarray(pi_mean), phi_j) / NS.NGM_F
        kw = dict(C["shared_by_s"][s])
        kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
        kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
        kw["phi_susc"] = phi_j
        kw["kappa"] = jnp.asarray(NS.KAP)
        kw["p_school"] = NS.BASE
        kw["p_work"] = NS.BASE
        st = simulate_jax_erlang_presymp(
            split_seed_to_erlang(C["st_by_s"][s]), w_presymp=NS.W, **kw,
            discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        pred = np.asarray(simulation_to_hira_by_age_jax(inc, gamma_j,
                                                        n_weeks=C["nw"]))
        obs = np.asarray(C["obs"][k])
        w = np.asarray(C["w"][k])[:, 0] > 0
        x = np.arange(C["nw"])
        ax.plot(x, C["obs_all"][k].sum(1), "-", color="#cccccc", lw=1.2,
                label="claims, A+B" if k == 0 else None)
        ax.plot(x, obs.sum(1), "o", color=GREY, ms=3,
                label="claims, influenza A" if k == 0 else None)
        ax.plot(x, pred.sum(1), "-", color=RED, lw=1.8,
                label="posterior mean model" if k == 0 else None)
        ax.axvline(np.argmax(~w) if (~w).any() else C["nw"], color="k",
                   lw=.6, ls=":", alpha=.5)
        ax.set_title(f"{s}   $R_0$={R0_mean[s]:.2f}", fontsize=10)
        ax.tick_params(labelsize=8)
        if k % ncol == 0:
            ax.set_ylabel("weekly claims", fontsize=9)
        if k // ncol == nrow - 1:
            ax.set_xlabel("week in season", fontsize=9)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Influenza-A fit — observed versus posterior-mean model",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fluA-npz", type=str, default=str(ED / "nuts_fluA_raw.npz"))
    # The manuscript's posterior is the PRE-reciprocity one. `outputs/eda/` was
    # overwritten by the corrected-C (v6) rerun on 2026-08-24, so the file that
    # main_v5 actually reports (pi = 0.442 / 0.298 / 0.065 / 0.195) survives only
    # under eda_v5_uncorrected/. Default to that; pass --v5-npz to compare
    # against the corrected rerun instead.
    ap.add_argument("--v5-npz", type=str,
                    default=str(REPO / "outputs" / "eda_v5_uncorrected"
                                / "nuts_seasonpop_merged.npz"))
    ap.add_argument("--seasons", nargs="*", default=None)
    ap.add_argument("--no-fit-figure", action="store_true")
    ap.add_argument("--tag", type=str, default="fluA",
                    help="output suffix; keeps runs with different season sets apart")
    args = ap.parse_args()
    tag = args.tag

    OUT.mkdir(parents=True, exist_ok=True)
    import fluA_setup as FS

    da = np.load(args.fluA_npz, allow_pickle=True)
    seasons = (args.seasons if args.seasons
               else (list(da["seasons"]) if "seasons" in da.files
                     else list(FS.SEASONS_DEFAULT)))
    seasons = [str(s) for s in seasons]

    fa = dict(pi={c: ci(da["pi"][:, i]) for i, c in enumerate(CHANNELS)},
              R0={s: ci(np.exp(da["log_R0"][:, j]))
                  for j, s in enumerate(seasons)},
              phi_nb=ci(da["phi_nb"]),
              n_draw=int(da["pi"].shape[0]))

    d5 = np.load(args.v5_npz)
    v5 = dict(pi={c: ci(d5["pi"][:, i]) for i, c in enumerate(CHANNELS)},
              R0={s: ci(np.exp(d5["log_R0"][:, j]))
                  for j, s in enumerate(V5_SEASONS)},
              phi_nb=ci(d5["phi_nb"]),
              n_draw=int(d5["pi"].shape[0]))

    print("=" * 78)
    print(f"v5   : {args.v5_npz}  ({v5['n_draw']} draws, {len(V5_SEASONS)} seasons)")
    print(f"fluA : {args.fluA_npz}  ({fa['n_draw']} draws, {len(seasons)} seasons)")
    print("=" * 78)
    print(f"\n{'channel':>8} {'v5 mean':>9} {'v5 95% CI':>20} "
          f"{'A mean':>9} {'A 95% CI':>20} {'overlap':>8}")
    pi_overlap = {}
    for c in CHANNELS:
        a, b = v5["pi"][c], fa["pi"][c]
        pi_overlap[c] = overlap(a, b)
        print(f"{c:>8} {a[0]:>9.3f} [{a[1]:>7.3f},{a[2]:>7.3f}]  "
              f"{b[0]:>9.3f} [{b[1]:>7.3f},{b[2]:>7.3f}] {pi_overlap[c]:>8.2f}")

    print(f"\n{'season':>11} {'v5 R0':>18} {'A R0':>18} {'shift':>8}")
    flags = []
    r0_shift = {}
    for s in seasons:
        b = fa["R0"][s]
        if s in v5["R0"]:
            a = v5["R0"][s]
            rel = (b[0] - a[0]) / a[0]
            r0_shift[s] = rel
            mark = ""
            if s in STABLE_SEASONS and abs(rel) > R0_STABLE_TOL:
                mark = "  <-- FLAG"
                flags.append(
                    f"{s}: influenza-A share is near 1.0 so the target barely "
                    f"changed, yet R0 moved {rel:+.1%}. Investigate the season "
                    f"set or the population before trusting the run.")
            print(f"{s:>11} {a[0]:>7.3f} [{a[1]:>5.3f},{a[2]:>5.3f}] "
                  f"{b[0]:>7.3f} [{b[1]:>5.3f},{b[2]:>5.3f}] {rel:>+7.1%}{mark}")
        else:
            print(f"{s:>11} {'—':>18} {b[0]:>7.3f} [{b[1]:>5.3f},{b[2]:>5.3f}] "
                  f"{'new':>8}")

    print(f"\nphi_nb   v5 {v5['phi_nb'][0]:.2f} "
          f"[{v5['phi_nb'][1]:.2f},{v5['phi_nb'][2]:.2f}]   "
          f"A {fa['phi_nb'][0]:.2f} [{fa['phi_nb'][1]:.2f},{fa['phi_nb'][2]:.2f}]")
    phi_note = ("overdispersion decreased (phi_nb up) — consistent with the "
                "influenza-B residual being removed"
                if fa["phi_nb"][0] > v5["phi_nb"][0] else
                "overdispersion increased (phi_nb down) — unexpected, the A "
                "correction should have removed residual structure, not added it")
    print(f"  -> {phi_note}")

    if flags:
        print("\n!! FLAGS")
        for f in flags:
            print(f"  - {f}")

    fig_forest(v5, fa, seasons, OUT / f"compare_v5_{tag}.png")
    print(f"\n[fig] {OUT / f'compare_v5_{tag}.png'}")

    if not args.no_fit_figure:
        C = FS.build_fluA_setup(seasons, verbose=False)
        pi_mean = np.array([fa["pi"][c][0] for c in CHANNELS])
        pi_mean = pi_mean / pi_mean.sum()
        fig_fit(C, seasons, pi_mean, {s: fa["R0"][s][0] for s in seasons},
                OUT / f"fit_total_{tag}.png")
        print(f"[fig] {OUT / f'fit_total_{tag}.png'}")

    payload = dict(
        v5_npz=args.v5_npz, fluA_npz=args.fluA_npz, seasons=seasons,
        v5=dict(pi=v5["pi"], R0=v5["R0"], phi_nb=v5["phi_nb"],
                n_draw=v5["n_draw"]),
        fluA=dict(pi=fa["pi"], R0=fa["R0"], phi_nb=fa["phi_nb"],
                  n_draw=fa["n_draw"]),
        pi_overlap=pi_overlap, r0_relative_shift=r0_shift,
        phi_nb_note=phi_note, flags=flags,
    )
    (OUT / f"compare_v5_{tag}.json").write_text(
        json.dumps(payload, indent=2, default=float))
    print(f"[json] {OUT / f'compare_v5_{tag}.json'}")


if __name__ == "__main__":
    main()
