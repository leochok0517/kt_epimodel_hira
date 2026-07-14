"""work_share sensitivity — UPPER extension (0.18 → 0.36).

Extends sens_workshare_kappa_v2 (which covered 0.03 → 0.15) into the upper
range needed to touch B (0.17) and A (0.29) literature targets and the
β_work detection floor (β = 0.05, reached around π_work ≈ 0.35).

Setup: identical to sens_workshare_kappa_v2 — φ U-shape, R(0) default, A-fix
coverage, γ = (0.40, 0.18, 0.25), σ_channel [.15, .01, .05, .15], HOLIDAY on,
non-work A relative share 0.408:0.085:0.507, κ ∈ {0.2, 0.4, 0.6}.

Grid (upper only): work_share ∈ {0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36}
× κ = 21 combos.  Same 12-start L-BFGS point fit.

Merged with existing v2 json → sens_workshare_full.json.
"""
from __future__ import annotations
import os, json, time, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sens_workshare_kappa_v2 import (
    build_setup, point_fit_at_ws, policy_averted_at_kappa, build_pi_target,
    run_forward, make_shared, build_gamma_15,
    PHI_USHAPE, GAMMA_CENTER, P_WORK_BASE, P_SCHOOL_BASE, P_WORK_SICK,
    SEASON_LABEL, SIGMA_PER_CHANNEL, A_REL, A_REL_NORM,
    HIRA_AGE_GROUPS,
)
from kt_epimodel_hira.calibration.hira_target import HIRA_GROUP_TO_NIMS_WEIGHTED
from kt_epimodel_hira.jax_model.solver_jax import daily_new_infection_by_age_jax
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON_UPPER = REPO_ROOT / "outputs" / "eda" / "sens_workshare_upper.json"
OUT_JSON_FULL = REPO_ROOT / "outputs" / "eda" / "sens_workshare_full.json"
OUT_JSON_V2 = REPO_ROOT / "outputs" / "eda" / "sens_workshare_kappa_v2.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "sens_workshare_full.png"
OUT_JSON_UPPER.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

WORK_SHARE_UPPER = [0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36]
KAPPA_GRID = [0.2, 0.4, 0.6]

BETA_FLOOR = 0.05
TARGETS = {"Italy": 0.033, "B": 0.17, "A": 0.29}


def build_hira_matrix():
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for j, w in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, j] = w
    return H


def redistribution_metrics(beta_4, kappa_scalar, setup, H, pop_6):
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15_j = jnp.asarray(build_gamma_15(*GAMMA_CENTER))
    shared_pol = make_shared(kappa_scalar, setup)
    beta_arr = np.array(beta_4)

    inc_b, _ = run_forward(beta_arr, phi_full_j, gamma_15_j,
                            P_SCHOOL_BASE, P_WORK_BASE, shared_pol, setup)
    inc_s, _ = run_forward(beta_arr, phi_full_j, gamma_15_j,
                            P_SCHOOL_BASE, P_WORK_SICK, shared_pol, setup)

    tot_15_b = np.asarray(inc_b).sum(axis=0)
    tot_15_s = np.asarray(inc_s).sum(axis=0)
    infb_6 = H @ tot_15_b
    infs_6 = H @ tot_15_s
    ab = infb_6 / pop_6; as_ = infs_6 / pop_6
    d_attack = as_ - ab
    d_counts = d_attack * pop_6
    pos = float(d_counts[d_counts > 0].sum())
    neg = float(-d_counts[d_counts < 0].sum())
    tr = pos / neg if neg > 1e-9 else float("inf")
    return dict(
        d_attack_by_age={a: float(d_attack[i])
                        for i, a in enumerate(HIRA_AGE_GROUPS)},
        attack_baseline_by_age={a: float(ab[i])
                                for i, a in enumerate(HIRA_AGE_GROUPS)},
        attack_sick_by_age={a: float(as_[i])
                            for i, a in enumerate(HIRA_AGE_GROUPS)},
        transfer_ratio=tr,
    )


def save_json(path, payload):
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    tmp.replace(path)


def main():
    print("=" * 78, flush=True)
    print(f"UPPER SENSITIVITY: work_share × κ  —  {SEASON_LABEL}", flush=True)
    print(f"  work_share ∈ {WORK_SHARE_UPPER}   κ ∈ {KAPPA_GRID}"
          f"   = {len(WORK_SHARE_UPPER)*len(KAPPA_GRID)} combos", flush=True)
    print(f"  targets: Italy=0.033  B=0.17  A=0.29   floor β_work={BETA_FLOOR}",
          flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()
    H = build_hira_matrix()
    pop_15_arr = np.asarray(setup["shared_base"]["pop_15"])
    if pop_15_arr.ndim == 2:
        pop_15_flat = pop_15_arr.sum(axis=1) if pop_15_arr.shape[0] == 15 \
            else pop_15_arr.sum(axis=0)
    else:
        pop_15_flat = pop_15_arr
    pop_6 = H @ pop_15_flat

    all_results = dict(
        season=SEASON_LABEL,
        config=dict(
            work_share_grid=WORK_SHARE_UPPER,
            kappa_grid=KAPPA_GRID,
            SIGMA_PER_CHANNEL=SIGMA_PER_CHANNEL.tolist(),
            A_REL=A_REL.tolist(),
            A_REL_NORM=A_REL_NORM.tolist(),
            beta_floor=BETA_FLOOR,
            targets=TARGETS,
        ),
        combos=[], errors=[], fits=[],
    )

    fits_by_ws = {}
    for ws in WORK_SHARE_UPPER:
        print(f"\n── fit ws = {ws:.2f}   pi_target = "
              f"{[round(float(x),4) for x in build_pi_target(ws)]}", flush=True)
        try:
            fit = point_fit_at_ws(ws, setup)
            fits_by_ws[ws] = fit
            all_results["fits"].append(fit)
            b = fit["beta_4"]
            mark = "ABOVE" if b[1] > BETA_FLOOR else "below"
            print(f"    R0={fit['R0']:.3f}  NLL={fit['nll']:.4e}  "
                  f"β_w={b[1]:.5f} [{mark} 0.05]  β_o={b[3]:.5f}  "
                  f"π={[round(p,4) for p in fit['pi']]}",
                  flush=True)
        except Exception:
            err = traceback.format_exc()
            print(f"  [FAIL fit ws={ws}] {err}", flush=True)
            all_results["errors"].append(f"fit ws={ws}: {err}")
        save_json(OUT_JSON_UPPER, all_results)

    print("\n" + "=" * 78, flush=True)
    print("Policy + redistribution at each (ws, κ)", flush=True)
    print("=" * 78, flush=True)
    for ws in WORK_SHARE_UPPER:
        fit = fits_by_ws.get(ws)
        if fit is None:
            for k in KAPPA_GRID:
                all_results["combos"].append(dict(
                    work_share=ws, kappa=k, policy=None, fit=None,
                    redistribution=None, error="no_fit",
                ))
            save_json(OUT_JSON_UPPER, all_results)
            continue
        for k in KAPPA_GRID:
            print(f"\n▶ ws={ws:.2f}  κ={k:.1f}", flush=True)
            try:
                pol = policy_averted_at_kappa(fit["beta_4"], k, setup)
                red = redistribution_metrics(fit["beta_4"], k, setup, H, pop_6)
                combo = dict(
                    work_share=ws, kappa=k, fit=fit,
                    policy=pol, redistribution=red,
                )
                tr = red["transfer_ratio"]
                tr_s = f"{tr:.3f}" if np.isfinite(tr) else "inf"
                print(f"  averted sick total = {pol['averted_sick_total_pct']:+.2f}%  "
                      f"school = {pol['averted_school_total_pct']:+.2f}%  "
                      f"transfer = {tr_s}", flush=True)
            except Exception:
                err = traceback.format_exc()
                combo = dict(work_share=ws, kappa=k, fit=fit,
                              policy=None, redistribution=None, error=err)
                print(f"  [FAIL] {err}", flush=True)
                all_results["errors"].append(f"combo ws={ws} k={k}: {err}")
            all_results["combos"].append(combo)
            save_json(OUT_JSON_UPPER, all_results)

    print(f"\nsaved {OUT_JSON_UPPER}", flush=True)

    # Merge with prior sens_workshare_kappa_v2 → sens_workshare_full
    print("\n" + "=" * 78, flush=True)
    print("Merging with sens_workshare_kappa_v2 → sens_workshare_full", flush=True)
    v2 = json.load(open(OUT_JSON_V2)) if OUT_JSON_V2.exists() else None
    merged = dict(
        season=SEASON_LABEL,
        config=all_results["config"],
        combos=[],
    )
    if v2 is not None:
        for c in v2.get("combos", []):
            merged["combos"].append(dict(
                work_share=c["work_share"], kappa=c["kappa"],
                fit=c.get("fit"), policy=c.get("policy"),
                source="v2_lower",
            ))
    for c in all_results["combos"]:
        merged["combos"].append(dict(
            work_share=c["work_share"], kappa=c["kappa"],
            fit=c.get("fit"), policy=c.get("policy"),
            redistribution=c.get("redistribution"),
            source="upper",
        ))
    save_json(OUT_JSON_FULL, merged)
    print(f"saved {OUT_JSON_FULL}  ({len(merged['combos'])} combos)", flush=True)

    make_full_plot(merged)
    print(f"saved {OUT_FIG}", flush=True)

    # Sign-flip boundary summary
    print_boundary_summary(merged)
    print("[DONE]", flush=True)


def make_full_plot(merged):
    combos = [c for c in merged["combos"] if c.get("policy")]
    fits = {(c["work_share"]): c["fit"] for c in combos if c.get("fit")}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    # Panel 1: averted sick % heatmap over full ws range
    ax = axes[0]
    ws_arr = sorted({c["work_share"] for c in combos})
    k_arr = sorted({c["kappa"] for c in combos})
    M = np.full((len(k_arr), len(ws_arr)), np.nan)
    for c in combos:
        i = k_arr.index(c["kappa"]); j = ws_arr.index(c["work_share"])
        M[i, j] = c["policy"]["averted_sick_total_pct"]
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", origin="lower",
                    vmin=-vmax, vmax=vmax)
    for i in range(len(k_arr)):
        for j in range(len(ws_arr)):
            v = M[i, j]
            if np.isfinite(v):
                col = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=8, color=col)
    ax.set_xticks(range(len(ws_arr)))
    ax.set_xticklabels([f"{x:.2f}" for x in ws_arr], rotation=45)
    ax.set_yticks(range(len(k_arr)))
    ax.set_yticklabels([f"{k:.1f}" for k in k_arr])
    ax.set_xlabel("work_share")
    ax.set_ylabel("κ")
    ax.set_title("averted sick_leave %  (full range)")
    for name, tgt in TARGETS.items():
        if tgt < ws_arr[0] - 0.01 or tgt > ws_arr[-1] + 0.01:
            continue
        j = np.interp(tgt, ws_arr, np.arange(len(ws_arr)))
        ax.axvline(j, color="black", ls=":", alpha=0.6)
        ax.text(j, len(k_arr)-0.4, name, color="black",
                fontsize=8, ha="center", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, label="averted %")

    # Panel 2: β_work vs π_work full range
    ax = axes[1]
    ws_sorted = sorted(fits.keys())
    bw = [fits[w]["beta_4"][1] for w in ws_sorted]
    bo = [fits[w]["beta_4"][3] for w in ws_sorted]
    ax.plot(ws_sorted, bw, "o-", color="C0", lw=2, label="β_work")
    ax.plot(ws_sorted, bo, "s--", color="C1", alpha=0.7, label="β_other")
    ax.axhline(BETA_FLOOR, color="red", ls=":", lw=1.5,
               label=f"detection floor β={BETA_FLOOR}")
    ax.axhspan(0, BETA_FLOOR, color="red", alpha=0.06)
    for name, tgt in TARGETS.items():
        ax.axvline(tgt, color="k", ls=":", alpha=0.4)
        ax.text(tgt, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 0.08,
                name, fontsize=8, ha="center", fontweight="bold")
    ax.set_xlabel("work_share  (= π_work pin)")
    ax.set_ylabel("β (fit)")
    ax.set_title("β_work vs π_work  —  full range")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"work_share sensitivity — full range 0.03 → 0.36  ({SEASON_LABEL})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)


def print_boundary_summary(merged):
    combos = [c for c in merged["combos"] if c.get("policy")]
    print("\n" + "=" * 78, flush=True)
    print("Sign-flip boundary per κ (averted sick total %)", flush=True)
    print("=" * 78, flush=True)
    for k in sorted({c["kappa"] for c in combos}):
        rows = sorted(
            [c for c in combos if c["kappa"] == k],
            key=lambda x: x["work_share"],
        )
        signs = [(r["work_share"], r["policy"]["averted_sick_total_pct"])
                for r in rows]
        # Find sign-change points
        first_pos = None
        for w, v in signs:
            if v > 0:
                first_pos = w; break
        print(f"  κ={k:.1f}  first positive averted at ws ≥ "
              f"{first_pos if first_pos is not None else 'none'}",
              flush=True)
        for w, v in signs:
            mark = "+" if v > 0 else "-"
            note = ""
            for name, tgt in TARGETS.items():
                if abs(w - tgt) < 0.02:
                    note = f"  ← near {name}({tgt})"
            print(f"    ws={w:.2f}  averted = {v:+7.2f}% [{mark}]{note}",
                  flush=True)


if __name__ == "__main__":
    main()
