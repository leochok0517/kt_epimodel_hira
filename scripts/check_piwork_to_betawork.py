"""π_work → β_work translation vs C2 detection floor (β = 0.05).

For each π_work in {0.03, 0.05, 0.06, 0.09, 0.12, 0.15} we run the same
point-fit as sens_workshare_kappa_v2 (φ U-shape, γ_report center, A-fix cov,
σ_per_channel [.15,.01,.05,.15]) and report the fitted β_work — is it above
or below the detection floor?

Setup reused verbatim from sens_workshare_kappa_v2 (imported, no fork).
"""
from __future__ import annotations
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse everything from sens_workshare_kappa_v2 — same setup, same fit.
from sens_workshare_kappa_v2 import (
    build_setup, point_fit_at_ws, build_pi_target, SEASON_LABEL,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "piwork_to_betawork.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "piwork_betawork.png"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

PI_WORK_GRID = [0.03, 0.05, 0.06, 0.09, 0.12, 0.15]
BETA_FLOOR = 0.05  # C2 detection floor


def main():
    print("=" * 78, flush=True)
    print(f"π_work → β_work check (detection floor β = {BETA_FLOOR}) — "
          f"{SEASON_LABEL}", flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    rows = []
    for pw in PI_WORK_GRID:
        print(f"\n── π_work pin = {pw:.2f} "
              f"(pi_target = {[round(float(x),4) for x in build_pi_target(pw)]}) ──",
              flush=True)
        fit = point_fit_at_ws(pw, setup)
        beta_w = fit["beta_4"][1]
        beta_o = fit["beta_4"][3]
        R0 = fit["R0"]
        pi_w_fit = fit["pi"][1]
        # β_w = (R0 / ρ) · π_w · (something) — but the simplest linear-in-π check
        # is the ratio β_w / π_w which should be roughly constant across the
        # scan (β_o / π_o similar). Also compute R0 / ρ implied.
        row = dict(
            pi_work_target=pw,
            pi_work_fit=pi_w_fit,
            beta_work=beta_w,
            beta_other=beta_o,
            R0=R0,
            beta_w_over_pi_w=beta_w / pi_w_fit,
            beta_o_over_pi_o=beta_o / fit["pi"][3],
            floor=BETA_FLOOR,
            above_floor=bool(beta_w > BETA_FLOOR),
            beta_w_ratio_to_beta_o=beta_w / beta_o,
        )
        rows.append(row)
        mark = "ABOVE" if row["above_floor"] else "below"
        print(f"    R0={R0:.3f}  β_work={beta_w:.5f}  β_other={beta_o:.5f}   "
              f"β_w/β_o={beta_w/beta_o:.3f}   [{mark} floor {BETA_FLOOR}]",
              flush=True)

    # Save JSON
    payload = dict(
        season=SEASON_LABEL,
        pi_work_grid=PI_WORK_GRID,
        beta_floor=BETA_FLOOR,
        rows=rows,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nsaved {OUT_JSON}", flush=True)

    # Console table
    print("\n" + "=" * 78, flush=True)
    print(f"{'π_work':>8s}  {'β_work':>9s}  {'β_other':>9s}  {'R0':>6s}  "
          f"{'β_w/β_o':>8s}  {'vs floor 0.05':>14s}", flush=True)
    print("-" * 78, flush=True)
    for r in rows:
        vs = "ABOVE" if r["above_floor"] else "below"
        print(f"{r['pi_work_target']:>8.2f}  {r['beta_work']:>9.5f}  "
              f"{r['beta_other']:>9.5f}  {r['R0']:>6.3f}  "
              f"{r['beta_w_ratio_to_beta_o']:>8.3f}  {vs:>14s}",
              flush=True)
    print("=" * 78, flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    pw = [r["pi_work_target"] for r in rows]
    bw = [r["beta_work"] for r in rows]
    bo = [r["beta_other"] for r in rows]
    ax.plot(pw, bw, "o-", color="C0", lw=2, label="β_work (fit)")
    ax.plot(pw, bo, "s--", color="C1", lw=1.5, alpha=0.7, label="β_other (fit)")
    ax.axhline(BETA_FLOOR, color="red", ls=":", lw=1.5,
                label=f"C2 detection floor β = {BETA_FLOOR}")
    for r in rows:
        y = r["beta_work"]
        ax.annotate(f"{y:.4f}", (r["pi_work_target"], y),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, ha="center")
    # Shade "data silence" region
    ax.axhspan(0, BETA_FLOOR, color="red", alpha=0.08)
    ax.text(pw[-1], BETA_FLOOR * 0.4,
            "data silence  (β < 0.05)",
            color="red", ha="right", fontsize=9, alpha=0.8)

    ax.set_xlabel("π_work (pin target)")
    ax.set_ylabel("β (fit)")
    ax.set_title("π_work → β_work translation vs C2 detection floor")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
