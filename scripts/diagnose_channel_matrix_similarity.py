"""Diagnose NIMS 4-channel contact matrix similarity — is work=0 an artefact of
C_work being structurally too similar to C_other?

Quantify:
1) Pairwise cosine similarity (flattened matrices) and Frobenius distance.
2) Diagonal dominance per channel = trace / total.
3) Row-sum profile variance per channel (flat profile → less specific to age).
4) Principal eigenvector (who dominates transmission if channel alone) —
   compare cosine between channels' eigenvectors.
5) Spectral gap λ₁/λ₂ per channel.

Both raw and row-normalised versions are reported (row-normalisation isolates
the "who-contacts-whom pattern" from the overall magnitude which is already
folded into β).

Decision guide (comments only):
- C_work ↔ C_other cosine > 0.9 AND eigenvector cosine ≈ 1
  → work/other structurally near-degenerate; matrix swap could revive β_w.
- C_work already visibly distinct → matrix is not the bottleneck; data itself
  gives no work signal.
"""
from __future__ import annotations
import os, json
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kt_data.data.load_contact import load_contact_matrices


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "channel_matrix_similarity.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "channel_similarity.png"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

CHANNELS = ["C_home", "C_work", "C_school", "C_other"]
SHORT = {"C_home": "home", "C_work": "work",
          "C_school": "school", "C_other": "other"}


def cosine_flat(A: np.ndarray, B: np.ndarray) -> float:
    a = A.reshape(-1); b = B.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / max(denom, 1e-30))


def frobenius(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro"))


def diagonal_dominance(A: np.ndarray) -> float:
    """trace / total. 1 = pure self-mixing, 0 = no self-contact."""
    tot = A.sum()
    return float(np.trace(A) / max(tot, 1e-30))


def row_profile_variance(A: np.ndarray) -> float:
    """Variance of row-sums (age-specific total contact profile)."""
    rs = A.sum(axis=1)
    return float(np.var(rs))


def principal_eigen(A: np.ndarray) -> tuple[complex, np.ndarray, float]:
    """Return (λ₁, v₁ real & normalised, λ₁/λ₂ spectral gap)."""
    eigvals, eigvecs = np.linalg.eig(A)
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    lam1 = eigvals[0]
    lam2 = eigvals[1] if len(eigvals) > 1 else 0.0
    v1 = np.real(eigvecs[:, 0])
    if v1.sum() < 0:
        v1 = -v1
    v1 = v1 / max(np.linalg.norm(v1), 1e-30)
    gap = float(np.abs(lam1) / max(np.abs(lam2), 1e-30))
    return complex(lam1), v1, gap


def row_normalise(A: np.ndarray) -> np.ndarray:
    rs = A.sum(axis=1, keepdims=True)
    rs_safe = np.where(rs > 1e-30, rs, 1e-30)
    return A / rs_safe


def total_normalise(A: np.ndarray) -> np.ndarray:
    tot = A.sum()
    return A / max(tot, 1e-30)


def pairwise_matrix(mats: dict[str, np.ndarray], fn) -> np.ndarray:
    keys = list(mats.keys())
    M = np.zeros((len(keys), len(keys)))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = fn(mats[ki], mats[kj])
    return M


def main():
    print("=" * 78)
    print("DIAGNOSE: NIMS 4-channel contact matrix similarity")
    print("=" * 78)

    data = load_contact_matrices()
    raws = {k: np.asarray(data[k]) for k in CHANNELS}
    for k in CHANNELS:
        print(f"  {k}: shape={raws[k].shape}  sum={raws[k].sum():.3f}")

    rows_n = {k: row_normalise(raws[k]) for k in CHANNELS}
    total_n = {k: total_normalise(raws[k]) for k in CHANNELS}

    # ─── Pairwise similarity (raw + normalised) ─────────────────
    print("\n[1] Pairwise cosine similarity  (flattened matrix)")
    for label, mats in [("raw", raws), ("row-normalised", rows_n),
                         ("total-normalised", total_n)]:
        print(f"\n  --- {label} ---")
        print("  " + "".join(f"{SHORT[k]:>10s}" for k in CHANNELS))
        C = pairwise_matrix(mats, cosine_flat)
        for i, ki in enumerate(CHANNELS):
            row = f"  {SHORT[ki]:6s}" + "".join(
                f"{C[i, j]:>10.4f}" for j in range(len(CHANNELS))
            )
            print(row)

    print("\n[1b] Pairwise Frobenius distance  (raw)")
    F = pairwise_matrix(raws, frobenius)
    print("  " + "".join(f"{SHORT[k]:>10s}" for k in CHANNELS))
    for i, ki in enumerate(CHANNELS):
        print(f"  {SHORT[ki]:6s}" + "".join(f"{F[i, j]:>10.3f}"
                                              for j in range(4)))

    # ─── Per-channel structural metrics ─────────────────────────
    print("\n[2/3/5] Per-channel metrics")
    print(f"  {'channel':>8s}  {'total sum':>10s}  "
          f"{'diag/total':>11s}  {'rowprof var':>12s}  "
          f"{'|λ1|':>9s}  {'|λ1/λ2|':>10s}")
    per_ch = {}
    eigvecs = {}
    for k in CHANNELS:
        A = raws[k]
        dd = diagonal_dominance(A)
        rv = row_profile_variance(A)
        lam1, v1, gap = principal_eigen(A)
        per_ch[k] = dict(
            total_sum=float(A.sum()),
            diag_dominance=dd,
            row_profile_var=rv,
            lambda1_abs=float(np.abs(lam1)),
            spectral_gap=gap,
            eigvec_v1=v1.tolist(),
        )
        eigvecs[k] = v1
        print(f"  {SHORT[k]:>8s}  {A.sum():>10.3f}  {dd:>11.4f}  "
              f"{rv:>12.4f}  {np.abs(lam1):>9.4f}  {gap:>10.4f}")

    # ─── Eigenvector cosine matrix ──────────────────────────────
    print("\n[4] Principal eigenvector cosine (raw matrices)")
    print("  " + "".join(f"{SHORT[k]:>10s}" for k in CHANNELS))
    EV = np.zeros((4, 4))
    for i, ki in enumerate(CHANNELS):
        for j, kj in enumerate(CHANNELS):
            vi = eigvecs[ki]; vj = eigvecs[kj]
            EV[i, j] = float(vi @ vj / max(np.linalg.norm(vi) *
                                             np.linalg.norm(vj), 1e-30))
        print(f"  {SHORT[ki]:6s}" + "".join(f"{EV[i, j]:>10.4f}"
                                              for j in range(4)))

    # ─── Highlight C_work vs C_other ────────────────────────────
    print("\n" + "=" * 78)
    print("  ★ C_work vs C_other highlights")
    print("=" * 78)
    print(f"  cosine similarity (raw)              = "
          f"{cosine_flat(raws['C_work'], raws['C_other']):.4f}")
    print(f"  cosine similarity (row-normalised)   = "
          f"{cosine_flat(rows_n['C_work'], rows_n['C_other']):.4f}")
    print(f"  cosine similarity (total-normalised) = "
          f"{cosine_flat(total_n['C_work'], total_n['C_other']):.4f}")
    print(f"  Frobenius distance (raw)             = "
          f"{frobenius(raws['C_work'], raws['C_other']):.3f}")
    print(f"  principal eigenvector cosine         = "
          f"{eigvecs['C_work'] @ eigvecs['C_other']:.4f}")
    print(f"  |λ1| work vs other                   = "
          f"{per_ch['C_work']['lambda1_abs']:.4f} vs "
          f"{per_ch['C_other']['lambda1_abs']:.4f}")
    print(f"  diag/total work vs other             = "
          f"{per_ch['C_work']['diag_dominance']:.4f} vs "
          f"{per_ch['C_other']['diag_dominance']:.4f}")

    # ─── Save JSON ─────────────────────────────────────────────
    out = dict(
        channels=CHANNELS,
        pairwise_cosine_raw=pairwise_matrix(raws, cosine_flat).tolist(),
        pairwise_cosine_rownorm=pairwise_matrix(rows_n, cosine_flat).tolist(),
        pairwise_cosine_totalnorm=pairwise_matrix(total_n, cosine_flat).tolist(),
        pairwise_frobenius_raw=pairwise_matrix(raws, frobenius).tolist(),
        eigenvector_cosine_raw=EV.tolist(),
        per_channel=per_ch,
        highlights_work_vs_other=dict(
            cosine_raw=cosine_flat(raws["C_work"], raws["C_other"]),
            cosine_rownorm=cosine_flat(rows_n["C_work"], rows_n["C_other"]),
            cosine_totalnorm=cosine_flat(total_n["C_work"], total_n["C_other"]),
            frobenius_raw=frobenius(raws["C_work"], raws["C_other"]),
            eigvec_cosine=float(eigvecs["C_work"] @ eigvecs["C_other"]),
        ),
    )
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Figure: 4 heatmaps + similarity bar ───────────────────
    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])
    vmax = max(A.max() for A in raws.values())
    for ci, k in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[0, ci])
        A = raws[k]
        im = ax.imshow(A, cmap="magma", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(f"{k}   sum={A.sum():.2f}\n"
                      f"diag/tot={per_ch[k]['diag_dominance']:.2f}  "
                      f"|λ1/λ2|={per_ch[k]['spectral_gap']:.2f}",
                      fontsize=10)
        ax.set_xlabel("contactee (0-14)")
        ax.set_ylabel("contactor (0-14)")
        fig.colorbar(im, ax=ax, fraction=0.045)

    # Bar: pairwise cosine matrix (raw and row-normalised)
    ax_bar = fig.add_subplot(gs[1, :])
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    pair_labels = [f"{SHORT[CHANNELS[i]]}↔{SHORT[CHANNELS[j]]}"
                    for i, j in pairs]
    cos_raw_vals = [pairwise_matrix(raws, cosine_flat)[i, j] for i, j in pairs]
    cos_rn_vals = [pairwise_matrix(rows_n, cosine_flat)[i, j] for i, j in pairs]
    eig_vals = [pairwise_matrix(raws, lambda A, B:
                                  eigvecs[[k for k in CHANNELS if raws[k] is A][0]]
                                  @ eigvecs[[k for k in CHANNELS if raws[k] is B][0]]
                                  if A is not B else 1.0)[i, j] for i, j in pairs]
    x = np.arange(len(pairs))
    w = 0.28
    ax_bar.bar(x - w, cos_raw_vals, w, label="cosine (raw)", color="#1a5490")
    ax_bar.bar(x, cos_rn_vals, w, label="cosine (row-norm)", color="#27ae60")
    ax_bar.bar(x + w, eig_vals, w, label="eigvec cosine", color="#c0392b")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(pair_labels, fontsize=9)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("similarity")
    ax_bar.axhline(0.9, color="grey", ls=":", lw=1)
    ax_bar.set_title("Channel-pair similarity  (grey dashed = 0.9 threshold)")
    ax_bar.legend(fontsize=9); ax_bar.grid(True, alpha=0.3, axis="y")

    fig.suptitle("NIMS 4-channel contact matrix similarity")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
