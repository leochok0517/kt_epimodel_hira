"""Verify that the influenza-A path uses the reciprocity-corrected matrices.

The method-A correction was deployed into the kt_data standard paths on
2026-08-24 (`deploy_reciprocity_matrices.py`), replacing
`empirical_matrices_15.npz` and `..._vacation.npz` in place and keeping the
originals as `..._raw_asymmetric.npz`. Because the correction lives in the data
files rather than in a call-site flag, anything that loads matrices through
`kt_data.data.load_contact.load_contact_matrices` picks it up automatically —
including `fluA_setup`, which inherits the shared block from
`sens_common.build_setup`.

This script proves that rather than assuming it. Checks, in order:

1. the deployed npz carries the correction metadata and the raw backup exists;
2. the deployed matrices actually differ from the raw ones;
3. reciprocity holds channel-wise on what the loader returns:
   G_i C_ij == G_j C_ji, in the loader's model convention;
4. method A is idempotent, so `nuts_seasonpop_v7_run.py --corrected 1`
   re-applying it on top of the deployed files was a no-op rather than a
   double correction;
5. the matrices that reach `build_fluA_setup`'s per-season shared block are
   the corrected ones (needs jax; skipped with --no-setup).

Exit code is non-zero if any check fails.

Usage:
    python scripts/check_reciprocity_fluA.py [--no-setup]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

CHANNELS = ["home", "work", "school", "other"]
G_SUM = 51_313_912
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-setup", action="store_true",
                    help="skip check 5 (avoids importing jax)")
    args = ap.parse_args()

    import phase1_reciprocity_v2 as P1
    from kt_data.data.load_contact import load_contact_matrices

    CDIR = P1.KT_DATA_CONTACT_DIR
    pairs = [
        ("school term", CDIR / "empirical_matrices_15.npz",
         CDIR / "empirical_matrices_15_raw_asymmetric.npz"),
        ("winter break", CDIR / "empirical_matrices_15_vacation.npz",
         CDIR / "empirical_matrices_15_vacation_raw_asymmetric.npz"),
    ]

    G, G_path = P1.load_G("pop_national_2024_01.csv", G_SUM)
    print(f"G: {G_path}  sum={G.sum():,}\n")

    for label, live, backup in pairs:
        print(f"--- {label}: {live.name}")

        d = np.load(live, allow_pickle=True)
        meta = [str(m) for m in d["meta"]] if "meta" in d.files else []
        check("deployed file carries the correction metadata",
              any("correction_method=A_channelwise_reciprocity" in m for m in meta),
              "; ".join(m for m in meta if "correction" in m or "deployed" in m))
        check("raw asymmetric backup exists", backup.exists(), str(backup.name))

        if not backup.exists():
            continue
        raw = {c: np.asarray(np.load(backup, allow_pickle=True)[c], float)
               for c in CHANNELS}
        cur = {c: np.asarray(d[c], float) for c in CHANNELS}

        diffs = {c: float(np.abs(cur[c] - raw[c]).max()) for c in CHANNELS}
        check("deployed matrices differ from the raw ones",
              all(v > 0 for v in diffs.values()),
              "max abs diff " + ", ".join(f"{c} {v:.4f}" for c, v in diffs.items()))

        pre = {c: float(np.abs(raw[c] - raw[c].T).max()) for c in CHANNELS}
        print(f"      raw ego-matrix asymmetry (max |m - m^T|): "
              + ", ".join(f"{c} {v:.3f}" for c, v in pre.items()))

        # 3. reciprocity in the model convention the loader hands out
        loaded = load_contact_matrices(path=live)
        recip = {}
        for c in CHANNELS:
            # loader transposes ego -> [contact, participant]; undo for the test
            m_ego = loaded[f"C_{c}"].T
            T = G[:, None] * m_ego
            recip[c] = float(np.abs(T - T.T).max())
        scale = max(float(np.abs(G[:, None] * loaded[f"C_{c}"].T).max())
                    for c in CHANNELS)
        tol = 1e-6 * max(scale, 1.0)
        check("reciprocity holds on what the loader returns",
              all(v <= tol for v in recip.values()),
              "max |T - T^T| " + ", ".join(f"{c} {v:.3e}" for c, v in recip.items()))

        # 4. idempotency
        again = P1.method_A(cur, G)
        idem = {c: float(np.abs(again[c] - cur[c]).max()) for c in CHANNELS}
        check("method A is idempotent (a second correction is a no-op)",
              all(v <= 1e-9 for v in idem.values()),
              "max change " + ", ".join(f"{c} {v:.2e}" for c, v in idem.items()))
        print()

    if args.no_setup:
        print("--- skipping the fluA_setup check (--no-setup)")
    else:
        print("--- matrices reaching build_fluA_setup")
        import fluA_setup as FS
        C = FS.build_fluA_setup(FS.SEASONS_DEFAULT[:1], verbose=False)
        s = FS.SEASONS_DEFAULT[0]
        sh = C["shared_by_s"][s]
        live = load_contact_matrices()
        same = {c: float(np.abs(np.asarray(sh[f"C_{c}"]) - live[f"C_{c}"]).max())
                for c in CHANNELS}
        check("per-season shared block uses the corrected matrices",
              all(v <= 1e-12 for v in same.values()),
              "max diff " + ", ".join(f"{c} {v:.2e}" for c, v in same.items()))
        vac_keys = [f"C_{c}_vac" for c in CHANNELS]
        check("winter-break matrices are present in the shared block",
              all(k in sh for k in vac_keys),
              ", ".join(k for k in vac_keys if k in sh) or "none found")

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): " + "; ".join(FAILS))
        sys.exit(1)
    print("All reciprocity checks passed — the influenza-A fit uses the "
          "corrected matrices.")


if __name__ == "__main__":
    main()
