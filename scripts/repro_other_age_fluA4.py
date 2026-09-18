"""T1 addendum -- other-channel reproduction share by age, fluA4 2019-2020.

main_v6.tex sec:theta (line ~574) needs "the exact other-channel reproduction
share and its 0-19 fraction", which reproduction_analysis_fluA4.py does not
compute directly: it decomposes R0 by channel (summed over age) and by age
(summed over channel) separately, but not jointly. This adds the joint
decomposition for the "other" channel only, using the same dominant
eigenvectors (u, v) of K_total as the rest of the static analysis, so that
the channel x age contributions are consistent with (sum to) the already
reported R_c["other"] and R0.

Reuses reproduction_analysis_fluA4's setup/posterior/build_K_channels code
unmodified (import, not re-derive) -- same N_POST/SEED, same representative
season, so the numbers are directly comparable to reproduction_numbers_fluA4_
2019-2020.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp

import reproduction_analysis_fluA4 as R

REPO = Path(__file__).resolve().parent.parent
FLUA = REPO / "outputs" / "fluA"


def main():
    season = R.S_REP
    C_all = R.FS.build_fluA_setup(R.SEASONS_FLUA4, verbose=False)
    setup = R._load_setup(season, C_all)
    pi_s, log_R0_s = R._load_posterior(R.N_POST, R.SEED)
    j_s = R.SEASONS_FLUA4.index(season)
    S_frac_15 = 1.0 - R.IMM

    N = pi_s.shape[0]
    R_other_by_age = np.zeros((N, 15))
    R0_check = np.zeros(N)
    R_other_total = np.zeros(N)

    for k in range(N):
        pi_k = pi_s[k]; R0 = float(np.exp(log_R0_s[k, j_s]))
        b0 = np.asarray(R.derive_beta_from_R0_simplex(setup["ngm3"], jnp.asarray(R0),
                                                         jnp.asarray(pi_k), jnp.asarray(R.PHI)))
        beta_h, beta_w, beta_s, beta_o = float(b0[0]), float(b0[1]), float(b0[2]), float(b0[3])
        Ks = R.build_K_channels(
            pop_15=setup["pop_15"], rho_flat=setup["rho"],
            C_home=setup["C_home"], C_school=setup["C_school"],
            C_work=setup["C_work"], C_other=setup["C_other"],
            beta_h=beta_h, beta_s=beta_s, beta_w=beta_w, beta_o=beta_o,
            phi_full=R.PHI, S_frac_15=S_frac_15, gamma=setup["gamma"],
            sf=1.0 + R.SEASONAL_AMP)
        rho, v, u = R.dominant_eigen(Ks["total"])
        R0_check[k] = rho
        R_other_by_age[k] = (u @ Ks["other"]) * v
        R_other_total[k] = float(u @ Ks["other"] @ v)

    def sm(x):
        x = np.asarray(x, dtype=float)
        return dict(mean=float(x.mean()), q05=float(np.quantile(x, 0.05)),
                    q95=float(np.quantile(x, 0.95)))

    share_0_19 = R_other_by_age[:, 0:4].sum(axis=1) / R_other_total
    out = dict(
        season=season, n_post=R.N_POST, seed=R.SEED,
        R_other_total=sm(R_other_total),
        R_other_share_pct_of_R0=sm(100.0 * R_other_total / R0_check),
        R_other_by_age_15={R.AGE15_LAB[b]: sm(R_other_by_age[:, b]) for b in range(15)},
        R_other_share_0_19=sm(share_0_19),
    )
    outp = FLUA / "repro_other_age_fluA4.json"
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"[{season}] other-channel R = {out['R_other_total']['mean']:.3f} "
          f"({out['R_other_share_pct_of_R0']['mean']:.1f}% of R0)")
    print(f"  0-19 share of other-channel R: {share_0_19.mean()*100:.1f}% "
          f"[{np.quantile(share_0_19,0.05)*100:.1f}, {np.quantile(share_0_19,0.95)*100:.1f}]")
    print(f"[json] {outp}")


if __name__ == "__main__":
    main()
