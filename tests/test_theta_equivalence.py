"""T1/T2 for erlang_presymp_theta.py (prompts/09_theta_reimplementation.md).

T1: theta=0 must reproduce erlang_presymp.py's trajectory exactly.
T2: p_school == policy_school_baseline and p_work == policy_work_baseline
    (no policy differential) must make the baseline epidemic invariant to
    theta, at theta in {-0.25, 0, 0.75} (main_v5 line 346).

Uses the real fluA setup for one season (not a synthetic fixture) so the
kwargs are guaranteed self-consistent with what the production pipeline
actually passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
from kt_epimodel_hira.jax_model.erlang_presymp_theta import (
    simulate_jax_erlang_presymp_theta,
)
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex


@pytest.fixture(scope="module")
def season_kwargs():
    import fluA_setup as FS
    import nuts_seasonpop as NS

    C = FS.build_fluA_setup(["2016-2017"], verbose=False)
    s = "2016-2017"
    sh = dict(C["shared_by_s"][s])
    ngm = C["ngm3_by_s"][s]
    st0 = split_seed_to_erlang(C["st_by_s"][s])

    pi = jnp.asarray([0.44, 0.30, 0.06, 0.20])
    R0 = jnp.asarray(2.2)
    beta = derive_beta_from_R0_simplex(ngm, R0, pi, jnp.asarray(NS.PHI)) / ngm_factor(W_PRESYMP)

    kw = dict(sh)
    kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(NS.PHI)
    kw["kappa"] = jnp.asarray(NS.KAP)
    return st0, kw


def _run_original(st0, kw):
    return np.asarray(simulate_jax_erlang_presymp(st0, w_presymp=W_PRESYMP, **kw,
                                                     discretize_time=False))


def _run_theta(st0, kw, theta):
    return np.asarray(simulate_jax_erlang_presymp_theta(st0, w_presymp=W_PRESYMP, **kw,
                                                           theta=theta, discretize_time=False))


class TestT1ThetaZeroEquivalence:
    """theta=0 must be numerically identical to the untouched pipeline."""

    def test_baseline_policy(self, season_kwargs):
        st0, kw = season_kwargs
        kw = dict(kw)
        kw["p_school"] = 0.4
        kw["p_work"] = 0.4
        kw["policy_school_start_day"], kw["policy_school_end_day"] = 70.0, 113.0
        kw["policy_work_start_day"], kw["policy_work_end_day"] = 70.0, 113.0
        kw["policy_school_baseline"] = 0.6
        kw["policy_work_baseline"] = 0.6
        orig = _run_original(st0, kw)
        theta0 = _run_theta(st0, kw, 0.0)
        assert np.allclose(orig, theta0, rtol=1e-10), (
            f"max abs diff = {np.abs(orig - theta0).max()}"
        )

    def test_no_policy_window(self, season_kwargs):
        """Default (infinite) policy window - the production-fit path."""
        st0, kw = season_kwargs
        orig = _run_original(st0, kw)
        theta0 = _run_theta(st0, kw, 0.0)
        assert np.allclose(orig, theta0, rtol=1e-10)


class TestT2BaselineInvariance:
    """p == policy_baseline => Delta_b == 0 => other-FOI invariant to theta."""

    @pytest.mark.parametrize("theta", [-0.25, 0.0, 0.75])
    def test_p_equals_baseline_is_theta_invariant(self, season_kwargs, theta):
        st0, kw = season_kwargs
        kw = dict(kw)
        # policy window active, but the target retention equals the baseline
        # retention - no actual policy differential.
        kw["policy_school_start_day"], kw["policy_school_end_day"] = 70.0, 113.0
        kw["policy_work_start_day"], kw["policy_work_end_day"] = 70.0, 113.0
        kw["policy_school_baseline"] = 0.6
        kw["policy_work_baseline"] = 0.6
        kw["p_school"] = 0.6   # == policy_school_baseline
        kw["p_work"] = 0.6     # == policy_work_baseline

        ref = _run_theta(st0, kw, 0.0)
        out = _run_theta(st0, kw, theta)
        assert np.allclose(ref, out, rtol=1e-10), (
            f"theta={theta}: max abs diff = {np.abs(ref - out).max()}"
        )

    def test_matches_original_at_baseline(self, season_kwargs):
        """Belt and suspenders: the p==baseline trajectory should also match
        the untouched pipeline exactly, independent of theta."""
        st0, kw = season_kwargs
        kw = dict(kw)
        kw["policy_school_start_day"], kw["policy_school_end_day"] = 70.0, 113.0
        kw["policy_work_start_day"], kw["policy_work_end_day"] = 70.0, 113.0
        kw["policy_school_baseline"] = 0.6
        kw["policy_work_baseline"] = 0.6
        kw["p_school"] = 0.6
        kw["p_work"] = 0.6
        orig = _run_original(st0, kw)
        theta_big = _run_theta(st0, kw, 0.75)
        assert np.allclose(orig, theta_big, rtol=1e-10)
