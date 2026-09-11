"""Phase 2 W-0 — logit_pi 비식별(평평한) 방향 제거 재모수화.

원본 scripts/nuts_seasonpop.py::build_model 은 건드리지 않는다(v5 재현용
보존). 이 모듈은 build_model 의 수정 사본이다 — logit_pi 4성분을 직접
표집하는 대신, 3성분만 표집하고 네 번째를 0으로 고정한다. pi=softmax(logit_pi)
는 이동불변이라 π 자체의 추론 대상은 바뀌지 않는다(같은 모형의 다른 좌표계).

make_joint_loss 는 nuts_seasonpop.py 에서 그대로 import 해서 재사용한다
(핵심 코드 미수정, logit_pi 좌표계와 무관하게 동작).

reparam=False 를 주면 원본과 동일한 4성분 좌표계 모형을 만든다 — 스모크에서
"기존 좌표계"와 "합0 고정" 을 나란히 비교하기 위함.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import nuts_seasonpop as NS   # reuse make_joint_loss, N_SEA, LOGIT_REF, SIGMA_PIN — no edits


def build_model_v7(C_all, reparam: bool, with_likelihood: bool = True):
    """reparam=True: logit_pi_free(3) 표집, 4번째 성분 0 고정 (좌표계만 변경).
    reparam=False: 원본과 동일한 logit_pi(4) ~ Uniform(-10,10)^4.
    with_likelihood=False: 데이터 우도 항 없이 사전분포(+ pin)만 (W-0 사전비교용).
    """
    logit_ref_j = jnp.asarray(NS.LOGIT_REF)
    sigma_pin_j = jnp.asarray(NS.SIGMA_PIN)
    loss_fn = NS.make_joint_loss(C_all) if (with_likelihood and C_all is not None) else None

    def model():
        log_R0 = numpyro.sample("log_R0",
            dist.Normal(jnp.log(2.0) * jnp.ones(NS.N_SEA), 0.5))

        if reparam:
            logit_pi_free = numpyro.sample("logit_pi_free",
                dist.Uniform(jnp.full(3, -10.0), jnp.full(3, 10.0)))
            logit_pi = jnp.concatenate([logit_pi_free, jnp.zeros(1)])
            numpyro.deterministic("logit_pi", logit_pi)
        else:
            logit_pi = numpyro.sample("logit_pi",
                dist.Uniform(jnp.full(4, -10.0), jnp.full(4, 10.0)))

        centered = logit_pi - jnp.mean(logit_pi)
        numpyro.factor("pin_prior",
            -0.5 * jnp.sum(((centered - logit_ref_j) / sigma_pin_j) ** 2))
        log_phi_nb = numpyro.sample("log_phi_nb", dist.Normal(jnp.log(10.0), 1.5))
        phi_nb = jnp.exp(log_phi_nb)
        numpyro.deterministic("phi_nb", phi_nb)
        numpyro.deterministic("pi", jax.nn.softmax(logit_pi))

        if loss_fn is not None:
            nll = loss_fn(log_R0, logit_pi, phi_nb)
            numpyro.factor("data_loglik", -nll)

    return model
