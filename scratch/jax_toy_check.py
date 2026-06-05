"""Stage M0: JAX/diffrax/numpyro infra check on Intel macOS.

Verifies:
- Imports + x64 enabled
- diffrax ODE forward + JIT
- jax.grad autodiff through ODE
- numpyro NUTS on toy SEIR posterior

Not production code; smoke test only.
"""
from __future__ import annotations
import os
# Don't let JAX hijack BLAS threading
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import diffrax
from diffrax import ODETerm, Tsit5, diffeqsolve, SaveAt, PIDController
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz

print("=" * 70)
print("Phase 1: Install + environment")
print("=" * 70)
print(f"jax:      {jax.__version__}")
print(f"diffrax:  {diffrax.__version__}")
print(f"numpyro:  {numpyro.__version__}")
print(f"arviz:    {arviz.__version__}")
print(f"devices:  {jax.devices()}")
print(f"backend:  {jax.default_backend()}")
print(f"x64:      {jnp.array([1.0]).dtype}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Phase 2: Toy SEIR autodiff")
print("=" * 70)


def toy_seir(t, y, args):
    S, E, I, R = y
    beta, sigma, gamma = args
    N = S + E + I + R
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return jnp.array([dS, dE, dI, dR])


def solve_toy(beta, sigma=0.5, gamma=0.25):
    y0 = jnp.array([9990.0, 5.0, 5.0, 0.0])
    term = ODETerm(toy_seir)
    sol = diffeqsolve(
        term, Tsit5(), t0=0.0, t1=100.0, dt0=0.1,
        y0=y0, args=(beta, sigma, gamma),
        saveat=SaveAt(ts=jnp.linspace(0.0, 100.0, 101)),
        stepsize_controller=PIDController(rtol=1e-4, atol=1e-6),
        max_steps=8192,
    )
    return sol.ys[:, 2]  # I(t)


# Forward sim (warmup)
_ = solve_toy(0.5).block_until_ready()
t0 = time.perf_counter()
I_traj = solve_toy(0.5).block_until_ready()
dt_fwd = time.perf_counter() - t0
print(f"forward sim (uncompiled, post-warmup): {dt_fwd*1000:.1f}ms, "
      f"peak I = {float(I_traj.max()):.1f}")

# JIT
solve_jit = jax.jit(solve_toy)
solve_jit(0.5).block_until_ready()  # warmup compile
t0 = time.perf_counter()
for _ in range(10):
    solve_jit(0.5).block_until_ready()
dt_jit = (time.perf_counter() - t0) / 10
print(f"JIT forward: {dt_jit*1000:.2f}ms/call (10 reps avg)")


def peak_I(beta):
    return solve_toy(beta).max()


# Autodiff gradient
grad_fn = jax.grad(peak_I)
grad_fn(0.5).block_until_ready()  # warmup

t0 = time.perf_counter()
g = float(grad_fn(0.5))
dt_grad = time.perf_counter() - t0
print(f"gradient d(peakI)/d(beta) = {g:.4f}  ({dt_grad*1000:.1f}ms)")

# Finite-diff comparison
eps = 1e-3
fd = (peak_I(0.5 + eps) - peak_I(0.5 - eps)) / (2 * eps)
print(f"finite-diff comparison:    {float(fd):.4f}  (rel diff {abs(g-float(fd))/abs(g)*100:.3f}%)")
print(f"autodiff / finite-diff cost ratio: {dt_grad/dt_fwd:.2f}x "
      f"(vs nominal 34 for finite-diff in 33-dim)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Phase 3: Toy NUTS via numpyro")
print("=" * 70)

# Synthetic data: beta=0.5 + noise
true_beta = 0.5
key = random.PRNGKey(0)
obs_true = solve_toy(true_beta)
obs = obs_true + random.normal(key, obs_true.shape) * 5.0


def model(obs=None):
    beta = numpyro.sample("beta", dist.Uniform(0.1, 1.0))
    I_pred = solve_toy(beta)
    numpyro.sample("y", dist.Normal(I_pred, 5.0), obs=obs)


nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=200, num_samples=200,
            num_chains=2, chain_method="sequential", progress_bar=False)
t0 = time.perf_counter()
mcmc.run(random.PRNGKey(1), obs=obs)
dt_nuts = time.perf_counter() - t0
print(f"NUTS (200 warmup + 200 samples × 2 chains): {dt_nuts:.1f}s")
mcmc.print_summary(prob=0.95)

samples = mcmc.get_samples()
beta_post = samples["beta"]
print(f"\nbeta posterior mean: {float(beta_post.mean()):.4f}  "
      f"(true {true_beta})")
print(f"beta posterior std:  {float(beta_post.std()):.4f}")
print(f"95% CI: [{float(jnp.quantile(beta_post, 0.025)):.4f}, "
      f"{float(jnp.quantile(beta_post, 0.975)):.4f}]")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
checks = {
    "x64 enabled": str(jnp.array([1.0]).dtype) == "float64",
    "diffrax forward sim works": float(I_traj.max()) > 0,
    "autodiff produces finite gradient": jnp.isfinite(g),
    "autodiff matches finite-diff": abs(g - float(fd)) / abs(g) < 0.01,
    "NUTS samples generated": len(beta_post) > 0,
    "beta posterior near true (rel diff <10%)": abs(float(beta_post.mean()) - true_beta) / true_beta < 0.10,
}
for label, ok in checks.items():
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
all_ok = all(checks.values())
print(f"\n  Overall: {'INFRA READY' if all_ok else 'NEEDS REVIEW'}")
