"""Tests for the vectorized Plackett-Luce prefix log-prob (logp_prefix_pl V2).

V2 replaced a length-n lax.scan (with a clip for gradient stability) by a
reverse cumulative logsumexp. We check that it (1) computes the true PL value,
(2) has finite gradients on the peaked distributions the clip used to guard
(the whole point of V2), and (3) matches the old clipped scan in the safe
small-k regime (where the clip never fired).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_blockchain.BlockEnv.BlockEnv import logp_prefix_pl

N = 200


def _true_pl_logp(log_p, perm, k):
    """Reference: exact sequential PL prefix log-prob in float64, no clip."""
    lp = np.asarray(log_p, dtype=np.float64)
    pm = np.asarray(perm)
    logW, total = 0.0, 0.0
    for t in range(int(k)):
        logp_t = lp[pm[t]] - logW
        total += logp_t
        logW = logW + np.log1p(-np.exp(logp_t))
    return total


def _old_clipped_scan(log_p, perm, k):
    """The previous implementation, kept here to check small-k compatibility."""
    def step(logW, t):
        def do(_):
            logp_t = log_p[perm[t]] - logW
            diff = jnp.clip(logp_t, max=-1e-6)
            return logW + jnp.log1p(-jnp.exp(diff)), logp_t
        def skip(_):
            return logW, jnp.array(0.0, dtype=log_p.dtype)
        return jax.lax.cond(t < k, do, skip, None)
    _, terms = jax.lax.scan(step, 0.0, jnp.arange(log_p.shape[0]))
    return jnp.sum(terms)


def _logits(seed, scale):
    key = jax.random.PRNGKey(seed)
    return jax.nn.log_softmax(jax.random.normal(key, (N,)) * scale)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("k", [4, 30, 80])
def test_v2_equals_true_pl(scale, k):
    lg = _logits(seed=k, scale=scale)
    perm = jnp.argsort(lg + jax.random.gumbel(jax.random.PRNGKey(k), (N,)))[::-1]
    got = float(logp_prefix_pl(lg, perm, jnp.int32(k)))
    ref = _true_pl_logp(lg, perm, k)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("scale", [2.0, 5.0, 8.0])
@pytest.mark.parametrize("k", [30, N])
def test_v2_gradient_is_finite_on_peaked(scale, k):
    """The reason V2 exists: no clip needed, gradient stays finite even when an
    item dominates the remaining mass."""
    lg = _logits(seed=7, scale=scale)
    perm = jnp.argsort(lg + jax.random.gumbel(jax.random.PRNGKey(1), (N,)))[::-1]
    # grad w.r.t. the raw pre-softmax logits (the real training path)
    g = jax.grad(lambda raw: logp_prefix_pl(jax.nn.log_softmax(raw), perm, jnp.int32(k)))(lg)
    assert bool(jnp.all(jnp.isfinite(g))), "V2 produced non-finite gradients"


@pytest.mark.parametrize("k", [4, 30])
def test_v2_matches_old_clipped_in_safe_regime(k):
    """For moderate distributions and small k the old clip never fires, so V2
    reproduces the previous values (documents backward compatibility there)."""
    lg = _logits(seed=3, scale=1.5)
    perm = jnp.argsort(lg + jax.random.gumbel(jax.random.PRNGKey(2), (N,)))[::-1]
    v2 = float(logp_prefix_pl(lg, perm, jnp.int32(k)))
    old = float(_old_clipped_scan(lg, perm, jnp.int32(k)))
    np.testing.assert_allclose(v2, old, rtol=1e-3, atol=1e-3)
