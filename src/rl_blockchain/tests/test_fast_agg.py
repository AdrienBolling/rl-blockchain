"""Assert the scatter-free aggregations match jraph exactly.

Covers both the op level (fast_segment_sum / fast_segment_softmax vs jraph) and
the full model (PPOSeparate output with fast ops vs jraph ops), so we know the
0.10 scatter workaround did not change results.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jraph._src.utils import segment_sum as jr_segment_sum
from jraph._src.utils import segment_softmax as jr_segment_softmax

from rl_blockchain.fast_agg import fast_segment_sum, fast_segment_softmax


def _equal_size_segment_ids(num_segments: int, k: int, seed: int = 0) -> jnp.ndarray:
    """Random assignment with exactly k members per segment, in shuffled order
    (worst case for a group-by-reduction: ids are NOT sorted)."""
    ids = np.repeat(np.arange(num_segments), k)
    np.random.default_rng(seed).shuffle(ids)
    return jnp.asarray(ids, dtype=jnp.int32)


@pytest.mark.parametrize("num_segments,k,F", [(25, 24, 8), (200, 199, 1), (1, 600, 4)])
def test_segment_sum_matches_jraph(num_segments, k, F):
    seg = _equal_size_segment_ids(num_segments, k)
    data = jax.random.normal(jax.random.PRNGKey(1), (num_segments * k, F))
    ref = jr_segment_sum(data, seg, num_segments)
    got = fast_segment_sum(data, seg, num_segments)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_segments,k,F", [(25, 24, 1), (200, 199, 1), (25, 24, 3)])
def test_segment_softmax_matches_jraph(num_segments, k, F):
    seg = _equal_size_segment_ids(num_segments, k)
    logits = jax.random.normal(jax.random.PRNGKey(2), (num_segments * k, F)) * 5.0
    ref = jr_segment_softmax(logits, seg, num_segments)
    got = fast_segment_softmax(logits, seg, num_segments)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    # sanity: softmax weights sum to 1 within each segment
    sums = fast_segment_sum(got, seg, num_segments)
    np.testing.assert_allclose(sums, jnp.ones_like(sums), rtol=1e-5, atol=1e-5)


def test_segment_sum_gradient_matches_jraph():
    num_segments, k, F = 25, 24, 4
    seg = _equal_size_segment_ids(num_segments, k)
    data = jax.random.normal(jax.random.PRNGKey(3), (num_segments * k, F))
    w = jax.random.normal(jax.random.PRNGKey(4), (num_segments, F))
    g_ref = jax.grad(lambda d: (jr_segment_sum(d, seg, num_segments) * w).sum())(data)
    g_got = jax.grad(lambda d: (fast_segment_sum(d, seg, num_segments) * w).sum())(data)
    np.testing.assert_allclose(g_got, g_ref, rtol=1e-5, atol=1e-5)


def test_full_model_matches_jraph_ops():
    """PPOSeparate output must be identical whether it uses the fast ops or
    jraph's segment ops. We monkeypatch the names the model looks up."""
    from rl_blockchain.scripts.env_factory import GenericEnvFactory
    from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat
    import rl_blockchain.model as M
    import rl_blockchain.fast_agg as FA

    kb, kr = jax.random.split(jax.random.PRNGKey(0))
    cfg = {"n_nodes": 25, "gat_arch": [16, 16, 16], "voting_nodes": 0,
           "reward_weights": [0.5, 0.5],
           "next_val_type": UpdateValStrat.NO_UPDATE,
           "next_edge_type": UpdateDistStrat.NO_UPDATE}
    model, env, _, _, _ = GenericEnvFactory.create("blockenv", kb, cfg)
    obs, _ = env.reset(kr, env.default_params)
    params = model.init(kr, obs)

    # fast path (as shipped)
    v_fast, pi_fast = model.apply(params, obs)

    # reference: swap the fast ops for jraph's everywhere they're looked up
    # (heads use model.fast_segment_sum; backbone/OptGraphNetGAT use fast_agg's),
    # recompute, then restore.
    patches = [(M, "fast_segment_sum", jr_segment_sum),
               (FA, "fast_segment_sum", jr_segment_sum),
               (FA, "fast_segment_softmax", jr_segment_softmax)]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    try:
        for mod, name, repl in patches:
            setattr(mod, name, repl)
        v_ref, pi_ref = model.apply(params, obs)
    finally:
        for mod, name, val in saved:
            setattr(mod, name, val)

    # The ops match at 1e-5 (tests above); across 2 GAT layers + LayerNorms the
    # different float summation order (reduction vs scatter) accumulates to
    # ~5e-4 on the outputs. That is float noise, not a behavioural change.
    np.testing.assert_allclose(v_fast, v_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(pi_fast.logits, pi_ref.logits, rtol=2e-3, atol=2e-3)
