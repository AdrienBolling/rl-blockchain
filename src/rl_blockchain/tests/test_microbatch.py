"""Assert gradient-accumulation (micro-batching) equals a full-batch update.

update_ppo(..., micro_batch_size=m) must produce the same gradient (hence the
same new params and metrics) as micro_batch_size=None, for any m dividing the
batch. This guarantees the VRAM optimization does not change training.

We use plain SGD (params_new = params - lr*grad) so the parameter delta is
*directly proportional to the gradient* -- comparing params then tests gradient
equivalence. (Adam would normalize near-zero gradients to ~lr and amplify the
~1e-6 float/autotuning noise into a misleading mismatch.)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl_blockchain.scripts.env_factory import GenericEnvFactory
from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat
from rl_blockchain.algo.ppo import create_ppo_state, update_ppo


def _make_batch(n_nodes=25, batch=8, seed=0):
    kb, kr = jax.random.split(jax.random.PRNGKey(seed))
    cfg = {"n_nodes": n_nodes, "gat_arch": [8, 8, 8], "voting_nodes": 0,
           "reward_weights": [0.5, 0.5],
           "next_val_type": UpdateValStrat.NO_UPDATE,
           "next_edge_type": UpdateDistStrat.NO_UPDATE}
    model, env, _, _, _ = GenericEnvFactory.create("blockenv", kb, cfg)
    ppo_state = create_ppo_state(resume_dir=None, env=env, seed=seed, lr=1e-3, model=model)

    # Plain SGD so param delta == -lr * grad (tests gradient equivalence).
    opt = optax.sgd(0.1)
    ppo_state = ppo_state.replace(opt_state=opt.init(ppo_state.params))

    # Distinct observation per sample (so accumulation across micro-batches matters).
    keys = jax.random.split(kr, batch)
    obs_list = [env.reset(k, env.default_params)[0] for k in keys]
    graphs = jax.tree.map(lambda *xs: jnp.stack(xs), *obs_list)

    perms = jax.vmap(lambda k: jax.random.permutation(k, n_nodes))(keys)
    old_lp = jax.random.normal(jax.random.PRNGKey(seed + 1), (batch,))
    returns = jax.random.normal(jax.random.PRNGKey(seed + 2), (batch,))
    adv = jax.random.normal(jax.random.PRNGKey(seed + 3), (batch,))
    old_val = jax.random.normal(jax.random.PRNGKey(seed + 4), (batch,))
    return model, ppo_state, opt, graphs, perms, old_lp, returns, adv, old_val


@pytest.mark.parametrize("micro", [1, 2, 4, 8])
def test_microbatch_matches_full_batch(micro):
    batch = 8
    model, st, opt, graphs, perms, old_lp, returns, adv, old_val = _make_batch(batch=batch)
    args = (st, graphs, perms, old_lp, returns, adv, old_val, model.apply, opt, 0.2)

    full = update_ppo(*args, micro_batch_size=None)
    acc = update_ppo(*args, micro_batch_size=micro)

    # Tolerance note: the accumulation is mathematically exact, but evaluating
    # the GNN unbatched (micro=1) vs batched (full) differs at ~5e-4 in float32
    # through the deep model. That noise SHRINKS with fewer chunks (micro=8 -> 0),
    # so it is float error, not an averaging bug (a wrong 1/n would be O(1) and
    # would not shrink). 2e-3 leaves margin while still catching structural bugs.
    for a, b in zip(jax.tree_util.tree_leaves(full[0].params),
                    jax.tree_util.tree_leaves(acc[0].params)):
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)
    for a, b in zip(full[1:6], acc[1:6]):
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)
