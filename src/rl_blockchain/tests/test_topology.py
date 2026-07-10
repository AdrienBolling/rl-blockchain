"""Assert dropping senders/receivers from stored observations changes nothing.

The rollout stores observations without their topology (it is a pure function of
the node count) and ``PPOBackbone`` rebuilds it. These tests pin the properties
that make that safe:

* the rebuilt topology is *bitwise* the one the environment produced -- edge
  features are positionally paired with their endpoints, so a different edge
  order would silently mispair them;
* the model's output is unchanged whether or not the observation carries its
  topology;
* a graph that is not the complete one this reconstruction assumes is rejected
  loudly rather than mispaired.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_blockchain.BlockEnv.BlockchainGraph import strip_topology, with_topology


def _make(n_nodes=25, seed=0):
    from rl_blockchain.scripts.env_factory import GenericEnvFactory
    from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat

    kb, kr = jax.random.split(jax.random.PRNGKey(seed))
    cfg = {"n_nodes": n_nodes, "gat_arch": [16, 16, 16], "voting_nodes": 0,
           "reward_weights": [0.5, 0.5],
           "next_val_type": UpdateValStrat.NO_UPDATE,
           "next_edge_type": UpdateDistStrat.NO_UPDATE}
    model, env, _, _, _ = GenericEnvFactory.create("blockenv", kb, cfg)
    obs, _ = env.reset(kr, env.default_params)
    return model, env, obs, kr


@pytest.mark.parametrize("n_nodes", [7, 10, 25])
def test_roundtrip_is_bitwise_identical(n_nodes):
    _, _, obs, _ = _make(n_nodes)
    rebuilt = with_topology(strip_topology(obs))

    np.testing.assert_array_equal(rebuilt.senders, obs.senders)
    np.testing.assert_array_equal(rebuilt.receivers, obs.receivers)
    assert rebuilt.senders.dtype == obs.senders.dtype
    # every other field untouched
    np.testing.assert_array_equal(rebuilt.edges, obs.edges)
    np.testing.assert_array_equal(rebuilt.nodes, obs.nodes)


def test_strip_removes_only_the_topology():
    _, _, obs, _ = _make()
    stripped = strip_topology(obs)
    assert stripped.senders is None and stripped.receivers is None
    # None fields are empty PyTree subtrees, so the graph still maps/batches
    assert len(jax.tree.leaves(stripped)) == len(jax.tree.leaves(obs)) - 2
    np.testing.assert_array_equal(stripped.edges, obs.edges)


def test_strip_topology_passes_through_non_graph_obs():
    x = jnp.arange(4.0)
    assert strip_topology(x) is x


def test_with_topology_is_a_noop_when_present():
    _, _, obs, _ = _make()
    assert with_topology(obs) is obs


def test_with_topology_keeps_the_batch_dimension():
    """A batched graph must get batched indices.

    Not cosmetic: XLA lowers the backward scatter of the aggregation ~1.2x slower
    when the indices are unbatched, so `update_ppo` restores the topology once for
    the whole batch rather than per-sample inside the vmap. An unbatched result
    here would still be *correct* -- and silently 20% slower.
    """
    _, _, obs, _ = _make(n_nodes=7)
    batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (4,) + x.shape), strip_topology(obs))

    filled = with_topology(batched)
    assert filled.senders.shape == (4, obs.senders.shape[0])
    assert filled.receivers.shape == (4, obs.receivers.shape[0])
    for i in range(4):
        np.testing.assert_array_equal(filled.senders[i], obs.senders)
        np.testing.assert_array_equal(filled.receivers[i], obs.receivers)

    # the single-graph case must stay unbatched
    assert with_topology(strip_topology(obs)).senders.shape == obs.senders.shape


def test_update_ppo_is_unchanged_by_stripping_the_topology():
    """update_ppo must produce the same new params from a stripped batch as from a
    batch that carries its topology -- it rebuilds it before the per-sample vmap."""
    from rl_blockchain.algo.ppo import create_ppo_state, update_ppo, make_optimizer

    model, env, obs, key = _make(n_nodes=7)
    ppo_state = create_ppo_state(resume_dir=None, env=env, seed=0, lr=1e-3, model=model)
    opt = make_optimizer(0.1)
    ppo_state = ppo_state.replace(opt_state=opt.init(ppo_state.params))

    batch = 4
    full = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), obs)
    # the actor's logits are one per node, so a perm ranks the nodes
    args = (jnp.tile(jnp.arange(obs.nodes.shape[0]), (batch, 1)),   # perms
            jnp.zeros(batch), jnp.ones(batch),                      # old_logps, returns
            jnp.ones(batch), jnp.zeros(batch))                      # advantages, old_values

    out_full = update_ppo(ppo_state, full, *args, model.apply, opt)
    out_strip = update_ppo(ppo_state, strip_topology(full), *args, model.apply, opt)

    assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.allclose(a, b, rtol=1e-6, atol=1e-6)),
                                     out_full[0].params, out_strip[0].params))


def test_with_topology_rejects_an_incomplete_graph():
    """A sparse graph must carry its own topology: rebuilding it would pair the
    edge features with the wrong endpoints, silently."""
    _, _, obs, _ = _make(n_nodes=7)
    sparse = strip_topology(obs)._replace(edges=obs.edges[:10])
    with pytest.raises(ValueError, match="Cannot derive the topology"):
        with_topology(sparse)


def test_model_output_is_identical_with_and_without_topology():
    model, _, obs, key = _make()
    params = model.init(key, obs)

    v_full, pi_full = model.apply(params, obs)
    v_stripped, pi_stripped = model.apply(params, strip_topology(obs))

    np.testing.assert_array_equal(v_stripped, v_full)
    np.testing.assert_array_equal(pi_stripped.logits, pi_full.logits)


def test_model_init_accepts_a_stripped_graph():
    """Same parameter tree either way, so checkpoints stay loadable."""
    model, _, obs, key = _make()
    assert jax.tree.structure(model.init(key, obs)) == \
           jax.tree.structure(model.init(key, strip_topology(obs)))


def test_memoized_helpers_never_cache_a_tracer():
    """The helpers are lru_cache'd AND build jnp arrays. If the first call lands
    inside a jit trace, a naive implementation memoizes a `DynamicJaxprTracer`,
    and every later trace dies with UnexpectedTracerError. `ensure_compile_time_eval`
    is what prevents this -- run in a subprocess so the cache starts cold."""
    import subprocess, sys, textwrap

    prog = textwrap.dedent("""
        import jax, jax.numpy as jnp
        import rl_blockchain.BlockEnv.BlockchainGraph as G

        # Force the very first call of each helper to happen inside a trace.
        jax.jit(lambda m: G.get_non_diag_indices(m.shape[0]).sum())(jnp.zeros((7, 7)))
        jax.jit(lambda m: G._create_pairwise_arrays(m.shape[0])[0].sum())(jnp.zeros((7, 7)))
        jax.jit(lambda m: G._topology(m.shape[0])[0].sum())(jnp.zeros((7, 7)))

        for name, val in [("get_non_diag_indices", G.get_non_diag_indices(7)),
                          ("_create_pairwise_arrays", G._create_pairwise_arrays(7)[0]),
                          ("_topology", G._topology(7)[0])]:
            assert not isinstance(val, jax.core.Tracer), f"{name} memoized a tracer"

        # And the cached values survive an independent, later trace.
        jax.jit(lambda m: G._topology(m.shape[0])[0].sum())(jnp.ones((7, 7)))
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    assert "OK" in out.stdout, f"tracer leaked:\n{out.stdout}\n{out.stderr[-1500:]}"


def test_rollout_does_not_stack_the_topology():
    """The whole point: the stored observation loses its 2 x n_edges int32 leaves."""
    from rl_blockchain.algo.ppo import rollout, create_ppo_state

    model, env, _, key = _make(n_nodes=7)
    ppo_state = create_ppo_state(resume_dir=None, env=env, seed=0, lr=1e-3, model=model)
    steps = 4
    out = jax.eval_shape(lambda k, s: rollout(k, env, model, s, env.default_params, steps),
                         key, ppo_state)
    observations = out[0]

    assert observations.senders is None and observations.receivers is None
    assert observations.edges.shape == (steps, env.nb_nodes * (env.nb_nodes - 1))

    # ... and the model consumes them anyway, per-sample, under vmap.
    graphs = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), observations)
    v, pi = jax.vmap(lambda g: model.apply(ppo_state.params, g))(graphs)
    assert v.shape == (steps,)
