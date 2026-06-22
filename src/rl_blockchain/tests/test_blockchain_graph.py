"""Tests for the BlockchainGraph object and its helper ops."""

import jax
import jax.numpy as jnp
import jraph

from rl_blockchain.graph import (
    BlockchainGraph,
    NodeFeatures,
    add_chosen,
    adjust_trust,
    num_chosen,
    reset_chosen,
    set_chosen,
    set_trust,
    top_k_by_trust,
)


def _graph():
    return BlockchainGraph.new(
        num_nodes=4,
        senders=[0, 1, 2],
        receivers=[1, 2, 3],
        trust_rating=[0.1, 0.9, 0.5, 0.7],
    )


def test_is_jraph_graphstuple():
    g = _graph()
    assert isinstance(g, jraph.GraphsTuple)
    assert type(g) is BlockchainGraph


def test_defaults_and_dtypes():
    g = BlockchainGraph.new(num_nodes=3)
    assert g.num_nodes == 3
    assert g.chosen.dtype == jnp.bool_
    assert g.trust_rating.dtype == jnp.float32
    assert g.nb_chosen.dtype == jnp.int32
    assert not bool(g.chosen.any())
    assert int(g.n_node[0]) == 3 and int(g.n_edge[0]) == 0


def test_pytree_roundtrip_preserves_type():
    g = _graph()
    leaves, treedef = jax.tree_util.tree_flatten(g)
    g2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(g2) is BlockchainGraph
    assert jnp.allclose(g2.trust_rating, g.trust_rating)


def test_set_chosen_bumps_counter():
    g = _graph()
    g = set_chosen(g, jnp.array([True, False, True, False]))
    assert int(num_chosen(g)) == 2
    assert g.nb_chosen.tolist() == [1, 0, 1, 0]
    # Re-selecting bumps the counter again.
    g = set_chosen(g, jnp.array([True, False, True, False]))
    assert g.nb_chosen.tolist() == [2, 0, 2, 0]


def test_add_chosen_unions_without_double_counting():
    g = _graph()
    g = set_chosen(g, jnp.array([True, False, False, False]))
    g = add_chosen(g, jnp.array([True, True, False, False]))
    assert g.chosen.tolist() == [True, True, False, False]
    # Node 0 was already chosen, so it is not double-counted.
    assert g.nb_chosen.tolist() == [1, 1, 0, 0]


def test_reset_chosen_keeps_history():
    g = _graph()
    g = set_chosen(g, jnp.array([True, True, False, False]))
    g = reset_chosen(g)
    assert int(num_chosen(g)) == 0
    assert g.nb_chosen.tolist() == [1, 1, 0, 0]  # history preserved


def test_trust_updates():
    g = _graph()
    g = adjust_trust(g, 0.5)
    assert jnp.allclose(g.trust_rating, jnp.array([0.6, 1.4, 1.0, 1.2]))
    g = set_trust(g, jnp.zeros(4))
    assert jnp.allclose(g.trust_rating, 0.0)


def test_top_k_by_trust_selection():
    g = _graph()
    mask = top_k_by_trust(g, 2)
    # Highest-trust nodes are index 1 (0.9) and 3 (0.7).
    assert mask.tolist() == [False, True, False, True]


def test_jit_through_full_step():
    @jax.jit
    def step(graph):
        return set_chosen(graph, top_k_by_trust(graph, 2))

    g = step(_graph())
    assert type(g) is BlockchainGraph
    assert int(num_chosen(g)) == 2


def test_replace_features_and_as_matrix():
    g = _graph()
    g = g.replace_features(chosen=jnp.array([True, False, False, True]))
    assert g.chosen.tolist() == [True, False, False, True]
    mat = g.features.as_matrix()
    assert mat.shape == (4, 3)


def test_node_features_create_infers_num_nodes():
    feats = NodeFeatures.create(trust_rating=jnp.array([1.0, 2.0]))
    assert feats.num_nodes == 2
    assert feats.chosen.tolist() == [False, False]
