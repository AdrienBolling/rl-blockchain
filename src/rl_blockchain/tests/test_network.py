"""Tests for fractal network topology generation."""

import jax
import jax.numpy as jnp
import numpy as np

from rl_blockchain.env import RewardParams, init_reward_state
from rl_blockchain.graph import (
    BlockchainGraph,
    NetworkConfig,
    NetworkTopology,
    generate_network,
)


def test_returns_graph_and_topology():
    g, topo = generate_network(jax.random.PRNGKey(0), NetworkConfig(num_nodes=20))
    assert isinstance(g, BlockchainGraph)
    assert isinstance(topo, NetworkTopology)
    assert g.num_nodes == 20
    assert topo.distance_matrix.shape == (20, 20)


def test_tree_structure_is_valid():
    _, topo = generate_network(jax.random.PRNGKey(1), NetworkConfig(num_nodes=50))
    parents = np.asarray(topo.router_parents)
    # Root first, parent precedes child (BFS order), at least one leaf.
    assert parents[0] == -1
    assert np.all(parents[1:] >= 0)
    assert np.all(parents[1:] < np.arange(1, len(parents)))
    assert bool(topo.router_is_leaf.any())
    # Every client is attached to a leaf router.
    leaves = set(np.flatnonzero(np.asarray(topo.router_is_leaf)).tolist())
    assert set(np.asarray(topo.client_router).tolist()) <= leaves


def test_distance_matrix_is_a_valid_metric_shape():
    _, topo = generate_network(jax.random.PRNGKey(2), NetworkConfig(num_nodes=30))
    d = topo.distance_matrix
    # Symmetric, zero diagonal, non-negative, finite (tree is connected).
    assert jnp.allclose(d, d.T)
    assert jnp.allclose(jnp.diag(d), 0.0)
    assert jnp.all(d >= 0.0)
    assert jnp.all(jnp.isfinite(d))


def test_reproducible_for_same_key():
    cfg = NetworkConfig(num_nodes=25)
    _, t1 = generate_network(jax.random.PRNGKey(7), cfg)
    _, t2 = generate_network(jax.random.PRNGKey(7), cfg)
    _, t3 = generate_network(jax.random.PRNGKey(8), cfg)
    assert jnp.allclose(t1.distance_matrix, t2.distance_matrix)
    assert not jnp.allclose(t1.distance_matrix, t3.distance_matrix)


def test_distances_match_latency_formula():
    # Force a fully deterministic tree: root with exactly 2 leaf children,
    # fixed link/access latencies.
    cfg = NetworkConfig(
        num_nodes=6,
        branching=(2,),
        stop_prob=0.0,
        min_children=2,
        router_latency_range=(2.0, 2.0),
        client_latency_range=(0.5, 0.5),
    )
    _, topo = generate_network(jax.random.PRNGKey(3), cfg)
    # 3 routers: root(0) + 2 leaves(1,2); router_dist between leaves = 2 + 2 = 4.
    assert len(topo.router_parents) == 3
    assert jnp.allclose(topo.router_distance_matrix[1, 2], 4.0)

    d = topo.distance_matrix
    cr = np.asarray(topo.client_router)
    n = cfg.num_nodes
    for i in range(n):
        for j in range(n):
            if i == j:
                expected = 0.0
            elif cr[i] == cr[j]:
                expected = 0.5 + 0.5  # same router: just the two access hops
            else:
                expected = 0.5 + 0.5 + 4.0  # plus router-to-router latency
            assert np.isclose(float(d[i, j]), expected)


def test_feeds_reward_state():
    g, topo = generate_network(jax.random.PRNGKey(4), NetworkConfig(num_nodes=40))
    rs = init_reward_state(g, RewardParams(), distance_matrix=topo.distance_matrix)
    assert rs.distance_matrix.shape == (40, 40)
    assert float(rs.good_avg_distance) > 0.0
    assert jnp.allclose(rs.distance_matrix, topo.distance_matrix)
