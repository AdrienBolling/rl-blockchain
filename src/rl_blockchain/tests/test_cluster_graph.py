"""Tests for the ClusterGraph object and its builder."""

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from rl_blockchain.decomp import (
    DEFAULT_PARTITIONER,
    PartitionContext,
    build_cluster_batch,
    build_cluster_graph,
)
from rl_blockchain.graph import (
    ClusterFeatures,
    ClusterGraph,
    NetworkConfig,
    generate_network,
    set_chosen,
)

MAX_SIZE = 100


def _setup(n=250, seed=0, max_clusters=None):
    g, topo = generate_network(jax.random.PRNGKey(seed), NetworkConfig(num_nodes=n))
    ctx = PartitionContext(num_nodes=n, distance_matrix=topo.distance_matrix)
    labels = DEFAULT_PARTITIONER(ctx, MAX_SIZE)
    batch = build_cluster_batch(labels, topo.distance_matrix, MAX_SIZE, max_clusters)
    node_dist = jnp.asarray(np.random.default_rng(0).random(n), dtype=jnp.float32)
    return g, topo, labels, batch, node_dist


def test_is_jraph_graphstuple():
    g, topo, labels, batch, node_dist = _setup()
    cg = build_cluster_graph(g, batch, node_dist)
    assert isinstance(cg, jraph.GraphsTuple)
    assert type(cg) is ClusterGraph
    assert isinstance(cg.features, ClusterFeatures)


def test_cluster_count_and_edges():
    g, topo, labels, batch, node_dist = _setup()
    k = int(labels.max()) + 1
    cg = build_cluster_graph(g, batch, node_dist)
    assert cg.num_clusters == k
    assert int(cg.n_node[0]) == k
    # fully connected with self-loops -> K*K edges
    assert int(cg.n_edge[0]) == k * k
    assert cg.feature_matrix().shape == (k, 7)


def test_num_nodes_feature_matches_cluster_sizes():
    g, topo, labels, batch, node_dist = _setup()
    cg = build_cluster_graph(g, batch, node_dist)
    assert jnp.array_equal(cg.features.num_nodes, batch.cluster_size.astype(jnp.float32))
    assert int(cg.features.num_nodes.sum()) == 250


def test_trust_aggregation_is_correct():
    g, topo, labels, batch, node_dist = _setup()
    cg = build_cluster_graph(g, batch, node_dist)
    trust = np.asarray(g.trust_rating)
    labels_np = np.asarray(labels)
    for c in range(int(labels_np.max()) + 1):
        members = trust[labels_np == c]
        assert np.isclose(float(cg.features.avg_trust[c]), members.mean(), atol=1e-5)
        assert np.isclose(float(cg.features.std_trust[c]), members.std(), atol=1e-5)


def test_distribution_aggregation_is_correct():
    g, topo, labels, batch, node_dist = _setup()
    cg = build_cluster_graph(g, batch, node_dist)
    nd = np.asarray(node_dist)
    labels_np = np.asarray(labels)
    for c in range(int(labels_np.max()) + 1):
        members = nd[labels_np == c]
        assert np.isclose(
            float(cg.features.avg_distribution[c]), members.mean(), atol=1e-5
        )


def test_padded_clusters_marked_invalid():
    g, topo, labels, batch, node_dist = _setup(max_clusters=None)
    k = int(labels.max()) + 1
    _, _, _, padded_batch, _ = _setup(max_clusters=k + 2)
    cg = build_cluster_graph(g, padded_batch, node_dist)
    assert cg.num_clusters == k + 2
    assert bool(cg.valid[:k].all())
    assert not bool(cg.valid[k:].any())
    # padded clusters have zero size / stats
    assert float(cg.features.num_nodes[-1]) == 0.0
    assert float(cg.features.avg_distance[-1]) == 0.0


def test_distances_are_positive_and_finite():
    g, topo, labels, batch, node_dist = _setup()
    cg = build_cluster_graph(g, batch, node_dist)
    assert jnp.all(jnp.isfinite(cg.features.avg_distance))
    assert jnp.all(cg.features.avg_distance >= 0.0)
    assert jnp.all(jnp.isfinite(cg.edges))


def test_edges_have_mean_and_std_features():
    g, topo, labels, batch, node_dist = _setup()
    k = int(labels.max()) + 1
    cg = build_cluster_graph(g, batch, node_dist)
    # [E, 2] = [mean, std] per directed edge.
    assert cg.edges.shape == (k * k, 2)
    assert jnp.all(cg.edges[:, 1] >= 0.0)  # std non-negative


def test_edge_features_match_direct_distance_stats():
    g, topo, labels, batch, node_dist = _setup()
    k = int(labels.max()) + 1
    cg = build_cluster_graph(g, batch, node_dist)
    d = np.asarray(topo.distance_matrix)
    labels_np = np.asarray(labels)
    edges = np.asarray(cg.edges).reshape(k, k, 2)
    # Check two distinct clusters (edge a->b uses all cross pairs).
    a, b = 0, 1
    block = d[np.ix_(labels_np == a, labels_np == b)]
    assert np.isclose(edges[a, b, 0], block.mean(), atol=1e-4)
    assert np.isclose(edges[a, b, 1], block.std(), atol=1e-4)


def test_reacts_to_node_state_changes():
    g, topo, labels, batch, node_dist = _setup()
    # choosing some nodes changes trust? no -> use distribution change instead
    cg1 = build_cluster_graph(g, batch, node_dist)
    cg2 = build_cluster_graph(g, batch, jnp.zeros_like(node_dist))
    assert not jnp.allclose(cg1.features.avg_distribution, cg2.features.avg_distribution)
    assert jnp.allclose(cg2.features.avg_distribution, 0.0)


def test_pytree_roundtrip_preserves_type():
    g, topo, labels, batch, node_dist = _setup()
    cg = build_cluster_graph(g, batch, node_dist)
    leaves, treedef = jax.tree_util.tree_flatten(cg)
    cg2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(cg2) is ClusterGraph


def test_builder_jits():
    g, topo, labels, batch, node_dist = _setup()
    cg = jax.jit(build_cluster_graph)(g, batch, node_dist)
    assert cg.feature_matrix().shape[1] == 7
