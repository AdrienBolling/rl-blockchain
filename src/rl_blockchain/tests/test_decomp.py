"""Tests for the decomposition substrate: partitioning + static batching."""

import jax
import jax.numpy as jnp
import numpy as np

from rl_blockchain.decomp import (
    DEFAULT_PARTITIONER,
    ClusterBatch,
    DistancePartitioner,
    PartitionContext,
    RouterPartitioner,
    aggregate_clusters,
    build_cluster_batch,
    gather_subgraph_features,
    scatter_selection,
    selection_counts,
)
from rl_blockchain.graph import NetworkConfig, generate_network, set_chosen

MAX_SIZE = 100


def _network(n, seed=0):
    return generate_network(jax.random.PRNGKey(seed), NetworkConfig(num_nodes=n))


# --- partitioning ----------------------------------------------------------- #
def test_distance_partition_sizes_and_coverage():
    g, topo = _network(250)
    ctx = PartitionContext(num_nodes=250, distance_matrix=topo.distance_matrix)
    labels = DEFAULT_PARTITIONER(ctx, MAX_SIZE)
    assert labels.shape == (250,)
    # ceil(250/100) = 3 clusters, each <= 100, every node covered exactly once.
    assert int(labels.max()) + 1 == 3
    sizes = np.bincount(labels)
    assert np.all(sizes <= MAX_SIZE)
    assert sizes.sum() == 250


def test_small_graph_single_cluster():
    g, topo = _network(40)
    ctx = PartitionContext(num_nodes=40, distance_matrix=topo.distance_matrix)
    labels = DistancePartitioner()(ctx, MAX_SIZE)
    assert set(labels.tolist()) == {0}


def test_router_partition_respects_max_size():
    g, topo = _network(300, seed=1)
    ctx = PartitionContext(num_nodes=300, topology=topo)
    labels = RouterPartitioner()(ctx, MAX_SIZE)
    sizes = np.bincount(labels)
    assert np.all(sizes <= MAX_SIZE)
    assert sizes.sum() == 300


def test_distance_partition_is_deterministic():
    _, topo = _network(250)
    ctx = PartitionContext(num_nodes=250, distance_matrix=topo.distance_matrix)
    l1 = DistancePartitioner()(ctx, MAX_SIZE)
    l2 = DistancePartitioner()(ctx, MAX_SIZE)
    assert np.array_equal(l1, l2)


# --- batching --------------------------------------------------------------- #
def _build(n=250, seed=0):
    g, topo = _network(n, seed)
    ctx = PartitionContext(num_nodes=n, distance_matrix=topo.distance_matrix)
    labels = DEFAULT_PARTITIONER(ctx, MAX_SIZE)
    batch = build_cluster_batch(labels, topo.distance_matrix, MAX_SIZE)
    return g, topo, labels, batch


def test_cluster_batch_shapes_and_masks():
    g, topo, labels, batch = _build(250)
    k = int(labels.max()) + 1
    assert isinstance(batch, ClusterBatch)
    assert batch.sub_index.shape == (k, MAX_SIZE)
    assert batch.sub_dist.shape == (k, MAX_SIZE, MAX_SIZE)
    assert batch.cluster_adj.shape == (k, k)
    # mask count matches cluster sizes; total real slots == N
    assert int(batch.sub_mask.sum()) == 250
    assert jnp.all(batch.sub_mask.sum(axis=1) == batch.cluster_size)
    assert bool(batch.cluster_mask.all())  # no padded clusters here


def test_cluster_padding():
    g, topo, labels, _ = _build(250)
    k_actual = int(labels.max()) + 1
    batch = build_cluster_batch(
        labels, topo.distance_matrix, MAX_SIZE, max_clusters=k_actual + 2
    )
    assert batch.sub_index.shape[0] == k_actual + 2
    assert int(batch.cluster_mask.sum()) == k_actual
    assert not bool(batch.cluster_mask[-1])
    assert int(batch.cluster_size[-1]) == 0


def test_scatter_is_inverse_of_gather():
    g, topo, labels, batch = _build(250)
    rng = np.random.default_rng(0)
    chosen = jnp.asarray(rng.random(250) < 0.3)
    # gather the chosen mask into per-cluster slots, then scatter back
    per_cluster = chosen[batch.sub_index] & batch.sub_mask
    recovered = scatter_selection(per_cluster, batch, num_nodes=250)
    assert jnp.array_equal(recovered, chosen)


def test_scatter_selecting_all_recovers_full_graph():
    g, topo, labels, batch = _build(250)
    recovered = scatter_selection(batch.sub_mask, batch, num_nodes=250)
    assert int(recovered.sum()) == 250


def test_selection_counts_matches_budget_check():
    g, topo, labels, batch = _build(250)
    # select the first two valid nodes in every cluster
    sel = jnp.zeros_like(batch.sub_mask).at[:, :2].set(True) & batch.sub_mask
    counts = selection_counts(sel, batch)
    assert jnp.all(counts == jnp.minimum(batch.cluster_size, 2))


def test_gather_and_aggregate_features():
    g, topo, labels, batch = _build(250)
    feats = g.features.as_matrix()  # [N, 3]
    sub = gather_subgraph_features(feats, batch)
    assert sub.shape == (batch.max_clusters, MAX_SIZE, 3)
    # padded slots are zeroed
    assert jnp.allclose(sub[~batch.sub_mask], 0.0)
    agg = aggregate_clusters(feats, batch)
    assert agg.shape == (batch.max_clusters, 3)


def test_pipeline_jits_end_to_end():
    g, topo, labels, batch = _build(250)

    @jax.jit
    def pick_and_apply(graph, batch):
        feats = graph.features.as_matrix()
        sub = gather_subgraph_features(feats, batch)  # [K, S, F]
        # toy "worker": select a node if its trust feature > 0.5
        selection = (sub[..., 1] > 0.5) & batch.sub_mask
        chosen = scatter_selection(selection, batch, num_nodes=250)
        return set_chosen(graph, chosen)

    g2 = pick_and_apply(g, batch)
    assert g2.chosen.shape == (250,)
