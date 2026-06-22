"""Build a :class:`ClusterGraph` from a partitioned :class:`BlockchainGraph`.

This is the per-step bridge feeding the supervisor: it aggregates current node
features into per-cluster statistics and wires up the coarse inter-cluster
graph. Jitted and whole-array — cluster count ``K`` is small, so this is cheap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from rl_blockchain.graph.blockchain_graph import BlockchainGraph
from rl_blockchain.graph.cluster_graph import ClusterFeatures, ClusterGraph

from .batching import ClusterBatch


def _masked_moments(values: Array, mask: Array) -> tuple[Array, Array]:
    """Mean and std of ``values[K, S]`` over the valid slots given by ``mask``."""
    count = mask.sum(axis=1)
    safe = jnp.maximum(count, 1)
    mean = (values * mask).sum(axis=1) / safe
    var = (((values - mean[:, None]) ** 2) * mask).sum(axis=1) / safe
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


def _masked_pair_moments(dist: Array, pair_mask: Array) -> tuple[Array, Array]:
    """Mean and std of ``dist[K, S, S]`` over the valid pairs in ``pair_mask``."""
    safe_dist = jnp.where(pair_mask, dist, 0.0)
    count = pair_mask.sum(axis=(1, 2))
    safe = jnp.maximum(count, 1)
    mean = safe_dist.sum(axis=(1, 2)) / safe
    var = (((safe_dist - mean[:, None, None]) ** 2) * pair_mask).sum(axis=(1, 2)) / safe
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))


@jax.jit
def build_cluster_graph(
    graph: BlockchainGraph, batch: ClusterBatch, node_distribution: Array
) -> ClusterGraph:
    """Aggregate node-level signals into the coarse cluster graph.

    Args:
        graph: The current blockchain graph (provides ``trust_rating``).
        batch: The static cluster decomposition.
        node_distribution: ``[N]`` per-node selection frequency over the rolling
            horizon (e.g. ``reward_state.choose_history.mean(axis=0)``).

    Returns:
        A :class:`ClusterGraph` with per-cluster features and a fully-connected
        coarse graph (self-loops included) whose edges carry the ``[mean, std]``
        of inter-cluster pairwise distances. Padded clusters are marked invalid
        via ``features.valid``.
    """
    s = batch.max_size
    k = batch.max_clusters

    # Per-node signals gathered into [K, S] slots.
    trust_slots = graph.trust_rating[batch.sub_index]
    distribution_slots = node_distribution[batch.sub_index]
    avg_trust, std_trust = _masked_moments(trust_slots, batch.sub_mask)
    avg_distribution, std_distribution = _masked_moments(
        distribution_slots, batch.sub_mask
    )

    # Intra-cluster distance over valid, off-diagonal pairs.
    pair_mask = (
        batch.sub_mask[:, :, None]
        & batch.sub_mask[:, None, :]
        & ~jnp.eye(s, dtype=bool)[None]
    )
    avg_distance, std_distance = _masked_pair_moments(batch.sub_dist, pair_mask)

    features = ClusterFeatures(
        num_nodes=batch.cluster_size.astype(jnp.float32),
        avg_trust=avg_trust,
        std_trust=std_trust,
        avg_distance=avg_distance,
        std_distance=std_distance,
        avg_distribution=avg_distribution,
        std_distribution=std_distribution,
        valid=batch.cluster_mask,
    )

    # Fully-connected coarse graph (static [K*K] edges, self-loops kept);
    # edge features = [mean, std] of inter-cluster pairwise distances. The GNN
    # masks padded clusters via features.valid.
    idx = jnp.arange(k, dtype=jnp.int32)
    senders = jnp.repeat(idx, k)
    receivers = jnp.tile(idx, k)
    edge_features = jnp.stack(
        (batch.cluster_adj.reshape(-1), batch.cluster_adj_std.reshape(-1)), axis=-1
    )

    return ClusterGraph.new(features, senders, receivers, edge_features)
