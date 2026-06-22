"""Static, padded batching of clusters — the bridge between a variable-size
graph and the fixed-shape policy networks.

Given a partition ``labels[N]`` (K clusters, each ≤ ``S = max_size``), we build a
:class:`ClusterBatch`: every cluster becomes a fixed ``S``-slot row, padded and
masked, and clusters themselves are padded up to ``K_max``. The policies then
see constant shapes regardless of ``N`` — which is exactly how the decomposition
absorbs variable graph sizes.

Layout:
* worker view  — ``[K_max, S, ...]`` padded subgraphs (one row per cluster).
* supervisor view — ``[K_max, ...]`` coarse "graph of clusters".

The batch is built **once at episode init** (host-side, like the topology). The
per-step operations (gather features in, scatter the selection back out) are
jitted, whole-array, and GPU-friendly.
"""

from __future__ import annotations

from functools import partial

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@flax.struct.dataclass
class ClusterBatch:
    """Static cluster decomposition of one graph (all fields JAX arrays).

    Shapes use ``K = K_max`` (padded cluster count) and ``S = max_size``.

    Attributes:
        sub_index: ``[K, S]`` global node id occupying each slot (0 where padded;
            always read together with ``sub_mask``).
        sub_mask: ``[K, S]`` bool — True for real nodes, False for padding.
        cluster_mask: ``[K]`` bool — True for real (non-padded) clusters.
        cluster_size: ``[K]`` int — number of real nodes per cluster.
        sub_dist: ``[K, S, S]`` intra-cluster distance sub-matrices.
        cluster_adj: ``[K, K]`` mean pairwise distance between clusters (diagonal
            = mean intra-cluster distance) — the supervisor's coarse graph.
        cluster_adj_std: ``[K, K]`` std of those pairwise distances.
    """

    sub_index: Array
    sub_mask: Array
    cluster_mask: Array
    cluster_size: Array
    sub_dist: Array
    cluster_adj: Array
    cluster_adj_std: Array

    @property
    def max_clusters(self) -> int:
        return self.sub_index.shape[0]

    @property
    def max_size(self) -> int:
        return self.sub_index.shape[1]


def build_cluster_batch(
    labels: np.ndarray,
    distance_matrix: Array,
    max_size: int,
    max_clusters: int | None = None,
) -> ClusterBatch:
    """Assemble a :class:`ClusterBatch` from a partition (host-side, run once).

    Args:
        labels: ``int[N]`` cluster id per node, ids in ``[0, K_actual)``.
        distance_matrix: ``[N, N]`` node distance matrix.
        max_size: Slot count ``S`` per cluster (must be ≥ the largest cluster).
        max_clusters: Padded cluster count ``K_max`` (defaults to ``K_actual``).
    """
    labels = np.asarray(labels)
    k_actual = int(labels.max()) + 1
    k_max = max_clusters if max_clusters is not None else k_actual
    s = max_size

    sub_index = np.zeros((k_max, s), dtype=np.int32)
    sub_mask = np.zeros((k_max, s), dtype=bool)
    cluster_size = np.zeros((k_max,), dtype=np.int32)
    for c in range(k_actual):
        members = np.flatnonzero(labels == c)
        m = members.shape[0]
        if m > s:
            raise ValueError(f"Cluster {c} has {m} nodes > max_size={s}.")
        sub_index[c, :m] = members
        sub_mask[c, :m] = True
        cluster_size[c] = m
    cluster_mask = np.arange(k_max) < k_actual

    # Coarse inter-cluster distance stats via a one-hot membership matrix M[K, N]:
    # sums = M D Mᵀ, normalised by the (size_a · size_b) pair counts. The std
    # comes from E[X²] − E[X]² using the same trick on the squared distances.
    d = np.asarray(distance_matrix, dtype=np.float64)
    membership = np.zeros((k_max, labels.shape[0]), dtype=np.float64)
    membership[labels, np.arange(labels.shape[0])] = 1.0
    sizes = membership.sum(axis=1)
    denom = np.maximum(np.outer(sizes, sizes), 1.0)
    valid_pair = np.outer(sizes, sizes) > 0
    cluster_adj = np.where(valid_pair, (membership @ d @ membership.T) / denom, 0.0)
    mean_sq = np.where(valid_pair, (membership @ (d**2) @ membership.T) / denom, 0.0)
    cluster_adj_std = np.sqrt(np.maximum(mean_sq - cluster_adj**2, 0.0))

    # Intra-cluster distance sub-matrices: sub_dist[k, a, b] = D[idx[k,a], idx[k,b]].
    si = jnp.asarray(sub_index)
    sub_dist = jnp.asarray(distance_matrix)[si[:, :, None], si[:, None, :]]

    return ClusterBatch(
        sub_index=si,
        sub_mask=jnp.asarray(sub_mask),
        cluster_mask=jnp.asarray(cluster_mask),
        cluster_size=jnp.asarray(cluster_size),
        sub_dist=sub_dist,
        cluster_adj=jnp.asarray(cluster_adj, dtype=jnp.float32),
        cluster_adj_std=jnp.asarray(cluster_adj_std, dtype=jnp.float32),
    )


# --------------------------------------------------------------------------- #
# Per-step jitted operations
# --------------------------------------------------------------------------- #
@jax.jit
def gather_subgraph_features(node_features: Array, batch: ClusterBatch) -> Array:
    """Scatter the flat ``[N, F]`` node features into ``[K, S, F]`` subgraphs.

    Padded slots are zeroed so they cannot leak into the worker.
    """
    feats = node_features[batch.sub_index]  # [K, S, F]
    return feats * batch.sub_mask[..., None]


@jax.jit
def aggregate_clusters(node_features: Array, batch: ClusterBatch) -> Array:
    """Per-cluster mean of node features → ``[K, F]`` (supervisor input).

    Padded slots are excluded from the mean; empty clusters give 0.
    """
    feats = gather_subgraph_features(node_features, batch)  # [K, S, F]
    counts = batch.sub_mask.sum(axis=1, keepdims=True)  # [K, 1]
    return feats.sum(axis=1) / jnp.maximum(counts, 1)


@partial(jax.jit, static_argnums=(2,))
def scatter_selection(selection: Array, batch: ClusterBatch, num_nodes: int) -> Array:
    """Map a per-cluster ``[K, S]`` selection back to a global ``chosen[N]`` mask.

    Padding (``~sub_mask``) is dropped; since the partition is exhaustive, each
    real node appears in exactly one slot.
    """
    idx = batch.sub_index.reshape(-1)
    val = (selection & batch.sub_mask).reshape(-1).astype(jnp.int32)
    chosen = jnp.zeros((num_nodes,), dtype=jnp.int32).at[idx].max(val)
    return chosen.astype(jnp.bool_)


@jax.jit
def selection_counts(selection: Array, batch: ClusterBatch) -> Array:
    """Number of selected (real) nodes per cluster → ``[K]`` int."""
    return (selection & batch.sub_mask).sum(axis=1)
