"""Graph partitioning: split a large graph into clusters of size ≤ ``max_size``.

Partitioning is a per-episode *preprocessing* step (the topology is fixed within
an episode), done on the host — like network generation — and its output (an
integer ``labels[N]`` array) feeds the static batching in
:mod:`rl_blockchain.decomp.batching`.

Two strategies behind a common :class:`Partitioner` interface:

* :class:`DistancePartitioner` (**default**) — a balanced, distance-aware split.
  We embed nodes on a 1-D classical-MDS axis (the top eigenvector of the
  double-centered squared-distance matrix), sort along it, and chop into
  ``ceil(N / max_size)`` equal chunks. This yields *exactly* ``ceil(N/max_size)``
  clusters, each ``≤ max_size``, with spatially coherent groupings. Cost is the
  ``O(N³)`` eigendecomposition at init — fine for moderate N; for very large N
  use the router strategy.

* :class:`RouterPartitioner` — groups nodes by their leaf router (free,
  ``O(N)``, network-local), splitting any oversized router group into chunks.
  Scales to large N but clusters are less size-balanced.
"""

from __future__ import annotations

import abc
import dataclasses
import math

import numpy as np
from jax import Array

from rl_blockchain.graph.network import NetworkTopology


@dataclasses.dataclass(frozen=True)
class PartitionContext:
    """Everything a partitioner might need; strategies use the relevant fields."""

    num_nodes: int
    distance_matrix: Array | None = None
    topology: NetworkTopology | None = None


class Partitioner(abc.ABC):
    """Maps a graph to a cluster label per node (each cluster ≤ ``max_size``)."""

    @abc.abstractmethod
    def __call__(
        self, ctx: PartitionContext, max_size: int, key: Array | None = None
    ) -> np.ndarray:
        """Return ``labels`` — an ``int32[N]`` array of cluster ids in ``[0, K)``."""


def _num_clusters(num_nodes: int, max_size: int) -> int:
    return max(1, math.ceil(num_nodes / max_size))


def _chunk_labels(order: np.ndarray, num_clusters: int) -> np.ndarray:
    """Assign consecutive (along ``order``) equal-size chunks to clusters."""
    labels = np.empty(order.shape[0], dtype=np.int32)
    for cluster, members in enumerate(np.array_split(order, num_clusters)):
        labels[members] = cluster
    return labels


class DistancePartitioner(Partitioner):
    """Balanced, distance-aware partition via a 1-D MDS embedding (default)."""

    def __call__(
        self, ctx: PartitionContext, max_size: int, key: Array | None = None
    ) -> np.ndarray:
        if ctx.distance_matrix is None:
            raise ValueError("DistancePartitioner requires ctx.distance_matrix.")
        n = ctx.num_nodes
        k = _num_clusters(n, max_size)
        if k <= 1:
            return np.zeros(n, dtype=np.int32)

        d = np.asarray(ctx.distance_matrix, dtype=np.float64)
        # Guard against unreachable pairs (inf) from non-tree distance matrices.
        if not np.all(np.isfinite(d)):
            finite_max = np.nanmax(np.where(np.isfinite(d), d, np.nan))
            d = np.where(np.isfinite(d), d, 2.0 * finite_max)

        # Classical MDS: B = -1/2 · J D² J ; sort by its leading eigenvector.
        d2 = d**2
        j = np.eye(n) - 1.0 / n
        b = -0.5 * (j @ d2 @ j)
        _, vecs = np.linalg.eigh(b)  # ascending eigenvalues
        order = np.argsort(vecs[:, -1])
        return _chunk_labels(order, k)


class RouterPartitioner(Partitioner):
    """Group nodes by leaf router, chunking oversized groups (scalable, O(N))."""

    def __call__(
        self, ctx: PartitionContext, max_size: int, key: Array | None = None
    ) -> np.ndarray:
        if ctx.topology is None:
            raise ValueError("RouterPartitioner requires ctx.topology.")
        client_router = np.asarray(ctx.topology.client_router)
        labels = np.empty(ctx.num_nodes, dtype=np.int32)
        next_label = 0
        for router in np.unique(client_router):
            members = np.flatnonzero(client_router == router)
            n_chunks = _num_clusters(len(members), max_size)
            for chunk in np.array_split(members, n_chunks):
                labels[chunk] = next_label
                next_label += 1
        return labels


DEFAULT_PARTITIONER: Partitioner = DistancePartitioner()
