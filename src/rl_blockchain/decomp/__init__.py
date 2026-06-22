"""Decomposition-based solving: cluster a large graph and batch the parts.

The pipeline:

1. :func:`partition <rl_blockchain.decomp.partition.Partitioner>` a graph into
   clusters of size ≤ ``max_handling_size``.
2. :func:`build_cluster_batch` into static, padded, masked tensors.
3. Per step: :func:`aggregate_clusters` feeds the supervisor (coarse graph),
   :func:`gather_subgraph_features` feeds the batched worker, and
   :func:`scatter_selection` reassembles the global vote mask.
"""

from .batching import (
    ClusterBatch,
    aggregate_clusters,
    build_cluster_batch,
    gather_subgraph_features,
    scatter_selection,
    selection_counts,
)
from .cluster_graph import build_cluster_graph
from .partition import (
    DEFAULT_PARTITIONER,
    DistancePartitioner,
    PartitionContext,
    Partitioner,
    RouterPartitioner,
)

__all__ = [
    "ClusterBatch",
    "aggregate_clusters",
    "build_cluster_batch",
    "build_cluster_graph",
    "gather_subgraph_features",
    "scatter_selection",
    "selection_counts",
    "DEFAULT_PARTITIONER",
    "DistancePartitioner",
    "PartitionContext",
    "Partitioner",
    "RouterPartitioner",
]
