"""``ClusterGraph`` — the coarse "graph of clusters" the supervisor reasons over.

After a large :class:`BlockchainGraph` is partitioned into clusters, each cluster
becomes a *single node* of a much smaller graph. The supervisor policy runs on
this ``ClusterGraph`` to allocate a voting budget per cluster.

Like :class:`BlockchainGraph`, this subclasses ``jraph.GraphsTuple`` (so it is a
pytree and works with every ``jraph`` utility) and stores a structured
:class:`ClusterFeatures` pytree in its ``nodes`` field. Edges carry the coarse
inter-cluster distance, so message passing can use network locality.

Note on naming: ``ClusterFeatures.num_nodes`` is the number of *blockchain*
nodes inside a cluster — a per-cluster feature. The number of nodes of the
``ClusterGraph`` itself (in the ``jraph`` sense) is the number of clusters,
available as :attr:`ClusterGraph.num_clusters`.
"""

from __future__ import annotations

import flax.struct
import jax.numpy as jnp
import jraph
from jax import Array

_FLOAT = jnp.float32


@flax.struct.dataclass
class ClusterFeatures:
    """Per-cluster features (each array batched along ``[K]`` = #clusters).

    Attributes:
        num_nodes: Number of blockchain nodes in the cluster.
        avg_trust / std_trust: Mean / std of node ``trust_rating`` in the cluster.
        avg_distance / std_distance: Mean / std of intra-cluster pairwise distance
            (compactness of the cluster).
        avg_distribution / std_distribution: Mean / std of the per-node selection
            frequency over the rolling horizon (the cluster's fairness state).
        valid: Whether this is a real cluster (``False`` for padded slots).
    """

    num_nodes: Array
    avg_trust: Array
    std_trust: Array
    avg_distance: Array
    std_distance: Array
    avg_distribution: Array
    std_distribution: Array
    valid: Array

    @property
    def num_clusters(self) -> int:
        return self.valid.shape[0]

    def as_matrix(self) -> Array:
        """Stack the numeric features into ``[K, 7]`` (excludes the ``valid`` mask).

        Column order: num_nodes, avg_trust, std_trust, avg_distance,
        std_distance, avg_distribution, std_distribution.
        """
        return jnp.stack(
            (
                self.num_nodes.astype(_FLOAT),
                self.avg_trust,
                self.std_trust,
                self.avg_distance,
                self.std_distance,
                self.avg_distribution,
                self.std_distribution,
            ),
            axis=-1,
        )


class ClusterGraph(jraph.GraphsTuple):
    """A ``jraph.GraphsTuple`` whose nodes are clusters (``ClusterFeatures``)."""

    __slots__ = ()

    @classmethod
    def new(
        cls,
        features: ClusterFeatures,
        senders: Array,
        receivers: Array,
        edge_features: Array | None = None,
        globals_: Array | None = None,
    ) -> "ClusterGraph":
        """Build a cluster graph.

        Args:
            features: Per-cluster :class:`ClusterFeatures` (length ``K``).
            senders / receivers: ``[E]`` int edge endpoints (cluster indices).
            edge_features: Optional ``[E, 2]`` per-edge features — the mean and
                std of the pairwise distances between the two clusters' nodes.
            globals_: Optional graph-level features.
        """
        k = features.valid.shape[0]
        senders = jnp.asarray(senders, dtype=jnp.int32)
        receivers = jnp.asarray(receivers, dtype=jnp.int32)
        edges = None if edge_features is None else jnp.asarray(edge_features, dtype=_FLOAT)
        return cls(
            nodes=features,
            edges=edges,
            senders=senders,
            receivers=receivers,
            globals=globals_,
            n_node=jnp.asarray([k], dtype=jnp.int32),
            n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
        )

    @property
    def features(self) -> ClusterFeatures:
        return self.nodes

    @property
    def valid(self) -> Array:
        """Boolean ``[K]`` mask of real (non-padded) clusters."""
        return self.nodes.valid

    @property
    def num_clusters(self) -> int:
        """Number of cluster slots ``K`` (including padding)."""
        return self.nodes.num_clusters

    def feature_matrix(self) -> Array:
        """Dense ``[K, 7]`` feature matrix for the supervisor network."""
        return self.nodes.as_matrix()
