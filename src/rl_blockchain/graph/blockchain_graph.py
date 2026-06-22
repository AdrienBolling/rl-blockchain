"""``BlockchainGraph`` — a ``jraph.GraphsTuple`` specialised for the voting graph.

A blockchain is modelled as a homogeneous graph where every node carries the
features defined in :class:`~rl_blockchain.graph.features.NodeFeatures`. Because
``jraph.GraphsTuple`` is a ``NamedTuple`` (and therefore a JAX pytree),
subclassing it gives us a type that:

* flows through ``jax.jit`` / ``vmap`` / ``grad`` for free — flatten/unflatten
  rebuilds the subclass via ``type(xs)(*children)``;
* is accepted by every ``jraph`` graph-net utility unchanged;
* exposes ergonomic, named accessors for the node features.

The node features are stored — as a single ``NodeFeatures`` pytree — in the
inherited ``nodes`` field.
"""

from __future__ import annotations

import jax.numpy as jnp
import jraph
from jax import Array

from .features import NodeFeatures


class BlockchainGraph(jraph.GraphsTuple):
    """A ``jraph.GraphsTuple`` whose ``nodes`` field is a ``NodeFeatures``.

    Use :meth:`new` to construct one. The standard ``GraphsTuple`` fields
    (``edges``, ``receivers``, ``senders``, ``globals``, ``n_node``, ``n_edge``)
    keep their usual meaning, so the object remains a drop-in for ``jraph``.
    """

    __slots__ = ()

    @classmethod
    def new(
        cls,
        num_nodes: int,
        senders: Array | None = None,
        receivers: Array | None = None,
        *,
        trust_rating: Array | None = None,
        chosen: Array | None = None,
        nb_chosen: Array | None = None,
        globals_: Array | None = None,
    ) -> "BlockchainGraph":
        """Create a single blockchain graph.

        Args:
            num_nodes: Number of nodes in the graph (static — drives shapes).
            senders: ``[n_edges]`` int array of edge source indices. Defaults to
                an empty edge set.
            receivers: ``[n_edges]`` int array of edge destination indices.
            trust_rating: Optional ``[num_nodes]`` float feature; defaults to 0.
            chosen: Optional ``[num_nodes]`` bool feature; defaults to ``False``.
            nb_chosen: Optional ``[num_nodes]`` int feature; defaults to 0.
            globals_: Optional graph-level features ``[1, ...]``.

        Returns:
            A :class:`BlockchainGraph`.
        """
        if (senders is None) != (receivers is None):
            raise ValueError("`senders` and `receivers` must be provided together.")
        if senders is None:
            senders = jnp.zeros((0,), dtype=jnp.int32)
            receivers = jnp.zeros((0,), dtype=jnp.int32)
        else:
            senders = jnp.asarray(senders, dtype=jnp.int32)
            receivers = jnp.asarray(receivers, dtype=jnp.int32)

        features = NodeFeatures.create(
            chosen=chosen,
            trust_rating=trust_rating,
            nb_chosen=nb_chosen,
            num_nodes=num_nodes,
        )

        return cls(
            nodes=features,
            edges=None,
            senders=senders,
            receivers=receivers,
            globals=globals_,
            n_node=jnp.asarray([num_nodes], dtype=jnp.int32),
            n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
        )

    # -- Ergonomic, named feature accessors -------------------------------- #

    @property
    def features(self) -> NodeFeatures:
        """The ``NodeFeatures`` pytree stored in ``nodes``."""
        return self.nodes

    @property
    def chosen(self) -> Array:
        return self.nodes.chosen

    @property
    def trust_rating(self) -> Array:
        return self.nodes.trust_rating

    @property
    def nb_chosen(self) -> Array:
        return self.nodes.nb_chosen

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (static)."""
        return self.nodes.num_nodes

    def replace_features(self, **updates: Array) -> "BlockchainGraph":
        """Return a copy with some node features replaced (functional update).

        Example: ``graph.replace_features(chosen=new_mask)``.
        """
        return self._replace(nodes=self.nodes.replace(**updates))
