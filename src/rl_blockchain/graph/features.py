"""Node feature container for :class:`BlockchainGraph`.

The features live in the ``nodes`` field of the underlying ``jraph.GraphsTuple``.
We keep them in a single ``flax.struct.dataclass`` so that:

* the whole bundle is a registered JAX pytree (works transparently with
  ``jax.jit`` / ``vmap`` / ``grad`` and with every ``jraph`` utility, since
  ``jraph`` treats ``nodes`` as an arbitrary ``ArrayTree``);
* features are accessed by name (``graph.nodes.trust_rating``) instead of by a
  fragile column index;
* immutable, functional updates are a one-liner via ``.replace(...)``.

All arrays are batched along the leading axis ``[n_nodes]`` exactly like the
``nodes`` array of a plain ``jraph.GraphsTuple``.
"""

from __future__ import annotations

import flax.struct
import jax.numpy as jnp
from jax import Array

# Canonical dtypes for the node features. Centralised so every constructor /
# helper stays consistent (important for `jit` cache hits and GPU kernels).
CHOSEN_DTYPE = jnp.bool_
TRUST_DTYPE = jnp.float32
NB_CHOSEN_DTYPE = jnp.int32


@flax.struct.dataclass
class NodeFeatures:
    """Per-node features of a homogeneous blockchain graph.

    Attributes:
        chosen: Boolean flag ``[n_nodes]``. ``True`` if the node was selected by
            the most recent action.
        trust_rating: Scalar float ``[n_nodes]``. How much the network trusts the
            node.
        nb_chosen: Integer counter ``[n_nodes]``. How many times the node has
            been chosen over the whole history.
    """

    chosen: Array
    trust_rating: Array
    nb_chosen: Array

    @classmethod
    def create(
        cls,
        *,
        chosen: Array | None = None,
        trust_rating: Array | None = None,
        nb_chosen: Array | None = None,
        num_nodes: int | None = None,
    ) -> "NodeFeatures":
        """Build a :class:`NodeFeatures`, filling in sensible defaults.

        Any feature left as ``None`` is initialised to a zero array of length
        ``num_nodes`` (which is otherwise inferred from a provided feature).
        Provided arrays are cast to the canonical dtypes.
        """
        if num_nodes is None:
            for feat in (trust_rating, chosen, nb_chosen):
                if feat is not None:
                    num_nodes = jnp.asarray(feat).shape[0]
                    break
            else:
                raise ValueError(
                    "Provide `num_nodes` or at least one feature array to infer it."
                )

        chosen = (
            jnp.zeros((num_nodes,), dtype=CHOSEN_DTYPE)
            if chosen is None
            else jnp.asarray(chosen, dtype=CHOSEN_DTYPE)
        )
        trust_rating = (
            jnp.zeros((num_nodes,), dtype=TRUST_DTYPE)
            if trust_rating is None
            else jnp.asarray(trust_rating, dtype=TRUST_DTYPE)
        )
        nb_chosen = (
            jnp.zeros((num_nodes,), dtype=NB_CHOSEN_DTYPE)
            if nb_chosen is None
            else jnp.asarray(nb_chosen, dtype=NB_CHOSEN_DTYPE)
        )
        return cls(chosen=chosen, trust_rating=trust_rating, nb_chosen=nb_chosen)

    @property
    def num_nodes(self) -> int:
        """Number of nodes (static — read from the leading axis)."""
        return self.trust_rating.shape[0]

    def as_matrix(self) -> Array:
        """Stack the features into a dense ``[n_nodes, 3]`` float matrix.

        Handy as a feed for downstream Flax modules / GNN layers.
        """
        return jnp.stack(
            (
                self.chosen.astype(TRUST_DTYPE),
                self.trust_rating,
                self.nb_chosen.astype(TRUST_DTYPE),
            ),
            axis=-1,
        )
