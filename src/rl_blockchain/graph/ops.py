"""Jitted, GPU-friendly helper functions for :class:`BlockchainGraph`.

Design notes
------------
* Every transform is **functional** — it returns a new graph, never mutates,
  which is what ``jax.jit`` and the rest of the JAX ecosystem expect.
* Selection is expressed with **boolean masks** of shape ``[n_nodes]`` rather
  than dynamic index lists. Masks keep all shapes static, so a single compiled
  kernel is reused across steps and the work vectorises cleanly on GPU
  (no gather/scatter with data-dependent sizes).
* The functions accept and return ``BlockchainGraph``; since it is a pytree,
  ``jax.jit`` traces straight through it.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .blockchain_graph import BlockchainGraph
from .features import NB_CHOSEN_DTYPE


@jax.jit
def set_chosen(graph: BlockchainGraph, mask: Array) -> BlockchainGraph:
    """Set the ``chosen`` flag to ``mask`` and bump ``nb_chosen`` accordingly.

    This represents applying one selection action: the nodes where ``mask`` is
    ``True`` become the currently-chosen set, and their lifetime ``nb_chosen``
    counter is incremented by one.

    Args:
        graph: Input graph.
        mask: Boolean ``[n_nodes]`` array — the newly selected nodes.

    Returns:
        Updated graph.
    """
    mask = mask.astype(jnp.bool_)
    nb_chosen = graph.nodes.nb_chosen + mask.astype(NB_CHOSEN_DTYPE)
    return graph.replace_features(chosen=mask, nb_chosen=nb_chosen)


@jax.jit
def add_chosen(graph: BlockchainGraph, mask: Array) -> BlockchainGraph:
    """Like :func:`set_chosen` but *unions* ``mask`` with the current selection.

    Useful when nodes are selected incrementally within a single voting round.
    ``nb_chosen`` is incremented only for nodes that were not already chosen.
    """
    mask = mask.astype(jnp.bool_)
    newly = mask & ~graph.nodes.chosen
    chosen = graph.nodes.chosen | mask
    nb_chosen = graph.nodes.nb_chosen + newly.astype(NB_CHOSEN_DTYPE)
    return graph.replace_features(chosen=chosen, nb_chosen=nb_chosen)


@jax.jit
def reset_chosen(graph: BlockchainGraph) -> BlockchainGraph:
    """Clear the ``chosen`` flag for all nodes (keeps ``nb_chosen`` history)."""
    chosen = jnp.zeros_like(graph.nodes.chosen)
    return graph.replace_features(chosen=chosen)


@jax.jit
def set_trust(graph: BlockchainGraph, trust_rating: Array) -> BlockchainGraph:
    """Overwrite the ``trust_rating`` of every node."""
    return graph.replace_features(
        trust_rating=jnp.asarray(trust_rating, dtype=graph.nodes.trust_rating.dtype)
    )


@jax.jit
def adjust_trust(graph: BlockchainGraph, delta: Array) -> BlockchainGraph:
    """Add ``delta`` (scalar or ``[n_nodes]``) to every node's ``trust_rating``."""
    return graph.replace_features(trust_rating=graph.nodes.trust_rating + delta)


@jax.jit
def num_chosen(graph: BlockchainGraph) -> Array:
    """Number of currently-chosen nodes (scalar int array)."""
    return jnp.sum(graph.nodes.chosen.astype(NB_CHOSEN_DTYPE))


# `k` controls the compiled shape, so it is a static argument.
@partial(jax.jit, static_argnums=(1,))
def top_k_by_trust(graph: BlockchainGraph, k: int) -> Array:
    """Boolean ``[n_nodes]`` mask selecting the ``k`` most-trusted nodes.

    A convenience selection criterion: pair with :func:`set_chosen` to apply it.
    """
    trust = graph.nodes.trust_rating
    # Indices of the top-k trust values; build a mask via a scatter of True.
    top_idx = jax.lax.top_k(trust, k)[1]
    mask = jnp.zeros_like(trust, dtype=jnp.bool_)
    return mask.at[top_idx].set(True)
