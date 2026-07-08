"""Scatter-free segment aggregations for the fixed complete-graph GNN.

Drop-in replacements for ``jraph``'s ``segment_sum`` / ``segment_softmax`` that
avoid ``lax.scatter_add`` -- which XLA 0.10 lowers to a pathologically slow GPU
kernel (see utils/jax_runtime.py). Instead of scattering, we group the leading
axis by segment id and do a plain axis **reduction**, which XLA lowers to a fast
reduce kernel and is GPU-independent (no XLA flag required).

HARD ASSUMPTION (validated by tests): every segment has the **same number of
members** ``k = data.shape[0] // num_segments``. This holds for the blockchain
env because the graph is the complete directed graph on ``n`` nodes: each node
sends to and receives from exactly ``n-1`` others, and there is no graph padding
(``model.apply`` runs on a single unpadded graph, vmapped over the batch). The
ORN_UHL_UPDATE mode only changes edge *features* (distances), not this topology,
so the assumption keeps holding.

If you ever switch to sparse / padded / variable-degree graphs, revert to
``jraph.segment_sum`` -- these functions would be silently wrong there.
"""

from __future__ import annotations

import jax.numpy as jnp
import jraph as jr


def _grouped(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int):
    """Return (grouped, order) where grouped[i] holds segment i's k members.

    ``order`` is the stable argsort of ``segment_ids`` (original -> grouped);
    kept so callers that need to scatter results back (softmax) can invert it.
    """
    assert num_segments is not None, "num_segments must be static"
    n = data.shape[0]
    assert n % num_segments == 0, (
        f"fast_agg requires equal-size segments: {n} rows not divisible by "
        f"{num_segments} segments. The graph is not the expected complete graph."
    )
    k = n // num_segments
    order = jnp.argsort(segment_ids, stable=True)          # groups equal ids
    grouped = data[order].reshape((num_segments, k) + data.shape[1:])
    return grouped, order


def fast_segment_sum(data, segment_ids, num_segments=None,
                     indices_are_sorted=False, unique_indices=False):
    """Scatter-free equivalent of ``jraph.segment_sum`` (equal-size segments)."""
    grouped, _ = _grouped(data, segment_ids, num_segments)
    return grouped.sum(axis=1)


def fast_segment_softmax(logits, segment_ids, num_segments=None,
                         indices_are_sorted=False, unique_indices=False):
    """Scatter-free equivalent of ``jraph.segment_softmax`` (equal-size segments).

    Numerically identical: subtracts the per-segment max before exp, same as
    jraph. Returns values in the original (ungrouped) edge order.
    """
    grouped, order = _grouped(logits, segment_ids, num_segments)
    maxs = grouped.max(axis=1, keepdims=True)               # matches jraph (no stop_grad)
    ex = jnp.exp(grouped - maxs)
    normalized = ex / ex.sum(axis=1, keepdims=True)
    flat_sorted = normalized.reshape(logits.shape)          # still grouped order
    inv = jnp.argsort(order, stable=True)                   # grouped -> original
    return flat_sorted[inv]


def OptGraphNetGAT(update_edge_fn, update_node_fn, attention_logit_fn,
                   attention_reduce_fn, update_global_fn=None,
                   aggregate_edges_for_nodes_fn=None,
                   aggregate_nodes_for_globals_fn=None,
                   aggregate_edges_for_globals_fn=None):
    """Scatter-free variant of :func:`jraph.GraphNetGAT`.

    Same interface and behaviour as ``jraph.GraphNetGAT``, but routes every
    aggregation through :func:`fast_segment_sum` and the attention normalization
    through :func:`fast_segment_softmax`. We build on ``jraph.GraphNetwork``
    rather than literally calling ``GraphNetGAT`` because the latter hardcodes
    ``segment_softmax`` and does not expose ``attention_normalize_fn``.

    The ``fast_*`` functions are resolved at call time (module attributes) so
    tests can monkeypatch them back to jraph's ops to assert equivalence.
    """
    if attention_logit_fn is None or attention_reduce_fn is None:
        raise ValueError("`attention_logit_fn` and `attention_reduce_fn` are "
                         "required for a Graph Attention network.")
    return jr.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
        attention_logit_fn=attention_logit_fn,
        attention_normalize_fn=fast_segment_softmax,
        attention_reduce_fn=attention_reduce_fn,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn or fast_segment_sum,
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn or fast_segment_sum,
        aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn or fast_segment_sum,
    )
