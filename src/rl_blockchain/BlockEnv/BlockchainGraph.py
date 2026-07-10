import json
import pathlib
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import jraph
import numpy as np


@jax.jit
def normalize_max(w: jax.Array) -> jax.Array:
    n = w.shape[0]
    # ignore la diagonale forcée à 0
    w_no_diag = w.at[jnp.diag_indices(n)].set(jnp.nan)
    w_max = jnp.nanmax(w_no_diag)

    w_norm = w / w_max
    w_norm = w_norm.at[jnp.diag_indices(n)].set(0.0)

    return w_norm


@lru_cache(maxsize=None)
def _create_pairwise_arrays(n):
    # ensure_compile_time_eval is REQUIRED, not an optimization: without it, the
    # first call -- which may happen inside a jit trace -- would return tracers,
    # and lru_cache would memoize them. Every later call then hands out a tracer
    # from a dead trace (UnexpectedTracerError). Forcing eager evaluation makes
    # the cached value a concrete array, which later traces embed as a constant.
    with jax.ensure_compile_time_eval():
        indices = jnp.arange(n)
        receivers, senders = jnp.meshgrid(indices, indices)
        senders = senders.flatten()
        receivers = receivers.flatten()
        return senders, receivers


@partial(jax.jit, static_argnames=['n_nodes'])
def create_rd_adj_matrix(n_nodes: int, key: jax.Array) -> jax.Array:
    positions = jax.random.uniform(key, (n_nodes, 2))
    diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 2)
    dists = jnp.linalg.norm(diff, axis=-1)
    return normalize_max(dists)


def import_positions_from_file(file_path: pathlib.Path) -> jax.Array:
    """
    Import node positions from a JSON file.
    :param file_path: Path to the JSON file containing node positions.
    :return: A dictionary mapping node indices to their positions.
    """
    with open(file_path, 'r') as f:
        position_map = json.load(f)
        position_map = [np.array(v, dtype=np.float32) for k, v in position_map.items()]

    return jnp.array(position_map)


def make_rd_closed_adj_matrix(list_nodes_position: jax.Array, key: jax.Array, std: float = 0.01) -> jax.Array:
    n_nodes = list_nodes_position.shape[0]
    max_height, max_width = list_nodes_position[:, 0].max(), list_nodes_position[:, 1].max()
    error_positions = jax.random.normal(key, (n_nodes, 2)) * std
    error_positions = error_positions.at[:, 0].set(error_positions[:, 0] * max_width)
    error_positions = error_positions.at[:, 1].set(error_positions[:, 1] * max_height)
    new_postions_map = list_nodes_position + error_positions
    return make_adj_matrix_from_positions(new_postions_map)


def make_adj_matrix_from_positions(list_nodes_position: jax.Array) -> jnp.ndarray:
    """
    Create an adjacency matrix from a dictionary of node positions.
    :param list_nodes_position: A dictionary mapping node indices to their positions.
    :return: A JAX array representing the adjacency matrix.
    """
    matrix = list_nodes_position[:, None, :] - list_nodes_position[None, :, :]  # (N, N, 2)
    matrix = jnp.linalg.norm(matrix, axis=-1)  # (N, N)
    return normalize_max(matrix)


def import_adj_matrix_from_file(file_path: pathlib.Path) -> jnp.ndarray:
    """
    Import an adjacency matrix from a CSV file.
    :param file_path: Path to the CSV file containing the adjacency matrix.
    :return: A JAX array representing the adjacency matrix.
    """
    position_map = import_positions_from_file(file_path)
    return make_adj_matrix_from_positions(position_map)


@jax.jit
def get_pair(k, n_nodes):
    i = k // (n_nodes - 1)
    j = k % (n_nodes - 1)
    j = j + (j >= i)  # décale j pour éviter la diagonale
    return i * n_nodes + j


@lru_cache(maxsize=None)
def get_non_diag_indices(n_nodes: int):
    """
    Get the indices of the non-diagonal elements in a flattened adjacency matrix.
    :param n_nodes:
    :return:
    """
    # See _create_pairwise_arrays: lru_cache + jnp ops must be evaluated eagerly,
    # or a first call from inside a trace memoizes a tracer.
    with jax.ensure_compile_time_eval():
        total = n_nodes * (n_nodes - 1)
        ks = jnp.arange(total)
        return jax.vmap(lambda k: get_pair(k, n_nodes))(ks)


@jax.jit
def create_jraph_from_adj_matrix(adj_matrix: jnp.ndarray) -> jraph.GraphsTuple:
    _n_nodes = adj_matrix.shape[0]
    mask = get_non_diag_indices(_n_nodes)
    return create_jraph_from_adj_matrix_fast(adj_matrix, mask)


def create_empty_jraph(n_nodes: int) -> jraph.GraphsTuple:
    mask = get_non_diag_indices(n_nodes)
    return create_empty_graph_fast(n_nodes, mask)


def create_empty_graph_fast(n_nodes: int, non_diag_mask: jnp.ndarray) -> jraph.GraphsTuple:
    senders, receivers = _create_pairwise_arrays(n_nodes)
    senders_no_loop = senders.take(non_diag_mask)
    receivers_no_loop = receivers.take(non_diag_mask)

    n_edges = n_nodes * (n_nodes - 1)

    graph = jraph.GraphsTuple(
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([n_edges]),
        nodes=None,
        edges=None,
        senders=senders_no_loop,
        receivers=receivers_no_loop,
        globals=None,
    )

    return graph


@jax.jit
def create_jraph_from_adj_matrix_fast(adj_matrix: jnp.ndarray, non_diag_mask: jnp.ndarray) -> jraph.GraphsTuple:
    empty_graph = create_empty_graph_fast(adj_matrix.shape[0], non_diag_mask)
    edge_features = adj_matrix.flatten().take(non_diag_mask)  # [:, None]
    return empty_graph._replace(edges=edge_features)


# ----------------------------------------------------------------------
# Topology (senders / receivers) is derived, not observed
# ----------------------------------------------------------------------
# The observation graph is always the complete directed graph on ``n_nodes``
# without self-loops, so ``senders``/``receivers`` are a pure function of
# ``n_nodes``: identical at every step of every rollout. Only ``edges`` (the
# distances) and ``nodes`` (the stake distribution) move.
#
# Stacking them over a rollout costs 2/3 of the observation buffer -- at
# n_nodes=200 that is 1.78 GB of 2.67 GB for 3 envs x 2000 steps. So a rollout
# can drop them with :func:`strip_topology` and the model puts them back with
# :func:`with_topology`, where they lower to XLA constants. Nothing in between
# (replay buffer, minibatching, the RL algorithm) needs to know about them.

@lru_cache(maxsize=None)
def _topology(n_nodes: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Goes through create_empty_graph_fast so the edge order is by construction the
    # one the environment produced -- edge features are paired with their endpoints
    # by position, so the two must never drift apart.
    # ensure_compile_time_eval: with_topology runs inside the model's trace, and a
    # memoized tracer would poison every later trace. See _create_pairwise_arrays.
    with jax.ensure_compile_time_eval():
        graph = create_empty_graph_fast(n_nodes, get_non_diag_indices(n_nodes))
        return graph.senders, graph.receivers


def strip_topology(graph):
    """Drop the derivable ``senders``/``receivers`` from an observation graph.

    A no-op on non-graph observations, so a rollout may call it unconditionally.
    ``jax.tree`` treats the resulting ``None`` fields as empty subtrees, so the
    stripped graph still indexes, batches and vmaps like any other PyTree.
    """
    if not isinstance(graph, jraph.GraphsTuple):
        return graph
    return graph._replace(senders=None, receivers=None)


def with_topology(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Inverse of :func:`strip_topology`; a no-op when the topology is present.

    Rebuilt through :func:`create_empty_graph_fast`, the same constructor the
    environment used, so the edge order matches ``edges`` element for element.
    ``n_nodes`` comes from the (static) node-axis length, so this works under
    ``jit`` and ``vmap``.

    Raises if the graph is not the complete one this reconstruction assumes: edge
    features are paired with their endpoints *by position*, so a sparse topology
    would otherwise be silently mispaired rather than rejected.
    """
    if graph.senders is not None:
        return graph
    n_nodes = graph.nodes.shape[0]
    n_edges = graph.edges.shape[0]
    if n_edges != n_nodes * (n_nodes - 1):
        raise ValueError(
            f"Cannot derive the topology of a graph with {n_nodes} nodes and "
            f"{n_edges} edges: expected {n_nodes * (n_nodes - 1)} (complete, no "
            f"self-loops). A sparse graph must carry its own senders/receivers."
        )
    senders, receivers = _topology(n_nodes)
    return graph._replace(senders=senders, receivers=receivers)


@jax.jit
def gini_coefficient(tensor: jnp.ndarray) -> jax.Array:
    """
    Compute the Gini coefficient of a 1D JAX array.
    :param tensor: jnp.ndarray of shape (n,)
    :return: scalar float (not DeviceArray)
    """
    sorted_tensor = jnp.sort(tensor)
    cumulative_sum = jnp.cumsum(sorted_tensor)
    n = tensor.shape[0]
    return (n + 1 - 2 * (jnp.sum(cumulative_sum) / cumulative_sum[-1])) / n


@partial(jax.jit, static_argnames=['nb_node'])
def gini_coefficient_worst(nb_val: float | jax.Array, nb_node: int) -> float | jax.Array:
    """
    Return the worst Gini index obtains with this parameters.
    :param nb_val: Number of validators
    :param nb_node: Total number of nodes
    :return: Worst Gini index
    """
    return 1.0 - nb_val / nb_node
