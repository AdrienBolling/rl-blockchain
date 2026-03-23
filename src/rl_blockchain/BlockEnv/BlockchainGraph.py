import json
import pathlib
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from rl_blockchain.BlockEnv.state_params import generate_unique_inverse_senders_receivers, Speeders


@jax.jit
def normalize_max(w: jax.Array) -> jax.Array:
    n = w.shape[0]
    # ignore la diagonale forcée à 0
    w_no_diag = w.at[jnp.diag_indices(n)].set(jnp.nan)
    w_max = jnp.nanmax(w_no_diag)

    w_norm = w / w_max
    w_norm = w_norm.at[jnp.diag_indices(n)].set(0.0)

    return w_norm


@partial(jax.jit, static_argnames=['n'])
def _create_pairwise_arrays(n):
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


@partial(jax.jit, static_argnames=['n_nodes'])
def get_non_diag_indices(n_nodes: int):
    """
    Get the indices of the non-diagonal elements in a flattened adjacency matrix.
    :param n_nodes:
    :return:
    """
    total = n_nodes * (n_nodes - 1)
    ks = jnp.arange(total)
    return jax.vmap(lambda k: get_pair(k, n_nodes))(ks)


@jax.jit
def create_jraph_from_adj_matrix(adj_matrix: jnp.ndarray) -> jraph.GraphsTuple:
    _n_nodes = adj_matrix.shape[0]
    mask = get_non_diag_indices(_n_nodes)
    return create_jraph_from_adj_matrix_fast(adj_matrix, mask)

@jax.jit
def create_empty_jraph(n_nodes: int) -> jraph.GraphsTuple:
    mask = get_non_diag_indices(n_nodes)
    return create_empty_graph_fast(n_nodes, mask)


def create_speeders(_n_nodes: int) -> Tuple[Tuple[jax.Array, jax.Array], Speeders]:
    senders, receivers = _create_pairwise_arrays(_n_nodes)
    unique, inverse = generate_unique_inverse_senders_receivers(senders, receivers, _n_nodes)
    return (senders, receivers), Speeders(unique, inverse)


@jax.jit
def create_unique_from_adj_matrix(adj_matrix: jnp.ndarray) -> jax.Array:
    _n_nodes = adj_matrix.shape[0]
    mask = get_non_diag_indices(_n_nodes)
    _, speeders = create_speeders(_n_nodes)
    return adj_matrix.flatten().take(mask).take(speeders.unique)


class DictOfMask(dict):
    """
    A dictionary to collect masks for different graphs.
    This is useful to avoid recomputing the masks multiple times.
    """

    def __getitem__(self, key):
        assert type(key) is int, "Key must be an integer representing the number of nodes."
        if key not in self:
            super().__setitem__(key, get_non_diag_indices(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        raise Exception("The dictionary is read-only. Use __getitem__ to access the masks.")


STATIC_MASKS_DICT = DictOfMask()


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
