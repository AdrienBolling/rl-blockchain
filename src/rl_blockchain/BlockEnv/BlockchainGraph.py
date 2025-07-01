import json
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import numpy as np


@partial(jax.jit, static_argnames=['n'])
def _create_pairwise_arrays(n):
    indices = jnp.arange(n)
    receivers, senders = jnp.meshgrid(indices, indices)
    senders = senders.flatten()
    receivers = receivers.flatten()
    return senders, receivers


@partial(jax.jit, static_argnames=['n_nodes'])
def create_rd_adj_matrix(n_nodes: int, key):
    A = jax.random.uniform(key, (n_nodes, n_nodes))
    return jnp.fill_diagonal((A + A.T) * 0.5, 0, inplace=False)


def compute_adjacency_matrix(dict_node: dict) -> np.ndarray:
    coords = np.array([dict_node[k] for k in sorted(dict_node.keys())], dtype=np.float32)
    nb_node = coords.shape[0]
    matrix = np.zeros((nb_node, nb_node), dtype=np.float32)
    for i in range(nb_node):
        for j in range(i + 1, nb_node):
            dist = np.linalg.norm(coords[i] - coords[j])
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix


def import_adj_matrix_from_file(file_path: str) -> jnp.ndarray:
    """
    Import an adjacency matrix from a CSV file.
    :param file_path: Path to the CSV file containing the adjacency matrix.
    :return: A JAX array representing the adjacency matrix.
    """
    with open(file_path, 'r') as f:
        position_map = json.load(f)
        position_map = {int(k): v for k, v in position_map.items()}
        matrix = compute_adjacency_matrix(position_map)
    return jnp.array(matrix, dtype=jnp.float32)


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


@jax.jit
def create_jraph_from_adj_matrix_fast(adj_matrix: jnp.ndarray, non_diag_mask: jnp.ndarray) -> jraph.GraphsTuple:
    senders, receivers = _create_pairwise_arrays(adj_matrix.shape[0])
    senders_no_loop = senders.take(non_diag_mask)
    receivers_no_loop = receivers.take(non_diag_mask)

    edge_features = adj_matrix.flatten().take(non_diag_mask)  # [:, None]

    n_nodes = adj_matrix.shape[0]
    n_edges = n_nodes * (n_nodes - 1)

    graph = jraph.GraphsTuple(
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([n_edges]),
        nodes=None,
        edges=edge_features,
        senders=senders_no_loop,
        receivers=receivers_no_loop,
        globals=None,
    )

    return graph


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
def gini_coefficient_worst(nb_val: float | jax.Array, nb_node: int) -> float:
    """
    Return the worst Gini index obtains with this parameters.
    :param nb_val: Number of validators
    :param nb_node: Total number of nodes
    :return: Worst Gini index
    """
    return 1.0 - nb_val / nb_node
