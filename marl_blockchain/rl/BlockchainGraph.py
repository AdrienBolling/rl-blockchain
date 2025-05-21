import jraph
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt

node_features_dict = {
    "node_id": 0,
    "chosen": 1,
    "distrib_chosen": 2,
}


def create_pairwise_arrays(n):
    indices = jnp.arange(n)
    receivers, senders = jnp.meshgrid(indices, indices)
    senders = senders.flatten()
    receivers = receivers.flatten()
    return senders, receivers


@partial(jit, static_argnums=(2,))
def edge_index(i, j, n):
    return i * n + j


@partial(jit, static_argnums=(1,))
def compute_random_mean_distance_from_matrix(node_distance_matrix: jnp.ndarray, num_nodes: int, key):
    """
    Compute the mean distance from a distance matrix.

    Params:
        node_distance_matrix: A matrix of distances between nodes.

    Returns:
        The mean distance.
    """
    # Choose num_nodes random nodes
    random_nodes = jax.random.choice(key, num_nodes, shape=(num_nodes,), replace=False)
    # Compute the mean distance between the random nodes
    d = node_distance_matrix[random_nodes, :][:, random_nodes]
    # Return the mean distance excluding the diagonal
    return jnp.sum(d) / (num_nodes * (num_nodes - 1))


def approximate_min_mean_distance(
        node_distance_matrix: jnp.ndarray,
        n_subnodes: int,
        n_samples: int,
        key,
):
    # generate a vector of random keys of size n_samples
    keys = jax.random.split(key, n_samples)
    # Compute the mean distance for each key
    mean_distances = jax.vmap(
        partial(compute_random_mean_distance_from_matrix, node_distance_matrix, n_subnodes)
    )(keys)
    # Return the minimum mean distance
    return jnp.min(mean_distances)


def create_blockchain_graph(
        node_distance_matrix: jnp.ndarray,
        node_features_size: int,
):
    """
    Create a graph from a distance matrix between nodes.

    Params:
        node_distance_matrix: A matrix of distances between nodes.
        node_features_size: The size of the features of each node.


    Returns:
        A jraph.GraphsTuple object.
    """

    # Create nodes features
    node_features = jnp.zeros((node_distance_matrix.shape[0], node_features_size))

    # Create edges
    senders, receivers = create_pairwise_arrays(node_distance_matrix.shape[0])

    # Create edges features
    edge_features = jnp.expand_dims(node_distance_matrix.flatten(), axis=1)

    # Save informations
    n_nodes = node_distance_matrix.shape[0]
    n_edges = n_nodes ** 2

    # Global features
    global_features = jnp.zeros((1, 1))

    # Create graph
    graph = jraph.GraphsTuple(
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([n_edges]),
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        globals=global_features,
    )

    return graph


@jax.jit
def voting_update(blockchain, voting_node_index):
    """
    Update the blockchain with a voting node.

    Params:
        blockchain: The blockchain to update.
        voting_node_index: The node to vote.

    Returns:
        The updated blockchain.
    """
    # Update the blockchain, at the index of the voting node, the chosen feature is set to 1

    # Get the nodes
    nodes = blockchain.nodes
    # Update the chosen feature of the voting node
    nodes = nodes.at[voting_node_index, node_features_dict["chosen"]].set(1)
    # Update the nodes
    blockchain = blockchain._replace(nodes=nodes)

    return blockchain


@partial(jax.jit, static_argnums=(1,))
def get_feature_all_nodes(blockchain, feature: str):
    """
    Get the feature of all the nodes in the blockchain.

    Params:
        blockchain: The blockchain.
        feature: The feature to get.

    Returns:
        The feature of all the nodes.
    """
    return blockchain.nodes[:, node_features_dict[feature]]


@jax.jit
def partial_reset(blockchain):
    """
    Reset the blockchain.

    Params:
        blockchain: The blockchain to reset.

    Returns:
        The reset blockchain.
    """
    # Compute the new distrib_chosen feature
    previous_distrib_chosen = get_feature_all_nodes(blockchain, "distrib_chosen")
    # Multiply the previous distrib_chosen by number of nodes
    new_distrib_chosen = previous_distrib_chosen * blockchain.nodes.shape[0]
    # Add the chosen feature to the new distrib_chosen
    new_distrib_chosen += get_feature_all_nodes(blockchain, "chosen")
    # Divide by the number of nodes
    new_distrib_chosen /= blockchain.nodes.shape[0]
    # Update the distrib_chosen feature

    blockchain = blockchain._replace(
        nodes=blockchain.nodes.at[
            :, node_features_dict["distrib_chosen"]
        ].set(new_distrib_chosen)
    )

    return blockchain


def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(
                int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
    nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
    pos = nx.spring_layout(nx_graph)
    # Create labels with features
    node_labels = {
        node: f"{node}\n{nx_graph.nodes[node]['node_feature']}"
        for node in nx_graph.nodes
    }
    edge_labels = {
        (sender, receiver): f"{nx_graph.edges[sender, receiver]['edge_feature']}"
        for sender, receiver in nx_graph.edges
    }
    nx.draw(
        nx_graph, pos=pos, with_labels=True, labels=node_labels, node_size=500, font_color='yellow')

    nx.draw_networkx_edge_labels(
        nx_graph, pos=pos, edge_labels=edge_labels, font_color='red')
    nx.draw_networkx_edges(nx_graph, pos=pos)
    # nx.draw_networkx_labels(nx_graph, pos=pos)
    # nx.draw_networkx_nodes(nx_graph, pos=pos)
    plt.show()
