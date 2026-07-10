import csv
import os
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jraph
from flax import struct
from gymnax.environments import environment

from rl_blockchain.BlockEnv.BlockchainGraph import create_rd_adj_matrix, normalize_max, \
    create_empty_jraph, _create_pairwise_arrays, get_non_diag_indices

node_features_dict = {
    "node_id": 0,
    "chosen": 1,
    "distrib_chosen": 2,
}

_box_clip = 4

Next_nb_val_fn = Callable[["EnvState", jax.Array], jax.Array]
Init_nb_val_fn = Callable[[jax.Array], jax.Array]
Next_map_fn = Callable[[jax.Array, "EnvState", "EnvParams"], jax.Array]


@struct.dataclass
class Speeders:
    senders : jax.Array
    receivers : jax.Array
    unique: jax.Array
    inverse: jax.Array

    @classmethod
    def create(cls, nb_nodes: int) -> 'Speeders':
        senders, receivers = _create_pairwise_arrays(nb_nodes)
        mask = get_non_diag_indices(nb_nodes)
        masked_senders = senders[mask]
        masked_receivers = receivers[mask]
        unique, inverse = generate_unique_inverse_senders_receivers(masked_senders, masked_receivers, nb_nodes)
        return cls(masked_senders, masked_receivers, unique, inverse)


@struct.dataclass
class EnvState(environment.EnvState):
    ring_history: jax.Array
    current_edges_unique: jax.Array
    nb_val: jax.Array

    def adj_matrix(self, speeders: Speeders) -> jnp.ndarray:
        # TODO
        return self.current_edges_unique[speeders.inverse]

    @classmethod
    def create_init_state(cls, key: jax.Array, params: "EnvParams", static_params: "StaticEnvParams") -> 'EnvState':
        new_nb_val = static_params.init_nb_val_fn(key)
        return cls(
            ring_history=jnp.ones((static_params.horizon, static_params.nb_nodes), dtype=jnp.bool),
            current_edges_unique=params.edge_config.edges_unique,
            nb_val=new_nb_val,
            time=0
        )

    @classmethod
    def next_state(cls, previous_state: 'EnvState', new_nb_val: jax.Array, new_edges_dist: jax.Array,
                   new_chosen_nodes_list: jax.Array) -> 'EnvState':
        horizon, nb_nodes = previous_state.ring_history.shape[0], previous_state.ring_history.shape[1]
        current_index = previous_state.time % horizon
        return cls(
            ring_history=previous_state.ring_history.at[current_index, :].set(new_chosen_nodes_list),
            nb_val=new_nb_val,
            current_edges_unique=new_edges_dist,
            time=previous_state.time + 1
        )


def load_min_max_array(filename: str) -> jax.Array:
    assert os.path.isfile(filename), f"the provide file {filename} must be the GA solved result"
    dict_opt = {}

    with (open(filename) as file):
        data = csv.reader(file)
        for row in data:
            if data.line_num == 1:
                continue
            nb_nodes = int(row[0])
            dict_opt[nb_nodes] = (float(row[1]), float(row[2]))
    opt_bounds_list = [(0.0, 0.0) for _ in range(nb_nodes + 1)]
    for k, v in dict_opt.items():
        opt_bounds_list[k] = v
    return jnp.array(opt_bounds_list)


@struct.dataclass
class StaticEnvParams:
    empty_network_graph: jraph.GraphsTuple
    speeder: Speeders
    init_nb_val_fn: Init_nb_val_fn = struct.field(pytree_node=False)
    next_nb_val_fn: Next_nb_val_fn = struct.field(pytree_node=False)
    next_map_fn: Next_map_fn = struct.field(pytree_node=False)
    nb_nodes: int
    distance_opt_array: jax.Array  # Dictionary of optimal distance bounds for each number of validators
    avg_distance: float  # Average distance for the environment, can be the last avg distance
    box_clip = _box_clip  # Clip value for node features

    horizon: int = 200
    node_features = ["distrib_chosen", "chosen"]

    rewards = ["gini", "distance"]

    @classmethod
    def create(cls, nb_nodes: int, filename: str,
               init_nb_val_fn: Init_nb_val_fn,
               next_nb_val_fn: Next_nb_val_fn,
               next_map_fn: Next_map_fn,
               horizon: int = 200) -> 'StaticEnvParams':
        min_max_array = load_min_max_array(filename)
        avg_distance = min_max_array[nb_nodes][0]
        return cls(
            empty_network_graph=create_empty_jraph(nb_nodes),
            speeder=Speeders.create(nb_nodes),
            init_nb_val_fn=init_nb_val_fn,
            next_nb_val_fn=next_nb_val_fn,
            next_map_fn=next_map_fn,
            nb_nodes=nb_nodes,
            distance_opt_array=min_max_array,
            avg_distance=avg_distance.item(),
            horizon=horizon
        )


@jax.jit
def white_param_fn(prev_state: EnvState, key: jax.Array) -> jax.Array:
    return prev_state.nb_val


def init_random_nb_val_factory(nb_node: int | jax.Array) -> Init_nb_val_fn:
    @jax.jit
    def init_random_nb_val_fn(key: jax.Array) -> jax.Array:
        return jax.random.randint(key, (), minval=4, maxval=nb_node, dtype=jnp.int32)

    return init_random_nb_val_fn


def init_fixed_nb_val_factory(nb_val: int) -> Init_nb_val_fn:
    @jax.jit
    def init_fixed_nb_val_fn(key: jax.Array) -> jax.Array:
        return jnp.asarray(nb_val, dtype=jnp.int32)

    return init_fixed_nb_val_fn


@partial(jax.jit, static_argnames=["nb_nodes"])
def generate_unique_inverse_senders_receivers(senders: jax.Array, receivers: jax.Array, nb_nodes: int) -> \
        Tuple[jax.Array, jax.Array]:
    i = jnp.minimum(senders, receivers)
    j = jnp.maximum(senders, receivers)
    pair_id = i * nb_nodes + j

    # unique sur pair_id
    _, unique, inverse = jnp.unique(
        pair_id,
        return_inverse=True,
        return_index=True,
        size=nb_nodes * (nb_nodes - 1) // 2,  # nombre de paires non orientées
    )
    return unique, inverse


@struct.dataclass
class EdgesConfig:
    edges_unique: jax.Array
    sigma: Optional[jax.Array]


@struct.dataclass
class EnvParams(environment.EnvParams):
    # network_graph: jraph.GraphsTuple = None  # Parameters # TODO should be removed
    # adj_matrix: jnp.ndarray = None  # same graph, but in a different struct //TODO
    rewards_weights: jax.Array = None  # Weights for the rewards
    edge_config: EdgesConfig = None
    max_steps_in_episode: jax.Array = 1000

    @classmethod
    def create(cls, adj_network_uniq: jnp.ndarray, list_sigma_value: Optional[jax.Array],
               rewards_weights: Optional[list | jax.Array] = None,
               max_steps_in_episode: int | None = 1000) -> 'EnvParams':

        if rewards_weights is None:
            rewards_weights = [1, 1]
        if list_sigma_value is None:
            list_sigma_value = jnp.zeros_like(adj_network_uniq)
        rewards_weights_jnp = jnp.array(rewards_weights, dtype=jnp.float32)

        return cls(
            rewards_weights=rewards_weights_jnp / rewards_weights_jnp.sum(),
            edge_config=EdgesConfig(adj_network_uniq, list_sigma_value),
            max_steps_in_episode=jnp.uint32(max_steps_in_episode),
        )

    @classmethod
    def create_old(cls, adj_network_graph: jnp.ndarray,
                   rewards_weights: list | jax.Array = None, max_steps_in_episode: int | None = 1000) -> 'EnvParams':
        if rewards_weights is None:
            rewards_weights = [1, 1]

        rewards_weights_jnp = jnp.array(rewards_weights, dtype=jnp.float32)

        norm_adj_matrix = normalize_max(adj_network_graph)

        return cls(
            # network_graph=create_jraph_from_adj_matrix(norm_adj_matrix),
            # adj_matrix=norm_adj_matrix,
            rewards_weights=rewards_weights_jnp / rewards_weights_jnp.sum(),
            max_steps_in_episode=jnp.uint32(max_steps_in_episode),
        )

    @classmethod
    def create_random(cls, nb_nodes: int, key: jax.Array, nb_validators: int | jax.Array = None,
                      rewards_weights: list | jax.Array = None,
                      max_steps: int | None = 1000) -> 'EnvParams':
        """
        Create randomized environment parameters.
        Args:
            nb_nodes (int): Number of nodes in the environment.
            key (jax.Array): JAX random key for reproducibility.
            nb_validators (int, optional): Number of validators. If None, a random number is generated.
            rewards_weights (list or jax.Array, optional): Weights for the rewards. If None, random weights are generated.
            init_nb_val_fn (Init_nb_val_fn, optional):
            next_nb_val_fn (Next_nb_val_fn, optional):
            max_steps (int, optional): Maximum number of steps in an episode. If None, it is set to `1000
        """
        key_mat, key_rew_weights = jax.random.split(key, 2)
        adj_mat = create_rd_adj_matrix(nb_nodes, key_mat)
        if rewards_weights is None:
            rewards_weights = jax.random.uniform(key_rew_weights, shape=(2,), minval=0.0, maxval=1.0)
        return cls.create(adj_mat, rewards_weights, max_steps)


@jax.jit
def get_stake_distribution(state: EnvState) -> jax.Array:
    nb_nodes = state.ring_history.shape[1]
    list_nb_val = jnp.sum(state.ring_history, axis=1).astype(jnp.float32)  # shape (rounds,)
    stake = nb_nodes / list_nb_val  # shape (rounds,)
    history_stake = state.ring_history * stake[:, None]  # broadcasting over columns
    stake_distribution = jnp.sum(history_stake, axis=0)  # shape (nodes,)
    return stake_distribution


@jax.jit
def preprocessing_validator_distribution(validator_distribution: jnp.ndarray, box_clip) -> jnp.ndarray:
    std = jnp.std(validator_distribution)
    mean = jnp.mean(validator_distribution)
    normalized = (validator_distribution - mean) / (std + 1e-8)
    clipped = jnp.clip(normalized, -box_clip, box_clip)
    result = jnp.where(std < 1e-7, jnp.zeros_like(validator_distribution), clipped)
    return result
