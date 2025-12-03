import csv
import os

import jax
import jax.numpy as jnp
import jraph
from flax import struct
from gymnax.environments import environment

from rl_blockchain.BlockEnv.BlockchainGraph import create_jraph_from_adj_matrix, create_rd_adj_matrix, normalize_max

node_features_dict = {
    "node_id": 0,
    "chosen": 1,
    "distrib_chosen": 2,
}

_box_clip = 4


@struct.dataclass
class EnvState(environment.EnvState):
    ring_history: jax.Array
    chosen_nodes: jax.Array  # Nodes chosen in the current round
    inner_step: int
    global_step: int

    @classmethod
    def create_init_state(cls, nb_nodes: int, horizon: int) -> 'EnvState':
        return cls(
            ring_history=jnp.ones((horizon, nb_nodes), dtype=jnp.bool),
            chosen_nodes=jnp.zeros((nb_nodes,), dtype=jnp.bool),
            inner_step=0,
            global_step=0,
            time=0
        )

    @classmethod
    def next_state_inner(cls, previous_state: 'EnvState', new_chosen_node: int) -> 'EnvState':
        new_chosen_nodes_list = previous_state.chosen_nodes.at[new_chosen_node].set(True)
        return cls(
            ring_history=previous_state.ring_history,
            chosen_nodes=new_chosen_nodes_list,
            inner_step=previous_state.inner_step + 1,
            global_step=previous_state.global_step,
            time=previous_state.time + 1
        )

    @classmethod
    def next_state_global(cls, previous_state: 'EnvState') -> 'EnvState':
        horizon, nb_nodes = previous_state.ring_history.shape[0], previous_state.ring_history.shape[1]
        current_index = previous_state.global_step % horizon
        return cls(
            ring_history=previous_state.ring_history.at[current_index, :].set(previous_state.chosen_nodes),
            chosen_nodes=jnp.zeros((nb_nodes,), dtype=jnp.bool),
            inner_step=0,
            global_step=previous_state.global_step + 1,
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
    nb_nodes: int
    distance_opt_array: jax.Array  # Dictionary of optimal distance bounds for each number of validators
    avg_distance: float  # Average distance for the environment, can be the last avg distance
    box_clip = _box_clip  # Clip value for node features

    horizon: int = 200
    node_features = ["distrib_chosen", "chosen"]

    rewards = ["gini", "distance"]

    @classmethod
    def create(cls, nb_nodes: int, filename: str,
               horizon: int = 200) -> 'StaticEnvParams':
        min_max_array = load_min_max_array(filename)
        avg_distance = min_max_array[nb_nodes][0]

        return cls(
            nb_nodes=nb_nodes,
            distance_opt_array=min_max_array,
            avg_distance=avg_distance.item(),
            horizon=horizon
        )


@struct.dataclass
class EnvParams(environment.EnvParams):
    network_graph: jraph.GraphsTuple = None  # Parameters
    adj_matrix: jnp.ndarray = None  # same graph, but in a different struct
    nb_validators: jax.Array = None
    rewards_weights: jax.Array = None  # Weights for the rewards
    max_outer_steps_in_episode: int = 1000  # TODO Set it to -1
    max_steps_in_episode: jax.Array = 1000

    @classmethod
    def create(cls, adj_network_graph: jnp.ndarray, nb_validators: int | jax.Array, key: jax.Array,
               rewards_weights: list | jax.Array = None, max_outer_step: int = 1000,
               max_steps: int | None = None) -> 'EnvParams':
        nb_nodes = adj_network_graph.shape[0]
        if rewards_weights is None:
            rewards_weights = [1, 1]

        rewards_weights_jnp = jnp.array(rewards_weights, dtype=jnp.float32)

        norm_adj_matrix = normalize_max(adj_network_graph)

        if (nb_validators is None) or (nb_validators == 0):
            # val_sample = jax.random.normal(key_nb_val) * (0.25 * nb_nodes) + (nb_nodes // 2)
            # nb_validators = jnp.clip(jnp.round(val_sample), 4, nb_nodes).astype(int)
            nb_validators = jax.random.randint(key, (), minval=4, maxval=nb_nodes)
            max_steps_in_episode = max_outer_step * (
                        jnp.asarray(nb_nodes//2, dtype=jnp.int32) + 1) if max_steps is None else jnp.asarray(max_steps, dtype=jnp.int32)
        else:
            nb_validators = jnp.asarray(nb_validators, dtype=jnp.int32)
            max_steps_in_episode = max_outer_step * (nb_validators + 1) if max_steps is None else jnp.asarray(max_steps, dtype=jnp.int32)

        return cls(
            network_graph=create_jraph_from_adj_matrix(norm_adj_matrix),
            adj_matrix=norm_adj_matrix,
            nb_validators=nb_validators,
            rewards_weights=rewards_weights_jnp / rewards_weights_jnp.sum(),
            max_steps_in_episode=max_steps_in_episode,
            max_outer_steps_in_episode=max_outer_step
        )

    @classmethod
    def create_random(cls, nb_nodes: int, key: jax.Array, nb_validators: int | jax.Array = None,
                      rewards_weights: list | jax.Array = None, max_outer_step: int = 1000,
                      max_steps: int | None = None) -> 'EnvParams':
        """
        Create randomized environment parameters.
        Args:
            nb_nodes (int): Number of nodes in the environment.
            key (jax.Array): JAX random key for reproducibility.
            nb_validators (int, optional): Number of validators. If None, a random number is generated.
            rewards_weights (list or jax.Array, optional): Weights for the rewards. If None, random weights are generated.
            max_outer_step (int, optional): Maximum number of outer steps in an episode. Default is 1000.
            max_steps (int, optional): Maximum number of steps in an episode. If None, it is set to `max_outer_step * (nb_validators + 1)`.
        """
        key_mat, key_rew_weights, key_next = jax.random.split(key, 3)
        adj_mat = create_rd_adj_matrix(nb_nodes, key_mat)
        if rewards_weights is None:
            rewards_weights = jax.random.uniform(key_rew_weights, shape=(2,), minval=0.0, maxval=1.0)
        return cls.create(adj_mat, nb_validators, key_next, rewards_weights, max_outer_step, max_steps)


@jax.jit
def get_stake_distribution(state: EnvState) -> jax.Array:
    nb_nodes = state.chosen_nodes.shape[0]
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
