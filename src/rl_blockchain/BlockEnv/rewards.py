import jax
import jax.numpy as jnp

from rl_blockchain.BlockEnv.BlockchainGraph import gini_coefficient, gini_coefficient_worst
from rl_blockchain.BlockEnv.state_params import EnvState, EnvParams, get_stake_distribution, StaticEnvParams


def gini_reward(state: EnvState, params: EnvParams) -> tuple[jax.Array, jax.Array]:
    """
    Compute the Gini reward based on the chosen nodes.
    :param state: The current state of the environment.
    :param params: The environment parameters.
    """
    sum_chosen_node = jnp.sum(state.chosen_nodes)
    nb_nodes = state.chosen_nodes.shape[0]

    stake_distribution = get_stake_distribution(state)
    current_gini = gini_coefficient(stake_distribution)
    worst_gini = gini_coefficient_worst(sum_chosen_node, nb_nodes)

    worst_gini_not_null = jnp.where(worst_gini == 0, 1.0, worst_gini)
    relative_gini = jnp.clip(current_gini / worst_gini_not_null, 0, 1)
    reward = 1.0 - relative_gini

    return jnp.where(sum_chosen_node == 0, reward, 0.0), relative_gini


# @jax.jit
def get_avg_distance(state: EnvState, params: EnvParams) -> jax.Array:
    """
    Compute the average distance of the current state.
    :param state: The current state of the environment.
    :param params: The environment parameters.
    :return: The average distance of the current state.
    """
    nb_val = jnp.sum(state.chosen_nodes)

    # masque sur les lignes : on garde que les lignes où chosen_nodes == 1
    masked_rows = params.adj_matrix * state.chosen_nodes[:, None]
    # masque sur les colonnes : on garde que les colonnes où chosen_nodes == 1
    submatrix = masked_rows * state.chosen_nodes[None, :]

    total = jnp.sum(submatrix)
    denom = (nb_val - 1) * nb_val
    return total / denom


# @partial(jax.jit, static_argnames=['params'])
# @jax.jit
def distance_reward(state: EnvState, params: EnvParams, static_params: StaticEnvParams) -> tuple[jax.Array, jax.Array]:
    # EnvParams
    """
    Make the reward relative to the best and worst value
    :return:
    """

    avg_delay = get_avg_distance(state, params)

    # the gain is the difference between the average delay of selected validators and the average delay of all nodes
    dist_min, dist_max = static_params.distance_opt_array[params.nb_validators]
    dist_ref = static_params.avg_distance

    gain = avg_delay - dist_ref
    scale = jnp.where(gain < 0, dist_ref - dist_min, dist_max - dist_ref)
    reward = gain / scale
    reward_clipped = jnp.clip(reward, -1, 1)
    reward_rescaled = (1 - reward_clipped) / 2
    return jnp.where(params.nb_validators == static_params.nb_nodes, 0.0, reward_rescaled), avg_delay


@jax.jit
def weighted_rewards(old_state: EnvState, new_state: EnvState, params: EnvParams, static_params: StaticEnvParams) -> (
        jax.Array, dict):
    gini_reward_value, gini_value = gini_reward(new_state, params)
    distance_reward_value, avg_value = distance_reward(old_state, params, static_params)
    weighted_value = jnp.array([gini_reward_value, distance_reward_value]) * params.rewards_weights
    return weighted_value.sum(), {"gini": gini_value, "gini_reward": gini_reward_value, "distance": avg_value,
                                  "distance_reward": distance_reward_value}


def null_reward() -> (jax.Array, dict):
    return jnp.array(0.0, dtype=jnp.float32), {
        "gini": jnp.array(0, dtype=jnp.float32),
        "gini_reward": jnp.array(0, dtype=jnp.float32),
        "distance": jnp.array(0, dtype=jnp.float32),
        "distance_reward": jnp.array(0, dtype=jnp.float32)
    }
