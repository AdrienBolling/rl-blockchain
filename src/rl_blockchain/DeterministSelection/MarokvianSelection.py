from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jraph as jr

from rl_blockchain.BlockEnv import BlockchainEnv, EnvParams
from rl_blockchain.scripts.env_factory import LOG_TYPE
import logging

logger = logging.getLogger(__name__)

@jax.jit
def build_adjacency_matrix(graph: jr.GraphsTuple) -> jnp.ndarray:
    nb_nodes = graph.nodes.shape[0]
    adjacency_matrix = jnp.zeros((nb_nodes, nb_nodes))
    adjacency_matrix = adjacency_matrix.at[(graph.senders, graph.receivers)].set(graph.edges.flatten())
    return adjacency_matrix


def no_validator_case(graph: jr.GraphsTuple) -> jax.Array:
    distribution = graph.nodes[:, 1]
    return jnp.argmin(distribution)


def select_center_with_validator(graph: jr.GraphsTuple) -> jax.Array:
    selected_nodes = graph.nodes[:, 0].astype(bool)
    adj_matrix = build_adjacency_matrix(graph)
    non_selected_nodes = jnp.logical_not(selected_nodes)
    mask_rows = non_selected_nodes[:, None]  # (nb_nodes, 1)
    adj_1 = jnp.where(mask_rows, 0.0, adj_matrix)
    mask_cols = non_selected_nodes[None, :]  # (1, nb_nodes)
    adj_2 = jnp.where(mask_cols, 0.0, adj_1)
    sum_adj_2 = jnp.sum(adj_2, axis=1)
    sum_adj_2 = jnp.where(non_selected_nodes, jnp.inf, sum_adj_2)
    center = jnp.argmin(sum_adj_2)
    return center


def compute_center(obs: jr.GraphsTuple) -> jax.Array:
    selected_nodes = obs.nodes[:, 0].astype(bool)
    nb_val = jnp.sum(selected_nodes)
    center = jax.lax.cond(nb_val == 0, no_validator_case, select_center_with_validator, obs)
    return center


gamma = 1000.0  # Paramètre de compensation, peut être ajusté selon les besoins


@jax.jit
def compensated_distance(graph: jr.GraphsTuple) -> jax.Array:
    """
    Calcule la distance compensée pour chaque noeud par rapport au centre.
    """
    distribution = graph.nodes[:, 1]
    selected_nodes = graph.nodes[:, 0].astype(bool)

    mat_adj = build_adjacency_matrix(graph)
    center = compute_center(graph)
    distance_from_center = mat_adj[center, :]  # distances du centre vers tous les noeuds
    compensated_dist = distance_from_center * jnp.exp(distribution / gamma)  # gamma=1.0
    compensated_dist = jnp.where(selected_nodes, jnp.inf, compensated_dist)

    return compensated_dist.argmin() + 1  # +1 for no-op action


def return_change_outer(_: jr.GraphsTuple) -> jax.Array:
    """Return a value for outer step change."""
    return jnp.array(0, dtype=jnp.int32)

def next_mark_action(obs: jr.GraphsTuple) -> jax.Array:
    """Select the next action based on the observation."""
    nb_val_required = obs.globals[0]
    selected_nodes = obs.nodes[:, 0].astype(bool)
    nb_val = jnp.sum(selected_nodes)

    # If the number of validators is equal to the required number, return a no-op action
    action = jax.lax.cond(nb_val == nb_val_required, return_change_outer, compensated_distance, obs)
    return action

@partial(jax.jit, static_argnames=('env', 'steps_in_episode'))
def rollout_markov(key_input, env: BlockchainEnv,
                   env_params: EnvParams, steps_in_episode: int):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    key_reset, key_episode = jax.random.split(key_input)
    first_obs, first_state = env.reset(key_reset, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, key = state_input
        next_key, key_step = jax.random.split(key, 2)


        action = next_mark_action(obs)

        next_obs, next_state, reward, done, infos = env.step(
            key_step, state, action, env_params
        )

        carry = (next_obs, next_state, next_key)
        traj = (obs, action, reward, done, infos)
        return carry, traj

    # Scan over episode step loop
    (obs_end, _, _), trajs = jax.lax.scan(
        policy_step,
        (first_obs, first_state, key_episode),
        None,
        steps_in_episode
    )

    # Return masked sum of rewards accumulated by agent in episode
    observations, actions, rewards, dones, infos = trajs
    return observations, actions, rewards, dones, infos

def eval_markov(env: BlockchainEnv, create_params_fn: Callable[[jax.Array], EnvParams], key:jax.Array, num_episodes: int = 10, log_fn: LOG_TYPE = None):
    @jax.jit
    def single_rollout(rng: jax.Array, new_param: EnvParams):
        return rollout_markov(rng, env, new_param, env.default_params.max_steps_in_episode)

    vm_rollouts = jax.vmap(single_rollout)

    params_map = jax.vmap(
        lambda key_map: create_params_fn(key_map)
    )

    # RNG split
    rollout_key, params_key = jax.random.split(key)
    subkeys = jax.random.split(rollout_key, num_episodes)
    subkeys_params = jax.random.split(params_key, num_episodes)

    params_list = params_map(subkeys_params)
    _, _, rews, dones, infos = vm_rollouts(subkeys, params_list)
    logger.info(f"Evaluated {num_episodes} episodes.")

    metrics = log_fn(infos, rews, dones)

    metrics["avg_returns_episode"] = rews.sum(axis=1).mean().tolist()
    sub_rewards = rews.mean(axis=1).tolist()
    for i, rew in enumerate(sub_rewards):
        metrics[f"reward_{i}"] = rew

    return metrics