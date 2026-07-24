from typing import Callable, Dict

import jax
import jax.numpy as jnp

from rl_blockchain.BlockEnv.BlockchainGraph import gini_coefficient, gini_coefficient_worst
from rl_blockchain.BlockEnv.state_params import EnvState, EnvParams, get_stake_distribution, StaticEnvParams


def _gen_post_filter(inflex_pts: float, inflex_value: float) -> Callable:
    """
    Generate a post-filtering function with the given inflexion points and values.
    """

    @jax.jit
    def post_filter(previous_reward: float) -> float:
        """
        Post-filtering function.
        """
        return jax.lax.cond(
            previous_reward <= inflex_pts,
            lambda r: r * inflex_value / inflex_pts,
            lambda r: 1 + (r - 1) * (1 - inflex_value) / (1 - inflex_pts),
            previous_reward
        )

    return post_filter


_post_filter_gini = _gen_post_filter(0.95 / 1.5, 0.95)  # Default inflexion point and value for Gini reward


def gini_reward(state: EnvState, params: EnvParams) -> tuple[jax.Array, jax.Array]:
    """
    Compute the Gini reward based on the chosen nodes.
    :param state: The current state of the environment.
    :param params: The environment parameters.
    """
    sum_chosen_node_mean = state.ring_history.sum(axis=1).mean()
    nb_nodes = state.ring_history.shape[1]

    stake_distribution = get_stake_distribution(state)
    current_gini = gini_coefficient(stake_distribution)
    worst_gini = gini_coefficient_worst(sum_chosen_node_mean, nb_nodes)

    worst_gini_not_null = jnp.where(worst_gini == 0, 1.0, worst_gini)
    relative_gini = jnp.clip(current_gini / worst_gini_not_null, 0, 1)
    reward = 1.0 - relative_gini
    post_filtered_reward = _post_filter_gini(reward)
    return post_filtered_reward, relative_gini


# @jax.jit
def get_avg_distance(action: jax.Array, state: EnvState, params: EnvParams,
                     static_params: StaticEnvParams) -> jax.Array:
    """
    Compute the average distance of the current state.
    :param action: The list of chosen nodes (1 for chosen, 0 for not chosen).
    :param state: The current state of the environment.
    :param params: The environment parameters.
    :param static_params: The static environment parameters.
    :return: The average distance of the current state.
    """
    edge_mask = action[static_params.speeder.senders] * action[static_params.speeder.receivers]
    distances = state.current_edges_unique[static_params.speeder.inverse]
    total = jnp.sum(edge_mask * distances)
    nb_val = jnp.sum(action)
    denom = (nb_val - 1) * nb_val
    return total / denom


_post_filter_distance = _gen_post_filter(0.5, 0.25)  # Default inflexion point and value for distance reward


def distance_reward(action: jax.Array, state: EnvState, params: EnvParams, static_params: StaticEnvParams) -> tuple[
    jax.Array, jax.Array]:
    # EnvParams
    """
    Make the reward relative to the best and worst value
    :return:
    """

    avg_delay = get_avg_distance(action, state, params, static_params)

    # the gain is the difference between the average delay of selected validators and the average delay of all nodes
    dist_min, dist_max = static_params.distance_opt_array[action.sum().astype(jnp.int32)]
    dist_ref = static_params.avg_distance

    gain = avg_delay - dist_ref
    scale = jnp.where(gain < 0, dist_ref - dist_min, dist_max - dist_ref)
    reward = gain / scale
    reward_clipped = jnp.clip(reward, -1, 1)
    reward_rescaled = (1 - reward_clipped) / 2
    post_filtered_reward = _post_filter_distance(reward_rescaled)
    return post_filtered_reward, avg_delay


def relative_stake_rank_reward(action: jax.Array, state: EnvState, params: EnvParams,
                               static_params: StaticEnvParams) -> jax.Array:
    """Per-step, fully action-attributable fairness reward.

    The windowed ``gini_reward`` is a poor policy-gradient signal: one action
    overwrites a single ring-history row, so its effect on the windowed gini is
    smeared across the next ``horizon`` steps and its GAE advantage is dominated
    by *other* steps' choices (measured: large advantage variance, tiny SNR w.r.t.
    the current action). This reward instead scores *this* action alone: reward is
    high when it selects the nodes that are currently the most under-represented in
    the stake distribution -- exactly the choice that reduces long-run inequality.

    reward = (picked - worst) / (best - worst) in [0, 1], where ``picked`` is the
    total stake *deficit* (mean stake minus node stake) of the chosen nodes, and
    best/worst are the deficits of the top-k / bottom-k nodes. k (=action.sum()) is
    dynamic (ORN-UHL), so top-k/bottom-k are taken via a sorted-cumulative mask
    rather than ``lax.top_k`` (which needs a static k).
    """
    d = get_stake_distribution(state)
    k = action.sum()
    n = d.shape[0]
    sorted_d = jnp.sort(d)
    mask = jnp.arange(n) < k
    min_sum = jnp.where(mask, sorted_d, 0.0).sum()
    max_sum = jnp.where(mask, sorted_d[::-1], 0.0).sum()
    picked_sum = jnp.sum(action * d)
    reward = jnp.clip((max_sum - picked_sum) / (max_sum - min_sum + 1e-8), 0.0, 1.0, )

    return reward


def differential_gini_reward(old_state: EnvState, new_state: EnvState, params: EnvParams) -> jax.Array:
    """Potential-based fairness reward: the per-step DECREASE in windowed relative gini.

    Potential ``Phi(s) = -relative_gini(s)``; the reward is ``Phi(new) - Phi(old) =
    G(old) - G(new)`` (>0 when the action made the committee window fairer). The
    windowed gini is itself a temporal integral over ``horizon`` rounds, so rewarding
    its *level* makes GAE integrate twice (which is why the level reward only learned
    at lambda=0). Differencing turns it back into a proper per-step increment: GAE
    re-integrates it (the discounted sum telescopes to ``G_0 - G_T``), so this
    optimizes the *true* windowed gini rather than a proxy, while being fully
    attributable to the current action -- the only thing that changed between old and
    new. No clip / post-filter: the decomposed critic whitens each component's
    advantage, so the (small, signed) scale is handled automatically.
    """
    g_old = gini_reward(old_state, params)[1]  # relative_gini before the action
    g_new = gini_reward(new_state, params)[1]  # relative_gini after the action
    return g_old - g_new


@jax.jit
def weighted_rewards(action: jax.Array, old_state: EnvState, new_state: EnvState, params: EnvParams,
                     static_params: StaticEnvParams) -> tuple[jax.Array, Dict[str, jax.Array]]:
    # Monitored metric is always the true windowed relative gini (unchanged). The
    # fairness *training* signal is selected by static_params.gini_reward_mode:
    #   "rank"         -> per-step action-attributable stake-rank surrogate (default)
    #   "differential" -> potential-based per-step decrease of the true windowed gini
    _, gini_value = gini_reward(new_state, params)
    if static_params.gini_reward_mode == "differential":
        gini_reward_value = differential_gini_reward(old_state, new_state, params)
    else:
        gini_reward_value = relative_stake_rank_reward(action, new_state, params, static_params)
    distance_reward_value, avg_value = distance_reward(action, new_state, params, static_params)
    weighted_value = jnp.array([gini_reward_value, distance_reward_value]) * params.rewards_weights
    weighted_value_sum = weighted_value.sum()
    return weighted_value_sum, {"gini": gini_value, "gini_reward": gini_reward_value, "distance": avg_value,
                                "distance_reward": distance_reward_value, "weighted_reward": weighted_value_sum}


def null_reward() -> tuple[jax.Array, Dict[str, jax.Array]]:
    return jnp.array(0.0, dtype=jnp.float32), {
        "gini": jnp.array(0, dtype=jnp.float32),
        "gini_reward": jnp.array(0, dtype=jnp.float32),
        "distance": jnp.array(0, dtype=jnp.float32),
        "distance_reward": jnp.array(0, dtype=jnp.float32),
        "weighted_reward": jnp.array(0, dtype=jnp.float32)
    }
