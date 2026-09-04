from typing import Callable, Dict

import jax
import jax.numpy as jnp

from rl_blockchain.BlockEnv.BlockchainGraph import gini_coefficient, gini_coefficient_worst
from rl_blockchain.BlockEnv.state_params import EnvState, EnvParams, get_stake_distribution, StaticEnvParams, \
    _get_stake_distribution_ring_history


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
    return _gini_reward_ring_history(state.ring_history)


def _gini_reward_ring_history(ring_history: jax.Array) -> tuple[jax.Array, jax.Array]:
    sum_chosen_node_mean = ring_history.sum(axis=1).mean()
    nb_nodes = ring_history.shape[1]

    stake_distribution = _get_stake_distribution_ring_history(ring_history)
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


def _base_stake_distribution(action: jax.Array, state: EnvState) -> tuple[jax.Array, jax.Array]:
    """Sliding-window stake distribution of ``state`` minus this action's own ring row.

    ``next_state`` *overwrites* row ``time % horizon``, so ``d_new != d_old + increment``.
    Subtracting the action's row from the post-step distribution gives the
    action-independent base; then ``dist(next_state(..., S)) == base + increment * 1_S``
    for any ``S`` of the same cardinality. (Zeroing the row instead divides by 0.)

    :param state: the state *after* ``action`` was applied.
    :return: ``(base, increment)``, ``increment = n / k`` being the stake a selected node gains.
    """
    d_new = get_stake_distribution(state)
    n = d_new.shape[0]
    k = action.sum()
    increment = n / jnp.maximum(k, 1.0)
    return d_new - increment * action.astype(d_new.dtype), increment


def relative_gini_rank_reward(
    action: jax.Array,
    state: EnvState,
    params: EnvParams,
    static_params: StaticEnvParams,
) -> jax.Array:
    """Normalized one-step Gini rank reward, ``(G_worst - G_action) / (G_worst - G_best)``.

    Same normalisation as :func:`relative_stake_rank_reward` but ranking the gini the
    action actually produces on the window instead of a stake-deficit proxy, so the score
    is rank-weighted rather than a plain stake sum.

    Bottom-k / top-k are the *exact* argmin / argmax, not a heuristic: swapping an
    unselected ``i`` for a selected ``j`` with ``b_i <= b_j`` maps the pair
    ``{b_i, b_j + c}`` to ``{b_i + c, b_j}``, same sum and spread ``|c - (b_j - b_i)|
    <= c + (b_j - b_i)`` -- a Robin Hood transfer, and gini is Schur-convex. This holds
    only because every selected node gains the *same* ``c = n / k``. So the reward is a
    true [0, 1] normalisation and the clip only absorbs float32 noise on ties.

    ``k`` is dynamic, hence the sorted mask over ``lax.top_k``.

    :param state: the state *after* ``action`` was applied.
    """
    base, increment = _base_stake_distribution(action, state)
    n = base.shape[0]
    k = action.sum()

    sorted_idx = jnp.argsort(base)
    mask = (jnp.arange(n) < k).astype(base.dtype)
    best_action = jnp.zeros(n, base.dtype).at[sorted_idx].set(mask)
    worst_action = jnp.zeros(n, base.dtype).at[sorted_idx[::-1]].set(mask)

    gini_action = gini_coefficient(base + increment * action.astype(base.dtype))
    gini_best = gini_coefficient(base + increment * best_action)
    gini_worst = gini_coefficient(base + increment * worst_action)

    # Degenerate when every k-subset is the same action (k == 0 or k == n): stay neutral
    # instead of letting the epsilon decide.
    spread = gini_worst - gini_best
    reward = jnp.where(spread > 1e-6, (gini_worst - gini_action) / spread, 0.5)

    return jnp.clip(reward, 0.0, 1.0)


def _relative_gini_of_row(row: jax.Array, ring_float: jax.Array, current_index: jax.Array) -> jax.Array:
    """Windowed ``relative_gini`` after writing ``row`` into a FLOAT ring buffer.

    The stored ``ring_history`` is boolean; ``jax.grad`` of ``new_gini_relative`` would
    be identically zero because the float->bool ``.set`` cast has a zero JVP. Operating
    on ``ring_float`` (a float copy) keeps the sort/cumsum path differentiable so the
    marginal effect of each node flows through.
    """
    new_ring = ring_float.at[current_index, :].set(row)
    return _gini_reward_ring_history(new_ring)[1]  # relative_gini scalar


# grad w.r.t. the row (argnums=0) built ONCE at import: it is a trace-time transform,
# not a jit cache entry, so this avoids re-creating the transformed function per call.
_relative_gini_grad = jax.grad(_relative_gini_of_row, argnums=0)


def gini_grad_reward(action: jax.Array, previous_state: EnvState) -> jax.Array:
    """Autodiff fairness reward: ``jax.grad`` of the new windowed relative gini w.r.t. the action.

    ``g_j = d relative_gini / d action_j`` is the marginal effect of including node ``j``
    on the windowed inequality (see :func:`_relative_gini_of_row` for the float relaxation
    that makes this non-zero). Lower gini is better, so the reward is ``-<action, g>``:
    positive when the chosen nodes are the ones whose inclusion *decreases* inequality
    (a smooth, first-order cousin of ``relative_stake_rank_reward``). Fully attributable
    to this action; no clip / post-filter (the decomposed critic whitens the advantage).
    """
    current_index = previous_state.time % previous_state.ring_history.shape[0]
    ring_float = previous_state.ring_history.astype(jnp.float32)
    grad = _relative_gini_grad(action.astype(jnp.float32), ring_float, current_index)
    return -jnp.sum(action * grad)


@jax.jit
def weighted_rewards(action: jax.Array, old_state: EnvState, new_state: EnvState, params: EnvParams,
                     static_params: StaticEnvParams) -> tuple[jax.Array, Dict[str, jax.Array]]:
    """Monitored metric is always the true windowed relative gini (unchanged). The
    fairness *training* signal is selected by static_params.gini_reward_mode:
      "rank"               -> per-step action-attributable stake-rank surrogate (default)
      "differential"       -> potential-based per-step decrease of the true windowed gini
                              (replaces the level; return is endpoint-only)
      "differential_shaped"-> level + beta * potential-based shaping: keeps the true
                              windowed-gini objective AND adds the dense shaping term
      "windowed"           -> the original level reward (1 - relative_gini, post-filtered);
                              integrative, so needs --gini-lambda 0 to learn
      "grad"               -> jax.grad of the new relative gini w.r.t. the action
      "gini_rank"          -> same rank normalisation as "rank" but scoring the one-step
                              gini the action produces, not a stake-deficit proxy
      "windowed_rank_mixed"-> convex blend (level + beta * gini_rank) / (1 + beta):
                              beta=0 recovers "windowed", large beta -> "gini_rank"
    Original windowed gini level reward (1 - relative_gini, post-filtered) + the
    monitored relative gini. Logged regardless of mode so runs stay comparable."""
    original_gini_reward, gini_value = gini_reward(new_state, params)
    fairness_shaping = jnp.zeros_like(gini_value)
    # Fairness *training* signal fed to the gini head (mode-dependent).
    if static_params.gini_reward_mode == "differential":
        fairness_shaping = static_params.gamma * original_gini_reward - gini_reward(old_state, params)[0]
        fairness_reward_value = fairness_shaping
    elif static_params.gini_reward_mode == "differential_shaped":
        fairness_shaping = static_params.gamma * original_gini_reward - gini_reward(old_state, params)[0]
        fairness_reward_value = original_gini_reward + static_params.shaping_beta * fairness_shaping
    elif static_params.gini_reward_mode == "windowed":
        fairness_reward_value = original_gini_reward
    elif static_params.gini_reward_mode == "grad":
        fairness_reward_value = gini_grad_reward(action, old_state)
    elif static_params.gini_reward_mode == "gini_rank":
        fairness_reward_value = relative_gini_rank_reward(action, new_state, params, static_params)
    elif static_params.gini_reward_mode == "windowed_rank_mixed":
        fairness_shaping = relative_gini_rank_reward(action, new_state, params, static_params)
        fairness_reward_value = ((original_gini_reward + static_params.shaping_beta * fairness_shaping)
                                 / (1.0 + static_params.shaping_beta))
    else:
        fairness_reward_value = relative_stake_rank_reward(action, new_state, params, static_params)
    distance_reward_value, avg_value = distance_reward(action, new_state, params, static_params)
    # The optimized reward uses the fairness training signal.
    weighted_value_sum = (jnp.array([fairness_reward_value, distance_reward_value]) * params.rewards_weights).sum()
    # Same weighting but with the ORIGINAL windowed gini -- monitoring only, comparable
    # across gini_reward_mode.
    weighted_original_sum = (jnp.array([original_gini_reward, distance_reward_value]) * params.rewards_weights).sum()
    return weighted_value_sum, {"gini": gini_value,
                                "fairness_reward": fairness_reward_value,
                                "fairness_shaping": fairness_shaping,
                                "gini_reward": original_gini_reward,
                                "distance": avg_value,
                                "distance_reward": distance_reward_value,
                                "weighted_reward": weighted_value_sum,
                                "weighted_original_reward": weighted_original_sum}


def null_reward() -> tuple[jax.Array, Dict[str, jax.Array]]:
    return jnp.array(0.0, dtype=jnp.float32), {
        "gini": jnp.array(0, dtype=jnp.float32),
        "fairness_reward": jnp.array(0, dtype=jnp.float32),
        "fairness_shaping": jnp.array(0, dtype=jnp.float32),
        "gini_reward": jnp.array(0, dtype=jnp.float32),
        "distance": jnp.array(0, dtype=jnp.float32),
        "distance_reward": jnp.array(0, dtype=jnp.float32),
        "weighted_reward": jnp.array(0, dtype=jnp.float32),
        "weighted_original_reward": jnp.array(0, dtype=jnp.float32)
    }
