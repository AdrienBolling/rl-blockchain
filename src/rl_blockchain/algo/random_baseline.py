"""Uniform-random committee baseline — the fairness lower bound.

A policy that draws a uniform ``k``-subset committee every round makes each node
validate ~``k/n`` of the time, so the windowed stake distribution is flat and the
relative gini → 0. It is therefore the natural *fairness benchmark*: no trained model
should be fairer than pure randomness, and its (poor) latency is the "no latency
optimization" reference at the other end of the Pareto front.

Deliberately kept apart from ``algo.ppo``: there is no model, no ``PPOState`` and no
value head here, so the eval rollout is much simpler than ``rollout_eval``. Mirrors the
batching / metric aggregation of :func:`algo.ppo.eval_ppo` so the returned ``metrics``
dict is drop-in comparable with a trained run's (same keys via ``log_fn``).
"""

import logging
from functools import partial, lru_cache
from typing import Callable

import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments.environment import TEnvParams

from rl_blockchain.BlockEnv import EnvParams
from rl_blockchain.BlockEnv.BlockEnv import uniform_k_true_mask
from rl_blockchain.scripts.env_factory import LOG_TYPE

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=("env", "steps_in_episode"))
def rollout_random(key_input, env: environment.Environment,
                   env_params_episode: EnvParams, steps_in_episode: int):
    """One episode of the uniform-random committee policy (no model involved).

    Draws ``action = uniform_k_true_mask(k=state.nb_val)`` each step and threads the
    env like the eval rollout, returning only ``(rewards, dones, infos)`` (the info
    dict already carries gini / distance / ... for aggregation).
    """
    key_reset, key_episode = jax.random.split(key_input)
    _, first_state = env.reset(key_reset, env_params_episode)

    def step(carry, _):
        state, key = carry
        next_key, key_step, key_act = jax.random.split(key, 3)
        action = uniform_k_true_mask(key_act, env.nb_nodes, state.nb_val).astype(jnp.bool_)
        _, next_state, reward, done, infos = env.step(
            key_step, state, action, env_params_episode)
        return (next_state, next_key), (reward, done, infos)

    (_, _), (rewards, dones, infos) = jax.lax.scan(
        step, (first_state, key_episode), None, steps_in_episode)
    return rewards, dones, infos


@lru_cache(maxsize=None)
def _vectorized_rollout_random(env, steps_in_episode: int):
    """Build (once) the jitted, vmapped random rollout. Cached on (env, steps)."""

    def single(rng, new_param):
        return rollout_random(rng, env, new_param, steps_in_episode)

    return jax.jit(jax.vmap(single, in_axes=(0, 0)))


def eval_random(env: environment.Environment, key: jax.Array,
                create_params_fn: Callable[[jax.Array], TEnvParams],
                num_episodes: int = 100, batch_size: int = 10,
                log_fn: LOG_TYPE = None) -> dict[str, jax.Array]:
    """Aggregate metrics of the uniform-random baseline over ``num_episodes``.

    Same batching + ``log_fn`` aggregation as :func:`algo.ppo.eval_ppo`, so the
    resulting metrics dict shares its keys (gini, distance, ...) and merges directly
    into the eval comparison table.
    """
    steps_in_episode = int(env.default_params.max_steps_in_episode)
    vm_rollouts = _vectorized_rollout_random(env, steps_in_episode)
    params_map = jax.vmap(create_params_fn)

    all_rewards, all_dones, all_infos = [], [], []
    num_batches = (num_episodes + batch_size - 1) // batch_size

    for this_batch_key in jax.random.split(key, num_batches):
        rollout_key, param_key = jax.random.split(this_batch_key)
        subkeys = jax.random.split(rollout_key, batch_size)
        params_list = params_map(jax.random.split(param_key, batch_size))
        rews, dones, infos = vm_rollouts(subkeys, params_list)
        all_rewards.append(rews)
        all_dones.append(dones)
        all_infos.append(infos)

    all_rewards = jnp.concatenate(all_rewards, axis=0)
    all_dones = jnp.concatenate(all_dones, axis=0)
    all_infos = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_infos)

    logger.info(f"Evaluated {num_episodes} random-baseline episodes "
                f"in {num_batches} batches of at most {batch_size} envs.")

    metrics = log_fn(all_infos, all_rewards, all_dones)
    metrics["avg_returns_episode"] = all_rewards.sum(axis=1).mean()
    return metrics
