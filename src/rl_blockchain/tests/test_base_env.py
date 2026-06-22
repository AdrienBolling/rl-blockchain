"""Tests for the stateless multi-objective env interface.

A tiny ``DummyEnv`` exercises the contract: abstractness, statelessness,
jit-ability, parallel ``vmap`` rollouts, ``lax.scan`` episodes, and the
pytree reward bundle.
"""

import jax
import jax.numpy as jnp
import pytest

from rl_blockchain.env import (
    BaseBlockchainEnv,
    EnvParams,
    EnvState,
    TimeStep,
    stack_rewards,
)
from rl_blockchain.graph import BlockchainGraph, set_chosen, top_k_by_trust


class DummyEnv(BaseBlockchainEnv):
    """Minimal concrete env used only for interface testing."""

    @property
    def default_params(self):
        return EnvParams(num_nodes=8, max_steps=5)

    @property
    def num_objectives(self):
        return 2

    def action_size(self, params):
        return params.num_nodes

    def observation_shape(self, params):
        return (params.num_nodes, 3)

    def reset(self, key, params):
        trust = jax.random.uniform(key, (params.num_nodes,))
        graph = BlockchainGraph.new(num_nodes=params.num_nodes, trust_rating=trust)
        state = EnvState(graph=graph, time=jnp.int32(0))
        return self.observe(state, params), state

    def observe(self, state, params):
        return state.graph.features.as_matrix()

    def step(self, key, state, action, params):
        graph = set_chosen(state.graph, action.astype(jnp.bool_))
        time = state.time + 1
        state = EnvState(graph=graph, time=time)
        reward = {
            "coverage": graph.chosen.mean(),
            "trust": (graph.chosen * graph.trust_rating).sum(),
        }
        done = time >= params.max_steps
        return TimeStep(obs=self.observe(state, params), state=state, reward=reward, done=done)


@pytest.fixture
def env():
    return DummyEnv()


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseBlockchainEnv()


def test_reset_returns_obs_and_state(env):
    p = env.default_params
    obs, state = env.reset(jax.random.PRNGKey(0), p)
    assert obs.shape == env.observation_shape(p)
    assert int(state.time) == 0
    assert state.graph.num_nodes == p.num_nodes


def test_reset_is_deterministic_in_key(env):
    p = env.default_params
    _, s1 = env.reset(jax.random.PRNGKey(0), p)
    _, s2 = env.reset(jax.random.PRNGKey(0), p)
    _, s3 = env.reset(jax.random.PRNGKey(1), p)
    assert jnp.allclose(s1.graph.trust_rating, s2.graph.trust_rating)
    assert not jnp.allclose(s1.graph.trust_rating, s3.graph.trust_rating)


def test_step_is_jittable_and_stateless(env):
    p = env.default_params
    _, state = env.reset_jit(jax.random.PRNGKey(0), p)
    action = top_k_by_trust(state.graph, 3)
    ts = env.step_jit(jax.random.PRNGKey(1), state, action, p)
    assert isinstance(ts, TimeStep)
    assert int(ts.state.time) == 1
    # original state untouched (stateless / functional)
    assert int(state.time) == 0
    assert set(ts.reward) == {"coverage", "trust"}


def test_done_triggers_at_max_steps(env):
    p = env.default_params
    _, state = env.reset(jax.random.PRNGKey(0), p)
    state = state.replace(time=jnp.int32(p.max_steps - 1))
    action = jnp.zeros(p.num_nodes, dtype=bool)
    ts = env.step(jax.random.PRNGKey(0), state, action, p)
    assert bool(ts.done)


def test_stack_rewards_shapes(env):
    p = env.default_params
    _, state = env.reset(jax.random.PRNGKey(0), p)
    ts = env.step(jax.random.PRNGKey(1), state, top_k_by_trust(state.graph, 3), p)
    assert stack_rewards(ts.reward).shape == (env.num_objectives,)


def test_vmap_parallel_envs(env):
    p = env.default_params
    batch = 16
    keys = jax.random.split(jax.random.PRNGKey(42), batch)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(keys, p)
    actions = jax.vmap(lambda g: top_k_by_trust(g, 3))(state.graph)
    ts = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, state, actions, p)
    assert obs.shape == (batch, *env.observation_shape(p))
    assert stack_rewards(ts.reward).shape == (batch, env.num_objectives)


def test_scan_episode_rollout(env):
    p = env.default_params

    def rollout(key):
        k0, k = jax.random.split(key)
        _, s0 = env.reset(k0, p)

        def body(carry, _):
            s, k = carry
            k, ks = jax.random.split(k)
            ts = env.step(ks, s, top_k_by_trust(s.graph, 3), p)
            return (ts.state, k), ts.reward

        _, rewards = jax.lax.scan(body, (s0, k), None, length=p.max_steps)
        return rewards

    rewards = jax.jit(rollout)(jax.random.PRNGKey(7))
    assert rewards["coverage"].shape == (p.max_steps,)
    assert rewards["trust"].shape == (p.max_steps,)
