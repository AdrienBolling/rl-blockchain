"""Integration tests: the fully-assembled BlockchainEnv is runnable end-to-end."""

import jax
import jax.numpy as jnp
import pytest

from rl_blockchain.env import (
    BlockchainEnv,
    BlockchainEnvParams,
    BlockchainEnvState,
    RewardParams,
    TimeStep,
    stack_rewards,
)
from rl_blockchain.graph import NetworkConfig


def _params(n=250, max_steps=5, max_size=100):
    return BlockchainEnvParams(
        network=NetworkConfig(num_nodes=n),
        reward=RewardParams(horizon=4, sigma=0.1),
        max_handling_size=max_size,
        max_steps=max_steps,
    )


@pytest.fixture
def env():
    return BlockchainEnv()


def test_reset_assembles_full_state(env):
    params = _params()
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    assert isinstance(state, BlockchainEnvState)
    assert obs.shape == env.observation_shape(params)
    assert state.graph.num_nodes == params.num_nodes
    assert int(state.time) == 0
    # decomposition is assembled and covers all nodes
    assert int(state.cluster_batch.sub_mask.sum()) == params.num_nodes
    # reward state has the latency distance matrix wired in
    assert state.reward_state.distance_matrix.shape == (params.num_nodes, params.num_nodes)
    assert float(state.reward_state.good_avg_distance) > 0.0


def test_step_returns_multiobjective_reward(env):
    params = _params()
    _, state = env.reset(jax.random.PRNGKey(0), params)
    action = jax.random.bernoulli(jax.random.PRNGKey(1), 0.2, (params.num_nodes,))
    ts = env.step(jax.random.PRNGKey(2), state, action, params)
    assert isinstance(ts, TimeStep)
    assert set(ts.reward.__dataclass_fields__) == {"voter_ratio", "distance", "distribution"}
    assert stack_rewards(ts.reward).shape == (env.num_objectives,)
    assert int(ts.state.time) == 1
    # action took effect
    assert int(ts.state.graph.chosen.sum()) == int(action.sum())


def test_step_is_jittable(env):
    params = _params()
    _, state = env.reset(jax.random.PRNGKey(0), params)
    action = jax.random.bernoulli(jax.random.PRNGKey(1), 0.2, (params.num_nodes,))
    ts = env.step_jit(jax.random.PRNGKey(2), state, action, params)
    assert int(ts.state.time) == 1


def test_episode_terminates_at_max_steps(env):
    params = _params(max_steps=5)
    _, state = env.reset(jax.random.PRNGKey(0), params)
    key = jax.random.PRNGKey(1)
    done_flags = []
    for t in range(5):
        key, ka, ks = jax.random.split(key, 3)
        action = jax.random.bernoulli(ka, 0.2, (params.num_nodes,))
        ts = env.step(ks, state, action, params)
        state = ts.state
        done_flags.append(bool(ts.done))
    assert done_flags == [False, False, False, False, True]


def test_scan_rollout_is_jittable(env):
    params = _params(max_steps=8)
    _, init_state = env.reset(jax.random.PRNGKey(0), params)

    @jax.jit
    def rollout(state, key):
        def body(carry, _):
            state, key = carry
            key, ka, ks = jax.random.split(key, 3)
            action = jax.random.bernoulli(ka, 0.2, (params.num_nodes,))
            ts = env.step(ks, state, action, params)
            return (ts.state, key), ts.reward

        (_, _), rewards = jax.lax.scan(body, (state, key), None, length=params.max_steps)
        return rewards

    rewards = rollout(init_state, jax.random.PRNGKey(1))
    assert rewards.voter_ratio.shape == (params.max_steps,)
    assert jnp.all(jnp.isfinite(stack_rewards(rewards)))


def test_reset_is_reproducible(env):
    params = _params()
    _, s1 = env.reset(jax.random.PRNGKey(0), params)
    _, s2 = env.reset(jax.random.PRNGKey(0), params)
    _, s3 = env.reset(jax.random.PRNGKey(1), params)
    assert jnp.allclose(s1.graph.trust_rating, s2.graph.trust_rating)
    assert jnp.allclose(s1.reward_state.distance_matrix, s2.reward_state.distance_matrix)
    assert not jnp.allclose(s1.graph.trust_rating, s3.graph.trust_rating)


def test_reset_jit_raises(env):
    with pytest.raises(NotImplementedError):
        env.reset_jit(jax.random.PRNGKey(0), _params())
