import jax
import jax.numpy as jnp

from rl_blockchain.BlockEnv.BlockEnv import uniform_k_true_mask
from rl_blockchain.BlockEnv.BlockchainGraph import STATIC_MASKS_DICT
from rl_blockchain.BlockEnv.state_params import (
    EnvParams,
    Speeders,
    StaticEnvParams,
    init_fixed_nb_val_factory,
    white_param_fn,
)
from rl_blockchain.BlockEnv import BlockchainEnv, create_rd_adj_matrix
from rl_blockchain.scripts.env_factory import next_edge_white
from rl_blockchain.scripts.parser import REF_FILENAME


def _build_test_env(seed: int = 0, nb_nodes: int = 7, nb_validators: int = 4, max_steps: int = 5):
    key = jax.random.PRNGKey(seed)
    non_diag_mask = STATIC_MASKS_DICT[nb_nodes]
    speeder = Speeders.create(nb_nodes)

    adj_matrix = create_rd_adj_matrix(nb_nodes, key)
    adj_unique = adj_matrix.flatten().take(non_diag_mask).take(speeder.unique)
    sigma = jnp.zeros_like(adj_unique)

    params = EnvParams.create(
        adj_network_uniq=adj_unique,
        list_sigma_value=sigma,
        rewards_weights=[0.5, 0.5],
        max_steps_in_episode=max_steps,
    )
    static_params = StaticEnvParams.create(
        nb_nodes=nb_nodes,
        filename=REF_FILENAME[nb_nodes],
        init_nb_val_fn=init_fixed_nb_val_factory(nb_validators),
        next_nb_val_fn=white_param_fn,
        next_map_fn=next_edge_white,
    )
    env = BlockchainEnv(params, static_params)
    return env, params


def test_reset_contract_is_consistent():
    env, params = _build_test_env(seed=11)
    obs, state = env.reset(jax.random.PRNGKey(123), params)

    assert state.time == 0
    assert state.ring_history.shape == (env._static_params.horizon, env.nb_nodes)
    assert state.ring_history.dtype == jnp.bool_
    assert 0 < int(state.nb_val) <= env.nb_nodes

    assert obs.n_node.shape == (1,)
    assert int(obs.n_node[0]) == env.nb_nodes
    assert obs.nodes.shape == (env.nb_nodes,)
    assert obs.nodes.dtype == jnp.float32
    assert obs.edges.shape[0] == env.nb_nodes * (env.nb_nodes - 1)
    assert obs.globals.shape == (1,)
    assert float(obs.globals[0]) == float(state.nb_val)


def test_step_with_legal_action_returns_valid_transition():
    env, params = _build_test_env(seed=7)
    reset_key = jax.random.PRNGKey(1)
    obs, state = env.reset(reset_key, params)

    action_key = jax.random.PRNGKey(2)
    legal_action = uniform_k_true_mask(action_key, env.nb_nodes, state.nb_val).astype(jnp.bool_)
    next_obs, next_state, reward, done, info = env.step(jax.random.PRNGKey(3), state, legal_action, params)

    assert bool(done) is False
    assert int(next_state.time) == int(state.time) + 1
    assert jnp.isfinite(reward)

    expected_info_keys = {
        "gini",
        "gini_reward",
        "distance",
        "distance_reward",
        "weighted_reward",
        "nb_validators",
    }
    assert set(info.keys()) == expected_info_keys
    assert int(info["nb_validators"]) == int(state.nb_val)
    assert next_obs.nodes.shape == obs.nodes.shape


def test_illegal_action_ends_episode_and_returns_null_reward():
    env, params = _build_test_env(seed=5)
    _, state = env.reset(jax.random.PRNGKey(21), params)

    illegal_action = jnp.zeros((env.nb_nodes,), dtype=jnp.bool_)
    _, _, reward, done, info = env.step(jax.random.PRNGKey(22), state, illegal_action, params)

    assert bool(done) is True
    assert float(reward) == 0.0
    assert float(info["weighted_reward"]) == 0.0
    assert float(info["gini_reward"]) == 0.0
    assert float(info["distance_reward"]) == 0.0


def test_step_auto_resets_when_max_steps_reached():
    env, params = _build_test_env(seed=9, max_steps=1)
    reset_key = jax.random.PRNGKey(100)
    _, state = env.reset(reset_key, params)

    action = uniform_k_true_mask(jax.random.PRNGKey(101), env.nb_nodes, state.nb_val).astype(jnp.bool_)
    step_key = jax.random.PRNGKey(102)

    _, key_reset = jax.random.split(step_key)
    expected_reset_obs, expected_reset_state = env.reset(key_reset, params)

    obs_after_step, state_after_step, _, done, _ = env.step(step_key, state, action, params)

    assert bool(done) is True
    assert int(state_after_step.time) == int(expected_reset_state.time) == 0
    assert jnp.array_equal(state_after_step.ring_history, expected_reset_state.ring_history)
    assert jnp.array_equal(obs_after_step.nodes, expected_reset_obs.nodes)
    assert jnp.array_equal(obs_after_step.globals, expected_reset_obs.globals)


def test_reset_and_step_are_deterministic_for_same_inputs():
    env, params = _build_test_env(seed=4)

    reset_key = jax.random.PRNGKey(200)
    obs_a, state_a = env.reset(reset_key, params)
    obs_b, state_b = env.reset(reset_key, params)

    assert jnp.array_equal(obs_a.nodes, obs_b.nodes)
    assert jnp.array_equal(obs_a.edges, obs_b.edges)
    assert jnp.array_equal(state_a.ring_history, state_b.ring_history)
    assert int(state_a.nb_val) == int(state_b.nb_val)

    action = uniform_k_true_mask(jax.random.PRNGKey(201), env.nb_nodes, state_a.nb_val).astype(jnp.bool_)
    step_key = jax.random.PRNGKey(202)

    out_a = env.step(step_key, state_a, action, params)
    out_b = env.step(step_key, state_a, action, params)

    obs_step_a, state_step_a, reward_a, done_a, info_a = out_a
    obs_step_b, state_step_b, reward_b, done_b, info_b = out_b

    assert jnp.array_equal(obs_step_a.nodes, obs_step_b.nodes)
    assert jnp.array_equal(state_step_a.ring_history, state_step_b.ring_history)
    assert float(reward_a) == float(reward_b)
    assert bool(done_a) == bool(done_b)
    assert float(info_a["weighted_reward"]) == float(info_b["weighted_reward"])
