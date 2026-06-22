"""Tests for the multi-objective reward definitions."""

import jax
import jax.numpy as jnp
import pytest

from rl_blockchain.env import (
    RewardParams,
    Rewards,
    all_pairs_shortest_paths,
    compute_rewards,
    distance_reward,
    distribution_reward,
    init_reward_state,
    push_chosen,
    stack_rewards,
    voter_ratio_reward,
)
from rl_blockchain.graph import BlockchainGraph, set_chosen


def path_graph(n=4):
    """Path graph 0-1-2-...-(n-1)."""
    senders = jnp.arange(n - 1)
    receivers = jnp.arange(1, n)
    return BlockchainGraph.new(num_nodes=n, senders=senders, receivers=receivers)


# --- distance matrix / good distance --------------------------------------- #
def test_all_pairs_shortest_paths_on_path():
    g = path_graph(4)
    d = all_pairs_shortest_paths(g)
    expected = jnp.array(
        [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=jnp.float32
    )
    assert jnp.allclose(d, expected)


def test_disconnected_nodes_are_inf():
    # Two disjoint edges: 0-1 and 2-3.
    g = BlockchainGraph.new(num_nodes=4, senders=[0, 2], receivers=[1, 3])
    d = all_pairs_shortest_paths(g)
    assert jnp.isinf(d[0, 2])
    assert d[0, 1] == 1.0


def test_good_avg_distance_excludes_inf():
    g = path_graph(4)
    rs = init_reward_state(g, RewardParams())
    # mean of off-diagonal ordered distances = 20 / 12
    assert jnp.allclose(rs.good_avg_distance, 20.0 / 12.0)


# --- voter ratio ------------------------------------------------------------ #
def test_voter_ratio():
    g = path_graph(4)
    g = set_chosen(g, jnp.array([True, False, True, False]))
    assert float(voter_ratio_reward(g)) == 0.5


# --- distance reward -------------------------------------------------------- #
def test_distance_reward_ratio():
    g = path_graph(4)
    rs = init_reward_state(g, RewardParams())
    g = set_chosen(g, jnp.array([True, False, False, True]))  # nodes 0 and 3, dist 3
    r = distance_reward(g, rs)
    assert jnp.allclose(r, (20.0 / 12.0) / 3.0)  # good_avg / avg


def test_distance_reward_grows_as_voters_cluster():
    g = path_graph(4)
    rs = init_reward_state(g, RewardParams())
    far = distance_reward(set_chosen(g, jnp.array([True, False, False, True])), rs)  # dist 3
    near = distance_reward(set_chosen(g, jnp.array([True, True, False, False])), rs)  # dist 1
    assert float(near) > float(far)


def test_distance_reward_zero_when_under_two_chosen():
    g = path_graph(4)
    rs = init_reward_state(g, RewardParams())
    g = set_chosen(g, jnp.array([True, False, False, False]))
    assert float(distance_reward(g, rs)) == 0.0


def test_distance_reward_skips_unreachable_pairs():
    g = BlockchainGraph.new(num_nodes=4, senders=[0, 2], receivers=[1, 3])
    rs = init_reward_state(g, RewardParams())
    # choose 0 and 2 — unreachable from each other -> no valid pair -> 0
    g = set_chosen(g, jnp.array([True, False, True, False]))
    assert float(distance_reward(g, rs)) == 0.0


# --- distribution reward ---------------------------------------------------- #
def test_push_chosen_rolls_window():
    g = path_graph(4)
    params = RewardParams(horizon=3)
    rs = init_reward_state(g, params)
    rs = push_chosen(rs, jnp.array([True, False, False, False]))
    rs = push_chosen(rs, jnp.array([False, True, False, False]))
    assert rs.choose_history.shape == (3, 4)
    # most recent row is last
    assert rs.choose_history[-1].tolist() == [False, True, False, False]
    assert rs.choose_history[0].tolist() == [False, False, False, False]


def test_distribution_reward_peaks_at_fair_share():
    n = 4
    g = path_graph(n)
    params = RewardParams(horizon=4, sigma=0.05)
    rs = init_reward_state(g, params)
    # Node 0 chosen exactly 1/4 of the window == fair share 1/n.
    rs = rs.replace(
        choose_history=jnp.array(
            [
                [True, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
    )
    fair = distribution_reward(g, rs, params)

    # All nodes never chosen (freq 0, far from 1/4) -> lower reward.
    rs0 = rs.replace(choose_history=jnp.zeros((4, n), dtype=bool))
    none = distribution_reward(g, rs0, params)
    assert float(fair) > float(none)


def test_distribution_reward_is_one_when_all_fair():
    n = 4
    g = path_graph(n)
    params = RewardParams(horizon=4, sigma=0.05)
    rs = init_reward_state(g, params)
    # Each node chosen exactly once in the window -> every freq == 1/n.
    rs = rs.replace(choose_history=jnp.eye(n, dtype=bool))
    assert jnp.allclose(distribution_reward(g, rs, params), 1.0)


def test_distribution_reward_bounded_unit_interval():
    n = 4
    g = path_graph(n)
    params = RewardParams(horizon=4, sigma=0.2)
    rs = init_reward_state(g, params)
    rs = rs.replace(choose_history=jnp.ones((4, n), dtype=bool))  # freq 1, far from fair
    r = float(distribution_reward(g, rs, params))
    assert 0.0 <= r <= 1.0


# --- combined + jax transforms --------------------------------------------- #
def test_compute_rewards_bundle_and_stack():
    g = path_graph(4)
    params = RewardParams()
    rs = init_reward_state(g, params)
    g = set_chosen(g, jnp.array([True, False, False, True]))
    rs = push_chosen(rs, g.chosen)
    rewards = compute_rewards(g, rs, params)
    assert isinstance(rewards, Rewards)
    vec = stack_rewards(rewards)
    assert vec.shape == (3,)  # voter_ratio, distance, distribution
    assert jnp.all(jnp.isfinite(vec))


def test_rewards_are_vmappable():
    params = RewardParams()
    batch = 8
    keys = jax.random.split(jax.random.PRNGKey(0), batch)

    def make(key):
        trust = jax.random.uniform(key, (4,))
        g = BlockchainGraph.new(
            num_nodes=4, senders=[0, 1, 2], receivers=[1, 2, 3], trust_rating=trust
        )
        g = set_chosen(g, jnp.array([True, False, False, True]))
        rs = init_reward_state(g, params)
        rs = push_chosen(rs, g.chosen)
        return compute_rewards(g, rs, params)

    rewards = jax.vmap(make)(keys)
    assert stack_rewards(rewards).shape == (batch, 3)
