"""Multi-objective reward definitions for the blockchain voting env.

Three objectives (all returned together in the :class:`Rewards` pytree, one
scalar leaf each — see :func:`rl_blockchain.env.types.stack_rewards`):

1. ``voter_ratio`` — fraction of nodes selected to vote
   (``n_chosen / num_nodes``). Cheap, depends only on the graph.

2. ``distance`` — how tightly grouped the chosen (voter) nodes are,
   **regularised by a reference "good" average distance**. We compute the
   all-pairs shortest-path distance matrix once at episode init (the topology is
   fixed within an episode) together with the mean pairwise distance over the
   whole graph (``good_avg_distance``). The reward is
   ``good_avg_distance / avg_chosen_distance``: it *grows* as the chosen nodes
   get closer together, equals 1 when they are as spread as a typical pair, and
   is bounded in ``(0, good_avg_distance]`` (connected pairs are ≥ 1 hop apart).

3. ``distribution`` — a fairness signal in ``[0, 1]``. For each node we take the
   fraction of the last ``horizon`` steps it was chosen, and score it with an
   (unnormalised) Gaussian centered at the fair share ``1 / num_nodes`` with std
   ``sigma``: ``exp(-½((freq - fair)/sigma)²)``. This is 1 when a node hits its
   fair share and decays towards 0 otherwise; ``sigma`` controls tolerance, not
   scale. The reward is the mean over all nodes.

Everything is a pure, jittable, whole-array function. The extra per-episode
arrays (distance matrix, rolling history) live in :class:`RewardState`, which a
concrete env carries inside its ``EnvState``.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp
from jax import Array

from rl_blockchain.graph import BlockchainGraph

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Config & state
# --------------------------------------------------------------------------- #
@flax.struct.dataclass
class RewardParams:
    """Static reward hyper-parameters.

    ``horizon`` drives the shape of the rolling history buffer, so it is a
    ``pytree_node=False`` (compile-time constant) field. ``sigma`` is an
    ordinary leaf so it can be swept without recompiling.
    """

    horizon: int = flax.struct.field(pytree_node=False, default=50)
    sigma: float = 0.05


@flax.struct.dataclass
class Rewards:
    """The multi-objective reward bundle (one scalar per objective)."""

    voter_ratio: Array
    distance: Array
    distribution: Array


@flax.struct.dataclass
class RewardState:
    """Per-episode arrays the reward functions need.

    Attributes:
        choose_history: ``[horizon, num_nodes]`` bool — rolling window of the
            ``chosen`` masks, oldest first, newest last.
        distance_matrix: ``[num_nodes, num_nodes]`` float — all-pairs
            shortest-path (hop) distances; ``inf`` for unreachable pairs.
        good_avg_distance: scalar float — mean pairwise distance over all
            reachable, distinct node pairs in the graph.
    """

    choose_history: Array
    distance_matrix: Array
    good_avg_distance: Array


# --------------------------------------------------------------------------- #
# Initialisation helpers
# --------------------------------------------------------------------------- #
@jax.jit
def all_pairs_shortest_paths(graph: BlockchainGraph) -> Array:
    """All-pairs shortest-path (unit-weight, undirected) distance matrix.

    Floyd–Warshall over a static ``num_nodes`` — jittable and GPU-friendly.
    Unreachable pairs are ``inf``; the diagonal is 0.
    """
    n = graph.num_nodes
    inf = jnp.float32(jnp.inf)

    dist = jnp.full((n, n), inf, dtype=jnp.float32)
    # Treat edges as undirected, unit weight.
    dist = dist.at[graph.senders, graph.receivers].set(1.0)
    dist = dist.at[graph.receivers, graph.senders].set(1.0)
    # Force a zero diagonal last (overrides any self-loops).
    dist = dist.at[jnp.diag_indices(n)].set(0.0)

    def relax(k, d):
        return jnp.minimum(d, d[:, k][:, None] + d[k, :][None, :])

    return jax.lax.fori_loop(0, n, relax, dist)


@jax.jit
def _mean_pairwise_distance(distance_matrix: Array) -> Array:
    """Mean over reachable, off-diagonal entries of a distance matrix."""
    n = distance_matrix.shape[0]
    finite = jnp.isfinite(distance_matrix)
    offdiag = ~jnp.eye(n, dtype=bool)
    mask = finite & offdiag
    safe = jnp.where(mask, distance_matrix, 0.0)
    total = jnp.sum(safe)
    count = jnp.sum(mask)
    return total / jnp.maximum(count, 1.0)


def init_reward_state(
    graph: BlockchainGraph,
    params: RewardParams,
    distance_matrix: Array | None = None,
) -> RewardState:
    """Build the per-episode :class:`RewardState` from a fresh graph.

    Args:
        graph: The freshly generated graph.
        params: Reward hyper-parameters.
        distance_matrix: Optional precomputed ``[N, N]`` distance matrix (e.g.
            the latency matrix from :func:`rl_blockchain.graph.generate_network`).
            If ``None``, falls back to unit-weight shortest paths over the
            graph's own edges.
    """
    if distance_matrix is None:
        distance_matrix = all_pairs_shortest_paths(graph)
    good_avg_distance = _mean_pairwise_distance(distance_matrix)
    choose_history = jnp.zeros((params.horizon, graph.num_nodes), dtype=jnp.bool_)
    return RewardState(
        choose_history=choose_history,
        distance_matrix=distance_matrix,
        good_avg_distance=good_avg_distance,
    )


@jax.jit
def push_chosen(reward_state: RewardState, chosen: Array) -> RewardState:
    """Append the latest ``chosen`` mask to the rolling history (drop oldest)."""
    new_row = chosen.astype(jnp.bool_)[None, :]
    history = jnp.concatenate([reward_state.choose_history[1:], new_row], axis=0)
    return reward_state.replace(choose_history=history)


# --------------------------------------------------------------------------- #
# Individual objectives
# --------------------------------------------------------------------------- #
@jax.jit
def voter_ratio_reward(graph: BlockchainGraph) -> Array:
    """Fraction of nodes currently selected to vote."""
    return jnp.mean(graph.chosen.astype(jnp.float32))


@jax.jit
def distance_reward(graph: BlockchainGraph, reward_state: RewardState) -> Array:
    """``good_avg_distance / avg_chosen_distance`` — grows as voters cluster.

    Bounded in ``(0, good_avg_distance]``. Returns 0 when fewer than two
    reachable chosen nodes exist (no valid pair to measure).
    """
    d = reward_state.distance_matrix
    n = d.shape[0]
    chosen = graph.chosen.astype(jnp.float32)
    pair = chosen[:, None] * chosen[None, :]
    finite = jnp.isfinite(d)
    offdiag = ~jnp.eye(n, dtype=bool)
    weight = pair * finite * offdiag
    # Replace inf before multiplying to avoid 0 * inf = nan.
    safe_d = jnp.where(finite, d, 0.0)
    total = jnp.sum(safe_d * weight)
    count = jnp.sum(weight)
    avg = total / jnp.maximum(count, 1.0)
    reward = reward_state.good_avg_distance / jnp.maximum(avg, _EPS)
    # No reachable pair of chosen nodes -> neutral 0 (avoid good / 0 blow-up).
    return jnp.where(count > 0, reward, 0.0)


@jax.jit
def distribution_reward(graph: BlockchainGraph, reward_state: RewardState, params: RewardParams) -> Array:
    """Mean fairness score over nodes, bounded in ``[0, 1]``.

    For each node, the selection frequency over the rolling horizon is scored by
    an unnormalised Gaussian centered at the fair share ``1 / num_nodes`` (std
    ``sigma``): ``exp(-½((freq - fair)/sigma)²)``, which equals 1 at the fair
    share and decays to 0. ``sigma`` controls tolerance only, not scale.
    """
    num_nodes = graph.num_nodes
    freq = jnp.mean(reward_state.choose_history.astype(jnp.float32), axis=0)
    fair_share = 1.0 / num_nodes
    z = (freq - fair_share) / params.sigma
    return jnp.mean(jnp.exp(-0.5 * z**2))


# --------------------------------------------------------------------------- #
# Combined
# --------------------------------------------------------------------------- #
@jax.jit
def compute_rewards(graph: BlockchainGraph, reward_state: RewardState, params: RewardParams) -> Rewards:
    """Compute all three objectives at once, as a :class:`Rewards` pytree."""
    return Rewards(
        voter_ratio=voter_ratio_reward(graph),
        distance=distance_reward(graph, reward_state),
        distribution=distribution_reward(graph, reward_state, params),
    )
