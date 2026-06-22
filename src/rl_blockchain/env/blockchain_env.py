"""``BlockchainEnv`` — the concrete, runnable environment.

Assembles every piece built so far into a working stateless MDP:

* **reset** (host-side, once per episode): generate a fractal network
  (:func:`generate_network`), sample initial node trust, partition the graph
  into clusters (:mod:`rl_blockchain.decomp`), build the static
  :class:`ClusterBatch`, and initialise the :class:`RewardState` with the
  latency distance matrix. Not jitted — network generation is irregular.
* **step** (jitted, the hot path): apply the per-node voting action, roll the
  selection history, and return the multi-objective reward pytree.

The static, per-episode topology (cluster batch, distance matrix inside the
reward state) is carried inside :class:`BlockchainEnvState`, so a single
jitted ``step`` works under ``jax.lax.scan`` and ``jax.vmap``.

The action is the fundamental decision of the MDP: a boolean ``[num_nodes]``
mask saying which nodes vote next. *How* a (hierarchical) policy produces that
mask is orthogonal to the environment — the cluster decomposition is exposed in
the state for the policy to use.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp
from jax import Array

from rl_blockchain.decomp import (
    DEFAULT_PARTITIONER,
    ClusterBatch,
    PartitionContext,
    Partitioner,
    build_cluster_batch,
)
from rl_blockchain.graph import BlockchainGraph, NetworkConfig, generate_network, set_chosen

from .base import BaseBlockchainEnv
from .rewards import RewardParams, RewardState, compute_rewards, init_reward_state, push_chosen
from .types import Action, Observation, PRNGKey, TimeStep


@flax.struct.dataclass
class BlockchainEnvParams:
    """Static configuration for :class:`BlockchainEnv`.

    ``network`` and the scalar caps are ``pytree_node=False`` (compile-time
    constants); ``reward`` is a nested pytree so ``sigma`` can be swept.
    """

    network: NetworkConfig = flax.struct.field(
        pytree_node=False, default=NetworkConfig(num_nodes=128)
    )
    reward: RewardParams = RewardParams()
    max_handling_size: int = flax.struct.field(pytree_node=False, default=100)
    max_steps: int = flax.struct.field(pytree_node=False, default=1000)

    @property
    def num_nodes(self) -> int:
        return self.network.num_nodes


@flax.struct.dataclass
class BlockchainEnvState:
    """Full mutable state of one environment instance.

    Attributes:
        graph: The blockchain graph with current node features.
        time: Step counter since reset.
        reward_state: Rolling history + distance matrix for the rewards.
        cluster_batch: Static cluster decomposition (constant within an episode;
            exposed for the hierarchical policy).
    """

    graph: BlockchainGraph
    time: Array
    reward_state: RewardState
    cluster_batch: ClusterBatch


class BlockchainEnv(BaseBlockchainEnv):
    """Stateless, multi-objective blockchain voting environment."""

    def __init__(self, partitioner: Partitioner = DEFAULT_PARTITIONER):
        self.partitioner = partitioner

    # --- descriptors ------------------------------------------------------- #
    @property
    def default_params(self) -> BlockchainEnvParams:
        return BlockchainEnvParams()

    @property
    def num_objectives(self) -> int:
        return 3  # voter_ratio, distance, distribution

    def action_size(self, params: BlockchainEnvParams) -> int:
        return params.num_nodes

    def observation_shape(self, params: BlockchainEnvParams) -> tuple[int, ...]:
        return (params.num_nodes, 3)

    # --- core transition functions ----------------------------------------- #
    def reset(
        self, key: PRNGKey, params: BlockchainEnvParams
    ) -> tuple[Observation, BlockchainEnvState]:
        """Generate a fresh network and dynamic state (host-side, not jitted)."""
        key_net, key_trust = jax.random.split(key)

        graph, topology = generate_network(key_net, params.network)
        # Sample initial per-node trust so the environment is non-trivial.
        trust = jax.random.uniform(key_trust, (params.num_nodes,))
        graph = graph.replace_features(trust_rating=trust)

        labels = self.partitioner(
            PartitionContext(
                num_nodes=params.num_nodes,
                distance_matrix=topology.distance_matrix,
                topology=topology,
            ),
            params.max_handling_size,
        )
        cluster_batch = build_cluster_batch(
            labels, topology.distance_matrix, params.max_handling_size
        )
        reward_state = init_reward_state(
            graph, params.reward, distance_matrix=topology.distance_matrix
        )

        state = BlockchainEnvState(
            graph=graph,
            time=jnp.int32(0),
            reward_state=reward_state,
            cluster_batch=cluster_batch,
        )
        return self.observe(state, params), state

    def observe(self, state: BlockchainEnvState, params: BlockchainEnvParams) -> Observation:
        return state.graph.features.as_matrix()

    def step(
        self,
        key: PRNGKey,
        state: BlockchainEnvState,
        action: Action,
        params: BlockchainEnvParams,
    ) -> TimeStep:
        """Apply the per-node voting mask and return the MORL reward (jitted)."""
        del key  # transition is currently deterministic given the action
        chosen = jnp.asarray(action).astype(jnp.bool_)

        graph = set_chosen(state.graph, chosen)
        reward_state = push_chosen(state.reward_state, chosen)
        reward = compute_rewards(graph, reward_state, params.reward)

        time = state.time + 1
        new_state = state.replace(graph=graph, reward_state=reward_state, time=time)
        done = time >= params.max_steps
        return TimeStep(obs=self.observe(new_state, params), state=new_state, reward=reward, done=done)

    def reset_jit(self, key: PRNGKey, params: BlockchainEnvParams):  # noqa: D102
        raise NotImplementedError(
            "reset performs host-side network generation and is not jittable; "
            "call reset() directly. Only step is jitted."
        )
