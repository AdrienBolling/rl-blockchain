from functools import partial
from typing import List

from flax import struct
import jax.numpy as jnp
import jax

import jraph
import rl_blockchain.rl.BlockchainGraph as BlockchainGraph
from rl_blockchain.rl.BlockchainGraph import voting_update, partial_reset, get_feature_all_nodes, \
    approximate_min_mean_distance


@struct.dataclass
class State:
    blockchain: jraph.GraphsTuple
    time_step: int
    inner_step: int
    global_step: int


class BlockchainEnv_intermediary:
    """
    A single-agent fully implemented environment for the blockchain. (No dependencies on any other class, to be able to
    use graph observations)

    This environment has intermediary steps, the agent will choose 1 voting node at each sub-timestep
    """

    def __init__(
            self,
            node_distance_matrix: jnp.ndarray,
            voting_nodes,
            random_key: jnp.ndarray,
    ):
        # Parameters
        self.node_distance_matrix = node_distance_matrix
        self.node_features = ["node_id", "chosen", "distrib_chosen"]
        self.voting_nodes = voting_nodes
        self.key = random_key

        # Settings
        self._rewards = ["gini", "distance"]
        self._rew_sigma = 0.2
        self._max_time_steps = 10000

        # Compute the approximate shortest mean distance for the environment
        self.shortest_mean_distance = approximate_min_mean_distance(node_distance_matrix, voting_nodes,
                                                                    node_distance_matrix.shape[0], key=self.key)

    @partial(jax.jit, static_argnums=[0])
    def _get_first_state(self) -> State:
        """
        Get the initial state of the blockchain.

        Returns:
            The initial state of the blockchain.
        """
        return State(
            blockchain=BlockchainGraph.create_blockchain_graph(
                node_distance_matrix=self.node_distance_matrix,
                node_features_size=len(self.node_features),
            ),
            time_step=0,
            inner_step=0,
            global_step=0,
        )

    @partial(jax.jit, static_argnums=[0])
    def reset(self):
        """
        Reset the environment.
        :return: The initial state of the environment.
        """
        return self._get_first_state()

    @partial(jax.jit, static_argnums=[0])
    def _inner_step(self, state: State, action: int) -> State:
        """
        Perform an inner step in the environment.

        Params:
            state: The current state of the environment.
            action: The action to take.

        Returns:
            The new state of the environment.
        """
        # Get the new voting node index
        voting_node_index = jnp.argmax(action)

        # Update the blobkchain
        new_state = State(
            blockchain=voting_update(state.blockchain, voting_node_index),
            time_step=state.time_step,
            inner_step=state.inner_step + 1,
            global_step=state.global_step + 1,
        )
        return new_state

    @partial(jax.jit, static_argnums=[0])
    def _outer_step(self, state: State) -> State:
        """
        Perform an outer step in the environment.

        Params:
            state: The current state of the environment.

        Returns:
            The new state of the environment.
        """
        # Partially reset the blockchain
        new_state = State(
            blockchain=partial_reset(state.blockchain),
            time_step=state.time_step + 1,
            inner_step=0,
            global_step=state.global_step + 1,
        )

        return new_state

    @partial(jax.jit, static_argnums=[0])
    def _compute_pseudo_gini(self, state: State, sigma=0.2):
        """
        Compute the pseudo-gini of the current state.

        Params:
            state: The current state of the environment.

        Returns:
            The pseudo-gini of the current state.
        """

        # Compute number of nodes
        n_nodes = state.blockchain.nodes.shape[0]
        p_id = 1 / n_nodes

        # Get the array of node features corresponding to the distribution of the nodes being chosen
        p = get_feature_all_nodes(state.blockchain, "distrib_chosen")

        # Compute the pseudo-gini
        return jnp.mean(jnp.exp(-((p - p_id) ** 2) / (2 * sigma ** 2))).astype(float)

    @partial(jax.jit, static_argnums=[0])
    def _compute_average_distance(self, state: State):
        """
        Compute the average distance of the current state.

        Params:
            state: The current state of the environment.

        Returns:
            The average distance of the current state.
        """
        # Compute the distance matrix
        distance_matrix = state.blockchain.edges[:, 0]

        # Compute the average distance, normalized by the approximate shortest mean distance
        return jnp.mean(distance_matrix).astype(float) / self.shortest_mean_distance

    @partial(jax.jit, static_argnums=[0])
    def _compute_reward(self, state: State, weights):
        """
        Compute the reward of the current state.

        Params:
            state: The current state of the environment.

        Returns:
            The reward of the current state.
        """

        # Compute the reward
        rewards = jnp.array([self._compute_pseudo_gini(state), self._compute_average_distance(state)])
        return jnp.dot(rewards, weights).astype(float)

    @partial(jax.jit, static_argnums=[0])
    def _compute_done(self, state: State) -> bool:
        """
        Compute whether the environment is done.

        Params:
            state: The current state of the environment.

        Returns:
            Whether the environment is done.
        """
        return state.time_step >= self._max_time_steps

    @partial(jax.jit, static_argnums=[0])
    def _compute_legal_actions(self, state: State) -> jnp.ndarray:
        """
        Compute the legal actions of the current state.
        A node that has already been chosen (feature "chosen" of the blockchain) cannot be chosen again during the same
        time step.

        Params:
            state: The current state of the environment.

        Returns:
            The legal actions of the current state.
        """

        # Get the array of node features corresponding to the distribution of the nodes being chosen
        chosen = get_feature_all_nodes(state.blockchain, "chosen")

        # Compute the legal actions
        return 1 - chosen

    @partial(jax.jit, static_argnums=[0])
    def _get_infos(self, state: State) -> dict:
        """
        Get the infos of the current state.

        Params:
            state: The current state of the environment.

        Returns:
            The infos of the current state.
        """
        return {
            "time_step": state.time_step,
            "inner_step": state.inner_step,
            "gini_reward": self._compute_pseudo_gini(state),
            "distance_reward": self._compute_average_distance(state),
        }

    def _compute_terminated(self, state: State, action: jnp.ndarray):
        """
        Compute whether the environment is terminated.
        Check the legality of the action taken.

        Params:
            state: The current state of the environment.
            action: The action to take.

        Returns:
            Whether the environment is terminated.
        """
        # Check the legality of the action taken
        legal_mask = self._compute_legal_actions(state)
        return (jnp.sum(action * legal_mask) == 0).astype(bool)

    @partial(jax.jit, static_argnames=('self',))
    def step(self, state: State, action: jnp.ndarray, weights: jnp.ndarray):
        # Normalize weights
        weights = weights / jnp.sum(weights)

        # Terminal flag
        terminated = self._compute_terminated(state, action)

        # Always do the inner step
        state = self._inner_step(state, action)

        # Conditionally do the outer step
        state = jax.lax.cond(
            state.inner_step == self.voting_nodes,
            lambda s: self._outer_step(s),
            lambda s: s,
            state,
        )

        # Conditionally pick -1.0 vs. computed reward
        reward = jax.lax.select(
            terminated,
            -1.0,
            self._compute_reward(state, weights),
        )

        # Done flag and infos
        done  = self._compute_done(state)
        infos = self._get_infos(state)

        return state, reward, done, infos


    @partial(jax.jit, static_argnums=[0])
    def sample_legal_action(self, state: State, key):
        """
        Sample a legal action in the environment.

        Params:
            state: The current state of the environment.

        Returns:
            A legal action.
        """
        legal_mask = self._compute_legal_actions(state)
        chosen = jax.random.choice(key, legal_mask)

        # Return as a one-hot vector
        return jnp.eye(legal_mask.shape[0])[chosen.astype(int)]

