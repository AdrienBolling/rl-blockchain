from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jraph
from gymnax.environments import environment, spaces
from jraph import GraphsTuple

from rl_blockchain.BlockEnv.BlockchainGraph import create_jraph_from_adj_matrix_fast, STATIC_MASKS_DICT, \
    create_rd_adj_matrix
from rl_blockchain.BlockEnv.rewards import weighted_rewards
from rl_blockchain.BlockEnv.state_params import EnvState, EnvParams, get_stake_distribution, \
    preprocessing_validator_distribution, StaticEnvParams


def action_to_selected_node(action: int) -> int:
    return action - 1


def selected_node_to_action(selected_node: int) -> int:
    return selected_node - 1


@partial(jax.jit, static_argnames=('nb_nodes',))
def _generate_random_chosen_nodes(nb_nodes: int, nb_val: jax.Array, key: jax.Array) -> jax.Array:
    key1, key2 = jax.random.split(key)
    init = jax.random.bernoulli(key1, p=0.5, shape=(nb_nodes,))
    not_selected = jax.random.permutation(key2, nb_nodes) < nb_val
    only_false = jnp.zeros(nb_nodes, dtype=bool)
    return jax.lax.select(
        not_selected,
        init,
        only_false
    )


class JraphSpace(spaces.Space):
    """A space for jraph.GraphsTuple."""
    """
    This class represent the observation space of the environment
    The observation is a graph with a dictionary of features
    """

    def __init__(self, features: spaces.Box, nb_nodes: int):
        super().__init__()
        self.features = spaces.Box(features.low, features.high, (nb_nodes,) + features.shape, features.dtype)
        self.boolean_features = spaces.Box(0, 1, (nb_nodes,), jnp.bool)
        self.validator_features = spaces.Discrete(nb_nodes)
        self.nb_nodes = nb_nodes

    def sample(self, key: jax.Array) -> jraph.GraphsTuple:
        """Sample a random graph from the space."""
        feature_key, validator_key, chosen_node_key, edges_key = jax.random.split(key, 4)
        sample_features = self.features.sample(feature_key)
        adj_matrix = create_rd_adj_matrix(self.nb_nodes, edges_key)
        graph: jraph.GraphsTuple = create_jraph_from_adj_matrix_fast(adj_matrix, STATIC_MASKS_DICT[self.nb_nodes])
        sample_nb_val = self.validator_features.sample(validator_key)

        bool_vector = _generate_random_chosen_nodes(self.nb_nodes, sample_nb_val, chosen_node_key)
        vector_chosen = bool_vector[:, None].astype(self.features.dtype)
        features_with_chosen = jnp.concat([vector_chosen, sample_features], axis=1)

        # Add the features to the graph
        graph_with_features = graph._replace(
            nodes=features_with_chosen,
            globals=jnp.array([sample_nb_val]),
        )
        return graph_with_features

    def contains(self, graph: jraph.GraphsTuple) -> bool:
        """Check whether the object is a jraph.GraphsTuple."""
        if not isinstance(graph, jraph.GraphsTuple):
            return False
        if graph.n_node.shape[0] != 1 or graph.n_edge.shape[0] != 1:
            return False
        if graph.nodes.shape[0] != self.nb_nodes:
            return False
        if graph.edges.shape[0] != self.nb_nodes * (self.nb_nodes - 1):
            return False
        if graph.edges.max().item() > 1 or graph.edges.min().item() < 0:
            return False
        chosen_nodes = graph.nodes[:, 0].astype(jnp.bool)
        features = graph.nodes[:, 1:]
        if not self.features.contains(features):
            return False
        if sum(chosen_nodes) > graph.globals.item():
            return False
        return True


class BlockchainEnv(environment.Environment[EnvState, EnvParams]):
    """
    A single-agent fully implemented environment for the blockchain. (No dependencies on any other class, to be able to
    use graph observations)

    This environment has intermediary steps, the agent will choose 1 voting node at each sub-timestep
    """

    def __init__(self, default_first_params: EnvParams, static_params: StaticEnvParams):
        # def __init__(self, network_graph: jnp.ndarray, nb_validators: int = None, max_edge_weights: float = None,
        #              filename: str = "grid_samples/grid_25.csv", weight_reward=None):
        super().__init__()

        self._first_params = default_first_params
        self._static_params = static_params

        # if weight_reward is None:
        #     weight_reward = [1, 1]
        # self.first_reward = weight_reward
        # self.first_network_adj_mat = network_graph
        #                                                                                      network_graph.max().item())
        # self.filename_min_max_dist = filename
        # self.nb_node = network_graph.shape[0]

        # if nb_validators is None:
        #     nb_validators = self.nb_node // 3
        # self._first_nb_validators = nb_validators

    @property
    def nb_nodes(self) -> int:
        """Number of nodes in the environment."""
        return self._static_params.nb_nodes

    @property
    def default_params(self) -> EnvParams:
        return self._first_params

    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1 + self._static_params.nb_nodes

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return spaces.Discrete(1 + self._static_params.nb_nodes)

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        node_feature = spaces.Box(
            low=jnp.array([-self._static_params.box_clip]),
            high=jnp.array([self._static_params.box_clip]),
            shape=(1,),
            dtype=jnp.float32
        )

        return JraphSpace(
            features=node_feature,
            nb_nodes=self._static_params.nb_nodes,
        )

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Dict({
            "ring_history": spaces.Box(
                low=0, high=1, shape=(self._static_params.horizon, self._static_params.nb_nodes), dtype=jnp.bool_
            ),
            "chosen_nodes": spaces.Box(
                low=0, high=1, shape=(self._static_params.nb_nodes,), dtype=jnp.bool),
            "inner_step": spaces.Discrete(self._static_params.nb_nodes + 1),  # +1 for the global step
            "global_step": spaces.Discrete(params.max_steps_in_episode),
        })

    def get_obs(self, state: EnvState, params: EnvParams = None, key=None) -> jraph.GraphsTuple:
        """Applies observation function to state."""
        stake_distribution_abs = get_stake_distribution(state)
        stake_distribution_relative = stake_distribution_abs / self._static_params.horizon / self._static_params.nb_nodes
        preprocessed_stake_distribution = preprocessing_validator_distribution(
            stake_distribution_relative, self._static_params.box_clip)
        node_features = jnp.column_stack((state.chosen_nodes, preprocessed_stake_distribution))
        global_features = jnp.array([params.nb_validators])

        obs_graph = params.network_graph._replace(nodes=node_features, globals=global_features)
        return obs_graph

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.array(done_steps)

    def step_env(self, key: jax.Array, state: EnvState, action: int | float | jax.Array, params: EnvParams) -> tuple[
        GraphsTuple, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        selected_node = action_to_selected_node(action)

        is_inner = jnp.array(selected_node != -1)

        operand_state = (state, selected_node)
        new_state = jax.lax.cond(is_inner,
                                 lambda tup: EnvState.next_state_inner(tup[0], tup[1]),
                                 lambda tup: EnvState.next_state_global(tup[0]),
                                 operand_state
                                 )

        new_obs = self.get_obs(new_state, params)
        mask = compute_legal_actions(new_obs)
        is_illegal_action = jnp.logical_not(mask[action])
        done = jnp.logical_or(self.is_terminal(state, params), is_illegal_action)

        operand_reward = (new_state, params, self._static_params)
        reward = jax.lax.cond(
            jnp.logical_or(is_illegal_action, is_inner),
            lambda tup: jnp.array(0.0, dtype=jnp.float32),
            lambda tup: weighted_rewards(tup[0], tup[1], tup[2]),
            operand_reward
        )

        return (
            jax.lax.stop_gradient(new_obs),
            jax.lax.stop_gradient(new_state),
            jnp.array(reward),
            done,
            {"action_taken": selected_node},
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(
            self,
            key: jax.Array,
            state: EnvState,
            action: int | float | jax.Array,
            params: EnvParams | None = None,
    ) -> tuple[GraphsTuple, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = obs_re._replace(nodes=jax.lax.select(done, obs_re.nodes, obs_st.nodes))

        return obs, state, reward, done, info

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[GraphsTuple, EnvState]:
        state = EnvState.create_init_state(self._static_params.nb_nodes, self._static_params.horizon)
        obs = self.get_obs(state, params)
        return obs, state


@jax.jit
def compute_legal_actions(obs: GraphsTuple) -> jnp.ndarray:
    chosen_nodes = obs.nodes[:, 0]
    nb_validators = obs.globals[0]

    current_nb_val = jnp.sum(chosen_nodes)
    available = jnp.logical_not(chosen_nodes)
    nb_nodes = chosen_nodes.shape[0]

    not_enough_validators = jnp.zeros((nb_nodes + 1,), dtype=bool)
    too_much_validators = jnp.zeros((nb_nodes + 1,), dtype=bool)

    not_enough_validators = not_enough_validators.at[1:].set(available)
    not_enough_validators = not_enough_validators.at[0].set(False)

    too_much_validators = too_much_validators.at[0].set(True)

    return jax.lax.select(
        current_nb_val < nb_validators,
        not_enough_validators,
        too_much_validators
    )
