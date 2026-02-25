from functools import partial
from typing import Any

import distrax
import jax
import jax.numpy as jnp
import jraph
from gymnax.environments import environment, spaces
from jax.random import gumbel
from jraph import GraphsTuple

from rl_blockchain.BlockEnv.BlockchainGraph import create_jraph_from_adj_matrix_fast, STATIC_MASKS_DICT, \
    create_rd_adj_matrix
from rl_blockchain.BlockEnv.rewards import weighted_rewards, null_reward
from rl_blockchain.BlockEnv.state_params import EnvState, EnvParams, get_stake_distribution, \
    preprocessing_validator_distribution, StaticEnvParams


def action_to_selected_node(action: int | jax.Array) -> jax.Array:
    return jnp.array(action - 1, dtype=jnp.int32)


def selected_node_to_action(selected_node: int | jax.Array) -> jax.Array:
    return jnp.array(selected_node + 1, dtype=jnp.int32)


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

        # Add the features to the graph
        graph_with_features = graph._replace(
            nodes=sample_features,
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
        super().__init__()

        self._first_params = default_first_params
        self._static_params = static_params

    @property
    def nb_nodes(self) -> int:
        """Number of nodes in the environment."""
        return self._static_params.nb_nodes

    @property
    def default_params(self) -> EnvParams:
        return self._first_params

    @property
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
            "global_step": spaces.Discrete(params.max_steps_in_episode.item()),
        })

    def get_obs(self, state: EnvState, params: EnvParams = None, key=None) -> jraph.GraphsTuple:
        """Applies observation function to state."""
        stake_distribution_abs = get_stake_distribution(state)
        stake_distribution_relative = stake_distribution_abs / self._static_params.horizon / self._static_params.nb_nodes
        preprocessed_stake_distribution = preprocessing_validator_distribution(
            stake_distribution_relative, self._static_params.box_clip)
        global_features = jnp.array([state.nb_val], dtype=jnp.float32)

        obs_graph = params.network_graph._replace(nodes=preprocessed_stake_distribution, globals=global_features)
        return obs_graph

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.array(done_steps)

    def step_env(self, key: jax.Array, state: EnvState, action: jax.Array, params: EnvParams) -> tuple[
        GraphsTuple, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """
        The params parameter define the status of the next obs, not the current one
        """
        new_state = EnvState.next_state(state, self._static_params.next_nb_val_fn(state, key), action)

        new_obs = self.get_obs(new_state, params)
        is_illegal_action = action.sum() != state.nb_val
        done = jnp.logical_or(self.is_terminal(new_state, params), is_illegal_action)

        operand_reward = (action, new_state, params, self._static_params)
        reward, info = jax.lax.cond(
            is_illegal_action,
            lambda tup: null_reward(),
            lambda tup: weighted_rewards(tup[0], tup[1], tup[2], tup[3]),
            operand_reward
        )
        reward_multiplied = reward

        infos_2 = dict(**info, nb_validators=state.nb_val)

        return (
            jax.lax.stop_gradient(new_obs),
            jax.lax.stop_gradient(new_state),
            jnp.array(reward_multiplied),
            done,
            infos_2,
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
        obs = obs_re._replace(nodes=jax.lax.select(done, obs_re.nodes, obs_st.nodes),
                              globals=jax.lax.select(done, obs_re.globals, obs_st.globals))

        return obs, state, reward, done, info

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[GraphsTuple, EnvState]:
        state = EnvState.create_init_state(key, self._static_params)
        obs = self.get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnames=('self',))
    def sample_legal_action(self, key: jax.Array) -> jax.Array:
        """
        Sample a legal action in the environment.

        Params:
            state: The current state of the environment.
            key: JAX random key for reproducibility.

        Returns:
            A legal action.
        """

        random_nb_val = self._static_params.init_nb_val_fn(key)

        return uniform_k_true_mask(key, self._static_params.nb_nodes, random_nb_val)


@partial(jax.jit, static_argnames=('N',))
def uniform_k_true_mask(key, N, k):
    """
    Returns:
        (N,) bool mask with exactly k True, uniformly sampled.
    """
    perm = jax.random.permutation(key, N)  # permutation uniforme
    return (perm < k).astype(jnp.float32)


def _pl_gumbel_permutation(key, logits):
    """
    logits: (n,) vrais logits du modèle
    returns: (n,) permutation pondérée (Plackett–Luce)
    """
    n = logits.shape[0]
    g = gumbel(key, shape=(n,), dtype=logits.dtype)
    scores = logits + g
    return jnp.argsort(scores)[::-1]  # ordre décroissant


def _mask_k_first_from_perm(perm, k):
    """
    perm: (n,)
    k: scalar int, dynamic
    returns mask (n,) where True iff index is in first k of perm
    """
    positions = jnp.argsort(perm)  # inverse permutation: positions[i] = rank of i
    return positions < k


def logp_prefix_pl(probs, perm, k):
    """
    logits: (n,)
    perm: (n,) permutation sampled from PL (e.g. by pl_gumbel_permutation)
    k: scalar int, dynamic
    returns scalar log-probability of the first k items of perm under PL(logits)
    """
    # Use scaled positive weights (scale cancels in ratios)
    W0 = 1
    eps = 1e-12

    def step(W, t):
        idx = perm[t]
        x = probs[idx]
        logp_t = jnp.log(jnp.clip(x, eps)) - jnp.log(jnp.clip(W, eps))
        return W - x, logp_t

    _, logp_terms = jax.lax.scan(step, W0, jnp.arange(probs.shape[0]))
    # Sum only first k terms without dynamic slicing
    t = jnp.arange(probs.shape[0])
    return jnp.sum(jnp.where(t < k, logp_terms, 0.0))


@jax.jit
def sample_subset_with_logp(key: jax.Array, distrib: distrax.Categorical, k: int | jax.Array) \
        -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sample a subset of k items from n with Plackett-Luce model defined by logits.
    Returns:
        perm: (n,) permutation sampled from PL(logits)
        mask: (n,) boolean mask of selected items
        logp: scalar log-probability of the selected subset under PL(logits)
    """
    perm = _pl_gumbel_permutation(key, distrib.logits)
    mask = _mask_k_first_from_perm(perm, k)
    logp = logp_prefix_pl(distrib.probs, perm, k)
    return perm, mask, logp


@jax.jit
def mode_subset(distrib: distrax.Categorical, k: int | jax.Array) -> jax.Array:
    """
    Get the mode subset of k items from n with Plackett-Luce model defined by logits.
    Returns:
        mask: (n,) boolean mask of selected items
    """
    sorted_indices = jnp.argsort(distrib.logits)[::-1]  # descending order
    mask = _mask_k_first_from_perm(sorted_indices, k)
    return mask
