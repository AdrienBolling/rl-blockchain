import abc
import logging
from functools import partial
from typing import Callable, Dict, Tuple, Any, List

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import jraph
from gymnax.environments.environment import Environment, TEnvParams

from rl_blockchain import BlockEnv
from rl_blockchain.BlockEnv import StaticEnvParams, BlockchainEnv, create_rd_adj_matrix
from rl_blockchain.BlockEnv.BlockchainGraph import import_positions_from_file, \
    make_adj_matrix_from_positions, get_non_diag_indices
from rl_blockchain.BlockEnv.state_params import Next_nb_val_fn, white_param_fn, EnvState, init_random_nb_val_factory, \
    EnvParams, Next_map_fn, init_fixed_nb_val_factory, Speeders
from rl_blockchain.model import CategoricalSeparateMLP, PPOSeparate
from rl_blockchain.scripts.parser import REF_FILENAME, UpdateValStrat, UpdateDistStrat

logger = logging.getLogger(__name__)

# Type alias
LOG_TYPE = Callable[[dict[str, jax.Array], jax.Array, jax.Array], dict[str, jax.Array]]
EnvInitOutput = Tuple[nn.Module, Environment, TEnvParams, Callable[[jax.Array], TEnvParams], LOG_TYPE]


def return_update_val_fn(update_mode: UpdateValStrat, nb_val: int, nb_node: int) -> Next_nb_val_fn:
    logger.debug("validator update strategy: %s", update_mode.name)
    if update_mode == UpdateValStrat.NO_UPDATE:
        return white_param_fn
    if update_mode == UpdateValStrat.THRESHOLD_UPDATE:
        return change_val_param_fn
    if update_mode == UpdateValStrat.ORN_UHL_UPDATE:
        mu = jnp.float32(nb_node // 4 if nb_val == 0 else nb_val)
        return partial(change_val_orn_uhl_fn, mu=mu)

    raise ValueError(f"Unsupported update mode: {update_mode}")


def return_update_maps_fn(update_mode: UpdateDistStrat) -> Next_map_fn:
    logger.debug("edge update strategy: %s", update_mode.name)
    if update_mode == UpdateDistStrat.NO_UPDATE:
        return next_edge_white
    if update_mode == UpdateDistStrat.ORN_UHL_UPDATE:
        return next_edge_orn_uhl
    raise ValueError(f"Unsupported update mode: {update_mode}")


@jax.jit
def change_val_param_fn(prev_state: EnvState, key: jax.Array) -> jax.Array:
    key_thresh, key_gen = jax.random.split(key)

    # threshold
    cond = jax.random.uniform(key_thresh) > 0.8

    # random integer in [4, n_nodes)
    max_n = prev_state.ring_history.shape[1]
    sampled = jax.random.randint(key_gen, (), minval=4, maxval=max_n)

    # cond-select instead of Python
    new_val = jnp.where(cond, sampled, prev_state.nb_val)

    return new_val


@jax.jit
def random_validator_ornstein_uhlenbeck(
        key: jax.Array,
        current_k: jax.Array,  # int32 scalar
        mu: jax.Array,  # float32 scalar
        max_nb_val: jax.Array,  # int32 scalar
        sigma: jax.Array = jnp.float32(0.10),
        alpha: jax.Array = jnp.float32(0.05),
) -> jax.Array:
    current_k = jnp.asarray(current_k, dtype=jnp.int32)
    max_nb_val = jnp.asarray(max_nb_val, dtype=jnp.int32)
    mu = jnp.asarray(mu, dtype=jnp.float32)

    min_k_f = jnp.float32(4.0)

    eps = jax.random.normal(key, ()) * sigma
    k_f = current_k.astype(jnp.float32)

    k_next = k_f + alpha * (mu - k_f) + k_f * eps
    k_next = jnp.clip(k_next, min_k_f, max_nb_val.astype(jnp.float32))
    return jnp.rint(k_next).astype(jnp.int32)


@jax.jit
def generate_unique_inverse(graph: jraph.GraphsTuple):
    senders = graph.senders
    receivers = graph.receivers
    n = graph.n_node[0]

    i = jnp.minimum(senders, receivers)
    j = jnp.maximum(senders, receivers)
    pair_id = i * n + j

    _, unique, inverse = jnp.unique(
        pair_id,
        return_inverse=True,
        return_index=True
    )
    return unique, inverse


@jax.jit
def _random_distance_ornstein_uhlenbeck(
        key: jax.Array,
        current_distance: jax.Array,
        target: jax.Array,
        sigma: jax.Array,
        alpha: jax.Array = jnp.float32(0.05),
) -> jax.Array:
    nb_link = current_distance.shape[0]
    eps = jax.random.normal(key, (nb_link,)) * sigma
    distance_next = current_distance + alpha * (target - current_distance) + current_distance * eps
    distance_next = jnp.clip(distance_next, 0)
    distance_next = distance_next / distance_next.max()  # Normalize
    return distance_next


@jax.jit
def next_edge_orn_uhl(key: jax.Array, state: EnvState, params: EnvParams) -> jax.Array:
    return _random_distance_ornstein_uhlenbeck(key, state.current_edges_unique, params.edge_config.edges_unique,
                                               params.edge_config.sigma)


@jax.jit
def next_edge_white(key: jax.Array, state: EnvState, params: EnvParams):
    return params.edge_config.edges_unique


def next_edge_full_rd(key: jax.Array, state: EnvState, params: EnvParams):
    nb_nodes = state.ring_history.shape[1]
    adj_mat = create_rd_adj_matrix(nb_nodes, key)
    # TODO ?
    return


def next_distance_orn_uhl_temp(key: jax.Array, graph: jraph.GraphsTuple, target: jax.Array, sigma: jax.Array,
                               speeders) -> jraph.GraphsTuple:
    list_distance = graph.edges
    unique_distances = list_distance[speeders[0]]
    new_single_distance = _random_distance_ornstein_uhlenbeck(key, unique_distances, target, sigma)
    new_distances = new_single_distance[speeders[1]]
    return graph._replace(edges=new_distances)


# ---------- param update fn ----------

@jax.jit
def change_val_orn_uhl_fn(
        prev_state: EnvState,
        key: jax.Array,
        mu: jax.Array,  # float32 scalar
) -> jax.Array:
    _, key_gen = jax.random.split(key)

    max_n = prev_state.ring_history.shape[1]

    return random_validator_ornstein_uhlenbeck(
        key_gen,
        prev_state.nb_val,
        mu,
        max_n,
    )


class EnvBuilder(abc.ABC):
    """
    Abstract interface for environment builders.
    """

    required_keys: List[str] = []

    @abc.abstractmethod
    def build(self, key_param: jax.Array, config: Dict[str, Any]) -> EnvInitOutput:
        """
        Build environment using a configuration dictionary.
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate that all required keys are present in the config.
        """
        missing = [key for key in self.required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing configuration keys: {missing}")

    def get_keys(self) -> List[str]:
        """
        Get the required configuration keys for this environment builder.
        """
        return self.required_keys

    @staticmethod
    @abc.abstractmethod
    def log(infos_env: dict[str, jax.Array], rews: jax.Array, dones: jax.Array) -> dict[str, jax.Array]:
        """
        Log environment-specific information.
        """
        pass


def compute_avg_value(infos: dict[str, jax.Array]) -> dict[str, jax.Array]:
    infos_keys = ["gini", "distance", "gini_reward", "fairness_reward", "distance_reward",
                  "weighted_reward", "weighted_original_reward", "nb_validators"]
    returned_infos = {}
    for key in infos_keys:
        returned_infos[key] = infos[key].mean()
    returned_infos["nb_validators_std"] = infos["nb_validators"].std()
    return returned_infos


class BlockchainEnvBuilder(EnvBuilder):
    required_keys = ["n_nodes", "gat_arch", "voting_nodes", "reward_weights"]

    @staticmethod
    def log(infos_env: dict[str, jax.Array], rews: jax.Array, dones: jax.Array) -> dict[str, jax.Array]:
        infos_env_refined = compute_avg_value(infos_env)
        total_rewards = rews.sum()
        total_dones = dones.sum()
        infos_env_refined["avg_rewards"] = jnp.mean(rews)
        infos_env_refined["nb_sequences_done"] = total_dones
        infos_env_refined["reward_mean_per_episode"] = total_rewards / total_dones if total_dones > 0 else total_rewards
        return infos_env_refined

    def build(self, key_param: jax.Array, config: Dict[str, Any]) -> EnvInitOutput:
        self.validate_config(config)
        backbone_gat_dim, actor_gcn_dim, critic_gnn_dim = config["gat_arch"]

        next_val_type = UpdateValStrat.NO_UPDATE if "next_val_type" not in config else config["next_val_type"]
        nb_nodes = config["n_nodes"]

        init_nb_val_fct = init_random_nb_val_factory(nb_nodes) if config["voting_nodes"] == 0 else \
            init_fixed_nb_val_factory(config["voting_nodes"])
        next_val_fct = return_update_val_fn(next_val_type, 0, nb_nodes)
        non_diag_mask = get_non_diag_indices(nb_nodes)
        speeder = Speeders.create(nb_nodes)

        @jax.jit
        def create_params_fn(key: jax.Array) -> BlockEnv.EnvParams:
            rd_adj_mat = create_rd_adj_matrix(nb_nodes, key)
            adj_mat_uniq = rd_adj_mat.flatten().take(non_diag_mask).take(speeder.unique)
            list_sigma = jax.random.uniform(key, (adj_mat_uniq.shape[0],), minval=0, maxval=0.01)  # TODO make the max val change

            return jax.lax.stop_gradient(BlockEnv.EnvParams.create(
                adj_mat_uniq,
                list_sigma,
                rewards_weights=config["reward_weights"],
            ))

        env_params = create_params_fn(key_param)
        next_edge_type = UpdateDistStrat.NO_UPDATE if "next_edge_type" not in config else config["next_edge_type"]
        next_edge_fn = return_update_maps_fn(next_edge_type)
        static_params = StaticEnvParams.create(config["n_nodes"], REF_FILENAME[config["n_nodes"]], init_nb_val_fct,
                                               next_val_fct, next_edge_fn,
                                               horizon=config.get("horizon", 200),
                                               gini_reward_mode=config.get("gini_reward_mode", "rank"),
                                               gamma=config.get("gamma", 0.99))
        env = BlockchainEnv(env_params, static_params)
        # model = PPO_NET_GAT(gat1_out, gat2_out, gat2_nodes_out, env.num_actions)
        model = PPOSeparate(env.num_actions, backbone_gat_dim, actor_gcn_dim, critic_gnn_dim)

        return model, env, env_params, create_params_fn, self.__class__.log


class BlockchainEnvCloseMapBuilder(BlockchainEnvBuilder):
    required_keys = ["gat_arch", "voting_nodes", "reward_weights", "ref_map_file"]

    def build(self, key_param: jax.Array, config: Dict[str, Any]) -> EnvInitOutput:
        self.validate_config(config)
        backbone_gat_dim, actor_gcn_dim, critic_gnn_dim = config["gat_arch"]

        positions = import_positions_from_file(config["ref_map_file"])

        next_val_type = UpdateValStrat.NO_UPDATE if "next_val_type" not in config else config["next_val_type"]
        next_edge_type = UpdateDistStrat.NO_UPDATE if "next_edge_type" not in config else config["next_edge_type"]
        nb_nodes = positions.shape[0]

        init_nb_val_fct = init_random_nb_val_factory(nb_nodes) if config["voting_nodes"] == 0 else \
            init_fixed_nb_val_factory(config["voting_nodes"])
        next_val_fct = return_update_val_fn(next_val_type, 0, nb_nodes)

        ref_adj_mat = make_adj_matrix_from_positions(positions)
        ref_adj_mat_uniq = ref_adj_mat.flatten().take(get_non_diag_indices(nb_nodes))
        list_sigma = jnp.ones((ref_adj_mat_uniq.shape[0],), dtype=jnp.float32) * 0.05

        @jax.jit
        def create_params_fn(key: jax.Array) -> BlockEnv.EnvParams:
            # key_mat, key_create = jax.random.split(key)
            # new_adj_mat = make_rd_closed_adj_matrix(positions, key_mat, 0.05)

            new_adj_mat_uniq = _random_distance_ornstein_uhlenbeck(
                key, ref_adj_mat_uniq, ref_adj_mat_uniq, list_sigma, 0)
            list_sigma_exp = jax.random.uniform(key, (nb_nodes,), minval=0, maxval=0.01)  # TODO make the max val change
            return jax.lax.stop_gradient(
                BlockEnv.EnvParams.create(new_adj_mat_uniq, list_sigma_exp, config["reward_weights"]))

        int_adj_mat = make_adj_matrix_from_positions(positions)
        env_params = BlockEnv.EnvParams.create(int_adj_mat, config["reward_weights"])
        next_edge_fn = return_update_maps_fn(next_edge_type)
        static_params = StaticEnvParams.create(nb_nodes, REF_FILENAME[nb_nodes], init_nb_val_fct, next_val_fct,
                                               next_map_fn=next_edge_fn, horizon=config.get("horizon", 200),
                                               gini_reward_mode=config.get("gini_reward_mode", "rank"),
                                               gamma=config.get("gamma", 0.99))
        env = BlockchainEnv(env_params, static_params)
        model = PPOSeparate(env.num_actions, backbone_gat_dim, actor_gcn_dim, critic_gnn_dim)

        return model, env, env_params, create_params_fn, self.__class__.log


class CartPoleEnvBuilder(EnvBuilder):

    @staticmethod
    def log(infos_env: dict[str, jax.Array], rews: jax.Array, dones: jax.Array) -> dict[str, jax.Array]:
        total_rewards = rews.sum()
        total_dones = dones.sum()
        reward_mean_per_episode = total_rewards / total_dones if total_dones > 0 else 0

        return {"reward_mean_per_episode": reward_mean_per_episode, "nb_sequences_done": jnp.sum(dones)}

    def build(self, key_param: jax.Array, config: Dict[str, Any]) -> EnvInitOutput:
        self.validate_config(config)
        env, env_params = gymnax.make("CartPole-v1")
        model = CategoricalSeparateMLP(env.num_actions, 64, 2)
        create_params_fn = jax.lax.stop_gradient(lambda key: env_params)

        return model, env, env_params, create_params_fn, self.__class__.log


class GenericEnvFactory:
    """
    Flexible, instance-based factory for RL environments.
    """

    _builders: Dict[str, EnvBuilder] = {}

    @classmethod
    def register(cls, name: str, builder: EnvBuilder) -> None:
        cls._builders[name] = builder

    @classmethod
    def create(cls, env_name: str, key_param: jax.Array, config: Dict[str, Any]) -> EnvInitOutput:
        if env_name not in cls._builders:
            raise ValueError(f"Unknown environment: {env_name}")
        return cls._builders[env_name].build(key_param, config)

    @classmethod
    def available_environments(cls):
        return list(cls._builders.keys())


# Registration
GenericEnvFactory.register("blockenv", BlockchainEnvBuilder())
GenericEnvFactory.register("blockenv_close_map", BlockchainEnvCloseMapBuilder())
GenericEnvFactory.register("cartpole", CartPoleEnvBuilder())
