import abc
from typing import Callable, Dict, Tuple, Any, List

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, TEnvParams

from rl_blockchain import BlockEnv
from rl_blockchain.BlockEnv import StaticEnvParams, BlockchainEnv, EnvParams
from rl_blockchain.BlockEnv.BlockchainGraph import make_rd_closed_adj_matrix, import_positions_from_file, \
    make_adj_matrix_from_positions
from rl_blockchain.model import CategoricalSeparateMLP, PPOSeparate
from rl_blockchain.scripts.parser import REF_FILENAME

# Type alias
LOG_TYPE = Callable[[dict[str, jax.Array], jax.Array, jax.Array], dict[str, jax.Array]]
Outer_param_fn = Callable[[TEnvParams, jax.Array, jax.Array], TEnvParams]
EnvInitOutput = Tuple[nn.Module, Environment, TEnvParams, Callable[[jax.Array], TEnvParams], LOG_TYPE]


@jax.jit
def white_param_fn(prev_param: TEnvParams, key: jax.Array, action: jax.Array) -> TEnvParams:
    return prev_param


@jax.jit
def change_val_param_fn(prev_param: EnvParams, key: jax.Array, action: jax.Array) -> EnvParams:
    return jax.lax.cond(action == 0, _sub_change_val_fn, _sub_white_param_fn, prev_param, key)


def _sub_white_param_fn(prev_param: TEnvParams, key: jax.Array) -> TEnvParams:
    return prev_param


def _sub_change_val_fn(prev_param: EnvParams, key: jax.Array) -> EnvParams:
    key_thresh, key_gen = jax.random.split(key)

    # threshold
    cond = jax.random.uniform(key_thresh) > 0.8

    # random integer in [4, n_nodes)
    max_n = prev_param.network_graph.n_node[0]
    sampled = jax.random.randint(key_gen, (), minval=4, maxval=max_n)

    # cond-select instead of Python
    new_val = jnp.where(cond, sampled, prev_param.nb_validators)

    return EnvParams(
        network_graph=prev_param.network_graph,
        adj_matrix=prev_param.adj_matrix,
        nb_validators=new_val,
        rewards_weights=prev_param.rewards_weights,
        max_steps_in_episode=prev_param.max_steps_in_episode,
        max_outer_steps_in_episode=prev_param.max_outer_steps_in_episode,
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
    infos_keys = ["gini", "distance", "gini_reward", "distance_reward", "weighted_reward", "nb_validators"]
    list_is_inner: jax.Array = infos["action_taken"] == -1
    sum_inner = list_is_inner.sum()
    returned_infos = {}
    for key in infos_keys:
        returned_infos[key] = ((infos[key] * list_is_inner).sum() / sum_inner).item()

    returned_infos["nb_validators_std"] = compute_std_validators(infos).item()
    return returned_infos


def compute_std_validators(infos: dict[str, jax.Array]) -> jax.Array:
    x = infos["nb_validators"]
    list_is_inner: jax.Array = infos["action_taken"] == -1
    sum_inner = list_is_inner.sum()
    masked_mean = (x * list_is_inner).sum() / sum_inner
    diff = x - masked_mean
    var = (list_is_inner * diff * diff).sum() / sum_inner
    std = jnp.sqrt(var)
    return std


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

        def create_params_fn(key: jax.Array) -> BlockEnv.EnvParams:
            return jax.lax.stop_gradient(BlockEnv.EnvParams.create_random(
                config["n_nodes"],
                key,
                config["voting_nodes"],
                config["reward_weights"]
            ))

        env_params = create_params_fn(key_param)
        static_params = StaticEnvParams.create(config["n_nodes"], REF_FILENAME[config["n_nodes"]])
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

        nb_nodes = positions.shape[0]

        def create_params_fn(key: jax.Array) -> BlockEnv.EnvParams:
            key_mat, key_create = jax.random.split(key)
            new_adj_mat = make_rd_closed_adj_matrix(positions, key_mat, 0.05)
            return jax.lax.stop_gradient(
                BlockEnv.EnvParams.create(new_adj_mat, config["voting_nodes"], key_create, config["reward_weights"]))

        int_adj_mat = make_adj_matrix_from_positions(positions)
        env_params = BlockEnv.EnvParams.create(int_adj_mat, config["voting_nodes"], key_param, config["reward_weights"])
        static_params = StaticEnvParams.create(nb_nodes, REF_FILENAME[nb_nodes])
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
