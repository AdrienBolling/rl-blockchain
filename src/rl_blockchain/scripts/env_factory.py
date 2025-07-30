import abc
from typing import Callable, Dict, Tuple, Any, List

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import wandb
from gymnax.environments.environment import Environment, TEnvParams

from rl_blockchain import BlockEnv
from rl_blockchain.BlockEnv import StaticEnvParams, BlockchainEnv
from rl_blockchain.model import CategoricalSeparateMLP, PPO_NET_GAT
from rl_blockchain.scripts.parser import REF_FILENAME

# Type alias
LOG_TYPE = Callable[[dict[str, jax.Array], jax.Array, jax.Array], dict[str, jax.Array]]
EnvInitOutput = Tuple[nn.Module, Environment, TEnvParams, Callable[[jax.Array], TEnvParams], LOG_TYPE]


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
    infos_keys = ["gini", "distance", "gini_reward", "distance_reward"]
    list_is_inner: jax.Array = infos["action_taken"] == -1
    sum_inner = list_is_inner.sum()
    returned_infos = {}
    for key in infos_keys:
        returned_infos[key] = ((infos[key] * list_is_inner).sum() / sum_inner).item()
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
        gat1_out, gat2_out, gat2_nodes_out = config["gat_arch"]

        def create_params_fn(key: jax.Array) -> BlockEnv.EnvParams:
            return BlockEnv.EnvParams.create_random(
                config["n_nodes"],
                key,
                config["voting_nodes"],
                config["reward_weights"]
            )

        env_params = create_params_fn(key_param)
        static_params = StaticEnvParams.create(config["n_nodes"], REF_FILENAME[config["n_nodes"]])
        env = BlockchainEnv(env_params, static_params)
        model = PPO_NET_GAT(gat1_out, gat2_out, gat2_nodes_out, env.num_actions)

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
        create_params_fn = lambda key: env_params

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
GenericEnvFactory.register("cartpole", CartPoleEnvBuilder())
