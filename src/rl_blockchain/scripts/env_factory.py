import abc
from functools import partial
from typing import Callable, Dict, Tuple, Any, List

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, TEnvParams

from rl_blockchain import BlockEnv
from rl_blockchain.BlockEnv import StaticEnvParams, BlockchainEnv
from rl_blockchain.BlockEnv.BlockchainGraph import make_rd_closed_adj_matrix, import_positions_from_file, \
    make_adj_matrix_from_positions
from rl_blockchain.BlockEnv.state_params import Next_nb_val_fn, white_param_fn, EnvState, init_random_nb_val_factory
from rl_blockchain.model import CategoricalSeparateMLP, PPOSeparate
from rl_blockchain.scripts.parser import REF_FILENAME, UpdateParams

# Type alias
LOG_TYPE = Callable[[dict[str, jax.Array], jax.Array, jax.Array], dict[str, jax.Array]]
EnvInitOutput = Tuple[nn.Module, Environment, TEnvParams, Callable[[jax.Array], TEnvParams], LOG_TYPE]


def return_update_params_fn(update_mode: UpdateParams, nb_val: int, nb_node: int) -> Next_nb_val_fn:
    if update_mode == UpdateParams.NO_UPDATE:
        return white_param_fn
    if update_mode == UpdateParams.THRESHOLD_UPDATE:
        return change_val_param_fn
    if update_mode == UpdateParams.ORN_UHL_UPDATE:
        mu = jnp.float32(nb_node // 4 if nb_val == 0 else nb_val)
        return partial(change_val_orn_uhl_fn, mu=mu)

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
    infos_keys = ["gini", "distance", "gini_reward", "distance_reward", "weighted_reward", "nb_validators"]
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

        next_val_type = UpdateParams.NO_UPDATE if "next_val_type" not in config else config["next_val_type"]
        nb_nodes = config["n_nodes"]

        init_nb_val_fct = init_random_nb_val_factory(nb_nodes) if config["voting_nodes"] == 0 else config[
            "voting_nodes"]
        next_val_fct = return_update_params_fn(next_val_type, 0, nb_nodes)

        def create_params_fn(key: jax.Array) -> BlockEnv.EnvParams:

            return jax.lax.stop_gradient(BlockEnv.EnvParams.create_random(
                config["n_nodes"],
                key,
                config["voting_nodes"],
                config["reward_weights"],
                init_nb_val_fct,
                next_val_fct
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

        next_val_type = UpdateParams.NO_UPDATE if "next_val_type" not in config else config["next_val_type"]
        nb_nodes = positions.shape[0]

        init_nb_val_fct = init_random_nb_val_factory(nb_nodes) if config["voting_nodes"] == 0 else config[
            "voting_nodes"]
        next_val_fct = return_update_params_fn(next_val_type, 0, nb_nodes)

        def create_params_fn(key: jax.Array) -> BlockEnv.EnvParams:
            key_mat, key_create = jax.random.split(key)
            new_adj_mat = make_rd_closed_adj_matrix(positions, key_mat, 0.05)
            # TODO need init et next_va_ fct
            return jax.lax.stop_gradient(
                BlockEnv.EnvParams.create(new_adj_mat, init_nb_val_fct, next_val_fct, config["reward_weights"]))

        int_adj_mat = make_adj_matrix_from_positions(positions)
        env_params = BlockEnv.EnvParams.create(int_adj_mat, init_nb_val_fct, next_val_type, config["reward_weights"])
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
