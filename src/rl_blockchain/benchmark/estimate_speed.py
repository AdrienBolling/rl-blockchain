import argparse
import time
from functools import partial

import gymnax
import jax
from gymnax.environments.environment import Environment, TEnvParams

from rl_blockchain.BlockEnv import BlockchainEnv, EnvParams
from rl_blockchain.BlockEnv.BlockEnv import uniform_k_true_mask
from rl_blockchain.scripts.env_factory import GenericEnvFactory
from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate the speed of the environment")
    parser.add_argument("--step", type=int, default=100000, help="number of steps")
    parser.add_argument("--num-envs", type=int, default=1, help="number of environments")
    parser.add_argument("--nb-nodes", type=int, default=7, help="number of nodes")

    return parser.parse_args()


def create_config(key: jax.Array, val_strat: UpdateValStrat, dist_strat: UpdateDistStrat, args: argparse.Namespace):
    config = {"n_nodes": args.nb_nodes, "gat_arch": [4, 4, 4], "voting_nodes": args.nb_nodes,
              "reward_weights": [0.5, 0.5], "next_edge_type": dist_strat, "next_val_type": val_strat}
    _, env, _, create_params_fn, _ = GenericEnvFactory.create("blockenv", key, config)
    return env, create_params_fn


def create_config_cartpole():
    env, env_params = gymnax.make("CartPole-v1")
    env_params = jax.lax.stop_gradient(env_params)
    create_params_fn = lambda key: env_params
    return env, create_params_fn


@partial(jax.jit, static_argnames=('env', 'steps_in_episode'))
def rollout_cartpole(key_input, env_params_episode: TEnvParams, env: Environment, steps_in_episode: int):
    key_reset, key_episode = jax.random.split(key_input)
    _, first_state = env.reset(key_reset, env_params_episode)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        state, key = state_input
        next_key, key_step, key_net = jax.random.split(key, 3)
        action = env.action_space(env_params_episode).sample(key_net)

        _, next_state, r, _, _ = env.step(
            key_step, state, action, env_params_episode
        )

        carry = [next_state, next_key]
        return carry, r

    # Scan over episode step loop
    (_, _), trajs = jax.lax.scan(
        policy_step,
        [first_state, key_episode],
        None,
        steps_in_episode
    )
    return trajs.sum()


@partial(jax.jit, static_argnames=('env', 'steps_in_episode', 'N'))
def rollout(key_input, env_params_episode: EnvParams, env: BlockchainEnv, N: int, steps_in_episode: int):
    key_reset, key_episode = jax.random.split(key_input)
    _, first_state = env.reset(key_reset, env_params_episode)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        state, key = state_input
        next_key, key_step, key_net = jax.random.split(key, 3)
        action = uniform_k_true_mask(key_net, N, state.nb_val)

        _, next_state, r, _, _ = env.step(
            key_step, state, action, env_params_episode
        )

        carry = [next_state, next_key]
        return carry, r

    # Scan over episode step loop
    (_, _), trajs = jax.lax.scan(
        policy_step,
        [first_state, key_episode],
        None,
        steps_in_episode
    )
    return trajs.sum()


def main():
    key = jax.random.PRNGKey(0)
    args = create_arg_parser()
    nb_envs: int = args.num_envs
    num_steps: int = args.step
    nb_nodes: int = args.nb_nodes

    key_params, key_config, rollout_key = jax.random.split(key, 3)
    subkey_params = jax.random.split(key_params, nb_envs)
    subkeys = jax.random.split(rollout_key, nb_envs)

    # =============
    env_cartpole, create_params_fn_cartpole = create_config_cartpole()
    params_map = jax.vmap(
        lambda key_map: create_params_fn_cartpole(key_map)  # Create new params for each env,
    )
    params_list = params_map(subkey_params)

    @jax.jit
    def single_rollout(rng: jax.Array, new_param: EnvParams):
        return rollout_cartpole(
            rng,
            new_param,
            env_cartpole,
            num_steps,
        )

    vm_rollout = jax.jit(jax.vmap(single_rollout))
    vm_rollout(subkeys, params_list).block_until_ready()

    start = time.time()
    vm_rollout(subkeys, params_list).block_until_ready()
    end = time.time()
    print(
        f"Reference Cartpole : Time for {nb_envs} envs and {num_steps} steps: {end - start:.2f} seconds")

    # =============

    for val_strat in UpdateValStrat:
        for dist_strat in UpdateDistStrat:
            env, create_params_fn = create_config(key, val_strat, dist_strat, args)
            params_map = jax.vmap(
                lambda key_map: create_params_fn(key_map)  # Create new params for each env,
            )
            params_list = params_map(subkey_params)

            @jax.jit
            def single_rollout(rng: jax.Array, new_param: EnvParams):
                return rollout(
                    rng,
                    new_param,
                    env,
                    nb_nodes,
                    num_steps,
                )

            vm_rollout = jax.jit(jax.vmap(single_rollout))
            vm_rollout(subkeys, params_list).block_until_ready()

            start = time.time()
            vm_rollout(subkeys, params_list).block_until_ready()
            end = time.time()

            print(
                f"Val strat: {val_strat}, Dist strat: {dist_strat}, Time for {nb_envs} envs and {num_steps} steps: {end - start:.2f} seconds")


if __name__ == '__main__':
    main()
