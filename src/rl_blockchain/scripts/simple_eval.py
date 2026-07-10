import argparse
import pathlib

import jax
import matplotlib.pyplot as plt
import numpy as np

from rl_blockchain.BlockEnv.BlockEnv import mode_subset
from rl_blockchain.algo.ppo import create_ppo_state
from rl_blockchain.scripts.env_factory import GenericEnvFactory, return_update_params_fn
from rl_blockchain.scripts.parser import UpdateParams
import jax.numpy as jnp

config = {"gat_arch": [64, 32, 16], "voting_nodes": 3, "n_nodes": 25,
          "reward_weights": [0.1, 0.9]}


def create_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default is 0.",
    )
    parser.add_argument(
        "chkpt_dir",
        type=pathlib.Path,
        help="Directory to save checkpoints. Default is 'checkpoints' in the cwd.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes to run during evaluation. Default is 100.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = create_argparse()
    key = jax.random.PRNGKey(args.seed)
    key_eval, state_key, key_param = jax.random.split(key, 3)

    # model, env, create_params_fn, log_fn, update_params_fn = get_env_config(args, key_param)

    model, env, params, create_params_fn, log_fn = GenericEnvFactory.create("blockenv", key_param,
                                                                            config)
    update_params_fn = return_update_params_fn(UpdateParams.ORN_UHL_UPDATE)

    chkpt_dir: pathlib.Path = args.chkpt_dir
    ppo_state = create_ppo_state(resume_dir=chkpt_dir, env=env, seed=args.seed,
                                 lr=1e-3, model=model)

    obs, state = env.reset(key, params)

    times_validator = np.zeros(config["n_nodes"])

    for i in range(args.eval_episodes):
        key_eval, key_step, key_params = jax.random.split(key_eval, 3)

        _, action_distribution = model.apply(ppo_state.params, obs)
        action = mode_subset(action_distribution, params.nb_validators)
        times_validator+=action

        obs, state, reward, done, infos = env.step(key_step, state, action, params)

        params = update_params_fn(params, key_params, action)

        print(f"Episode {i + 1} - Reward: {reward}, Done: {done}, Infos: {infos}, nb_val: {params.nb_validators}")
        print(f"Observation: {obs}")
        indices = jnp.where(action)[0]
        print(f"action: {indices}")
        print("action distribution:", action_distribution.probs)
        print("----")

    plt.figure()
    plt.bar(range(config["n_nodes"]), times_validator)
    plt.xlabel("Node Index")
    plt.ylabel("Times Chosen as Validator")
    plt.title("Validator Selection Frequency")
    plt.xticks(range(config["n_nodes"]))
    plt.grid(axis='y')
    plt.show()