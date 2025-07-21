import argparse

import jax

from rl_blockchain import BlockEnv
from rl_blockchain.BlockEnv import StaticEnvParams, BlockchainEnv
from rl_blockchain.DeterministSelection.MarokvianSelection import eval_markov
from rl_blockchain.scripts.env_factory import BlockchainEnvBuilder
from rl_blockchain.scripts.parser import REF_FILENAME
import logging

logger = logging.getLogger(__name__)

def create_parser() -> argparse.Namespace:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Run the Markovian selection algorithm.")
    parser.add_argument('--n_nodes', type=int, default=25, help='Number of nodes in the environment.')
    parser.add_argument('--n_val', type=int, default=10, help='Number of validators to select.')
    parser.add_argument('--reward_weights', type=float, nargs='+', default=[1.0, 1.0], help="Weights for the rewards.")
    parser.add_argument('--val', type=int, default=10, help='Number of validators.')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run.')
    return parser.parse_args()


if __name__ == '__main__':
    args = create_parser()

    key = jax.random.PRNGKey(0)
    key_param, key = jax.random.split(key)


    def create_params_fn(key: jax.Array) -> BlockEnv.EnvParams:
        return BlockEnv.EnvParams.create_random(
            args.n_nodes,
            key,
            args.n_val,
            args.reward_weights,
        )


    # TODO wandb integration
    env_params = create_params_fn(key_param)
    static_params = StaticEnvParams.create(args.n_nodes, REF_FILENAME[args.n_nodes])
    env = BlockchainEnv(env_params, static_params)
    log_fn = BlockchainEnvBuilder.log

    logger.info("Starting evaluation of Markovian selection algorithm...")
    metrics = eval_markov(env, create_params_fn, key, args.episodes, log_fn)
    logger.info("Evaluation completed.")
    print("Metrics:", metrics)
    logger.info(metrics)
