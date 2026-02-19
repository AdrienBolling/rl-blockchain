import logging

from jax import config

from rl_blockchain.scripts.parser import _parse_args
from rl_blockchain.scripts.ppo_func import train_ppo, eval_ppo_run
from rl_blockchain.utils.logging import setup_logging, setup_wandb

logger = logging.getLogger(__name__)


def main():
    # Parse the arguments
    args = _parse_args()

    if args.jax_log_compiles:
        config.update("jax_log_compiles", True)
        logger.info("JAX compilation logging enabled.")

    # Init the logging

    # Setup Weights & Biases (wandb) logging

    # Check the mode and call the appropriate function
    if args.algo == "ppo":
        logger.info(f"Running {args.mode} mode with PPO algorithm.")
        if args.mode == "train":
            run = setup_wandb(args)
            logfile = setup_logging(args, run)
            logging.info("Logging démarré")
            logging.debug(f"Fichier log actif : {logfile}")
            logger.info("Starting training...")
            train_ppo(args)
        elif args.mode == "eval":
            logger.info("Starting evaluation...")
            eval_ppo_run(args)
        else:
            logger.error("Invalid mode. Use 'train' or 'eval'.")
            raise ValueError("Invalid mode. Use 'train' or 'eval'.")


if __name__ == "__main__":
    main()
