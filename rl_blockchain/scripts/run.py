import logging
from rl_blockchain.utils.logging import setup_logging, setup_wandb
from rl_blockchain.scripts.ppo_func import train_ppo, eval_ppo
from rl_blockchain.scripts.parser import _parse_args

def main():
    # Parse the arguments
    args = _parse_args()

    # Init the logging
    setup_logging(args)
    logger = logging.getLogger(__name__)
    # Setup Weights & Biases (wandb) logging
    setup_wandb(args)

    # Check the mode and call the appropriate function
    if args.algo == "ppo":
        logger.info(f"Running {args.mode} mode with PPO algorithm.")
        if args.mode == "train":
            logger.info("Starting training...")
            train_ppo(args)
        elif args.mode == "eval":
            logger.info("Starting evaluation...")
            eval_ppo(args)
        else:
            logger.error("Invalid mode. Use 'train' or 'eval'.")
            raise ValueError("Invalid mode. Use 'train' or 'eval'.")


if __name__ == "__main__":
    main()