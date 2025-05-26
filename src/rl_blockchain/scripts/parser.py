from argparse import ArgumentParser
import os


def _parse_args():

    # Create an argument parser
    parser = ArgumentParser(description="Run PPO training or evaluation.")
    
    parser.add_argument(
        "--env",
        type=str,
        default="BlockchainEnv_intermediary",
        help="Environment to use. Default is 'BlockchainEnv_intermediary'.",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=20,
        help="Number of nodes in the environment. Default is 20.",
    )
    parser.add_argument(
        "--voting-nodes",
        type=int,
        default=5,
        help="Number of voting nodes in the environment. Default is 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default is 0.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(os.getcwd(), "checkpoints"),
        help="Directory to save checkpoints. Default is 'checkpoints' in the cwd.",
    )
    
    # Logging args
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(os.getcwd(), "logs"),
        help="Directory to save logs. Default is 'logs' in the cwd.",
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level. Default is 'INFO'.",
    )
    parser.add_argument(
        "--logging-format",
        type=str,
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        help="Logging format. Default is '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.",
    )
    parser.add_argument(
        "--logging-datefmt",
        type=str,
        default="%Y-%m-%d %H:%M:%S",
        help="Date format for logging. Default is '%Y-%m-%d %H:%M:%S'.",
    )
    parser.add_argument(
        "--logging-filename",
        type=str,
        default=None,
        help="Name of the log file. Default is None (logs to console).",
    )
    parser.add_argument(
        "--logging-filemode",
        type=str,
        default="a",
        help="File mode for logging. Default is 'a' (append).",
    )
    parser.add_argument(
        "--logging-stream",
        action="store_true",
        default=False,
        help="If True, logs to console as well. Default is False.",
    )
    
    # Wandb args
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="rl_blockchain",
        help="Wandb project name. Default is 'rl_blockchain'.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="bolling-adrien",
        help="Wandb entity name. Default is 'your_entity'.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        type=str,
        default=["default"],
        help="Wandb tags for the run. Default is ['default'].",
    )

    algo_subparser = parser.add_subparsers(dest="algo", required=True)

    #### PPO Subparser ####
    ppo_parser = algo_subparser.add_parser("ppo", help="PPO algorithm.")
    mode_subparsers = ppo_parser.add_subparsers(dest="mode", required=True)
    # Add a subparser for the 'train' mode
    train_parser = mode_subparsers.add_parser("train", help="Train the PPO agent.")
    # Add arguments for the 'train' mode
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default is 100.",
    )
    train_parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of steps per epoch. Default is 1000.",
    )
    train_parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments. Default is 1.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training. Default is 64.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Learning rate for the optimizer. Default is 0.0003.",
    )
    train_parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards. Default is 0.99.",
    )
    train_parser.add_argument(
        "--lambda_",
        type=float,
        default=0.95,
        help="GAE lambda parameter. Default is 0.95.",
    )
    train_parser.add_argument(
        "--clip-ratio",
        type=float,
        default=0.2,
        help="PPO clip ratio. Default is 0.2.",
    )
    train_parser.add_argument(
        "--reward-weights",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="Weights for the rewards as space separated values (e.g: 0.5 0.5). Default is [0.5, 0.5].",
    )
    train_parser.add_argument(
        "--gat-arch",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="GAT architecture as three space separated values (e.g: 64 64 64). Default is [64, 64, 64].",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Name of the checkpoint to load. Default is None. 'latest' will load the latest checkpoint available.",
    )
    train_parser.add_argument(
        "--warm-start",
        type=bool,
        default=False,
        help="If True, warm start the training from the checkpoint. Default is False.",
    )
    
    # Eval of the PPO agent training
    train_parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Interval (in epochs) to evaluate the agent during training. Default is 10.",
    )
    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes to run for evaluation. Default is 10.",
    )



    return parser.parse_args()