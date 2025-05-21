from argparse import ArgumentParser
import os
from pathlib import Path


def _parse_args():

    # Create an argument parser
    parser = ArgumentParser(description="Run PPO training or evaluation.")

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
        "--lambda",
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

    # Create a subparser for the 'eval' mode # TODO

    eval_parser = parser.add_subparsers(dest="eval_mode")
