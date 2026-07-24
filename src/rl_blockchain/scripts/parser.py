import os
import pathlib
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from enum import Enum

REF_FILENAME = {
    7: "ref_grid_min_max/grid_7.csv",
    10: "ref_grid_min_max/grid_10.csv",
    25: "ref_grid_min_max/grid_25.csv",
    200: "ref_grid_min_max/grid_200.csv",
}


class ArgparseEnum(Enum):
    """
    Base Enum with argparse-compatible parsing.
    Accepts enum name (case-insensitive) or integer value.
    """

    def __str__(self):
        return self.name

    @classmethod
    def parse(cls, value: str):
        # Try by name
        try:
            return cls[value.upper()]
        except KeyError:
            pass

        # Try by integer value
        try:
            return cls(int(value))
        except (ValueError, KeyError):
            raise ArgumentTypeError(
                f"Invalid value '{value}'. "
                f"Allowed names: {[e.name for e in cls]} "
                f"or values: {[e.value for e in cls]}"
            )

    @classmethod
    def help(cls, desc:str) -> str:
        return (
                f"{desc}. "
                "Allowed values: "
                + ", ".join([f"{e.name} ({e.value})" for e in cls])
                + f". Default is '{cls(0)}'."
        )


class UpdateValStrat(ArgparseEnum):
    NO_UPDATE = 0
    THRESHOLD_UPDATE = 1
    ORN_UHL_UPDATE = 2


class UpdateDistStrat(ArgparseEnum):
    NO_UPDATE = 0
    ORN_UHL_UPDATE = 1


def _parse_args() -> Namespace:
    # Create an argument parser
    parser = ArgumentParser(description="Run PPO training or evaluation.")

    parser.add_argument(
        "--jax-log-compiles",
        action="store_true",
        default=False,
        help="If True, enable JAX debug mode. Default is False.",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="blockenv",
        help="Environment to use. Default is 'BlockEnv'.",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=25,
        help="Number of nodes in the environment. Default is 25.",
    )
    parser.add_argument(
        "--voting-nodes",
        type=int,
        default=7,
        help="Number of voting nodes in the environment. Default is 7.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Episode horizon: length of the validator-history window (ring_history). "
             "Default is 200.",
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
        help=(
            "Logging format. Default is "
            "'%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s'."
        ),
    )
    parser.add_argument(
        "--logging-datefmt",
        type=str,
        default="%Y-%m-%d %H:%M:%S",
        help="Date format for logging. Default is "
             "'%%Y-%%m-%%d %%H:%%M:%%S'.",
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
    parser.add_argument(
        "--logging-prefix",
        type=str,
        default="run",
        help="Prefix for logging. Default is 'run'.",
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

    ppo_parser.add_argument(
        "--reward-weights",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="Weights for the rewards as space separated values. Default is [0.5, 0.5].",
    )
    ppo_parser.add_argument(
        "--gini-reward-mode",
        choices=["rank", "differential", "windowed", "grad"],
        default="windowed",
        help="Fairness training signal for the gini head. 'rank': per-step "
             "action-attributable stake-rank surrogate. 'differential': potential-based "
             "per-step decrease of the true windowed gini (G_t - G_{t+1}); GAE re-integrates "
             "it. 'windowed' (default): the original level reward (1 - relative_gini) -- integrative, "
             "needs --gini-lambda 0. 'grad': jax.grad of the new relative gini w.r.t. the "
             "action. The monitored env/gini metric is the true windowed gini in all cases.",
    )
    ppo_parser.add_argument(
        "--gat-arch",
        nargs="+",
        type=int,
        default=[64, 32, 16],
        help="GAT architecture as three space separated values. Default is [64, 32, 16].",
    )

    ppo_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes to run for evaluation. Default is 10.",
    )

    ppo_parser.add_argument(
        "--ref-map-file",
        type=pathlib.Path,
        default=None,
        help="Path to the reference map file for the environment. Default is None.",
    )

    ppo_parser.add_argument(
        "--update-params",
        type=UpdateValStrat.parse,
        default=UpdateValStrat.NO_UPDATE,
        help=UpdateValStrat.help("Update strategy for the number of validators"),
    )

    ppo_parser.add_argument(
        "--next-edge-type",
        type=UpdateDistStrat.parse,
        default=UpdateDistStrat.NO_UPDATE,
        help=UpdateDistStrat.help("Update strategy for edges at each step"),
    )

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
        default=10000,
        help="Number of steps per epoch. Default is 10000.",
    )
    train_parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments. Default is 5.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training. Default is 64.",
    )
    train_parser.add_argument(
        "--learning-rate",
        nargs="+",
        type=float,
        default=[0.0003],
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
        "--gini-lambda",
        type=float,
        default=0.0,
        help="GAE lambda for the gini (fairness) reward head only; distance keeps --lambda_. "
             "The marginal gini reward is per-step action-attributable, so a high lambda sums "
             "future-action noise (from the rotating fairness target) and buries its signal. "
             "lambda=0 (default) gives a clean one-step advantage and is what makes fairness "
             "learnable (measured: rel_gini 0.28 -> 0.08 vs flat at 0.99).",
    )
    train_parser.add_argument(
        "--clip-ratio",
        nargs="+",
        type=float,
        default=[0.2],
        help="PPO clip ratio. Default is 0.2.",
    )
    train_parser.add_argument(
        "--value-coef",
        type=float,
        default=0.5,
        help="Coefficient for the value function loss. Default is 0.5.",
    )
    train_parser.add_argument(
        "--entropy-coef",
        nargs="+",
        type=float,
        default=[0.01],
        help="Coefficient for the entropy loss. Default is 0.01.",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path of the checkpoint to load. Default is None.",
    )
    train_parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Step (epoch) to resume from inside --checkpoint. Default: the latest.",
    )
    train_parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Load only the network weights from --checkpoint: reset the optimizer "
             "moments, the RNG stream and the epoch schedules. Default is a full "
             "resume (weights + optimizer + RNG, schedules continue).",
    )
    train_parser.add_argument(
        "--checkpoint-max-to-keep",
        type=int,
        default=5,
        help="Number of most-recent checkpoints to retain. Default is 5.",
    )
    train_parser.add_argument(
        "--checkpoint-keep-period",
        type=int,
        default=50,
        help="Additionally retain every N-th epoch forever, so old models stay "
             "loadable. Set to 0 to disable. Default is 50.",
    )

    # Eval of the PPO agent training
    train_parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Interval (in epochs) to evaluate the agent during training. Default is 10.",
    )

    train_parser.add_argument(
        "--no-norm-advantages",
        action="store_true",
        default=False,
        help="If True, normalize the advantages during training. Default is False.",
    )

    train_parser.add_argument(
        "--no-norm-rewards",
        action="store_true",
        default=False,
        help="If True, normalize the rewards during training. Default is False.",
    )

    train_parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="If set, split each PPO minibatch into micro-batches of this size and "
             "accumulate gradients (lax.scan) to cut peak VRAM. Must divide "
             "--batch-size. Result is identical to a full-batch update. Default: "
             "None (no accumulation).",
    )

    eval_parser = mode_subparsers.add_parser("eval", help="Eval the PPO agent.")

    eval_parser.add_argument(
        "chkpt_dir",
        type=pathlib.Path,
        help="Directory to load the checkpoint from.",
    )
    eval_parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Step (epoch) to evaluate inside chkpt_dir. Default: the latest.",
    )
    eval_parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Path to write the JSON evaluation results. Default: "
             "'eval_<checkpoint-name>.json' in the cwd.",
    )

    return parser.parse_args()
