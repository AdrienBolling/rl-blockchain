from argparse import ArgumentParser
import os

def parse_args():
# Create a parser
    parser = ArgumentParser(description="Test the RL blockchain environment.")

    # Env parameters
    parser.add_argument(
        "--env",
        type=str,
        default="BlockchainEnv_intermediary",
        help="Environment to use. Default is 'BlockchainEnv_intermediary'.",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=500,
        help="Number of nodes in the environment. Default is 20.",
    )
    parser.add_argument(
        "--voting-nodes",
        type=int,
        default=20,
        help="Number of voting nodes in the environment. Default is 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default is 0.",
    )
    
    # Logging args
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(os.getcwd(), "profiling_results"),
        help="Directory to save results. Default is 'results'.",
    )
    
    # Test type subparser
    subparsers = parser.add_subparsers(dest="test_type", required=True)
    
    # Test type: profiling
    profiling_parser = subparsers.add_parser(
        "profiling",
        help="Run profiling tests.",
    )
    profiling_parser.add_argument(
        "--n-steps",
        type=int,
        default=100000,
        help="Number of steps to run for profiling. Default is 1000.",
    )
    profiling_parser.add_argument(
        "--n-envs",
        type=int,
        default=10,
        help="Number of envs to run for profiling. Default is 10.",
    )
    
    # What to profile subparser
    target_subparsers = profiling_parser.add_subparsers(dest="target", required=True)
    
    # Target: env
    env_parser = target_subparsers.add_parser(
        "env",
        help="Profile the environment only (no training just steps through the env with random legal actions).",
    )
    return parser.parse_args()