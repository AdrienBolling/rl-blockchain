from rl_blockchain.tests.test_parser import parse_args
import os

def main():
    args = parse_args()
    
    # Initialise the results dir
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.test_type == "profiling":
        if args.target == "env":
            from rl_blockchain.tests.profiling import env_profiling
            env_profiling(args)
            
if __name__ == "__main__":
    main()