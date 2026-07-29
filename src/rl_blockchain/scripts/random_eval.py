"""Evaluate the uniform-random committee baseline and write JSON + CSV.

The random policy draws a uniform ``k``-subset every round, so each node validates
~``k/n`` of the time: the windowed stake distribution is flat and the relative gini
→ 0 (the fairness *lower bound*), while latency is left to chance (the "no latency
optimization" reference at the other end of the Pareto front).

The output schema matches :func:`scripts.ppo_func.eval_ppo_run`, so the resulting row
drops straight into ``eval_all.sh``'s ``comparison.csv``. Only the mode-INDEPENDENT
metrics (``gini`` / ``distance`` / ``weighted_original_reward``) are meaningful to
compare across selections; ``fairness_reward`` / ``weighted_reward`` are differential
signals (~0 at any stationary policy) and are logged only for completeness.

Usage:
    uv run src/rl_blockchain/scripts/random_eval.py \
        --n-nodes 25 --voting-nodes 7 --horizon 200 --eval-episodes 100 \
        --output eval_results/eval_random_baseline.json
"""

import argparse
import json
import logging
import pathlib

import jax
import pandas as pd

from rl_blockchain.algo.random_baseline import eval_random
from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat
from rl_blockchain.scripts.ppo_func import get_env_config

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uniform-random committee baseline evaluation.")
    # Env structure (must match the trained runs it is compared against).
    p.add_argument("--env", type=str, default="blockenv")
    p.add_argument("--n-nodes", type=int, default=25)
    p.add_argument("--voting-nodes", type=int, default=7)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--gat-arch", nargs="+", type=int, default=[64, 32, 16],
                   help="Unused by the random policy; kept so get_env_config can build the env.")
    p.add_argument("--reward-weights", nargs="+", type=float, default=[0.5, 0.5])
    p.add_argument("--update-params", type=UpdateValStrat.parse, default=UpdateValStrat.NO_UPDATE,
                   help=UpdateValStrat.help("Update strategy for the number of validators"))
    p.add_argument("--next-edge-type", type=UpdateDistStrat.parse, default=UpdateDistStrat.NO_UPDATE,
                   help=UpdateDistStrat.help("Update strategy for edges at each step"))
    p.add_argument("--ref-map-file", type=pathlib.Path, default=None,
                   help="Reference map file (only for env=blockenv_close_map).")
    p.add_argument("--gini-reward-mode",
                   choices=["rank", "differential", "differential_shaped", "windowed", "grad"],
                   default="windowed")
    p.add_argument("--gini-shaping-beta", type=float, default=1.0,
                   help="Shaping weight for --gini-reward-mode differential_shaped (see PPO parser).")
    # Eval knobs.
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=pathlib.Path, default=None,
                   help="Output path; the extension picks the format (.json = nested "
                        "results, .csv = flat one-row table for eval_all.sh). Omit to "
                        "print the results to stdout instead of writing a file.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    key = jax.random.PRNGKey(args.seed)
    key_eval, key_param = jax.random.split(key)

    # Reuse the standard env factory so the env-param distribution (adjacency, OU
    # dynamics, committee size) is identical to what the trained runs saw.
    _, env, create_params_fn, log_fn = get_env_config(args, key_param)

    metrics = eval_random(
        env=env,
        key=key_eval,
        create_params_fn=create_params_fn,
        num_episodes=args.eval_episodes,
        batch_size=min(args.batch_size, args.eval_episodes),
        log_fn=log_fn,
    )
    metrics_f = {k: float(v) for k, v in metrics.items()}
    reward_weights = [float(w) for w in args.reward_weights]

    # Tag the row so it stands out in the comparison table. gini_lambda is N/A here.
    results = {
        "model": "random_baseline",
        "checkpoint_dir": None,
        "checkpoint_step": None,
        "gini_reward_mode": "random_baseline",
        "gini_lambda": None,
        "seed": args.seed,
        "num_episodes": args.eval_episodes,
        "n_nodes": args.n_nodes,
        "horizon": args.horizon,
        "reward_weights": reward_weights,
        "metrics": metrics_f,
    }

    # One flat row, same columns as eval_ppo_run so pd.concat aligns in eval_all.sh.
    row = {
        "model": "random_baseline",
        "gini_reward_mode": "random_baseline",
        "gini_lambda": None,
        "reward_weight_gini": reward_weights[0] if len(reward_weights) > 0 else None,
        "reward_weight_distance": reward_weights[1] if len(reward_weights) > 1 else None,
        "seed": args.seed,
        "n_nodes": args.n_nodes,
        "horizon": args.horizon,
        "num_episodes": args.eval_episodes,
        "checkpoint_step": None,
        **metrics_f,
    }

    # No --output: print to stdout. Otherwise the extension picks the format:
    #   .json -> nested results dict, .csv -> the flat one-row table.
    if args.output is None:
        print(json.dumps(results, indent=2))
        return

    out_path: pathlib.Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        pd.DataFrame([row]).to_csv(out_path, index=False)
    elif suffix == ".json":
        out_path.write_text(json.dumps(results, indent=2))
    else:
        raise SystemExit(
            f"Unsupported output extension '{out_path.suffix}'. Use .json or .csv "
            f"(or omit --output to print to stdout).")
    print(f"Random-baseline results written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
