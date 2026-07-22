"""Interactive single-episode evaluation of a trained PPO checkpoint.

Rolls the greedy (mode) policy through the BlockEnv and plots how often each node
is picked as a validator. Unlike ``ppo eval`` (aggregate metrics) this is an
inspection tool: it prints per-step detail and the validator-selection histogram.

The model architecture (n_nodes, gat_arch, ...) is restored from the
``run_config.json`` written next to the checkpoint, so you only choose:

* the checkpoint (``chkpt_dir`` + optional ``--checkpoint-step``),
* the reward weights (``--reward-weights``, defaults to the trained values), and
* how the validator count behaves:
    - ``--val-mode fixed``  : constant ``--voting-nodes`` every step;
    - ``--val-mode ornuhl`` : drifts each step (Ornstein-Uhlenbeck, as in
      training ``--update-params 2``), clipped to ``--val-range MIN MAX``.

Results (rewards, global reward, validator frequency, env metrics, ...) are
written to a JSON file (``--output``).
"""

import argparse
import json
import pathlib

# XLA flags must be set before the first `import jax` (see utils/jax_runtime).
from rl_blockchain.utils.jax_runtime import configure_compilation_cache, configure_xla_flags

configure_xla_flags()
configure_compilation_cache()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rl_blockchain.BlockEnv.BlockEnv import mode_subset
from rl_blockchain.algo.ppo import create_ppo_state
from rl_blockchain.scripts.parser import UpdateValStrat
from rl_blockchain.scripts.ppo_func import get_env_config
from rl_blockchain.utils.run_config import apply_model_config


def create_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "chkpt_dir",
        type=pathlib.Path,
        help="Directory to load the checkpoint from (must contain run_config.json).",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Step (epoch) to load inside chkpt_dir. Default: the latest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default is 0.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Number of environment steps to roll out. Default is 100.",
    )
    parser.add_argument(
        "--reward-weights",
        nargs=2,
        type=float,
        default=None,
        metavar=("GINI", "DISTANCE"),
        help="Override the reward weights. Default: the values the run was trained with.",
    )

    # ---- validator-count behaviour ----
    parser.add_argument(
        "--val-mode",
        choices=("fixed", "ornuhl"),
        default="fixed",
        help="'fixed': constant --voting-nodes every step. 'ornuhl': the count "
             "drifts each step (Ornstein-Uhlenbeck, like training), clipped to "
             "--val-range. Default: fixed.",
    )
    parser.add_argument(
        "--voting-nodes",
        type=int,
        default=6,
        help="Validator count for --val-mode fixed. Default is 6.",
    )
    parser.add_argument(
        "--val-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("MIN", "MAX"),
        help="Clip bounds for the drifting count in --val-mode ornuhl. "
             "Default: [4, n_nodes].",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Path to write the JSON results. Default: "
             "'simple_eval_<checkpoint-name>.json' in the cwd.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the validator-frequency plot (useful when running headless).",
    )
    return parser.parse_args()


def _clip_nb_val(state, lo, hi):
    """Force the (drifting) validator count back into ``[lo, hi]``."""
    return state.replace(nb_val=jnp.clip(state.nb_val, lo, hi).astype(jnp.int32))


def main() -> None:
    args = create_argparse()
    key = jax.random.PRNGKey(args.seed)
    key_reset, key_param, key_eval = jax.random.split(key, 3)

    # Capture the CLI overrides before apply_model_config clobbers them with the
    # trained values.
    user_reward_weights = args.reward_weights
    user_voting_nodes = args.voting_nodes

    # Restore the architecture the checkpoint was trained with (n_nodes, gat_arch,
    # ...), so the user does not have to re-supply the training flags.
    apply_model_config(args, args.chkpt_dir)

    # Re-apply the user-selectable overrides (these change env behaviour, not the
    # weight shapes, so they are safe to differ from the trained config).
    if user_reward_weights is not None:
        args.reward_weights = user_reward_weights
    if args.val_mode == "fixed":
        args.voting_nodes = user_voting_nodes          # fixed init count...
        args.update_params = UpdateValStrat.NO_UPDATE  # ...and keep it constant.
    else:  # ornuhl
        args.voting_nodes = 0                           # random init, then clipped
        args.update_params = UpdateValStrat.ORN_UHL_UPDATE

    model, env, create_params_fn, log_fn = get_env_config(args, key_param)
    params = create_params_fn(key_param)

    # lr only shapes the (unused) optimizer state of the restore target.
    ppo_state = create_ppo_state(resume_dir=args.chkpt_dir, env=env, seed=args.seed,
                                 lr=1e-3, model=model, step=args.checkpoint_step)

    n_nodes = args.n_nodes
    ornuhl = args.val_mode == "ornuhl"
    val_lo, val_hi = args.val_range if args.val_range is not None else (4, n_nodes)

    obs, state = env.reset(key_reset, params)
    if ornuhl:
        state = _clip_nb_val(state, val_lo, val_hi)

    times_validator = np.zeros(n_nodes)
    rewards: list[float] = []
    nb_val_per_step: list[int] = []
    metrics: dict[str, list[float]] = {}  # per-step env infos (gini, distance, ...)

    for i in range(args.eval_steps):
        key_eval, key_step = jax.random.split(key_eval)

        _, action_distribution = model.apply(ppo_state.params, obs)
        action = mode_subset(action_distribution, state.nb_val)
        times_validator += np.asarray(action)
        nb_val_per_step.append(int(state.nb_val))

        obs, state, reward, done, infos = env.step(key_step, state, action, params)
        # In ornuhl mode env.step already drifted nb_val; tighten it to the range.
        if ornuhl:
            state = _clip_nb_val(state, val_lo, val_hi)

        rewards.append(float(reward))
        for k, v in infos.items():
            metrics.setdefault(k, []).append(float(v))

        chosen = np.asarray(jnp.where(action)[0]).tolist()
        print(f"Step {i + 1} - Reward: {float(reward):.4f}, Done: {bool(done)}, "
              f"nb_val: {nb_val_per_step[-1]}, chosen: {chosen}")

    # ---- assemble and write the JSON results ----
    checkpoint_name = args.chkpt_dir.resolve().name
    results = {
        "model": checkpoint_name,
        "checkpoint_dir": str(args.chkpt_dir.resolve()),
        "checkpoint_step": args.checkpoint_step,  # None => latest
        "seed": args.seed,
        "num_tests": args.eval_steps,
        "n_nodes": n_nodes,
        "val_mode": args.val_mode,
        "voting_nodes": args.voting_nodes if not ornuhl else None,
        "val_range": [val_lo, val_hi] if ornuhl else None,
        # [gini, distance] weights used to build the env (from the run config,
        # or the --reward-weights override).
        "reward_weights": [float(w) for w in args.reward_weights],
        "rewards": rewards,
        "global_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "nb_val_per_step": nb_val_per_step,
        "validator_selection_frequency": times_validator.astype(int).tolist(),
        # Mean of every per-step env metric (gini, distance, *_reward, ...).
        "metrics_mean": {k: float(np.mean(v)) for k, v in metrics.items()},
    }

    out_path = args.output or pathlib.Path(f"simple_eval_{checkpoint_name}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nGlobal reward: {results['global_reward']:.4f} "
          f"(mean {results['mean_reward']:.4f}) over {args.eval_steps} steps")
    print(f"Results written to {out_path.resolve()}")

    if not args.no_plot:
        mode_label = (f"fixed={args.voting_nodes}" if not ornuhl
                      else f"orn-uhl clipped to [{val_lo}, {val_hi}]")
        plt.figure()
        plt.bar(range(n_nodes), times_validator)
        plt.xlabel("Node Index")
        plt.ylabel("Times Chosen as Validator")
        plt.title(f"Validator Selection Frequency ({args.eval_steps} steps, {mode_label})")
        plt.xticks(range(n_nodes))
        plt.grid(axis="y")
        plt.show()


if __name__ == "__main__":
    main()