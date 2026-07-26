import json
import logging
import pathlib
from argparse import Namespace
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pandas as pd
import wandb
from gymnax.environments.environment import TEnvParams, Environment
from tqdm import tqdm

from rl_blockchain.BlockEnv import BlockchainEnv
from rl_blockchain.BlockEnv.NormailzationWrapper import NormalizationWrapper
from rl_blockchain.algo.ppo import create_checkpoint_manager, create_ppo_state, train_epoch, \
    latest_checkpoint_step, make_optimizer
from rl_blockchain.algo.ppo import eval_ppo
from rl_blockchain.scripts.env_factory import GenericEnvFactory, LOG_TYPE
from rl_blockchain.utils.run_config import apply_model_config, save_run_config, load_run_config
from rl_blockchain.utils.run_counter import project_dir

logger = logging.getLogger(__name__)


def make_fct_value(inputs: list[float], nb_step: int) -> Callable[[int], float]:
    if len(inputs) == 1:
        return lambda _: inputs[0]
    elif len(inputs) == 2:
        init = inputs[0]
        last_value = inputs[1]
        return lambda x: init + (last_value - init) * (x / (nb_step - 1))
    raise ValueError("Unexpected number of inputs: {}, must be 1 or 2".format(len(inputs)))


def make_fct_value_array(inputs: list[float], nb_step: int) -> Callable[[int], jax.Array]:
    return lambda x: jnp.float32(make_fct_value(inputs, nb_step)(x))


def train_ppo(ARGS: Namespace):
    """
    Train a PPO agent on the Blockchain environment.

    Args:
        ARGS (argparse.Namespace): Command line arguments containing hyperparameters and environment settings.

    Returns:
        None
    """
    # Unpack arguments
    num_steps = ARGS.num_steps
    num_envs = ARGS.num_envs
    num_epochs = ARGS.num_epochs
    batch_size = ARGS.batch_size
    normalize_rewards = not ARGS.no_norm_rewards
    lr_fn = make_fct_value_array(ARGS.learning_rate, num_epochs)
    gamma = ARGS.gamma
    lambda_ = ARGS.lambda_
    # None means "no separate schedule": the gini head reuses the general GAE lambda.
    gini_lambda = ARGS.gini_lambda if ARGS.gini_lambda is not None else lambda_
    norm_advantages = not ARGS.no_norm_advantages
    clip_ratio_fn = make_fct_value(ARGS.clip_ratio, num_epochs)
    value_coef = jnp.float32(ARGS.value_coef)
    entropy_coef_fn = make_fct_value_array(ARGS.entropy_coef, num_epochs)
    key = jax.random.PRNGKey(ARGS.seed)
    key, key_param = jax.random.split(key)
    # Create environment parameters

    env_step = 0
    # TODO

    model, env, create_params_fn, log_fn = get_env_config(ARGS, key_param)
    env_train = env

    # If we need to resume a training, get the name of the checkpoint
    load_chkpt_name: pathlib.Path = ARGS.checkpoint

    # Name reserved by setup_wandb; deriving it again here shifted it by one.
    chkpt_dir = project_dir(ARGS) / ARGS.run_name

    # Create the checkpointmanager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=chkpt_dir,
        max_to_keep=ARGS.checkpoint_max_to_keep,
        save_interval_steps=1,
        keep_period=ARGS.checkpoint_keep_period or None,
    )
    # The weights alone cannot be reloaded without the architecture that made them.
    save_run_config(chkpt_dir, ARGS)

    # Create the PPO state
    ppo_state = create_ppo_state(resume_dir=load_chkpt_name, env=env, seed=ARGS.seed, lr=lr_fn(0), model=model,
                                 step=ARGS.checkpoint_step, warm_start=ARGS.warm_start)

    # A full resume continues where the checkpoint stopped, so the lr / clip / entropy
    # schedules and the wandb x-axis pick up at the right epoch instead of restarting.
    start_epoch = 0
    if load_chkpt_name is not None and not ARGS.warm_start:
        last_step = ARGS.checkpoint_step
        if last_step is None:
            last_step = latest_checkpoint_step(load_chkpt_name)
        start_epoch = last_step + 1
        env_step = start_epoch * ((num_steps * num_envs) // batch_size) * batch_size
        logger.info(f"Resuming at epoch {start_epoch} (env_step={env_step})")

    key, key_eval = jax.random.split(key)

    if normalize_rewards and isinstance(env_train, BlockchainEnv):
        # If using normalization, ensure the environment is wrapped accordingly
        env_train = NormalizationWrapper(env_train)

    # Train the PPO agent
    try:
        for epoch in tqdm(range(start_epoch, num_epochs)):
            # Memoized on the float lr so a constant schedule reuses one stable
            # optimizer object -> update_ppo compiles once instead of every epoch.
            model_opt = make_optimizer(float(lr_fn(epoch)))
            # Train for one epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            ppo_state, env_step = train_epoch(ppo_state=ppo_state, epoch=epoch, env=env_train,
                                              model=model, num_steps=num_steps,
                                              num_envs=num_envs, create_params_fn=create_params_fn,
                                              batch_size=batch_size, model_opt=model_opt, gamma=gamma,
                                              lambda_=lambda_,
                                              clip_ratio=clip_ratio_fn(epoch), log_fn=log_fn,
                                              env_step=env_step, value_coef=value_coef,
                                              entropy_coef=entropy_coef_fn(epoch),
                                              norm_advantage=norm_advantages,
                                              micro_batch_size=getattr(ARGS, "micro_batch_size", None),
                                              gini_lambda=gini_lambda)
            key, _ = jax.random.split(key)
            if epoch % ARGS.eval_interval == 0:
                logger.info(f"Evaluating PPO agent at epoch {epoch + 1}/{num_epochs}")
                # Evaluate the PPO agent
                metrics = eval_ppo(
                    ppo_state=ppo_state,
                    key=key_eval,
                    env=env,
                    model=model,
                    create_params_fn=create_params_fn,
                    num_episodes=ARGS.eval_episodes,
                    recorded_episodes=5,
                    batch_size=min(num_envs, ARGS.eval_episodes),
                    log_fn=log_fn
                )

                wandb.log({"eval": metrics}, step=env_step)
                logger.info(metrics)

            # Save the checkpoint
            checkpoint_manager.save(step=epoch, args=ocp.args.StandardSave(ppo_state))
            # logger.info(f"Epoch {epoch} - Policy Loss: {policy_loss}, Value Loss: {value_loss}")
    finally:
        # orbax saves asynchronously. Without this the last save is still an
        # uncommitted '<step>.orbax-checkpoint-tmp' when the interpreter exits, so
        # the final epoch is silently lost (and orbax dies on shutdown). In a
        # `finally` so a crash or a SLURM time-out still commits what it can.
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
    wandb.finish()


def get_env_config(ARGS: Namespace, key_param: jax.Array) \
        -> Tuple[flax.linen.Module, Environment, Callable[[jax.Array], TEnvParams], LOG_TYPE]:
    env_name = ARGS.env.lower()
    horizon = getattr(ARGS, "horizon", 200)  # older checkpoints predate --horizon
    if env_name == "blockenv":
        config = {"n_nodes": ARGS.n_nodes, "gat_arch": ARGS.gat_arch, "voting_nodes": ARGS.voting_nodes,
                  "reward_weights": ARGS.reward_weights, "next_val_type": ARGS.update_params,
                  "next_edge_type": ARGS.next_edge_type, "horizon": horizon,
                  "gini_reward_mode": getattr(ARGS, "gini_reward_mode", "rank")}
    elif env_name == "blockenv_close_map":
        assert ARGS.ref_map_file is not None, "ref_map_file must be provided for blockenv_close_map"
        config = {"gat_arch": ARGS.gat_arch, "voting_nodes": ARGS.voting_nodes,
                  "reward_weights": ARGS.reward_weights, "ref_map_file": ARGS.ref_map_file,
                  "next_val_type": ARGS.update_params, "next_edge_type": ARGS.next_edge_type,
                  "horizon": horizon,
                  "gini_reward_mode": getattr(ARGS, "gini_reward_mode", "rank")}
    elif env_name == "cartpole":
        config = {}
    else:
        raise ValueError(
            f"Unknown environment: {env_name}. Available environments: {GenericEnvFactory.available_environments()}")
    model, env, _, create_params_fn, log_fn = GenericEnvFactory.create(env_name, key_param, config)
    return model, env, create_params_fn, log_fn


def eval_ppo_run(args: Namespace):
    key = jax.random.PRNGKey(args.seed)
    key_eval, state_key, key_param = jax.random.split(key, 3)

    chkpt_dir: pathlib.Path = args.chkpt_dir
    # Rebuild the exact architecture the checkpoint was trained with, so the user
    # does not have to re-supply --n-nodes / --gat-arch / ... from memory.
    apply_model_config(args, chkpt_dir)

    model, env, create_params_fn, log_fn = get_env_config(args, key_param)

    # lr only shapes the (unused) optimizer state of the restore target.
    ppo_state = create_ppo_state(resume_dir=chkpt_dir, env=env, seed=args.seed, lr=1e-3,
                                 model=model, step=args.checkpoint_step)

    # Evaluate the PPO agent
    metrics = eval_ppo(
        ppo_state=ppo_state,
        env=env,
        key=key_eval,
        model=model,
        create_params_fn=create_params_fn,
        num_episodes=args.eval_episodes,
        recorded_episodes=5,
        log_fn=log_fn
    )

    # Write the aggregate results to JSON (same spirit as simple_eval): no wandb.
    checkpoint_name = chkpt_dir.resolve().name
    # Pull the training-time labels from the saved run config so rows are correctly
    # tagged by mode/lambda regardless of what was passed on the eval command line.
    saved_cfg = load_run_config(chkpt_dir)
    gini_reward_mode = saved_cfg.get("gini_reward_mode", getattr(args, "gini_reward_mode", None))
    gini_lambda = saved_cfg.get("gini_lambda", getattr(args, "gini_lambda", None))
    # None means the gini head reused the general lambda at train time; surface the
    # effective value so the comparison table never shows a bare None.
    if gini_lambda is None:
        gini_lambda = saved_cfg.get("lambda_", getattr(args, "lambda_", None))
    reward_weights = [float(w) for w in args.reward_weights]

    metrics_f = {k: float(v) for k, v in metrics.items()}
    results = {
        "model": checkpoint_name,
        "checkpoint_dir": str(chkpt_dir.resolve()),
        "checkpoint_step": args.checkpoint_step,  # None => latest
        "gini_reward_mode": gini_reward_mode,
        "gini_lambda": gini_lambda,
        "seed": args.seed,
        "num_episodes": args.eval_episodes,
        "n_nodes": args.n_nodes,
        "horizon": getattr(args, "horizon", 200),
        "reward_weights": reward_weights,
        # Every aggregate metric eval_ppo produced. The mode-INDEPENDENT comparables
        # across runs are `gini`, `distance` and `weighted_original_reward`; the
        # mode-dependent `fairness_reward`/`weighted_reward` are only meaningful within
        # a single mode.
        "metrics": metrics_f,
    }

    out_path = args.output or pathlib.Path(f"eval_{checkpoint_name}.json")
    out_path.write_text(json.dumps(results, indent=2))

    # Also emit a one-row CSV (flat) so the 12 runs concat/merge trivially in pandas:
    #   pd.concat([pd.read_csv(f) for f in glob("eval_*.csv")])
    row = {
        "model": checkpoint_name,
        "gini_reward_mode": gini_reward_mode,
        "gini_lambda": gini_lambda,
        "reward_weight_gini": reward_weights[0] if len(reward_weights) > 0 else None,
        "reward_weight_distance": reward_weights[1] if len(reward_weights) > 1 else None,
        "seed": args.seed,
        "n_nodes": args.n_nodes,
        "horizon": getattr(args, "horizon", 200),
        "num_episodes": args.eval_episodes,
        "checkpoint_step": args.checkpoint_step,
        **metrics_f,
    }
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    logger.info(metrics)
    print(f"Results written to {out_path.resolve()} and {csv_path.resolve()}")
