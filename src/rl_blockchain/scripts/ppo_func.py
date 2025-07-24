import logging
import pathlib
import re
from argparse import Namespace
from typing import Callable

import jax
import optax
import orbax.checkpoint as ocp
import wandb
from tqdm import tqdm

from rl_blockchain.algo.ppo import create_checkpoint_manager, create_ppo_state, train_epoch, load_ppo_state
from rl_blockchain.algo.ppo import eval_ppo
from rl_blockchain.scripts.env_factory import GenericEnvFactory

logger = logging.getLogger(__name__)


def make_fct_value(inputs: list[float], nb_step: int) -> Callable[[int], float]:
    if len(inputs) == 1:
        return lambda _: inputs[0]
    elif len(inputs) == 2:
        init = inputs[0]
        last_value = inputs[1]
        return lambda x: init + (last_value - init) * (x / (nb_step - 1))
    raise ValueError("Unexpected number of inputs: {}, must be 1 or 2".format(len(inputs)))


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
    lr_fn = make_fct_value(ARGS.learning_rate, num_epochs)
    gamma = ARGS.gamma
    lambda_ = ARGS.lambda_
    clip_ratio_fn = make_fct_value(ARGS.clip_ratio, num_epochs)
    value_coef = ARGS.value_coef
    entropy_coef_fn = make_fct_value(ARGS.entropy_coef, num_epochs)
    key = jax.random.PRNGKey(ARGS.seed)
    key, key_param = jax.random.split(key)
    # Create environment parameters

    sub_epoch = 0

    model, env, create_params_fn, log_fn = get_env_config(ARGS, key_param)

    # If we need to resume a training, get the name of the checkpoint
    chkpt_name = ARGS.checkpoint

    # If the checkpoint is 'latest', get the latest run id
    api = wandb.Api()
    runs = api.runs(
        f"{ARGS.wandb_entity}/{ARGS.wandb_project}",
        order="created_at",
    )
    try:
        if chkpt_name == "latest":
            chkpt_name = runs[-1].id
        elif chkpt_name is None:
            chkpt_name = f"run_{len(runs)}"
        else:
            # Check if the name respects the format 'run_<int>' with a regex
            if not re.match(r"^run_\d+$", chkpt_name):
                raise ValueError(f"Checkpoint name '{chkpt_name}' is not valid. It should be 'run_<int>' or 'latest'.")
    except ValueError:
        # When the project does not exist yet, assume no runs
        chkpt_name = "run_0"

    # Checkpoint_dir
    chkpt_dir = f"{ARGS.checkpoint_dir}/{ARGS.wandb_entity}_{ARGS.wandb_project}/{chkpt_name}"

    # Create the checkpointmanager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=chkpt_dir,
        max_to_keep=1,
        save_interval_steps=1,
    )

    # Create the PPO state
    ppo_state = create_ppo_state(
        checkpoint_manager=checkpoint_manager,
        resume_dir=chkpt_dir if ARGS.checkpoint else None,
        warm_start=ARGS.warm_start,
        env=env,
        seed=ARGS.seed,
        lr=lr_fn(0),
        model=model
    )

    key, key_eval = jax.random.split(key)

    # Train the PPO agent
    for epoch in tqdm(range(num_epochs)):
        model_opt = optax.adam(lr_fn(epoch))
        # Train for one epoch
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        ppo_state, sub_epoch = train_epoch(ppo_state=ppo_state, epoch=epoch, env=env, model_opt=model_opt, model=model,
                                           create_params_fn=create_params_fn,
                                           num_steps=num_steps,
                                           num_envs=num_envs, batch_size=batch_size, gamma=gamma,
                                           lambda_=lambda_, clip_ratio=clip_ratio_fn(epoch),
                                           value_coef=value_coef, entropy_coef=entropy_coef_fn(epoch),
                                           sub_epoch=sub_epoch, log_fn=log_fn,
                                           normalize_rewards=True)
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
                log_fn=log_fn
            )

            wandb.log({"eval": metrics}, step=sub_epoch)
            logger.info(metrics)

        # Save the checkpoint
        checkpoint_manager.save(step=epoch, args=ocp.args.StandardSave(ppo_state))
        # logger.info(f"Epoch {epoch} - Policy Loss: {policy_loss}, Value Loss: {value_loss}")
    wandb.finish()


def get_env_config(ARGS: Namespace, key_param: jax.Array):
    env_name = ARGS.env.lower()
    if env_name == "blockenv":
        config = {"n_nodes": ARGS.n_nodes, "gat_arch": ARGS.gat_arch, "voting_nodes": ARGS.voting_nodes,
                  "reward_weights": ARGS.reward_weights}
    elif env_name == "cartpole":
        config = {}
    else:
        raise ValueError(
            f"Unknown environment: {env_name}. Available environments: {GenericEnvFactory.available_environments()}")
    model, env, _, create_params_fn, log_fn = GenericEnvFactory.create(env_name, key_param, config)
    return model, env, create_params_fn, log_fn


def eval_ppo_run(args: Namespace):
    key = jax.random.PRNGKey(args.seed)
    key, state_key, key_param = jax.random.split(key, 3)

    model, env, create_params_fn, log_fn = get_env_config(args, key_param)

    chkpt_dir :pathlib.Path= args.chkpt_dir
    chkpt_dir.absolute()
    print(type(chkpt_dir), chkpt_dir.absolute())

    ppo_state = load_ppo_state(        chkpt_dir, key    )

    # Evaluate the PPO agent
    metrics = eval_ppo(
        ppo_state=ppo_state,
        env=env,
        model=model,
        create_params_fn=create_params_fn,
        num_episodes=args.eval_episodes,
        log_fn=log_fn
    )

    logger.info(metrics)
    print(metrics)
