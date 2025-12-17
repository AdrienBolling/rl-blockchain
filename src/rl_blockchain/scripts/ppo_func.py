import logging
import pathlib
from argparse import Namespace
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from gymnax.environments.environment import TEnvParams, Environment
from tqdm import tqdm

from rl_blockchain.BlockEnv import BlockchainEnv
from rl_blockchain.BlockEnv.NormailzationWrapper import NormalizationWrapper
from rl_blockchain.algo.ppo import create_checkpoint_manager, create_ppo_state, train_epoch, load_ppo_state
from rl_blockchain.algo.ppo import eval_ppo
from rl_blockchain.scripts.env_factory import GenericEnvFactory, change_val_param_fn, white_param_fn, Outer_param_fn, \
    LOG_TYPE

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
    normalize_rewards = True
    lr_fn = make_fct_value_array(ARGS.learning_rate, num_epochs)
    gamma = ARGS.gamma
    lambda_ = ARGS.lambda_
    norm_advantages = not ARGS.no_norm_advantages
    clip_ratio_fn = make_fct_value(ARGS.clip_ratio, num_epochs)
    value_coef = jnp.float32(ARGS.value_coef)
    entropy_coef_fn = make_fct_value_array(ARGS.entropy_coef, num_epochs)
    key = jax.random.PRNGKey(ARGS.seed)
    key, key_param = jax.random.split(key)
    # Create environment parameters

    sub_epoch = 0

    model, env, create_params_fn, log_fn, update_params_fn = get_env_config(ARGS, key_param)

    # If we need to resume a training, get the name of the checkpoint
    load_chkpt_name: pathlib.Path = ARGS.checkpoint

    # If the checkpoint is 'latest', get the latest run id
    api = wandb.Api()
    runs = api.runs(
        f"{ARGS.wandb_entity}/{ARGS.wandb_project}",
        order="created_at",
    )

    new_run_chkpt_name = f"run_{len(runs)}"

    # Checkpoint_dir
    chkpt_dir = f"{ARGS.checkpoint_dir}/{ARGS.wandb_entity}_{ARGS.wandb_project}/{new_run_chkpt_name}"

    # Create the checkpointmanager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=chkpt_dir,
        max_to_keep=1,
        save_interval_steps=1,
    )

    # Create the PPO state
    ppo_state = create_ppo_state(resume_dir=load_chkpt_name, env=env, seed=ARGS.seed, lr=lr_fn(0), model=model)

    key, key_eval = jax.random.split(key)

    if normalize_rewards and isinstance(env, BlockchainEnv):
        # If using normalization, ensure the environment is wrapped accordingly
        env = NormalizationWrapper(env)

    print(f"INIT ids env : {id(env)}")
    print(f"INIT hashs env : {hash(env)}")

    # Train the PPO agent
    for epoch in tqdm(range(num_epochs)):
        model_opt = optax.adam(lr_fn(epoch))
        # Train for one epoch
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        ppo_state, sub_epoch = train_epoch(ppo_state=ppo_state, epoch=epoch, env=env,
                                           model=model, num_steps=num_steps,
                                           num_envs=num_envs, create_params_fn=create_params_fn,
                                           update_params_fn=update_params_fn,
                                           batch_size=batch_size, model_opt=model_opt, gamma=gamma,
                                           lambda_=lambda_,
                                           clip_ratio=clip_ratio_fn(epoch), log_fn=log_fn,
                                           sub_epoch=sub_epoch, value_coef=value_coef,
                                           entropy_coef=entropy_coef_fn(epoch),
                                           norm_advantage=norm_advantages)
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
                update_params_fn=update_params_fn,
                num_episodes=ARGS.eval_episodes,
                recorded_episodes=5,
                log_fn=log_fn
            )

            wandb.log({"eval": metrics}, step=sub_epoch)
            logger.info(metrics)

        # Save the checkpoint
        checkpoint_manager.save(step=epoch, args=ocp.args.StandardSave(ppo_state))
        # logger.info(f"Epoch {epoch} - Policy Loss: {policy_loss}, Value Loss: {value_loss}")
    wandb.finish()


def get_env_config(ARGS: Namespace, key_param: jax.Array) \
        -> Tuple[flax.linen.Module, Environment, Callable[[jax.Array], TEnvParams], LOG_TYPE, Outer_param_fn]:
    env_name = ARGS.env.lower()
    if env_name == "blockenv":
        config = {"n_nodes": ARGS.n_nodes, "gat_arch": ARGS.gat_arch, "voting_nodes": ARGS.voting_nodes,
                  "reward_weights": ARGS.reward_weights}
    elif env_name == "blockenv_close_map":
        assert ARGS.ref_map_file is not None, "ref_map_file must be provided for blockenv_close_map"
        config = {"gat_arch": ARGS.gat_arch, "voting_nodes": ARGS.voting_nodes,
                  "reward_weights": ARGS.reward_weights, "ref_map_file": ARGS.ref_map_file}
    elif env_name == "cartpole":
        config = {}
    else:
        raise ValueError(
            f"Unknown environment: {env_name}. Available environments: {GenericEnvFactory.available_environments()}")
    model, env, _, create_params_fn, log_fn = GenericEnvFactory.create(env_name, key_param, config)
    update_params_fn = change_val_param_fn if ARGS.update_params else white_param_fn
    return model, env, create_params_fn, log_fn, update_params_fn


def eval_ppo_run(args: Namespace):
    key = jax.random.PRNGKey(args.seed)
    key_eval, state_key, key_param = jax.random.split(key, 3)

    model, env, create_params_fn, log_fn, update_params_fn = get_env_config(args, key_param)

    chkpt_dir: pathlib.Path = args.chkpt_dir
    ppo_state = load_ppo_state(chkpt_dir, state_key)

    # Evaluate the PPO agent
    metrics = eval_ppo(
        ppo_state=ppo_state,
        env=env,
        key=key_eval,
        model=model,
        create_params_fn=create_params_fn,
        update_params_fn=update_params_fn,
        num_episodes=args.eval_episodes,
        recorded_episodes=5,
        log_fn=log_fn
    )

    logger.info(metrics)
    print(metrics)
