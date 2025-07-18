import logging
import re
import string
from argparse import Namespace

import jax
import optax
import orbax.checkpoint as ocp
import wandb
from tqdm import tqdm

from rl_blockchain.algo.ppo import create_checkpoint_manager, create_ppo_state, train_epoch
from rl_blockchain.algo.ppo import eval_ppo as ev_ppo
from rl_blockchain.scripts.env_factory import GenericEnvFactory

logger = logging.getLogger(__name__)


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
    lr = ARGS.learning_rate
    gamma = ARGS.gamma
    lambda_ = ARGS.lambda_
    clip_ratio = ARGS.clip_ratio
    key = jax.random.PRNGKey(ARGS.seed)
    key, key_param = jax.random.split(key)
    # Create environment parameters

    sub_epoch = 0

    env_name =  ARGS.env.lowercase()
    if env_name == "blockchain":
        config = {"n_nodes": ARGS.n_nodes, "gat_arch": ARGS.gat_arch, "voting_nodes": ARGS.voting_nodes,
                  "reward_weights": ARGS.reward_weights}
        model, env, first_param, create_params_fn = GenericEnvFactory.create("blockchain", key_param, config)
    elif env_name == "cartpole":
        config = {}
        model, env, first_param, create_params_fn = GenericEnvFactory.create("cartpole", key_param, config)
    else:
        raise ValueError(f"Unknown environment: {env_name}. Available environments: {GenericEnvFactory.available_environments()}")

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
        lr=lr,
        model=model
    )

    model_opt = optax.adam(lr)

    # Train the PPO agent
    for epoch in tqdm(range(num_epochs)):
        # Train for one epoch
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        ppo_state, sub_epoch = train_epoch(ppo_state=ppo_state, epoch=epoch, env=env, model_opt=model_opt, model=model,
                                           create_params_fn=create_params_fn,
                                           num_steps=num_steps,
                                           num_envs=num_envs, batch_size=batch_size, gamma=gamma,
                                           lambda_=lambda_, clip_ratio=clip_ratio, sub_epoch=sub_epoch, to_log=True,
                                           normalize_rewards=True)
        key, subkey = jax.random.split(key)
        if epoch % ARGS.eval_interval == 0:
            logger.info(f"Evaluating PPO agent at epoch {epoch + 1}/{num_epochs}")
            # Evaluate the PPO agent
            metrics = ev_ppo(
                ppo_state=ppo_state,
                env=env,
                model=model,
                num_episodes=ARGS.eval_episodes,
                key=subkey,
            )

            wandb.log({"eval": metrics}, step=sub_epoch)
            logger.info(metrics)

        # Save the checkpoint
        checkpoint_manager.save(step=epoch, args=ocp.args.StandardSave(ppo_state))
        # logger.info(f"Epoch {epoch} - Policy Loss: {policy_loss}, Value Loss: {value_loss}")
    wandb.finish()


def eval_ppo():
    pass
