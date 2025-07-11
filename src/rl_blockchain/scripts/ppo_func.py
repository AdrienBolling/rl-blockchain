import logging
import re
from argparse import Namespace

import jax
import numpy as np
import orbax.checkpoint as ocp
import wandb
from tqdm import tqdm

from rl_blockchain.BlockEnv import BlockchainEnv, StaticEnvParams
from rl_blockchain.BlockEnv import EnvParams
from rl_blockchain.algo.ppo import create_checkpoint_manager, create_ppo_state, train_epoch
from rl_blockchain.algo.ppo import eval_ppo as ev_ppo
from rl_blockchain.scripts.parser import REF_FILENAME

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

    env_params = EnvParams.create_random(ARGS.n_nodes, key_param, ARGS.voting_nodes, ARGS.reward_weights)
    static_params = StaticEnvParams.create(ARGS.n_nodes, REF_FILENAME[ARGS.n_nodes])
    env = BlockchainEnv(env_params, static_params)

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

    gat1_out, gat2_out, gat2_nodes_out = ARGS.gat_arch

    # Create the PPO state
    ppo_state = create_ppo_state(
        checkpoint_manager=checkpoint_manager,
        resume_dir=chkpt_dir if ARGS.checkpoint else None,
        warm_start=ARGS.warm_start,
        env=env,
        seed=ARGS.seed,
        lr=lr,
        gat1_out=gat1_out,
        gat2_out=gat2_out,
        gat2_nodes_out=gat2_nodes_out,
    )

    # Train the PPO agent
    for epoch in tqdm(range(num_epochs)):
        # Train for one epoch
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        ppo_state, sub_epoch = train_epoch(ppo_state=ppo_state, epoch=epoch, env=env, num_steps=num_steps,
                                           num_envs=num_envs, batch_size=batch_size, lr=lr, gamma=gamma,
                                           lambda_=lambda_, clip_ratio=clip_ratio, gat1_out=gat1_out, gat2_out=gat2_out,
                                           gat2_nodes_out=gat2_nodes_out, sub_epoch=sub_epoch, to_log=True)
        key, subkey = jax.random.split(key)
        if epoch % ARGS.eval_interval == 0:
            logger.info(f"Evaluating PPO agent at epoch {epoch + 1}/{num_epochs}")
            # Evaluate the PPO agent
            metrics = ev_ppo(
                ppo_state=ppo_state,
                env=env,
                num_episodes=ARGS.eval_episodes,
                key=subkey,
                gat_1_out=gat1_out,
                gat_2_out=gat2_out,
                gat_2_nodes_out=gat2_nodes_out,
            )


            wandb.log({"eval": metrics}, step=sub_epoch)
            logger.info(metrics)

        # Save the checkpoint
        checkpoint_manager.save(step=epoch, args=ocp.args.StandardSave(ppo_state))
        # logger.info(f"Epoch {epoch} - Policy Loss: {policy_loss}, Value Loss: {value_loss}")
    wandb.finish()


def eval_ppo():
    pass
