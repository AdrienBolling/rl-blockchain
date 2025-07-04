import jax
import jax.numpy as jnp
import numpy as np
import wandb
import re
from rl_blockchain.algo.ppo import create_checkpoint_manager, create_ppo_state, train_epoch
from rl_blockchain.algo.ppo import eval_ppo as ev_ppo
import os
import logging
import orbax.checkpoint as ocp
from rl_blockchain.rl.env import BlockchainEnv_intermediary
from tqdm import tqdm

logger = logging.getLogger(__name__)

def make_env_params(key, n_nodes, voting_nodes):
    # Create a random symmetric distance matrix
    A = jax.random.uniform(key, (n_nodes, n_nodes))
    dist = (A + A.T) * 0.5
    # Create environment parameters
    env_params = {
        "node_distance_matrix": dist,
        "voting_nodes": voting_nodes,
        "random_key": key,
    }
    return env_params

def log_to_wandb(logs: dict, mode: str = "train"):
    # Wandb is already initialized in the main script, so we just need to log the arguments
    if "mode" == "train+eval":
        aggregated_eval_metrics = {}
        avg_return = np.asarray(logs["eval"]["return"]).mean()
        avg_length = np.asarray(logs["eval"]["length"]).mean()
        individual_rewards = np.asarray(logs["eval"]["infos"]["rewards"]).mean(axis=0)
        aggregated_eval_metrics["avg_return"] = avg_return
        aggregated_eval_metrics["avg_length"] = avg_length
        for i, rew in enumerate(logs["eval"]["infos"]["rewards"]):
            aggregated_eval_metrics[f"reward_{i}"] = individual_rewards[i]
        
        
    elif mode == "train":
        # If we are in training mode, we log the training metrics
        aggregated_eval_metrics = {
        }
        
    logs["eval"] = aggregated_eval_metrics
    wandb.log(logs, step=logs["train"]["epoch"])

def train_ppo(ARGS):
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
    reward_weights = jnp.array(ARGS.reward_weights)
    key = jax.random.PRNGKey(ARGS.seed)
    key, subkey = jax.random.split(key)
    # Create environment parameters
    env_params = make_env_params(
        key=subkey,
        n_nodes=ARGS.n_nodes,
        voting_nodes=ARGS.voting_nodes
    )
    
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
    
    # Create the environment
    env_fn = BlockchainEnv_intermediary
    
    gat1_out, gat2_out, gat2_nodes_out = ARGS.gat_arch
    
    # Create the PPO state
    ppo_state = create_ppo_state(
        checkpoint_manager=checkpoint_manager,
        resume_dir=chkpt_dir if ARGS.checkpoint else None,
        warm_start=ARGS.warm_start,
        env_fn=env_fn,
        env_params=env_params,
        seed=ARGS.seed,
        lr=lr,
        gat1_out=gat1_out,
        gat2_out=gat2_out,
        gat2_nodes_out=gat2_nodes_out,
    )
    
    # Train the PPO agent
    for epoch in tqdm(range(num_epochs)):
        # Train for one epoch
        ppo_state, policy_loss, value_loss = train_epoch(
            ppo_state=ppo_state,
            epoch=epoch,
            env_fn=env_fn,
            env_params=env_params,
            num_steps=num_steps,
            num_envs=num_envs,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            lambda_=lambda_,
            clip_ratio=clip_ratio,
            reward_weights=reward_weights,
            gat1_out=gat1_out,
            gat2_out=gat2_out,
            gat2_nodes_out=gat2_nodes_out,
        )
        key, subkey = jax.random.split(key)
        if epoch % ARGS.eval_interval == 0:
            # Evaluate the PPO agent
            metrics = ev_ppo(
                ppo_state=ppo_state,
                env_fn=env_fn,
                env_params=env_params,
                reward_weights=reward_weights,
                num_episodes=ARGS.eval_episodes,
                key=subkey,
                gat_1_out=gat1_out,
                gat_2_out=gat2_out,
                gat_2_nodes_out=gat2_nodes_out,
            )
            
            logs = {
                "train": {
                    "epoch": epoch,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                },
                "eval": metrics,
            }
            
            log_to_wandb(logs, mode="train+eval")
        else:
            logs = {
                "train": {
                    "epoch": epoch,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                }
            }
            log_to_wandb(logs, mode="train")
            
        
        # Save the checkpoint
        checkpoint_manager.save(step=epoch, args=ocp.args.StandardSave(ppo_state))
        logger.info(f"Epoch {epoch} - Policy Loss: {policy_loss}, Value Loss: {value_loss}")
        
        
def eval_ppo():
    pass