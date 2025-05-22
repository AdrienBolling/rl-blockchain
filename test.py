import jax
import jax.numpy as jnp
from rl_blockchain.algo.ppo import train_ppo, eval_ppo_and_log
from rl_blockchain.rl.env import BlockchainEnv_intermediary


def make_env_params(key: jnp.ndarray,
                     n_nodes: int = 5,
                     voting_nodes: int = 2):
    """
    Create synthetic environment parameters for a quick test run.
    - n_nodes: number of nodes in the graph
    - n_features: size of the node feature vector (unused by env except for shaping)
    - voting_nodes: number of votes per outer step
    """
    # Split the key for reproducibility
    key, subkey = jax.random.split(key)
    # Random symmetric distance matrix
    A = jax.random.uniform(key, (n_nodes, n_nodes))
    dist = (A + A.T) * 0.5
    return {
        "node_distance_matrix": dist,
        "voting_nodes": voting_nodes,
        "random_key": subkey,
    }


def main():
    # Hyperparameters
    num_steps = 100                # steps per rollout
    num_envs  = 4                  # parallel environments
    num_epochs = 5                 # training epochs
    batch_size = 32
    lr         = 3e-4
    gamma      = 0.99
    lambda_    = 0.95
    clip_ratio = 0.2
    
    # ----- Random Key -----
    # Set the random key for reproducibility
    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)

    # Create environment params
    env_params = make_env_params(key=subkey,
                                 n_nodes=6,
                                 n_features=4,
                                 voting_nodes=3)
    
    key, subkey = jax.random.split(subkey)

    # ===== Training =====
    print("[TRAIN] Starting PPO training...")
    # Note: train_ppo currently prints metrics but does not return the final state.
    # If you update train_ppo to return PPOState, you can capture it like:
    # final_state = train_ppo(...)
    ppo_state = train_ppo(
        BlockchainEnv_intermediary,
        env_params,
        num_steps, num_envs, num_epochs,
        batch_size, lr,
        gamma, lambda_, clip_ratio, subkey
    )
    key, subkey = jax.random.split(subkey)
    # ===== Evaluation =====
    print("[EVAL] Running evaluation with default (random) policy...")
    # Since train_ppo does not return the trained state, this will evaluate the initial policy.
    # To evaluate the truly trained policy, modify train_ppo to return PPOState and pass that here.
    eval_ppo_and_log(
        BlockchainEnv_intermediary,
        env_params,
        ppo_state,
        reward_weights=jnp.array([0.5, 0.5]),
        num_episodes=1,
        key=subkey,
    )
    print("[EVAL] Evaluation completed.")


if __name__ == "__main__":
    main()