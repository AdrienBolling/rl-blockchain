import gymnax
import jax
import jax.numpy as jnp

from rl_blockchain.BlockEnv import create_rd_adj_matrix, EnvParams, BlockchainEnv, StaticEnvParams
from rl_blockchain.algo.ppo import train_ppo, eval_ppo_and_log
from rl_blockchain.model import CategoricalSeparateMLP
from rl_blockchain.scripts.env_factory import white_param_fn


def main():
    # Hyperparameters
    num_steps = 2048*4                # steps per rollout
    num_envs  = 8                  # parallel environments
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



    env, env_params = gymnax.make("CartPole-v1")


    # ===== Training =====
    print("[TRAIN] Starting PPO training...")
    # Note: train_ppo currently prints metrics but does not return the final state.
    # If you update train_ppo to return PPOState, you can capture it like:
    # final_state = train_ppo(...)

    print("num_actions:", env.num_actions)
    mlpCat = CategoricalSeparateMLP(env.num_actions, 64, 2)
    create_params_fn = lambda key_map : env_params
    ppo_state = train_ppo(env, mlpCat, create_params_fn, white_param_fn, num_steps, num_envs, num_epochs, batch_size, lr, gamma,
                          lambda_, clip_ratio, subkey)
    key, subkey = jax.random.split(subkey)
    # ===== Evaluation =====
    print("[EVAL] Running evaluation with default (random) policy...")
    # Since train_ppo does not return the trained state, this will evaluate the initial policy.
    # To evaluate the truly trained policy, modify train_ppo to return PPOState and pass that here.
    eval_ppo_and_log(env, mlpCat, ppo_state, num_episodes=1, key=subkey)
    print("[EVAL] Evaluation completed.")


if __name__ == "__main__":
    main()