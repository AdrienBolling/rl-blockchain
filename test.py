import jax

from rl_blockchain.algo.ppo import train_ppo, eval_ppo_and_log
from rl_blockchain.scripts.env_factory import white_param_fn, GenericEnvFactory, change_val_param_fn


def main():
    # Hyperparameters
    num_steps = 10000  # steps per rollout
    num_envs = 1  # parallel environments
    num_epochs = 5  # training epochs
    batch_size = 32
    lr = 3e-4
    gamma = 0.99
    lambda_ = 0.95
    clip_ratio = 0.2

    # ----- Random Key -----
    # Set the random key for reproducibility
    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)

    # Create environment params

    config = {"n_nodes": 25, "gat_arch": [4, 4, 4], "voting_nodes": 3,
              "reward_weights": [0.5, 0.5]}
    model, env, env_params, create_params_fn, log_fn = GenericEnvFactory.create("blockenv", key, config)
    update_params_fn = change_val_param_fn

    key, subkey = jax.random.split(subkey)

    # ===== Training =====
    print("[TRAIN] Starting PPO training...")
    # Note: train_ppo currently prints metrics but does not return the final state.
    # If you update train_ppo to return PPOState, you can capture it like:
    # final_state = train_ppo(...)
    ppo_state = train_ppo(env, model, create_params_fn, num_steps, num_envs, num_epochs, batch_size, lr, gamma, lambda_,
                          clip_ratio, subkey)
    key, subkey = jax.random.split(subkey)
    # ===== Evaluation =====
    print("[EVAL] Running evaluation with default (random) policy...")
    # Since train_ppo does not return the trained state, this will evaluate the initial policy.
    # To evaluate the truly trained policy, modify train_ppo to return PPOState and pass that here.
    eval_ppo_and_log(env, model, ppo_state, num_episodes=1, key=subkey)
    print("[EVAL] Evaluation completed.")


if __name__ == "__main__":
    main()
