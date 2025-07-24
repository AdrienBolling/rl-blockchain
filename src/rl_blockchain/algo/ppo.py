import logging
import os
from functools import partial
from typing import Optional, Tuple, Union, Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph as jr
import optax
import orbax.checkpoint as ocp
import wandb
from flax import struct
from gymnax.environments import environment
from gymnax.environments.environment import TEnvParams
from pathlib import Path
from optax._src.base import GradientTransformationExtraArgs

from rl_blockchain.BlockEnv import EnvParams
from rl_blockchain.BlockEnv.BlockEnv import BlockchainEnv
from rl_blockchain.BlockEnv.NormailzationWrapper import NormalizationWrapper
from rl_blockchain.scripts.env_factory import LOG_TYPE

logger = logging.getLogger(__name__)


@struct.dataclass
class PPOState:
    params: dict
    opt_state: optax.OptState
    rng_key: jnp.ndarray


@partial(jax.jit, static_argnames=('model', 'env', 'steps_in_episode'))
def rollout(key_input, env: environment.Environment,
            model: nn.Module, ppo_state: PPOState,
            env_params: EnvParams, steps_in_episode: int):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    key_reset, key_episode = jax.random.split(key_input)
    first_obs, first_state = env.reset(key_reset, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, key = state_input
        next_key, key_step, key_net = jax.random.split(key, 3)
        value, action_distribution = model.apply(ppo_state.params, obs, )
        action = action_distribution.sample(seed=key_net)
        logp = action_distribution.log_prob(action)

        next_obs, next_state, reward, done, infos = env.step(
            key_step, state, action, env_params
        )

        carry = [next_obs, next_state, next_key]
        traj = (obs, action, logp, reward, done, value, infos)
        return carry, traj

    # Scan over episode step loop
    (obs_end, _, _), trajs = jax.lax.scan(
        policy_step,
        [first_obs, first_state, key_episode],
        None,
        steps_in_episode
    )

    last_value, _ = model.apply(ppo_state.params, obs_end)
    # Return masked sum of rewards accumulated by agent in episode
    observations, actions, logps, rewards, dones, values, infos = trajs
    return observations, actions, logps, rewards, dones, values, last_value, infos


@partial(jax.jit, static_argnames=('model', 'env', 'steps_in_episode'))
def rollout_eval(key_input, env: environment.Environment,
                 model: nn.Module, ppo_state: PPOState,
                 env_params: EnvParams, steps_in_episode: int):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    key_reset, key_episode = jax.random.split(key_input)
    first_obs, first_state = env.reset(key_reset, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, key = state_input
        next_key, key_step, key_net = jax.random.split(key, 3)
        _, action_distribution = model.apply(ppo_state.params, obs)
        action = action_distribution.mode()

        next_obs, next_state, reward, done, infos = env.step(
            key_step, state, action, env_params
        )

        carry = [next_obs, next_state, next_key]
        traj = (obs, action, reward, done, infos)
        return carry, traj

    # Scan over episode step loop
    (obs_end, _, _), trajs = jax.lax.scan(
        policy_step,
        [first_obs, first_state, key_episode],
        None,
        steps_in_episode
    )

    # Return masked sum of rewards accumulated by agent in episode
    observations, actions, rewards, dones, infos = trajs
    return observations, actions, rewards, dones, infos


@jax.jit
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lambda_=0.95):
    values = jnp.concatenate([values, last_value[None]], axis=0)

    def fn(carry, idx):
        adv, next_val = carry
        r = rewards[idx]
        v = values[idx]
        d = dones[idx]
        delta = r + gamma * next_val * (1 - d) - v
        adv = delta + gamma * lambda_ * adv * (1 - d)
        return (adv, v), adv

    (_, _), advs = jax.lax.scan(
        fn, (0.0, last_value), jnp.arange(values.shape[0] - 1)[::-1]
    )
    return advs[::-1]


@partial(jax.jit, static_argnames=('model_apply', 'model_optimizer', 'clip_ratio'))
def update_ppo(
        ppo_state: PPOState,
        observation: jr.GraphsTuple,
        actions: jnp.ndarray,
        old_logps: jnp.ndarray,
        returns: jnp.ndarray,
        advantages: jnp.ndarray,
        old_values: jnp.ndarray,
        model_apply,
        model_optimizer,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
) -> tuple[PPOState, float, float, float, dict[str, Any]]:
    """
    Performs a PPO update over a batch of transitions.

    Args:
        ppo_state: Current PPOState
        observation: Batched GraphsTuple of observations, shape [B, ...]
        actions: Actions array, shape [B, action_dim]
        old_logps: Log probabilities under old policy, shape [B]
        returns: Discounted returns, shape [B]
        advantages: GAE advantages, shape [B]
        policy_apply: Policy network apply function
        value_apply: Value network apply function
        policy_optimizer: Optax optimizer for policy
        value_optimizer: Optax optimizer for value
        clip_ratio: PPO clipping parameter
        value_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy regularization

    Returns:
        new_state: Updated PPOState
        mean_policy_loss: Scalar
        mean_value_loss: Scalar
    """

    info_coef = {
        "clip_ratio": clip_ratio,
        "value_coef": value_coef,
        "entropy_coef": entropy_coef,
    }

    # Loss function with aux outputs
    def loss_fn(model_params):
        # compute per-sample losses
        def sample_loss(m_params, graph, a, old_lp, ret, adv, old_val):
            value_pred, dist = model_apply(m_params, graph)
            new_lp = dist.log_prob(a)
            ratio = jnp.exp(new_lp - old_lp)

            clipp_actor = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -jnp.minimum(ratio * adv, clipp_actor * adv)
            entropy = dist.entropy()

            value_pred_clipped = old_val + (value_pred - old_val).clip(
                -clip_ratio, clip_ratio)
            value_loss = jnp.square(value_pred - ret)
            value_loss_clipped = jnp.square(value_pred_clipped - ret)
            value_loss = jnp.maximum(value_loss, value_loss_clipped)

            total_loss = policy_loss + value_coef * value_loss - entropy * entropy_coef
            return total_loss, (policy_loss, value_loss, entropy)

        # Vectorize over batch
        total_loss, (pl_batch, vl_batch, ent_batch) = jax.vmap(
            sample_loss,
            in_axes=(None, 0, 0, 0, 0, 0, 0),
            out_axes=(0, (0, 0, 0))
        )(
            model_params,
            observation,
            actions,
            old_logps,
            returns,
            advantages,
            old_values
        )
        # total_loss is array of shape [B], pl_batch/ vl_batch each shape [B]
        mean_loss = jnp.mean(total_loss)
        mean_pl_batch = jnp.mean(pl_batch)
        mean_vl_batch = jnp.mean(vl_batch)
        mean_ent_batch = jnp.mean(ent_batch)
        # return mean total_loss as loss, and policy/value losses as aux
        return mean_loss, (mean_pl_batch, mean_vl_batch, mean_ent_batch)

    # Compute gradients
    (loss_val, (mean_pl, mean_vl, mean_ent)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(ppo_state.params)

    model_updates, new_model_opt_state = model_optimizer.update(
        grads, ppo_state.opt_state
    )

    new_model_params = optax.apply_updates(ppo_state.params, model_updates)

    # Construct new state
    new_state = ppo_state.replace(
        params=new_model_params,
        opt_state=new_model_opt_state,
    )

    return new_state, mean_pl, mean_vl, mean_ent, info_coef


def compute_avg_value(infos: dict[str, jax.Array]) -> dict[str, jax.Array]:
    infos_keys = ["gini", "distance", "gini_reward", "distance_reward"]
    list_is_inner: jax.Array = infos["action_taken"] == -1
    sum_inner = list_is_inner.sum()
    returned_infos = {}
    for key in infos_keys:
        returned_infos[key] = ((infos[key] * list_is_inner).sum() / sum_inner).item()
    return returned_infos


def train_ppo(
        env: BlockchainEnv,
        model: nn.module,
        create_params_fn: Callable[[jax.Array], TEnvParams],
        num_steps,
        num_envs,
        num_epochs,
        batch_size,
        lr,
        gamma,
        lambda_,
        clip_ratio,
        key,
) -> PPOState:
    # init_state = env.reset()

    obs_key, pol_key, val_key, ppo_key = jax.random.split(key, 4)

    first_obs, first_state = env.reset(obs_key, env.default_params)

    # Initialize with GraphsTuple

    model_vars = model.init(obs_key, first_obs)

    model_opt = optax.adam(lr)
    model_opt_state = model_opt.init(model_vars)
    ppo_state = PPOState(model_vars, model_opt_state, ppo_key)

    # rollout fns expect graph inputs inside rollout
    @jax.jit
    def single_rollout(rng: jax.Array, new_param: EnvParams):
        return rollout(
            rng,
            env,
            model,
            ppo_state,
            new_param,
            num_steps,
        )

    vm_rollout = jax.vmap(single_rollout)

    params_map = jax.vmap(
        lambda key_map: create_params_fn(key_map)  # Create new params for each env,
    )

    for epoch in range(num_epochs):
        new_ppo_key, rollout_key, params_key, permutation_key = jax.random.split(ppo_state.rng_key, 4)
        subkeys = jax.random.split(rollout_key, num_envs)
        subkeys_params = jax.random.split(params_key, num_envs)

        params_list = params_map(subkeys_params)

        observations, acts, logps, rews, dones, vals, last_values, infos = vm_rollout(subkeys, params_list)
        # refined_value = compute_avg_value(infos)

        # Extract graphs and last graphs
        # GAE over each env
        advantages = jax.vmap(
            lambda r, v, d, last_value: compute_gae(
                r,
                v,
                d,
                last_value,
                gamma,
                lambda_,
            )
        )(rews, vals, dones, last_values)
        returns = advantages + vals
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Helper to flatten env × time dims
        def flatten(x):
            return x.reshape(-1, *x.shape[2:])

        # Flatten your action/logp/return/adv arrays
        flat_a = flatten(acts)
        flat_lp = flatten(logps)
        flat_r = flatten(returns)
        flat_adv = flatten(advantages_norm)
        flat_val = flatten(vals)

        # Flatten *each* leaf in the GraphsTuple of states.blockchain
        flat_graphs = jax.tree.map(flatten, observations)

        # Permute to get randomized minibatches
        idx = jax.random.permutation(permutation_key, flat_a.shape[0])
        for start in range(0, idx.shape[0], batch_size):
            batch_idx = idx[start: start + batch_size]

            # Slice out a minibatch of graphs
            batch_graphs = jax.tree.map(lambda x: x[batch_idx], flat_graphs)

            # Now call update_ppo with the exact signature you defined:
            ppo_state, policy_loss, value_loss, entropy, info_coef = update_ppo(
                ppo_state,
                batch_graphs,  # env_states: a GraphsTuple PyTree
                flat_a[batch_idx],  # actions
                flat_lp[batch_idx],  # old_logps
                flat_r[batch_idx],  # returns
                flat_adv[batch_idx],  # advantages
                flat_val[batch_idx],
                model.apply,  # policy_apply
                model_opt,  # policy_optimizer (optax.OptState)
                clip_ratio  # clip_ratio
            )
        v, pi = model.apply(ppo_state.params, first_obs)
        print(f"Policy: {pi.logits}, Value: {v}")
        ppo_state = ppo_state.replace(rng_key=new_ppo_key)
        print(f"Epoch {epoch}: PolicyLoss={policy_loss:.3f}, ValueLoss={value_loss:.3f}, Entropy={entropy:.3f}")
        print(f"rewards : {rews.sum():.3f}, longueur {rews.shape}, dones {dones.sum():.3f}")
        # print("Infos -> ", infos)
        # string_builder = ""
        # for key, value in refined_value.items():
        #     string_builder += f"{key}: {value:.3f}, "
        # print("Infos Value -> ", string_builder)

    return ppo_state


def eval_ppo_and_log(env: BlockchainEnv, model: nn.module, ppo_state: PPOState, num_episodes: int = 10, key=None):
    returns = []
    env_params = env.default_params
    for _ in range(num_episodes):
        key, subkey_mat, subkey_st = jax.random.split(key, 3)
        # temp_params = EnvParams.create_random(env.nb_nodes, subkey_mat, env_params.nb_validators,
        #                                       env_params.rewards_weights)
        temp_params = env_params  # Use default params for evaluation
        obs, st = env.reset(subkey_st, temp_params)
        done = False
        tot = 0.0
        while not done:
            key, subkey = jax.random.split(key)
            _, dist = model.apply(ppo_state.params, obs)
            a = dist.mode()
            obs, st, r, done, _ = env.step(subkey, st, a, temp_params)
            tot += r
        returns.append(tot)
    avg = sum(returns) / len(returns)
    print(f"Eval over {num_episodes} eps: avg return={avg:.3f}")


def train_epoch(ppo_state: PPOState, epoch: int, env: environment.Environment, model: nn.module, num_steps: int,
                num_envs: int,
                create_params_fn: Callable[[jax.Array], TEnvParams],
                batch_size: int,
                model_opt: GradientTransformationExtraArgs, gamma: float, lambda_: float,
                clip_ratio: float, normalize_rewards: bool = False, log_fn: LOG_TYPE = None,
                sub_epoch: int = 0, value_coef: float = 0.5, entropy_coef: float = 0.01, ) -> Tuple[PPOState, int]:
    """
    Perform one PPO training epoch using the provided hyperparameters.
    Returns the updated PPOState.
    """

    # TODO normalization of rewards
    if normalize_rewards and isinstance(env, BlockchainEnv):
        # If using normalization, ensure the environment is wrapped accordingly
        env = NormalizationWrapper(env)

    # Vectorized rollout
    @jax.jit
    def single_rollout(rng: jax.Array, new_param: EnvParams):
        return rollout(
            rng,
            env,
            model,
            ppo_state,
            new_param,
            num_steps,
        )

    vm_rollout = jax.vmap(single_rollout)

    params_map = jax.vmap(
        # lambda key_map: EnvParams.create_random(env.nb_nodes, key_map, env.default_params.nb_validators,
        #                                        env.default_params.rewards_weights),
        lambda key_map: create_params_fn(key_map)
    )

    # Split RNG keys for rollouts and parameter sampling
    rollout_key, params_key, perm_key, new_ppo_key = jax.random.split(ppo_state.rng_key, 4)

    subkeys = jax.random.split(rollout_key, num_envs)
    subkeys_params = jax.random.split(params_key, num_envs)

    params_list = params_map(subkeys_params)
    observations, acts, logps, rews, dones, vals, last_values, infos_env = vm_rollout(subkeys, params_list)
    logger.info(f"Epoch {epoch}: Collected {num_steps * num_envs} steps.")

    if log_fn is not None:
        infos_env_refined = log_fn(infos_env, rews, dones)
        wandb.log({"env": infos_env_refined}, step=sub_epoch)

    # Compute advantages and returns
    advantages = jax.vmap(
        lambda r, v, d, last_value: compute_gae(
            r,
            v,
            d,
            last_value,
            gamma,
            lambda_,
        )
    )(rews, vals, dones, last_values)
    returns = advantages + vals
    advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten data
    def flatten(x):
        return x.reshape(-1, *x.shape[2:])

    flat_a = flatten(acts)
    flat_lp = flatten(logps)
    flat_r = flatten(returns)
    flat_adv = flatten(advantages_norm)
    flat_val = flatten(vals)

    flat_graphs = jax.tree.map(flatten, observations)

    # Shuffle and minibatch updates
    perm = jax.random.permutation(perm_key, flat_a.shape[0])
    for start in range(0, perm.shape[0], batch_size):
        logger.info(f"Epoch {epoch}: Processing batch {start // batch_size + 1} / {perm.shape[0] // batch_size + 1}")
        idx = perm[start: start + batch_size]
        batch_graphs = jax.tree.map(lambda x: x[idx], flat_graphs)
        ppo_state, policy_loss, value_loss, entropy, info_coef = update_ppo(
            ppo_state,
            batch_graphs,
            flat_a[idx],
            flat_lp[idx],
            flat_r[idx],
            flat_adv[idx],
            flat_val[idx],
            model.apply,
            model_opt,
            clip_ratio,
            value_coef, entropy_coef
        )
        info_train = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }
        if log_fn is not None:
            wandb.log({"coef": info_coef, "train": info_train, "epoch": epoch}, step=sub_epoch)
        sub_epoch += 1
        logger.info(f"Epoch {epoch}/batch {start} - Policy Loss: {policy_loss}, Value Loss: {value_loss}")

    # Update RNG and log progress
    ppo_state = ppo_state.replace(rng_key=new_ppo_key)

    return ppo_state, sub_epoch


def create_checkpoint_manager(
        checkpoint_dir: Union[str, Path],
        max_to_keep: int = 5,
        save_interval_steps: int = 1
) -> ocp.CheckpointManager:
    """
    Build and return an Orbax CheckpointManager that will keep at most
    `max_to_keep` checkpoints and only saves every `save_interval_steps`.
    """
    # make sure the directory exists

    os.makedirs(checkpoint_dir, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
        create=True,
    )
    manager = ocp.CheckpointManager(
        str(checkpoint_dir),
        options=options,
    )
    return manager


def eval_ppo(ppo_state: PPOState, env: environment.Environment, model: nn.Module, key: jax.Array,
             create_params_fn: Callable[[jax.Array], TEnvParams], num_episodes: int = 10, log_fn: LOG_TYPE = None):
    @jax.jit
    def single_rollout(rng: jax.Array, new_param: EnvParams):
        return rollout_eval(rng, env, model, ppo_state, new_param, env.default_params.max_steps_in_episode)

    vm_rollouts = jax.vmap(single_rollout)

    params_map = jax.vmap(
        lambda key_map: create_params_fn(key_map)
    )

    # RNG split
    rollout_key, params_key = jax.random.split(key)
    subkeys = jax.random.split(rollout_key, num_episodes)
    subkeys_params = jax.random.split(params_key, num_episodes)

    params_list = params_map(subkeys_params)
    _, _, rews, dones, infos = vm_rollouts(subkeys, params_list)
    logger.info(f"Evaluated {num_episodes} episodes.")

    metrics = log_fn(infos, rews, dones)

    metrics["avg_returns_episode"] = rews.sum(axis=1).mean().tolist()
    sub_rewards = rews.mean(axis=1).tolist()
    for i, rew in enumerate(sub_rewards):
        metrics[f"reward_{i}"] = rew

    return metrics


# Modified create_ppo_state to use the manager
def create_ppo_state(
        checkpoint_manager: ocp.CheckpointManager,
        resume_dir: Optional[Path],
        warm_start: bool,
        env: environment.Environment,
        seed: int,
        lr: float,
        model: nn.Module,
) -> PPOState:
    """
    Initialize or restore a PPOState.  If `resume_dir` is provided, uses
    `checkpoint_manager` to restore the latest checkpoint; if `warm_start`
    is True, reinitializes optimizer states with loaded network weights.
    Otherwise, does a fresh init.
    """
    model_opt = optax.adam(lr)

    # --- restore path ---
    if resume_dir:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found in {resume_dir}")
        # restore the entire PPOState PYTree
        state: PPOState = checkpoint_manager.restore(step)
        print(f"Loaded checkpoint from step {step}")
        if warm_start:
            state = state.replace(
                opt_state=model_opt.init(state.params),
            )
            print("Optimizer states reinitialized for warm start.")
        return state

    # --- fresh initialization ---
    key = jax.random.PRNGKey(seed)
    obs_key, ppo_key, state_key = jax.random.split(key, 3)
    first_obs, first_state = env.reset(obs_key, env.default_params)
    model_vars = model.init(ppo_key, first_obs)
    model_opt_state = model_opt.init(model_vars)
    print("Initialized new PPOState.")
    return PPOState(model_vars, model_opt_state, ppo_key)


def load_ppo_state(resume_dir: Path, key:jax.Array) -> PPOState:
    """
    Load a PPOState from a checkpoint or initialize a new one.
    """
    checkpoint_manager = create_checkpoint_manager(resume_dir.absolute())
    step = checkpoint_manager.latest_step()
    if step is None:
        raise ValueError(f"No checkpoints found in {resume_dir}")
    # restore the entire PPOState PYTree
    restored_state = checkpoint_manager.restore(step)
    state = PPOState(
        params=restored_state["params"],
        opt_state=None,
        rng_key=key
    )
    print(type(state))
    print(f"Loaded checkpoint from step {step}")
    return state
