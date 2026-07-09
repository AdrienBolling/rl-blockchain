import logging
import os
from functools import partial, lru_cache
from pathlib import Path
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
from optax._src.base import GradientTransformationExtraArgs

from rl_blockchain.BlockEnv import EnvParams
from rl_blockchain.BlockEnv.BlockEnv import BlockchainEnv, sample_subset_with_logp, mode_subset, logp_prefix_pl
from rl_blockchain.scripts.env_factory import LOG_TYPE

logger = logging.getLogger(__name__)


@struct.dataclass
class PPOState:
    params: dict
    opt_state: optax.OptState
    rng_key: jnp.ndarray


@lru_cache(maxsize=None)
def make_optimizer(lr: float) -> optax.GradientTransformation:
    """Build an Adam optimizer, memoized on the (python float) learning rate.

    ``update_ppo`` receives the optimizer as a *static* argument. A fresh
    ``optax.adam(lr)`` object never compares equal to another -- even for an
    identical lr (verified: ``hash(optax.adam(1e-3)) != hash(optax.adam(1e-3))``)
    -- so constructing one per epoch forces ``update_ppo`` to recompile every
    epoch, and under jax 0.10 each recompile re-runs the expensive Triton-GEMM
    autotuner. Memoizing on the lr value makes a constant schedule reuse a single
    stable object, so ``update_ppo`` compiles once. A changing lr still yields a
    new (but cached) object per distinct value.
    """
    return optax.adam(lr)


@partial(jax.jit, static_argnames=('model', 'env', 'steps_in_episode'))
def rollout(key_input, env: environment.Environment,
            model: nn.Module, ppo_state: PPOState,
            env_params_episode: EnvParams, steps_in_episode: int):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    key_reset, key_episode = jax.random.split(key_input)
    first_obs, first_state = env.reset(key_reset, env_params_episode)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, key = state_input
        next_key, key_step, key_net = jax.random.split(key, 3)
        value, action_distribution = model.apply(ppo_state.params, obs, )
        perm, action, logp = sample_subset_with_logp(key_net, action_distribution, state.nb_val)

        next_obs, next_state, reward, done, infos = env.step(
            key_step, state, action, env_params_episode
        )

        carry = [next_obs, next_state, next_key]
        traj = (obs, perm, logp, reward, done, value, infos)
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
    observations, perms, logps, rewards, dones, values, infos = trajs
    return observations, perms, logps, rewards, dones, values, last_value, infos


@partial(jax.jit, static_argnames=('model', 'env', 'steps_in_episode'))
def rollout_eval(key_input, env: environment.Environment,
                 model: nn.Module, ppo_state: PPOState,
                 env_params_episode: EnvParams,
                 steps_in_episode: int):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    key_reset, key_episode = jax.random.split(key_input)
    first_obs, first_state = env.reset(key_reset, env_params_episode)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, key = state_input
        next_key, key_step, key_net = jax.random.split(key, 3)
        _, action_distribution = model.apply(ppo_state.params, obs)
        action = mode_subset(action_distribution, state.nb_val)

        next_obs, next_state, reward, done, infos = env.step(
            key_step, state, action, env_params_episode
        )

        carry = [next_obs, next_state, next_key]
        # NOTE: do NOT emit obs/action here. eval only needs reward/done/infos;
        # stacking the full 200-node observation graph (~39.8k edges) over every
        # step x env is a large buffer that eval_ppo immediately discards and
        # that can OOM at max_steps_in_episode (~1000) -> looks like a hang.
        traj = (reward, done, infos)
        return carry, traj

    # Scan over episode step loop
    (obs_end, _, _), trajs = jax.lax.scan(
        policy_step,
        [first_obs, first_state, key_episode],
        None,
        steps_in_episode
    )

    rewards, dones, infos = trajs
    return rewards, dones, infos


@lru_cache(maxsize=None)
def _vectorized_rollout(env, model, steps_in_episode: int):
    """Build (once) a jitted, vmapped training rollout.

    ``ppo_state`` is a *traced* argument (in_axes=None) so the network weights
    are never baked in as constants. Cached on (env, model, steps) so the same
    compiled executable is reused across all epochs -> one compilation total
    instead of one per epoch.
    """

    def single(rng, ppo_state, new_param):
        return rollout(rng, env, model, ppo_state, new_param, steps_in_episode)

    return jax.jit(jax.vmap(single, in_axes=(0, None, 0)))


@lru_cache(maxsize=None)
def _vectorized_rollout_eval(env, model, steps_in_episode: int):
    """Same as :func:`_vectorized_rollout` for the deterministic eval rollout."""

    def single(rng, ppo_state, new_param):
        return rollout_eval(rng, env, model, ppo_state, new_param, steps_in_episode)

    return jax.jit(jax.vmap(single, in_axes=(0, None, 0)))


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


@partial(jax.jit, static_argnames=('model_apply', 'model_optimizer', 'clip_ratio', 'micro_batch_size'))
def update_ppo(
        ppo_state: PPOState,
        observation: jr.GraphsTuple,
        perms: jnp.ndarray,
        old_logps: jnp.ndarray,
        returns: jnp.ndarray,
        advantages: jnp.ndarray,
        old_values: jnp.ndarray,
        model_apply,
        model_optimizer,
        clip_ratio: float = 0.2,
        value_coef: jax.Array = jnp.float32(0.5),
        entropy_coef: jax.Array = jnp.float32(0.01),
        micro_batch_size: Optional[int] = None,
) -> tuple[PPOState, float, float, float, float, float, dict[str, Any]]:
    """
    Performs a PPO update over a batch of transitions.

    Args:
        ppo_state: Current PPOState
        observation: Batched GraphsTuple of observations, shape [B, ...]
        perms: Actions taken, shape [B]
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

    # Per-sample PPO loss.
    def sample_loss(m_params, graph, perm, old_lp, ret, adv, old_val):
        value_pred, dist = model_apply(m_params, graph)
        nb_validators = graph.globals[0]
        new_lp = logp_prefix_pl(dist.logits, perm, nb_validators)
        log_ratio = new_lp - old_lp
        ratio = jnp.exp(log_ratio)

        clipp_actor = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = -jnp.minimum(ratio * adv, clipp_actor * adv)
        entropy = dist.entropy()  # TODO

        value_pred_clipped = old_val + (value_pred - old_val).clip(
            -clip_ratio, clip_ratio)
        value_loss = jnp.square(value_pred - ret)
        value_loss_clipped = jnp.square(value_pred_clipped - ret)
        value_loss = jnp.maximum(value_loss, value_loss_clipped)

        total_loss = policy_loss + value_coef * value_loss - entropy * entropy_coef

        approx_kl = ratio - 1.0 - log_ratio
        is_clipped = (jnp.abs(ratio - 1.0) > clip_ratio).astype(jnp.float32)
        return total_loss, (policy_loss, value_loss, entropy, approx_kl, is_clipped)

    # Mean loss + stacked aux metrics [pl, vl, ent, kl, cf] over a set of samples.
    def batch_loss(model_params, graphs, perms_, lp_, ret_, adv_, val_):
        total_loss, aux = jax.vmap(
            sample_loss,
            in_axes=(None, 0, 0, 0, 0, 0, 0),
            out_axes=(0, (0, 0, 0, 0, 0)),
        )(model_params, graphs, perms_, lp_, ret_, adv_, val_)
        aux5 = jnp.stack([jnp.mean(a) for a in aux])
        return jnp.mean(total_loss), aux5

    grad_fn = jax.value_and_grad(batch_loss, has_aux=True)

    batch = perms.shape[0]
    full_inputs = (observation, perms, old_logps, returns, advantages, old_values)

    if micro_batch_size is None or micro_batch_size >= batch:
        # Single pass over the whole batch (default behaviour).
        (_, aux5), grads = grad_fn(ppo_state.params, *full_inputs)
    else:
        # Gradient accumulation: split the batch into micro-batches and sum their
        # gradients one at a time (lax.scan keeps only one micro-batch's
        # activations live), then average. Mathematically identical to the
        # full-batch gradient; peak activation VRAM scales with micro_batch_size
        # instead of batch. Exact here because there is no cross-sample coupling
        # (advantage normalization happens upstream; the model uses LayerNorm).
        assert batch % micro_batch_size == 0, (
            f"batch_size ({batch}) must be divisible by "
            f"micro_batch_size ({micro_batch_size})")
        n_micro = batch // micro_batch_size

        def _to_micro(x):
            return x.reshape((n_micro, micro_batch_size) + x.shape[1:])

        micro_inputs = jax.tree.map(_to_micro, full_inputs)

        def _accumulate(carry, micro):
            grad_acc, aux_acc = carry
            (_, aux5_m), g = grad_fn(ppo_state.params, *micro)
            return (jax.tree.map(jnp.add, grad_acc, g), aux_acc + aux5_m), None

        init = (jax.tree.map(jnp.zeros_like, ppo_state.params), jnp.zeros(5))
        (grad_sum, aux_sum), _ = jax.lax.scan(_accumulate, init, micro_inputs)
        grads = jax.tree.map(lambda x: x / n_micro, grad_sum)
        aux5 = aux_sum / n_micro

    mean_pl, mean_vl, mean_ent, mean_kl, mean_cf = aux5

    model_updates, new_model_opt_state = model_optimizer.update(
        grads, ppo_state.opt_state
    )

    new_model_params: dict = optax.apply_updates(ppo_state.params, model_updates)

    new_state = PPOState(new_model_params, new_model_opt_state, ppo_state.rng_key)

    return new_state, mean_pl, mean_vl, mean_ent, mean_kl, mean_cf, info_coef


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

    num_steps = ((num_steps + batch_size - 1) // batch_size) * batch_size

    # rollout fns expect graph inputs inside rollout. ppo_state is threaded as a
    # traced argument (see _vectorized_rollout) to avoid per-epoch recompilation.
    vm_rollout = _vectorized_rollout(env, model, num_steps)

    params_map = jax.vmap(
        lambda key_map: create_params_fn(key_map)  # Create new params for each env,
    )

    for epoch in range(num_epochs):
        new_ppo_key, rollout_key, params_key, permutation_key = jax.random.split(ppo_state.rng_key, 4)
        subkeys = jax.random.split(rollout_key, num_envs)
        subkeys_params = jax.random.split(params_key, num_envs)

        params_list = params_map(subkeys_params)

        observations, perms, logps, rews, dones, vals, last_values, infos = vm_rollout(
            subkeys, ppo_state, params_list)
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
        flat_perms = flatten(perms)
        flat_lp = flatten(logps)
        flat_r = flatten(returns)
        flat_adv = flatten(advantages_norm)
        flat_val = flatten(vals)

        # Flatten *each* leaf in the GraphsTuple of states.blockchain
        flat_graphs = jax.tree.map(flatten, observations)

        # Permute to get randomized minibatches
        idx = jax.random.permutation(permutation_key, flat_perms.shape[0])
        for start in range(0, idx.shape[0] - batch_size + 1, batch_size):  # Drop the last batch to avoid recomputation
            batch_idx = idx[start: start + batch_size]

            # Slice out a minibatch of graphs
            batch_graphs = jax.tree.map(lambda x: x[batch_idx], flat_graphs)

            # Now call update_ppo with the exact signature you defined:
            ppo_state, policy_loss, value_loss, entropy, approx_kl, clip_frac, info_coef = update_ppo(
                ppo_state,
                batch_graphs,  # env_states: a GraphsTuple PyTree
                flat_perms[batch_idx],  # actions
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
        print(
            f"Epoch {epoch}: PolicyLoss={policy_loss:.3f}, ValueLoss={value_loss:.3f}, Entropy={entropy:.3f}, approxKL={approx_kl:.3f}, clipFract={clip_frac:.3f}, info_coef={info_coef}")
        print(f"rewards : {rews.sum():.3f}, longueur {rews.shape}, dones {dones.sum():.3f}")

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


def index_graph(x, idx, num_steps: int):
    env_idx = idx // num_steps
    step_idx = idx % num_steps
    return x[env_idx, step_idx]


def train_epoch(ppo_state: PPOState, epoch: int, env: environment.Environment, model: nn.Module, num_steps: int,
                num_envs: int, create_params_fn: Callable[[jax.Array], TEnvParams],
                batch_size: int,
                model_opt: GradientTransformationExtraArgs, gamma: float, lambda_: float,
                clip_ratio: float, log_fn: LOG_TYPE = None,
                env_step: int = 0, value_coef: jax.Array = jnp.float32(0.5),
                entropy_coef: jax.Array = jnp.float32(0.01),
                norm_advantage: bool = False,
                micro_batch_size: Optional[int] = None) -> Tuple[PPOState, int]:
    """
    Perform one PPO training epoch using the provided hyperparameters.
    Returns the updated PPOState.
    """

    # Vectorized rollout.
    # NOTE: ``ppo_state`` is threaded as a *traced argument* (in_axes=None),
    # not captured by closure. Capturing it would bake the current network
    # weights in as compile-time constants, forcing a full recompilation (and,
    # under jax 0.10, a fresh Triton-GEMM autotuning pass) on *every* epoch.
    vm_rollout = _vectorized_rollout(env, model, num_steps)

    params_map = jax.vmap(create_params_fn)

    # Split RNG keys for rollouts and parameter sampling
    rollout_key, params_key, perm_key, new_ppo_key = jax.random.split(ppo_state.rng_key, 4)

    subkeys = jax.random.split(rollout_key, num_envs)
    subkeys_params = jax.random.split(params_key, num_envs)

    params_list = params_map(subkeys_params)
    observations, perms, logps, rews, dones, vals, last_values, infos_env = vm_rollout(
        subkeys, ppo_state, params_list)
    logger.info(f"Epoch {epoch}: Collected {num_steps * num_envs} steps.")

    if log_fn is not None:
        infos_env_refined = log_fn(infos_env, rews, dones)
        wandb.log({"env": infos_env_refined}, step=env_step)

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
    if norm_advantage:
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages_norm = advantages

    # Flatten data
    def flatten(x):
        return x.reshape(-1, *x.shape[2:])

    flat_perms = flatten(perms)
    flat_lp = flatten(logps)
    flat_r = flatten(returns)
    flat_adv = flatten(advantages_norm)
    flat_val = flatten(vals)

    # flat_graphs = jax.tree.map(flatten, observations)

    # Shuffle and minibatch updates
    perm = jax.random.permutation(perm_key, flat_perms.shape[0])
    for start in range(0, perm.shape[0] - batch_size + 1, batch_size):  # Drop the last batch to avoid recomputation
        logger.info(f"Epoch {epoch}: Processing batch {start // batch_size + 1} / {perm.shape[0] // batch_size + 1}")
        idx = perm[start: start + batch_size]
        # batch_graphs = jax.tree.map(lambda x: x[idx], flat_graphs)
        batch_graphs = jax.tree.map(lambda x: index_graph(x, idx, num_steps), observations)
        # print(f"ID elements : model.apply:{id(model.apply)}, model_opt:{id(model_opt)}, clip_ratio:{id(clip_ratio)}")
        ppo_state, policy_loss, value_loss, entropy, approx_kl, clip_fract, info_coef = update_ppo(
            ppo_state,
            batch_graphs,
            flat_perms[idx],
            flat_lp[idx],
            flat_r[idx],
            flat_adv[idx],
            flat_val[idx],
            model.apply,
            model_opt,
            clip_ratio,
            value_coef,
            entropy_coef,
            micro_batch_size,
        )
        info_train = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fract": clip_fract,
        }
        if log_fn is not None:
            wandb.log({"coef": info_coef, "train": info_train, "epoch": epoch}, step=env_step)
        env_step += batch_size
        logger.info(f"Epoch {epoch}/batch {start} - Policy Loss: {policy_loss}, Value Loss: {value_loss}")

    # Update RNG and log progress
    ppo_state = ppo_state.replace(rng_key=new_ppo_key)

    return ppo_state, env_step


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
             create_params_fn: Callable[[jax.Array], TEnvParams],
             num_episodes: int = 10,
             recorded_episodes: int = 10, batch_size: int = 10,
             log_fn: LOG_TYPE = None) -> dict[str, jax.Array]:
    steps_in_episode = int(env.default_params.max_steps_in_episode)

    vm_rollouts = _vectorized_rollout_eval(env, model, steps_in_episode)
    params_map = jax.vmap(create_params_fn)

    all_rewards = []
    all_dones = []
    all_infos = []

    num_batches = (num_episodes + batch_size - 1) // batch_size

    batch_key = jax.random.split(key, num_batches)

    for this_batch_key in batch_key:
        rollout_key, param_key = jax.random.split(this_batch_key)
        subkeys = jax.random.split(rollout_key, batch_size)
        subkeys_params = jax.random.split(param_key, batch_size)

        params_list = params_map(subkeys_params)
        rews, dones, infos = vm_rollouts(subkeys, ppo_state, params_list)

        all_rewards.append(rews)
        all_dones.append(dones)
        all_infos.append(infos)

    # Concaténer les résultats
    all_rewards = jnp.concatenate(all_rewards, axis=0)
    all_dones = jnp.concatenate(all_dones, axis=0)
    all_infos = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_infos)

    logger.info(f"Evaluated {num_episodes} episodes in {num_batches} batches of at most {batch_size} envs.")

    metrics = log_fn(all_infos, all_rewards, all_dones)
    metrics["avg_returns_episode"] = all_rewards.sum(axis=1).mean()

    sub_rewards = all_rewards.mean(axis=1)
    for i in range(min(recorded_episodes, num_episodes)):
        metrics[f"reward_{i}"] = sub_rewards[i]

    return metrics


# Modified create_ppo_state to use the manager
def create_ppo_state(resume_dir: Optional[Path], env: environment.Environment, seed: int, lr: float,
                     model: nn.Module) -> PPOState:
    """
    Initialize or restore a PPOState.  If `resume_dir` is provided, uses
    `checkpoint_manager` to restore the latest checkpoint; if `warm_start`
    is True, reinitializes optimizer states with loaded network weights.
    Otherwise, does a fresh init.
    """
    model_opt = make_optimizer(float(lr))
    key = jax.random.PRNGKey(seed)

    # --- restore path ---
    if resume_dir:
        # restore the entire PPOState PYTree
        state: PPOState = load_ppo_state(resume_dir, key)
        state = state.replace(opt_state=model_opt.init(state.params))
        return state

    # --- fresh initialization ---
    obs_key, ppo_key, state_key = jax.random.split(key, 3)
    first_obs, first_state = env.reset(obs_key, env.default_params)
    model_vars = model.init(ppo_key, first_obs)
    model_opt_state = model_opt.init(model_vars)
    print("Initialized new PPOState.")
    return PPOState(model_vars, model_opt_state, ppo_key)


def load_ppo_state(resume_dir: Path, key: jax.Array) -> PPOState:
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
