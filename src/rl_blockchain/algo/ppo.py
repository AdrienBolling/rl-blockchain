import os
from functools import partial
from typing import Optional, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph as jr
import optax
import orbax.checkpoint as ocp
from flax import struct
from gymnax.environments import environment
from matplotlib.path import Path
from tqdm import tqdm

from rl_blockchain.BlockEnv import EnvParams
from rl_blockchain.BlockEnv.BlockEnv import compute_legal_actions_obs, BlockchainEnv


@struct.dataclass
class PPOState:
    policy_params: dict
    value_params: dict
    policy_opt_state: optax.OptState
    value_opt_state: optax.OptState
    rng_key: jnp.ndarray


def make_embed_fn(latent_size):
    def embed(inputs):
        return nn.Dense(latent_size)(inputs)

    return embed


def _attention_logit_fn(
        sender_attr: jnp.ndarray, receiver_attr: jnp.ndarray, edges: jnp.ndarray
) -> jnp.ndarray:
    edges = edges[:, None]
    x = jnp.concatenate((sender_attr, receiver_attr, edges), axis=1)
    return nn.Dense(1)(x)


class PolicyNET_GAT(nn.Module):
    gat1_output_dim: int
    gat2_output_dim: int
    gat2_nodes_output_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple) -> distrax.Categorical:
        mask = compute_legal_actions_obs(graph)
        # Two GCN layers
        gcn1 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(
                make_embed_fn(self.gat1_output_dim)(n)
            ),
            add_self_edges=True,
        )
        gcn2 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(
                make_embed_fn(self.gat2_output_dim)(n)
            ),
            add_self_edges=True,
        )
        # Two GAT layers
        gat1 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat1_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=None,
        )
        gat2 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat2_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=lambda n: make_embed_fn(self.gat2_nodes_output_dim)(n),
        )
        # Initialize globals to zero of shape [batch, action_dim]
        graph = graph._replace(
            globals=jnp.zeros((graph.globals.shape[0], self.action_dim))
        )

        @jr.concatenated_args
        def edge_fn(attrs):
            return jax.nn.relu(make_embed_fn(self.gat1_output_dim)(attrs))

        @jr.concatenated_args
        def node_fn(attrs):
            return jax.nn.relu(make_embed_fn(self.gat1_output_dim)(attrs))

        @jr.concatenated_args
        def global_fn(attrs):
            return jax.nn.relu(make_embed_fn(self.action_dim)(attrs))

        gnn = jr.GraphNetwork(
            update_edge_fn=edge_fn,
            update_node_fn=node_fn,
            update_global_fn=global_fn,
        )

        graph = gcn1(graph)
        graph = gcn2(graph)
        graph = gat1(graph)
        graph = gat2(graph)
        graph = graph._replace(edges=graph.edges[:, None])
        graph = gnn(graph)

        squeezed_globals = graph.globals.squeeze()
        full_inf = jnp.full((self.action_dim,), -jnp.inf)
        masked_globals = jax.lax.select(mask, squeezed_globals, full_inf)

        return distrax.Categorical(logits=masked_globals)


class ValueNET_GAT(nn.Module):
    gat1_output_dim: int
    gat2_output_dim: int
    gat2_nodes_output_dim: int

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple) -> jnp.ndarray:
        # Two GCN layers
        gcn1 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(
                make_embed_fn(self.gat1_output_dim)(n)
            ),
            add_self_edges=True,
        )
        gcn2 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(
                make_embed_fn(self.gat2_output_dim)(n)
            ),
            add_self_edges=True,
        )
        # Two GAT layers
        gat1 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat1_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=None,
        )
        gat2 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat2_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=lambda n: make_embed_fn(self.gat2_nodes_output_dim)(n),
        )
        # Initialize globals to zero of shape [batch,1]
        graph = graph._replace(globals=jnp.zeros((graph.globals.shape[0], 1)))

        @jr.concatenated_args
        def edge_fn(attrs):
            return jax.nn.relu(make_embed_fn(self.gat1_output_dim)(attrs))

        @jr.concatenated_args
        def node_fn(attrs):
            return jax.nn.relu(make_embed_fn(self.gat1_output_dim)(attrs))

        @jr.concatenated_args
        def global_fn(attrs):
            return jax.nn.relu(make_embed_fn(1)(attrs))

        gnn = jr.GraphNetwork(
            update_edge_fn=edge_fn,
            update_node_fn=node_fn,
            update_global_fn=global_fn,
        )

        graph = gcn1(graph)
        graph = gcn2(graph)
        graph = gat1(graph)
        graph = gat2(graph)
        graph = graph._replace(edges=graph.edges[:, None])
        graph = gnn(graph)

        # Return shape [batch]
        return graph.globals.squeeze()


@partial(jax.jit, static_argnames=('pol_model', 'val_model', 'env', 'steps_in_episode'))
def rollout(key_input, env: environment.Environment,
            pol_model: nn.Module, val_model: nn.Module, ppo_state: PPOState,
            env_params: EnvParams, steps_in_episode: int):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    key_reset, key_episode = jax.random.split(key_input)
    first_obs, first_state = env.reset(key_reset, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, key = state_input
        next_key, key_step, key_net = jax.random.split(key, 3)
        action_distribution = pol_model.apply(ppo_state.policy_params, obs)
        action = action_distribution.sample(seed=key_net)
        logp = action_distribution.log_prob(action)
        value = val_model.apply(ppo_state.value_params, obs)

        next_obs, next_state, reward, done, _ = env.step(
            key_step, state, action, env_params
        )

        carry = [next_obs, next_state, next_key]
        traj = (next_obs, action, logp, reward, done, value)
        return carry, traj

    # Scan over episode step loop
    (obs_end, _, _), trajs = jax.lax.scan(
        policy_step,
        [first_obs, first_state, key_episode],
        None,
        steps_in_episode
    )

    last_value = val_model.apply(ppo_state.value_params, obs_end)
    # Return masked sum of rewards accumulated by agent in episode
    observations, actions, logps, rewards, dones, values = trajs
    return observations, actions, logps, rewards, dones, values, last_value


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
        fn, (0.0, last_value), jnp.arange(values.shape[0] - 2, -1, -1)
    )
    return advs[::-1]


@partial(jax.jit, static_argnames=('policy_apply', 'value_apply', 'policy_optimizer', 'value_optimizer', 'clip_ratio'))
def update_ppo(
        ppo_state: PPOState,
        observation: jr.GraphsTuple,
        actions: jnp.ndarray,
        old_logps: jnp.ndarray,
        returns: jnp.ndarray,
        advantages: jnp.ndarray,
        policy_apply,
        value_apply,
        policy_optimizer,
        value_optimizer,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
) -> tuple[PPOState, float, float, float]:
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

    # Loss function with aux outputs
    def loss_fn(policy_params, value_params):
        # compute per-sample losses
        def sample_loss(p_params, v_params, graph, a, old_lp, ret, adv):
            dist = policy_apply(p_params, graph)
            new_lp = dist.log_prob(a)
            ratio = jnp.exp(new_lp - old_lp)

            clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -jnp.minimum(ratio * adv, clipped * adv)
            entropy = dist.entropy()

            value_pred = value_apply(v_params, graph)
            diff = ret - value_pred
            value_loss = jnp.inner(diff, diff)
            return policy_loss + value_coef * value_loss - entropy * entropy_coef, (policy_loss, value_loss, entropy)

        # Vectorize over batch
        total_loss, (pl_batch, vl_batch, ent_batch) = jax.vmap(
            sample_loss,
            in_axes=(None, None, 0, 0, 0, 0, 0),
            out_axes=(0, (0, 0, 0))
        )(
            policy_params,
            value_params,
            observation,
            actions,
            old_logps,
            returns,
            advantages,
        )
        # total_loss is array of shape [B], pl_batch/ vl_batch each shape [B]
        mean_loss = jnp.mean(total_loss)
        mean_pl_batch = jnp.mean(pl_batch)
        mean_vl_batch = jnp.mean(vl_batch)
        mean_ent_batch = jnp.mean(ent_batch)
        # return mean total_loss as loss, and policy/value losses as aux
        return mean_loss, (mean_pl_batch, mean_vl_batch, mean_ent_batch)

    # Compute gradients
    (loss_val, (mean_pl, mean_vl, mean_ent)), (policy_grads, value_grads) = jax.value_and_grad(
        loss_fn, has_aux=True, argnums=(0, 1)
    )(ppo_state.policy_params, ppo_state.value_params)

    # Apply policy optimizer step
    policy_updates, new_pol_opt_state = policy_optimizer.update(
        policy_grads, ppo_state.policy_opt_state
    )
    new_policy_params = optax.apply_updates(
        ppo_state.policy_params, policy_updates
    )

    # Apply value optimizer step
    value_updates, new_val_opt_state = value_optimizer.update(
        value_grads, ppo_state.value_opt_state
    )
    new_value_params = optax.apply_updates(
        ppo_state.value_params, value_updates
    )

    # Construct new state
    new_state = ppo_state.replace(
        policy_params=new_policy_params,
        value_params=new_value_params,
        policy_opt_state=new_pol_opt_state,
        value_opt_state=new_val_opt_state,
    )

    return new_state, mean_pl, mean_vl, mean_ent


def train_ppo(
        env: BlockchainEnv,
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

    default_params = env.default_params
    action_range = env.action_space(default_params).n
    sample_obs = env.observation_space(default_params).sample(obs_key)

    pol_net = PolicyNET_GAT(64, 64, 64, action_range)
    val_net = ValueNET_GAT(64, 64, 64)
    # Initialize with GraphsTuple
    pol_vars = pol_net.init(pol_key, sample_obs)
    val_vars = val_net.init(val_key, sample_obs)

    pol_opt = optax.adam(lr)
    val_opt = optax.adam(lr)
    pol_opt_state = pol_opt.init(pol_vars)
    val_opt_state = val_opt.init(val_vars)
    ppo_state = PPOState(pol_vars, val_vars, pol_opt_state, val_opt_state, ppo_key)

    # rollout fns expect graph inputs inside rollout
    @jax.jit
    def single_rollout(rng: jax.Array, new_param: EnvParams):
        return rollout(
            rng,
            env,
            pol_net,
            val_net,
            ppo_state,
            new_param,
            num_steps,
        )

    vm_rollout = jax.vmap(single_rollout)

    params_map = jax.vmap(
        lambda key_map: EnvParams.create_random(env.nb_nodes, key_map, env.default_params.nb_validators,
                                                env.default_params.rewards_weights),
    )

    for epoch in range(num_epochs):
        new_ppo_key, rollout_key, params_key = jax.random.split(ppo_state.rng_key, 3)
        subkeys = jax.random.split(rollout_key, num_envs)
        subkeys_params = jax.random.split(params_key, num_envs)

        params_list = params_map(subkeys_params)

        observations, acts, logps, rews, dones, vals, last_values = vm_rollout(subkeys, params_list)

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

        # Helper to flatten env × time dims
        def flatten(x):
            return x.reshape(-1, *x.shape[2:])

        # Flatten your action/logp/return/adv arrays
        flat_a = flatten(acts)
        flat_lp = flatten(logps)
        flat_r = flatten(returns)
        flat_adv = flatten(advantages)

        # Flatten *each* leaf in the GraphsTuple of states.blockchain
        flat_graphs = jax.tree.map(flatten, observations)

        # Permute to get randomized minibatches
        idx = jax.random.permutation(key, flat_a.shape[0])
        for start in range(0, idx.shape[0], batch_size):
            batch_idx = idx[start: start + batch_size]

            # Slice out a minibatch of graphs
            batch_graphs = jax.tree.map(lambda x: x[batch_idx], flat_graphs)

            # Now call update_ppo with the exact signature you defined:
            ppo_state, policy_loss, value_loss, entropy = update_ppo(
                ppo_state,
                batch_graphs,  # env_states: a GraphsTuple PyTree
                flat_a[batch_idx],  # actions
                flat_lp[batch_idx],  # old_logps
                flat_r[batch_idx],  # returns
                flat_adv[batch_idx],  # advantages
                pol_net.apply,  # policy_apply
                val_net.apply,  # value_apply
                pol_opt,  # policy_optimizer (optax.OptState)
                val_opt,  # value_optimizer
                clip_ratio  # clip_ratio
            )
        ppo_state = ppo_state.replace(rng_key=new_ppo_key)
        print(f"Epoch {epoch}: PolicyLoss={policy_loss:.3f}, ValueLoss={value_loss:.3f}, Entropy={entropy:.3f}")
        print(f"rewards : {rews.sum():.3f}, longueur {rews.shape}, dones {dones.sum():.3f}")

    return ppo_state


def eval_ppo_and_log(env: BlockchainEnv, ppo_state: PPOState, num_episodes: int = 10, key=None):
    returns = []
    env_params = env.default_params
    pol_net = PolicyNET_GAT(64, 64, 64,
                            env.action_space(env_params).n)
    for _ in range(num_episodes):
        key, subkey_mat, subkey_st = jax.random.split(key, 3)
        temp_params = EnvParams.create_random(env.nb_nodes, subkey_mat, env_params.nb_validators,
                                              env_params.rewards_weights)
        obs, st = env.reset(subkey_st, temp_params)
        done = False
        tot = 0.0
        while not done:
            key, subkey = jax.random.split(key)
            dist = pol_net.apply(ppo_state.policy_params, obs)
            a = dist.mode()
            obs, st, r, done, _ = env.step(subkey, st, a, temp_params)
            tot += r
        returns.append(tot)
    avg = sum(returns) / len(returns)
    print(f"Eval over {num_episodes} eps: avg return={avg:.3f}")


def eval_ppo(ppo_state: PPOState, env: BlockchainEnv, num_episodes: int = 10, key: Optional[jnp.ndarray] = None,
             gat_1_out: int = 64, gat_2_out: int = 64, gat_2_nodes_out: int = 64):
    metrics = {
        "returns": [],
        "lengths": [],
        "rewards": [],
        "infos": [],
    }
    key, subkey = jax.random.split(key)
    env_params = env.default_params
    pol_net = PolicyNET_GAT(gat_1_out, gat_2_out, gat_2_nodes_out,
                            env.action_space(env_params).n)
    for _ in tqdm(range(num_episodes)):
        key, subkey_mat, subkey_st = jax.random.split(key, 3)
        temp_params = EnvParams.create_random(env.nb_nodes, subkey_mat, env_params.nb_validators,
                                              env_params.rewards_weights)
        obs, st = env.reset(subkey_st, temp_params)
        done = False
        total_reward = 0.0
        rewards = []
        lengths = 0
        infos_ = []

        while not done:
            key, subkey = jax.random.split(key)
            dist = pol_net.apply(ppo_state.policy_params, obs)
            a = dist.mode()
            obs, st, r, done, infos = env.step(subkey, st, a, temp_params)
            total_reward += r
            rewards.append(r)
            lengths += 1
            infos_.append(infos)

        metrics["returns"].append(total_reward)
        metrics["lengths"].append(lengths)
        metrics["rewards"].append(rewards)
        metrics["infos"].append(infos_)

    return metrics


def train_epoch(ppo_state: PPOState, epoch: int, env: BlockchainEnv, num_steps: int, num_envs: int, batch_size: int,
                lr: float, gamma: float, lambda_: float, clip_ratio: float, gat1_out: int, gat2_out: int,
                gat2_nodes_out: int, normalize_rewards: bool = False) -> Tuple[PPOState, float, float, float]:
    """
    Perform one PPO training epoch using the provided hyperparameters.
    Returns the updated PPOState.
    """

    # TODO normalization of rewards
    # if normalize_rewards:
    #    # If using normalization, ensure the environment is wrapped accordingly
    #    env = NormalizationWrapper(env)

    action_dim = env.action_space(env.default_params).n
    pol_net = PolicyNET_GAT(gat1_out, gat2_out, gat2_nodes_out, action_dim)
    val_net = ValueNET_GAT(gat1_out, gat2_out, gat2_nodes_out)

    # Vectorized rollout
    @jax.jit
    def single_rollout(rng: jax.Array, new_param: EnvParams):
        return rollout(
            rng,
            env,
            pol_net,
            val_net,
            ppo_state,
            new_param,
            num_steps,
        )

    vm_rollout = jax.vmap(single_rollout)

    params_map = jax.vmap(
        lambda key_map: EnvParams.create_random(env.nb_nodes, key_map, env.default_params.nb_validators,
                                                env.default_params.rewards_weights),
    )

    # Split RNG keys for rollouts and parameter sampling
    rollout_key, params_key, perm_key, new_ppo_key = jax.random.split(ppo_state.rng_key, 4)

    subkeys = jax.random.split(rollout_key, num_envs)
    subkeys_params = jax.random.split(params_key, num_envs)

    params_list = params_map(subkeys_params)
    observations, acts, logps, rews, dones, vals, last_values = vm_rollout(subkeys, params_list)

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

    # Flatten data
    def flatten(x):
        return x.reshape(-1, *x.shape[2:])

    flat_a = flatten(acts)
    flat_lp = flatten(logps)
    flat_r = flatten(returns)
    flat_adv = flatten(advantages)
    flat_graphs = jax.tree.map(flatten, observations)

    # Shuffle and minibatch updates
    perm = jax.random.permutation(perm_key, flat_a.shape[0])
    for start in range(0, perm.shape[0], batch_size):
        idx = perm[start: start + batch_size]
        batch_graphs = jax.tree.map(lambda x: x[idx], flat_graphs)
        ppo_state, policy_loss, value_loss, entropy = update_ppo(
            ppo_state,
            batch_graphs,
            flat_a[idx],
            flat_lp[idx],
            flat_r[idx],
            flat_adv[idx],
            pol_net.apply,
            val_net.apply,
            optax.adam(lr),
            optax.adam(lr),
            clip_ratio
        )

    # Update RNG and log progress
    ppo_state = ppo_state.replace(rng_key=new_ppo_key)
    return ppo_state, policy_loss, value_loss, entropy


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


# Modified create_ppo_state to use the manager
def create_ppo_state(
        checkpoint_manager: ocp.CheckpointManager,
        resume_dir: Optional[Path],
        warm_start: bool,
        env: environment.Environment,
        seed: int,
        lr: float,
        gat1_out: int,
        gat2_out: int,
        gat2_nodes_out: int
) -> PPOState:
    """
    Initialize or restore a PPOState.  If `resume_dir` is provided, uses
    `checkpoint_manager` to restore the latest checkpoint; if `warm_start`
    is True, reinitializes optimizer states with loaded network weights.
    Otherwise, does a fresh init.
    """
    # --- restore path ---
    if resume_dir:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found in {resume_dir}")
        # restore the entire PPOState PYTree
        state: PPOState = checkpoint_manager.restore(step)
        print(f"Loaded checkpoint from step {step}")
        if warm_start:
            pol_opt = optax.adam(lr)
            val_opt = optax.adam(lr)
            state = state.replace(
                policy_opt_state=pol_opt.init(state.policy_params),
                value_opt_state=val_opt.init(state.value_params)
            )
            print("Optimizer states reinitialized for warm start.")
        return state

    # --- fresh initialization ---
    key = jax.random.PRNGKey(seed)
    graph_key, act_key, pol_key, val_key = jax.random.split(key, 4)
    dummy_graph = env.observation_space(env.default_params).sample(graph_key)

    pol_net = PolicyNET_GAT(gat1_out, gat2_out, gat2_nodes_out, env.action_space(env.default_params).n)
    val_net = ValueNET_GAT(gat1_out, gat2_out, gat2_nodes_out)

    pol_vars = pol_net.init(pol_key, dummy_graph)
    val_vars = val_net.init(val_key, dummy_graph)

    pol_opt = optax.adam(lr)
    val_opt = optax.adam(lr)
    pol_opt_state = pol_opt.init(pol_vars)
    val_opt_state = val_opt.init(val_vars)

    print("Initialized new PPOState.")
    return PPOState(
        policy_params=pol_vars,
        value_params=val_vars,
        policy_opt_state=pol_opt_state,
        value_opt_state=val_opt_state,
        rng_key=key
    )
