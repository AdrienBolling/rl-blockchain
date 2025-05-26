from functools import partial
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from matplotlib.path import Path
import optax
from flax import struct
import jraph as jr
import distrax
import flax.linen as nn
import orbax.checkpoint as ocp
from tqdm import tqdm

from rl_blockchain.rl.wrappers.NormalizationWrapper import NormalizationWrapper


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
    x = jnp.concatenate((sender_attr, receiver_attr, edges), axis=1)
    return nn.Dense(1)(x)


class PolicyNET_GAT(nn.Module):
    gat1_output_dim: int
    gat2_output_dim: int
    gat2_nodes_output_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple):
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
        graph = gnn(graph)

        return distrax.Categorical(logits=graph.globals)


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
        graph = gnn(graph)

        # Return shape [batch]
        return graph.globals.squeeze()


@partial(jax.jit, static_argnames=('env', 'policy_apply', 'value_apply', 'num_steps'))
def rollout(rng_key, env, policy_apply, value_apply, state, num_steps, reward_weights):
    """Runs one episode for num_steps, returns trajectories and final state."""

    def step_fn(carry, _):
        key, st = carry
        key, subkey = jax.random.split(key)

        # Unwrap graph from Env State
        graph = st.blockchain
        dist = policy_apply(graph)
        act = dist.sample(seed=subkey)
        logp = dist.log_prob(act)
        val = value_apply(graph)

        next_st, rew, done, info = env.step(st, act, reward_weights)
        next_st = jax.lax.cond(
            done, lambda _: env.reset(), lambda _: next_st, operand=None
        )

        traj = (st, act, logp, rew, done, val)
        return (key, next_st), traj

    (key_final, st_end), trajs = jax.lax.scan(
        step_fn, (rng_key, state), None, length=num_steps
    )
    states, actions, logps, rewards, dones, values = trajs
    return states, actions, logps, rewards, dones, values, st_end, key_final


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
    state: PPOState,
    env_states: jr.GraphsTuple,
    actions: jnp.ndarray,
    old_logps: jnp.ndarray,
    returns: jnp.ndarray,
    advantages: jnp.ndarray,
    policy_apply,
    value_apply,
    policy_optimizer,
    value_optimizer,
    clip_ratio: float = 0.2
) -> tuple[PPOState, float, float]:
    """
    Performs a PPO update over a batch of transitions.

    Args:
        state: Current PPOState
        env_states: Batched GraphsTuple of observations, shape [B, ...]
        actions: Actions array, shape [B, action_dim]
        old_logps: Log probabilities under old policy, shape [B]
        returns: Discounted returns, shape [B]
        advantages: GAE advantages, shape [B]
        policy_apply: Policy network apply function
        value_apply: Value network apply function
        policy_optimizer: Optax optimizer for policy
        value_optimizer: Optax optimizer for value
        clip_ratio: PPO clipping parameter

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
            value_pred = value_apply(v_params, graph)
            value_loss = (ret - value_pred) ** 2
            return policy_loss + 0.5 * value_loss, (policy_loss, value_loss)

        # Vectorize over batch
        total_loss, (pl_batch, vl_batch) = jax.vmap(
            sample_loss,
            in_axes=(None, None, 0, 0, 0, 0, 0),
            out_axes=(0, (0, 0))
        )(
            policy_params,
            value_params,
            env_states,
            actions,
            old_logps,
            returns,
            advantages,
        )
        # total_loss is array of shape [B], pl_batch/ vl_batch each shape [B]
        mean_loss = jnp.mean(total_loss)
        mean_pl = jnp.mean(pl_batch)
        mean_vl = jnp.mean(vl_batch)
        # return mean total_loss as loss, and policy/value losses as aux
        return mean_loss, (mean_pl, mean_vl)

    # Compute gradients
    (loss_val, (mean_pl, mean_vl)), (policy_grads, value_grads) = jax.value_and_grad(
        loss_fn, has_aux=True, argnums=(0, 1)
    )(state.policy_params, state.value_params)

    # Apply policy optimizer step
    policy_updates, new_pol_opt_state = policy_optimizer.update(
        policy_grads, state.policy_opt_state
    )
    new_policy_params = optax.apply_updates(
        state.policy_params, policy_updates
    )

    # Apply value optimizer step
    value_updates, new_val_opt_state = value_optimizer.update(
        value_grads, state.value_opt_state
    )
    new_value_params = optax.apply_updates(
        state.value_params, value_updates
    )

    # Construct new state
    new_state = state.replace(
        policy_params=new_policy_params,
        value_params=new_value_params,
        policy_opt_state=new_pol_opt_state,
        value_opt_state=new_val_opt_state,
    )

    return new_state, mean_pl, mean_vl


def train_ppo(
    env_fn,
    env_params,
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
    env = env_fn(**env_params)
    init_state = env.reset()
    dummy_graph = init_state.blockchain
    key, subkey = jax.random.split(key)
    dummy_act = env.sample_legal_action(init_state, key=subkey)
    key, sub = jax.random.split(key)
    pol_net = PolicyNET_GAT(64, 64, 64, dummy_act.shape[0])
    val_net = ValueNET_GAT(64, 64, 64)
    # Initialize with GraphsTuple
    pol_vars = pol_net.init(sub, dummy_graph)
    val_vars = val_net.init(sub, dummy_graph)

    pol_opt = optax.adam(lr)
    val_opt = optax.adam(lr)
    pol_opt_state = pol_opt.init(pol_vars)
    val_opt_state = val_opt.init(val_vars)
    ppo_state = PPOState(pol_vars, val_vars, pol_opt_state, val_opt_state, key)

    # rollout fns expect graph inputs inside rollout
    def single_rollout(rng):
        return rollout(
            rng,
            env,
            lambda g: pol_net.apply(ppo_state.policy_params, g),
            lambda g: val_net.apply(ppo_state.value_params, g),
            env.reset(),
            num_steps,
            jnp.array([0.5, 0.5]),
        )

    vm_rollout = jax.vmap(single_rollout)

    for epoch in range(num_epochs):
        key, *subkeys = jax.random.split(ppo_state.rng_key, num_envs + 1)
        subkeys = jnp.stack(subkeys)
        states, acts, logps, rews, dones, vals, last_states, _ = vm_rollout(subkeys)
        # Extract graphs and last graphs
        # GAE over each env
        advantages = jax.vmap(
            lambda r, v, d, st: compute_gae(
                r,
                v,
                d,
                val_net.apply(ppo_state.value_params, st.blockchain),
                gamma,
                lambda_,
            )
        )(rews, vals, dones, last_states)
        returns = advantages + vals
    # Helper to flatten env × time dims
        def flatten(x):
            return x.reshape(-1, *x.shape[2:])

        # Flatten your action/logp/return/adv arrays
        flat_a   = flatten(acts)
        flat_lp  = flatten(logps)
        flat_r   = flatten(returns)
        flat_adv = flatten(advantages)

        # Flatten *each* leaf in the GraphsTuple of states.blockchain
        flat_graphs = jax.tree.map(flatten, states.blockchain)

        # Permute to get randomized minibatches
        idx = jax.random.permutation(key, flat_a.shape[0])
        for start in range(0, idx.shape[0], batch_size):
            batch_idx = idx[start : start + batch_size]

            # Slice out a minibatch of graphs
            batch_graphs = jax.tree.map(lambda x: x[batch_idx], flat_graphs)

            # Now call update_ppo with the exact signature you defined:
            ppo_state, policy_loss, value_loss = update_ppo(
                ppo_state,
                batch_graphs,             # env_states: a GraphsTuple PyTree
                flat_a[batch_idx],        # actions
                flat_lp[batch_idx],       # old_logps
                flat_r[batch_idx],        # returns
                flat_adv[batch_idx],      # advantages
                pol_net.apply,         # policy_apply
                val_net.apply,          # value_apply
                pol_opt,                  # policy_optimizer (optax.OptState)
                val_opt,                  # value_optimizer
                clip_ratio                # clip_ratio
            )
        ppo_state = ppo_state.replace(rng_key=key)
        print(f"Epoch {epoch}: PolicyLoss={policy_loss:.3f}, ValueLoss={value_loss:.3f}")

    return ppo_state


def eval_ppo_and_log(env_fn, env_params, ppo_state, reward_weights, num_episodes=10, key=None):
    env = env_fn(**env_params)
    returns = []
    for _ in range(num_episodes):
        st = env.reset()
        done = False
        tot = 0.0
        while not done:
            graph = st.blockchain
            key, subkey = jax.random.split(key)
            dist = PolicyNET_GAT(
                gat1_output_dim=64,
                gat2_output_dim=64,
                gat2_nodes_output_dim=64,
                action_dim=env.sample_legal_action(st, key=subkey).shape[0],
            ).apply(ppo_state.policy_params, graph)
            a = dist.mode()
            st, r, done, _ = env.step(st, a, reward_weights)
            tot += r
        returns.append(tot)
    avg = sum(returns) / len(returns)
    print(f"Eval over {num_episodes} eps: avg return={avg:.3f}")
    
def eval_ppo(
    ppo_state: PPOState,
    env_fn: Callable[..., Any],
    env_params: Dict[str, Any],
    reward_weights: jnp.ndarray,
    num_episodes: int = 10,
    key: Optional[jnp.ndarray] = None,
    gat_1_out: int = 64,
    gat_2_out: int = 64,
    gat_2_nodes_out: int = 64,
):
    env = env_fn(**env_params)
    metrics = {
        "returns": [],
        "lengths": [],
        "rewards": [],
        "infos": [],
    }
    key, subkey = jax.random.split(key)
    pol_net = PolicyNET_GAT(gat_1_out, gat_2_out, gat_2_nodes_out, env.sample_legal_action(env.reset(), key=subkey).shape[0])
    for _ in tqdm(range(num_episodes)):
        st = env.reset()
        done = False
        total_reward = 0.0
        rewards = []
        lengths = 0
        infos_ = []

        while not done:
            graph = st.blockchain
            key, subkey = jax.random.split(key)
            dist = pol_net.apply(ppo_state.policy_params, graph)
            action = dist.mode()
            st, reward, done, infos = env.step(st, action, reward_weights)
            total_reward += reward
            rewards.append(reward)
            lengths += 1
            infos_.append(infos)
            
        metrics["returns"].append(total_reward)
        metrics["lengths"].append(lengths)
        metrics["rewards"].append(rewards)
        metrics["infos"].append(infos_)
        
    return metrics

def train_epoch(
    ppo_state: PPOState,
    epoch: int,
    env_fn: Callable[..., Any],
    env_params: Dict[str, Any],
    num_steps: int,
    num_envs: int,
    batch_size: int,
    lr: float,
    gamma: float,
    lambda_: float,
    clip_ratio: float,
    reward_weights: jnp.ndarray,
    gat1_out: int,
    gat2_out: int,
    gat2_nodes_out: int,
    normalize_rewards: bool = True,
) -> Tuple[PPOState, float, float]:
    """
    Perform one PPO training epoch using the provided hyperparameters.
    Returns the updated PPOState.
    """
    env = env_fn(**env_params)
    
    if normalize_rewards:
        # If using normalization, ensure the environment is wrapped accordingly
        env = NormalizationWrapper(env)

    # Split RNG for rollouts
    key, *subkeys = jax.random.split(ppo_state.rng_key, num_envs + 1)
    subkeys = jnp.stack(subkeys)

    # Determine action dim
    init_state = env.reset()
    key, subkey = jax.random.split(key)
    dummy_act = env.sample_legal_action(init_state, key=subkey)

    pol_net = PolicyNET_GAT(gat1_out, gat2_out, gat2_nodes_out, dummy_act.shape[0])
    val_net = ValueNET_GAT(gat1_out, gat2_out, gat2_nodes_out)

    # Vectorized rollout
    def single_rollout(rng):
        return rollout(
            rng,
            env,
            lambda g: pol_net.apply(ppo_state.policy_params, g),
            lambda g: val_net.apply(ppo_state.value_params, g),
            env.reset(),
            num_steps,
            reward_weights,
        )

    vm_rollout = jax.vmap(single_rollout)
    states, acts, logps, rews, dones, vals, last_states, _ = vm_rollout(subkeys)

    # Compute advantages and returns
    advantages = jax.vmap(
        lambda r, v, d, st: compute_gae(
            r, v, d,
            val_net.apply(ppo_state.value_params, st.blockchain),
            gamma, lambda_
        )
    )(rews, vals, dones, last_states)
    returns = advantages + vals

    # Flatten data
    def flatten(x):
        return x.reshape(-1, *x.shape[2:])

    flat_a   = flatten(acts)
    flat_lp  = flatten(logps)
    flat_r   = flatten(returns)
    flat_adv = flatten(advantages)
    flat_graphs = jax.tree.map(lambda x: flatten(x), states.blockchain)

    # Shuffle and minibatch updates
    perm = jax.random.permutation(key, flat_a.shape[0])
    for start in range(0, perm.shape[0], batch_size):
        idx = perm[start: start + batch_size]
        batch_graphs = jax.tree.map(lambda x: x[idx], flat_graphs)
        ppo_state, policy_loss, value_loss = update_ppo(
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
    ppo_state = ppo_state.replace(rng_key=key)
    return ppo_state, policy_loss, value_loss

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
    env_fn: Callable[..., Any],
    env_params: Dict[str, Any],
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
    env = env_fn(**env_params)
    init_state = env.reset()
    dummy_graph = init_state.blockchain
    key, subkey = jax.random.split(key)
    dummy_act = env.sample_legal_action(init_state, key=subkey)

    pol_net = PolicyNET_GAT(gat1_out, gat2_out, gat2_nodes_out, dummy_act.shape[0])
    val_net = ValueNET_GAT(gat1_out, gat2_out, gat2_nodes_out)

    pol_vars = pol_net.init(subkey, dummy_graph)
    key, subkey = jax.random.split(key)
    val_vars = val_net.init(subkey, dummy_graph)

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
