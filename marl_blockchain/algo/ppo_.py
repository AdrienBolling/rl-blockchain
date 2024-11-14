from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct
import jraph as jr
import distrax
import flax.linen as nn


@struct.dataclass
class PPOState:
    policy_params: dict
    value_params: dict
    policy_opt_state: optax.OptState
    value_opt_state: optax.OptState
    rng_key: jax.random.KeyArray


def make_embed_fn(latent_size):
    def embed(inputs):
        return nn.Dense(latent_size)(inputs)

    return embed


def _attention_logit_fn(sender_attr: jnp.ndarray,
                        receiver_attr: jnp.ndarray,
                        edges: jnp.ndarray) -> jnp.ndarray:
    x = jnp.concatenate((sender_attr, receiver_attr, edges), axis=1)
    return nn.Dense(1)(x)


class PolicyNET_GAT(nn.Module):
    """
    Graph Attention Network (GAT) module using flax and jraph

    The policy network for the PPO algorithm, outputting a categorical distribution over actions
    """

    gat1_output_dim: int
    gat2_output_dim: int
    gat2_nodes_output_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        gcn1 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(make_embed_fn(self.gat1_output_dim)(n)),
            add_self_edges=True,
        )
        gcn2 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(make_embed_fn(self.gat2_output_dim)(n)),
            add_self_edges=True,
        )
        gat1 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat1_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=None
        )
        gat2 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat2_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=lambda n: make_embed_fn(self.gat2_nodes_output_dim)(n)
        )
        x._replace(globals=jnp.zeros((x.globals.shape[0], self.action_dim)))

        @jr.concatenated_args
        def concat_fn(x):
            return jax.nn.relu(make_embed_fn(self.gat1_output_dim)(x))

        @jr.concatenated_args
        def concat_fn_final(x):
            return jax.nn.relu(make_embed_fn(self.action_dim)(x))

        gnn = jr.GraphNetwork(
            update_edge_fn=concat_fn,
            update_node_fn=concat_fn,
            update_global_fn=concat_fn_final
        )
        x = gcn1(x)
        x = gcn2(x)
        x = gat1(x)
        x = gat2(x)
        x = gnn(x)

        pi = distrax.Categorical(logits=x.globals)
        return pi


class ValueNET_GAT(nn.Module):
    """
    Graph Attention Network (GAT) module using flax and jraph

    A copy of the policy network, but with a different output layer of size 1
    """

    gat1_output_dim: int
    gat2_output_dim: int
    gat2_nodes_output_dim: int

    @nn.compact
    def __call__(self, x):
        gcn1 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(make_embed_fn(self.gat1_output_dim)(n)),
            add_self_edges=True,
        )
        gcn2 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(make_embed_fn(self.gat2_output_dim)(n)),
            add_self_edges=True,
        )
        gat1 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat1_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=None
        )
        gat2 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat2_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=lambda n: make_embed_fn(self.gat2_nodes_output_dim)(n)
        )
        x._replace(globals=jnp.zeros((x.globals.shape[0], 1)))

        @jr.concatenated_args
        def concat_fn(x):
            return jax.nn.relu(make_embed_fn(self.gat1_output_dim)(x))

        @jr.concatenated_args
        def concat_fn_final(x):
            return jax.nn.relu(make_embed_fn(1)(x))

        gnn = jr.GraphNetwork(
            update_edge_fn=concat_fn,
            update_node_fn=concat_fn,
            update_global_fn=concat_fn_final
        )
        x = gcn1(x)
        x = gcn2(x)
        x = gat1(x)
        x = gat2(x)
        x = gnn(x)

        return x.globals


@partial(jax.jit, static_argnums=(1,5))
def rollout(rng_key, env, policy_fn, state, num_steps, reward_weights):
    """Collects a rollout using the policy over multiple steps with environment resets for early terminations."""

    def step_fn(carry, step):
        rng_key, state = carry
        rng_key, subkey = jax.random.split(rng_key)

        # Get the action distribution from the policy and sample an action
        action_dist = policy_fn(state)
        action = action_dist.sample(seed=subkey)

        # Step the environment
        next_state, reward, done, info = env.step(state=state, action=action, weights=reward_weights)

        # If done, reset the environment, else continue with the next state
        next_state = jax.lax.cond(done,
                                  lambda _: env.reset(),  # Reset if done
                                  lambda _: next_state,  # Continue with next state otherwise
                                  operand=None)

        return (rng_key, next_state), (state, action, reward, done)

    # Run the scan over the number of steps
    _, (states, actions, rewards, dones) = jax.lax.scan(step_fn, (rng_key, state), jnp.arange(num_steps))

    return states, actions, rewards, dones


@jax.jit
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lambda_=0.95):
    """Generalized Advantage Estimation (GAE) computation optimized with JAX."""

    # Append the last value for bootstrapping
    values = jnp.append(values, last_value)

    def scan_fn(carry, idx):
        advantage, next_value = carry
        reward = rewards[idx]
        value = values[idx]
        done = dones[idx]

        delta = reward + gamma * next_value * (1.0 - done) - value
        advantage = delta + gamma * lambda_ * advantage * (1.0 - done)

        return (advantage, value), advantage

    # Initialize advantage and next_value with 0.0 and the last_value, respectively
    init = (0.0, last_value)

    # Use lax.scan to reverse iterate and accumulate the advantages
    _, advantages = jax.lax.scan(scan_fn, init, jnp.arange(len(rewards) - 1, -1, -1))

    # Reverse the advantages to match the original order
    return jnp.flip(advantages)


@jax.jit
def update_ppo(state, states, actions, log_probs, returns, advantages, policy_fn, value_fn, policy_optimizer,
               value_optimizer, clip_ratio=0.2):
    """JIT-compilable PPO update step for separate policy and value networks."""

    def loss_fn(policy_params, value_params):
        """Computes the combined PPO loss (policy + value loss) with separate params."""
        # Calculate the policy distribution and value predictions
        new_policy_dist = policy_fn.apply(policy_params, states)
        new_log_probs = new_policy_dist.log_prob(actions)

        # Compute the policy loss (PPO clipping)
        ratio = jnp.exp(new_log_probs - log_probs)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = jnp.mean(-jnp.minimum(ratio * advantages, clipped_ratio * advantages))

        # Compute the value loss (mean squared error)
        values = value_fn.apply(value_params, states)
        value_loss = jnp.mean(jnp.square(returns - values))

        # Return combined loss (policy loss + value loss)
        total_loss = policy_loss + 0.5 * value_loss
        return total_loss, (policy_loss, value_loss)  # Return the individual losses as well

    # Combine policy and value params into a tuple for efficient loss and gradient computation
    combined_params = (state.policy_params, state.value_params)

    # Compute the loss and gradients with respect to both policy and value params
    (total_loss, (policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(*combined_params)

    # Split the gradients for policy and value networks
    policy_grads, value_grads = grads

    # Update the policy network parameters
    policy_updates, policy_opt_state = policy_optimizer.update(policy_grads, state.policy_opt_state)
    new_policy_params = optax.apply_updates(state.policy_params, policy_updates)

    # Update the value network parameters
    value_updates, value_opt_state = value_optimizer.update(value_grads, state.value_opt_state)
    new_value_params = optax.apply_updates(state.value_params, value_updates)

    # Create the new updated state
    new_state = state.replace(
        policy_params=new_policy_params,
        value_params=new_value_params,
        policy_opt_state=policy_opt_state,
        value_opt_state=value_opt_state
    )

    return new_state, policy_loss, value_loss


def train_ppo(env_fn, env_params, num_steps, num_envs_rollout, num_epochs, batch_size, lr, gamma, lambda_, clip_ratio):
    """Trains the PPO agent using parallel environments and optimized update steps."""

    # Initialize environment
    env = env_fn(**env_params)
    dummy_state = env.reset()
    dummy_action = env.sample_legal_action(dummy_state)

    # Initialize both networks
    rng_key = jax.random.PRNGKey(0)
    rng_key, subkey = jax.random.split(rng_key)

    policy_net = PolicyNET_GAT(gat1_output_dim=64, gat2_output_dim=64, gat2_nodes_output_dim=64,
                               action_dim=dummy_action.shape[0])
    value_net = ValueNET_GAT(gat1_output_dim=64, gat2_output_dim=64, gat2_nodes_output_dim=64)

    # Initialize the PPO state
    policy_params = policy_net.init(subkey, dummy_state)
    value_params = value_net.init(subkey, dummy_state)
    policy_opt = optax.adam(lr)
    value_opt = optax.adam(lr)
    policy_opt_state = policy_opt.init(policy_params)
    value_opt_state = value_opt.init(value_params)
    ppo_state = PPOState(policy_params, value_params, policy_opt_state, value_opt_state, rng_key)

    # Initialize an experience buffer
    buffer = []

    @jax.jit
    def parallel_rollout(env, key):
        state = env.reset()

        policy_fn = lambda s: policy_net.apply(ppo_state.policy_params, s)

        states, actions, rewards, dones = rollout(key, env, policy_fn, state, num_steps)
        return states, actions, rewards, dones

    # Run the training loop
    for epoch in range(num_epochs):

        # Collect rollouts from multiple environments
        rng_key, subkey = jax.random.split(rng_key)
        env_keys = jax.random.split(subkey, num_envs_rollout)
        envs = [env_fn(**env_params) for _ in range(num_envs_rollout)]

        states, actions, rewards, dones = jax.vmap(parallel_rollout)(envs, env_keys)

        # Add the collected experiences to the buffer
        for i in range(num_envs_rollout):
            buffer.extend(list(zip(states[i], actions[i], rewards[i], dones[i])))

        # Update the policy and value networks
        for _ in range(num_steps // batch_size):

            # Sample a batch of experiences
            batch = jax.random.choice(rng_key, buffer, shape=(batch_size,), replace=False)
            states, actions, returns, advantages = zip(*batch)

            # Update the PPO state
            ppo_state, policy_loss, value_loss = update_ppo(ppo_state, states, actions, returns, advantages, policy_net.apply, value_net.apply, opt)

        # Log the loss
        # print(f"Epoch {epoch + 1}: Policy Loss: {policy_loss} - Value Loss: {value_loss}")

def eval_ppo_and_log():
