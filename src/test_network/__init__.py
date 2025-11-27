import distrax
import jax
import jax.numpy as jnp
import jraph as jr
import optax
from flax.training.train_state import TrainState

from rl_blockchain.BlockEnv import EnvParams, EnvState, BlockchainEnv
from rl_blockchain.scripts.env_factory import GenericEnvFactory

key = jax.random.PRNGKey(0)

key_param, key_model, key_obs, key = jax.random.split(key, 4)
size_batch = 8

lr = 1e-3
config = {"gat_arch": [16, 8, 4], "voting_nodes": 4, "ref_map_file": "grid/grid_7_nodes.json",
          "reward_weights": [0.1, 0.9]}


def l2_loss_graph(pred: tuple[jax.Array, distrax.Categorical], target: tuple[jax.Array, jax.Array]) -> jax.Array:
    loss_values, loss_probs = optax.l2_loss(pred[0], target[0]), optax.l2_loss(pred[1].probs, target[1])
    return loss_values + loss_probs.sum()


def peerwise_1(params: EnvParams, init_state: EnvState) -> tuple[EnvState, tuple[jax.Array, jax.Array]]:
    nb_nodes = params.network_graph.n_node[0]

    node_validator_init = jnp.zeros((nb_nodes,))
    action_space_init = jnp.zeros((nb_nodes + 1,))

    node_validator = node_validator_init.at[jnp.array([0, 1])].set(1)

    state_1 = EnvState(
        ring_history=init_state.ring_history,
        chosen_nodes=node_validator,
        inner_step=init_state.inner_step,
        global_step=init_state.global_step,
        time=init_state.time
    )

    prob_1 = action_space_init.at[2].set(1)  # Choosing node 2
    value_1 = jnp.array(40, dtype=jnp.float32)
    return state_1, (value_1, prob_1)


def peerwise_2(params: EnvParams, init_state: EnvState) -> tuple[EnvState, tuple[jax.Array, jax.Array]]:
    nb_nodes = params.network_graph.n_node[0]

    node_validator_init = jnp.zeros((nb_nodes,))
    action_space_init = jnp.zeros((nb_nodes + 1,))

    node_validator = node_validator_init.at[jnp.array([3, 4])].set(1)

    state_1 = EnvState(
        ring_history=init_state.ring_history,
        chosen_nodes=node_validator,
        inner_step=init_state.inner_step,
        global_step=init_state.global_step,
        time=init_state.time
    )

    prob_1 = action_space_init.at[5].set(1)  # Choosing node 2
    value_1 = jnp.array(20, dtype=jnp.float32)
    return state_1, (value_1, prob_1)


def create_peerwise_pred(env: BlockchainEnv, params: EnvParams, init_state: EnvState) -> list[
    tuple[jr.GraphsTuple, tuple[jax.Array, jax.Array]]]:
    list_peer = []
    list_fn_gen = [peerwise_1, peerwise_2]

    for fn_gen in list_fn_gen:
        state_i, output_i = fn_gen(params, init_state)
        obs_i = env.get_obs(state_i, params)
        list_peer.append((obs_i, output_i))

    return list_peer


def build_balanced_batch(list_peerwise, batch_size, key):
    n = len(list_peerwise)

    # indices répétés pour atteindre batch_size
    reps = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=n)

    obs_batch = [list_peerwise[int(i)][0] for i in reps]
    batched_obs = jax.tree.map(lambda *xs: jnp.stack(xs), *obs_batch)
    vals_batch = jnp.array([list_peerwise[int(i)][1][0] for i in reps], dtype=jnp.float32)
    probs_batch = jnp.array([list_peerwise[int(i)][1][1] for i in reps], dtype=jnp.float32)

    return batched_obs, (vals_batch, probs_batch)


if __name__ == '__main__':
    model, env, _, _, _ = GenericEnvFactory.create("blockenv_close_map", key_param, config)

    first_obs, first_state = env.reset(key_obs, env.default_params)
    tx = optax.adam(lr)
    model_vars = model.init(key_model, first_obs)

    state = TrainState.create(
        apply_fn=model.apply,
        params=model_vars,
        tx=tx)

    list_peerwise = create_peerwise_pred(env, env.default_params, first_state)

    obs_batch, (vals_batch, probs_batch) = build_balanced_batch(list_peerwise, batch_size=size_batch, key=key)

    # Simple training loop to overfit on the synthetic peerwise dataset
    num_epochs = 5000


    @jax.jit
    def loss_batch(params, obs_batch, vals_batch, probs_batch):

        def loss_fn(obs: jr.GraphsTuple, val, prob):
            prediction = model.apply(params, obs)
            return l2_loss_graph(prediction, (val, prob))

        batched_loss = jax.vmap(loss_fn)(obs_batch, vals_batch, probs_batch)
        return batched_loss.mean()


    grad_loss_fn = jax.value_and_grad(loss_batch)


    def train_step(state: TrainState):

        loss, grads = grad_loss_fn(state.params, obs_batch, vals_batch, probs_batch)
        state = state.apply_gradients(grads=grads)
        return state, loss


    for epoch in range(num_epochs):
        total_loss = 0.0

        # loop on the artificial supervised pairs
        for (obs_i, target_i) in list_peerwise:
            state, loss = train_step(state)
            total_loss += loss

        if epoch % 200 == 0:
            print("Epoch:", epoch, "Loss:", float(total_loss))

    # ---- Verification of overfitting ----
    print("\nFinal evaluation on training examples:")
    for idx, (obs_i, target_i) in enumerate(list_peerwise):
        pred_value, pred_dist = state.apply_fn(state.params, obs_i)
        print("Example", idx)
        print("  Target value:", float(target_i[0]), "Pred value:", float(pred_value))
        print("  Target action:", target_i[1])
        print("  Predicted probs:", pred_dist.probs)
