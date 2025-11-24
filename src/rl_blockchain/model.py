import distrax
import jax
import jraph as jr
from flax import linen as nn
from jax import numpy as jnp
from jraph._src.utils import segment_sum

from rl_blockchain.BlockEnv.BlockEnv import compute_legal_actions_obs


def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)


class CategoricalSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    model_name: str = "separate-mlp"
    flatten_2d: bool = False  # Catch case
    flatten_3d: bool = False  # Rooms/minatar case

    @nn.compact
    def __call__(self, x):
        # Flatten a single 2D image
        if self.flatten_2d and len(x.shape) == 2:
            x = x.reshape(-1)
        # Flatten a batch of 2d images into a batch of flat vectors
        if self.flatten_2d and len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Flatten a single 3D image
        if self.flatten_3d and len(x.shape) == 3:
            x = x.reshape(-1)
        # Flatten a batch of 3d images into a batch of flat vectors
        if self.flatten_3d and len(x.shape) > 3:
            x = x.reshape(x.shape[0], -1)
        x_v = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_critic + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i + 1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        v = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v",
            bias_init=default_mlp_init(),
        )(x_v)

        x_a = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        logits = nn.Dense(
            self.num_output_units,
            bias_init=default_mlp_init(),
        )(x_a)
        pi = distrax.Categorical(logits=logits)
        return v.squeeze(), pi


def make_mlp(
        layers_dim: int | list[int],
        activation=nn.gelu,
        pre_norm: bool = False,
        last_activation: bool = True,
        dtype=jnp.float32,
        kernel_init=nn.initializers.xavier_uniform(),
):
    if isinstance(layers_dim, int):
        layers_dim = [layers_dim]

    layers: list = []
    if pre_norm:
        layers.append(nn.LayerNorm())
    for width in layers_dim[:-1]:
        layers.append(nn.Dense(width, dtype=dtype, kernel_init=kernel_init))
        layers.append(activation)
        layers.append(nn.LayerNorm())

    # Projection finale vers latent_size
    layers.append(nn.Dense(layers_dim[-1], dtype=dtype, kernel_init=kernel_init))
    if last_activation:
        layers.append(activation)

    return nn.Sequential(layers)


def make_update_fn(layers_dim: int | list[int], pre_norm: bool = True, last_activation: bool = True):
    sequential_layer = make_mlp(layers_dim, pre_norm=pre_norm, last_activation=last_activation)

    @jr.concatenated_args
    def embed(inputs) -> jnp.ndarray:
        return sequential_layer(inputs)

    return embed


def attention_reduce_fn(edges: jnp.ndarray, attention: jnp.ndarray):
    return edges * attention


class PPO_NET_BACK(nn.Module):
    action_dim: int
    backbone_gat_dim: int
    actor_gcn_dim: int
    critic_gnn_dim: int

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple):
        mask = compute_legal_actions_obs(graph)
        graph = graph._replace(edges=graph.edges[:, None], globals=graph.globals[:, None])

        projector = jr.GraphMapFeatures(
            embed_edge_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
            embed_node_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
            embed_global_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
        )

        back_gat_1 = jr.GraphNetGAT(
            update_edge_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_node_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_global_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            attention_logit_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim, 1],
                last_activation=False),
            attention_reduce_fn=attention_reduce_fn,
            aggregate_edges_for_nodes_fn=segment_sum,
            aggregate_nodes_for_globals_fn=segment_sum,
            aggregate_edges_for_globals_fn=segment_sum,
        )

        back_gat_2 = jr.GraphNetGAT(
            update_edge_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_node_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_global_fn=make_update_fn(
                [self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            attention_logit_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim, 1],
                last_activation=False),
            attention_reduce_fn=attention_reduce_fn,
            aggregate_edges_for_nodes_fn=segment_sum,
            aggregate_nodes_for_globals_fn=segment_sum,
            aggregate_edges_for_globals_fn=segment_sum,
        )

        actor_gcn_1 = jr.GraphConvolution(
            update_node_fn=make_update_fn([self.actor_gcn_dim, self.actor_gcn_dim * 4, self.actor_gcn_dim]),
            add_self_edges=True,
            symmetric_normalization=True
        )

        actor_gcn_2 = jr.GraphConvolution(
            update_node_fn=make_update_fn([self.actor_gcn_dim, self.actor_gcn_dim * 4, self.actor_gcn_dim, 1],
                                          last_activation=False),
        )

        critic_graphnet = jr.GraphNetwork(
            update_edge_fn=make_update_fn([self.critic_gnn_dim, self.critic_gnn_dim]),
            update_node_fn=make_update_fn([self.critic_gnn_dim, self.critic_gnn_dim]),
            update_global_fn=make_update_fn([self.critic_gnn_dim * 2, self.critic_gnn_dim, 1], last_activation=False),
            aggregate_edges_for_nodes_fn=segment_sum,
            aggregate_nodes_for_globals_fn=segment_sum,
        )

        graph_projected = projector(graph)

        shared_graph_1 = back_gat_1(graph_projected)
        shared_graph_1 = shared_graph_1._replace(
            nodes=shared_graph_1.nodes + graph_projected.nodes,
            edges=shared_graph_1.edges + graph_projected.edges,
            globals=shared_graph_1.globals + graph_projected.globals,
        )
        shared_graph_2 = back_gat_2(shared_graph_1)
        shared_graph_2 = shared_graph_2._replace(
            nodes=shared_graph_2.nodes + shared_graph_1.nodes,
            edges=shared_graph_2.edges + shared_graph_1.edges,
            globals=shared_graph_2.globals + shared_graph_1.globals,
        )

        act_graph_1 = actor_gcn_1(shared_graph_2)
        act_graph = actor_gcn_2(act_graph_1)

        crit_graph = critic_graphnet(shared_graph_2)
        val = crit_graph.globals.squeeze()

        full_pol = jnp.concatenate([jnp.zeros(1), act_graph.nodes.squeeze()]).squeeze()

        full_inf = jnp.full((self.action_dim,), -jnp.inf)
        masked_globals = jax.lax.select(mask, full_pol, full_inf)

        pi = distrax.Categorical(logits=masked_globals)

        return val, pi
