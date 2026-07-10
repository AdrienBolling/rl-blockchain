import distrax
import jraph as jr
from flax import linen as nn
from jax import numpy as jnp
from rl_blockchain.BlockEnv.BlockchainGraph import with_topology
from rl_blockchain.fast_agg import fast_segment_sum, OptGraphNetGAT


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


class PPOBackbone(nn.Module):
    backbone_gat_dim: int

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple):
        # Accept observations stored without their topology (see `strip_topology`);
        # a no-op when senders/receivers are already there.
        graph = with_topology(graph)
        graph = graph._replace(nodes=graph.nodes[:, None], edges=graph.edges[:, None], globals=graph.globals[:, None])

        projector = jr.GraphMapFeatures(
            embed_edge_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
            embed_node_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
            embed_global_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
        )

        back_gat_1 = OptGraphNetGAT(
            update_edge_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_node_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_global_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            attention_logit_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim, 1],
                last_activation=False),
            attention_reduce_fn=attention_reduce_fn)

        back_gat_2 = OptGraphNetGAT(
            update_edge_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_node_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_global_fn=make_update_fn(
                [self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            attention_logit_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim * 2, self.backbone_gat_dim, 1],
                last_activation=False),
            attention_reduce_fn=attention_reduce_fn)

        gp = projector(graph)
        g1 = back_gat_1(gp)
        g1 = g1._replace(
            nodes=g1.nodes + gp.nodes,
            edges=g1.edges + gp.edges,
            globals=g1.globals + gp.globals)

        g2 = back_gat_2(g1)
        g2 = g2._replace(
            nodes=g2.nodes + g1.nodes,
            edges=g2.edges + g1.edges,
            globals=g2.globals + g1.globals)

        return g2


class PPOActorHead(nn.Module):
    action_dim: int
    actor_gcn_dim: int

    @nn.compact
    def __call__(self, shared_graph):
        graph_pi = jr.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_update_fn(
                [self.actor_gcn_dim, self.actor_gcn_dim * 2, self.actor_gcn_dim, 1], last_activation=False),
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=fast_segment_sum)(shared_graph)

        return distrax.Categorical(logits=graph_pi.nodes.squeeze())


class PPOCriticHead(nn.Module):
    critic_gnn_dim: int

    @nn.compact
    def __call__(self, shared_graph):
        crit = jr.GraphNetwork(
            update_edge_fn=make_update_fn([self.critic_gnn_dim, self.critic_gnn_dim]),
            update_node_fn=make_update_fn([self.critic_gnn_dim, self.critic_gnn_dim]),
            update_global_fn=make_update_fn([self.critic_gnn_dim * 2, self.critic_gnn_dim, 1], last_activation=False),
            aggregate_edges_for_nodes_fn=fast_segment_sum,
            aggregate_nodes_for_globals_fn=fast_segment_sum,
        )(shared_graph)
        return crit.globals.squeeze()


class PPOSeparate(nn.Module):
    action_dim: int
    backbone_gat_dim: int
    actor_gcn_dim: int
    critic_gnn_dim: int

    @nn.compact
    def __call__(self, graph):
        shared = PPOBackbone(self.backbone_gat_dim)(graph)
        pi = PPOActorHead(self.action_dim, self.actor_gcn_dim)(shared)
        v = PPOCriticHead(self.critic_gnn_dim)(shared)

        return v, pi
