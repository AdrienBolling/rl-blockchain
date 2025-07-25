import distrax
import jax
import jraph as jr
from flax import linen as nn
from jax import numpy as jnp

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


def make_embed_fn(latent_size):
    sequential_layer = nn.Sequential([
        nn.LayerNorm(),
        nn.Dense(latent_size),
        nn.relu,
        nn.Dense(latent_size),
    ])

    @jr.concatenated_args
    def embed(inputs):
        return sequential_layer(inputs)

    return embed


def make_attention_logit_fn(latent_size):
    sequential_layer = nn.Sequential([
        nn.LayerNorm(),
        nn.Dense(latent_size),
        nn.relu,
        nn.Dense(latent_size)
    ])

    @jr.concatenated_args
    def attention_logit_fn(inputs) -> jnp.ndarray:
        return sequential_layer(inputs)

    return attention_logit_fn


def attention_reduce_fn(edges: jnp.ndarray, attention: jnp.ndarray):
    return edges * attention


class PPO_NET_GAT(nn.Module):
    backbone_gat1_output_dim: int
    backbone_gat2_output_dim: int
    gat2_nodes_output_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple):
        mask = compute_legal_actions_obs(graph)
        graph = graph._replace(edges=graph.edges[:, None], globals=graph.globals[:, None])

        net_gat_1 = jr.GraphNetGAT(update_edge_fn=make_embed_fn(self.backbone_gat1_output_dim),
                                   update_node_fn=make_embed_fn(self.backbone_gat1_output_dim),
                                   update_global_fn=make_embed_fn(self.backbone_gat1_output_dim),
                                   attention_logit_fn=make_attention_logit_fn(self.backbone_gat1_output_dim),
                                   attention_reduce_fn=attention_reduce_fn
                                   )

        net_gat_2 = jr.GraphNetGAT(update_edge_fn=make_embed_fn(self.backbone_gat2_output_dim),
                                   update_node_fn=make_embed_fn(self.backbone_gat2_output_dim),
                                   update_global_fn=make_embed_fn(self.backbone_gat2_output_dim),
                                   attention_logit_fn=make_attention_logit_fn(self.backbone_gat2_output_dim),
                                   attention_reduce_fn=attention_reduce_fn
                                   )

        net_gnn_val = jr.GraphNetwork(
            update_edge_fn=make_embed_fn(10),
            update_node_fn=make_embed_fn(self.gat2_nodes_output_dim),
            update_global_fn=make_embed_fn(self.gat2_nodes_output_dim),
        )

        net_last_gnn_val = jr.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_embed_fn(self.gat2_nodes_output_dim),
            update_global_fn=make_embed_fn(self.gat2_nodes_output_dim),
        )

        net_last_MLP_val = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.gat2_nodes_output_dim, bias_init=default_mlp_init()),
            nn.relu,
            nn.Dense(1, bias_init=default_mlp_init()),
        ])

        net_gnn_pol = jr.GraphNetwork(
            update_edge_fn=make_embed_fn(10),
            update_node_fn=make_embed_fn(self.gat2_nodes_output_dim),
            update_global_fn=None,
        )

        net_last_gnn_pol = jr.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_embed_fn(self.gat2_nodes_output_dim),
            update_global_fn=make_embed_fn(self.gat2_nodes_output_dim),
        )

        net_last_MLP_pol = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.gat2_nodes_output_dim, bias_init=default_mlp_init()),
            nn.relu,
            nn.Dense(self.action_dim, bias_init=default_mlp_init()),
        ])

        graph_1 = net_gat_1(graph)
        graph_2 = net_gat_2(graph_1)

        val_graph_3 = net_gnn_val(graph_2)
        val_graph_4 = net_last_gnn_val(val_graph_3)
        val_concat = jnp.concat([val_graph_4.globals, val_graph_4.nodes], axis=0)
        val_concat = val_concat.reshape(-1)
        last_val = net_last_MLP_val(val_concat)
        val = last_val.squeeze()

        pol_graph_3 = net_gnn_pol(graph_2)
        pol_graph_4 = net_last_gnn_pol(pol_graph_3)

        pol_concat = jnp.concat([pol_graph_4.globals, pol_graph_4.nodes], axis=0)
        pol_concat = pol_concat.reshape(-1)
        # jax.debug.print("pol_concat shape: {shape}", shape=pol_concat.shape)
        last_pol = net_last_MLP_pol(pol_concat)
        # jax.debug.print("last_pol shape: {shape}", shape=last_pol.shape)

        squeezed_globals = last_pol.squeeze()
        # jax.debug.print("squeezed_globals shape: {shape}", shape=squeezed_globals.shape)
        full_inf = jnp.full((self.action_dim,), -jnp.inf)
        masked_globals = jax.lax.select(mask, squeezed_globals, full_inf)

        pi = distrax.Categorical(logits=masked_globals)

        return val, pi
