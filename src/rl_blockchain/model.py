from typing import Callable

import distrax
import jax
import jraph as jr
import jraph._src.models as jr_m
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
        latent_size: int,
        hidden_layers: list[int] | None = None,
        *,
        activation=nn.gelu,
        pre_norm: bool = False,
        last_activation: bool = False,
        dtype=jnp.float32,
        kernel_init=nn.initializers.lecun_normal(),
):
    if hidden_layers is None:
        hidden_layers = [latent_size]

    layers: list = []
    if pre_norm:
        layers.append(nn.LayerNorm())
    for width in hidden_layers:
        layers.append(nn.Dense(width, dtype=dtype, kernel_init=kernel_init))
        layers.append(activation)

    # Projection finale vers latent_size
    layers.append(nn.Dense(latent_size, dtype=dtype, kernel_init=kernel_init))
    if last_activation:
        layers.append(activation)

    return nn.Sequential(layers)


def make_embed_fn(latent_size, hidden_layers: list[int] | None = None, pre_norm: bool = False):
    sequential_layer = make_mlp(latent_size, hidden_layers=hidden_layers, pre_norm=pre_norm)

    @jr.concatenated_args
    def embed(inputs) -> jnp.ndarray:
        return sequential_layer(inputs)

    return embed


def make_attention_logit_fn(latent_size, hidden_layers: list[int] | None = None):
    """
    Reuse the same function for attention logits.
    """
    if hidden_layers is None:
        hidden_layers = [latent_size]
    # hidden_layers.append(1)  # Ensure the last layer is of size 1 for logits
    hidden_layers = (*hidden_layers, 1)
    return make_embed_fn(latent_size, hidden_layers=hidden_layers)


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


def make_norm_layer():
    return jr.GraphMapFeatures(
        embed_edge_fn=nn.LayerNorm(),
        embed_node_fn=nn.LayerNorm(),
        embed_global_fn=nn.LayerNorm(),
    )


class Transf_GAT(nn.Module):
    layer_output_dim: int
    layer_hidden_dim: list[int]
    layer_hidden_edge_dim: list[int]

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple):
        norm_layer_0 = make_norm_layer()

        h0 = norm_layer_0(graph)

        net_gat = jr.GraphNetGAT(
            update_edge_fn=make_embed_fn(self.layer_output_dim, hidden_layers=self.layer_hidden_edge_dim),
            update_node_fn=make_embed_fn(self.layer_output_dim, hidden_layers=self.layer_hidden_dim),
            update_global_fn=make_embed_fn(self.layer_output_dim, hidden_layers=self.layer_hidden_dim),
            attention_logit_fn=make_attention_logit_fn(self.layer_output_dim, hidden_layers=self.layer_hidden_edge_dim),
            attention_reduce_fn=attention_reduce_fn
        )
        d_h0 = net_gat(h0)

        h1 = h0._replace(
            nodes=d_h0.nodes + graph.nodes,
            edges=d_h0.edges + graph.edges,
            globals=d_h0.globals + graph.globals,
        )

        norm_layer_1 = make_norm_layer()
        norm_h1 = norm_layer_1(h1)

        mlp_layer = jr.GraphMapFeatures(
            embed_edge_fn=make_embed_fn(self.layer_output_dim, hidden_layers=self.layer_hidden_edge_dim),
            embed_node_fn=make_embed_fn(self.layer_output_dim, hidden_layers=self.layer_hidden_dim),
            embed_global_fn=make_embed_fn(self.layer_output_dim, hidden_layers=self.layer_hidden_dim),
        )
        mlp_h1 = mlp_layer(norm_h1)
        h2 = h1._replace(
            nodes=mlp_h1.nodes + h1.nodes,
            edges=mlp_h1.edges + h1.edges,
            globals=mlp_h1.globals + h1.globals,
        )

        norm_layer_2 = make_norm_layer()
        norm_h2 = norm_layer_2(h2)
        return norm_h2


def DeepSetsGlobalised(
        update_node_fn: Callable[[jr_m.NodeFeatures, jr_m.Globals], jr_m.NodeFeatures] | None,
        update_global_fn: Callable[[jr_m.NodeFeatures, jr_m.Globals], jr_m.Globals],
        aggregate_nodes_for_globals_fn:
        jr_m.AggregateNodesToGlobalsFn = segment_sum):
    return jr.GraphNetwork(
        update_edge_fn=None,
        update_node_fn=None if update_node_fn is None else lambda n, s, r, g: update_node_fn(n, g),
        update_global_fn=lambda n, e, g: update_global_fn(n, g),
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn)


class PPO_SKIP(nn.Module):
    action_dim: int
    embedded_dim: int = 10
    embedding_hidden_dim: int = 32
    trans_gat_dim: int = 64
    trans_mlp_dim_val: int = 64
    trans_mlp_dim_pol: int = 128

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple):
        mask = compute_legal_actions_obs(graph)
        graph = graph._replace(edges=graph.edges[:, None], globals=graph.globals[:, None])

        embedding_encoder = jr.GraphMapFeatures(
            embed_edge_fn=make_embed_fn(self.embedded_dim,
                                        hidden_layers=[self.embedding_hidden_dim, self.embedding_hidden_dim],
                                        pre_norm=True),
            embed_node_fn=make_embed_fn(self.embedded_dim,
                                        hidden_layers=[self.embedding_hidden_dim, self.embedding_hidden_dim],
                                        pre_norm=True),
            embed_global_fn=make_embed_fn(self.embedded_dim, hidden_layers=[self.embedding_hidden_dim // 2,
                                                                            self.embedding_hidden_dim // 2],
                                          pre_norm=True),
        )

        gat_1 = Transf_GAT(self.embedded_dim, [self.trans_gat_dim, self.trans_gat_dim],
                           [self.trans_gat_dim // 2, self.trans_gat_dim // 2])
        gat_2 = Transf_GAT(self.embedded_dim, [self.trans_gat_dim, self.trans_gat_dim],
                           [self.trans_gat_dim // 2, self.trans_gat_dim // 2])

        gat_3 = Transf_GAT(self.embedded_dim, [self.trans_gat_dim, self.trans_gat_dim],
                           [self.trans_gat_dim // 2, self.trans_gat_dim // 2])

        deep_set_val = DeepSetsGlobalised(
            update_node_fn=None,
            update_global_fn=make_embed_fn(1, hidden_layers=[self.trans_mlp_dim_val, self.trans_mlp_dim_val])
        )

        deep_set_pol = DeepSetsGlobalised(
            update_node_fn=make_embed_fn(1, hidden_layers=[self.trans_mlp_dim_pol, self.trans_mlp_dim_pol,
                                                           self.trans_mlp_dim_pol]),
            update_global_fn=make_embed_fn(1, hidden_layers=[self.trans_mlp_dim_pol, self.trans_mlp_dim_pol,
                                                             self.trans_mlp_dim_pol])
        )

        embedded_graph = embedding_encoder(graph)
        graph_1 = gat_1(embedded_graph)
        graph_2 = gat_2(graph_1)
        graph_3 = gat_3(graph_2)

        g_val = deep_set_val(graph_3)
        val = g_val.globals.squeeze()

        g_pol = deep_set_pol(graph_3)
        pol_unmasked = jnp.concat([g_pol.globals, g_pol.nodes], axis=0).squeeze()
        full_inf = jnp.full((self.action_dim,), -jnp.inf)
        masked_globals = jax.lax.select(mask, pol_unmasked, full_inf)
        pi = distrax.Categorical(logits=masked_globals)

        return val, pi
