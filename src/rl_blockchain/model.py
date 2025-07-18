import distrax
import jax
import jraph as jr
from flax import linen as nn
from jax import numpy as jnp

from rl_blockchain.BlockEnv.BlockEnv import compute_legal_actions_obs


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


class PPO_NET_GAT(nn.Module):
    gat1_output_dim: int
    gat2_output_dim: int
    gat2_nodes_output_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, graph: jr.GraphsTuple) -> (jax.Array, distrax.Categorical):
        mask = compute_legal_actions_obs(graph)
        # Two GCN layers
        pol_gcn1 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(
                make_embed_fn(self.gat1_output_dim)(n)
            ),
            add_self_edges=True,
        )
        pol_gcn2 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(
                make_embed_fn(self.gat2_output_dim)(n)
            ),
            add_self_edges=True,
        )
        # Two GAT layers
        pol_gat1 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat1_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=None,
        )
        pol_gat2 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat2_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=lambda n: make_embed_fn(self.gat2_nodes_output_dim)(n),
        )

        # # Initialize globals to zero of shape [batch, action_dim]
        # # TODO: Check if this is correct
        # pol_graph = graph._replace(
        #     globals=jnp.zeros((graph.globals.shape[0], self.action_dim))
        # )

        @jr.concatenated_args
        def edge_fn(attrs):
            return jax.nn.relu(make_embed_fn(self.gat1_output_dim)(attrs))

        @jr.concatenated_args
        def node_fn(attrs):
            return jax.nn.relu(make_embed_fn(self.gat1_output_dim)(attrs))

        @jr.concatenated_args
        def global_fn(attrs):
            return jax.nn.relu(make_embed_fn(self.action_dim)(attrs))

        pol_gnn = jr.GraphNetwork(
            update_edge_fn=edge_fn,
            update_node_fn=node_fn,
            update_global_fn=global_fn,
        )

        pol_graph = pol_gcn1(graph)
        pol_graph = pol_gcn2(pol_graph)
        pol_graph = pol_gat1(pol_graph)
        pol_graph = pol_gat2(pol_graph)
        pol_graph = pol_graph._replace(edges=pol_graph.edges[:, None])
        pol_graph = pol_gnn(pol_graph)

        squeezed_globals = pol_graph.globals.squeeze()
        full_inf = jnp.full((self.action_dim,), -jnp.inf)
        masked_globals = jax.lax.select(mask, squeezed_globals, full_inf)

        pi = distrax.Categorical(logits=masked_globals)

        # Two GCN layers
        val_gcn1 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(
                make_embed_fn(self.gat1_output_dim)(n)
            ),
            add_self_edges=True,
        )
        val_gcn2 = jr.GraphConvolution(
            update_node_fn=lambda n: jax.nn.relu(
                make_embed_fn(self.gat2_output_dim)(n)
            ),
            add_self_edges=True,
        )
        # Two GAT layers
        val_gat1 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat1_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=None,
        )
        val_gat2 = jr.GAT(
            attention_query_fn=lambda n: make_embed_fn(self.gat2_output_dim)(n),
            attention_logit_fn=_attention_logit_fn,
            node_update_fn=lambda n: make_embed_fn(self.gat2_nodes_output_dim)(n),
        )
        # Initialize globals to zero of shape [batch,1]
        # graph = graph._replace(globals=jnp.zeros((graph.globals.shape[0], 1)))

        val_gnn = jr.GraphNetwork(
            update_edge_fn=edge_fn,
            update_node_fn=node_fn,
            update_global_fn=global_fn,
        )

        val_graph = val_gcn1(graph)
        val_graph = val_gcn2(val_graph)
        val_graph = val_gat1(val_graph)
        val_graph = val_gat2(val_graph)
        val_graph = val_graph._replace(edges=val_graph.edges[:, None])
        val_graph = val_gnn(val_graph)

        # Return shape [batch]
        val = val_graph.globals.squeeze()
        return val, pi
