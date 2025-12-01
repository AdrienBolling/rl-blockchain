import distrax
import jax
import jraph as jr
from flax import linen as nn
from jax import numpy as jnp
from jraph._src.utils import segment_sum

from rl_blockchain.BlockEnv.BlockEnv import compute_legal_actions_obs
from rl_blockchain.model import make_update_fn, attention_reduce_fn


class PPOSeparate_Test(nn.Module):
    action_dim: int
    backbone_gat_dim: int
    actor_gcn_dim: int
    critic_gnn_dim: int

    @nn.compact
    def __call__(self, graph):
        mask = compute_legal_actions_obs(graph)

        graph = graph._replace(edges=graph.edges[:, None], globals=graph.globals[:, None])

        projector = jr.GraphMapFeatures(
            embed_edge_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
            embed_node_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
            embed_global_fn=make_update_fn(self.backbone_gat_dim, pre_norm=False, last_activation=False),
        )

        gate_1 = jr.GraphNetGAT(
            update_edge_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_node_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim]),
            update_global_fn=make_update_fn(
                [self.backbone_gat_dim]),
            attention_logit_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim, 1],
                last_activation=False),
            attention_reduce_fn=attention_reduce_fn,
            aggregate_edges_for_nodes_fn=segment_sum,
            aggregate_nodes_for_globals_fn=segment_sum,
            aggregate_edges_for_globals_fn=segment_sum)

        gate_2 = jr.GraphNetGAT(
            update_edge_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim]),
            update_node_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim]),
            update_global_fn=make_update_fn(
                [self.backbone_gat_dim]),
            attention_logit_fn=make_update_fn(
                [self.backbone_gat_dim, self.backbone_gat_dim * 2, self.backbone_gat_dim, 1],
                last_activation=False),
            attention_reduce_fn=attention_reduce_fn,
            aggregate_edges_for_nodes_fn=segment_sum,
            aggregate_nodes_for_globals_fn=segment_sum,
            aggregate_edges_for_globals_fn=segment_sum)

        gate_3 = jr.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_update_fn(
                [1], last_activation=False),
            update_global_fn=make_update_fn(
                [1], last_activation=False),
            aggregate_edges_for_nodes_fn=segment_sum,
            aggregate_nodes_for_globals_fn=segment_sum,
            aggregate_edges_for_globals_fn=segment_sum)

        g_p = projector(graph)
        g_1 = gate_1(g_p)
        g_1 = g_1._replace(
            nodes=g_1.nodes + g_p.nodes,
            edges=g_1.edges + g_p.edges,
            globals=g_1.globals + g_p.globals
        )
        g_2 = gate_2(g_1)
        g_2 = g_2._replace(
            nodes=g_2.nodes + g_1.nodes,
            edges=g_2.edges + g_1.edges,
            globals=g_2.globals + g_1.globals
        )

        g_3 = gate_3(g_2)

        logits = jnp.concatenate([jnp.zeros(1), g_3.nodes.squeeze()]).squeeze()

        full_inf = jnp.full((self.action_dim,), -jnp.inf)
        masked_logits = jax.lax.select(mask, logits, full_inf)

        pi = distrax.Categorical(logits=masked_logits)

        v = g_3.globals.squeeze()

        return v, pi
