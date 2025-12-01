import distrax
import jax
import jraph as jr
from flax import linen as nn
from jax import numpy as jnp
from jraph._src.utils import segment_sum

from rl_blockchain.BlockEnv.BlockEnv import compute_legal_actions_obs
from rl_blockchain.model import make_update_fn, attention_reduce_fn, PPOBackbone, PPOCriticHead


class PPOSeparate_Test(nn.Module):
    action_dim: int
    backbone_gat_dim: int
    actor_gcn_dim: int
    critic_gnn_dim: int

    @nn.compact
    def __call__(self, graph):
        mask = compute_legal_actions_obs(graph)

        shared = PPOBackbone(self.backbone_gat_dim)(graph)

        gate_pi = jr.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_update_fn(
                [1], last_activation=False),
            update_global_fn=None)

        # gate_val = jr.GraphNetwork(
        #     update_edge_fn=None,
        #     update_node_fn=None,
        #     update_global_fn=make_update_fn(
        #         [1], last_activation=False),
        #     aggregate_edges_for_nodes_fn=segment_sum,
        #     aggregate_nodes_for_globals_fn=segment_sum,
        #     aggregate_edges_for_globals_fn=segment_sum)
        gate_val = PPOCriticHead(self.critic_gnn_dim)


        g_pi = gate_pi(shared)
        g_val = gate_val(shared)

        logits = jnp.concatenate([jnp.zeros(1), g_pi.nodes.squeeze()]).squeeze()

        full_inf = jnp.full((self.action_dim,), -jnp.inf)
        masked_logits = jax.lax.select(mask, logits, full_inf)

        pi = distrax.Categorical(logits=masked_logits)

        v = g_val

        return v, pi
