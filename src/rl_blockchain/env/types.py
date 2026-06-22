"""Core datatypes for the stateless blockchain environment.

Everything here is a JAX pytree (``flax.struct.dataclass``) so it can be carried
through ``jax.jit``, ``jax.vmap`` (for parallel/batched rollouts) and
``jax.lax.scan`` (for whole-episode unrolls) without any special handling.

Conventions
-----------
* The environment is **stateless**: the mutable simulation state lives entirely
  in :class:`EnvState`, which is passed in and returned by every env function.
* Randomness is **explicit**: a ``jax.random.PRNGKey`` is threaded through
  ``reset`` / ``step`` by the caller. No global RNG, so runs are reproducible
  and trivially parallelisable (``vmap`` over a batch of keys).
* Rewards are a **pytree** (multi-objective / MORL): ``step`` returns a bundle
  of named scalar rewards rather than a single scalar.
* Static configuration (anything that drives array shapes) lives in
  :class:`EnvParams` as ``pytree_node=False`` fields, so it is treated as a
  compile-time constant by ``jit``.
"""

from __future__ import annotations

from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
from jax import Array

from rl_blockchain.graph import BlockchainGraph

# --- Type aliases (documentation only; all are JAX pytrees / arrays) -------- #
PRNGKey = Array
PyTree = Any
Observation = Any  # whatever the concrete env exposes (e.g. a feature matrix)
Action = Any  # e.g. a [n_nodes] selection mask or an index vector
Reward = Any  # a pytree of scalar rewards — see `stack_rewards`


@flax.struct.dataclass
class EnvParams:
    """Static environment configuration.

    Fields are ``pytree_node=False`` so they ride in the pytree *aux data* and
    are seen as concrete Python constants at trace time — safe to use for array
    shapes inside a jitted function.
    """

    num_nodes: int = flax.struct.field(pytree_node=False, default=100)
    max_steps: int = flax.struct.field(pytree_node=False, default=1000)


@flax.struct.dataclass
class EnvState:
    """The complete mutable state of one environment instance.

    Concrete environments may subclass this to add fields; the two below are
    the minimum the base interface relies on.
    """

    graph: BlockchainGraph
    time: Array  # int32 scalar — number of steps taken since reset


@flax.struct.dataclass
class TimeStep:
    """The result of one :meth:`BaseBlockchainEnv.step`.

    ``reward`` is intentionally a pytree (MORL): each leaf is a scalar reward
    (or a ``[batch]`` vector once the step is ``vmap``-ed).
    """

    obs: Observation
    state: EnvState
    reward: Reward
    done: Array  # bool scalar — episode termination flag
    info: dict[str, Any] = flax.struct.field(default_factory=dict)


def stack_rewards(reward: Reward) -> Array:
    """Flatten a reward pytree into a dense vector along the last axis.

    Leaves are taken in a deterministic (sorted-key) order. A single reward
    bundle becomes ``[n_objectives]``; a ``vmap``-ed batch becomes
    ``[batch, n_objectives]``. Handy for MORL scalarisation / logging.
    """
    leaves = jax.tree_util.tree_leaves(reward)
    if not leaves:
        return jnp.zeros((0,))
    return jnp.stack(jnp.broadcast_arrays(*leaves), axis=-1)
