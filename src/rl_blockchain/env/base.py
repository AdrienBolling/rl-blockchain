"""``BaseBlockchainEnv`` — the stateless, multi-objective env interface.

The contract, in one place:

* **Stateless.** No array state is stored on the instance. The whole mutable
  state is an :class:`EnvState` pytree passed in and returned by every method.
  This is what lets a single env object drive thousands of parallel rollouts via
  ``jax.vmap`` / ``jax.lax.scan``.
* **Explicit RNG.** Every stochastic method takes a ``PRNGKey``. Split it; never
  reuse a key. The caller owns key management (e.g. ``vmap`` over a batch of
  keys for parallel envs).
* **Multi-objective.** ``step`` returns a :class:`TimeStep` whose ``reward`` is
  a pytree of named scalar rewards (use :func:`stack_rewards` to vectorise).
* **Jit/GPU first.** All concrete implementations must be ``jax.jit``-able and
  written with whole-array ops (no Python-level per-node loops, no
  data-dependent shapes). The instance holds only static config, so
  ``jax.jit(env.step)`` works out of the box.

Subclasses implement the abstract methods; the small concrete helpers
(:meth:`reset_jit`, :meth:`step_jit`) provide ready-made jitted entry points.
"""

from __future__ import annotations

import abc
from functools import partial

import jax

from .types import (
    Action,
    EnvParams,
    EnvState,
    Observation,
    PRNGKey,
    TimeStep,
)


class BaseBlockchainEnv(abc.ABC):
    """Abstract base class for stateless blockchain voting environments."""

    # --- Static descriptors ------------------------------------------------ #

    @property
    @abc.abstractmethod
    def default_params(self) -> EnvParams:
        """Default static configuration for this environment."""

    @property
    @abc.abstractmethod
    def num_objectives(self) -> int:
        """Number of reward objectives (leaves in the reward pytree)."""

    @abc.abstractmethod
    def action_size(self, params: EnvParams) -> int:
        """Size of the (flat) action space, e.g. number of selectable nodes."""

    @abc.abstractmethod
    def observation_shape(self, params: EnvParams) -> tuple[int, ...]:
        """Shape of the observation returned by :meth:`observe`."""

    # --- Core stateless transition functions ------------------------------- #

    @abc.abstractmethod
    def reset(
        self, key: PRNGKey, params: EnvParams
    ) -> tuple[Observation, EnvState]:
        """Sample a fresh initial state.

        Args:
            key: PRNG key for any randomness in initialisation.
            params: Static configuration.

        Returns:
            ``(obs, state)`` — the first observation and the initial state.
        """

    @abc.abstractmethod
    def step(
        self,
        key: PRNGKey,
        state: EnvState,
        action: Action,
        params: EnvParams,
    ) -> TimeStep:
        """Advance the environment by one step.

        Implementations should split ``key`` for any stochastic dynamics, apply
        ``action`` to ``state.graph``, compute the multi-objective ``reward``
        pytree, advance ``state.time``, and set ``done`` (e.g. on reaching
        ``params.max_steps``). Must not mutate ``state`` in place.

        Returns:
            A :class:`TimeStep` (obs, next state, reward pytree, done, info).
        """

    @abc.abstractmethod
    def observe(self, state: EnvState, params: EnvParams) -> Observation:
        """Compute the agent-facing observation for ``state`` (pure function)."""

    # --- Ready-made jitted entry points ------------------------------------ #
    # Only `self` is static. `params` is passed as a normal pytree: its
    # shape-driving fields are `pytree_node=False`, so they live in the pytree
    # aux data and are already concrete Python values at trace time. These are
    # convenience wrappers; subclasses need not override them.

    @partial(jax.jit, static_argnums=(0,))
    def reset_jit(
        self, key: PRNGKey, params: EnvParams
    ) -> tuple[Observation, EnvState]:
        """Jitted :meth:`reset`."""
        return self.reset(key, params)

    @partial(jax.jit, static_argnums=(0,))
    def step_jit(
        self,
        key: PRNGKey,
        state: EnvState,
        action: Action,
        params: EnvParams,
    ) -> TimeStep:
        """Jitted :meth:`step`."""
        return self.step(key, state, action, params)
