import jax
import jax.numpy as jnp
from flax.struct import dataclass

from rl_blockchain.BlockEnv import EnvState, BlockchainEnv, EnvParams


# ======================================================================
# Reward Normalization Wrapper
# ======================================================================

@dataclass
class NormRewState(EnvState):
    """
    Extension of the State class to include normalization parameters for rewards.
    """
    rew_count: int = 0
    rew_mean: float = 0.0
    rew_M2: float = 0.0

    @classmethod
    def create_from_state(cls, state: EnvState, rew_count: int = 0, rew_mean: float | jax.Array = 0.0,
                          rew_M2: float | jax.Array = 0.0) -> 'NormRewState':
        return cls(
            ring_history=state.ring_history,
            rew_count=rew_count,
            rew_mean=rew_mean,
            rew_M2=rew_M2,
            time=state.time,
        )


class NormalizationWrapper(BlockchainEnv):
    """
    A wrapper for normalizing rewards in a reinforcement learning environment.
    This wrapper maintains the mean and variance of the rewards to normalize them.
    """

    def __init__(self, block_env: BlockchainEnv, eps: float = 1e-8, clip_range: float = 10.0):
        """
        Initialize the NormalizationWrapper.

        Args:
            env: The environment to wrap.
            eps: A small value to avoid division by zero.
            clip_range: The range to clip the normalized rewards.
        """
        super().__init__(block_env._first_params, block_env._static_params)
        self.eps = eps
        self.clip_range = clip_range

    def reset_env(self, key: jax.Array, params: EnvParams):
        """
        Reset the environment and return the initial state.

        Returns:
            The initial state of the environment.
        """
        obs, state = super().reset_env(key, params)

        # Create a new state with normalization parameters
        norm_state = NormRewState.create_from_state(state)
        return obs, norm_state

    def step_env(self, key: jax.Array, state: NormRewState, action: int | float | jax.Array, params: EnvParams):
        new_obs, new_state, reward, done, info = super().step_env(key, state, action, params)

        # Perform normalization of the reward

        # 2) Welford update
        r = jnp.array(reward, dtype=jnp.float32)
        cnt = state.rew_count + 1
        delta = r - state.rew_mean
        mean = state.rew_mean + delta / cnt
        m2 = state.rew_M2 + delta * (r - mean)
        var = m2 / cnt
        std = jnp.sqrt(var)

        # 3) Normalize
        norm_r = r / (std + self.eps)

        # 4) Conditionally clip if clip_range is defined
        clipped_norm_r = jnp.clip(norm_r, -self.clip_range, self.clip_range)

        # 5) Pack new state
        new_state_norm = NormRewState.create_from_state(
            new_state,
            rew_count=cnt,
            rew_mean=mean,
            rew_M2=m2
        )

        return new_obs, new_state_norm, clipped_norm_r, done, info
