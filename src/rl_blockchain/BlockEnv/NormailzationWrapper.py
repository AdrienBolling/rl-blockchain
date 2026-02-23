import jax
import jax.numpy as jnp
from flax.struct import dataclass

from rl_blockchain.BlockEnv import EnvState, BlockchainEnv, EnvParams


# ======================================================================
# Reward normalization state (wrapper-only, not env core)
# ======================================================================

@dataclass
class NormRewStats:
    count: jax.Array  # shape=(), int32
    mean: jax.Array  # shape=(), float32
    m2: jax.Array  # shape=(), float32

    @classmethod
    def init(cls):
        return cls(
            count=jnp.array(0, dtype=jnp.int32),
            mean=jnp.array(0.0, dtype=jnp.float32),
            m2=jnp.array(0.0, dtype=jnp.float32),
        )


@dataclass
class NormWrappedState:
    env_state: EnvState
    stats: NormRewStats


# ======================================================================
# Reward Normalization Wrapper
# ======================================================================

class NormalizationWrapper(BlockchainEnv):
    """
    Running mean / std reward normalization using Welford.
    """

    def __init__(
            self,
            block_env: BlockchainEnv,
            eps: float = 1e-8,
            clip_range: float | None = None,
            min_count: int = 2,
    ):
        super().__init__(block_env._first_params, block_env._static_params)
        self.eps = eps
        self.clip_range = clip_range
        self.min_count = min_count

    def reset_env(self, key: jax.Array, params: EnvParams):
        obs, env_state = super().reset_env(key, params)
        state = NormWrappedState(
            env_state=env_state,
            stats=NormRewStats.init(),
        )
        return obs, state

    def step_env(
            self,
            key: jax.Array,
            state: NormWrappedState,
            action,
            params: EnvParams,
    ):
        obs, env_state, reward, done, info = super().step_env(
            key, state.env_state, action, params
        )

        r = jnp.asarray(reward, dtype=jnp.float32)

        # Welford update
        count = state.stats.count + 1
        delta = r - state.stats.mean
        mean = state.stats.mean + delta / count
        delta2 = r - mean
        m2 = state.stats.m2 + delta * delta2

        # variance / std (safe for early steps)
        var = jnp.where(
            count >= self.min_count,
            m2 / (count - 1),
            jnp.array(1.0, dtype=jnp.float32),
        )
        std = jnp.sqrt(var)

        # normalize (centered)
        norm_r = (r - mean) / (std + self.eps)

        # optional clipping
        if self.clip_range is not None:
            norm_r = jnp.clip(norm_r, -self.clip_range, self.clip_range)

        new_state = NormWrappedState(
            env_state=env_state,
            stats=NormRewStats(
                count=count,
                mean=mean,
                m2=m2,
            ),
        )

        # diagnostics
        info = dict(info)
        info.update({
            "rew_raw": r,
            "rew_norm": norm_r,
            "rew_mean": mean,
            "rew_std": std,
            "rew_count": count,
        })

        return obs, new_state, norm_r, done, info
