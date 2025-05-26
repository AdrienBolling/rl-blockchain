from flax.struct import dataclass
from rl_blockchain.rl.env import State
from jax import jit
from functools import partial
import jax.numpy as jnp
import jax

from rl_blockchain.rl.wrappers.Wrapper import Wrapper

#======================================================================
# Reward Normalization Wrapper
#======================================================================

@dataclass
class NormRewState(State):
    """
    Extension of the State class to include normalization parameters for rewards.
    """
    rew_count: int = 0
    rew_mean: float = 0.0
    rew_M2: float = 0.0
    
class NormalizationWrapper(Wrapper):
    """
    A wrapper for normalizing rewards in a reinforcement learning environment.
    This wrapper maintains the mean and variance of the rewards to normalize them.
    """
    
    def __init__(self, env, eps:float = 1e-8, clip_range: float = 10.0):
        """
        Initialize the NormalizationWrapper.
        
        Args:
            env: The environment to wrap.
            eps: A small value to avoid division by zero.
            clip_range: The range to clip the normalized rewards.
        """
        self.env = env
        self.eps = eps
        self.clip_range = clip_range
    
    @partial(jit, static_argnums=[0])
    def reset(self):
        """
        Reset the environment and return the initial state.
        
        Returns:
            The initial state of the environment.
        """
        base_state = self.env.reset()
        
        # Create a new state with normalization parameters
        norm_state = NormRewState(
            blockchain=base_state.blockchain,
            time_step=base_state.time_step,
            inner_step=base_state.inner_step,
            global_step=base_state.global_step,
            rew_count=0,
            rew_mean=0.0,
            rew_M2=0.0
        )
        return norm_state
    
    @partial(jit, static_argnames=('self',))
    def step(self, state: State, action: jnp.ndarray, weights: jnp.ndarray):
        
        next_state, reward, done, info = self.env.step(state, action, weights)
        
        # Perform normalization of the reward
        
        # 2) Welford update
        r     = jnp.array(reward, dtype=jnp.float32)
        cnt   = state.rew_count + 1
        delta = r - state.rew_mean
        mean  = state.rew_mean + delta / cnt
        M2    = state.rew_M2 + delta * (r - mean)
        var   = M2 / cnt
        std   = jnp.sqrt(var)

        # 3) Normalize
        norm_r = r / (std + self.eps)

        # 4) Conditionally clip if clip_range is defined
        norm_r = jax.lax.cond(
            self.clip_range is not None,
            lambda rr: jnp.clip(rr, -self.clip_range, self.clip_range),
            lambda rr: rr,
            norm_r
        )

        # 5) Pack new state
        new_state = NormRewState(
            blockchain=next_state.blockchain,
            time_step=next_state.time_step,
            inner_step=next_state.inner_step,
            global_step=next_state.global_step,
            rew_count=cnt,
            rew_mean=mean,
            rew_M2=M2
        )

        return new_state, norm_r, done, info