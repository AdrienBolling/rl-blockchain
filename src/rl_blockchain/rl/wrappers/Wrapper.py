import jax
from functools import partial
import jax.numpy as jnp
from rl_blockchain.rl.env import State

class Wrapper:
    
    """
    A base wrappe class for this environment.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper with the environment.
        
        Args:
            env: The environment to wrap.
        """
        self.env = env
            
    @partial(jax.jit, static_argnums=[0])
    def reset(self):
        """
        Reset the environment and return the initial state.
        
        Returns:
            The initial state of the environment.
        """
        return self.env.reset()
    
    @partial(jax.jit, static_argnames=('self',))
    def step(self, state: State, action: jnp.ndarray, weights: jnp.ndarray):
        """
        Perform a step in the environment.
        
        Args:
            state: The current state of the environment.
            action: The action to take.
            weights: Weights for the action (if applicable).
        
        Returns:
            A tuple containing the new state, reward, done flag, and info.
        """
        return self.env.step(state, action, weights)
    
    @partial(jax.jit, static_argnums=[0])
    def sample_legal_action(self, state: State, key: jax.random.PRNGKey):
        """
        Sample a legal action from the environment.
        
        Args:
            state: The current state of the environment.
            key: A random key for sampling.
        
        Returns:
            A legal action for the current state.
        """
        return self.env.sample_legal_action(state, key)