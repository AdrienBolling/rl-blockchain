import cProfile
from rl_blockchain.rl.env import BlockchainEnv_intermediary
import jax
import os
import jax.numpy as jnp
from jax import profiler as jpr
import time

def make_env_params(key, n_nodes, voting_nodes):
    # Create a random symmetric distance matrix
    A = jax.random.uniform(key, (n_nodes, n_nodes))
    dist = (A + A.T) * 0.5
    # Create environment parameters
    env_params = {
        "node_distance_matrix": dist,
        "voting_nodes": voting_nodes,
        "random_key": key,
    }
    return env_params

def env_profiling(args):
    """
    Run profiling tests on the environment.
    
    Args:
        args: Parsed arguments from the command line.
    """

    # Create a random key for JAX
    key = jax.random.PRNGKey(args.seed)
    
    # Create environment parameters
    env_params = make_env_params(
        key=key,
        n_nodes=args.n_nodes,
        voting_nodes=args.voting_nodes
    )
    test_weights = jax.numpy.array([0.5, 0.5])
    
    # Start a JAX Trace
    with jpr.trace(os.path.join(args.results_dir, "jax_trace"), create_perfetto_trace=True):
    
    
        # Start with a single environment, step over it and profile the time taken
        env_fn = BlockchainEnv_intermediary


        # Begin profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Create the environment
        env = env_fn(**env_params)
        state = env.reset()
        
        start = time.time()
        # Step through the environment
        for _ in range(args.n_steps):
            
            key, subkey = jax.random.split(key)
            action = env.sample_legal_action(state, subkey)
            state, reward, done, infos = env.step(state, action, test_weights)
            if done:
                state = env.reset()
        end = time.time()
        print(f"Time taken for {args.n_steps} steps: {end - start:.4f} seconds (1 env)")
            
            
        # Stop profiling
        profiler.disable()
        
        # Save the profiling results
        profiler.dump_stats(os.path.join(args.results_dir, "single_env_profiling.prof"))
        
        

        
        # Now run the same test with multiple environments
        key, subkey = jax.random.split(key)
        B = args.n_envs
        T = args.n_steps
        env = env_fn(**env_params)
        vmapped_reset = jax.vmap(lambda _: env.reset(), in_axes=(0,))
        init_dummy_state = jnp.arange(B)
        initial_states = vmapped_reset(init_dummy_state)
        
        # Pre-split the key for multiple environments
        key, subkey = jax.random.split(key)
        flat_keys = jax.random.split(subkey, T * B)
        step_keys = flat_keys.reshape((T, B, 2))
        
        def step_fn(state, key):
            action = env.sample_legal_action(state, key)
            state, reward, done, infos = env.step(state, action, test_weights)
            return state, reward, done, infos
        
        vmapped_step = jax.vmap(step_fn, in_axes=(0, 0), out_axes=(0, 0, 0, 0))
        
        @jax.jit
        def run_all_steps(states, all_keys):
            T, B, _ = all_keys.shape
            def body_fn(i, st):
                rng = all_keys[i]
                new_states, rewards, dones, infos = vmapped_step(st, rng)
                
                resets = vmapped_reset(jnp.arange(B))
                
                def mask_fn(ns, rs):
                    # ns.shape == rs.shape == (B, d1, d2, …)
                    # first reshape to (B, 1, 1, …) then broadcast
                    cond = dones.reshape((B,) + (1,) * (ns.ndim - 1))
                    cond = jnp.broadcast_to(cond, ns.shape)
                    return jnp.where(cond, rs, ns)
            
                masked_states = jax.tree.map(mask_fn, new_states, resets)
                return masked_states
            
            final_states = jax.lax.fori_loop(0, T, body_fn, states)
            return final_states
        
        # Begin profiling
        profiler = cProfile.Profile()
        profiler.enable()
        start = time.time()
        # Run the steps
        final_states = run_all_steps(initial_states, step_keys)
        
        # Block until all computations are done
        jax.tree.map(lambda x: x.block_until_ready(), final_states)
        # Stop profiling
        end = time.time()
        profiler.disable()
        profiler.dump_stats(os.path.join(args.results_dir, "multi_env_profiling.prof"))=
        print(f"Time taken for {args.n_steps} steps: {end - start:.4f} seconds ({args.n_envs} envs)")

                
        