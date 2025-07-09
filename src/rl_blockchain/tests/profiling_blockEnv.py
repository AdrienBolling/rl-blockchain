import cProfile
import os
import time

import jax
import jax.numpy as jnp
from jax import profiler as jpr

from rl_blockchain.BlockEnv import StaticEnvParams, EnvParams, BlockchainEnv
from rl_blockchain.scripts.parser import REF_FILENAME


def env_profiling(args):
    """
    Run profiling tests on the environment.
    
    Args:
        args: Parsed arguments from the command line.
    """

    # Create a random key for JAX
    key = jax.random.PRNGKey(args.seed)

    test_weights = jax.numpy.array([0.5, 0.5])

    static_params = StaticEnvParams.create(args.n_nodes, REF_FILENAME[args.n_nodes])
    params = EnvParams.create_random(args.n_nodes, key, args.voting_nodes, test_weights)

    # Start a JAX Trace
    with jpr.trace(os.path.join(args.results_dir, "jax_trace_BlockEnv"), create_perfetto_trace=True):

        # Start with a single environment, step over it and profile the time taken

        # Begin profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Create the environment
        env = BlockchainEnv(params, static_params)
        _, state = env.reset(key, params)

        start = time.time()
        # Step through the environment
        for _ in range(args.n_steps):

            key, subkey = jax.random.split(key)
            action = env.sample_legal_action(state, params, subkey)
            obs, state, reward, done, infos = env.step(subkey, state, action, params)
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
        vmapped_reset = jax.vmap(lambda subkey_reset: env.reset(subkey_reset, params), in_axes=(0,))
        key_reset = jax.random.split(subkey, args.n_envs)
        initial_obs, initial_states = vmapped_reset(key_reset)

        # Pre-split the key for multiple environments
        key, subkey = jax.random.split(key)
        flat_keys = jax.random.split(subkey, T * B)
        step_keys = flat_keys.reshape((T, B, 2))

        def step_fn(state, key):
            action = env.sample_legal_action(state, params, key)
            obs, state, reward, done, infos = env.step(key, state, action, params)
            return obs, state, reward, done, infos

        vmapped_step = jax.vmap(step_fn, in_axes=(0, 0), out_axes=(0, 0, 0, 0, 0))

        @jax.jit
        def run_all_steps(states, all_keys):
            T, B, _ = all_keys.shape

            def body_fn(i, st):
                rng = all_keys[i]
                new_obs, new_states, rewards, dones, infos = vmapped_step(st, rng)

                key_reset = jax.random.split(subkey, B)
                sub_obs, sub_states = vmapped_reset(key_reset)

                def mask_fn(ns, rs):
                    # ns.shape == rs.shape == (B, d1, d2, …)
                    # first reshape to (B, 1, 1, …) then broadcast
                    cond = dones.reshape((B,) + (1,) * (ns.ndim - 1))
                    cond = jnp.broadcast_to(cond, ns.shape)
                    return jnp.where(cond, rs, ns)

                masked_states = jax.tree.map(mask_fn, new_states, sub_states)
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
        profiler.dump_stats(os.path.join(args.results_dir, "multi_env_profiling.prof"))
        print(f"Time taken for {args.n_steps} steps: {end - start:.4f} seconds ({args.n_envs} envs)")
