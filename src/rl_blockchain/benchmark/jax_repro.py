"""Stage-by-stage repro / profiler for the blockchain PPO model.

Builds the real env + GNN and runs rollout, GAE, PPO update, and evaluation,
timing and isolating each stage while reporting GPU memory. Separates compile
time (first call) from run time (subsequent calls). The update stage is the
dominant VRAM consumer and its peak scales ~linearly with batch size.

Examples::

    # full run at 200 nodes:
    uv run --no-sync python -m rl_blockchain.benchmark.jax_repro --n-nodes 200

    # sweep batch size to see how the update's peak VRAM scales:
    for b in 16 32 64; do
      uv run --no-sync python -m rl_blockchain.benchmark.jax_repro --n-nodes 200 --batch-size $b
    done
"""

from __future__ import annotations

import argparse
import time
import traceback
from contextlib import contextmanager

from rl_blockchain.utils.jax_runtime import configure_xla_flags

_XLA = configure_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _mem() -> str:
    try:
        s = jax.devices()[0].memory_stats() or {}
        return (f"mem in_use={s.get('bytes_in_use', 0) / 1e9:.2f}GB "
                f"peak={s.get('peak_bytes_in_use', 0) / 1e9:.2f}GB")
    except Exception:
        return "mem n/a"


@contextmanager
def stage(name: str):
    print(f"[ .. ] {name}: starting ...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        print(f"[FAIL] {name}: raised after {time.perf_counter() - t0:6.1f}s  ({_mem()})", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
    print(f"[ OK ] {name}: {time.perf_counter() - t0:6.1f}s  ({_mem()})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-nodes", type=int, default=200)
    p.add_argument("--gat-arch", type=int, nargs=3, default=[16, 16, 16],
                   metavar=("BACKBONE", "ACTOR", "CRITIC"))
    p.add_argument("--voting-nodes", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-steps", type=int, default=None,
                   help="episode length for the eval stage (default: env max_steps)")
    p.add_argument("--eval-envs", type=int, default=None,
                   help="parallel envs for the eval stage (default: num-envs)")
    p.add_argument("--micro-batch-size", type=int, default=None,
                   help="gradient-accumulation micro-batch for the update stage "
                        "(must divide --batch-size); reduces update peak VRAM")
    args = p.parse_args()

    print("=" * 68)
    print(f"jax {jax.__version__}  backend {jax.default_backend()}  devices {jax.devices()}")
    print(f"XLA_FLAGS: {_XLA or '(none)'}")
    print(f"n_nodes={args.n_nodes} gat_arch={args.gat_arch} num_steps={args.num_steps} "
          f"num_envs={args.num_envs} batch_size={args.batch_size} "
          f"micro_batch_size={args.micro_batch_size}")
    print(f"edges/graph = {args.n_nodes * (args.n_nodes - 1)}   ({_mem()})")
    print("=" * 68)

    from rl_blockchain.scripts.env_factory import GenericEnvFactory
    from rl_blockchain.scripts.parser import UpdateValStrat, UpdateDistStrat
    from rl_blockchain.algo.ppo import (
        create_ppo_state, make_optimizer, update_ppo, compute_gae,
        _vectorized_rollout, _vectorized_rollout_eval,
    )

    key = jax.random.PRNGKey(0)
    kb, krun = jax.random.split(key)
    config = {"n_nodes": args.n_nodes, "gat_arch": args.gat_arch,
              "voting_nodes": args.voting_nodes, "reward_weights": [0.5, 0.5],
              "next_val_type": UpdateValStrat.NO_UPDATE,
              "next_edge_type": UpdateDistStrat.NO_UPDATE}

    with stage("build env + model"):
        model, env, _, create_params_fn, _ = GenericEnvFactory.create("blockenv", kb, config)

    with stage("model.init + optimizer (create_ppo_state)"):
        ppo_state = create_ppo_state(resume_dir=None, env=env, seed=0, lr=1e-3, model=model)
        model_opt = make_optimizer(1e-3)

    rollout_key, params_key = jax.random.split(krun)
    subkeys = jax.random.split(rollout_key, args.num_envs)
    params_list = jax.vmap(create_params_fn)(jax.random.split(params_key, args.num_envs))

    # -------- rollout --------
    vm_rollout = _vectorized_rollout(env, model, args.num_steps)
    with stage("rollout COMPILE + run"):
        obs, perms, logps, rews, dones, vals, last_values, infos = vm_rollout(
            subkeys, ppo_state, params_list)
        jax.block_until_ready((rews, vals))
    with stage(f">>> rollout EXEC only (num_steps={args.num_steps}) <<<"):
        jax.block_until_ready(vm_rollout(subkeys, ppo_state, params_list)[0].nodes)

    # -------- GAE --------
    with stage("GAE"):
        advantages = jax.vmap(lambda r, v, d, lv: compute_gae(r, v, d, lv, 0.99, 0.95))(
            rews, vals, dones, last_values)
        returns = advantages + vals
        jax.block_until_ready(returns)

    def flatten(x):
        return x.reshape(-1, *x.shape[2:])

    flat_perms, flat_lp, flat_r = flatten(perms), flatten(logps), flatten(returns)
    flat_adv, flat_val = flatten(advantages), flatten(vals)
    idx = jnp.arange(args.batch_size)
    batch_graphs = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:])[idx], obs)

    def do_update():
        return update_ppo(ppo_state, batch_graphs, flat_perms[idx], flat_lp[idx],
                          flat_r[idx], flat_adv[idx], flat_val[idx],
                          model.apply, model_opt, 0.2,
                          micro_batch_size=args.micro_batch_size)

    # -------- update (dominant VRAM stage; micro-batching lowers its peak) --------
    mb = args.micro_batch_size
    with stage(f"update_ppo COMPILE + run (micro_batch_size={mb})"):
        jax.block_until_ready(do_update()[0].params)
    for i in range(3):
        with stage(f">>> update_ppo EXEC only, call {i + 1}/3 (micro={mb}) <<<"):
            jax.block_until_ready(do_update()[0].params)

    # -------- eval --------
    eval_steps = args.eval_steps or int(env.default_params.max_steps_in_episode)
    eval_envs = args.eval_envs or args.num_envs
    vm_eval = _vectorized_rollout_eval(env, model, eval_steps)
    ek = jax.random.split(jax.random.PRNGKey(1), eval_envs)
    ep = jax.vmap(create_params_fn)(jax.random.split(jax.random.PRNGKey(2), eval_envs))
    with stage(f"eval rollout COMPILE + run (steps={eval_steps}, envs={eval_envs})"):
        jax.block_until_ready(vm_eval(ek, ppo_state, ep))
    with stage(">>> eval rollout EXEC only <<<"):
        jax.block_until_ready(vm_eval(ek, ppo_state, ep))

    print("=" * 68)
    print(f"ALL STAGES OK at n_nodes={args.n_nodes}.  {_mem()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
