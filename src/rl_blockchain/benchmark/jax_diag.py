"""Standalone JAX/XLA diagnostic + micro-benchmark.

Separates, for a workload that mimics the project's GNN (many tiny GEMMs inside
a ``lax.scan``):

  * compile time (first call, includes XLA autotuning)
  * execution time after warmup
  * number of XLA compilations
  * backend + device placement

Run it with and without the XLA flags to quantify the Triton-GEMM autotuning
cost that regressed under jax 0.10::

    # baseline stack behaviour (Triton GEMM autotuning ON):
    RLB_DISABLE_XLA_FLAGS=1 uv run --no-sync python -m rl_blockchain.benchmark.jax_diag

    # with the project's default fix (cuBLAS GEMMs):
    uv run --no-sync python -m rl_blockchain.benchmark.jax_diag

    # log every compilation to confirm the recompilation count:
    JAX_LOG_COMPILES=1 uv run --no-sync python -m rl_blockchain.benchmark.jax_diag
"""

from __future__ import annotations

import argparse
import time

# Configure XLA flags BEFORE importing jax.
from rl_blockchain.utils.jax_runtime import configure_xla_flags, compile_counter

_XLA_FLAGS = configure_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _report_env() -> None:
    print("=" * 64)
    print(f"jax        : {jax.__version__}")
    print(f"jaxlib     : {jax.lib.__version__}")
    print(f"backend    : {jax.default_backend()}")
    print(f"devices    : {jax.devices()}")
    print(f"XLA_FLAGS  : {_XLA_FLAGS or '(none)'}")
    print("=" * 64)


def _make_workload(hidden: int, n_layers: int, steps: int, batch: int):
    """Many tiny dense layers inside a scan -- like the GAT backbone."""
    key = jax.random.PRNGKey(0)
    params = [
        jax.random.normal(jax.random.fold_in(key, i), (hidden, hidden)) * 0.1
        for i in range(n_layers)
    ]
    x = jnp.ones((batch, hidden), dtype=jnp.float32)

    @jax.jit
    def net(params, x):
        def step(h, _):
            for p in params:
                h = jnp.tanh(h @ p)
            return h, h.sum()
        h, ys = jax.lax.scan(step, x, None, length=steps)
        return ys.sum()

    return net, params, x


def benchmark(hidden: int, n_layers: int, steps: int, batch: int, iters: int) -> None:
    net, params, x = _make_workload(hidden, n_layers, steps, batch)

    with compile_counter() as c:
        t0 = time.perf_counter()
        y = net(params, x)
        y.block_until_ready()
        t_compile = time.perf_counter() - t0

    print(f"array device        : {y.device}")

    t0 = time.perf_counter()
    for _ in range(iters):
        y = net(params, x)
    y.block_until_ready()
    t_exec = (time.perf_counter() - t0) / iters

    print(f"compile + first call: {t_compile * 1e3:8.1f} ms")
    print(f"exec after warmup   : {t_exec * 1e3:8.3f} ms/call  (mean of {iters})")
    if c.available:
        print(f"XLA compilations    : {c.count}")
    else:
        print("XLA compilations    : (counter unavailable on this jax build)")
    print("=" * 64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--expect-gpu", action="store_true",
                   help="fail if JAX is not on the GPU")
    args = p.parse_args()

    _report_env()
    if args.expect_gpu:
        from rl_blockchain.utils.jax_runtime import require_gpu
        require_gpu(expect_gpu=True)
    benchmark(args.hidden, args.layers, args.steps, args.batch, args.iters)


if __name__ == "__main__":
    main()