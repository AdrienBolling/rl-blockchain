"""JAX/XLA runtime configuration and safety checks.

This module centralises two concerns that turned out to matter a lot when
moving the project from ``jax==0.9.0.1`` to ``jax>=0.10``:

1. **XLA GPU compilation flags.** JAX 0.10 ships a newer XLA that, by default,
   lowers matmuls through the Triton GEMM emitter and *autotunes* every
   ``__triton_nested_gemm_fusion`` region with a ``cuda_timer`` "delay kernel".
   On the many *tiny* GEMMs produced by the GNN/GAT model, that autotuning is
   both slow to compile and prone to the
   ``cuda_timer.cc:87] Delay kernel timed out: measured time has sub-optimal
   accuracy`` warnings, which lead to poor kernel selection. Routing GEMMs back
   to cuBLAS (``--xla_gpu_enable_triton_gemm=false``) restores the 0.9 behaviour
   without disabling the GPU. See :func:`configure_xla_flags`.

2. **Fail loud, never fall back to CPU silently.** :func:`require_gpu` raises if
   the GPU backend is not the one JAX picked when we expect GPU execution.

IMPORTANT: :func:`configure_xla_flags` only mutates ``os.environ`` and therefore
**must be imported and called before the first ``import jax``** anywhere in the
process. It deliberately does not import jax itself.
"""

from __future__ import annotations

import os
import warnings

# Named XLA-flag profiles. IMPORTANT: none of these are applied by default.
#
# We benchmarked them (see rl_blockchain.benchmark.jax_diag) and the result is
# *hardware dependent*: on a healthy GPU the Triton GEMM emitter produces faster
# kernels than cuBLAS, it just costs more to compile/autotune. So disabling it
# is only worth it on GPUs where the autotuner misbehaves -- exactly the
# "cuda_timer.cc:87 Delay kernel timed out: measured time has sub-optimal
# accuracy" situation seen on the target server. Opt in explicitly, after
# measuring on the target machine, via RLB_XLA_PROFILE.
_XLA_PROFILES = {
    # Route GEMMs through cuBLAS instead of the Triton GEMM emitter. Direct
    # counter to __triton_nested_gemm_fusion / xtile_compiler / delay-kernel
    # autotuning blow-ups. Faster compile, sometimes slower steady-state.
    "cublas": "--xla_gpu_enable_triton_gemm=false",
    # Keep Triton but skip autotuning entirely (use default tilings). Avoids the
    # delay-kernel timeout warnings without giving up Triton codegen.
    "no-autotune": "--xla_gpu_autotune_level=0",
    # Both, for GPUs where autotuning is badly broken.
    "safe": "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
    "none": "",
}


def configure_xla_flags() -> str:
    """Apply an opt-in XLA_FLAGS profile for GPU compilation. Idempotent.

    Must be called *before* jax is imported. By default this is a **no-op**
    (upstream XLA behaviour) so the code-level fixes stay the primary solution.

    Env controls (checked in this order):
      * ``RLB_DISABLE_XLA_FLAGS=1`` -- do not touch XLA_FLAGS at all.
      * ``RLB_XLA_FLAGS=<string>``  -- full override, used verbatim.
      * ``RLB_XLA_PROFILE=<name>``  -- one of ``cublas``, ``no-autotune``,
        ``safe``, ``none``. Recommended only after benchmarking on the target
        GPU (``python -m rl_blockchain.benchmark.jax_diag``).

    Returns the resulting ``XLA_FLAGS`` string.
    """
    if os.environ.get("RLB_DISABLE_XLA_FLAGS") == "1":
        return os.environ.get("XLA_FLAGS", "")

    existing = os.environ.get("XLA_FLAGS", "")
    override = os.environ.get("RLB_XLA_FLAGS")
    profile = os.environ.get("RLB_XLA_PROFILE")

    if override is not None:
        extra = override
    elif profile:
        if profile not in _XLA_PROFILES:
            raise ValueError(
                f"Unknown RLB_XLA_PROFILE={profile!r}; choose from {sorted(_XLA_PROFILES)}"
            )
        extra = _XLA_PROFILES[profile]
    else:
        extra = ""  # default: change nothing

    # Only add flags the user has not already set explicitly.
    to_add = [f for f in extra.split() if f.split("=")[0] not in existing]
    merged = " ".join(filter(None, [existing, *to_add]))
    os.environ["XLA_FLAGS"] = merged
    return merged


def require_gpu(expect_gpu: bool = True) -> None:
    """Fail clearly if the active JAX backend is not what we expect.

    With ``expect_gpu=True`` (the default for training/eval) this raises a
    ``RuntimeError`` when JAX has fallen back to CPU, instead of silently
    training ~100x slower. Set ``RLB_ALLOW_CPU=1`` to downgrade to a warning
    (useful for laptops / CI without a GPU).
    """
    import jax  # local import: must happen after configure_xla_flags()

    backend = jax.default_backend()
    devices = jax.devices()
    allow_cpu = os.environ.get("RLB_ALLOW_CPU") == "1"

    if expect_gpu and backend != "gpu":
        msg = (
            f"Expected JAX to use the GPU but default_backend()={backend!r} "
            f"(devices={devices}). This usually means the CUDA plugin failed to "
            f"load. Refusing to run on CPU. Check `uv run python -c "
            f"'import jax; print(jax.print_environment_info())'`. "
            f"Set RLB_ALLOW_CPU=1 to override."
        )
        if allow_cpu:
            warnings.warn(msg)
        else:
            raise RuntimeError(msg)

    _warn_on_plugin_driver_mismatch()


def _warn_on_plugin_driver_mismatch() -> None:
    """Warn if the installed CUDA *plugin* major version differs from the driver.

    On this project the pyproject requests ``jax[cuda13]`` but uv can resolve the
    ``jax-cuda12-plugin`` wheels, which then run against a CUDA 13 driver. That
    mismatch is a common source of flaky XLA autotuning on 0.10.
    """
    try:
        import importlib.metadata as md

        installed = {d.metadata["Name"].lower() for d in md.distributions()}
        plugin_majors = {
            name.split("-")[2][4:]  # jax-cuda12-plugin -> "12"
            for name in installed
            if name.startswith("jax-cuda") and name.endswith("-plugin")
        }
        if len(plugin_majors) > 1:
            warnings.warn(f"Multiple jax-cuda*-plugin majors installed: {plugin_majors}")
    except Exception:
        pass


class compile_counter:
    """Context manager counting XLA compilations that happen inside it.

    Uses jax's ``jax_log_compiles`` machinery via a lightweight monkeypatch of
    the JIT lowering cache-miss path. Best-effort: if the internal hook is not
    available on this jax version, ``.count`` stays at 0 and ``.available`` is
    False.

    Usage::

        with compile_counter() as c:
            f(x); g(y)
        print(c.count)
    """

    def __init__(self):
        self.count = 0
        self.available = False
        self._orig = None
        self._mod = None
        self._attr = None

    def __enter__(self):
        import jax

        # jax exposes a cache-miss callback we can wrap. The exact location has
        # moved across versions, so probe a couple of candidates.
        candidates = [
            ("jax._src.compiler", "compile_or_get_cached"),
        ]
        for modname, attr in candidates:
            try:
                mod = __import__(modname, fromlist=[attr])
                orig = getattr(mod, attr)
            except Exception:
                continue

            counter = self

            def wrapped(*a, __orig=orig, **k):
                counter.count += 1
                return __orig(*a, **k)

            setattr(mod, attr, wrapped)
            self._mod, self._attr, self._orig = mod, attr, orig
            self.available = True
            break
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            setattr(self._mod, self._attr, self._orig)
        return False