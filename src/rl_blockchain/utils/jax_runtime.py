"""JAX/XLA runtime configuration and a GPU safety check.

Two small helpers, both called from ``scripts/run.py``:

* :func:`configure_xla_flags` applies the one XLA GPU flag that this project's
  GNN needs on jax 0.10 (see the comment on ``_SCATTER_FIX`` below). It only
  touches ``os.environ`` and must run **before the first ``import jax``**.
* :func:`require_gpu` refuses to run silently on CPU when GPU is expected.
"""

from __future__ import annotations

import os
import warnings

# MEASURED fix for the jax-0.10 "hangs forever" at 200 nodes: the GNN's
# segment_sum / segment_softmax aggregations lower to scatter-add, and 0.10's
# default GPU scatter lowering is ~1100x slower on the target GPU (tabriz). The
# scatter-determinism expander rewrites it into a fast -- and deterministic --
# sorted-segment form: model.apply @200 nodes goes 1342ms -> 5.2ms.
#
# NOTE: the optimal value is GPU-dependent. On GPUs whose native atomic scatter
# is already fast, `true` can be slower -- override with RLB_XLA_FLAGS there.
_SCATTER_FIX = "--xla_gpu_enable_scatter_determinism_expander=true"


def configure_xla_flags() -> str:
    """Prepend the project's XLA GPU fix to ``XLA_FLAGS``. Idempotent.

    Must be called before jax is imported. Anything already in ``XLA_FLAGS``
    is preserved (a flag we would add is skipped if the user already set it).

    Env overrides:
      * ``RLB_DISABLE_XLA_FLAGS=1`` -- leave ``XLA_FLAGS`` untouched.
      * ``RLB_XLA_FLAGS=<string>``  -- use this instead of the default fix.

    Returns the resulting ``XLA_FLAGS`` string.
    """
    if os.environ.get("RLB_DISABLE_XLA_FLAGS") == "1":
        return os.environ.get("XLA_FLAGS", "")

    existing = os.environ.get("XLA_FLAGS", "")
    extra = os.environ.get("RLB_XLA_FLAGS", _SCATTER_FIX)

    to_add = [f for f in extra.split() if f.split("=")[0] not in existing]
    merged = " ".join(filter(None, [existing, *to_add]))
    os.environ["XLA_FLAGS"] = merged
    return merged


def require_gpu(expect_gpu: bool = True) -> None:
    """Fail clearly if JAX did not pick the GPU when GPU execution is expected.

    Raises ``RuntimeError`` instead of silently training ~100x slower on CPU.
    Set ``RLB_ALLOW_CPU=1`` to downgrade to a warning (CPU-only dev / CI).
    """
    import jax  # local import: must happen after configure_xla_flags()

    backend = jax.default_backend()
    if expect_gpu and backend != "gpu":
        msg = (
            f"Expected JAX to use the GPU but default_backend()={backend!r} "
            f"(devices={jax.devices()}). The CUDA plugin likely failed to load; "
            f"refusing to run on CPU. Set RLB_ALLOW_CPU=1 to override."
        )
        if os.environ.get("RLB_ALLOW_CPU") == "1":
            warnings.warn(msg)
        else:
            raise RuntimeError(msg)
