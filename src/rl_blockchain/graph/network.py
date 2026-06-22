"""Fractal (hierarchical) network topology generation.

We model the physical network as a **tree of routers** carrying latencies, with
the blockchain nodes ("clients") hanging off the leaf routers:

* A root router branches into a random number of child routers (up to
  ``branching[0]``); each of those branches again (up to ``branching[1]``), and
  so on. Each router may also *stop early* with probability ``stop_prob``, so
  individual branches reach different depths — a self-similar, fractal-ish
  backbone. Every router→parent link carries a random latency.
* Each blockchain node is attached to a (randomly chosen) leaf router with its
  own random access latency.
* The distance between two nodes is the network latency of the unique path
  between them: ``lat_i + routerdist(r_i, r_j) + lat_j``.

Why it is efficient
--------------------
The router backbone is a *tree*, so the number of routers ``R`` is tiny next to
the number of nodes ``N``. We therefore:

1. Build the (irregular) router tree once on the host — O(R).
2. Solve the ``R×R`` router distance matrix (weighted Floyd–Warshall, jitted).
   Small, since ``R ≪ N``.
3. Assemble the full ``N×N`` node distance matrix with a single vectorised
   gather + broadcast on the GPU. This O(N²) step is the only unavoidable cost
   (a dense distance matrix is inherently N²) and is fused into one kernel.

The random *structure* is generated on the host (Python control flow) because
variable branching/depth is ragged and does not fit ``jit``'s static shapes —
but all sampling uses ``jax.random`` with explicit key splitting, so this part,
while not jitted, stays reproducible and consistent with the rest of the
codebase.
"""

from __future__ import annotations

import dataclasses

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .blockchain_graph import BlockchainGraph


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    """Configuration for :func:`generate_network`.

    Attributes:
        num_nodes: Number of blockchain nodes (graph clients) to create.
        branching: Per-level cap on the number of child routers. ``len`` sets
            the maximum tree depth. E.g. ``(10, 6, 3)`` ⇒ root spawns ≤10, each
            ≤6, each ≤3.
        stop_prob: Probability that a non-root router stops branching early
            (becomes a leaf), giving branches of random depth.
        min_children: Minimum children a branching router spawns.
        router_latency_range: ``(lo, hi)`` uniform range for router-link latency.
        client_latency_range: ``(lo, hi)`` uniform range for node access latency.
    """

    num_nodes: int
    branching: tuple[int, ...] = (10, 6, 3)
    stop_prob: float = 0.3
    min_children: int = 1
    router_latency_range: tuple[float, float] = (1.0, 10.0)
    client_latency_range: tuple[float, float] = (0.1, 2.0)


@flax.struct.dataclass
class NetworkTopology:
    """Result of :func:`generate_network` (all fields are JAX arrays).

    Router-level (length ``R``):
        router_parents: Parent index per router (``-1`` for the root).
        router_depth: Depth of each router (root = 0).
        router_edge_latency: Latency of each router→parent link (0 for root).
        router_is_leaf: Boolean mask of leaf routers.
        router_distance_matrix: ``[R, R]`` shortest-path latency between routers.

    Node-level (length ``N``):
        client_router: Index of the leaf router each node is attached to.
        client_latency: Access latency of each node to its router.
        distance_matrix: ``[N, N]`` node↔node latency distance (the main output;
            feed it straight into the reward state).
    """

    router_parents: Array
    router_depth: Array
    router_edge_latency: Array
    router_is_leaf: Array
    router_distance_matrix: Array
    client_router: Array
    client_latency: Array
    distance_matrix: Array


# --------------------------------------------------------------------------- #
# Host-side tree construction (irregular -> NumPy)
# --------------------------------------------------------------------------- #
def _build_router_tree(
    key: Array, cfg: NetworkConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the random router tree (BFS order ⇒ every parent precedes its child).

    Sampling uses ``jax.random`` (split per branching router); control flow is
    host-side because the structure is ragged. Returns ``(parents, depth,
    edge_latency)`` as NumPy arrays of length ``R``.
    """
    parents = [-1]
    depth = [0]
    edge_latency = [0.0]
    queue = [0]
    max_depth = len(cfg.branching)
    lo, hi = cfg.router_latency_range

    while queue:
        r = queue.pop(0)
        d = depth[r]
        if d >= max_depth:
            continue  # reached the configured maximum depth -> leaf
        key, k_stop, k_n, k_lat = jax.random.split(key, 4)
        if d > 0 and float(jax.random.uniform(k_stop)) < cfg.stop_prob:
            continue  # random early stop -> leaf
        max_children = cfg.branching[d]
        n_children = int(jax.random.randint(k_n, (), cfg.min_children, max_children + 1))
        latencies = jax.random.uniform(k_lat, (n_children,), minval=lo, maxval=hi)
        for c in range(n_children):
            child = len(parents)
            parents.append(r)
            depth.append(d + 1)
            edge_latency.append(float(latencies[c]))
            queue.append(child)

    return (
        np.asarray(parents, dtype=np.int32),
        np.asarray(depth, dtype=np.int32),
        np.asarray(edge_latency, dtype=np.float32),
    )


def _router_adjacency(parents: np.ndarray, edge_latency: np.ndarray) -> np.ndarray:
    """Dense ``[R, R]`` undirected adjacency with ``inf`` off-tree, 0 diagonal."""
    r = len(parents)
    adj = np.full((r, r), np.inf, dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    children = np.arange(1, r)  # every router except the root has a parent edge
    par = parents[children]
    lat = edge_latency[children]
    adj[children, par] = lat
    adj[par, children] = lat
    return adj


# --------------------------------------------------------------------------- #
# Jitted distance computations
# --------------------------------------------------------------------------- #
@jax.jit
def _weighted_all_pairs(adjacency: Array) -> Array:
    """Weighted all-pairs shortest paths (Floyd–Warshall) over ``R`` routers."""
    r = adjacency.shape[0]

    def relax(k, d):
        return jnp.minimum(d, d[:, k][:, None] + d[k, :][None, :])

    return jax.lax.fori_loop(0, r, relax, adjacency)


@jax.jit
def _client_distance_matrix(
    router_distance: Array, client_router: Array, client_latency: Array
) -> Array:
    """Assemble the ``[N, N]`` node distance matrix from router distances.

    ``dist[i, j] = lat_i + lat_j + routerdist[r_i, r_j]`` (diagonal forced to 0).
    A single fused gather + broadcast — the GPU-friendly O(N²) step.
    """
    n = client_latency.shape[0]
    routed = router_distance[client_router][:, client_router]  # [N, N]
    dist = client_latency[:, None] + client_latency[None, :] + routed
    return dist.at[jnp.diag_indices(n)].set(0.0)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_network(
    key: Array, cfg: NetworkConfig
) -> tuple[BlockchainGraph, NetworkTopology]:
    """Generate a fractal router backbone with clients and their distance matrix.

    Args:
        key: ``jax`` PRNGKey — split across tree, client-assignment and latency
            sampling for reproducibility.
        cfg: Topology configuration.

    Returns:
        ``(graph, topology)``. ``graph`` is a :class:`BlockchainGraph` with
        ``cfg.num_nodes`` client nodes (default features, no inter-client edges —
        connectivity is modelled by the latency ``distance_matrix``).
        ``topology.distance_matrix`` is the ``[N, N]`` latency matrix to feed the
        reward state.
    """
    key_tree, key_assign, key_latency = jax.random.split(key, 3)

    parents, depth, edge_latency = _build_router_tree(key_tree, cfg)
    num_routers = len(parents)

    is_leaf = np.ones(num_routers, dtype=bool)
    internal = np.unique(parents[parents >= 0])
    is_leaf[internal] = False
    leaves = np.flatnonzero(is_leaf)

    # Attach each client to a random leaf router with a random access latency.
    client_leaf = jax.random.randint(key_assign, (cfg.num_nodes,), 0, len(leaves))
    client_router = jnp.asarray(leaves)[client_leaf].astype(jnp.int32)
    lo, hi = cfg.client_latency_range
    client_latency = jax.random.uniform(
        key_latency, (cfg.num_nodes,), minval=lo, maxval=hi
    ).astype(jnp.float32)

    router_distance = _weighted_all_pairs(
        jnp.asarray(_router_adjacency(parents, edge_latency))
    )
    distance_matrix = _client_distance_matrix(
        router_distance, client_router, client_latency
    )

    topology = NetworkTopology(
        router_parents=jnp.asarray(parents),
        router_depth=jnp.asarray(depth),
        router_edge_latency=jnp.asarray(edge_latency),
        router_is_leaf=jnp.asarray(is_leaf),
        router_distance_matrix=router_distance,
        client_router=jnp.asarray(client_router),
        client_latency=jnp.asarray(client_latency),
        distance_matrix=distance_matrix,
    )
    graph = BlockchainGraph.new(num_nodes=cfg.num_nodes)
    return graph, topology
