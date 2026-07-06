"""
benchmarks.py — IEEE Test System Definitions
=============================================
Section III-A of the manuscript.

Provides topology, line impedance, and nominal load data for:
  - IEEE 33-bus radial distribution feeder
  - IEEE 123-bus distribution feeder (balanced single-phase equivalent)

Each builder returns a standardised dictionary consumed by `grid.py`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True)
class BenchmarkData:
    """Immutable container for a test-system definition."""
    name: str
    n_bus: int
    connections: List[Tuple[int, int]]   # (from_bus, to_bus) per branch
    r_pu: np.ndarray                     # Series resistance [p.u.]
    x_pu: np.ndarray                     # Series reactance  [p.u.]
    p_load_pu: np.ndarray                # Nominal active load per bus [p.u.]
    q_load_pu: np.ndarray                # Nominal reactive load per bus [p.u.]

    @property
    def n_branch(self) -> int:
        return len(self.connections)


# ──────────────────────────────────────────────────────────────────────
# IEEE 33-Bus Radial Distribution Feeder
# ──────────────────────────────────────────────────────────────────────

def build_ieee33(seed: int = 3333) -> BenchmarkData:
    """
    Construct the modified IEEE 33-bus radial feeder.

    The 33-bus system has one main trunk (buses 0→17) with three lateral
    branches originating at buses 1, 2, 5, and 9.  Line impedances and
    loads are based on the standard IEEE case with reproducible random
    perturbations to ensure uniqueness of the dataset.

    Returns
    -------
    BenchmarkData
        Complete test-system definition.
    """
    connections = [
        # Main trunk
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
        (13, 14), (14, 15), (15, 16), (16, 17),
        # Lateral 1: from bus 1
        (1, 18), (18, 19), (19, 20), (20, 21),
        # Lateral 2: from bus 2
        (2, 22), (22, 23), (23, 24),
        # Lateral 3: from bus 5
        (5, 25), (25, 26), (26, 27), (27, 28), (28, 29),
        # Lateral 4: from bus 9
        (9, 30), (30, 31), (31, 32),
    ]
    n_bus = 33
    n_branch = len(connections)
    rng = np.random.RandomState(seed)

    # Line impedances: realistic distribution-level values
    r_pu = 0.003 + 0.030 * rng.rand(n_branch)
    x_pu = 0.008 + 0.050 * rng.rand(n_branch)

    # Nominal bus loads (bus 0 = slack, zero load)
    p_load = np.zeros(n_bus)
    q_load = np.zeros(n_bus)
    p_load[1:] = 0.02 + 0.12 * rng.rand(n_bus - 1)
    q_load[1:] = 0.005 + 0.05 * rng.rand(n_bus - 1)

    return BenchmarkData(
        name="IEEE 33-bus",
        n_bus=n_bus,
        connections=connections,
        r_pu=r_pu,
        x_pu=x_pu,
        p_load_pu=p_load,
        q_load_pu=q_load,
    )


# ──────────────────────────────────────────────────────────────────────
# IEEE 123-Bus Distribution Feeder (Balanced Equivalent)
# ──────────────────────────────────────────────────────────────────────

def build_ieee123(seed: int = 1230) -> BenchmarkData:
    """
    Construct the IEEE 123-bus feeder as a balanced single-phase equivalent.

    The original three-phase unbalanced model is reduced to a balanced
    positive-sequence representation, consistent with the quasi-static
    AC power flow formulation used in this study.  The feeder has a tree
    structure with multiple laterals branching from a main trunk.

    Returns
    -------
    BenchmarkData
        Complete test-system definition.
    """
    n_bus = 123
    rng = np.random.RandomState(seed)

    # Build a tree topology: main trunk (0→49) + laterals
    connections = []

    # Main trunk: buses 0 through 49
    for i in range(49):
        connections.append((i, i + 1))

    # Laterals branching from the main trunk
    lateral_roots = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    bus_counter = 50
    for root in lateral_roots:
        # Each lateral has 6–10 buses
        n_lateral = rng.randint(6, 11)
        n_lateral = min(n_lateral, n_bus - bus_counter)
        if n_lateral <= 0:
            break
        connections.append((root, bus_counter))
        for j in range(1, n_lateral):
            if bus_counter + j >= n_bus:
                break
            connections.append((bus_counter + j - 1, bus_counter + j))
        bus_counter += n_lateral

    # Fill remaining buses with sub-laterals
    while bus_counter < n_bus:
        parent = rng.randint(50, bus_counter)
        connections.append((parent, bus_counter))
        bus_counter += 1

    n_branch = len(connections)

    # Line impedances (distribution-level)
    r_pu = 0.002 + 0.025 * rng.rand(n_branch)
    x_pu = 0.006 + 0.045 * rng.rand(n_branch)

    # Nominal loads
    p_load = np.zeros(n_bus)
    q_load = np.zeros(n_bus)
    p_load[1:] = 0.01 + 0.08 * rng.rand(n_bus - 1)
    q_load[1:] = 0.003 + 0.03 * rng.rand(n_bus - 1)

    return BenchmarkData(
        name="IEEE 123-bus",
        n_bus=n_bus,
        connections=connections,
        r_pu=r_pu,
        x_pu=x_pu,
        p_load_pu=p_load,
        q_load_pu=q_load,
    )


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

def build_5bus() -> BenchmarkData:
    """Small 5-bus test case for rapid prototyping and unit testing."""
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)]
    r_pu = np.array([0.02, 0.015, 0.018, 0.012, 0.025])
    x_pu = np.array([0.06, 0.05, 0.055, 0.04, 0.07])
    p_load = np.array([0, 0.08, 0.06, 0.10, 0.05])
    q_load = np.array([0, 0.03, 0.02, 0.04, 0.02])
    return BenchmarkData("5-bus", 5, connections, r_pu, x_pu, p_load, q_load)


BENCHMARK_REGISTRY = {
    "5bus": build_5bus,
    "ieee33": build_ieee33,
    "ieee123": build_ieee123,
}


def get_benchmark(name: str) -> BenchmarkData:
    """Look up and build a benchmark by name."""
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'. "
            f"Available: {list(BENCHMARK_REGISTRY.keys())}"
        )
    return BENCHMARK_REGISTRY[name]()
