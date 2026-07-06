"""
graph.py — Graph Construction and Input Features
=================================================
Section III-D of the manuscript.

Defines the graph representation of the distribution network:
  - Node features (9-dimensional per bus per timestep)
  - Edge features (3-dimensional per branch)
  - Adjacency matrix and admittance-weighted adjacency
  - Normalisation utilities

The graph G = (V, E) is undirected: each physical line produces two
directed edges.  Self-loops are excluded.
"""

from __future__ import annotations
from typing import Dict
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Node Feature Vector (Section III-D, 9-dimensional)
# ──────────────────────────────────────────────────────────────────────

def build_node_features(
    scenario: Dict[str, np.ndarray],
    t: int,
    pv_buses: list,
    bess_buses: list,
    p_load_nom: np.ndarray,
    q_load_nom: np.ndarray,
    n_bus: int,
) -> np.ndarray:
    """
    Construct the node feature matrix for a single timestep.

    Feature vector per bus (9 dimensions):
      [0] P_net   : net active power injection (PV + BESS − Load) [p.u.]
      [1] Q_net   : net reactive power injection [p.u.]
      [2] P_pv    : PV active power at this bus [p.u.]
      [3] P_bess  : BESS active power at this bus [p.u.]
      [4] P_load0 : nominal active load [p.u.]
      [5] Q_load0 : nominal reactive load [p.u.]
      [6] is_pv   : binary indicator (1 if PV present)
      [7] is_bess : binary indicator (1 if BESS present)
      [8] is_slack: binary indicator (1 if slack bus)

    Parameters
    ----------
    scenario : dict
        Scenario data with 'p_pv', 'p_bess', 'p_load', 'q_pv', 'q_load'.
    t : int
        Timestep index.

    Returns
    -------
    ndarray, shape (n_bus, 9)
    """
    p_net = scenario["p_pv"][t] + scenario["p_bess"][t] - scenario["p_load"][t]
    q_net = scenario["q_pv"][t] - scenario["q_load"][t]

    is_pv = np.array([1.0 if i in pv_buses else 0.0 for i in range(n_bus)])
    is_bess = np.array([1.0 if i in bess_buses else 0.0 for i in range(n_bus)])
    is_slack = np.zeros(n_bus)
    is_slack[0] = 1.0

    features = np.stack([
        p_net,                       # [0] Net active injection
        q_net,                       # [1] Net reactive injection
        scenario["p_pv"][t],         # [2] PV active power
        scenario["p_bess"][t],       # [3] BESS active power
        p_load_nom,                  # [4] Nominal active load
        q_load_nom,                  # [5] Nominal reactive load
        is_pv,                       # [6] PV presence indicator
        is_bess,                     # [7] BESS presence indicator
        is_slack,                    # [8] Slack bus indicator
    ], axis=-1)  # (n_bus, 9)

    return features


NODE_FEATURE_DIM = 9
NODE_FEATURE_NAMES = [
    "P_net", "Q_net", "P_pv", "P_bess",
    "P_load_nom", "Q_load_nom",
    "is_pv", "is_bess", "is_slack",
]


# ──────────────────────────────────────────────────────────────────────
# Edge Feature Vector (Section III-D, 3-dimensional)
# ──────────────────────────────────────────────────────────────────────

def build_edge_features(
    edge_index: np.ndarray,
    r_pu: np.ndarray,
    x_pu: np.ndarray,
    connections: list,
) -> np.ndarray:
    """
    Construct the edge feature matrix.

    Feature vector per directed edge (3 dimensions):
      [0] r   : series resistance [p.u.]
      [1] x   : series reactance  [p.u.]
      [2] |y| : admittance magnitude 1/|z| [p.u.]

    The admittance magnitude |y| serves as the attention weight in the
    GNN message-passing aggregation, embedding Kirchhoff's circuit law
    directly into the graph convolution.

    Parameters
    ----------
    edge_index : ndarray, shape (2, n_edges)
        Directed edge list (includes both directions).

    Returns
    -------
    ndarray, shape (n_edges, 3)
    """
    n_edges = edge_index.shape[1]
    features = np.zeros((n_edges, 3))

    for k in range(n_edges):
        line_idx = k // 2  # Each line produces 2 directed edges
        if line_idx < len(r_pu):
            r = r_pu[line_idx]
            x = x_pu[line_idx]
            z_mag = abs(complex(r, x))
            features[k, 0] = r
            features[k, 1] = x
            features[k, 2] = 1.0 / z_mag if z_mag > 1e-10 else 0.0

    return features


EDGE_FEATURE_DIM = 3
EDGE_FEATURE_NAMES = ["r", "x", "|y|"]


# ──────────────────────────────────────────────────────────────────────
# Adjacency Matrices
# ──────────────────────────────────────────────────────────────────────

def build_adjacency(
    n_bus: int,
    connections: list,
) -> np.ndarray:
    """Build the binary adjacency matrix (undirected, no self-loops)."""
    A = np.zeros((n_bus, n_bus))
    for i, j in connections:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def build_admittance_weighted_adjacency(
    n_bus: int,
    connections: list,
    r_pu: np.ndarray,
    x_pu: np.ndarray,
) -> np.ndarray:
    """
    Build the admittance-weighted adjacency matrix.

    W[i,j] = 1/|Z_ij| for connected buses, 0 otherwise.
    Used as message-passing weights in the GNN encoder.
    """
    W = np.zeros((n_bus, n_bus))
    for k, (i, j) in enumerate(connections):
        if k < len(r_pu):
            z_mag = abs(complex(r_pu[k], x_pu[k]))
            weight = 1.0 / z_mag if z_mag > 1e-10 else 0.0
            W[i, j] = weight
            W[j, i] = weight
    return W


# ──────────────────────────────────────────────────────────────────────
# Feature Normalisation
# ──────────────────────────────────────────────────────────────────────

class FeatureNormaliser:
    """
    Per-feature standardisation (zero mean, unit variance).

    Statistics are computed on the training set only and applied
    identically to validation, test, and OOD sets.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._fitted = False

    def fit(self, features: np.ndarray) -> None:
        """Compute mean and std from a (N, D) feature matrix."""
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-8
        self._fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply standardisation."""
        if not self._fitted:
            raise RuntimeError("Normaliser must be fit() before transform().")
        return (features - self.mean) / self.std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)
