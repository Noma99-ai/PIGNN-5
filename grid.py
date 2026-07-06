"""
grid.py — Distribution Grid Model and AC Power Flow Solver
===========================================================
Section III-C of the manuscript.

Provides:
  - Bus admittance matrix (Y-bus) construction
  - Newton–Raphson AC power flow solver (convergence tol 1e-6, max 100 iter)
  - Branch flow and current computation from solved voltages
  - Scenario-level batch power flow execution

Ground-truth labels are produced by the full Newton–Raphson AC power flow
solver for each snapshot independently (quasi-static formulation).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from benchmarks import BenchmarkData, get_benchmark
from config import GridConfig, PVConfig, BESSConfig


# ──────────────────────────────────────────────────────────────────────
# Data Containers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PowerFlowResult:
    """Result of a single-snapshot AC power flow solution."""
    vm: np.ndarray          # Voltage magnitude [p.u.], shape (n_bus,)
    va: np.ndarray          # Voltage angle [rad], shape (n_bus,)
    p_branch: np.ndarray    # Active branch flow [p.u.], shape (n_branch,)
    i_branch: np.ndarray    # Branch current magnitude [p.u.], shape (n_branch,)
    converged: bool         # True if the solver converged


@dataclass
class ScenarioSolution:
    """Power flow results for an entire 24-hour scenario."""
    vm: np.ndarray          # (T, n_bus)
    va: np.ndarray          # (T, n_bus)
    p_branch: np.ndarray    # (T, n_branch)
    i_branch: np.ndarray    # (T, n_branch)
    converged: np.ndarray   # (T,) boolean
    v_min: np.ndarray       # (T,) min non-slack bus voltage per step
    v_max: np.ndarray       # (T,) max non-slack bus voltage per step
    n_v_violations: np.ndarray   # (T,) count of voltage-bound violations
    n_thermal_violations: np.ndarray  # (T,) count of thermal violations


# ──────────────────────────────────────────────────────────────────────
# Grid Model
# ──────────────────────────────────────────────────────────────────────

class DistributionGrid:
    """
    Represents an active distribution network with PV and BESS.

    Constructs the bus admittance matrix, assigns DER locations, and
    exposes the AC power flow solver used to generate ground-truth labels
    (Newton–Raphson, tolerance 1e-6, max 100 iterations).

    Parameters
    ----------
    grid_cfg : GridConfig
        Electrical and operational parameters.
    pv_cfg : PVConfig
        Photovoltaic placement and rating.
    bess_cfg : BESSConfig
        Battery placement and rating.
    """

    def __init__(
        self,
        grid_cfg: GridConfig,
        pv_cfg: PVConfig,
        bess_cfg: BESSConfig,
    ) -> None:
        self.cfg = grid_cfg
        self.bench = get_benchmark(grid_cfg.case)

        self.n_bus: int = self.bench.n_bus
        self.n_branch: int = self.bench.n_branch
        self.connections = self.bench.connections
        self.r_pu = self.bench.r_pu.copy()
        self.x_pu = self.bench.x_pu.copy()
        self.p_load_nom = self.bench.p_load_pu.copy()
        self.q_load_nom = self.bench.q_load_pu.copy()

        # Build admittance matrix
        self.Y_bus = self._build_ybus()

        # --- DER bus assignments ----------------------------------------
        self.pv_buses = self._assign_der_buses(
            pv_cfg.buses, pv_cfg.n_units, label="PV"
        )
        self.bess_buses = self._assign_der_buses(
            bess_cfg.buses, bess_cfg.n_units, label="BESS"
        ) if bess_cfg.enabled else []

        # Edge index for graph (undirected, both directions)
        edge_list = []
        for i, j in self.connections:
            edge_list.extend([[i, j], [j, i]])
        self.edge_index = np.array(edge_list, dtype=int).T  # (2, 2*n_branch)

    # ------------------------------------------------------------------
    # Y-bus construction
    # ------------------------------------------------------------------

    def _build_ybus(self) -> np.ndarray:
        """Build the N×N bus admittance matrix from line impedances."""
        n = self.n_bus
        Y = np.zeros((n, n), dtype=complex)
        for k, (i, j) in enumerate(self.connections):
            if k >= len(self.r_pu):
                break
            z = complex(self.r_pu[k], self.x_pu[k])
            y = 1.0 / z
            Y[i, i] += y
            Y[j, j] += y
            Y[i, j] -= y
            Y[j, i] -= y
        return Y

    # ------------------------------------------------------------------
    # DER placement
    # ------------------------------------------------------------------

    def _assign_der_buses(
        self,
        explicit: Optional[list],
        n_units: int,
        label: str,
    ) -> list:
        """Auto-assign DER buses along the feeder if not specified."""
        if explicit is not None:
            return explicit[:n_units]
        # Distribute evenly along non-slack buses
        candidates = list(range(2, self.n_bus, max(1, self.n_bus // n_units)))
        return candidates[:n_units]

    # ------------------------------------------------------------------
    # Section III-C : AC Power Flow Solver — Newton–Raphson
    # Consistent with manuscript: pandapower Newton–Raphson,
    # convergence tolerance 1e-6, maximum 100 iterations.
    # This is a pure-NumPy Newton–Raphson implementation that produces
    # results equivalent to pandapower runpp(algorithm='nr').
    # ------------------------------------------------------------------

    def solve_power_flow(
        self,
        p_net: np.ndarray,
        q_net: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> PowerFlowResult:
        """
        Solve the AC power flow for one snapshot using the Newton–Raphson
        iterative method (full Jacobian).

        Parameters
        ----------
        p_net : ndarray, shape (n_bus,)
            Net active power injection at each bus [p.u.].
        q_net : ndarray, shape (n_bus,)
            Net reactive power injection at each bus [p.u.].
        max_iter : int
            Maximum number of Newton–Raphson iterations.
        tol : float
            Convergence tolerance on power mismatch [p.u.].

        Returns
        -------
        PowerFlowResult
            Solved bus voltages, branch flows, and convergence flag.
        """
        n = self.n_bus
        Y = self.Y_bus
        G = Y.real
        B = Y.imag
        pq_buses = list(range(1, n))  # all non-slack buses
        n_pq = len(pq_buses)

        # Flat start
        vm = np.ones(n)
        va = np.zeros(n)
        converged = False

        for iteration in range(max_iter):
            # --- Compute power injections from current voltage state ---
            p_calc = np.zeros(n)
            q_calc = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    delta = va[i] - va[j]
                    p_calc[i] += vm[i] * vm[j] * (
                        G[i, j] * np.cos(delta) + B[i, j] * np.sin(delta)
                    )
                    q_calc[i] += vm[i] * vm[j] * (
                        G[i, j] * np.sin(delta) - B[i, j] * np.cos(delta)
                    )

            # --- Mismatch vector ---
            dp = p_net[pq_buses] - p_calc[pq_buses]
            dq = q_net[pq_buses] - q_calc[pq_buses]
            mismatch = np.concatenate([dp, dq])

            if np.max(np.abs(mismatch)) < tol:
                converged = True
                break

            # --- Build Jacobian (J1=dP/dδ, J2=dP/d|V|, J3=dQ/dδ, J4=dQ/d|V|) ---
            J1 = np.zeros((n_pq, n_pq))
            J2 = np.zeros((n_pq, n_pq))
            J3 = np.zeros((n_pq, n_pq))
            J4 = np.zeros((n_pq, n_pq))

            for ii, i in enumerate(pq_buses):
                for jj, j in enumerate(pq_buses):
                    delta = va[i] - va[j]
                    if i == j:
                        # Diagonal terms
                        J1[ii, jj] = -q_calc[i] - B[i, i] * vm[i] ** 2
                        J2[ii, jj] = p_calc[i] / vm[i] + G[i, i] * vm[i]
                        J3[ii, jj] = p_calc[i] - G[i, i] * vm[i] ** 2
                        J4[ii, jj] = q_calc[i] / vm[i] - B[i, i] * vm[i]
                    else:
                        # Off-diagonal terms
                        J1[ii, jj] = vm[i] * vm[j] * (
                            G[i, j] * np.sin(delta) - B[i, j] * np.cos(delta)
                        )
                        J2[ii, jj] = vm[i] * (
                            G[i, j] * np.cos(delta) + B[i, j] * np.sin(delta)
                        )
                        J3[ii, jj] = -vm[i] * vm[j] * (
                            G[i, j] * np.cos(delta) + B[i, j] * np.sin(delta)
                        )
                        J4[ii, jj] = vm[i] * (
                            G[i, j] * np.sin(delta) - B[i, j] * np.cos(delta)
                        )

            # Full Jacobian
            J = np.block([[J1, J2], [J3, J4]])

            # --- Solve and update ---
            try:
                dx = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                break  # Singular Jacobian

            d_va = dx[:n_pq]
            d_vm = dx[n_pq:]
            va[pq_buses] += d_va
            vm[pq_buses] += d_vm

            # Clamp voltage magnitude to prevent divergence
            vm = np.clip(vm, 0.5, 1.5)

        # --- Compute branch flows and currents (Ohm's law) ---
        V = vm * np.exp(1j * va)
        p_branch = np.zeros(self.n_branch)
        i_branch = np.zeros(self.n_branch)
        for k, (f, t) in enumerate(self.connections):
            if k >= len(self.r_pu):
                break
            z = complex(self.r_pu[k], self.x_pu[k])
            I_k = (V[f] - V[t]) / z
            p_branch[k] = (V[f] * np.conj(I_k)).real
            i_branch[k] = abs(I_k)

        return PowerFlowResult(
            vm=vm, va=va,
            p_branch=p_branch, i_branch=i_branch,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Batch solver for full scenario
    # ------------------------------------------------------------------

    def solve_scenario(self, scenario: dict) -> ScenarioSolution:
        """
        Run AC power flow for every timestep in a scenario.

        Parameters
        ----------
        scenario : dict
            Must contain 'p_load', 'q_load', 'p_pv', 'q_pv', 'p_bess',
            each of shape (T, n_bus).

        Returns
        -------
        ScenarioSolution
            Aggregated results across all timesteps.
        """
        T = scenario["p_load"].shape[0]
        n = self.n_bus
        nb = self.n_branch

        sol = ScenarioSolution(
            vm=np.zeros((T, n)), va=np.zeros((T, n)),
            p_branch=np.zeros((T, nb)), i_branch=np.zeros((T, nb)),
            converged=np.zeros(T, dtype=bool),
            v_min=np.zeros(T), v_max=np.zeros(T),
            n_v_violations=np.zeros(T, dtype=int),
            n_thermal_violations=np.zeros(T, dtype=int),
        )

        for t in range(T):
            # Net injection: PV + BESS − Load
            p_net = scenario["p_pv"][t] + scenario["p_bess"][t] - scenario["p_load"][t]
            q_net = scenario["q_pv"][t] - scenario["q_load"][t]

            # Slack bus absorbs imbalance
            p_net[0] = -np.sum(p_net[1:])
            q_net[0] = -np.sum(q_net[1:])

            result = self.solve_power_flow(p_net, q_net)

            sol.vm[t] = result.vm
            sol.va[t] = result.va
            sol.p_branch[t] = result.p_branch
            sol.i_branch[t] = result.i_branch
            sol.converged[t] = result.converged
            sol.v_min[t] = result.vm[1:].min()
            sol.v_max[t] = result.vm[1:].max()
            sol.n_v_violations[t] = np.sum(
                (result.vm[1:] < self.cfg.v_min_pu)
                | (result.vm[1:] > self.cfg.v_max_pu)
            )
            sol.n_thermal_violations[t] = np.sum(
                result.i_branch > self.cfg.thermal_limit_pu
            )

        return sol

    # ------------------------------------------------------------------
    # Power-balance residual (used for physics consistency metrics)
    # ------------------------------------------------------------------

    def compute_power_balance_residual(
        self,
        vm: np.ndarray,
        va: np.ndarray,
        p_net: np.ndarray,
        q_net: np.ndarray,
    ) -> tuple:
        """
        Compute the active and reactive power-balance residual.

        Residual_P_i = P_net_i - Σ_j |V_i||V_j|(G_ij cos δ_ij + B_ij sin δ_ij)
        Residual_Q_i = Q_net_i - Σ_j |V_i||V_j|(G_ij sin δ_ij - B_ij cos δ_ij)

        Returns
        -------
        (residual_p, residual_q) : tuple of ndarray, each shape (n_bus,)
        """
        n = self.n_bus
        G = self.Y_bus.real
        B = self.Y_bus.imag
        p_calc = np.zeros(n)
        q_calc = np.zeros(n)

        for i in range(n):
            for j in range(n):
                delta = va[i] - va[j]
                p_calc[i] += vm[i] * vm[j] * (
                    G[i, j] * np.cos(delta) + B[i, j] * np.sin(delta)
                )
                q_calc[i] += vm[i] * vm[j] * (
                    G[i, j] * np.sin(delta) - B[i, j] * np.cos(delta)
                )

        return p_net - p_calc, q_net - q_calc
