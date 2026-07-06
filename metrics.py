"""
metrics.py — Evaluation Metrics
================================
Section III-I of the manuscript.

Metrics are organised across four orthogonal dimensions:

  (a) Predictive accuracy
      - RMSE and MAE_max for |V| and δ

  (b) Physical consistency
      - Mean power-balance residual (active + reactive)
      - Percentage of snapshots with voltage bound violations
      - Slack bus reference deviation

  (c) Computational performance
      - Single-snapshot inference time [ms]
      - Full-scenario inference time [ms]
      - Speed-up ratio vs AC power flow solver

  (d) Robustness / generalisation
      - ΔRMSE under OOD conditions
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import time
import numpy as np


@dataclass
class PredictiveMetrics:
    """Predictive accuracy metrics."""
    vm_rmse: float       # Voltage magnitude RMSE [p.u.]
    va_rmse: float       # Voltage angle RMSE [rad]
    vm_mae_max: float    # Maximum absolute error |V|
    va_mae_max: float    # Maximum absolute error δ


@dataclass
class PhysicsMetrics:
    """Physical consistency metrics."""
    mean_p_residual: float    # Mean |ΔP| across all buses [p.u.]
    mean_q_residual: float    # Mean |ΔQ| across all buses [p.u.]
    pct_v_violations: float   # % of snapshots with ≥1 voltage violation
    slack_deviation: float    # |V₀ − 1.0| + |δ₀ − 0|


@dataclass
class ComputationalMetrics:
    """Computational performance metrics."""
    single_inference_ms: float    # Mean per-snapshot inference [ms]
    scenario_inference_ms: float  # Full 96-step scenario inference [ms]
    speedup_vs_pf: float          # Speed-up ratio vs Newton–Raphson


@dataclass
class FullEvaluation:
    """Aggregated evaluation across all metric dimensions."""
    accuracy: PredictiveMetrics
    physics: PhysicsMetrics
    compute: ComputationalMetrics
    model_name: str
    n_parameters: int


# ──────────────────────────────────────────────────────────────────────
# Metric Computation Functions
# ──────────────────────────────────────────────────────────────────────

def compute_predictive_metrics(
    vm_pred: np.ndarray,
    va_pred: np.ndarray,
    vm_true: np.ndarray,
    va_true: np.ndarray,
) -> PredictiveMetrics:
    """
    Compute RMSE and maximum absolute error for |V| and δ.

    Parameters
    ----------
    vm_pred, va_pred : ndarray, shape (n_samples, n_bus) or (n_bus,)
    vm_true, va_true : ndarray, same shape
    """
    return PredictiveMetrics(
        vm_rmse=float(np.sqrt(np.mean((vm_pred - vm_true) ** 2))),
        va_rmse=float(np.sqrt(np.mean((va_pred - va_true) ** 2))),
        vm_mae_max=float(np.max(np.abs(vm_pred - vm_true))),
        va_mae_max=float(np.max(np.abs(va_pred - va_true))),
    )


def compute_physics_metrics(
    vm_pred: np.ndarray,
    va_pred: np.ndarray,
    p_net: np.ndarray,
    q_net: np.ndarray,
    Y_bus: np.ndarray,
    v_min: float,
    v_max: float,
) -> PhysicsMetrics:
    """
    Evaluate physical consistency of predictions.

    Operates on a single snapshot (1-D arrays) or batch (2-D).
    """
    if vm_pred.ndim == 1:
        vm_pred = vm_pred[np.newaxis, :]
        va_pred = va_pred[np.newaxis, :]
        p_net = p_net[np.newaxis, :]
        q_net = q_net[np.newaxis, :]

    n_samples, n_bus = vm_pred.shape
    G = Y_bus.real
    B = Y_bus.imag

    total_p_res = 0.0
    total_q_res = 0.0
    violation_count = 0

    for s in range(n_samples):
        p_calc = np.zeros(n_bus)
        q_calc = np.zeros(n_bus)
        for i in range(n_bus):
            for j in range(n_bus):
                delta = va_pred[s, i] - va_pred[s, j]
                p_calc[i] += vm_pred[s, i] * vm_pred[s, j] * (
                    G[i, j] * np.cos(delta) + B[i, j] * np.sin(delta)
                )
                q_calc[i] += vm_pred[s, i] * vm_pred[s, j] * (
                    G[i, j] * np.sin(delta) - B[i, j] * np.cos(delta)
                )
        total_p_res += np.mean(np.abs(p_net[s] - p_calc))
        total_q_res += np.mean(np.abs(q_net[s] - q_calc))

        # Voltage violations (non-slack buses)
        if np.any((vm_pred[s, 1:] < v_min) | (vm_pred[s, 1:] > v_max)):
            violation_count += 1

    slack_dev = float(
        np.mean(np.abs(vm_pred[:, 0] - 1.0) + np.abs(va_pred[:, 0]))
    )

    return PhysicsMetrics(
        mean_p_residual=total_p_res / n_samples,
        mean_q_residual=total_q_res / n_samples,
        pct_v_violations=100.0 * violation_count / n_samples,
        slack_deviation=slack_dev,
    )


def compute_computational_metrics(
    model,
    grid,
    scenario: dict,
    edge_features: np.ndarray,
    T: int,
) -> ComputationalMetrics:
    """
    Measure inference time and compare against Newton–Raphson.

    Runs inference for all T timesteps and measures wall-clock time.
    """
    from .graph import build_node_features

    adjacency = (np.abs(grid.Y_bus) > 0).astype(float)

    # Model inference timing
    t0 = time.perf_counter()
    for t in range(T):
        nf = build_node_features(
            scenario, t, grid.pv_buses, grid.bess_buses,
            grid.p_load_nom, grid.q_load_nom, grid.n_bus,
        )
        try:
            model.forward(nf, adjacency, grid.edge_index, edge_features)
        except TypeError:
            model.forward(nf)
    model_time = (time.perf_counter() - t0) * 1000  # ms

    # Power flow solver timing
    t0 = time.perf_counter()
    for t in range(T):
        p_net = scenario["p_pv"][t] + scenario["p_bess"][t] - scenario["p_load"][t]
        q_net = scenario["q_pv"][t] - scenario["q_load"][t]
        p_net[0] = -np.sum(p_net[1:])
        q_net[0] = -np.sum(q_net[1:])
        grid.solve_power_flow(p_net, q_net)
    pf_time = (time.perf_counter() - t0) * 1000

    return ComputationalMetrics(
        single_inference_ms=model_time / T,
        scenario_inference_ms=model_time,
        speedup_vs_pf=pf_time / max(model_time, 1e-6),
    )


# ──────────────────────────────────────────────────────────────────────
# Full Evaluation Pipeline
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    grid,
    test_scenarios: list,
    cfg,
    model_name: str = "PIGNN",
    max_eval_scenarios: int = 50,
) -> FullEvaluation:
    """
    Run the complete evaluation pipeline on the test set.

    Parameters
    ----------
    model : any model with .forward() and .count_parameters()
    grid : DistributionGrid
    test_scenarios : list of scenario dicts
    cfg : ExperimentConfig
    model_name : str
    max_eval_scenarios : int
        Limit for computational tractability.

    Returns
    -------
    FullEvaluation
    """
    from graph import build_node_features, build_edge_features

    edge_features = build_edge_features(
        grid.edge_index, grid.r_pu, grid.x_pu, grid.connections,
    )
    adjacency = (np.abs(grid.Y_bus) > 0).astype(float)
    T = cfg.scenario.timesteps_per_day

    all_vm_pred, all_va_pred = [], []
    all_vm_true, all_va_true = [], []
    all_p_net, all_q_net = [], []

    n_eval = min(max_eval_scenarios, len(test_scenarios))
    for sc in test_scenarios[:n_eval]:
        for t in range(0, T, 4):  # Sample every 4th timestep
            nf = build_node_features(
                sc, t, grid.pv_buses, grid.bess_buses,
                grid.p_load_nom, grid.q_load_nom, grid.n_bus,
            )
            try:
                pred = model.forward(nf, adjacency, grid.edge_index, edge_features)
            except TypeError:
                pred = model.forward(nf)

            p_net = sc["p_pv"][t] + sc["p_bess"][t] - sc["p_load"][t]
            q_net = sc["q_pv"][t] - sc["q_load"][t]
            p_net[0] = -np.sum(p_net[1:])
            q_net[0] = -np.sum(q_net[1:])
            pf = grid.solve_power_flow(p_net, q_net)

            all_vm_pred.append(pred["vm"])
            all_va_pred.append(pred["va"])
            all_vm_true.append(pf.vm)
            all_va_true.append(pf.va)
            all_p_net.append(p_net)
            all_q_net.append(q_net)

    vm_pred = np.array(all_vm_pred)
    va_pred = np.array(all_va_pred)
    vm_true = np.array(all_vm_true)
    va_true = np.array(all_va_true)

    import os
    os.makedirs("results_raw", exist_ok=True)

    np.save("results_raw/vm_pred.npy", vm_pred)
    np.save("results_raw/va_pred.npy", va_pred)
    np.save("results_raw/vm_true.npy", vm_true)
    np.save("results_raw/va_true.npy", va_true)

    print("Saved raw prediction arrays to results_raw/")
    p_net_arr = np.array(all_p_net)
    q_net_arr = np.array(all_q_net)

    accuracy = compute_predictive_metrics(vm_pred, va_pred, vm_true, va_true)
    physics = compute_physics_metrics(
        vm_pred, va_pred, p_net_arr, q_net_arr,
        grid.Y_bus, cfg.grid.v_min_pu, cfg.grid.v_max_pu,
    )
    compute = compute_computational_metrics(
        model, grid, test_scenarios[0], edge_features, T,
    )

    return FullEvaluation(
        accuracy=accuracy,
        physics=physics,
        compute=compute,
        model_name=model_name,
        n_parameters=model.count_parameters(),
    )


def print_evaluation(ev: FullEvaluation) -> None:
    """Pretty-print evaluation results."""
    print(f"\n  {'─' * 55}")
    print(f"  {ev.model_name} ({ev.n_parameters:,} params)")
    print(f"  {'─' * 55}")
    print(f"  Predictive Accuracy:")
    print(f"    |V| RMSE       : {ev.accuracy.vm_rmse:.6f} p.u.")
    print(f"    δ   RMSE       : {ev.accuracy.va_rmse:.6f} rad")
    print(f"    |V| MAE_max    : {ev.accuracy.vm_mae_max:.6f} p.u.")
    print(f"    δ   MAE_max    : {ev.accuracy.va_mae_max:.6f} rad")
    print(f"  Physical Consistency:")
    print(f"    Mean |ΔP|      : {ev.physics.mean_p_residual:.6f} p.u.")
    print(f"    Mean |ΔQ|      : {ev.physics.mean_q_residual:.6f} p.u.")
    print(f"    V violations   : {ev.physics.pct_v_violations:.1f}%")
    print(f"    Slack deviation: {ev.physics.slack_deviation:.6f}")
    print(f"  Computational:")
    print(f"    Inference/snap : {ev.compute.single_inference_ms:.2f} ms")
    print(f"    Scenario (96t) : {ev.compute.scenario_inference_ms:.1f} ms")
    print(f"    Speed-up vs PF : {ev.compute.speedup_vs_pf:.1f}×")
