"""
robustness.py — Robustness and Generalisation Tests
====================================================
Section III-K of the manuscript.

Four stress tests applied to the trained PIGNN without retraining:

  (R1) Elevated PV penetration (120%, 150% of training max)
  (R2) Load uncertainty (30% above training max)
  (R3) Measurement noise (σ = 1%, 3%, 5% on all inputs)
  (R4) Topology perturbation (single-line N−1 contingency)

These tests challenge the model's extrapolation capability without
conflating Layer 5 evaluation with higher-level modules.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from copy import deepcopy
import numpy as np

from config import ExperimentConfig
from graph import build_node_features, build_edge_features
from metrics import compute_predictive_metrics, PredictiveMetrics


@dataclass
class RobustnessResult:
    """Result of one robustness test."""
    test_id: str
    description: str
    condition: str
    baseline_metrics: PredictiveMetrics   # In-distribution reference
    stress_metrics: PredictiveMetrics     # Under stress condition
    delta_vm_rmse: float                  # Degradation
    delta_va_rmse: float


# ──────────────────────────────────────────────────────────────────────
# Individual Stress Tests
# ──────────────────────────────────────────────────────────────────────

def _evaluate_on_scenarios(model, grid, scenarios, cfg, edge_features):
    """Helper: evaluate model on a list of scenarios, return vm/va arrays."""
    adjacency = (np.abs(grid.Y_bus) > 0).astype(float)
    T = cfg.scenario.timesteps_per_day
    all_vm_pred, all_va_pred, all_vm_true, all_va_true = [], [], [], []

    for sc in scenarios[:20]:  # Limit for speed
        for t in range(0, T, 8):
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

    return compute_predictive_metrics(
        np.array(all_vm_pred), np.array(all_va_pred),
        np.array(all_vm_true), np.array(all_va_true),
    )


def test_elevated_pv(
    model, grid, base_scenarios, cfg, edge_features,
    baseline_metrics: PredictiveMetrics,
    pv_multiplier: float,
) -> RobustnessResult:
    """R1: Evaluate under elevated PV penetration."""
    stressed = []
    for sc in base_scenarios[:20]:
        sc_new = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc.items()}
        sc_new["p_pv"] = sc["p_pv"] * pv_multiplier
        sc_new["q_pv"] = sc["q_pv"] * pv_multiplier
        stressed.append(sc_new)

    metrics = _evaluate_on_scenarios(model, grid, stressed, cfg, edge_features)
    return RobustnessResult(
        test_id="R1", description="Elevated PV penetration",
        condition=f"{pv_multiplier:.0%} of training max",
        baseline_metrics=baseline_metrics, stress_metrics=metrics,
        delta_vm_rmse=metrics.vm_rmse - baseline_metrics.vm_rmse,
        delta_va_rmse=metrics.va_rmse - baseline_metrics.va_rmse,
    )


def test_load_uncertainty(
    model, grid, base_scenarios, cfg, edge_features,
    baseline_metrics: PredictiveMetrics,
    load_multiplier: float,
) -> RobustnessResult:
    """R2: Evaluate under increased load."""
    stressed = []
    for sc in base_scenarios[:20]:
        sc_new = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc.items()}
        sc_new["p_load"] = sc["p_load"] * load_multiplier
        sc_new["q_load"] = sc["q_load"] * load_multiplier
        stressed.append(sc_new)

    metrics = _evaluate_on_scenarios(model, grid, stressed, cfg, edge_features)
    return RobustnessResult(
        test_id="R2", description="Load uncertainty",
        condition=f"Loads × {load_multiplier:.1f}",
        baseline_metrics=baseline_metrics, stress_metrics=metrics,
        delta_vm_rmse=metrics.vm_rmse - baseline_metrics.vm_rmse,
        delta_va_rmse=metrics.va_rmse - baseline_metrics.va_rmse,
    )


def test_measurement_noise(
    model, grid, base_scenarios, cfg, edge_features,
    baseline_metrics: PredictiveMetrics,
    noise_sigma: float,
    seed: int = 99,
) -> RobustnessResult:
    """R3: Evaluate with Gaussian measurement noise on all inputs."""
    rng = np.random.RandomState(seed)
    stressed = []
    for sc in base_scenarios[:20]:
        sc_new = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc.items()}
        for key in ["p_pv", "q_pv", "p_load", "q_load"]:
            sc_new[key] = sc[key] + rng.normal(0, noise_sigma, sc[key].shape)
            if "pv" in key:
                sc_new[key] = np.clip(sc_new[key], 0, None)
        stressed.append(sc_new)

    metrics = _evaluate_on_scenarios(model, grid, stressed, cfg, edge_features)
    return RobustnessResult(
        test_id="R3", description="Measurement noise",
        condition=f"σ = {noise_sigma:.0%} of nominal",
        baseline_metrics=baseline_metrics, stress_metrics=metrics,
        delta_vm_rmse=metrics.vm_rmse - baseline_metrics.vm_rmse,
        delta_va_rmse=metrics.va_rmse - baseline_metrics.va_rmse,
    )


def test_n1_contingency(
    model, grid, base_scenarios, cfg, edge_features,
    baseline_metrics: PredictiveMetrics,
    line_idx: int = 3,
) -> RobustnessResult:
    """R4: Single-line outage (N−1 contingency)."""
    # Temporarily disable one line by setting its impedance very high
    orig_r = grid.r_pu[line_idx]
    orig_x = grid.x_pu[line_idx]
    grid.r_pu[line_idx] = 1e6
    grid.x_pu[line_idx] = 1e6
    grid.Y_bus = grid._build_ybus()

    # Recompute edge features with modified line
    ef_mod = build_edge_features(
        grid.edge_index, grid.r_pu, grid.x_pu, grid.connections,
    )

    metrics = _evaluate_on_scenarios(model, grid, base_scenarios, cfg, ef_mod)

    # Restore original line
    grid.r_pu[line_idx] = orig_r
    grid.x_pu[line_idx] = orig_x
    grid.Y_bus = grid._build_ybus()

    return RobustnessResult(
        test_id="R4", description="N−1 line outage",
        condition=f"Line {line_idx} removed",
        baseline_metrics=baseline_metrics, stress_metrics=metrics,
        delta_vm_rmse=metrics.vm_rmse - baseline_metrics.vm_rmse,
        delta_va_rmse=metrics.va_rmse - baseline_metrics.va_rmse,
    )


# ──────────────────────────────────────────────────────────────────────
# Full Robustness Suite
# ──────────────────────────────────────────────────────────────────────

def run_robustness_tests(
    model,
    grid,
    test_scenarios: list,
    cfg: ExperimentConfig,
    baseline_metrics: PredictiveMetrics,
) -> List[RobustnessResult]:
    """
    Run all four robustness tests (Section III-K).

    Parameters
    ----------
    model : trained PIGNN model
    grid : DistributionGrid
    test_scenarios : in-distribution test scenarios
    cfg : ExperimentConfig
    baseline_metrics : in-distribution accuracy (for ΔRMSE)

    Returns
    -------
    List of RobustnessResult
    """
    edge_features = build_edge_features(
        grid.edge_index, grid.r_pu, grid.x_pu, grid.connections,
    )
    results = []

    print("\n  Running robustness tests (Section III-K)...")

    # R1: Elevated PV
    for mult in cfg.robustness.elevated_pv_multipliers:
        print(f"    R1: PV × {mult}")
        results.append(test_elevated_pv(
            model, grid, test_scenarios, cfg, edge_features,
            baseline_metrics, mult,
        ))

    # R2: Load uncertainty
    print(f"    R2: Load × {cfg.robustness.load_stress_multiplier}")
    results.append(test_load_uncertainty(
        model, grid, test_scenarios, cfg, edge_features,
        baseline_metrics, cfg.robustness.load_stress_multiplier,
    ))

    # R3: Measurement noise
    for sigma in cfg.robustness.noise_sigma_levels:
        print(f"    R3: Noise σ = {sigma:.0%}")
        results.append(test_measurement_noise(
            model, grid, test_scenarios, cfg, edge_features,
            baseline_metrics, sigma,
        ))

    # R4: N−1 contingency
    line_idx = min(3, grid.n_branch - 1)
    print(f"    R4: N−1 contingency (line {line_idx})")
    results.append(test_n1_contingency(
        model, grid, test_scenarios, cfg, edge_features,
        baseline_metrics, line_idx,
    ))

    return results


def print_robustness_summary(results: List[RobustnessResult]) -> None:
    """Print formatted robustness results table."""
    print(f"\n  {'=' * 72}")
    print(f"  ROBUSTNESS TEST RESULTS (Section III-K)")
    print(f"  {'=' * 72}")
    print(f"  {'Test':<6} {'Condition':<28} {'|V| RMSE':>10} "
          f"{'Δ|V| RMSE':>11} {'Δδ RMSE':>10}")
    print(f"  {'─' * 72}")
    for r in results:
        print(
            f"  {r.test_id:<6} {r.condition:<28} "
            f"{r.stress_metrics.vm_rmse:>10.6f} "
            f"{r.delta_vm_rmse:>+10.6f} "
            f"{r.delta_va_rmse:>+10.6f}"
        )
    print(f"  {'─' * 72}")
