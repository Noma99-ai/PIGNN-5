"""
ablation.py — Ablation Study Design
====================================
Section III-J of the manuscript.

Five ablation variants isolate the marginal contribution of each
architectural component of the PIGNN:

  (A1) No graph topology       — GNN replaced by flat MLP
  (A2) No physics loss          — Only supervised data loss
  (A3) No constrained outputs   — Linear output heads
  (A4) No edge electrical feat. — Binary adjacency, uniform weights
  (A5) Reduced MP depth         — K=1 instead of K=3
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from config import ExperimentConfig
from models.pignn import PIGNN
from models.baselines import StandardNN
from graph import NODE_FEATURE_DIM
from training import train_model
from metrics import evaluate_model, FullEvaluation, print_evaluation


@dataclass
class AblationResult:
    """Result of one ablation variant."""
    variant_id: str
    description: str
    scientific_purpose: str
    evaluation: FullEvaluation


def run_ablation_study(
    grid,
    train_scenarios: list,
    val_scenarios: list,
    test_scenarios: list,
    cfg: ExperimentConfig,
    seed: int = 42,
) -> List[AblationResult]:
    """
    Execute the complete ablation study (Section III-J).

    Returns a list of AblationResult, one per variant.
    """
    rng = np.random.RandomState(seed)
    results = []

    # ── (A1) No graph topology ────────────────────────────────────────
    print("\n  Ablation A1: No graph topology (flat MLP)")
    model_a1 = StandardNN(grid.n_bus, NODE_FEATURE_DIM, rng)
    train_model(model_a1, grid, train_scenarios, val_scenarios, cfg,
                use_physics_loss=False, label="A1-NoGraph")
    ev_a1 = evaluate_model(model_a1, grid, test_scenarios, cfg, "A1-NoGraph")
    results.append(AblationResult(
        variant_id="A1",
        description="GNN encoder replaced by flat MLP of equivalent capacity",
        scientific_purpose="Measures the benefit of exploiting the physical "
                          "grid topology through message passing",
        evaluation=ev_a1,
    ))

    # ── (A2) No physics loss ──────────────────────────────────────────
    print("\n  Ablation A2: No physics loss (data-only)")
    model_a2 = PIGNN(grid.n_bus, cfg, np.random.RandomState(seed))
    train_model(model_a2, grid, train_scenarios, val_scenarios, cfg,
                use_physics_loss=False, label="A2-NoPhysics")
    ev_a2 = evaluate_model(model_a2, grid, test_scenarios, cfg, "A2-NoPhysics")
    results.append(AblationResult(
        variant_id="A2",
        description="All physics penalty terms set to zero, retaining only "
                    "supervised data loss",
        scientific_purpose="Quantifies the regularisation value of embedding "
                          "AC power flow equations in the training objective",
        evaluation=ev_a2,
    ))

    # ── (A3) No constrained output activations ────────────────────────
    print("\n  Ablation A3: No constrained outputs (linear heads)")
    cfg_a3 = deepcopy(cfg)
    cfg_a3.pinn.vm_range = (-10.0, 10.0)  # Effectively unconstrained
    cfg_a3.pinn.va_range = (-10.0, 10.0)
    model_a3 = PIGNN(grid.n_bus, cfg_a3, np.random.RandomState(seed))
    train_model(model_a3, grid, train_scenarios, val_scenarios, cfg,
                use_physics_loss=True, label="A3-NoConstrained")
    ev_a3 = evaluate_model(model_a3, grid, test_scenarios, cfg, "A3-NoConstrained")
    results.append(AblationResult(
        variant_id="A3",
        description="Sigmoid and tanh output scalings removed; unconstrained "
                    "linear output heads",
        scientific_purpose="Isolates the effect of hard physical bounds on "
                          "prediction feasibility",
        evaluation=ev_a3,
    ))

    # ── (A4) No edge electrical features ──────────────────────────────
    print("\n  Ablation A4: No edge electrical features (binary adjacency)")
    cfg_a4 = deepcopy(cfg)
    cfg_a4.gnn.use_edge_gating = False
    model_a4 = PIGNN(grid.n_bus, cfg_a4, np.random.RandomState(seed))
    # Train with None edge features
    train_model(model_a4, grid, train_scenarios, val_scenarios, cfg,
                use_physics_loss=True, label="A4-NoEdgeFeat")
    ev_a4 = evaluate_model(model_a4, grid, test_scenarios, cfg, "A4-NoEdgeFeat")
    results.append(AblationResult(
        variant_id="A4",
        description="Edge features (r, x, |y|) removed; message passing uses "
                    "binary adjacency with uniform weights",
        scientific_purpose="Tests whether encoding line impedance parameters "
                          "improves accuracy beyond topological connectivity",
        evaluation=ev_a4,
    ))

    # ── (A5) Reduced message-passing depth (K=1) ─────────────────────
    print("\n  Ablation A5: Reduced message-passing (K=1)")
    cfg_a5 = deepcopy(cfg)
    cfg_a5.gnn.n_message_passing = 1
    model_a5 = PIGNN(grid.n_bus, cfg_a5, np.random.RandomState(seed))
    train_model(model_a5, grid, train_scenarios, val_scenarios, cfg,
                use_physics_loss=True, label="A5-K1")
    ev_a5 = evaluate_model(model_a5, grid, test_scenarios, cfg, "A5-K=1")
    results.append(AblationResult(
        variant_id="A5",
        description="Message-passing rounds reduced from K=3 to K=1 (immediate "
                    "neighbours only)",
        scientific_purpose="Evaluates whether multi-hop information propagation "
                          "is necessary for accurate voltage-state estimation",
        evaluation=ev_a5,
    ))

    return results


def print_ablation_summary(results: List[AblationResult]) -> None:
    """Print a formatted ablation study summary table."""
    print(f"\n  {'=' * 75}")
    print(f"  ABLATION STUDY RESULTS (Section III-J)")
    print(f"  {'=' * 75}")
    print(f"  {'Variant':<18} {'|V| RMSE':>10} {'δ RMSE':>10} "
          f"{'V viol%':>9} {'PB resid':>10} {'Params':>8}")
    print(f"  {'─' * 75}")
    for r in results:
        ev = r.evaluation
        print(
            f"  {r.variant_id + ' ' + ev.model_name:<18} "
            f"{ev.accuracy.vm_rmse:>10.6f} "
            f"{ev.accuracy.va_rmse:>10.6f} "
            f"{ev.physics.pct_v_violations:>8.1f}% "
            f"{ev.physics.mean_p_residual:>10.6f} "
            f"{ev.n_parameters:>8,}"
        )
    print(f"  {'─' * 75}")
