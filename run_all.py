"""
run_all.py — Full Experiment Orchestrator
==========================================
Executes the complete PIGNN simulation pipeline as defined in
Section III of the IEEE Transactions manuscript:

  Step 1:  Initialise test system (Section III-A)
  Step 2:  Generate scenario dataset (Section III-B)
  Step 3:  Train PIGNN and all baselines (Sections III-E–H)
  Step 4:  Evaluate all models (Section III-I)
  Step 5:  Ablation study (Section III-J)
  Step 6:  Robustness tests (Section III-K)
  Step 7:  Print results summary

Usage:
    python -m pignn_simulation.run_all
    python -m pignn_simulation.run_all --case ieee33 --epochs 300
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np

from config import ExperimentConfig
from grid import DistributionGrid
from scenarios import generate_dataset
from graph import build_edge_features, NODE_FEATURE_DIM
from models.pignn import PIGNN
from models.baselines import StandardNN, GNNOnly, PINNOnly
from training import train_model
from metrics import evaluate_model, print_evaluation, FullEvaluation
from ablation import run_ablation_study, print_ablation_summary
from robustness import run_robustness_tests, print_robustness_summary


def run_experiment(cfg: ExperimentConfig | None = None) -> dict:
    """
    Execute the full simulation pipeline.

    Parameters
    ----------
    cfg : ExperimentConfig or None
        If None, uses default configuration.

    Returns
    -------
    dict with all results for downstream analysis or plotting.
    """
    if cfg is None:
        cfg = ExperimentConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)
    print(cfg.summary())

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Initialise Test System (Section III-A)
    # ══════════════════════════════════════════════════════════════════
    print("\n  STEP 1: Initialising test system...")
    grid = DistributionGrid(cfg.grid, cfg.pv, cfg.bess)
    print(f"    Grid      : {grid.bench.name}")
    print(f"    Buses     : {grid.n_bus}")
    print(f"    Branches  : {grid.n_branch}")
    print(f"    PV buses  : {grid.pv_buses}")
    print(f"    BESS buses: {grid.bess_buses}")

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Generate Scenario Dataset (Section III-B)
    # ══════════════════════════════════════════════════════════════════
    print("\n  STEP 2: Generating scenario dataset...")
    t0 = time.time()
    dataset = generate_dataset(grid, cfg, seed=cfg.training.seeds[0])
    print(f"    Generated in {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Train PIGNN and Baselines (Sections III-E through III-H)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("  STEP 3: Training all models")
    print(f"{'=' * 65}")

    seed = cfg.training.seeds[0]
    rng = np.random.RandomState(seed)

    # 3a. PIGNN (proposed model)
    pignn = PIGNN(grid.n_bus, cfg, rng)
    print(f"\n{pignn.summary()}")
    hist_pignn = train_model(
        pignn, grid, dataset["train"], dataset["val"], cfg,
        use_physics_loss=True, label="PIGNN",
    )

    # 3b. Standard NN baseline
    model_nn = StandardNN(grid.n_bus, NODE_FEATURE_DIM, np.random.RandomState(seed))
    hist_nn = train_model(
        model_nn, grid, dataset["train"], dataset["val"], cfg,
        use_physics_loss=False, label="StandardNN",
    )

    # 3c. GNN-only baseline
    model_gnn = GNNOnly(grid.n_bus, cfg, np.random.RandomState(seed))
    hist_gnn = train_model(
        model_gnn, grid, dataset["train"], dataset["val"], cfg,
        use_physics_loss=False, label="GNN-Only",
    )

    # 3d. PINN-only baseline
    model_pinn = PINNOnly(grid.n_bus, NODE_FEATURE_DIM, cfg, np.random.RandomState(seed))
    hist_pinn = train_model(
        model_pinn, grid, dataset["train"], dataset["val"], cfg,
        use_physics_loss=True, label="PINN-Only",
    )

    # ══════════════════════════════════════════════════════════════════
    # Step 4: Evaluate All Models (Section III-I)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("  STEP 4: Evaluating all models on test set")
    print(f"{'=' * 65}")

    eval_pignn = evaluate_model(pignn, grid, dataset["test"], cfg, "PIGNN")
    eval_nn = evaluate_model(model_nn, grid, dataset["test"], cfg, "StandardNN")
    eval_gnn = evaluate_model(model_gnn, grid, dataset["test"], cfg, "GNN-Only")
    eval_pinn = evaluate_model(model_pinn, grid, dataset["test"], cfg, "PINN-Only")

    all_evals = [eval_pignn, eval_nn, eval_gnn, eval_pinn]
    for ev in all_evals:
        print_evaluation(ev)

    # Comparative summary table
    print(f"\n  {'=' * 72}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"  {'=' * 72}")
    print(f"  {'Model':<15} {'|V| RMSE':>10} {'δ RMSE':>10} "
          f"{'V viol%':>9} {'PB resid':>10} {'Speed-up':>10}")
    print(f"  {'─' * 72}")
    for ev in all_evals:
        print(
            f"  {ev.model_name:<15} "
            f"{ev.accuracy.vm_rmse:>10.6f} "
            f"{ev.accuracy.va_rmse:>10.6f} "
            f"{ev.physics.pct_v_violations:>8.1f}% "
            f"{ev.physics.mean_p_residual:>10.6f} "
            f"{ev.compute.speedup_vs_pf:>9.1f}×"
        )
    print(f"  {'─' * 72}")

    # ══════════════════════════════════════════════════════════════════
    # Step 5: Ablation Study (Section III-J)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("  STEP 5: Ablation study")
    print(f"{'=' * 65}")

    ablation_results = run_ablation_study(
        grid, dataset["train"], dataset["val"], dataset["test"], cfg, seed,
    )
    print_ablation_summary(ablation_results)

    # ══════════════════════════════════════════════════════════════════
    # Step 6: Robustness Tests (Section III-K)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("  STEP 6: Robustness and generalisation tests")
    print(f"{'=' * 65}")

    robustness_results = run_robustness_tests(
        pignn, grid, dataset["test"], cfg, eval_pignn.accuracy,
    )
    print_robustness_summary(robustness_results)

    # ══════════════════════════════════════════════════════════════════
    # Step 7: Final Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Test system    : {grid.bench.name}")
    print(f"  PIGNN params   : {pignn.count_parameters():,}")
    print(f"  |V| RMSE       : {eval_pignn.accuracy.vm_rmse:.6f} p.u.")
    print(f"  V violations   : {eval_pignn.physics.pct_v_violations:.1f}%")
    print(f"  Speed-up vs PF : {eval_pignn.compute.speedup_vs_pf:.1f}×")
    print(f"  Training time  : {hist_pignn.total_time_s:.1f}s")
    print(f"  Ablation tests : {len(ablation_results)}")
    print(f"  Robustness tests: {len(robustness_results)}")
    print(f"{'=' * 65}\n")

    return {
        "grid": grid,
        "dataset": dataset,
        "models": {
            "pignn": pignn, "nn": model_nn,
            "gnn": model_gnn, "pinn": model_pinn,
        },
        "histories": {
            "pignn": hist_pignn, "nn": hist_nn,
            "gnn": hist_gnn, "pinn": hist_pinn,
        },
        "evaluations": {
            "pignn": eval_pignn, "nn": eval_nn,
            "gnn": eval_gnn, "pinn": eval_pinn,
        },
        "ablation": ablation_results,
        "robustness": robustness_results,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIGNN IEEE Transactions — Full Simulation Pipeline"
    )
    parser.add_argument(
        "--case", type=str, default="ieee33",
        choices=["ieee33", "ieee123"],
        help="Test system benchmark (default: ieee33)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum training epochs (default: 100, paper: 300)",
    )
    parser.add_argument(
        "--scenarios", type=int, default=200,
        help="Number of scenarios (default: 200, paper: 10000)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=48,
        help="Timesteps per day (default: 48, paper: 96)",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.grid.case = args.case
    cfg.training.max_epochs = args.epochs
    cfg.scenario.n_scenarios = args.scenarios
    cfg.scenario.timesteps_per_day = args.timesteps
    cfg.output_dir = args.output
    cfg.verbose = not args.quiet

    # Reduce OOD scenarios proportionally
    cfg.scenario.n_ood_scenarios = max(10, args.scenarios // 20)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
from config import ExperimentConfig
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig

