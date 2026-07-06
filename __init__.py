"""
PIGNN Simulation Platform — IEEE Transactions on Power Systems
================================================================
PIGNN-L5: Fast and Physically Consistent Graph Neural Surrogate for AC Power
Flow Under Stochastic Renewable Penetration in Active Distribution Networks.

This package implements the complete simulation setup described in
Section III of the manuscript. Each module maps directly to a
subsection of the paper:

    config.py        → All experimental configuration
    benchmarks.py    → Section III-A : Test Systems
    grid.py          → Section III-C : Ground-Truth AC Power Flow Solver
    scenarios.py     → Section III-B : Scenario Generation
    graph.py         → Section III-D : Graph Construction & Input Features
    models/
        gnn.py       → Section III-E : GNN Topology Encoder
        pinn.py      → Section III-E : PINN Physics Decoder
        pignn.py     → Section III-E : Combined PIGNN Model
        baselines.py → Section III-G : Baseline Models (NN, GNN-only, PINN-only)
    losses.py        → Section III-F : Loss Function & Physics Constraints
    training.py      → Section III-H : Training Protocol
    metrics.py       → Section III-I : Evaluation Metrics
    ablation.py      → Section III-J : Ablation Study Design
    robustness.py    → Section III-K : Robustness & Generalisation Tests
    run_all.py       → Full experiment orchestrator

Author  : [Author Name]
Date    : 2026
License : MIT
"""

__version__ = "4.1.0"
