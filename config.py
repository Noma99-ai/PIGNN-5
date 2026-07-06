"""
config.py — Centralised Experiment Configuration
=================================================
All hyperparameters, grid parameters, and experimental settings for the
PIGNN IEEE Transactions study — PIGNN-L5: Fast and Physically Consistent
Graph Neural Surrogate for AC Power Flow Under Stochastic Renewable
Penetration in Active Distribution Networks.  Modify only this file to change any
aspect of the simulation; all other modules read from these dataclasses.

Reference: Section III of the manuscript.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
# Section III-A : Test System Parameters
# ──────────────────────────────────────────────────────────────────────
@dataclass
class GridConfig:
    """Electrical and operational parameters of the test feeder."""
    case: str = "ieee33"              # "ieee33" or "ieee123"
    v_min_pu: float = 0.95            # Lower statutory voltage bound [p.u.]
    v_max_pu: float = 1.05            # Upper statutory voltage bound [p.u.]
    s_base_mva: float = 10.0          # System base power [MVA]
    thermal_limit_pu: float = 1.0     # Branch thermal limit [p.u.]
    slack_bus: int = 0                # Slack / reference bus index
    slack_vm_pu: float = 1.0          # Slack bus voltage setpoint [p.u.]
    slack_va_rad: float = 0.0         # Slack bus angle setpoint [rad]


# ──────────────────────────────────────────────────────────────────────
# Section III-A : DER Placement
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PVConfig:
    """Photovoltaic generator configuration."""
    n_units: int = 6                  # Number of PV units
    buses: Optional[List[int]] = None # Explicit bus placement (auto if None)
    peak_mw: float = 0.4             # Rated peak per unit [MW]
    noise_std: float = 0.04          # Irradiance noise σ [fraction of 1000 W/m²]
    temp_coeff: float = -0.004       # Temperature derating [1/°C relative to 25°C]
    penetration_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )


@dataclass
class BESSConfig:
    """Battery energy storage system configuration."""
    enabled: bool = True
    n_units: int = 2
    buses: Optional[List[int]] = None
    capacity_mwh: float = 2.0        # Energy capacity [MWh]
    p_max_mw: float = 1.0            # Max charge/discharge rate [MW]
    efficiency: float = 0.92         # Round-trip efficiency
    soc_min: float = 0.10            # Minimum SOC bound
    soc_max: float = 0.90            # Maximum SOC bound
    soc_initial: float = 0.50        # Initial SOC


# ──────────────────────────────────────────────────────────────────────
# Section III-E : PIGNN Model Configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class GNNConfig:
    """GNN topology encoder hyperparameters (Table II)."""
    hidden_dim: int = 64              # Node embedding dimension d
    n_message_passing: int = 3        # Number of MP rounds K
    aggregation: str = "mean"         # Neighbour aggregation method
    use_edge_gating: bool = True      # Admittance-based edge gating
    use_residual: bool = True         # Skip connections for k ≥ 2
    use_layer_norm: bool = True       # LayerNorm after final MP round


@dataclass
class PINNConfig:
    """PINN physics decoder hyperparameters (Table II)."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    vm_range: tuple = (0.90, 1.05)    # Sigmoid output range [p.u.]
    va_range: tuple = (-0.30, 0.30)   # Tanh output range [rad]


# ──────────────────────────────────────────────────────────────────────
# Section III-F : Loss Weights
# ──────────────────────────────────────────────────────────────────────
@dataclass
class LossConfig:
    """Physics-informed loss function weights."""
    lambda_data: float = 1.0          # Supervised data-fitting
    lambda_power_balance: float = 10.0  # KCL active + reactive
    lambda_voltage_bounds: float = 5.0  # Voltage limit penalty
    lambda_slack_ref: float = 2.0     # Slack bus reference
    lambda_thermal: float = 3.0       # Branch thermal limit


# ──────────────────────────────────────────────────────────────────────
# Section III-B : Scenario Generation
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ScenarioConfig:
    """Data generation parameters."""
    n_scenarios: int = 30_000         # 30-day horizon: 30,000 daily scenarios
    timesteps_per_day: int = 96       # Δt = 15 min → 96 per day
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    n_ood_scenarios: int = 1_500      # 3× enlarged OOD test set
    ood_pv_multipliers: List[float] = field(default_factory=lambda: [1.2, 1.5])
    ood_load_multiplier: float = 1.3
    ood_noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05])


# ──────────────────────────────────────────────────────────────────────
# Section III-H : Training Protocol
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Optimiser, scheduler, and reproducibility settings."""
    learning_rate: float = 5e-4       # Manuscript: 5×10⁻⁴
    lr_min: float = 1e-6              # Cosine-annealing floor
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_epochs: int = 300
    early_stop_patience: int = 30
    batch_size: int = 256             # Manuscript: 256 snapshots
    batch_snapshots: int = 4          # Snapshots per gradient step
    seeds: List[int] = field(default_factory=lambda: [42, 123, 256, 512, 1024])
    gradient_clip_norm: float = 1.0   # Max gradient norm
    n_grad_samples: int = 20          # FD samples per layer (v3 upgrade)


# ──────────────────────────────────────────────────────────────────────
# Section III-K : Robustness Test Parameters
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RobustnessConfig:
    """Stress-test parameters for generalisation evaluation."""
    noise_sigma_levels: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05])
    elevated_pv_multipliers: List[float] = field(default_factory=lambda: [1.2, 1.5])
    load_stress_multiplier: float = 1.3
    n1_contingency_lines: Optional[List[int]] = None  # Auto-select if None


# ──────────────────────────────────────────────────────────────────────
# Top-Level Configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """Master configuration aggregating all sub-configs."""
    grid: GridConfig = field(default_factory=GridConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    bess: BESSConfig = field(default_factory=BESSConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    pinn: PINNConfig = field(default_factory=PINNConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    output_dir: str = "results"
    verbose: bool = True

    def summary(self) -> str:
        """Return a human-readable summary of key settings."""
        lines = [
            "=" * 68,
            "  PIGNN Experiment Configuration",
            "=" * 68,
            f"  Test system       : {self.grid.case}",
            f"  Scenarios         : {self.scenario.n_scenarios} "
            f"(train {self.scenario.train_ratio:.0%} / "
            f"val {self.scenario.val_ratio:.0%} / "
            f"test {self.scenario.test_ratio:.0%})",
            f"  Timesteps/day     : {self.scenario.timesteps_per_day}",
            f"  PV units          : {self.pv.n_units}   |  BESS units: {self.bess.n_units}",
            f"  GNN hidden dim    : {self.gnn.hidden_dim}  |  MP rounds : {self.gnn.n_message_passing}",
            f"  PINN hidden       : {self.pinn.hidden_dims}",
            f"  |V| output range  : {self.pinn.vm_range}",
            f"  δ   output range  : {self.pinn.va_range}",
            f"  Loss weights      : data={self.loss.lambda_data}, "
            f"PB={self.loss.lambda_power_balance}, "
            f"VB={self.loss.lambda_voltage_bounds}, "
            f"SL={self.loss.lambda_slack_ref}, "
            f"TH={self.loss.lambda_thermal}",
            f"  Optimiser         : Adam (lr={self.training.learning_rate})",
            f"  Max epochs        : {self.training.max_epochs}  |  "
            f"Early stop: {self.training.early_stop_patience}",
            f"  Random seeds      : {self.training.seeds}",
            "=" * 68,
        ]
        return "\n".join(lines)
