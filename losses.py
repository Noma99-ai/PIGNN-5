"""
losses.py — Loss Function and Physics Constraints
==================================================
Section III-F of the manuscript.

Total training loss:
    L_total = λ_data · L_data + λ_PB · L_PB + λ_VB · L_VB
            + λ_SL · L_SL + λ_TH · L_TH

where:
  L_data  — MSE between predicted and ground-truth |V| and δ
  L_PB    — KCL power-balance residual (active + reactive)
  L_VB    — Voltage bound violation penalty (one-sided hinge)
  L_SL    — Slack bus reference condition (|V₀|=1, δ₀=0)
  L_TH    — Branch thermal limit penalty (one-sided hinge)

Note: BESS SOC energy conservation is intentionally excluded because
the snapshot-based formulation has no temporal coupling.  BESS active
power is retained as an exogenous input feature.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from config import LossConfig, GridConfig


@dataclass
class LossBreakdown:
    """Itemised loss components for logging and analysis."""
    total: float
    data: float
    power_balance: float
    voltage_bounds: float
    slack_ref: float
    thermal: float


# ──────────────────────────────────────────────────────────────────────
# Individual Loss Terms
# ──────────────────────────────────────────────────────────────────────

def data_loss(
    vm_pred: np.ndarray,
    va_pred: np.ndarray,
    vm_true: np.ndarray,
    va_true: np.ndarray,
) -> float:
    """
    L_data: Mean squared error on voltage magnitude and angle.

    L_data = (1/N) Σ_i [ (|V̂_i| − |V_i|)² + (δ̂_i − δ_i)² ]
    """
    return float(np.mean((vm_pred - vm_true) ** 2 + (va_pred - va_true) ** 2))


def power_balance_loss(
    vm: np.ndarray,
    va: np.ndarray,
    p_net: np.ndarray,
    q_net: np.ndarray,
    Y_bus: np.ndarray,
) -> float:
    """
    L_PB: KCL power-balance residual for active and reactive power.

    Penalises the mismatch between the net bus injections and the power
    computed from predicted voltages via the bus admittance matrix.

    L_PB = (1/N) Σ_i [ (P_net_i − P_calc_i)² + (Q_net_i − Q_calc_i)² ]
    """
    n = len(vm)
    G = Y_bus.real
    B = Y_bus.imag
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

    return float(np.mean((p_net - p_calc) ** 2 + (q_net - q_calc) ** 2))


def voltage_bounds_loss(
    vm: np.ndarray,
    v_min: float,
    v_max: float,
) -> float:
    """
    L_VB: One-sided squared hinge loss for voltage bound violations.

    Only penalises non-slack buses outside [V_min, V_max].
    L_VB = (1/(N-1)) Σ_{i≠0} [ max(0, V_min − |V_i|)² + max(0, |V_i| − V_max)² ]
    """
    vm_non_slack = vm[1:]  # Exclude slack bus
    low_violation = np.maximum(0, v_min - vm_non_slack) ** 2
    high_violation = np.maximum(0, vm_non_slack - v_max) ** 2
    return float(np.mean(low_violation + high_violation))


def slack_reference_loss(
    vm: np.ndarray,
    va: np.ndarray,
    slack_vm: float = 1.0,
    slack_va: float = 0.0,
) -> float:
    """
    L_SL: Enforce slack bus reference condition.

    L_SL = (|V̂₀| − 1.0)² + (δ̂₀ − 0)²
    """
    return float(abs(vm[0] - slack_vm) ** 2 + abs(va[0] - slack_va) ** 2)


def thermal_limit_loss(
    i_branch: Optional[np.ndarray],
    thermal_limit: float = 1.0,
) -> float:
    """
    L_TH: One-sided squared hinge loss on branch current magnitude.

    L_TH = (1/L) Σ_l max(0, |I_l| − I_max)²

    Branch currents are computed from predicted voltages and known
    line impedances via Ohm's law.
    """
    if i_branch is None:
        return 0.0
    violation = np.maximum(0, i_branch - thermal_limit) ** 2
    return float(np.mean(violation))


# ──────────────────────────────────────────────────────────────────────
# Combined Loss Function
# ──────────────────────────────────────────────────────────────────────

def compute_total_loss(
    vm_pred: np.ndarray,
    va_pred: np.ndarray,
    vm_true: np.ndarray,
    va_true: np.ndarray,
    p_net: np.ndarray,
    q_net: np.ndarray,
    Y_bus: np.ndarray,
    grid_cfg: GridConfig,
    loss_cfg: LossConfig,
    i_branch: Optional[np.ndarray] = None,
) -> LossBreakdown:
    """
    Compute the total physics-informed training loss (Eq. in Section III-F).

    Parameters
    ----------
    vm_pred, va_pred : predicted voltage state
    vm_true, va_true : ground-truth voltage state
    p_net, q_net     : net bus power injections [p.u.]
    Y_bus            : bus admittance matrix
    grid_cfg         : grid operational limits
    loss_cfg         : loss weighting coefficients
    i_branch         : branch current magnitudes (optional)

    Returns
    -------
    LossBreakdown with total loss and individual components.
    """
    l_data = data_loss(vm_pred, va_pred, vm_true, va_true)
    l_pb = power_balance_loss(vm_pred, va_pred, p_net, q_net, Y_bus)
    l_vb = voltage_bounds_loss(vm_pred, grid_cfg.v_min_pu, grid_cfg.v_max_pu)
    l_sl = slack_reference_loss(vm_pred, va_pred, grid_cfg.slack_vm_pu, grid_cfg.slack_va_rad)
    l_th = thermal_limit_loss(i_branch, grid_cfg.thermal_limit_pu)

    total = (
        loss_cfg.lambda_data * l_data
        + loss_cfg.lambda_power_balance * l_pb
        + loss_cfg.lambda_voltage_bounds * l_vb
        + loss_cfg.lambda_slack_ref * l_sl
        + loss_cfg.lambda_thermal * l_th
    )

    return LossBreakdown(
        total=total,
        data=l_data,
        power_balance=l_pb,
        voltage_bounds=l_vb,
        slack_ref=l_sl,
        thermal=l_th,
    )
