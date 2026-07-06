"""
training.py — Training Protocol
================================
Section III-H of the manuscript.

Implements:
  - Adam optimiser with cosine-annealing learning rate schedule
  - Numerical gradient computation (finite differences)
  - Early stopping with patience monitoring
  - Multi-seed training loop for variance quantification
  - Training history logging
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np

from config import ExperimentConfig
from losses import compute_total_loss, LossBreakdown


# ──────────────────────────────────────────────────────────────────────
# Optimiser
# ──────────────────────────────────────────────────────────────────────

def adam_update(
    layers: list,
    lr: float,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """
    Apply one Adam optimiser step to all DenseLayer parameters.

    Uses bias-corrected first and second moment estimates as specified
    in the training protocol (Section III-H).
    """
    for layer in layers:
        # Weight update
        layer.mW = beta1 * layer.mW + (1 - beta1) * layer.dW
        layer.vW = beta2 * layer.vW + (1 - beta2) * layer.dW ** 2
        mW_hat = layer.mW / (1 - beta1 ** step)
        vW_hat = layer.vW / (1 - beta2 ** step)
        layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        # Bias update
        layer.mb = beta1 * layer.mb + (1 - beta1) * layer.db
        layer.vb = beta2 * layer.vb + (1 - beta2) * layer.db ** 2
        mb_hat = layer.mb / (1 - beta1 ** step)
        vb_hat = layer.vb / (1 - beta2 ** step)
        layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


def cosine_annealing_lr(
    epoch: int,
    max_epochs: int,
    lr_init: float,
    lr_min: float,
) -> float:
    """
    Cosine-annealing learning rate schedule.

    lr(t) = lr_min + 0.5 * (lr_init - lr_min) * (1 + cos(π * t / T_max))
    """
    return lr_min + 0.5 * (lr_init - lr_min) * (
        1 + np.cos(np.pi * epoch / max_epochs)
    )


# ──────────────────────────────────────────────────────────────────────
# Numerical Gradient Estimation
# ──────────────────────────────────────────────────────────────────────

def numerical_gradient_step(
    model,
    loss_fn,
    eps: float = 1e-4,
    n_samples: int = 20,
) -> None:
    """
    Estimate gradients via finite differences for each trainable layer.

    For production use, replace with automatic differentiation (PyTorch).
    This implementation samples a subset of parameters per layer for
    computational feasibility. v3 upgrade: increased default n_samples
    from 10 to 20 for better gradient coverage.

    Parameters
    ----------
    model : any model with .get_trainable_layers()
    loss_fn : callable returning (loss_value, loss_breakdown)
    eps : float
        Finite-difference step size.
    n_samples : int
        Number of parameters to sample per layer.
    """
    for layer in model.get_trainable_layers():
        # Weight gradients
        n_w = min(n_samples, layer.W.size)
        indices = np.random.choice(layer.W.size, n_w, replace=False)
        grad_W = np.zeros_like(layer.W)
        for idx in indices:
            multi_idx = np.unravel_index(idx, layer.W.shape)
            layer.W[multi_idx] += eps
            loss_plus, _ = loss_fn()
            layer.W[multi_idx] -= 2 * eps
            loss_minus, _ = loss_fn()
            layer.W[multi_idx] += eps  # restore
            grad_W[multi_idx] = (loss_plus - loss_minus) / (2 * eps)
        layer.dW = grad_W

        # Bias gradients (sample more elements for better coverage)
        n_b = min(n_samples // 2, layer.b.size)
        indices_b = np.random.choice(layer.b.size, max(n_b, 1), replace=False)
        for j in indices_b:
            layer.b[j] += eps
            loss_plus, _ = loss_fn()
            layer.b[j] -= 2 * eps
            loss_minus, _ = loss_fn()
            layer.b[j] += eps
            layer.db[j] = (loss_plus - loss_minus) / (2 * eps)


# ──────────────────────────────────────────────────────────────────────
# Training History
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    """Records per-epoch training and validation metrics."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    loss_components: List[Dict[str, float]] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_time_s: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Early Stopping Monitor
# ──────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation loss does not decrease for `patience`
    consecutive epochs (Section III-H).
    """

    def __init__(self, patience: int = 30):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────

def train_model(
    model,
    grid,
    train_scenarios: list,
    val_scenarios: list,
    cfg: ExperimentConfig,
    use_physics_loss: bool = True,
    label: str = "PIGNN",
) -> TrainingHistory:
    """
    Train a model using the protocol defined in Section III-H.

    Parameters
    ----------
    model : PIGNN, StandardNN, GNNOnly, or PINNOnly
    grid : DistributionGrid
    train_scenarios : list of scenario dicts
    val_scenarios : list of scenario dicts
    cfg : ExperimentConfig
    use_physics_loss : bool
        If False, only supervised data loss is used (for NN baseline).
    label : str
        Display label for progress logging.

    Returns
    -------
    TrainingHistory
    """
    from graph import build_node_features, build_edge_features

    history = TrainingHistory()
    early_stop = EarlyStopping(patience=cfg.training.early_stop_patience)
    T = cfg.scenario.timesteps_per_day

    # Precompute edge features (static per grid)
    edge_features = build_edge_features(
        grid.edge_index, grid.r_pu, grid.x_pu, grid.connections,
    )

    # Solve ground truth for a subset of training scenarios
    if cfg.verbose:
        print(f"\n  Training {label} ({model.count_parameters():,} params)...")

    adam_step = 0
    t_start = time.time()

    for epoch in range(cfg.training.max_epochs):
        lr = cosine_annealing_lr(
            epoch, cfg.training.max_epochs,
            cfg.training.learning_rate, cfg.training.lr_min,
        )
        history.learning_rates.append(lr)

        # --- Training step (sample multiple snapshots per epoch) ---
        batch_loss = 0.0
        n_batch = min(cfg.training.batch_snapshots, len(train_scenarios))
        batch_bd = None
        for b in range(n_batch):
            sc_idx = (epoch * n_batch + b) % len(train_scenarios)
            t_idx = (epoch * n_batch + b) % T
            sc = train_scenarios[sc_idx]

            # Solve ground truth for this snapshot
            p_net = sc["p_pv"][t_idx] + sc["p_bess"][t_idx] - sc["p_load"][t_idx]
            q_net = sc["q_pv"][t_idx] - sc["q_load"][t_idx]
            p_net_slack = p_net.copy()
            p_net_slack[0] = -np.sum(p_net[1:])
            q_net_slack = q_net.copy()
            q_net_slack[0] = -np.sum(q_net[1:])
            pf_result = grid.solve_power_flow(p_net_slack, q_net_slack)
            if not pf_result.converged:
                continue
            vm_true = pf_result.vm
            va_true = pf_result.va

            # Node features
            node_feat = build_node_features(
                sc, t_idx, grid.pv_buses, grid.bess_buses,
                grid.p_load_nom, grid.q_load_nom, grid.n_bus,
            )

            # Define loss function for this snapshot
            def compute_loss(nf=node_feat, vmt=vm_true, vat=va_true,
                             pns=p_net_slack, qns=q_net_slack, ibr=pf_result.i_branch):
                try:
                    pred = model.forward(
                        nf, (np.abs(grid.Y_bus) > 0).astype(float),
                        grid.edge_index, edge_features,
                    )
                except TypeError:
                    pred = model.forward(nf)

                if use_physics_loss:
                    lb = compute_total_loss(
                        pred["vm"], pred["va"], vmt, vat,
                        pns, qns, grid.Y_bus,
                        cfg.grid, cfg.loss, ibr,
                    )
                    return lb.total, lb
                else:
                    from losses import data_loss, LossBreakdown
                    dl = data_loss(pred["vm"], pred["va"], vmt, vat)
                    return dl, LossBreakdown(dl, dl, 0, 0, 0, 0)

            # Gradient estimation and update
            numerical_gradient_step(model, compute_loss)
            adam_step += 1
            adam_update(
                model.get_trainable_layers(), lr, adam_step,
                cfg.training.beta1, cfg.training.beta2, cfg.training.epsilon,
            )

            loss_val, bd = compute_loss()
            batch_loss += loss_val
            batch_bd = bd

        if batch_bd is None:
            continue
        train_loss_val = batch_loss / max(n_batch, 1)
        loss_bd = batch_bd
        history.train_loss.append(train_loss_val)
        history.loss_components.append({
            "data": loss_bd.data,
            "power_balance": loss_bd.power_balance,
            "voltage_bounds": loss_bd.voltage_bounds,
            "slack_ref": loss_bd.slack_ref,
            "thermal": loss_bd.thermal,
        })

        # --- Validation (average over multiple snapshots) ---
        val_losses = []
        for vb in range(min(3, len(val_scenarios))):
            val_sc = val_scenarios[(epoch * 3 + vb) % len(val_scenarios)]
            val_t = (epoch * 3 + vb) % T
            val_pnet = val_sc["p_pv"][val_t] + val_sc["p_bess"][val_t] - val_sc["p_load"][val_t]
            val_qnet = val_sc["q_pv"][val_t] - val_sc["q_load"][val_t]
            val_pnet_s = val_pnet.copy()
            val_qnet_s = val_qnet.copy()
            val_pnet_s[0] = -np.sum(val_pnet[1:])
            val_qnet_s[0] = -np.sum(val_qnet[1:])
            val_pf = grid.solve_power_flow(val_pnet_s, val_qnet_s)
            if not val_pf.converged:
                continue

            val_nf = build_node_features(
                val_sc, val_t, grid.pv_buses, grid.bess_buses,
                grid.p_load_nom, grid.q_load_nom, grid.n_bus,
            )
            try:
                val_pred = model.forward(
                    val_nf,
                    (np.abs(grid.Y_bus) > 0).astype(float),
                    grid.edge_index, edge_features,
                )
            except TypeError:
                val_pred = model.forward(val_nf)

            from losses import data_loss
            val_losses.append(data_loss(val_pred["vm"], val_pred["va"], val_pf.vm, val_pf.va))

        val_loss_val = np.mean(val_losses) if val_losses else history.best_val_loss
        history.val_loss.append(val_loss_val)

        if val_loss_val < history.best_val_loss:
            history.best_val_loss = val_loss_val
            history.best_epoch = epoch

        # Early stopping
        if early_stop.step(val_loss_val):
            if cfg.verbose:
                print(f"    Early stopping at epoch {epoch + 1}")
            break

        # Progress logging
        if cfg.verbose and (epoch + 1) % 50 == 0:
            print(
                f"    Epoch {epoch + 1:4d}/{cfg.training.max_epochs}: "
                f"train={train_loss_val:.6f}  val={val_loss_val:.6f}  "
                f"lr={lr:.2e}  "
                f"PB={loss_bd.power_balance:.5f}  VB={loss_bd.voltage_bounds:.5f}"
            )

    history.total_time_s = time.time() - t_start
    if cfg.verbose:
        print(f"    Training complete: {history.total_time_s:.1f}s, "
              f"best_val={history.best_val_loss:.6f} (epoch {history.best_epoch + 1})")

    return history
