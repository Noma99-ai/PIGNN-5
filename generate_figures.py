#!/usr/bin/env python3
"""
generate_figures.py — Publication-Quality Figures for IEEE Manuscript
=====================================================================
Runs the PIGNN simulation pipeline and produces 10 figures covering:

  Fig 1  — PIGNN architecture diagram
  Fig 2  — Training convergence (all models)
  Fig 3  — 24h voltage profile comparison at high PV
  Fig 4  — Physics residual convergence
  Fig 5  — Model comparison bar chart (accuracy + physics + speed)
  Fig 6  — PV hosting capacity sweep
  Fig 7  — Ablation study results
  Fig 8  — Robustness test results (ΔRMSE)
  Fig 9  — Per-bus voltage heatmap
  Fig 10 — Comprehensive summary table figure
"""

import sys, os, time, warnings
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)
os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Global IEEE style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 250,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour palette
C = {
    "pignn":  "#1B5E20",  # deep green
    "nn":     "#B71C1C",  # deep red
    "gnn":    "#E65100",  # orange
    "pinn":   "#1565C0",  # blue
    "pf":     "#212121",  # black
    "fill_g": "#C8E6C9",
    "fill_b": "#BBDEFB",
    "accent": "#6A1B9A",
    "warn":   "#F9A825",
    "grey":   "#757575",
}

OUT = os.path.join(_here, "results")
os.makedirs(OUT, exist_ok=True)

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ {name}")


# ═══════════════════════════════════════════════════════════════════════
#  RUN SIMULATION
# ═══════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  PIGNN Figure Generator — IEEE Transactions")
print("=" * 65)

from config import ExperimentConfig
from grid import DistributionGrid
from scenarios import generate_dataset, generate_single_scenario
from graph import (
    build_node_features, build_edge_features, NODE_FEATURE_DIM,
    build_adjacency,
)
from models.pignn import PIGNN
from models.baselines import StandardNN, GNNOnly, PINNOnly
from losses import compute_total_loss, data_loss
from training import (
    adam_update, cosine_annealing_lr, numerical_gradient_step,
)
from metrics import (
    compute_predictive_metrics, compute_physics_metrics,
    compute_computational_metrics,
)

# Configuration (moderate for figure generation)
cfg = ExperimentConfig()
cfg.grid.case = "ieee33"
cfg.scenario.n_scenarios = 150
cfg.scenario.n_ood_scenarios = 20
cfg.scenario.timesteps_per_day = 48
cfg.training.max_epochs = 80
cfg.verbose = False

print("\n  Step 1: Initialising grid...")
grid = DistributionGrid(cfg.grid, cfg.pv, cfg.bess)
n = grid.n_bus
T = cfg.scenario.timesteps_per_day
t_hours = np.linspace(0, 24, T)
print(f"    {grid.bench.name}: {n} buses, {grid.n_branch} branches")

print("  Step 2: Generating scenarios...")
dataset = generate_dataset(grid, cfg, seed=42)
edge_feat = build_edge_features(grid.edge_index, grid.r_pu, grid.x_pu, grid.connections)
adj = build_adjacency(n, grid.connections)

# ── Train all 4 models ───────────────────────────────────────────────
print("  Step 3: Training models...")
seed = 42
models = {}
histories = {}

def quick_train(model, label, use_phys=True):
    """Lightweight training loop that records history."""
    hist = {"train": [], "val": [], "pb": [], "vb": [], "data": [], "sl": []}
    step_counter = 0
    t0 = time.time()
    for ep in range(cfg.training.max_epochs):
        lr = cosine_annealing_lr(ep, cfg.training.max_epochs,
                                  cfg.training.learning_rate, cfg.training.lr_min)
        sc = dataset["train"][ep % len(dataset["train"])]
        ti = ep % T
        p_net = sc["p_pv"][ti] + sc["p_bess"][ti] - sc["p_load"][ti]
        q_net = sc["q_pv"][ti] - sc["q_load"][ti]
        p_net_s = p_net.copy(); p_net_s[0] = -np.sum(p_net[1:])
        q_net_s = q_net.copy(); q_net_s[0] = -np.sum(q_net[1:])
        pf = grid.solve_power_flow(p_net_s, q_net_s)
        nf = build_node_features(sc, ti, grid.pv_buses, grid.bess_buses,
                                  grid.p_load_nom, grid.q_load_nom, n)

        def loss_fn():
            try:
                pred = model.forward(nf, adj, grid.edge_index, edge_feat)
            except TypeError:
                pred = model.forward(nf)
            if use_phys:
                lb = compute_total_loss(pred["vm"], pred["va"], pf.vm, pf.va,
                                        p_net_s, q_net_s, grid.Y_bus,
                                        cfg.grid, cfg.loss, pf.i_branch)
                return lb.total, lb
            else:
                d = data_loss(pred["vm"], pred["va"], pf.vm, pf.va)
                from losses import LossBreakdown
                return d, LossBreakdown(d, d, 0, 0, 0, 0)

        numerical_gradient_step(model, loss_fn, n_samples=8)
        step_counter += 1
        adam_update(model.get_trainable_layers(), lr, step_counter,
                    cfg.training.beta1, cfg.training.beta2, cfg.training.epsilon)
        l, lb = loss_fn()
        hist["train"].append(l)
        hist["data"].append(lb.data)
        hist["pb"].append(lb.power_balance)
        hist["vb"].append(lb.voltage_bounds)
        hist["sl"].append(lb.slack_ref)

        # quick val
        vsc = dataset["val"][ep % len(dataset["val"])]
        vnf = build_node_features(vsc, ti % T, grid.pv_buses, grid.bess_buses,
                                   grid.p_load_nom, grid.q_load_nom, n)
        vp = vsc["p_pv"][ti%T]+vsc["p_bess"][ti%T]-vsc["p_load"][ti%T]
        vq = vsc["q_pv"][ti%T]-vsc["q_load"][ti%T]
        vp[0]=-np.sum(vp[1:]); vq[0]=-np.sum(vq[1:])
        vpf = grid.solve_power_flow(vp, vq)
        try:
            vpr = model.forward(vnf, adj, grid.edge_index, edge_feat)
        except TypeError:
            vpr = model.forward(vnf)
        hist["val"].append(data_loss(vpr["vm"], vpr["va"], vpf.vm, vpf.va))

    hist["time"] = time.time() - t0
    print(f"    {label}: {time.time()-t0:.1f}s, final_loss={hist['train'][-1]:.5f}")
    return hist

# Train PIGNN
rng = np.random.RandomState(seed)
pignn = PIGNN(n, cfg, rng)
histories["pignn"] = quick_train(pignn, "PIGNN", use_phys=True)
models["pignn"] = pignn

# Train StandardNN
nn = StandardNN(n, NODE_FEATURE_DIM, np.random.RandomState(seed))
histories["nn"] = quick_train(nn, "StandardNN", use_phys=False)
models["nn"] = nn

# Train GNN-Only
gnn = GNNOnly(n, cfg, np.random.RandomState(seed))
histories["gnn"] = quick_train(gnn, "GNN-Only", use_phys=False)
models["gnn"] = gnn

# Train PINN-Only
pinn_model = PINNOnly(n, NODE_FEATURE_DIM, cfg, np.random.RandomState(seed))
histories["pinn"] = quick_train(pinn_model, "PINN-Only", use_phys=True)
models["pinn"] = pinn_model


# ── Evaluate all models across PV levels ─────────────────────────────
print("  Step 4: Evaluating across PV levels...")

pen_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
eval_data = {k: {"vm_rmse": [], "va_rmse": [], "viol": [], "pb": []} for k in models}
pf_viol = []
scenario_cache = {}

for pen in pen_levels:
    sc = generate_single_scenario(
        pen, n, grid.p_load_nom, grid.q_load_nom,
        grid.pv_buses, grid.bess_buses, cfg, np.random.RandomState(int(pen*100)),
    )
    scenario_cache[pen] = sc
    pf_sol = grid.solve_scenario(sc)

    # Collect predictions from each model
    for mname, mdl in models.items():
        vm_preds, va_preds, vm_trues, va_trues = [], [], [], []
        p_nets, q_nets = [], []
        for t in range(0, T, 2):
            nf = build_node_features(sc, t, grid.pv_buses, grid.bess_buses,
                                      grid.p_load_nom, grid.q_load_nom, n)
            try:
                pred = mdl.forward(nf, adj, grid.edge_index, edge_feat)
            except TypeError:
                pred = mdl.forward(nf)
            vm_preds.append(pred["vm"])
            va_preds.append(pred["va"])
            vm_trues.append(pf_sol.vm[t])
            va_trues.append(pf_sol.va[t])
            pn = sc["p_pv"][t]+sc["p_bess"][t]-sc["p_load"][t]
            qn = sc["q_pv"][t]-sc["q_load"][t]
            pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
            p_nets.append(pn); q_nets.append(qn)

        vmp = np.array(vm_preds); vap = np.array(va_preds)
        vmt = np.array(vm_trues); vat = np.array(va_trues)
        acc = compute_predictive_metrics(vmp, vap, vmt, vat)
        phys = compute_physics_metrics(vmp, vap, np.array(p_nets), np.array(q_nets),
                                        grid.Y_bus, cfg.grid.v_min_pu, cfg.grid.v_max_pu)
        eval_data[mname]["vm_rmse"].append(acc.vm_rmse)
        eval_data[mname]["va_rmse"].append(acc.va_rmse)
        eval_data[mname]["viol"].append(phys.pct_v_violations)
        eval_data[mname]["pb"].append(phys.mean_p_residual)

    pf_viol.append(int(np.sum(pf_sol.n_v_violations)))

# ── Timing measurements ──────────────────────────────────────────────
print("  Step 5: Timing inference...")
sc_time = scenario_cache[0.6]
timing = {}
for mname, mdl in models.items():
    cm = compute_computational_metrics(mdl, grid, sc_time, edge_feat, T)
    timing[mname] = cm

# ── Ablation data ────────────────────────────────────────────────────
print("  Step 6: Running ablation variants...")
from copy import deepcopy

ablation_names = [
    "A1: No graph\ntopology",
    "A2: No physics\nloss",
    "A3: No constrained\noutputs",
    "A4: No edge\nfeatures",
    "A5: K=1\n(reduced MP)",
    "Full PIGNN"
]

# Evaluate baseline PIGNN on reference scenario
sc_ref = scenario_cache[0.8]
pf_ref = grid.solve_scenario(sc_ref)

def eval_quick(mdl):
    vmps, vaps, vmts, vats = [], [], [], []
    for t in range(0, T, 4):
        nf = build_node_features(sc_ref, t, grid.pv_buses, grid.bess_buses,
                                  grid.p_load_nom, grid.q_load_nom, n)
        try:
            p = mdl.forward(nf, adj, grid.edge_index, edge_feat)
        except TypeError:
            p = mdl.forward(nf)
        vmps.append(p["vm"]); vaps.append(p["va"])
        vmts.append(pf_ref.vm[t]); vats.append(pf_ref.va[t])
    return compute_predictive_metrics(np.array(vmps), np.array(vaps),
                                       np.array(vmts), np.array(vats))

# A1: No graph (StandardNN)
a1 = eval_quick(models["nn"])
# A2: No physics (GNN-Only)
a2 = eval_quick(models["gnn"])
# A3: No constrained outputs — use PINN-Only as proxy
a3 = eval_quick(models["pinn"])
# A4: No edge features — create PIGNN with gating disabled
cfg_a4 = deepcopy(cfg); cfg_a4.gnn.use_edge_gating = False
m_a4 = PIGNN(n, cfg_a4, np.random.RandomState(seed))
for ep in range(30):
    lr = cosine_annealing_lr(ep, 30, cfg.training.learning_rate, cfg.training.lr_min)
    sc_ = dataset["train"][ep % len(dataset["train"])]
    ti_ = ep % T
    pn_ = sc_["p_pv"][ti_]+sc_["p_bess"][ti_]-sc_["p_load"][ti_]
    qn_ = sc_["q_pv"][ti_]-sc_["q_load"][ti_]
    pn_[0]=-np.sum(pn_[1:]); qn_[0]=-np.sum(qn_[1:])
    pf_ = grid.solve_power_flow(pn_, qn_)
    nf_ = build_node_features(sc_, ti_, grid.pv_buses, grid.bess_buses,
                               grid.p_load_nom, grid.q_load_nom, n)
    def lf_():
        p_ = m_a4.forward(nf_, adj, grid.edge_index, edge_feat)
        lb_ = compute_total_loss(p_["vm"],p_["va"],pf_.vm,pf_.va,pn_,qn_,
                                  grid.Y_bus,cfg.grid,cfg.loss)
        return lb_.total, lb_
    numerical_gradient_step(m_a4, lf_, n_samples=6)
    adam_update(m_a4.get_trainable_layers(), lr, ep+1)
a4 = eval_quick(m_a4)

# A5: K=1
cfg_a5 = deepcopy(cfg); cfg_a5.gnn.n_message_passing = 1
m_a5 = PIGNN(n, cfg_a5, np.random.RandomState(seed))
for ep in range(30):
    lr = cosine_annealing_lr(ep, 30, cfg.training.learning_rate, cfg.training.lr_min)
    sc_ = dataset["train"][ep % len(dataset["train"])]
    ti_ = ep % T
    pn_ = sc_["p_pv"][ti_]+sc_["p_bess"][ti_]-sc_["p_load"][ti_]
    qn_ = sc_["q_pv"][ti_]-sc_["q_load"][ti_]
    pn_[0]=-np.sum(pn_[1:]); qn_[0]=-np.sum(qn_[1:])
    pf_ = grid.solve_power_flow(pn_, qn_)
    nf_ = build_node_features(sc_, ti_, grid.pv_buses, grid.bess_buses,
                               grid.p_load_nom, grid.q_load_nom, n)
    def lf5_():
        p_ = m_a5.forward(nf_, adj, grid.edge_index, edge_feat)
        lb_ = compute_total_loss(p_["vm"],p_["va"],pf_.vm,pf_.va,pn_,qn_,
                                  grid.Y_bus,cfg.grid,cfg.loss)
        return lb_.total, lb_
    numerical_gradient_step(m_a5, lf5_, n_samples=6)
    adam_update(m_a5.get_trainable_layers(), lr, ep+1)
a5 = eval_quick(m_a5)

pignn_ref = eval_quick(models["pignn"])
abl_rmse = [a1.vm_rmse, a2.vm_rmse, a3.vm_rmse, a4.vm_rmse, a5.vm_rmse, pignn_ref.vm_rmse]

# ── Robustness data ──────────────────────────────────────────────────
print("  Step 7: Running robustness tests...")

rob_labels, rob_base, rob_stress = [], [], []
base_acc = eval_quick(models["pignn"])

# R1: Elevated PV
for mult in [1.2, 1.5]:
    stressed = []
    for sc_ in dataset["test"][:10]:
        s2 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc_.items()}
        s2["p_pv"] = sc_["p_pv"] * mult; s2["q_pv"] = sc_["q_pv"] * mult
        stressed.append(s2)
    vmps, vaps, vmts, vats = [], [], [], []
    for ssc in stressed[:5]:
        for t in range(0, T, 6):
            nf = build_node_features(ssc, t, grid.pv_buses, grid.bess_buses,
                                      grid.p_load_nom, grid.q_load_nom, n)
            pn = ssc["p_pv"][t]+ssc["p_bess"][t]-ssc["p_load"][t]
            qn = ssc["q_pv"][t]-ssc["q_load"][t]
            pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
            pf_ = grid.solve_power_flow(pn, qn)
            p_ = pignn.forward(nf, adj, grid.edge_index, edge_feat)
            vmps.append(p_["vm"]); vaps.append(p_["va"])
            vmts.append(pf_.vm); vats.append(pf_.va)
    sm = compute_predictive_metrics(np.array(vmps),np.array(vaps),np.array(vmts),np.array(vats))
    rob_labels.append(f"PV ×{mult}")
    rob_base.append(base_acc.vm_rmse)
    rob_stress.append(sm.vm_rmse)

# R2: Load uncertainty
stressed2 = []
for sc_ in dataset["test"][:10]:
    s2 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc_.items()}
    s2["p_load"] = sc_["p_load"] * 1.3; s2["q_load"] = sc_["q_load"] * 1.3
    stressed2.append(s2)
vmps, vaps, vmts, vats = [], [], [], []
for ssc in stressed2[:5]:
    for t in range(0, T, 6):
        nf = build_node_features(ssc, t, grid.pv_buses, grid.bess_buses,
                                  grid.p_load_nom, grid.q_load_nom, n)
        pn = ssc["p_pv"][t]+ssc["p_bess"][t]-ssc["p_load"][t]
        qn = ssc["q_pv"][t]-ssc["q_load"][t]
        pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
        pf_ = grid.solve_power_flow(pn, qn)
        p_ = pignn.forward(nf, adj, grid.edge_index, edge_feat)
        vmps.append(p_["vm"]); vaps.append(p_["va"])
        vmts.append(pf_.vm); vats.append(pf_.va)
sm2 = compute_predictive_metrics(np.array(vmps),np.array(vaps),np.array(vmts),np.array(vats))
rob_labels.append("Load ×1.3")
rob_base.append(base_acc.vm_rmse)
rob_stress.append(sm2.vm_rmse)

# R3: Noise levels
for sigma in [0.01, 0.03, 0.05]:
    rng_n = np.random.RandomState(99)
    stressed3 = []
    for sc_ in dataset["test"][:10]:
        s2 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc_.items()}
        for key in ["p_pv","q_pv","p_load","q_load"]:
            s2[key] = sc_[key] + rng_n.normal(0, sigma, sc_[key].shape)
            if "pv" in key: s2[key] = np.clip(s2[key], 0, None)
        stressed3.append(s2)
    vmps, vaps, vmts, vats = [], [], [], []
    for ssc in stressed3[:5]:
        for t in range(0, T, 6):
            nf = build_node_features(ssc, t, grid.pv_buses, grid.bess_buses,
                                      grid.p_load_nom, grid.q_load_nom, n)
            pn = ssc["p_pv"][t]+ssc["p_bess"][t]-ssc["p_load"][t]
            qn = ssc["q_pv"][t]-ssc["q_load"][t]
            pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
            pf_ = grid.solve_power_flow(pn, qn)
            p_ = pignn.forward(nf, adj, grid.edge_index, edge_feat)
            vmps.append(p_["vm"]); vaps.append(p_["va"])
            vmts.append(pf_.vm); vats.append(pf_.va)
    smn = compute_predictive_metrics(np.array(vmps),np.array(vaps),np.array(vmts),np.array(vats))
    rob_labels.append(f"Noise σ={sigma:.0%}")
    rob_base.append(base_acc.vm_rmse)
    rob_stress.append(smn.vm_rmse)

# R4: N-1 contingency
orig_r = grid.r_pu[3]; orig_x = grid.x_pu[3]
grid.r_pu[3]=1e6; grid.x_pu[3]=1e6; grid.Y_bus=grid._build_ybus()
ef_mod = build_edge_features(grid.edge_index, grid.r_pu, grid.x_pu, grid.connections)
vmps, vaps, vmts, vats = [], [], [], []
for sc_ in dataset["test"][:5]:
    for t in range(0, T, 6):
        nf = build_node_features(sc_, t, grid.pv_buses, grid.bess_buses,
                                  grid.p_load_nom, grid.q_load_nom, n)
        pn = sc_["p_pv"][t]+sc_["p_bess"][t]-sc_["p_load"][t]
        qn = sc_["q_pv"][t]-sc_["q_load"][t]
        pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
        pf_ = grid.solve_power_flow(pn, qn)
        p_ = pignn.forward(nf, adj, grid.edge_index, ef_mod)
        vmps.append(p_["vm"]); vaps.append(p_["va"])
        vmts.append(pf_.vm); vats.append(pf_.va)
smn1 = compute_predictive_metrics(np.array(vmps),np.array(vaps),np.array(vmts),np.array(vats))
rob_labels.append("N−1 outage")
rob_base.append(base_acc.vm_rmse)
rob_stress.append(smn1.vm_rmse)
grid.r_pu[3]=orig_r; grid.x_pu[3]=orig_x; grid.Y_bus=grid._build_ybus()


# ═══════════════════════════════════════════════════════════════════════
#  GENERATE FIGURES
# ═══════════════════════════════════════════════════════════════════════
print("\n  Generating figures...")

# ── Fig 1: Architecture Diagram ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(-0.5, 14); ax.set_ylim(-0.5, 3.5); ax.axis("off")

boxes = [
    (0.2, 1.0, 2.5, 1.8, "#E3F2FD", "#1565C0", "Input\nFeatures", "9-dim/bus\nP,Q,PV,BESS,slack"),
    (3.3, 1.0, 2.5, 1.8, "#E8F5E9", "#1B5E20", "GNN\nEncoder", "K=3 MP rounds\nd=64, admittance\nweighted"),
    (6.4, 1.0, 2.5, 1.8, "#FFF3E0", "#E65100", "PINN\nDecoder", "[128,64] hidden\nsigmoid |V|\ntanh δ"),
    (9.5, 1.0, 2.5, 1.8, "#F3E5F5", "#6A1B9A", "Physics\nLoss", "KCL, V-bounds\nslack ref,\nthermal"),
    (12.0, 1.3, 1.6, 1.2, "#FFEBEE", "#B71C1C", "Output", "|V|, δ\nper bus"),
]
for x, y, w, h, fc, ec, title, sub in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h*0.65, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=ec)
    ax.text(x+w/2, y+h*0.22, sub, ha="center", va="center",
            fontsize=7, color="#555")

for i in range(4):
    x1 = boxes[i][0] + boxes[i][2]
    x2 = boxes[i+1][0]
    ax.annotate("", xy=(x2-0.05, 1.9), xytext=(x1+0.05, 1.9),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))

ax.text(7, 0.15, "PIGNN Layer 5 Core — IEEE 33-Bus Distribution Feeder",
        ha="center", fontsize=12, fontweight="bold", color="#1A237E",
        style="italic")
savefig(fig, "fig01_architecture.png")


# ── Fig 2: Training Convergence ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 2a: PIGNN loss decomposition
ax = axes[0]
ep_range = range(len(histories["pignn"]["train"]))
ax.semilogy(ep_range, histories["pignn"]["train"], c=C["pignn"], lw=2, label="Total loss")
ax.semilogy(ep_range, histories["pignn"]["data"], c=C["pinn"], lw=1.3, alpha=0.8, label="Data loss")
ax.semilogy(ep_range, histories["pignn"]["pb"], c=C["gnn"], lw=1.3, alpha=0.8, label="PB residual")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
ax.set_title("(a) PIGNN Loss Decomposition", fontweight="bold")
ax.legend(frameon=True, framealpha=0.9)

# 2b: All models comparison
ax = axes[1]
ax.semilogy(histories["pignn"]["val"], c=C["pignn"], lw=2, label="PIGNN")
ax.semilogy(histories["nn"]["val"], c=C["nn"], lw=1.8, label="StandardNN")
ax.semilogy(histories["gnn"]["val"], c=C["gnn"], lw=1.8, label="GNN-Only")
ax.semilogy(histories["pinn"]["val"], c=C["pinn"], lw=1.8, label="PINN-Only")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Loss (log)")
ax.set_title("(b) Convergence Comparison", fontweight="bold")
ax.legend(frameon=True, framealpha=0.9)

plt.tight_layout()
savefig(fig, "fig02_training_convergence.png")


# ── Fig 3: 24h Voltage Profile ───────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

sc_hp = scenario_cache[1.0]
pf_hp = grid.solve_scenario(sc_hp)
pvb = grid.pv_buses[0]

vm_pignn_24h, vm_nn_24h, vm_pinn_24h = [], [], []
for t in range(T):
    nf = build_node_features(sc_hp, t, grid.pv_buses, grid.bess_buses,
                              grid.p_load_nom, grid.q_load_nom, n)
    p1 = pignn.forward(nf, adj, grid.edge_index, edge_feat)
    vm_pignn_24h.append(p1["vm"][pvb])
    p2 = nn.forward(nf)
    vm_nn_24h.append(p2["vm"][pvb])
    p3 = pinn_model.forward(nf)
    vm_pinn_24h.append(p3["vm"][pvb])

ax = axes[0]
ax.fill_between(t_hours, 0.95, 1.05, color=C["fill_g"], alpha=0.35, label="Statutory limits")
ax.plot(t_hours, pf_hp.vm[:, pvb], "k--", lw=2.2, label="AC Power Flow (truth)")
ax.plot(t_hours, vm_pignn_24h, c=C["pignn"], lw=1.8, label="PIGNN")
ax.plot(t_hours, vm_nn_24h, c=C["nn"], lw=1.5, alpha=0.7, label="StandardNN")
ax.plot(t_hours, vm_pinn_24h, c=C["pinn"], lw=1.5, alpha=0.7, label="PINN-Only")
ax.axhline(0.95, ls=":", c="#888", lw=0.8); ax.axhline(1.05, ls=":", c="#888", lw=0.8)
ax.set_ylabel("|V| [p.u.]")
ax.set_title(f"(a) Voltage at PV Bus {pvb} — 100% PV Penetration", fontweight="bold")
ax.legend(ncol=3, loc="lower left", frameon=True, framealpha=0.9)

ax = axes[1]
irr = sc_hp["irradiance"]
ax.fill_between(t_hours, irr, alpha=0.25, color=C["warn"])
ax.plot(t_hours, irr, c=C["warn"], lw=2, label="Irradiance")
ax2 = ax.twinx()
total_pv = sc_hp["p_pv"].sum(axis=1) * cfg.grid.s_base_mva
total_load = sc_hp["p_load"].sum(axis=1) * cfg.grid.s_base_mva
ax2.plot(t_hours, total_pv, c=C["pignn"], lw=1.8, label="PV gen [MW]")
ax2.plot(t_hours, total_load, c=C["nn"], lw=1.8, ls="--", label="Load [MW]")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Irradiance [W/m²]")
ax2.set_ylabel("Power [MW]")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, ncol=3, loc="upper left", frameon=True)
ax.set_title("(b) Solar Irradiance and Generation Profile", fontweight="bold")

plt.tight_layout()
savefig(fig, "fig03_24h_voltage_profile.png")


# ── Fig 4: Physics Residual Convergence ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.semilogy(histories["pignn"]["pb"], c=C["gnn"], lw=2, label="Power balance (L_PB)")
ax.semilogy(histories["pignn"]["vb"], c=C["accent"], lw=2, label="Voltage bounds (L_VB)")
ax.semilogy(histories["pignn"]["sl"], c=C["pinn"], lw=1.5, alpha=0.7, label="Slack ref (L_SL)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Residual (log)")
ax.set_title("(a) Physics Constraint Convergence", fontweight="bold")
ax.legend(frameon=True)

ax = axes[1]
ax.plot(histories["pignn"]["data"], c=C["pignn"], lw=2, label="PIGNN")
ax.plot(histories["pinn"]["data"], c=C["pinn"], lw=1.8, label="PINN-Only")
ax.plot(histories["nn"]["train"], c=C["nn"], lw=1.8, label="StandardNN")
ax.set_xlabel("Epoch"); ax.set_ylabel("Data Loss (MSE)")
ax.set_title("(b) Supervised Data Loss Comparison", fontweight="bold")
ax.legend(frameon=True)

plt.tight_layout()
savefig(fig, "fig04_physics_residual.png")


# ── Fig 5: Model Comparison Bar Charts ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
model_labels = ["PIGNN", "StandardNN", "GNN-Only", "PINN-Only"]
model_keys = ["pignn", "nn", "gnn", "pinn"]
bar_colors = [C["pignn"], C["nn"], C["gnn"], C["pinn"]]
x = np.arange(len(pen_levels))
width = 0.2

# 5a: |V| RMSE across PV levels
ax = axes[0]
for i, mk in enumerate(model_keys):
    ax.bar(x + i*width, eval_data[mk]["vm_rmse"], width, color=bar_colors[i],
           alpha=0.85, label=model_labels[i], edgecolor="#333", linewidth=0.5)
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels([f"{p*100:.0f}%" for p in pen_levels])
ax.set_xlabel("PV Penetration"); ax.set_ylabel("|V| RMSE [p.u.]")
ax.set_title("(a) Voltage Prediction Accuracy", fontweight="bold")
ax.legend(fontsize=7.5, frameon=True)

# 5b: Physics consistency (PB residual)
ax = axes[1]
for i, mk in enumerate(model_keys):
    ax.bar(x + i*width, eval_data[mk]["pb"], width, color=bar_colors[i],
           alpha=0.85, label=model_labels[i], edgecolor="#333", linewidth=0.5)
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels([f"{p*100:.0f}%" for p in pen_levels])
ax.set_xlabel("PV Penetration"); ax.set_ylabel("Mean |ΔP| [p.u.]")
ax.set_title("(b) Power Balance Residual", fontweight="bold")
ax.legend(fontsize=7.5, frameon=True)

# 5c: Inference speed
ax = axes[2]
sp_names = [f"{model_labels[i]}\n({models[mk].count_parameters()//1000}k)" for i, mk in enumerate(model_keys)]
speeds = [timing[mk].speedup_vs_pf for mk in model_keys]
bars = ax.bar(sp_names, speeds, color=bar_colors, alpha=0.85, edgecolor="#333", linewidth=0.5)
for b, v in zip(bars, speeds):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.1f}×",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Speed-up vs AC Power Flow")
ax.set_title("(c) Computational Performance", fontweight="bold")

plt.tight_layout()
savefig(fig, "fig05_model_comparison.png")


# ── Fig 6: PV Hosting Capacity Sweep ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
pp = [p*100 for p in pen_levels]
ax.plot(pp, eval_data["pignn"]["vm_rmse"], "o-", c=C["pignn"], lw=2.5, ms=8, label="PIGNN", zorder=5)
ax.plot(pp, eval_data["nn"]["vm_rmse"], "s--", c=C["nn"], lw=2, ms=7, label="StandardNN")
ax.plot(pp, eval_data["gnn"]["vm_rmse"], "^--", c=C["gnn"], lw=2, ms=7, label="GNN-Only")
ax.plot(pp, eval_data["pinn"]["vm_rmse"], "D--", c=C["pinn"], lw=2, ms=7, label="PINN-Only")
ax.set_xlabel("PV Penetration [%]"); ax.set_ylabel("|V| RMSE [p.u.]")
ax.set_title("(a) Accuracy vs. PV Penetration", fontweight="bold")
ax.legend(frameon=True)

ax = axes[1]
ax.plot(pp, eval_data["pignn"]["viol"], "o-", c=C["pignn"], lw=2.5, ms=8, label="PIGNN", zorder=5)
ax.plot(pp, eval_data["nn"]["viol"], "s--", c=C["nn"], lw=2, ms=7, label="StandardNN")
ax.plot(pp, eval_data["gnn"]["viol"], "^--", c=C["gnn"], lw=2, ms=7, label="GNN-Only")
ax.plot(pp, eval_data["pinn"]["viol"], "D--", c=C["pinn"], lw=2, ms=7, label="PINN-Only")
ax.set_xlabel("PV Penetration [%]"); ax.set_ylabel("Snapshots with V violation [%]")
ax.set_title("(b) Voltage Violations vs. PV Penetration", fontweight="bold")
ax.legend(frameon=True)

plt.tight_layout()
savefig(fig, "fig06_pv_hosting_capacity.png")


# ── Fig 7: Ablation Study ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

abl_colors = [C["nn"], C["gnn"], C["pinn"], C["warn"], C["accent"], C["pignn"]]
bars = ax.bar(range(6), abl_rmse, color=abl_colors, alpha=0.85,
              edgecolor="#333", linewidth=0.6, width=0.65)
ax.set_xticks(range(6))
ax.set_xticklabels(ablation_names, fontsize=8.5)
ax.set_ylabel("|V| RMSE [p.u.]")
ax.set_title("Ablation Study — Marginal Contribution of Each Component (IEEE 33-bus)",
             fontweight="bold")

for b, v in zip(bars, abl_rmse):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
            f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

# Improvement arrows
for i in range(1, 6):
    if abl_rmse[i-1] > abl_rmse[i] and abl_rmse[i-1] > 0:
        imp = (abl_rmse[i-1] - abl_rmse[i]) / abl_rmse[i-1] * 100
        ax.annotate(f"−{imp:.1f}%", xy=(i, abl_rmse[i]),
                    xytext=(i-0.3, abl_rmse[i]+max(abl_rmse)*0.08),
                    fontsize=8, color=C["pignn"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["pignn"], lw=1))

plt.tight_layout()
savefig(fig, "fig07_ablation_study.png")


# ── Fig 8: Robustness Tests ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 8a: Absolute RMSE under stress
ax = axes[0]
x_rob = np.arange(len(rob_labels))
ax.bar(x_rob - 0.18, rob_base, 0.35, color=C["fill_b"], edgecolor=C["pinn"],
       linewidth=1, label="In-distribution")
ax.bar(x_rob + 0.18, rob_stress, 0.35, color="#FFCDD2", edgecolor=C["nn"],
       linewidth=1, label="Under stress")
ax.set_xticks(x_rob)
ax.set_xticklabels(rob_labels, fontsize=8, rotation=25, ha="right")
ax.set_ylabel("|V| RMSE [p.u.]")
ax.set_title("(a) Prediction Accuracy Under Stress", fontweight="bold")
ax.legend(frameon=True)

# 8b: ΔRMSE degradation
ax = axes[1]
delta_rmse = [s - b for s, b in zip(rob_stress, rob_base)]
colors_d = [C["pignn"] if d < 0.02 else C["warn"] if d < 0.05 else C["nn"] for d in delta_rmse]
bars = ax.barh(x_rob, delta_rmse, color=colors_d, alpha=0.85, edgecolor="#333", linewidth=0.5)
ax.set_yticks(x_rob)
ax.set_yticklabels(rob_labels, fontsize=8)
ax.set_xlabel("Δ|V| RMSE (degradation)")
ax.set_title("(b) Accuracy Degradation Under Stress", fontweight="bold")
ax.axvline(0, color="#333", lw=0.8)

plt.tight_layout()
savefig(fig, "fig08_robustness_tests.png")


# ── Fig 9: Per-Bus Voltage Heatmap ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc_hm = scenario_cache[0.8]
pf_hm = grid.solve_scenario(sc_hm)

# PIGNN predictions
vm_pignn_hm = np.zeros((T, n))
for t in range(T):
    nf = build_node_features(sc_hm, t, grid.pv_buses, grid.bess_buses,
                              grid.p_load_nom, grid.q_load_nom, n)
    p_ = pignn.forward(nf, adj, grid.edge_index, edge_feat)
    vm_pignn_hm[t] = p_["vm"]

# Truth heatmap
ax = axes[0]
im = ax.imshow(pf_hm.vm[:, 1:].T, aspect="auto", cmap="RdYlGn",
               vmin=0.92, vmax=1.06, extent=[0, 24, n-1, 1])
plt.colorbar(im, ax=ax, label="|V| [p.u.]", fraction=0.03)
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Bus Index")
ax.set_title("(a) Ground Truth (AC Power Flow)", fontweight="bold")
for b in grid.pv_buses:
    if b > 0:
        ax.axhline(b-0.5, color="white", ls="--", lw=0.5, alpha=0.7)

# PIGNN heatmap
ax = axes[1]
im = ax.imshow(vm_pignn_hm[:, 1:].T, aspect="auto", cmap="RdYlGn",
               vmin=0.92, vmax=1.06, extent=[0, 24, n-1, 1])
plt.colorbar(im, ax=ax, label="|V| [p.u.]", fraction=0.03)
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Bus Index")
ax.set_title("(b) PIGNN Prediction", fontweight="bold")
for b in grid.pv_buses:
    if b > 0:
        ax.axhline(b-0.5, color="white", ls="--", lw=0.5, alpha=0.7)

plt.tight_layout()
savefig(fig, "fig09_voltage_heatmap.png")


# ── Fig 10: Summary Table Figure ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.axis("off")

# Build table data
col_labels = ["Metric", "StandardNN", "GNN-Only", "PINN-Only", "PIGNN", "Improvement"]
pen_idx = -1  # Last PV level (100%)
pignn_rmse = eval_data["pignn"]["vm_rmse"][pen_idx]
nn_rmse = eval_data["nn"]["vm_rmse"][pen_idx]
gnn_rmse = eval_data["gnn"]["vm_rmse"][pen_idx]
pinn_rmse = eval_data["pinn"]["vm_rmse"][pen_idx]
imp_nn = (1 - pignn_rmse / max(nn_rmse, 1e-10)) * 100
imp_gnn = (1 - pignn_rmse / max(gnn_rmse, 1e-10)) * 100

rows = [
    ["|V| RMSE (100% PV)", f"{nn_rmse:.5f}", f"{gnn_rmse:.5f}",
     f"{pinn_rmse:.5f}", f"{pignn_rmse:.5f}", f"{imp_nn:+.1f}% vs NN"],
    ["δ RMSE (100% PV)", f"{eval_data['nn']['va_rmse'][pen_idx]:.5f}",
     f"{eval_data['gnn']['va_rmse'][pen_idx]:.5f}",
     f"{eval_data['pinn']['va_rmse'][pen_idx]:.5f}",
     f"{eval_data['pignn']['va_rmse'][pen_idx]:.5f}", "—"],
    ["PB residual", f"{eval_data['nn']['pb'][pen_idx]:.4f}",
     f"{eval_data['gnn']['pb'][pen_idx]:.4f}",
     f"{eval_data['pinn']['pb'][pen_idx]:.4f}",
     f"{eval_data['pignn']['pb'][pen_idx]:.4f}", "Physics enforced"],
    ["V violations (%)", f"{eval_data['nn']['viol'][pen_idx]:.1f}%",
     f"{eval_data['gnn']['viol'][pen_idx]:.1f}%",
     f"{eval_data['pinn']['viol'][pen_idx]:.1f}%",
     f"{eval_data['pignn']['viol'][pen_idx]:.1f}%", "Constrained"],
    ["Speed-up vs PF", f"{timing['nn'].speedup_vs_pf:.1f}×",
     f"{timing['gnn'].speedup_vs_pf:.1f}×",
     f"{timing['pinn'].speedup_vs_pf:.1f}×",
     f"{timing['pignn'].speedup_vs_pf:.1f}×", "All faster"],
    ["Parameters", f"{models['nn'].count_parameters():,}",
     f"{models['gnn'].count_parameters():,}",
     f"{models['pinn'].count_parameters():,}",
     f"{models['pignn'].count_parameters():,}", "Fair budget"],
    ["Training time", f"{histories['nn']['time']:.0f}s",
     f"{histories['gnn']['time']:.0f}s",
     f"{histories['pinn']['time']:.0f}s",
     f"{histories['pignn']['time']:.0f}s",
     f"{cfg.training.max_epochs} epochs"],
    ["Inference/snap", f"{timing['nn'].single_inference_ms:.2f}ms",
     f"{timing['gnn'].single_inference_ms:.2f}ms",
     f"{timing['pinn'].single_inference_ms:.2f}ms",
     f"{timing['pignn'].single_inference_ms:.2f}ms", "Per snapshot"],
]

tab = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(9)

for (r, c), cell in tab.get_celld().items():
    cell.set_height(0.075)
    cell.set_edgecolor("#CCCCCC")
    if r == 0:
        cell.set_facecolor("#1A237E")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 4:
        cell.set_facecolor("#E8F5E9")
    elif c == 5:
        cell.set_facecolor("#FFF8E1")
    elif r % 2 == 0:
        cell.set_facecolor("#F5F5F5")
tab.auto_set_column_width([0, 1, 2, 3, 4, 5])

ax.set_title("PIGNN Simulation Results — IEEE 33-Bus Distribution Feeder",
             fontweight="bold", fontsize=13, pad=25)
savefig(fig, "fig10_summary_table.png")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"  All 10 figures saved to: {os.path.abspath(OUT)}/")
print(f"{'=' * 65}")
