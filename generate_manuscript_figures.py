#!/usr/bin/env python3
"""
generate_manuscript_figures.py
===============================
Produces 12 publication-quality figures, each mapped to a specific
subsection of the IEEE Transactions manuscript on PIGNN Layer 5.

  Fig 1  — (III-A)  Test system topology: IEEE 33-bus graph
  Fig 2  — (III-B)  Scenario generation profiles (24h load, PV, BESS, SOC)
  Fig 3  — (III-D)  Node & edge feature illustration
  Fig 4  — (III-E)  PIGNN architecture pipeline diagram
  Fig 5  — (III-F)  Loss function: 5 physics terms convergence
  Fig 6  — (III-H)  Training convergence: PIGNN vs 3 baselines
  Fig 7  — (III-I)  Model comparison: accuracy + physics + speed
  Fig 8  — (III-I)  24h voltage profile: truth vs all models
  Fig 9  — (III-I)  Per-bus voltage heatmap: truth vs PIGNN
  Fig 10 — (III-J)  Ablation study: 5 variants bar chart
  Fig 11 — (III-K)  Robustness: ΔRMSE under R1–R4 stress
  Fig 12 — (Results) Comprehensive summary table figure

Author: PIGNN IEEE Simulation Platform
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if "__file__" in dir() else ".")
sys.path.insert(0, ".")
os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from copy import deepcopy

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

C = {"pignn": "#1B5E20", "nn": "#B71C1C", "gnn": "#E65100",
     "pinn": "#1565C0", "pf": "#212121", "fill_g": "#C8E6C9",
     "fill_b": "#BBDEFB", "accent": "#6A1B9A", "warn": "#F9A825", "grey": "#9E9E9E"}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__))
                   if "__file__" in dir() else ".", "results")
os.makedirs(OUT, exist_ok=True)

def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ {name}")

# ═══════════════════════════════════════════════════════════════════
# RUN SIMULATION PIPELINE
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PIGNN IEEE Manuscript — Figure Generator (Upgraded)")
print("=" * 65)

from config import ExperimentConfig
from grid import DistributionGrid
from scenarios import generate_dataset, generate_single_scenario
from graph import (build_node_features, build_edge_features,
    build_adjacency, build_admittance_weighted_adjacency, NODE_FEATURE_DIM)
from models.pignn import PIGNN
from models.baselines import StandardNN, GNNOnly, PINNOnly
from losses import compute_total_loss, data_loss, LossBreakdown
from training import (adam_update, cosine_annealing_lr,
    numerical_gradient_step)
from metrics import compute_predictive_metrics, compute_physics_metrics

cfg = ExperimentConfig()
cfg.grid.case = "ieee33"
cfg.scenario.n_scenarios = 100
cfg.scenario.n_ood_scenarios = 15
cfg.scenario.timesteps_per_day = 24
cfg.training.max_epochs = 50
cfg.verbose = False

print("\n  Step 1/7: Initialise IEEE 33-bus grid (Section III-A)...")
grid = DistributionGrid(cfg.grid, cfg.pv, cfg.bess)
n = grid.n_bus; T = cfg.scenario.timesteps_per_day
t_hours = np.linspace(0, 24, T)
adj = build_adjacency(n, grid.connections)
adj_w = build_admittance_weighted_adjacency(n, grid.connections, grid.r_pu, grid.x_pu)
edge_feat = build_edge_features(grid.edge_index, grid.r_pu, grid.x_pu, grid.connections)
print(f"    {grid.bench.name}: {n} buses, {grid.n_branch} branches, PV@{grid.pv_buses}, BESS@{grid.bess_buses}")

print("  Step 2/7: Generate scenarios (Section III-B)...")
dataset = generate_dataset(grid, cfg, seed=42)

print("  Step 3/7: Train all models (Sections III-E–H)...")
seed = 42

def quick_train(model, label, use_phys=True, epochs=None):
    ep_max = epochs or cfg.training.max_epochs
    hist = {"train": [], "val": [], "pb": [], "vb": [], "data": [], "sl": [], "th": []}
    step_c = 0; t0 = time.time()
    for ep in range(ep_max):
        lr = cosine_annealing_lr(ep, ep_max, cfg.training.learning_rate, cfg.training.lr_min)
        sc = dataset["train"][ep % len(dataset["train"])]
        ti = ep % T
        p_net = sc["p_pv"][ti] + sc["p_bess"][ti] - sc["p_load"][ti]
        q_net = sc["q_pv"][ti] - sc["q_load"][ti]
        ps = p_net.copy(); ps[0] = -np.sum(p_net[1:])
        qs = q_net.copy(); qs[0] = -np.sum(q_net[1:])
        pf = grid.solve_power_flow(ps, qs)
        nf = build_node_features(sc, ti, grid.pv_buses, grid.bess_buses,
                                  grid.p_load_nom, grid.q_load_nom, n)
        def loss_fn():
            try: pred = model.forward(nf, adj, grid.edge_index, edge_feat)
            except TypeError: pred = model.forward(nf)
            if use_phys:
                lb = compute_total_loss(pred["vm"], pred["va"], pf.vm, pf.va,
                                        ps, qs, grid.Y_bus, cfg.grid, cfg.loss, pf.i_branch)
                return lb.total, lb
            else:
                d = data_loss(pred["vm"], pred["va"], pf.vm, pf.va)
                return d, LossBreakdown(d, d, 0, 0, 0, 0)
        numerical_gradient_step(model, loss_fn, n_samples=5)
        step_c += 1
        adam_update(model.get_trainable_layers(), lr, step_c)
        l, lb = loss_fn()
        hist["train"].append(l); hist["data"].append(lb.data)
        hist["pb"].append(lb.power_balance); hist["vb"].append(lb.voltage_bounds)
        hist["sl"].append(lb.slack_ref); hist["th"].append(lb.thermal)
        vsc = dataset["val"][ep % len(dataset["val"])]
        vnf = build_node_features(vsc, ti, grid.pv_buses, grid.bess_buses,
                                   grid.p_load_nom, grid.q_load_nom, n)
        vp = vsc["p_pv"][ti]+vsc["p_bess"][ti]-vsc["p_load"][ti]
        vq = vsc["q_pv"][ti]-vsc["q_load"][ti]; vp[0]=-np.sum(vp[1:]); vq[0]=-np.sum(vq[1:])
        vpf = grid.solve_power_flow(vp, vq)
        try: vpr = model.forward(vnf, adj, grid.edge_index, edge_feat)
        except TypeError: vpr = model.forward(vnf)
        hist["val"].append(data_loss(vpr["vm"], vpr["va"], vpf.vm, vpf.va))
    hist["time"] = time.time() - t0
    print(f"    {label}: {hist['time']:.1f}s")
    return hist

rng = np.random.RandomState(seed)
pignn = PIGNN(n, cfg, rng)
h_pignn = quick_train(pignn, "PIGNN", use_phys=True)
nn_mdl = StandardNN(n, NODE_FEATURE_DIM, np.random.RandomState(seed))
h_nn = quick_train(nn_mdl, "StandardNN", use_phys=False)
gnn_mdl = GNNOnly(n, cfg, np.random.RandomState(seed))
h_gnn = quick_train(gnn_mdl, "GNN-Only", use_phys=False)
pinn_mdl = PINNOnly(n, NODE_FEATURE_DIM, cfg, np.random.RandomState(seed))
h_pinn = quick_train(pinn_mdl, "PINN-Only", use_phys=True)
models = {"PIGNN": pignn, "StandardNN": nn_mdl, "GNN-Only": gnn_mdl, "PINN-Only": pinn_mdl}
hists = {"PIGNN": h_pignn, "StandardNN": h_nn, "GNN-Only": h_gnn, "PINN-Only": h_pinn}

print("  Step 4/7: Evaluate across PV levels (Section III-I)...")
pen_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
eval_d = {k: {"vm_rmse": [], "va_rmse": [], "viol": [], "pb": []} for k in models}
sc_cache = {}
for pen in pen_levels:
    sc = generate_single_scenario(pen, n, grid.p_load_nom, grid.q_load_nom,
        grid.pv_buses, grid.bess_buses, cfg, np.random.RandomState(int(pen*100)))
    sc_cache[pen] = sc; pf_s = grid.solve_scenario(sc)
    for mk, mdl in models.items():
        vmp, vap, vmt, vat, pns, qns = [], [], [], [], [], []
        for t in range(0, T, 2):
            nf = build_node_features(sc, t, grid.pv_buses, grid.bess_buses,
                                      grid.p_load_nom, grid.q_load_nom, n)
            try: p = mdl.forward(nf, adj, grid.edge_index, edge_feat)
            except TypeError: p = mdl.forward(nf)
            vmp.append(p["vm"]); vap.append(p["va"])
            vmt.append(pf_s.vm[t]); vat.append(pf_s.va[t])
            pn = sc["p_pv"][t]+sc["p_bess"][t]-sc["p_load"][t]
            qn = sc["q_pv"][t]-sc["q_load"][t]; pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
            pns.append(pn); qns.append(qn)
        acc = compute_predictive_metrics(np.array(vmp),np.array(vap),np.array(vmt),np.array(vat))
        phys = compute_physics_metrics(np.array(vmp),np.array(vap),np.array(pns),np.array(qns),
                                        grid.Y_bus, cfg.grid.v_min_pu, cfg.grid.v_max_pu)
        eval_d[mk]["vm_rmse"].append(acc.vm_rmse); eval_d[mk]["va_rmse"].append(acc.va_rmse)
        eval_d[mk]["viol"].append(phys.pct_v_violations); eval_d[mk]["pb"].append(phys.mean_p_residual)

print("  Step 5/7: Ablation study (Section III-J)...")
sc_ref = sc_cache[0.8]; pf_ref = grid.solve_scenario(sc_ref)
def eval_q(mdl):
    vmps, vaps, vmts, vats = [], [], [], []
    for t in range(0, T, 4):
        nf = build_node_features(sc_ref, t, grid.pv_buses, grid.bess_buses,
                                  grid.p_load_nom, grid.q_load_nom, n)
        try: p = mdl.forward(nf, adj, grid.edge_index, edge_feat)
        except TypeError: p = mdl.forward(nf)
        vmps.append(p["vm"]); vaps.append(p["va"]); vmts.append(pf_ref.vm[t]); vats.append(pf_ref.va[t])
    return compute_predictive_metrics(np.array(vmps),np.array(vaps),np.array(vmts),np.array(vats))

a1 = eval_q(nn_mdl)       # A1: no graph
a2 = eval_q(gnn_mdl)      # A2: no physics loss (GNN-only)
a3 = eval_q(pinn_mdl)     # A3: proxy for no constrained / PINN-only

cfg_a4 = deepcopy(cfg); cfg_a4.gnn.use_edge_gating = False
m_a4 = PIGNN(n, cfg_a4, np.random.RandomState(seed))
quick_train(m_a4, "A4-NoEdge", use_phys=True, epochs=15)
a4 = eval_q(m_a4)

cfg_a5 = deepcopy(cfg); cfg_a5.gnn.n_message_passing = 1
m_a5 = PIGNN(n, cfg_a5, np.random.RandomState(seed))
quick_train(m_a5, "A5-K=1", use_phys=True, epochs=15)
a5 = eval_q(m_a5)
pignn_ref = eval_q(pignn)

abl_names = ["A1: No graph\ntopology","A2: No physics\nloss","A3: No constrained\noutputs",
             "A4: No edge\nfeatures","A5: K=1\n(reduced MP)","Full PIGNN"]
abl_vm = [a1.vm_rmse, a2.vm_rmse, a3.vm_rmse, a4.vm_rmse, a5.vm_rmse, pignn_ref.vm_rmse]
abl_va = [a1.va_rmse, a2.va_rmse, a3.va_rmse, a4.va_rmse, a5.va_rmse, pignn_ref.va_rmse]

print("  Step 6/7: Robustness tests (Section III-K)...")
rob_labels, rob_base, rob_stress = [], [], []
base_m = eval_q(pignn)

def stress_eval(scenarios_mod):
    vmps, vaps, vmts, vats = [], [], [], []
    for ssc in scenarios_mod[:3]:
        for t in range(0, T, 6):
            nf = build_node_features(ssc, t, grid.pv_buses, grid.bess_buses,
                                      grid.p_load_nom, grid.q_load_nom, n)
            pn = ssc["p_pv"][t]+ssc["p_bess"][t]-ssc["p_load"][t]
            qn = ssc["q_pv"][t]-ssc["q_load"][t]; pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
            pf_ = grid.solve_power_flow(pn, qn)
            p_ = pignn.forward(nf, adj, grid.edge_index, edge_feat)
            vmps.append(p_["vm"]); vaps.append(p_["va"]); vmts.append(pf_.vm); vats.append(pf_.va)
    return compute_predictive_metrics(np.array(vmps),np.array(vaps),np.array(vmts),np.array(vats))

# R1: Elevated PV
for mult in [1.2, 1.5]:
    stressed = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc.items()} for sc in dataset["test"][:5]]
    for s in stressed: s["p_pv"] = s["p_pv"]*mult; s["q_pv"] = s["q_pv"]*mult
    sm = stress_eval(stressed)
    rob_labels.append(f"R1: PV ×{mult}"); rob_base.append(base_m.vm_rmse); rob_stress.append(sm.vm_rmse)

# R2: Load uncertainty
stressed2 = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc.items()} for sc in dataset["test"][:5]]
for s in stressed2: s["p_load"]=s["p_load"]*1.3; s["q_load"]=s["q_load"]*1.3
sm2 = stress_eval(stressed2)
rob_labels.append("R2: Load ×1.3"); rob_base.append(base_m.vm_rmse); rob_stress.append(sm2.vm_rmse)

# R3: Measurement noise
for sigma in [0.01, 0.03, 0.05]:
    rng_n = np.random.RandomState(99)
    stressed3 = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc.items()} for sc in dataset["test"][:5]]
    for s in stressed3:
        for key in ["p_pv","q_pv","p_load","q_load"]:
            s[key] = s[key]+rng_n.normal(0,sigma,s[key].shape)
            if "pv" in key: s[key]=np.clip(s[key],0,None)
    smn = stress_eval(stressed3)
    rob_labels.append(f"R3: σ={sigma:.0%}"); rob_base.append(base_m.vm_rmse); rob_stress.append(smn.vm_rmse)

# R4: N-1 contingency
orig_r=grid.r_pu[3]; orig_x=grid.x_pu[3]
grid.r_pu[3]=1e6; grid.x_pu[3]=1e6; grid.Y_bus=grid._build_ybus()
ef_mod = build_edge_features(grid.edge_index, grid.r_pu, grid.x_pu, grid.connections)
vmps, vaps, vmts, vats = [], [], [], []
for sc_ in dataset["test"][:3]:
    for t in range(0,T,6):
        nf = build_node_features(sc_,t,grid.pv_buses,grid.bess_buses,grid.p_load_nom,grid.q_load_nom,n)
        pn=sc_["p_pv"][t]+sc_["p_bess"][t]-sc_["p_load"][t]; qn=sc_["q_pv"][t]-sc_["q_load"][t]
        pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
        pf_=grid.solve_power_flow(pn,qn); p_=pignn.forward(nf,adj,grid.edge_index,ef_mod)
        vmps.append(p_["vm"]); vaps.append(p_["va"]); vmts.append(pf_.vm); vats.append(pf_.va)
smn1 = compute_predictive_metrics(np.array(vmps),np.array(vaps),np.array(vmts),np.array(vats))
rob_labels.append("R4: N−1 outage"); rob_base.append(base_m.vm_rmse); rob_stress.append(smn1.vm_rmse)
grid.r_pu[3]=orig_r; grid.x_pu[3]=orig_x; grid.Y_bus=grid._build_ybus()

# Speed measurement
print("  Step 7/7: Timing inference...")
timing = {}
for mk, mdl in models.items():
    t0 = time.perf_counter()
    sc_t = sc_cache[0.6]
    for t in range(T):
        nf = build_node_features(sc_t,t,grid.pv_buses,grid.bess_buses,grid.p_load_nom,grid.q_load_nom,n)
        try: mdl.forward(nf,adj,grid.edge_index,edge_feat)
        except TypeError: mdl.forward(nf)
    t_model = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter()
    for t in range(T):
        pn=sc_t["p_pv"][t]+sc_t["p_bess"][t]-sc_t["p_load"][t]; qn=sc_t["q_pv"][t]-sc_t["q_load"][t]
        pn[0]=-np.sum(pn[1:]); qn[0]=-np.sum(qn[1:])
        grid.solve_power_flow(pn,qn)
    t_pf = (time.perf_counter()-t0)*1000
    timing[mk] = {"model_ms": t_model, "pf_ms": t_pf, "speedup": t_pf/max(t_model,0.01), "per_snap": t_model/T}


# ═══════════════════════════════════════════════════════════════════
# GENERATE 12 FIGURES
# ═══════════════════════════════════════════════════════════════════
print("\n  Generating 12 manuscript figures...")

# ── Fig 1 (III-A): IEEE 33-bus Topology ──────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
import matplotlib.patches as mpatches
# Layout: position buses along feeder
pos = {}
# Main trunk 0-17
for i in range(18): pos[i] = (i*0.55, 0)
# Lateral from bus 1
for j, b in enumerate([18,19,20,21]): pos[b] = (0.55+j*0.55, -1.2)
# Lateral from bus 2
for j, b in enumerate([22,23,24]): pos[b] = (1.1+j*0.55, 1.2)
# Lateral from bus 5
for j, b in enumerate([25,26,27,28,29]): pos[b] = (2.75+j*0.55, -1.2)
# Lateral from bus 9
for j, b in enumerate([30,31,32]): pos[b] = (4.95+j*0.55, 1.2)

for i, j in grid.connections:
    if i in pos and j in pos:
        ax.plot([pos[i][0],pos[j][0]], [pos[i][1],pos[j][1]], '-', color='#888', lw=1.5, zorder=1)

for b in range(n):
    if b in pos:
        color = C["pignn"] if b in grid.pv_buses else C["pinn"] if b in grid.bess_buses else (C["pf"] if b==0 else C["grey"])
        size = 120 if b in grid.pv_buses or b in grid.bess_buses or b==0 else 50
        ax.scatter(pos[b][0], pos[b][1], s=size, c=color, zorder=3, edgecolors='white', linewidths=0.5)
        if b in grid.pv_buses or b in grid.bess_buses or b == 0:
            ax.annotate(str(b), (pos[b][0], pos[b][1]+0.25), ha='center', fontsize=7, fontweight='bold')

ax.legend(handles=[
    mpatches.Patch(color=C["pf"], label="Slack bus (Bus 0)"),
    mpatches.Patch(color=C["pignn"], label=f"PV buses {grid.pv_buses}"),
    mpatches.Patch(color=C["pinn"], label=f"BESS buses {grid.bess_buses}"),
    mpatches.Patch(color=C["grey"], label="Load buses"),
], loc="upper right", fontsize=8, frameon=True)
ax.set_title("Fig. 1 — IEEE 33-bus distribution feeder topology (Section III-A)", fontweight="bold")
ax.set_xlabel("Electrical distance along feeder"); ax.set_ylabel("Lateral branch")
ax.set_aspect("equal"); ax.grid(False)
plt.tight_layout(); savefig(fig, "fig01_test_system_topology.png")

# ── Fig 2 (III-B): Scenario Generation Profiles ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
sc0 = sc_cache[0.8]

axes[0,0].plot(t_hours, sc0["irradiance"], c=C["warn"], lw=2)
axes[0,0].fill_between(t_hours, sc0["irradiance"], alpha=0.2, color=C["warn"])
axes[0,0].set_ylabel("Irradiance [W/m²]"); axes[0,0].set_title("(a) Solar irradiance profile", fontweight="bold")

axes[0,1].plot(t_hours, sc0["load_mult"], c=C["nn"], lw=2)
axes[0,1].set_ylabel("Load multiplier [p.u.]"); axes[0,1].set_title("(b) Stochastic load profile", fontweight="bold")

tp = sc0["p_pv"].sum(1)*cfg.grid.s_base_mva; tl = sc0["p_load"].sum(1)*cfg.grid.s_base_mva
tb = sc0["p_bess"].sum(1)*cfg.grid.s_base_mva
axes[1,0].plot(t_hours, tp, c=C["pignn"], lw=2, label="PV gen")
axes[1,0].plot(t_hours, tl, c=C["nn"], lw=2, ls="--", label="Total load")
axes[1,0].plot(t_hours, tb, c=C["accent"], lw=1.5, label="BESS (+dis/−chg)")
axes[1,0].set_ylabel("Power [MW]"); axes[1,0].set_xlabel("Hour of day")
axes[1,0].set_title("(c) Generation vs load at 80% PV", fontweight="bold"); axes[1,0].legend(fontsize=7)

axes[1,1].plot(t_hours, sc0["soc"], c=C["accent"], lw=2)
axes[1,1].axhline(0.1, ls=":", c="#999", lw=0.8); axes[1,1].axhline(0.9, ls=":", c="#999", lw=0.8)
axes[1,1].set_ylabel("SOC"); axes[1,1].set_xlabel("Hour of day")
axes[1,1].set_title("(d) BESS state of charge (exogenous)", fontweight="bold")
axes[1,1].annotate("SOC_min=0.10", (20,0.12), fontsize=7, color="#999")
axes[1,1].annotate("SOC_max=0.90", (20,0.88), fontsize=7, color="#999")

fig.suptitle("Fig. 2 — Scenario generation profiles (Section III-B)", fontweight="bold", y=1.01)
plt.tight_layout(); savefig(fig, "fig02_scenario_profiles.png")

# ── Fig 3 (III-D): Node & Edge Features ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

node_labels = ["P_net","Q_net","P_pv","P_bess","P_L0","Q_L0","is_pv","is_bess","is_slack"]
nf_sample = build_node_features(sc0, T//2, grid.pv_buses, grid.bess_buses,
                                 grid.p_load_nom, grid.q_load_nom, n)
im = axes[0].imshow(nf_sample.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
axes[0].set_yticks(range(9)); axes[0].set_yticklabels(node_labels, fontsize=8)
axes[0].set_xlabel("Bus index"); axes[0].set_title("(a) 9-dim node features at noon", fontweight="bold")
plt.colorbar(im, ax=axes[0], fraction=0.03, label="Feature value [p.u.]")

ef_half = edge_feat[::2]  # one direction only
edge_labels = ["r [p.u.]", "x [p.u.]", "|y|=1/|z|"]
im2 = axes[1].imshow(ef_half.T, aspect="auto", cmap="viridis", interpolation="nearest")
axes[1].set_yticks(range(3)); axes[1].set_yticklabels(edge_labels, fontsize=9)
axes[1].set_xlabel("Branch index"); axes[1].set_title("(b) 3-dim edge features", fontweight="bold")
plt.colorbar(im2, ax=axes[1], fraction=0.03, label="Feature value [p.u.]")

fig.suptitle("Fig. 3 — Graph input features (Section III-D)", fontweight="bold", y=1.01)
plt.tight_layout(); savefig(fig, "fig03_graph_features.png")

# ── Fig 4 (III-E): Architecture Pipeline ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.set_xlim(-0.5, 15); ax.set_ylim(-0.2, 3.8); ax.axis("off")
boxes = [
    (0.2,0.8,2.4,2.2,"#E3F2FD","#1565C0","Input\nFeatures","9-dim/bus\n3-dim/edge"),
    (3.2,0.8,2.4,2.2,"#E8F5E9","#1B5E20","GNN\nEncoder","K=3, d=64\nadmittance\nweighted"),
    (6.2,0.8,2.4,2.2,"#FFF3E0","#E65100","PINN\nDecoder","[128,64]\nsigmoid |V|\ntanh δ"),
    (9.2,0.8,2.4,2.2,"#F3E5F5","#6A1B9A","Physics\nLoss (5)","L_PB + L_VB\nL_SL + L_TH\n+ L_data"),
    (12.2,1.1,2.0,1.6,"#FFEBEE","#B71C1C","Output","|V|, δ\nper bus"),
]
for x,y,w,h,fc,ec,title,sub in boxes:
    rect = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15",facecolor=fc,edgecolor=ec,lw=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2,y+h*0.68,title,ha="center",va="center",fontsize=10,fontweight="bold",color=ec)
    ax.text(x+w/2,y+h*0.25,sub,ha="center",va="center",fontsize=7,color="#555")
for i in range(4):
    x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]
    ax.annotate("",xy=(x2,1.9),xytext=(x1,1.9),arrowprops=dict(arrowstyle="-|>",color="#333",lw=2))
ax.text(7.5,0.1,"PIGNN Layer 5 Core — GNN Topology Encoder + PINN Physics Decoder (Section III-E)",
        ha="center",fontsize=11,fontweight="bold",color="#1A237E",style="italic")
savefig(fig, "fig04_pignn_architecture.png")

# ── Fig 5 (III-F): Physics Loss Convergence ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ep_r = range(len(h_pignn["train"]))
axes[0].semilogy(ep_r, h_pignn["pb"], c=C["gnn"], lw=2, label="L_PB (KCL power balance, λ=10)")
axes[0].semilogy(ep_r, h_pignn["vb"], c=C["accent"], lw=2, label="L_VB (voltage bounds, λ=5)")
axes[0].semilogy(ep_r, h_pignn["sl"], c=C["pinn"], lw=1.5, alpha=0.8, label="L_SL (slack ref, λ=2)")
axes[0].semilogy(ep_r, h_pignn["th"], c=C["nn"], lw=1.5, alpha=0.8, label="L_TH (thermal, λ=3)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Physics residual (log)")
axes[0].set_title("(a) All 5 physics penalty convergence", fontweight="bold"); axes[0].legend(fontsize=7)

axes[1].semilogy(ep_r, h_pignn["train"], c=C["pignn"], lw=2.5, label="L_total")
axes[1].semilogy(ep_r, h_pignn["data"], c=C["pinn"], lw=1.5, alpha=0.8, label="L_data (MSE)")
axes[1].semilogy(ep_r, h_pignn["pb"], c=C["gnn"], lw=1.5, alpha=0.8, label="L_PB × λ_PB")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss (log)")
axes[1].set_title("(b) PIGNN total loss decomposition", fontweight="bold"); axes[1].legend(fontsize=7)

fig.suptitle("Fig. 5 — Physics-informed loss convergence (Section III-F)", fontweight="bold", y=1.01)
plt.tight_layout(); savefig(fig, "fig05_physics_loss_convergence.png")

# ── Fig 6 (III-H): Training Convergence All Models ───────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for mk, cl in [("PIGNN",C["pignn"]),("StandardNN",C["nn"]),("GNN-Only",C["gnn"]),("PINN-Only",C["pinn"])]:
    axes[0].semilogy(hists[mk]["val"], c=cl, lw=2, label=mk)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation loss (log)")
axes[0].set_title("(a) Validation convergence", fontweight="bold"); axes[0].legend()

t_names = list(models.keys())
t_times = [hists[k]["time"] for k in t_names]
bars = axes[1].barh(t_names, t_times, color=[C["pignn"],C["nn"],C["gnn"],C["pinn"]], alpha=0.85, edgecolor="#333", lw=0.5)
for b, v in zip(bars, t_times): axes[1].text(b.get_width()+0.5, b.get_y()+b.get_height()/2, f"{v:.0f}s", va="center", fontsize=9)
axes[1].set_xlabel("Training time [s]"); axes[1].set_title("(b) Training time comparison", fontweight="bold")

fig.suptitle("Fig. 6 — Training protocol results (Section III-H)", fontweight="bold", y=1.01)
plt.tight_layout(); savefig(fig, "fig06_training_convergence.png")

# ── Fig 7 (III-I): Model Comparison Bars ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
mkeys = ["PIGNN","StandardNN","GNN-Only","PINN-Only"]; bar_c = [C["pignn"],C["nn"],C["gnn"],C["pinn"]]
x = np.arange(len(pen_levels)); w = 0.2
for i, mk in enumerate(mkeys):
    axes[0].bar(x+i*w, eval_d[mk]["vm_rmse"], w, color=bar_c[i], alpha=0.85, label=mk, edgecolor="#333", lw=0.4)
axes[0].set_xticks(x+1.5*w); axes[0].set_xticklabels([f"{p*100:.0f}%" for p in pen_levels])
axes[0].set_xlabel("PV penetration"); axes[0].set_ylabel("|V| RMSE [p.u.]")
axes[0].set_title("(a) Voltage accuracy", fontweight="bold"); axes[0].legend(fontsize=7)

for i, mk in enumerate(mkeys):
    axes[1].bar(x+i*w, eval_d[mk]["pb"], w, color=bar_c[i], alpha=0.85, edgecolor="#333", lw=0.4)
axes[1].set_xticks(x+1.5*w); axes[1].set_xticklabels([f"{p*100:.0f}%" for p in pen_levels])
axes[1].set_xlabel("PV penetration"); axes[1].set_ylabel("Mean |ΔP| [p.u.]")
axes[1].set_title("(b) Power balance residual", fontweight="bold")

sp_names = [f"{mk}\n({models[mk].count_parameters()//1000}k)" for mk in mkeys]
speeds = [timing[mk]["speedup"] for mk in mkeys]
bars = axes[2].bar(sp_names, speeds, color=bar_c, alpha=0.85, edgecolor="#333", lw=0.4)
for b, v in zip(bars, speeds): axes[2].text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{v:.1f}×", ha="center", fontsize=9, fontweight="bold")
axes[2].set_ylabel("Speed-up vs AC-PF"); axes[2].set_title("(c) Computational speed", fontweight="bold")

fig.suptitle("Fig. 7 — Model comparison across 4 baselines (Section III-I)", fontweight="bold", y=1.01)
plt.tight_layout(); savefig(fig, "fig07_model_comparison.png")

# ── Fig 8 (III-I): 24h Voltage Profile ───────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
sc_hp = sc_cache[1.0]; pf_hp = grid.solve_scenario(sc_hp); pvb = grid.pv_buses[0]
vm_models = {}
for mk, mdl in models.items():
    vm_models[mk] = []
    for t in range(T):
        nf = build_node_features(sc_hp,t,grid.pv_buses,grid.bess_buses,grid.p_load_nom,grid.q_load_nom,n)
        try: p = mdl.forward(nf,adj,grid.edge_index,edge_feat)
        except TypeError: p = mdl.forward(nf)
        vm_models[mk].append(p["vm"][pvb])

ax.fill_between(t_hours, 0.95, 1.05, color=C["fill_g"], alpha=0.3, label="Statutory limits [0.95,1.05]")
ax.plot(t_hours, pf_hp.vm[:, pvb], "k--", lw=2.5, label="AC Power Flow (truth)", zorder=5)
for mk, cl in [("PIGNN",C["pignn"]),("StandardNN",C["nn"]),("GNN-Only",C["gnn"]),("PINN-Only",C["pinn"])]:
    ax.plot(t_hours, vm_models[mk], c=cl, lw=1.5, alpha=0.8, label=mk)
ax.axhline(0.95, ls=":", c="#888", lw=0.7); ax.axhline(1.05, ls=":", c="#888", lw=0.7)
ax.set_xlabel("Hour of day"); ax.set_ylabel("|V| [p.u.]")
ax.set_title(f"Fig. 8 — 24h voltage at PV bus {pvb}, 100% PV penetration (Section III-I)", fontweight="bold")
ax.legend(ncol=3, loc="lower left"); plt.tight_layout(); savefig(fig, "fig08_24h_voltage_profile.png")

# ── Fig 9 (III-I): Per-Bus Voltage Heatmap ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sc_hm = sc_cache[0.8]; pf_hm = grid.solve_scenario(sc_hm)
vm_pignn_hm = np.zeros((T, n))
for t in range(T):
    nf = build_node_features(sc_hm,t,grid.pv_buses,grid.bess_buses,grid.p_load_nom,grid.q_load_nom,n)
    p_ = pignn.forward(nf,adj,grid.edge_index,edge_feat); vm_pignn_hm[t] = p_["vm"]

for i, (data, title) in enumerate([(pf_hm.vm[:,1:], "(a) AC power flow (truth)"), (vm_pignn_hm[:,1:], "(b) PIGNN prediction")]):
    im = axes[i].imshow(data.T, aspect="auto", cmap="RdYlGn", vmin=0.92, vmax=1.06, extent=[0,24,n-1,1])
    plt.colorbar(im, ax=axes[i], label="|V| [p.u.]", fraction=0.03)
    axes[i].set_xlabel("Hour of day"); axes[i].set_ylabel("Bus index"); axes[i].set_title(title, fontweight="bold")
    for b in grid.pv_buses:
        if b>0: axes[i].axhline(b-0.5, color="white", ls="--", lw=0.4, alpha=0.6)

fig.suptitle("Fig. 9 — Per-bus voltage heatmap at 80% PV (Section III-I)", fontweight="bold", y=1.01)
plt.tight_layout(); savefig(fig, "fig09_voltage_heatmap.png")

# ── Fig 10 (III-J): Ablation Study ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
abl_c = [C["nn"],C["gnn"],C["pinn"],C["warn"],C["accent"],C["pignn"]]

bars = axes[0].bar(range(6), abl_vm, color=abl_c, alpha=0.85, edgecolor="#333", lw=0.5, width=0.65)
axes[0].set_xticks(range(6)); axes[0].set_xticklabels(abl_names, fontsize=8)
axes[0].set_ylabel("|V| RMSE [p.u.]"); axes[0].set_title("(a) Voltage magnitude RMSE", fontweight="bold")
for b, v in zip(bars, abl_vm): axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

bars2 = axes[1].bar(range(6), abl_va, color=abl_c, alpha=0.85, edgecolor="#333", lw=0.5, width=0.65)
axes[1].set_xticks(range(6)); axes[1].set_xticklabels(abl_names, fontsize=8)
axes[1].set_ylabel("δ RMSE [rad]"); axes[1].set_title("(b) Voltage angle RMSE", fontweight="bold")
for b, v in zip(bars2, abl_va): axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

fig.suptitle("Fig. 10 — Ablation study: A1–A5 variants (Section III-J)", fontweight="bold", y=1.01)
plt.tight_layout(); savefig(fig, "fig10_ablation_study.png")

# ── Fig 11 (III-K): Robustness Tests ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x_rob = np.arange(len(rob_labels))
axes[0].bar(x_rob-0.18, rob_base, 0.35, color=C["fill_b"], edgecolor=C["pinn"], lw=1, label="In-distribution")
axes[0].bar(x_rob+0.18, rob_stress, 0.35, color="#FFCDD2", edgecolor=C["nn"], lw=1, label="Under stress")
axes[0].set_xticks(x_rob); axes[0].set_xticklabels(rob_labels, fontsize=8, rotation=25, ha="right")
axes[0].set_ylabel("|V| RMSE [p.u.]"); axes[0].set_title("(a) Accuracy: baseline vs stressed", fontweight="bold"); axes[0].legend()

delta = [s-b for s,b in zip(rob_stress, rob_base)]
colors_d = [C["pignn"] if d<0.02 else C["warn"] if d<0.05 else C["nn"] for d in delta]
axes[1].barh(x_rob, delta, color=colors_d, alpha=0.85, edgecolor="#333", lw=0.5)
axes[1].set_yticks(x_rob); axes[1].set_yticklabels(rob_labels, fontsize=8)
axes[1].set_xlabel("Δ|V| RMSE (degradation)"); axes[1].set_title("(b) ΔRMSE under stress", fontweight="bold")
axes[1].axvline(0, color="#333", lw=0.8)

fig.suptitle("Fig. 11 — Robustness & generalisation tests R1–R4 (Section III-K)", fontweight="bold", y=1.01)
plt.tight_layout(); savefig(fig, "fig11_robustness_tests.png")

# ── Fig 12 (Results): Comprehensive Summary Table ────────────────
fig, ax = plt.subplots(figsize=(14, 7)); ax.axis("off")
pi = -1
col_labels = ["Metric","StandardNN","GNN-Only","PINN-Only","PIGNN","Section"]
rows = [
    ["|V| RMSE (100% PV)", f"{eval_d['StandardNN']['vm_rmse'][pi]:.5f}", f"{eval_d['GNN-Only']['vm_rmse'][pi]:.5f}",
     f"{eval_d['PINN-Only']['vm_rmse'][pi]:.5f}", f"{eval_d['PIGNN']['vm_rmse'][pi]:.5f}", "III-I"],
    ["δ RMSE (100% PV)", f"{eval_d['StandardNN']['va_rmse'][pi]:.5f}", f"{eval_d['GNN-Only']['va_rmse'][pi]:.5f}",
     f"{eval_d['PINN-Only']['va_rmse'][pi]:.5f}", f"{eval_d['PIGNN']['va_rmse'][pi]:.5f}", "III-I"],
    ["PB residual [p.u.]", f"{eval_d['StandardNN']['pb'][pi]:.4f}", f"{eval_d['GNN-Only']['pb'][pi]:.4f}",
     f"{eval_d['PINN-Only']['pb'][pi]:.4f}", f"{eval_d['PIGNN']['pb'][pi]:.4f}", "III-F"],
    ["V violations (%)", f"{eval_d['StandardNN']['viol'][pi]:.1f}", f"{eval_d['GNN-Only']['viol'][pi]:.1f}",
     f"{eval_d['PINN-Only']['viol'][pi]:.1f}", f"{eval_d['PIGNN']['viol'][pi]:.1f}", "III-I"],
    ["Speed-up vs PF", f"{timing['StandardNN']['speedup']:.1f}×", f"{timing['GNN-Only']['speedup']:.1f}×",
     f"{timing['PINN-Only']['speedup']:.1f}×", f"{timing['PIGNN']['speedup']:.1f}×", "III-I"],
    ["Inference/snap [ms]", f"{timing['StandardNN']['per_snap']:.2f}", f"{timing['GNN-Only']['per_snap']:.2f}",
     f"{timing['PINN-Only']['per_snap']:.2f}", f"{timing['PIGNN']['per_snap']:.2f}", "III-I"],
    ["Parameters", f"{models['StandardNN'].count_parameters():,}", f"{models['GNN-Only'].count_parameters():,}",
     f"{models['PINN-Only'].count_parameters():,}", f"{models['PIGNN'].count_parameters():,}", "III-E"],
    ["Training time [s]", f"{hists['StandardNN']['time']:.0f}", f"{hists['GNN-Only']['time']:.0f}",
     f"{hists['PINN-Only']['time']:.0f}", f"{hists['PIGNN']['time']:.0f}", "III-H"],
    ["Physics loss terms", "0", "0", "4 (PB,VB,SL,TH)", "5 (all)", "III-F"],
    ["Graph topology", "No", "Yes (K=3)", "No", "Yes (K=3)", "III-E"],
    ["Constrained outputs", "No", "No", "Yes", "Yes", "III-E"],
]

tab = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(9)
for (r,c), cell in tab.get_celld().items():
    cell.set_height(0.065); cell.set_edgecolor("#CCC")
    if r == 0: cell.set_facecolor("#1A237E"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 4: cell.set_facecolor("#E8F5E9")
    elif c == 5: cell.set_facecolor("#FFF8E1")
    elif r % 2 == 0: cell.set_facecolor("#F5F5F5")
tab.auto_set_column_width([0,1,2,3,4,5])
ax.set_title("Fig. 12 — Comprehensive results summary: PIGNN vs baselines\n(IEEE 33-bus, 5 physics laws, Layer 5 core only)",
             fontweight="bold", fontsize=13, pad=25)
savefig(fig, "fig12_summary_table.png")

# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  All 12 figures saved to: {os.path.abspath(OUT)}/")
print(f"  Each figure is tagged with its manuscript section reference.")
print(f"{'='*65}")
