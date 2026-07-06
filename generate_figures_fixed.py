#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt

os.makedirs("results_fixed", exist_ok=True)

# -----------------------------
# Final simulation metrics
# -----------------------------
models = ["PIGNN", "StandardNN", "GNN-Only", "PINN-Only"]

v_rmse = [0.345876, 0.401464, 0.240251, 0.342355]
pb_resid = [1.490058, 2.615444, 7.683616, 0.605011]
v_viol = [11.1, 100.0, 100.0, 0.0]
speedup = [372.3, 13678.0, 441.8, 9346.5]

# -----------------------------
# Helper
# -----------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(f"results_fixed/{name}.png", dpi=300)
    plt.savefig(f"results_fixed/{name}.pdf")
    plt.close()
    print(f"Saved: results_fixed/{name}.png")

# -----------------------------
# Figure 1: IEEE 33-bus topology
# -----------------------------
edges = [
    (1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
    (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),
    (2,19),(19,20),(20,21),(21,22),
    (3,23),(23,24),(24,25),
    (6,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33)
]

pos = {}
for i in range(1, 19):
    pos[i] = (i, 0)

pos.update({
    19:(2,-1.5), 20:(3,-1.5), 21:(4,-1.5), 22:(5,-1.5),
    23:(3,1.5), 24:(4,1.5), 25:(5,1.5),
    26:(6,-3), 27:(7,-3), 28:(8,-3), 29:(9,-3),
    30:(10,-3), 31:(11,-3), 32:(12,-3), 33:(13,-3)
})

pv_buses = [2, 7, 12, 17, 22, 27]
bess_buses = [2, 18]

plt.figure(figsize=(12, 4.5))
for a, b in edges:
    xa, ya = pos[a]
    xb, yb = pos[b]
    plt.plot([xa, xb], [ya, yb], linewidth=1.6)

for bus in range(1, 34):
    x, y = pos[bus]
    if bus == 1:
        marker, size = "s", 180   # slack
    elif bus in bess_buses:
        marker, size = "D", 140   # BESS
    elif bus in pv_buses:
        marker, size = "^", 140   # PV
    else:
        marker, size = "o", 80    # normal bus

    plt.scatter(x, y, s=size, marker=marker, edgecolors="black")
    plt.text(x, y + 0.22, str(bus), ha="center", fontsize=8)

plt.title("IEEE 33-Bus Feeder Topology (Simulation View)")
plt.xlabel("Bus order")
plt.ylabel("Radial branches")
plt.grid(True, linewidth=0.3)
plt.axis("equal")
savefig("fig01_ieee33_topology_fixed")

# -----------------------------
# Figure 2: Voltage RMSE
# -----------------------------
plt.figure(figsize=(8, 4.5))
plt.bar(models, v_rmse)
plt.ylabel("|V| RMSE (p.u.)")
plt.title("Voltage Magnitude RMSE Comparison")
plt.grid(True, axis="y", linewidth=0.3)
for i, v in enumerate(v_rmse):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
savefig("fig02_voltage_rmse")

# -----------------------------
# Figure 3: Power-balance residual
# -----------------------------
plt.figure(figsize=(8, 4.5))
plt.bar(models, pb_resid)
plt.ylabel("Mean Power-Balance Residual (p.u.)")
plt.title("Physics Consistency Comparison")
plt.grid(True, axis="y", linewidth=0.3)
for i, v in enumerate(pb_resid):
    plt.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
savefig("fig03_power_balance_residual")

# -----------------------------
# Figure 4: Voltage violations
# -----------------------------
plt.figure(figsize=(8, 4.5))
plt.bar(models, v_viol)
plt.ylabel("Voltage Violations (%)")
plt.title("Voltage Constraint Violations")
plt.ylim(0, 110)
plt.grid(True, axis="y", linewidth=0.3)
for i, v in enumerate(v_viol):
    plt.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=8)
savefig("fig04_voltage_violations")

# -----------------------------
# Figure 5: Speed-up
# -----------------------------
plt.figure(figsize=(8, 4.5))
plt.bar(models, speedup)
plt.yscale("log")
plt.ylabel("Speed-up vs Power Flow Solver")
plt.title("Jetson Inference Speed-up")
plt.grid(True, axis="y", linewidth=0.3)
for i, v in enumerate(speedup):
    plt.text(i, v * 1.1, f"{v:.0f}x", ha="center", fontsize=8)
savefig("fig05_speedup")

# -----------------------------
# Figure 6: Angle diagnostic note
# -----------------------------
plt.figure(figsize=(9, 4.5))
plt.axis("off")
plt.text(
    0.02, 0.95,
    "Angle scaling diagnostic\n\n"
    "The simulation reported:\n"
    "δ RMSE ≈ 394 rad\n"
    "δ MAE_max ≈ 9003 rad\n\n"
    "These values are physically unrealistic for\n"
    "distribution-system voltage angles.\n\n"
    "Therefore, raw angle plots are excluded here.\n"
    "The angle target/output scaling must be fixed\n"
    "before using δ figures in the IEEE paper.",
    va="top",
    fontsize=11
)
savefig("fig06_angle_scaling_note")

print("All fixed figures generated successfully in results_fixed/")#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt

os.makedirs("results_fixed", exist_ok=True)

models = ["PIGNN", "StandardNN", "GNN-Only", "PINN-Only"]
v_rmse = [0.345876, 0.401464, 0.240251, 0.342355]

plt.figure(figsize=(8, 4.5))
plt.bar(models, v_rmse)
plt.ylabel("|V| RMSE (p.u.)")
plt.title("Voltage Magnitude RMSE Comparison")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("results_fixed/test_voltage_rmse.png", dpi=300)
plt.savefig("results_fixed/test_voltage_rmse.pdf")
plt.close()

print("Figure generated successfully in results_fixed/")


