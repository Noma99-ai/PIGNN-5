#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt

OUT = "results_more"
os.makedirs(OUT, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close()
    print("Saved:", f"{OUT}/{name}.png")

# -------------------------------------------------
# Completed simulation results from your terminal
# -------------------------------------------------
models = ["PIGNN", "StandardNN", "GNN-Only", "PINN-Only"]

v_rmse = [0.345876, 0.401464, 0.240251, 0.342355]
pb_resid = [1.490058, 2.615444, 7.683616, 0.605011]
v_viol = [11.1, 100.0, 100.0, 0.0]
speedup = [372.3, 13678.0, 441.8, 9346.5]
train_time = [1511.4, 770.8, 1435.4, 835.6]

robust_tests = [
    "PV 120%", "PV 150%", "Load x1.3",
    "Noise 1%", "Noise 3%", "Noise 5%", "N-1 line 3"
]
robust_vrmse = [0.335498, 0.332867, 0.430395, 0.294891, 0.230345, 0.174353, 0.471161]

ablation_names = ["A1-NoGraph", "A2-NoPhysics", "A3-NoConstr", "A4-NoEdgeFeat", "A5-K1"]
ablation_vrmse = [0.404660, 0.334663, 1.426610, 0.343550, 0.344203]
ablation_pb = [2.609507, 1.679951, 25.083717, 0.998562, 1.726588]

# -------------------------------------------------
# 1. Voltage RMSE line graph
# -------------------------------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(models, v_rmse, marker="o", linewidth=2)
plt.ylabel("|V| RMSE (p.u.)")
plt.title("Voltage Magnitude RMSE by Model")
plt.grid(True, linewidth=0.3)
for i, v in enumerate(v_rmse):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
savefig("fig01_voltage_rmse_line")

# -------------------------------------------------
# 2. Power-balance residual line graph
# -------------------------------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(models, pb_resid, marker="o", linewidth=2)
plt.ylabel("Mean Power-Balance Residual (p.u.)")
plt.title("Physics Consistency by Model")
plt.grid(True, linewidth=0.3)
for i, v in enumerate(pb_resid):
    plt.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)
savefig("fig02_power_balance_line")

# -------------------------------------------------
# 3. Voltage-violation line graph
# -------------------------------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(models, v_viol, marker="o", linewidth=2)
plt.ylabel("Voltage Violations (%)")
plt.title("Voltage Constraint Violations by Model")
plt.ylim(0, 110)
plt.grid(True, linewidth=0.3)
for i, v in enumerate(v_viol):
    plt.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=8)
savefig("fig03_voltage_violations_line")

# -------------------------------------------------
# 4. Speed-up line graph (log scale)
# -------------------------------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(models, speedup, marker="o", linewidth=2)
plt.yscale("log")
plt.ylabel("Speed-up vs PF Solver")
plt.title("Inference Speed-up by Model")
plt.grid(True, linewidth=0.3)
for i, v in enumerate(speedup):
    plt.text(i, v * 1.08, f"{v:.0f}x", ha="center", fontsize=8)
savefig("fig04_speedup_line")

# -------------------------------------------------
# 5. Training time graph
# -------------------------------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(models, train_time, marker="o", linewidth=2)
plt.ylabel("Training Time (s)")
plt.title("Training Time by Model")
plt.grid(True, linewidth=0.3)
for i, v in enumerate(train_time):
    plt.text(i, v + 30, f"{v:.1f}s", ha="center", fontsize=8)
savefig("fig05_training_time_line")

# -------------------------------------------------
# 6. Robustness graph
# -------------------------------------------------
plt.figure(figsize=(10, 4.8))
plt.plot(robust_tests, robust_vrmse, marker="o", linewidth=2)
plt.ylabel("|V| RMSE (p.u.)")
plt.title("PIGNN Robustness Tests")
plt.grid(True, linewidth=0.3)
plt.xticks(rotation=25, ha="right")
for i, v in enumerate(robust_vrmse):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
savefig("fig06_robustness_line")

# -------------------------------------------------
# 7. Ablation graph: RMSE
# -------------------------------------------------
plt.figure(figsize=(10, 4.8))
plt.plot(ablation_names, ablation_vrmse, marker="o", linewidth=2, label="|V| RMSE")
plt.axhline(y=0.345876, linestyle="--", linewidth=1.5, label="Baseline PIGNN")
plt.ylabel("|V| RMSE (p.u.)")
plt.title("Ablation Study: Voltage RMSE")
plt.grid(True, linewidth=0.3)
plt.xticks(rotation=25, ha="right")
plt.legend()
savefig("fig07_ablation_vrmse")

# -------------------------------------------------
# 8. Ablation graph: power-balance residual
# -------------------------------------------------
plt.figure(figsize=(10, 4.8))
plt.plot(ablation_names, ablation_pb, marker="o", linewidth=2, label="PB Residual")
plt.axhline(y=1.490058, linestyle="--", linewidth=1.5, label="Baseline PIGNN")
plt.ylabel("Mean Power-Balance Residual (p.u.)")
plt.title("Ablation Study: Physics Residual")
plt.grid(True, linewidth=0.3)
plt.xticks(rotation=25, ha="right")
plt.legend()
savefig("fig08_ablation_pb")

# -------------------------------------------------
# 9. Angle diagnostic
# -------------------------------------------------
plt.figure(figsize=(9, 4.5))
plt.axis("off")
plt.text(
    0.02, 0.95,
    "Angle diagnostic\n\n"
    "The completed simulation reported very large angle errors:\n"
    "δ RMSE ≈ 394 rad and δ MAE_max ≈ 9003 rad.\n\n"
    "This strongly suggests an angle scaling/evaluation issue.\n"
    "Therefore, raw angle-result graphs are intentionally excluded\n"
    "until the angle target/output scaling is corrected.",
    va="top",
    fontsize=11
)
savefig("fig09_angle_diagnostic")

print("All additional result graphs saved in:", OUT)


