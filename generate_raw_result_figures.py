#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "results_raw_figures"
os.makedirs(OUT, exist_ok=True)

vm_pred = np.load("results_raw/vm_pred.npy")
va_pred = np.load("results_raw/va_pred.npy")
vm_true = np.load("results_raw/vm_true.npy")
va_true = np.load("results_raw/va_true.npy")

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close()
    print("Saved:", f"{OUT}/{name}.png")

# Ensure shape is samples x buses
if vm_pred.ndim == 1:
    vm_pred = vm_pred[None, :]
    vm_true = vm_true[None, :]
    va_pred = va_pred[None, :]
    va_true = va_true[None, :]

n_samples, n_bus = vm_pred.shape
buses = np.arange(1, n_bus + 1)
samples = np.arange(n_samples)

# 1. Voltage profile averaged over test set
plt.figure(figsize=(10, 5))
plt.plot(buses, vm_true.mean(axis=0), marker="o", linewidth=2, label="Ground truth")
plt.plot(buses, vm_pred.mean(axis=0), marker="s", linewidth=2, label="PIGNN prediction")
plt.xlabel("Bus number")
plt.ylabel("Voltage magnitude (p.u.)")
plt.title("IEEE 33-Bus Average Voltage Profile")
plt.grid(True, linewidth=0.3)
plt.legend()
savefig("fig01_average_voltage_profile")

# 2. Voltage profile for first test snapshot
plt.figure(figsize=(10, 5))
plt.plot(buses, vm_true[0], marker="o", linewidth=2, label="Ground truth")
plt.plot(buses, vm_pred[0], marker="s", linewidth=2, label="PIGNN prediction")
plt.xlabel("Bus number")
plt.ylabel("Voltage magnitude (p.u.)")
plt.title("IEEE 33-Bus Voltage Profile: Test Snapshot 1")
plt.grid(True, linewidth=0.3)
plt.legend()
savefig("fig02_snapshot_voltage_profile")

# 3. Voltage prediction error by bus
vm_error_bus = np.sqrt(np.mean((vm_pred - vm_true) ** 2, axis=0))
plt.figure(figsize=(10, 5))
plt.bar(buses, vm_error_bus)
plt.xlabel("Bus number")
plt.ylabel("Voltage RMSE (p.u.)")
plt.title("Per-Bus Voltage Prediction Error")
plt.grid(True, axis="y", linewidth=0.3)
savefig("fig03_per_bus_voltage_error")

# 4. Error heatmap: samples x buses
plt.figure(figsize=(11, 5))
plt.imshow(np.abs(vm_pred - vm_true), aspect="auto", origin="lower")
plt.colorbar(label="|Voltage error| (p.u.)")
plt.xlabel("Bus number")
plt.ylabel("Test sample")
plt.title("Voltage Prediction Error Heatmap")
plt.xticks(np.arange(n_bus), buses, fontsize=7)
savefig("fig04_voltage_error_heatmap")

# 5. Average voltage over samples
plt.figure(figsize=(10, 5))
plt.plot(samples, vm_true.mean(axis=1), linewidth=2, label="Ground truth")
plt.plot(samples, vm_pred.mean(axis=1), linewidth=2, label="PIGNN prediction")
plt.xlabel("Test sample")
plt.ylabel("Average voltage magnitude (p.u.)")
plt.title("Average Feeder Voltage Across Test Samples")
plt.grid(True, linewidth=0.3)
plt.legend()
savefig("fig05_average_voltage_over_samples")

# 6. Angle diagnostic only, because angle scale is still suspicious
va_rmse = np.sqrt(np.mean((va_pred - va_true) ** 2))
plt.figure(figsize=(9, 4.5))
plt.axis("off")
plt.text(
    0.02, 0.95,
    f"Voltage-angle diagnostic\n\n"
    f"Raw angle RMSE from saved arrays: {va_rmse:.4f} rad\n\n"
    "If this value is very large, the angle target/output scaling is still wrong.\n"
    "For IEEE paper figures, use voltage-magnitude plots until angle scaling is corrected.",
    va="top",
    fontsize=11
)
savefig("fig06_angle_diagnostic")

print("Done. Raw simulation figures saved in:", OUT)
print("Loaded shapes:")
print("vm_pred:", vm_pred.shape)
print("vm_true:", vm_true.shape)
print("va_pred:", va_pred.shape)
print("va_true:", va_true.shape)

