"""
scenarios.py — Scenario Generation Pipeline
============================================
Section III-B of the manuscript.

Generates 24-hour operating scenarios for training, validation, and
testing.  Each scenario contains time-series profiles for:
  - Bus-level active and reactive loads
  - PV irradiance and active/reactive generation
  - BESS charge/discharge schedule (exogenous, rule-based)

Also generates out-of-distribution (OOD) stress-test scenarios with
elevated PV, increased loads, and measurement noise.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import (
    ExperimentConfig, ScenarioConfig, PVConfig, BESSConfig, GridConfig,
)


# ──────────────────────────────────────────────────────────────────────
# Profile Generators
# ──────────────────────────────────────────────────────────────────────

def generate_solar_irradiance(
    T: int,
    rng: np.random.RandomState,
    noise_std: float = 0.04,
) -> np.ndarray:
    """
    Synthesise a 24-hour irradiance profile [W/m²].

    A truncated Gaussian bell centred at solar noon (12:30) with σ = 3.2 h
    is modulated by correlated cloud noise (3-point moving-average filter).

    Parameters
    ----------
    T : int
        Number of timesteps in the day.
    rng : RandomState
        Reproducible random generator.
    noise_std : float
        Cloud-noise intensity as a fraction of 1000 W/m².

    Returns
    -------
    ndarray, shape (T,)
        Irradiance profile clipped to [0, 1100] W/m².
    """
    t_hours = np.linspace(0, 24, T)
    clear_sky = 1000.0 * np.exp(-0.5 * ((t_hours - 12.5) / 3.2) ** 2)
    cloud_noise = np.convolve(
        rng.normal(0, noise_std * 200, T), np.ones(3) / 3, mode="same"
    )
    return np.clip(clear_sky + cloud_noise, 0, 1100)


def generate_temperature(
    T: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Generate a daily ambient temperature profile [°C].

    Follows a sinusoidal pattern: min ≈ 14 °C at dawn, max ≈ 34 °C in
    the afternoon, with small additive Gaussian noise (σ = 0.5 °C).
    """
    t_hours = np.linspace(0, 24, T)
    return 20.0 + 10.0 * np.sin(np.pi * (t_hours - 6) / 18) + rng.normal(0, 0.5, T)


def generate_load_multiplier(
    T: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Generate a stochastic daily load multiplier profile.

    Double-Gaussian residential pattern with morning (09:00) and evening
    (19:30) peaks, plus additive noise σ = 0.015 p.u.
    """
    t_hours = np.linspace(0, 24, T)
    morning = 0.25 * np.exp(-0.5 * ((t_hours - 9.0) / 2.0) ** 2)
    evening = 0.35 * np.exp(-0.5 * ((t_hours - 19.5) / 2.5) ** 2)
    base = 0.40 + morning + evening + rng.normal(0, 0.015, T)
    return np.clip(base, 0.25, 1.15)


def compute_pv_power(
    irradiance: np.ndarray,
    temperature: np.ndarray,
    penetration: float,
    pv_buses: List[int],
    n_bus: int,
    pv_cfg: PVConfig,
    s_base: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Compute per-bus PV active power injection [p.u.].

    Includes temperature-dependent derating and small stochastic noise.

    Returns
    -------
    ndarray, shape (T, n_bus)
    """
    T = len(irradiance)
    p_pv = np.zeros((T, n_bus))
    base_power = (irradiance / 1000.0) * pv_cfg.peak_mw / s_base
    temp_derating = np.clip(1.0 + pv_cfg.temp_coeff * (temperature - 25.0), 0.7, 1.0)

    for bus in pv_buses:
        p_pv[:, bus] = np.clip(
            base_power * temp_derating * penetration + rng.normal(0, 0.001, T),
            0, None,
        )
    return p_pv


def compute_bess_schedule(
    T: int,
    irradiance: np.ndarray,
    bess_cfg: BESSConfig,
    bess_buses: List[int],
    n_bus: int,
    s_base: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rule-based BESS charge–discharge schedule (Section III-B).

    - Charges during peak irradiance (10:00–15:00) when PV surplus exists.
    - Discharges during evening peak (18:00–22:00).
    - SOC bounded between soc_min and soc_max.
    - BESS power is treated as an exogenous input (not optimised).

    Returns
    -------
    p_bess : ndarray, shape (T, n_bus)
        Signed active power: positive = discharge, negative = charge [p.u.].
    soc : ndarray, shape (T,)
        State-of-charge trajectory.
    """
    p_bess_total = np.zeros(T)
    soc = np.zeros(T + 1)
    soc[0] = bess_cfg.soc_initial

    if not bess_cfg.enabled:
        return np.zeros((T, n_bus)), np.ones(T) * bess_cfg.soc_initial

    p_max_pu = bess_cfg.p_max_mw / s_base
    cap_pu = bess_cfg.capacity_mwh / s_base
    dt_hours = 24.0 / T

    for t in range(T):
        hour = t * 24.0 / T

        if 10.0 <= hour <= 15.0 and irradiance[t] > 400:
            # Charge: absorb surplus PV
            p_charge = min(p_max_pu, (bess_cfg.soc_max - soc[t]) * cap_pu)
            p_bess_total[t] = -p_charge  # negative = charging
            soc[t + 1] = soc[t] + p_charge * bess_cfg.efficiency * dt_hours / cap_pu

        elif 18.0 <= hour <= 22.0:
            # Discharge: support evening peak
            p_discharge = min(p_max_pu, (soc[t] - bess_cfg.soc_min) * cap_pu)
            p_bess_total[t] = p_discharge  # positive = discharging
            soc[t + 1] = soc[t] - p_discharge / bess_cfg.efficiency * dt_hours / cap_pu

        else:
            soc[t + 1] = soc[t]

        soc[t + 1] = np.clip(soc[t + 1], bess_cfg.soc_min, bess_cfg.soc_max)

    # Distribute across BESS buses
    p_bess = np.zeros((T, n_bus))
    n_bess = max(len(bess_buses), 1)
    for bus in bess_buses:
        p_bess[:, bus] = p_bess_total / n_bess

    return p_bess, soc[:T]


# ──────────────────────────────────────────────────────────────────────
# Scenario Assembly
# ──────────────────────────────────────────────────────────────────────

def generate_single_scenario(
    penetration: float,
    n_bus: int,
    p_load_nom: np.ndarray,
    q_load_nom: np.ndarray,
    pv_buses: List[int],
    bess_buses: List[int],
    cfg: ExperimentConfig,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """
    Generate one complete 24-hour scenario.

    Returns a dictionary with keys: 'p_load', 'q_load', 'p_pv', 'q_pv',
    'p_bess', 'soc', 'irradiance', 'temperature', 'load_mult',
    'penetration', 't_hours'.
    """
    T = cfg.scenario.timesteps_per_day

    # Stochastic profiles
    irradiance = generate_solar_irradiance(T, rng, cfg.pv.noise_std)
    temperature = generate_temperature(T, rng)
    load_mult = generate_load_multiplier(T, rng)

    # 30-day upgrade: seasonal irradiance scaling (winter-to-summer)
    seasonal_factor = rng.uniform(0.6, 1.0)
    irradiance = irradiance * seasonal_factor

    # 30-day upgrade: weekday/weekend load variation (30% weekend)
    is_weekend = rng.random() < 0.3
    if is_weekend:
        t_hours_wd = np.linspace(0, 24, T)
        morning_wd = 0.20 * np.exp(-0.5 * ((t_hours_wd - 11.0) / 2.5) ** 2)
        evening_wd = 0.30 * np.exp(-0.5 * ((t_hours_wd - 20.0) / 2.5) ** 2)
        load_mult = 0.38 + morning_wd + evening_wd + rng.normal(0, 0.015, T)
        load_mult = np.clip(load_mult, 0.25, 1.10)

    # Bus-level loads
    p_load = np.outer(load_mult, p_load_nom)  # (T, n_bus)
    q_load = np.outer(load_mult, q_load_nom)

    # PV generation
    p_pv = compute_pv_power(
        irradiance, temperature, penetration,
        pv_buses, n_bus, cfg.pv, cfg.grid.s_base_mva, rng,
    )
    q_pv = p_pv * 0.05  # Near-unity power factor for PV inverters

    # Stochastic reactive power factor per scenario (Section III-B):
    # "Reactive loads are scaled with a fixed power factor uniformly
    #  drawn from [0.85, 0.97] per scenario."
    pf_scenario = rng.uniform(0.85, 0.97)
    tan_phi = np.sqrt(1 - pf_scenario**2) / pf_scenario
    q_load = p_load * tan_phi  # Override nominal Q with stochastic PF

    # BESS schedule
    p_bess, soc = compute_bess_schedule(
        T, irradiance, cfg.bess, bess_buses, n_bus, cfg.grid.s_base_mva,
    )

    return {
        "p_load": p_load, "q_load": q_load,
        "p_pv": p_pv, "q_pv": q_pv,
        "p_bess": p_bess, "soc": soc,
        "irradiance": irradiance, "temperature": temperature,
        "load_mult": load_mult, "penetration": penetration,
        "t_hours": np.linspace(0, 24, T),
    }


# ──────────────────────────────────────────────────────────────────────
# Full Dataset Generation
# ──────────────────────────────────────────────────────────────────────

def generate_dataset(
    grid,
    cfg: ExperimentConfig,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Generate the full dataset with stratified train/val/test split.

    Each PV penetration level is represented equally across splits to
    ensure balanced coverage (stratified sampling).

    Parameters
    ----------
    grid : DistributionGrid
        The test system providing topology and nominal loads.
    cfg : ExperimentConfig
        Full experiment configuration.
    seed : int
        Master random seed.

    Returns
    -------
    dict with keys 'train', 'val', 'test', 'ood', each a list of
    scenario dictionaries.
    """
    rng = np.random.RandomState(seed)
    pen_levels = cfg.pv.penetration_levels
    n_total = cfg.scenario.n_scenarios
    n_per_level = n_total // len(pen_levels)

    all_scenarios = []
    for pen in pen_levels:
        for _ in range(n_per_level):
            sc = generate_single_scenario(
                penetration=pen,
                n_bus=grid.n_bus,
                p_load_nom=grid.p_load_nom,
                q_load_nom=grid.q_load_nom,
                pv_buses=grid.pv_buses,
                bess_buses=grid.bess_buses,
                cfg=cfg,
                rng=rng,
            )
            all_scenarios.append(sc)

    # Stratified split
    rng.shuffle(all_scenarios)
    n_train = int(len(all_scenarios) * cfg.scenario.train_ratio)
    n_val = int(len(all_scenarios) * cfg.scenario.val_ratio)

    dataset = {
        "train": all_scenarios[:n_train],
        "val": all_scenarios[n_train:n_train + n_val],
        "test": all_scenarios[n_train + n_val:],
    }

    # OOD scenarios (Section III-B)
    ood_scenarios = []
    for _ in range(cfg.scenario.n_ood_scenarios):
        pen_mult = rng.choice(cfg.scenario.ood_pv_multipliers)
        sc = generate_single_scenario(
            penetration=pen_mult,
            n_bus=grid.n_bus,
            p_load_nom=grid.p_load_nom * cfg.scenario.ood_load_multiplier,
            q_load_nom=grid.q_load_nom * cfg.scenario.ood_load_multiplier,
            pv_buses=grid.pv_buses,
            bess_buses=grid.bess_buses,
            cfg=cfg,
            rng=rng,
        )
        # Add measurement noise
        sigma = rng.choice(cfg.scenario.ood_noise_levels)
        for key in ["p_pv", "q_pv", "p_load", "q_load"]:
            sc[key] = sc[key] + rng.normal(0, sigma, sc[key].shape)
            if "pv" in key:
                sc[key] = np.clip(sc[key], 0, None)
        sc["ood"] = True
        sc["ood_pv_mult"] = pen_mult
        sc["ood_noise_sigma"] = sigma
        ood_scenarios.append(sc)

    dataset["ood"] = ood_scenarios

    if cfg.verbose:
        print(f"  Dataset generated: "
              f"train={len(dataset['train'])}, "
              f"val={len(dataset['val'])}, "
              f"test={len(dataset['test'])}, "
              f"ood={len(dataset['ood'])}")

    return dataset
