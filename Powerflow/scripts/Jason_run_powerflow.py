#!/usr/bin/env python3
"""
Jason_run_powerflow.py
=======================
Executes UNBALANCED 3-PHASE power flow calculations using pandapower
on the translated Icelandic grid data.

Uses pp.runpp_3ph() with asymmetric loads to model phase imbalance
in the LV distribution system.

Supports two modes:
  - unbalanced: original per-phase loads (actual smart meter assignment)
  - balanced: total load divided equally across 3 phases (theoretical optimum)

Extracts I²R line losses (phase + neutral conductors) for loss analysis.

Saves per-phase results to Powerflow/output/pf_results/{substation_id}/{mode}/

Prerequisites:
  - Jason_prepare_powerflow.py must have been run first (produces per-phase parquets).

Usage:
    python Jason_run_powerflow.py [--max-timesteps N] [--aggregate] [--mode unbalanced|balanced|both]
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandapower as pp
import warnings
import time
import argparse
import re
import logging
from pathlib import Path
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PF_INPUT_DIR = PROJECT_ROOT / "Powerflow" / "output" / "pf_input"
PF_RESULTS_DIR = PROJECT_ROOT / "Powerflow" / "output" / "pf_results"

# Voltage levels (grid constants, not substation-specific)
V_LV = 0.4  # kV (line-to-line)
V_MV = 11   # kV (Icelandic distribution, from transformers.csv FORSPENNA=11000)

np.random.seed(42)


def compute_line_losses(net, line_indices):
    """
    Compute I²R losses for each line after a converged runpp_3ph().

    Phase losses:   (I_a² + I_b² + I_c²) × R_phase × L  [kW]
    Neutral losses: I_n² × R_neutral × L                  [kW]

    R_neutral = R_phase for Icelandic 4-equal-conductor cables (e.g. "4 x 50 Al").
    Currents from net.res_line_3ph in kA; R in ohm/km; L in km.

    Returns:
        phase_losses_kw: np.array of phase losses per line (kW)
        neutral_losses_kw: np.array of neutral losses per line (kW)
    """
    r_ohm_per_km = net.line.loc[line_indices, 'r_ohm_per_km'].values
    length_km = net.line.loc[line_indices, 'length_km'].values

    # Phase currents in kA
    i_a = net.res_line_3ph.loc[line_indices, 'i_a_from_ka'].values
    i_b = net.res_line_3ph.loc[line_indices, 'i_b_from_ka'].values
    i_c = net.res_line_3ph.loc[line_indices, 'i_c_from_ka'].values

    # I²R for phase conductors: sum of (I² × R × L) for each phase
    # Units: (kA)² × (ohm/km) × km = kA² × ohm
    #   1 kA² × 1 ohm = (10³ A)² × ohm = 10⁶ W = 10³ kW
    # So result in kW = I²_kA × R × L × 1000
    phase_losses_kw = (i_a**2 + i_b**2 + i_c**2) * r_ohm_per_km * length_km * 1000

    # Neutral current
    if 'i_n_from_ka' in net.res_line_3ph.columns:
        i_n = net.res_line_3ph.loc[line_indices, 'i_n_from_ka'].values
    else:
        i_n = np.zeros(len(line_indices))

    # R_neutral = R_phase (4 equal conductor cables)
    neutral_losses_kw = i_n**2 * r_ohm_per_km * length_km * 1000

    return phase_losses_kw, neutral_losses_kw


def build_balanced_loads(Ploads_sub, Qloads_sub):
    """
    Create balanced load dicts: total load per meter divided equally across 3 phases.

    For each meter at each timestep:
        P_balanced_per_phase = (P_a + P_b + P_c) / 3
        Q_balanced_per_phase = (Q_a + Q_b + Q_c) / 3

    Args:
        Ploads_sub: dict {'a': DataFrame, 'b': DataFrame, 'c': DataFrame} in MW
        Qloads_sub: dict {'a': DataFrame, 'b': DataFrame, 'c': DataFrame} in Mvar

    Returns:
        Ploads_bal, Qloads_bal: same structure with balanced values
    """
    P_total = Ploads_sub['a'] + Ploads_sub['b'] + Ploads_sub['c']
    Q_total = Qloads_sub['a'] + Qloads_sub['b'] + Qloads_sub['c']

    P_per_phase = P_total / 3.0
    Q_per_phase = Q_total / 3.0

    Ploads_bal = {'a': P_per_phase.copy(), 'b': P_per_phase.copy(), 'c': P_per_phase.copy()}
    Qloads_bal = {'a': Q_per_phase.copy(), 'b': Q_per_phase.copy(), 'c': Q_per_phase.copy()}

    return Ploads_bal, Qloads_bal


def find_reachable_buses(net, root_bus):
    """BFS from root_bus through net.line AND net.trafo to find all reachable buses."""
    visited = {root_bus}
    queue = deque([root_bus])
    while queue:
        bus = queue.popleft()
        # Neighbors via lines
        neighbors = net.line.loc[net.line.from_bus == bus, 'to_bus'].tolist()
        neighbors += net.line.loc[net.line.to_bus == bus, 'from_bus'].tolist()
        # Neighbors via transformers (hv_bus <-> lv_bus)
        neighbors += net.trafo.loc[net.trafo.hv_bus == bus, 'lv_bus'].tolist()
        neighbors += net.trafo.loc[net.trafo.lv_bus == bus, 'hv_bus'].tolist()
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


def run_powerflow_loop(net, ts_index, Ploads_run, Qloads_run, Pgens_sub, Qgens_sub,
                       buses_dict, lines_dict, loads_dict, gens_dict,
                       buses_loads, buses_loads_unique, buses_gens, buses_gens_unique,
                       aggregate, result_path, run_label, sub):
    """
    Run the time-series 3-phase power flow loop and save all results.

    This is the core simulation loop, extracted from main() to allow
    running it for both unbalanced and balanced modes.

    Args:
        net: pandapower network (loads will be overwritten each timestep)
        ts_index: DatetimeIndex of timesteps
        Ploads_run, Qloads_run: per-phase load dicts for this run
        Pgens_sub, Qgens_sub: per-phase generation dicts (same for both runs)
        buses_dict, lines_dict, loads_dict, gens_dict: network element mappings
        buses_loads, buses_loads_unique: bus-meter aggregation DataFrames
        buses_gens, buses_gens_unique: bus-generator aggregation DataFrames
        aggregate: bool, whether loads are aggregated per bus
        result_path: Path to save results
        run_label: 'unbalanced' or 'balanced'
        sub: substation ID
    """
    result_path.mkdir(parents=True, exist_ok=True)
    n_steps = len(ts_index)

    bus_keys = list(buses_dict.keys())
    line_keys = list(lines_dict.keys())
    bus_indices = list(buses_dict.values())
    line_indices = list(lines_dict.values())

    # --- Initialize result DataFrames ---
    # Per-phase bus voltages
    V_buses_a = pd.DataFrame(columns=bus_keys, index=ts_index)
    V_buses_b = pd.DataFrame(columns=bus_keys, index=ts_index)
    V_buses_c = pd.DataFrame(columns=bus_keys, index=ts_index)
    VUF = pd.DataFrame(columns=bus_keys, index=ts_index)

    # Per-phase line currents
    i_lines_a = pd.DataFrame(columns=line_keys, index=ts_index)
    i_lines_b = pd.DataFrame(columns=line_keys, index=ts_index)
    i_lines_c = pd.DataFrame(columns=line_keys, index=ts_index)
    i_lines_n = pd.DataFrame(columns=line_keys, index=ts_index)
    i_lines_perc = pd.DataFrame(columns=line_keys, index=ts_index)

    # Grid exchange
    power_grid = pd.DataFrame(columns=['P', 'Q'], index=ts_index)

    # Transformer loading + per-phase power (for validation against Janitza / smart meters)
    trafo_3ph = pd.DataFrame(
        columns=['loading_percent', 'loading_a_percent', 'loading_b_percent', 'loading_c_percent',
                 'P_a_kw', 'P_b_kw', 'P_c_kw',
                 'Q_a_kvar', 'Q_b_kvar', 'Q_c_kvar',
                 'S_a_kva', 'S_b_kva', 'S_c_kva'],
        index=ts_index
    )

    # Line losses
    loss_lines_phase = pd.DataFrame(columns=line_keys, index=ts_index, dtype=float)
    loss_lines_neutral = pd.DataFrame(columns=line_keys, index=ts_index, dtype=float)

    # Transformer losses
    loss_trafo = pd.DataFrame(columns=['copper_loss_kw', 'iron_loss_kw'], index=ts_index, dtype=float)

    # --- Time-series power flow loop ---
    logger.info(f"\nRunning 3-phase power flow [{run_label}] for {n_steps} timesteps...")
    start_time = time.time()
    converged_count = 0

    for t, ts in enumerate(ts_index):
        # Update asymmetric load values
        if aggregate:
            for bus_load in buses_loads_unique:
                meters_on_bus = buses_loads.loc[buses_loads['BUS_NUMBER'] == bus_load, 'METER_NUMBER']
                idx = loads_dict[bus_load]
                for ph_col, sub_dict in [('p_a_mw', Ploads_run['a']), ('p_b_mw', Ploads_run['b']),
                                          ('p_c_mw', Ploads_run['c'])]:
                    val = sum(sub_dict.loc[ts, int(m)] for m in meters_on_bus)
                    net.asymmetric_load.loc[idx, ph_col] = val
                for ph_col, sub_dict in [('q_a_mvar', Qloads_run['a']), ('q_b_mvar', Qloads_run['b']),
                                          ('q_c_mvar', Qloads_run['c'])]:
                    val = sum(sub_dict.loc[ts, int(m)] for m in meters_on_bus)
                    net.asymmetric_load.loc[idx, ph_col] = val

            for bus_gen in buses_gens_unique:
                meters_on_bus = buses_gens.loc[buses_gens['BUS_NUMBER'] == bus_gen, 'METER_NUMBER']
                idx = gens_dict[bus_gen]
                for ph_col, sub_dict in [('p_a_mw', Pgens_sub['a']), ('p_b_mw', Pgens_sub['b']),
                                          ('p_c_mw', Pgens_sub['c'])]:
                    val = sum(sub_dict.loc[ts, int(m)] for m in meters_on_bus)
                    net.asymmetric_sgen.loc[idx, ph_col] = val
                for ph_col, sub_dict in [('q_a_mvar', Qgens_sub['a']), ('q_b_mvar', Qgens_sub['b']),
                                          ('q_c_mvar', Qgens_sub['c'])]:
                    val = sum(sub_dict.loc[ts, int(m)] for m in meters_on_bus)
                    net.asymmetric_sgen.loc[idx, ph_col] = val
        else:
            for meter in Ploads_run['a'].columns:
                idx = loads_dict[str(meter)]
                net.asymmetric_load.loc[idx, 'p_a_mw'] = Ploads_run['a'].loc[ts, meter]
                net.asymmetric_load.loc[idx, 'p_b_mw'] = Ploads_run['b'].loc[ts, meter]
                net.asymmetric_load.loc[idx, 'p_c_mw'] = Ploads_run['c'].loc[ts, meter]
                net.asymmetric_load.loc[idx, 'q_a_mvar'] = Qloads_run['a'].loc[ts, meter]
                net.asymmetric_load.loc[idx, 'q_b_mvar'] = Qloads_run['b'].loc[ts, meter]
                net.asymmetric_load.loc[idx, 'q_c_mvar'] = Qloads_run['c'].loc[ts, meter]

            for meter in Pgens_sub['a'].columns:
                idx = gens_dict[str(meter)]
                net.asymmetric_sgen.loc[idx, 'p_a_mw'] = Pgens_sub['a'].loc[ts, meter]
                net.asymmetric_sgen.loc[idx, 'p_b_mw'] = Pgens_sub['b'].loc[ts, meter]
                net.asymmetric_sgen.loc[idx, 'p_c_mw'] = Pgens_sub['c'].loc[ts, meter]
                net.asymmetric_sgen.loc[idx, 'q_a_mvar'] = Qgens_sub['a'].loc[ts, meter]
                net.asymmetric_sgen.loc[idx, 'q_b_mvar'] = Qgens_sub['b'].loc[ts, meter]
                net.asymmetric_sgen.loc[idx, 'q_c_mvar'] = Qgens_sub['c'].loc[ts, meter]

        # Run 3-phase unbalanced power flow
        try:
            pp.runpp_3ph(net)
            converged_count += 1
        except Exception as e:
            logger.warning(f"3ph power flow did not converge at t={t} ({ts}): {e}")
            continue  # skip this timestep, try next

        # ---- Extract per-phase results ----

        # Bus voltages (from res_bus_3ph)
        V_buses_a.loc[ts, :] = net.res_bus_3ph.loc[bus_indices, 'vm_a_pu'].values
        V_buses_b.loc[ts, :] = net.res_bus_3ph.loc[bus_indices, 'vm_b_pu'].values
        V_buses_c.loc[ts, :] = net.res_bus_3ph.loc[bus_indices, 'vm_c_pu'].values

        # Voltage unbalance factor (built-in in pandapower)
        if 'unbalance_percent' in net.res_bus_3ph.columns:
            VUF.loc[ts, :] = net.res_bus_3ph.loc[bus_indices, 'unbalance_percent'].values
        else:
            # Manual VUF calculation as fallback
            a = np.exp(1j * 2 * np.pi / 3)
            for bus_name, bus_idx in buses_dict.items():
                rb = net.res_bus_3ph.loc[bus_idx]
                va = rb['vm_a_pu'] * np.exp(1j * np.deg2rad(rb['va_a_degree']))
                vb = rb['vm_b_pu'] * np.exp(1j * np.deg2rad(rb['va_b_degree']))
                vc = rb['vm_c_pu'] * np.exp(1j * np.deg2rad(rb['va_c_degree']))
                v_pos = (va + a * vb + a**2 * vc) / 3
                v_neg = (va + a**2 * vb + a * vc) / 3
                VUF.loc[ts, bus_name] = abs(v_neg) / abs(v_pos) * 100 if abs(v_pos) > 1e-6 else 0

        # Line currents per phase (from res_line_3ph)
        i_lines_a.loc[ts, :] = net.res_line_3ph.loc[line_indices, 'i_a_from_ka'].values * 1000
        i_lines_b.loc[ts, :] = net.res_line_3ph.loc[line_indices, 'i_b_from_ka'].values * 1000
        i_lines_c.loc[ts, :] = net.res_line_3ph.loc[line_indices, 'i_c_from_ka'].values * 1000
        if 'i_n_from_ka' in net.res_line_3ph.columns:
            i_lines_n.loc[ts, :] = net.res_line_3ph.loc[line_indices, 'i_n_from_ka'].values * 1000
        i_lines_perc.loc[ts, :] = net.res_line_3ph.loc[line_indices, 'loading_percent'].values

        # Grid exchange (res_ext_grid_3ph has per-phase columns)
        if hasattr(net, 'res_ext_grid_3ph') and len(net.res_ext_grid_3ph) > 0:
            eg = net.res_ext_grid_3ph.loc[0]
            p_a = eg['p_a_mw'] * 1000   # kW
            p_b = eg['p_b_mw'] * 1000
            p_c = eg['p_c_mw'] * 1000
            q_a = eg['q_a_mvar'] * 1000  # kvar
            q_b = eg['q_b_mvar'] * 1000
            q_c = eg['q_c_mvar'] * 1000
            power_grid.loc[ts, 'P'] = p_a + p_b + p_c
            power_grid.loc[ts, 'Q'] = q_a + q_b + q_c
            # Per-phase power for validation against Janitza / smart meters
            trafo_3ph.loc[ts, 'P_a_kw'] = p_a
            trafo_3ph.loc[ts, 'P_b_kw'] = p_b
            trafo_3ph.loc[ts, 'P_c_kw'] = p_c
            trafo_3ph.loc[ts, 'Q_a_kvar'] = q_a
            trafo_3ph.loc[ts, 'Q_b_kvar'] = q_b
            trafo_3ph.loc[ts, 'Q_c_kvar'] = q_c
            trafo_3ph.loc[ts, 'S_a_kva'] = np.sqrt(p_a**2 + q_a**2)
            trafo_3ph.loc[ts, 'S_b_kva'] = np.sqrt(p_b**2 + q_b**2)
            trafo_3ph.loc[ts, 'S_c_kva'] = np.sqrt(p_c**2 + q_c**2)
        elif len(net.res_ext_grid) > 0:
            p_total = net.res_ext_grid.loc[0, 'p_mw'] * 1000
            q_total = net.res_ext_grid.loc[0, 'q_mvar'] * 1000
            power_grid.loc[ts, 'P'] = p_total
            power_grid.loc[ts, 'Q'] = q_total
            # Balanced fallback: split equally across phases
            trafo_3ph.loc[ts, 'P_a_kw'] = p_total / 3
            trafo_3ph.loc[ts, 'P_b_kw'] = p_total / 3
            trafo_3ph.loc[ts, 'P_c_kw'] = p_total / 3
            trafo_3ph.loc[ts, 'Q_a_kvar'] = q_total / 3
            trafo_3ph.loc[ts, 'Q_b_kvar'] = q_total / 3
            trafo_3ph.loc[ts, 'Q_c_kvar'] = q_total / 3
            s_per_phase = np.sqrt((p_total/3)**2 + (q_total/3)**2)
            trafo_3ph.loc[ts, 'S_a_kva'] = s_per_phase
            trafo_3ph.loc[ts, 'S_b_kva'] = s_per_phase
            trafo_3ph.loc[ts, 'S_c_kva'] = s_per_phase

        # Transformer (per-phase loading from res_trafo_3ph)
        if hasattr(net, 'res_trafo_3ph') and len(net.res_trafo_3ph) > 0:
            tr = net.res_trafo_3ph.loc[0]
            trafo_3ph.loc[ts, 'loading_percent'] = tr['loading_percent']
            trafo_3ph.loc[ts, 'loading_a_percent'] = tr['loading_a_percent']
            trafo_3ph.loc[ts, 'loading_b_percent'] = tr['loading_b_percent']
            trafo_3ph.loc[ts, 'loading_c_percent'] = tr['loading_c_percent']
        elif len(net.res_trafo) > 0:
            trafo_3ph.loc[ts, 'loading_percent'] = net.res_trafo.loc[0, 'loading_percent']

        # Transformer losses (copper = load-dependent, iron = no-load/core)
        if hasattr(net, 'res_trafo_3ph') and len(net.res_trafo_3ph) > 0:
            tr = net.res_trafo_3ph.loc[0]
            loss_trafo.loc[ts, 'copper_loss_kw'] = tr['pl_mw'] * 1000 if 'pl_mw' in tr.index else 0
            loss_trafo.loc[ts, 'iron_loss_kw'] = tr['pfe_mw'] * 1000 if 'pfe_mw' in tr.index else 0
        elif len(net.res_trafo) > 0:
            tr = net.res_trafo.loc[0]
            loss_trafo.loc[ts, 'copper_loss_kw'] = tr['pl_mw'] * 1000 if 'pl_mw' in tr.index else 0
            loss_trafo.loc[ts, 'iron_loss_kw'] = tr['pfe_mw'] * 1000 if 'pfe_mw' in tr.index else 0

        # Line losses (I²R)
        phase_loss, neutral_loss = compute_line_losses(net, line_indices)
        loss_lines_phase.loc[ts, :] = phase_loss
        loss_lines_neutral.loc[ts, :] = neutral_loss

        pp.reset_results(net)

        if (t + 1) % 100 == 0 or t == 0:
            elapsed = time.time() - start_time
            logger.info(f"  [{run_label}] {ts} - {t+1}/{n_steps} ({elapsed:.1f}s, {converged_count} converged)")

    elapsed = time.time() - start_time
    logger.info(f"\n[{run_label}] 3ph power flow completed: {converged_count}/{n_steps} converged ({elapsed:.1f}s)")

    # ---- Save results ----
    logger.info(f"\nSaving [{run_label}] results to {result_path}/")
    V_buses_a.to_parquet(result_path / 'V_buses_a.parquet')
    V_buses_b.to_parquet(result_path / 'V_buses_b.parquet')
    V_buses_c.to_parquet(result_path / 'V_buses_c.parquet')
    VUF.to_parquet(result_path / 'VUF.parquet')
    i_lines_a.to_parquet(result_path / 'i_lines_a.parquet')
    i_lines_b.to_parquet(result_path / 'i_lines_b.parquet')
    i_lines_c.to_parquet(result_path / 'i_lines_c.parquet')
    i_lines_n.to_parquet(result_path / 'i_lines_n.parquet')
    i_lines_perc.to_parquet(result_path / 'i_lines_perc.parquet')
    power_grid.to_parquet(result_path / 'power_grid.parquet')
    trafo_3ph.to_parquet(result_path / 'trafo_3ph.parquet')
    loss_lines_phase.to_parquet(result_path / 'loss_lines_phase.parquet')
    loss_lines_neutral.to_parquet(result_path / 'loss_lines_neutral.parquet')
    loss_trafo.to_parquet(result_path / 'loss_trafo.parquet')

    # ---- Generate plots ----
    logger.info(f"\nGenerating [{run_label}] plots...")

    # 1. Per-phase bus voltages
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=150, sharex=True)
    for ax, (label, V_df) in zip(axes, [('A', V_buses_a), ('B', V_buses_b), ('C', V_buses_c)]):
        V_num = V_df.apply(pd.to_numeric, errors='coerce')
        ax.plot(V_num, alpha=0.7, linewidth=0.5)
        ax.set_ylabel(f'V phase {label} [p.u.]')
        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='±5% limit')
        ax.axhline(y=1.05, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.9, 1.1)
    axes[0].set_title(f'Per-Phase Bus Voltages [{run_label}] - Substation {sub}')
    fig.tight_layout()
    fig.savefig(result_path / 'Vbuses_3ph.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Voltage Unbalance Factor
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    VUF_num = VUF.apply(pd.to_numeric, errors='coerce')
    ax.plot(VUF_num, alpha=0.5, linewidth=0.5)
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='EN 50160 limit (2%)')
    ax.set_ylabel('Voltage Unbalance Factor [%]')
    ax.set_title(f'VUF [{run_label}] - Substation {sub}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.savefig(result_path / 'VUF.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Per-phase max line currents
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    for label, i_df, color in [('A', i_lines_a, 'tab:red'), ('B', i_lines_b, 'tab:blue'), ('C', i_lines_c, 'tab:green')]:
        i_num = i_df.apply(pd.to_numeric, errors='coerce')
        ax.plot(i_num.max(axis=1), label=f'Max I phase {label}', color=color, alpha=0.7, linewidth=0.8)
    ax.set_ylabel('Line Current [A]')
    ax.set_title(f'Maximum Line Currents per Phase [{run_label}] - Substation {sub}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(result_path / 'Ilines_3ph.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. Neutral current
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    i_n_num = i_lines_n.apply(pd.to_numeric, errors='coerce')
    ax.plot(i_n_num, alpha=0.5, linewidth=0.5)
    ax.set_ylabel('Neutral Current [A]')
    ax.set_title(f'Neutral Line Currents [{run_label}] - Substation {sub}')
    ax.grid(True, alpha=0.3)
    fig.savefig(result_path / 'Ilines_neutral.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 5. Transformer loading (%) + per-phase apparent power (kVA)
    trafo_num = trafo_3ph.apply(pd.to_numeric, errors='coerce')
    trafo_cap_kva = net.trafo.sn_mva.iloc[0] * 1000  # kVA

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=150, sharex=True)

    # Top: loading %
    ax1.plot(trafo_num['loading_percent'], color='tab:purple', linewidth=0.8)
    ax1.axhline(y=100, color='red', linestyle='--', label='100% loading')
    ax1.set_ylabel('Transformer Loading [%]')
    ax1.set_title(f'Transformer Loading [{run_label}] — Substation {sub} ({trafo_cap_kva:.0f} kVA)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom: per-phase apparent power in kVA
    for phase, col, color in [('A', 'S_a_kva', 'tab:red'), ('B', 'S_b_kva', 'tab:blue'), ('C', 'S_c_kva', 'tab:green')]:
        ax2.plot(trafo_num[col], label=f'S phase {phase}', color=color, alpha=0.7, linewidth=0.8)
    s_total = trafo_num['S_a_kva'] + trafo_num['S_b_kva'] + trafo_num['S_c_kva']
    ax2.plot(s_total, label='S total', color='black', linewidth=1.0)
    ax2.axhline(y=trafo_cap_kva, color='red', linestyle='--', alpha=0.5, label=f'Rated {trafo_cap_kva:.0f} kVA')
    ax2.set_ylabel('Apparent Power [kVA]')
    ax2.set_xlabel('Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(result_path / 'trafo_loading_3ph.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 6. Total network losses over time
    loss_phase_num = loss_lines_phase.apply(pd.to_numeric, errors='coerce')
    loss_neutral_num = loss_lines_neutral.apply(pd.to_numeric, errors='coerce')
    total_phase = loss_phase_num.sum(axis=1)
    total_neutral = loss_neutral_num.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.fill_between(total_phase.index, 0, total_phase, alpha=0.4, color='tab:blue', label='Phase losses')
    ax.fill_between(total_phase.index, total_phase, total_phase + total_neutral,
                    alpha=0.4, color='tab:orange', label='Neutral losses')
    ax.set_ylabel('Line Losses [kW]')
    ax.set_title(f'Total Network Line Losses [{run_label}] - Substation {sub}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(result_path / 'line_losses.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"\n[{run_label}] Substation {sub} complete!")


def main():
    parser = argparse.ArgumentParser(description='Run 3-phase unbalanced power flow analysis')
    parser.add_argument('--max-timesteps', type=int, default=None,
                        help='Limit number of timesteps (for testing)')
    parser.add_argument('--aggregate', action='store_true', default=True,
                        help='Aggregate loads per bus (default: True)')
    parser.add_argument('--mode', choices=['unbalanced', 'balanced', 'both'], default='both',
                        help='Run unbalanced, balanced, or both (default: both)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Jason 3-Phase Power Flow Analysis")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 80)

    # --- Load per-phase input data ---
    logger.info("\nLoading per-phase input data...")

    Pload_a = pd.read_parquet(PF_INPUT_DIR / "Pload_a.parquet")
    Pload_b = pd.read_parquet(PF_INPUT_DIR / "Pload_b.parquet")
    Pload_c = pd.read_parquet(PF_INPUT_DIR / "Pload_c.parquet")
    Pprod_a = pd.read_parquet(PF_INPUT_DIR / "Pprod_a.parquet")
    Pprod_b = pd.read_parquet(PF_INPUT_DIR / "Pprod_b.parquet")
    Pprod_c = pd.read_parquet(PF_INPUT_DIR / "Pprod_c.parquet")
    Qload_a = pd.read_parquet(PF_INPUT_DIR / "Qload_a.parquet")
    Qload_b = pd.read_parquet(PF_INPUT_DIR / "Qload_b.parquet")
    Qload_c = pd.read_parquet(PF_INPUT_DIR / "Qload_c.parquet")
    Qprod_a = pd.read_parquet(PF_INPUT_DIR / "Qprod_a.parquet")
    Qprod_b = pd.read_parquet(PF_INPUT_DIR / "Qprod_b.parquet")
    Qprod_c = pd.read_parquet(PF_INPUT_DIR / "Qprod_c.parquet")
    topology = pd.read_parquet(PF_INPUT_DIR / "topology.parquet")
    meter_and_cabinet = pd.read_parquet(PF_INPUT_DIR / "meter_and_cabinet.parquet")

    logger.info(f"  Pload_a: {Pload_a.shape} ({Pload_a.index[0]} to {Pload_a.index[-1]})")
    logger.info(f"  Topology: {len(topology)} lines")
    logger.info(f"  Meters: {len(meter_and_cabinet)}")

    # Collect all DataFrames for aligned trimming
    all_dfs = [Pload_a, Pload_b, Pload_c, Pprod_a, Pprod_b, Pprod_c,
               Qload_a, Qload_b, Qload_c, Qprod_a, Qprod_b, Qprod_c]

    # Align time windows
    starting = max(df.index.min() for df in all_dfs)
    ending = min(df.index.max() for df in all_dfs)
    all_dfs = [df.loc[starting:ending] for df in all_dfs]

    if args.max_timesteps:
        all_dfs = [df.iloc[:args.max_timesteps] for df in all_dfs]
        logger.info(f"  Limited to {args.max_timesteps} timesteps")

    (Pload_a, Pload_b, Pload_c, Pprod_a, Pprod_b, Pprod_c,
     Qload_a, Qload_b, Qload_c, Qprod_a, Qprod_b, Qprod_c) = all_dfs

    # Reference index/columns from phase A
    ts_index = Pload_a.index

    # --- Process each substation ---
    substations = topology['Substation'].unique()
    logger.info(f"\nSubstations to process: {substations}")

    for sub in substations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Substation {sub} (3-phase)")
        logger.info(f"{'='*60}")

        topology_sub = topology[topology['Substation'] == sub].reset_index(drop=True)
        trafo_cap = topology_sub['TRANSFORMER_CAPACITY'].unique()[0] / 1000  # MVA
        vector_group_raw = topology_sub['VECTOR_GROUP'].iloc[0] if 'VECTOR_GROUP' in topology_sub.columns else 'Dyn5'
        vector_group = re.sub(r'\d+$', '', vector_group_raw)
        logger.info(f"  Transformer: {trafo_cap*1000:.0f} kVA, vector group: {vector_group} (from {vector_group_raw})")

        # ---- Build pandapower network ----
        net = pp.create_empty_network()
        buses_dict = {}
        lines_dict = {}
        loads_dict = {}
        gens_dict = {}

        # Transformer buses
        buses_dict['mv_trafo'] = pp.create_bus(net, vn_kv=V_MV, name='trafo_MV')
        buses_dict['lv_trafo'] = pp.create_bus(net, vn_kv=V_LV, name='trafo_LV')

        # External grid (slack)
        pp.create_ext_grid(net, bus=buses_dict['mv_trafo'], vm_pu=1.0, va_degree=0,
                           s_sc_max_mva=1000, rx_max=0.1,
                           name="slack_grid")
        net.ext_grid['x0x_max'] = 1.0
        net.ext_grid['r0x0_max'] = 0.1

        # Transformer with vector group from CSV
        pp.create_transformer_from_parameters(
            net, hv_bus=buses_dict['mv_trafo'], lv_bus=buses_dict['lv_trafo'],
            sn_mva=trafo_cap,
            vn_hv_kv=V_MV,
            vn_lv_kv=V_LV,
            vkr_percent=0.7,
            vk_percent=6,
            pfe_kw=0.45,
            i0_percent=0.2,
            vector_group=vector_group,
            vk0_percent=6,
            vkr0_percent=0.7,
            mag0_percent=100,
            mag0_rx=0,
            si0_hv_partial=0.9,
        )

        # LV buses
        dummy = topology_sub[~topology_sub['NODE1'].str.contains('LvFeeder', na=False)]
        buses = np.unique(np.concatenate((
            dummy['NODE1_value'].unique(),
            topology_sub['NODE2_value'].unique()
        )))
        for bus in buses:
            buses_dict[str(bus)] = pp.create_bus(net, vn_kv=V_LV, name=str(bus))

        # Lines with zero-sequence parameters
        feeder_count = 0
        for line_idx in topology_sub.index:
            row = topology_sub.loc[line_idx]
            r1 = float(row['RESISTANCE'])
            x1 = float(row['REACTANCE'])
            length_km = float(row['CABLE_LENGTH'] / 1000)

            if row['NODE1_type'] == 'LvFeeder':
                feeder_count += 1
                i_max = min(row['LV_FEEDER_FUSE_SIZE'], row['CURRENT_CARRYING_CAPACITY_PIPE'])
                lines_dict[f'feeder{feeder_count}'] = pp.create_line_from_parameters(
                    net,
                    from_bus=buses_dict['lv_trafo'],
                    to_bus=buses_dict[str(row['NODE2_value'])],
                    length_km=length_km,
                    r_ohm_per_km=r1, x_ohm_per_km=x1, c_nf_per_km=140.0,
                    r0_ohm_per_km=r1 * 4.0, x0_ohm_per_km=x1, c0_nf_per_km=140.0,
                    max_i_ka=float(i_max / 1000),
                    name=str(line_idx)
                )
            else:
                lines_dict[str(line_idx)] = pp.create_line_from_parameters(
                    net,
                    from_bus=buses_dict[str(row['NODE1_value'])],
                    to_bus=buses_dict[str(row['NODE2_value'])],
                    length_km=length_km,
                    r_ohm_per_km=r1, x_ohm_per_km=x1, c_nf_per_km=140.0,
                    r0_ohm_per_km=r1 * 4.0, x0_ohm_per_km=x1 * 4.0, c0_nf_per_km=140.0,
                    max_i_ka=float(row['CURRENT_CARRYING_CAPACITY_PIPE'] / 1000),
                    name=str(line_idx)
                )

        # Drop buses not reachable from the transformer (handles both
        # degree-zero buses AND disconnected subgraphs/islands)
        reachable = find_reachable_buses(net, buses_dict['lv_trafo'])
        disconnected = set(net.bus.index) - reachable
        if disconnected:
            logger.warning(f"Dropping {len(disconnected)} disconnected buses (not reachable from transformer)")
            for bus in sorted(disconnected, reverse=True):
                pp.drop_buses(net, [bus])
                to_remove = [k for k, v in buses_dict.items() if v == bus]
                for k in to_remove:
                    del buses_dict[k]
            # Sync lines_dict: pp.drop_buses also removes lines connected to dropped buses
            surviving_lines = set(net.line.index)
            stale = [k for k, v in lines_dict.items() if v not in surviving_lines]
            for k in stale:
                del lines_dict[k]
            if stale:
                logger.info(f"  Also removed {len(stale)} lines connected to dropped buses")

        # ---- Map meters to buses and isolate per-phase loads ----
        mapping = meter_and_cabinet.set_index('METER_NUMBER')

        # Build per-phase sub-matrices (W -> MW)
        phase_labels = ['a', 'b', 'c']
        Ploads_sub = {ph: pd.DataFrame(index=ts_index) for ph in phase_labels}
        Qloads_sub = {ph: pd.DataFrame(index=ts_index) for ph in phase_labels}
        Pgens_sub = {ph: pd.DataFrame(index=ts_index) for ph in phase_labels}
        Qgens_sub = {ph: pd.DataFrame(index=ts_index) for ph in phase_labels}

        pload_phases = {'a': Pload_a, 'b': Pload_b, 'c': Pload_c}
        qload_phases = {'a': Qload_a, 'b': Qload_b, 'c': Qload_c}
        pprod_phases = {'a': Pprod_a, 'b': Pprod_b, 'c': Pprod_c}
        qprod_phases = {'a': Qprod_a, 'b': Qprod_b, 'c': Qprod_c}

        for meter in Pload_a.columns:
            if meter in mapping.index:
                cabinet = str(mapping.loc[meter, 'CABINET_value'])
                if cabinet in buses_dict:
                    for ph in phase_labels:
                        Ploads_sub[ph] = pd.concat([Ploads_sub[ph], pload_phases[ph][[meter]] / 1e6], axis=1)
                        Qloads_sub[ph] = pd.concat([Qloads_sub[ph], qload_phases[ph][[meter]] / 1e6], axis=1)

        for meter in Pprod_a.columns:
            if meter in mapping.index:
                cabinet = str(mapping.loc[meter, 'CABINET_value'])
                if cabinet in buses_dict:
                    for ph in phase_labels:
                        Pgens_sub[ph] = pd.concat([Pgens_sub[ph], pprod_phases[ph][[meter]] / 1e6], axis=1)
                        Qgens_sub[ph] = pd.concat([Qgens_sub[ph], qprod_phases[ph][[meter]] / 1e6], axis=1)

        n_load_meters = Ploads_sub['a'].shape[1]
        logger.info(f"  Loads: {n_load_meters} meters")
        logger.info(f"  Generators: {Pgens_sub['a'].shape[1]} meters")

        if n_load_meters == 0:
            logger.warning("No loads found for this substation, skipping")
            continue

        # ---- Create asymmetric load/gen objects ----
        buses_loads = None
        buses_loads_unique = []
        buses_gens = None
        buses_gens_unique = []

        if args.aggregate:
            buses_loads = pd.DataFrame(index=range(n_load_meters))
            for m, meter in enumerate(Ploads_sub['a'].columns):
                cabinet = str(mapping.loc[meter, 'CABINET_value'])
                buses_loads.loc[m, 'METER_NUMBER'] = meter
                buses_loads.loc[m, 'BUS_NUMBER'] = buses_dict[cabinet]

            buses_loads_unique = buses_loads['BUS_NUMBER'].unique()
            for bus_load in buses_loads_unique:
                loads_dict[bus_load] = pp.create_asymmetric_load(
                    net, bus=int(bus_load),
                    p_a_mw=0.0, p_b_mw=0.0, p_c_mw=0.0,
                    q_a_mvar=0.0, q_b_mvar=0.0, q_c_mvar=0.0,
                    type='wye', name=str(int(bus_load))
                )

            buses_gens = pd.DataFrame(index=range(Pgens_sub['a'].shape[1]))
            for m, meter in enumerate(Pgens_sub['a'].columns):
                cabinet = str(mapping.loc[meter, 'CABINET_value'])
                buses_gens.loc[m, 'METER_NUMBER'] = meter
                buses_gens.loc[m, 'BUS_NUMBER'] = buses_dict[cabinet]

            buses_gens_unique = buses_gens['BUS_NUMBER'].unique()
            for bus_gen in buses_gens_unique:
                gens_dict[bus_gen] = pp.create_asymmetric_sgen(
                    net, bus=int(bus_gen),
                    p_a_mw=0.0, p_b_mw=0.0, p_c_mw=0.0,
                    q_a_mvar=0.0, q_b_mvar=0.0, q_c_mvar=0.0,
                    type='wye', name=str(int(bus_gen))
                )
        else:
            for meter in Ploads_sub['a'].columns:
                cabinet = str(mapping.loc[meter, 'CABINET_value'])
                loads_dict[str(meter)] = pp.create_asymmetric_load(
                    net, bus=buses_dict[cabinet],
                    p_a_mw=0.0, p_b_mw=0.0, p_c_mw=0.0,
                    q_a_mvar=0.0, q_b_mvar=0.0, q_c_mvar=0.0,
                    type='wye', name=str(meter)
                )
            for meter in Pgens_sub['a'].columns:
                cabinet = str(mapping.loc[meter, 'CABINET_value'])
                gens_dict[str(meter)] = pp.create_asymmetric_sgen(
                    net, bus=buses_dict[cabinet],
                    p_a_mw=0.0, p_b_mw=0.0, p_c_mw=0.0,
                    q_a_mvar=0.0, q_b_mvar=0.0, q_c_mvar=0.0,
                    type='wye', name=str(meter)
                )

        # ---- Build run list based on mode ----
        runs = []
        if args.mode in ('unbalanced', 'both'):
            runs.append(('unbalanced', Ploads_sub, Qloads_sub))
        if args.mode in ('balanced', 'both'):
            Ploads_bal, Qloads_bal = build_balanced_loads(Ploads_sub, Qloads_sub)
            runs.append(('balanced', Ploads_bal, Qloads_bal))

        for run_label, P_run, Q_run in runs:
            result_path = PF_RESULTS_DIR / str(sub) / run_label
            run_powerflow_loop(
                net=net,
                ts_index=ts_index,
                Ploads_run=P_run,
                Qloads_run=Q_run,
                Pgens_sub=Pgens_sub,
                Qgens_sub=Qgens_sub,
                buses_dict=buses_dict,
                lines_dict=lines_dict,
                loads_dict=loads_dict,
                gens_dict=gens_dict,
                buses_loads=buses_loads,
                buses_loads_unique=buses_loads_unique,
                buses_gens=buses_gens,
                buses_gens_unique=buses_gens_unique,
                aggregate=args.aggregate,
                result_path=result_path,
                run_label=run_label,
                sub=sub,
            )

        logger.info(f"\nSubstation {sub} complete!")

    logger.info("\n" + "=" * 80)
    logger.info("3-Phase power flow analysis complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
