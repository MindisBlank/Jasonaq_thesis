#!/usr/bin/env python3
"""
Jason_prepare_powerflow.py
===========================
Converts smart meter parquet data and translated topology into per-phase
parquet format for unbalanced 3-phase power flow.

Output files (saved to Powerflow/output/pf_input/):
  - Pload_a/b/c.parquet : Active power load per phase (W)
  - Qload_a/b/c.parquet : Reactive power load per phase (var)
  - Pprod_a/b/c.parquet : Active power production per phase (W), zeros
  - Qprod_a/b/c.parquet : Reactive power production per phase (var), zeros
  - Pload.parquet        : Total active power (backward-compatible)
  - Qload.parquet        : Total reactive power (backward-compatible)
  - topology.parquet     : Network topology with UPPERCASE columns
  - meter_and_cabinet.parquet : Meter-to-cabinet mapping
  - meter_phase_assignment.csv : Phase assignment for traceability

Prerequisites:
  - Jason_data_adapter.py must have been run first

Usage:
    python Jason_prepare_powerflow.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from Jason_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Powerflow" / "output" / "pf_input"

# Set from config in main()
SUBSTATION_ID = None
LV_FEEDER_ID = None
SMARTMETER_FILE = None


def load_sm_profiles(project_root):
    """
    Load SM profiles from the source parquet file in data/smartmeter/.
    Returns (profiles, per_completed) where:
        profiles: dict of meter_id (str) -> DataFrame
        per_completed: float or None (coverage ratio from parquet)
    """
    parquet_file = project_root / "data" / "smartmeter" / SMARTMETER_FILE
    if not parquet_file.exists():
        logger.error(f"Source parquet not found: {parquet_file}")
        return None, None

    logger.info(f"Loading raw SM data from {parquet_file.name}")
    sm_df = pd.read_parquet(parquet_file)

    # Extract per_completed before splitting into per-meter dicts
    per_completed = None
    if 'per_completed' in sm_df.columns:
        pc_values = sm_df['per_completed'].dropna().unique()
        if len(pc_values) == 1:
            per_completed = float(pc_values[0])
        elif len(pc_values) > 1:
            logger.warning(f"Multiple per_completed values found: {pc_values}")

    # Convert to per-meter DataFrames using husveita_fastanumer (200 unique meters)
    sm_df['timestamp'] = pd.to_datetime(sm_df['ts'])
    sm_df['meter_id'] = sm_df['husveita_fastanumer'].astype(str)

    profiles = {}
    for meter_id, group in sm_df.groupby('meter_id'):
        df = group.set_index('timestamp').sort_index()
        profiles[meter_id] = df

    logger.info(f"Loaded {len(profiles)} meter profiles from source")
    return profiles, per_completed


def determine_coverage_ratio(per_completed_from_parquet, config_value):
    """
    Determine the coverage ratio for unmetered load scaling.

    Priority: config override > per_completed from parquet > 1.0 (no scaling)

    Args:
        per_completed_from_parquet: float or None from the smart meter parquet
        config_value: float or None from jason_config.yaml

    Returns:
        (coverage_ratio: float, source: str)
    """
    # 1. Config override
    if config_value is not None:
        cr = float(config_value)
        if not (0.0 < cr <= 1.0):
            raise ValueError(f"coverage_ratio must be in (0, 1], got {cr}")
        return cr, "config override"

    # 2. per_completed from parquet
    if per_completed_from_parquet is not None:
        cr = per_completed_from_parquet
        if 0.0 < cr <= 1.0:
            return cr, "parquet per_completed"
        else:
            logger.warning(f"per_completed={cr} out of range (0,1], defaulting to 1.0")

    # 3. Fallback
    return 1.0, "default (no scaling)"


# ---------------------------------------------------------------------------
#  Phase classification and assignment
# ---------------------------------------------------------------------------

def classify_meter_phases(profiles):
    """
    Determine which meters are single-phase vs 3-phase.

    Returns dict: {meter_id_int: True/False} where True = 3-phase meter.
    """
    is_3phase = {}

    for meter_id, df in profiles.items():
        meter_int = int(meter_id)

        # Source data has P_a, P_b, P_c columns
        has_b = False
        has_c = False
        if 'P_b' in df.columns:
            p_b = pd.to_numeric(df['P_b'], errors='coerce')
            has_b = p_b.abs().sum() > 0
        if 'P_c' in df.columns:
            p_c = pd.to_numeric(df['P_c'], errors='coerce')
            has_c = p_c.abs().sum() > 0
        is_3phase[meter_int] = has_b or has_c

    return is_3phase


def assign_phases(single_phase_meters, phase_ratio, seed=42):
    """
    Assign single-phase meters to phases A, B, C according to the ratio.

    Args:
        single_phase_meters: list of meter IDs (int)
        phase_ratio: [pct_a, pct_b, pct_c] summing to 100
        seed: random seed for reproducibility

    Returns dict: {meter_id: 'A'|'B'|'C'}
    """
    rng = np.random.RandomState(seed)
    n = len(single_phase_meters)

    n_a = int(round(n * phase_ratio[0] / 100))
    n_b = int(round(n * phase_ratio[1] / 100))
    n_c = n - n_a - n_b  # remainder to C to ensure sum = n

    assignments = ['A'] * n_a + ['B'] * n_b + ['C'] * n_c
    rng.shuffle(assignments)

    # Sort meter list for deterministic assignment before shuffle
    sorted_meters = sorted(single_phase_meters)
    return dict(zip(sorted_meters, assignments))


# ---------------------------------------------------------------------------
#  Per-phase P/Q matrix builder
# ---------------------------------------------------------------------------

def build_pload_qload_3ph(profiles, phase_assignment, is_3phase_map):
    """
    Build per-phase Pload and Qload DataFrames.

    Args:
        profiles: dict of meter_id -> DataFrame
        phase_assignment: dict of meter_id_int -> 'A'|'B'|'C' (single-phase only)
        is_3phase_map: dict of meter_id_int -> bool

    Returns:
        (Pload_a, Pload_b, Pload_c, Qload_a, Qload_b, Qload_c) — 6 DataFrames
        Index: DatetimeIndex, Columns: meter IDs (int), Values: Watts / var
    """
    pa_series, pb_series, pc_series = {}, {}, {}
    qa_series, qb_series, qc_series = {}, {}, {}

    for meter_id, df in profiles.items():
        meter_int = int(meter_id)
        _build_source_meter(df, meter_int, is_3phase_map, phase_assignment,
                            pa_series, pb_series, pc_series,
                            qa_series, qb_series, qc_series)

    # Assemble DataFrames
    result = []
    for series_dict in [pa_series, pb_series, pc_series,
                        qa_series, qb_series, qc_series]:
        frame = pd.DataFrame(series_dict)
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index().fillna(0)
        result.append(frame)

    return tuple(result)


def _build_source_meter(df, meter_int, is_3phase_map, phase_assignment,
                        pa, pb, pc, qa, qb, qc):
    """Extract per-phase P/Q from raw source data for one meter."""
    p_a = pd.to_numeric(df.get('P_a', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    p_b = pd.to_numeric(df.get('P_b', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    p_c = pd.to_numeric(df.get('P_c', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    q_total = pd.to_numeric(df.get('Q_total', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

    zero = pd.Series(0.0, index=df.index)

    if is_3phase_map.get(meter_int, False):
        # 3-phase meter: use per-phase P and Q directly
        pa[meter_int] = p_a
        pb[meter_int] = p_b
        pc[meter_int] = p_c
        qa[meter_int] = pd.to_numeric(df.get('Q_a', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        qb[meter_int] = pd.to_numeric(df.get('Q_b', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        qc[meter_int] = pd.to_numeric(df.get('Q_c', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    else:
        # Single-phase: all power in P_a, assign to designated network phase
        phase = phase_assignment.get(meter_int, 'A')
        # P_a holds the total for single-phase meters
        p_total = p_a
        if phase == 'A':
            pa[meter_int] = p_total; pb[meter_int] = zero; pc[meter_int] = zero
            qa[meter_int] = q_total; qb[meter_int] = zero; qc[meter_int] = zero
        elif phase == 'B':
            pa[meter_int] = zero; pb[meter_int] = p_total; pc[meter_int] = zero
            qa[meter_int] = zero; qb[meter_int] = q_total; qc[meter_int] = zero
        else:  # C
            pa[meter_int] = zero; pb[meter_int] = zero; pc[meter_int] = p_total
            qa[meter_int] = zero; qb[meter_int] = zero; qc[meter_int] = q_total


# ---------------------------------------------------------------------------
#  Topology and meter-cabinet
# ---------------------------------------------------------------------------

def build_topology_parquet(project_root):
    """
    Build topology.parquet from adapter output.

    Expected columns (UPPERCASE):
      Substation, TRANSFORMER_CAPACITY, NODE1, NODE1_type, NODE1_value,
      NODE2, NODE2_value, CABLE_LENGTH, RESISTANCE, REACTANCE,
      LV_FEEDER_FUSE_SIZE, CURRENT_CARRYING_CAPACITY_PIPE
    """
    topo_path = project_root / "Powerflow" / "output" / "topology" / "lv_topology.csv"
    logger.info(f"Loading topology from {topo_path}")

    topo_df = pd.read_csv(topo_path)

    sub_mask = topo_df['secondary_substation'] == f'SecondarySubstation.{SUBSTATION_ID}'
    topo_sub = topo_df[sub_mask].copy()
    logger.info(f"Substation {SUBSTATION_ID}: {len(topo_sub)} topology rows")

    if len(topo_sub) == 0:
        logger.error(f"No topology data found for SecondarySubstation.{SUBSTATION_ID}")
        return None

    out = pd.DataFrame()
    out['Substation'] = topo_sub['secondary_substation'].apply(
        lambda x: int(x.split('.')[-1]) if '.' in str(x) else x
    )
    out['TRANSFORMER_CAPACITY'] = topo_sub['transformer_capacity'].astype(float)
    out['VECTOR_GROUP'] = topo_sub['vector_group'].values if 'vector_group' in topo_sub.columns else 'Dyn5'
    out['NODE1'] = topo_sub['node1'].values
    out['NODE1_type'] = topo_sub['node1'].apply(
        lambda x: x.split('.')[0] if '.' in str(x) else 'Cabinet'
    )
    out['NODE1_value'] = topo_sub['node1'].apply(
        lambda x: x.split('.')[-1] if '.' in str(x) else x
    )
    out['NODE2'] = topo_sub['node2'].values
    out['NODE2_value'] = topo_sub['node2'].apply(
        lambda x: x.split('.')[-1] if '.' in str(x) else x
    )
    out['CABLE_LENGTH'] = topo_sub['cable_length'].astype(float)
    out['RESISTANCE'] = topo_sub['resistance'].astype(float)
    out['REACTANCE'] = topo_sub['reactance'].astype(float)
    out['LV_FEEDER_FUSE_SIZE'] = topo_sub['lv_feeder_fuse_size'].astype(float)
    out['CURRENT_CARRYING_CAPACITY_PIPE'] = topo_sub['cable_capacity'].astype(float)

    return out


def build_meter_cabinet_parquet(project_root):
    """Build meter_and_cabinet.parquet with METER_NUMBER (int), CABINET_value (str)."""
    conn_path = project_root / "Powerflow" / "output" / "topology" / "meter_cabinet_connection.csv"
    logger.info(f"Loading meter-cabinet mapping from {conn_path}")

    conn_df = pd.read_csv(conn_path, dtype=str)

    feeder_mask = conn_df['lv_feeder'] == f'LvFeeder.{LV_FEEDER_ID}'
    our_meters = conn_df[feeder_mask].copy()

    if len(our_meters) == 0:
        logger.warning("No meters found with LvFeeder filter, using all meters")
        our_meters = conn_df.copy()

    logger.info(f"Meter-cabinet connections: {len(our_meters)}")

    out = pd.DataFrame()
    out['METER_NUMBER'] = our_meters['meter_number'].apply(lambda x: int(float(x)))
    out['CABINET_value'] = our_meters['cabinet'].apply(
        lambda x: x.split('.')[-1] if '.' in str(x) else x
    )
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Prepare power flow input files (3-phase)')
    parser.add_argument('--use-raw', action='store_true', default=True,
                        help='Kept for backward compatibility (always uses source data)')
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config()
    global SUBSTATION_ID, LV_FEEDER_ID, SMARTMETER_FILE
    SUBSTATION_ID = cfg['substation_id']
    LV_FEEDER_ID = cfg['lv_feeder_id']
    SMARTMETER_FILE = cfg['smartmeter_file']

    logger.info("=" * 80)
    logger.info("Jason Power Flow Input Preparation (3-phase unbalanced)")
    logger.info(f"Substation: {SUBSTATION_ID}")
    logger.info(f"Phase ratio: {cfg['phase_ratio']} (seed={cfg['phase_seed']})")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load SM profiles ---
    profiles, per_completed = load_sm_profiles(PROJECT_ROOT)
    if not profiles:
        logger.error("No SM data available. Run Jason_data_adapter.py first.")
        return

    logger.info(f"Loaded profiles for {len(profiles)} meters")

    # --- Classify meters (single-phase vs 3-phase) ---
    logger.info("\nClassifying meters...")
    is_3phase_map = classify_meter_phases(profiles)
    single_phase_meters = [m for m, is3 in is_3phase_map.items() if not is3]
    three_phase_meters = [m for m, is3 in is_3phase_map.items() if is3]
    logger.info(f"  Single-phase meters: {len(single_phase_meters)}")
    logger.info(f"  Three-phase meters:  {len(three_phase_meters)}")

    # --- Assign phases to single-phase meters ---
    phase_assignment = assign_phases(single_phase_meters, cfg['phase_ratio'], cfg['phase_seed'])
    for ph in ['A', 'B', 'C']:
        count = sum(1 for v in phase_assignment.values() if v == ph)
        logger.info(f"  Phase {ph}: {count} single-phase meters")

    # --- Build per-phase P/Q matrices ---
    logger.info("\nBuilding per-phase Pload/Qload matrices...")
    Pload_a, Pload_b, Pload_c, Qload_a, Qload_b, Qload_c = build_pload_qload_3ph(
        profiles, phase_assignment, is_3phase_map
    )
    logger.info(f"  Shape: {Pload_a.shape[0]} timesteps x {Pload_a.shape[1]} meters")

    # --- Unmetered load compensation ---
    coverage_ratio, cr_source = determine_coverage_ratio(
        per_completed, cfg.get('coverage_ratio')
    )
    alpha = 1.0 / coverage_ratio

    logger.info(f"\nUnmetered load compensation:")
    logger.info(f"  Coverage ratio: {coverage_ratio:.4f} (source: {cr_source})")
    logger.info(f"  Alpha (scaling factor): {alpha:.4f}")
    logger.info(f"  Load increase: {(alpha - 1) * 100:.1f}%")

    if alpha != 1.0:
        Pload_a *= alpha
        Pload_b *= alpha
        Pload_c *= alpha
        Qload_a *= alpha
        Qload_b *= alpha
        Qload_c *= alpha
        logger.info(f"  Applied scaling to all 6 load matrices")
    else:
        logger.info(f"  No scaling applied (alpha=1.0)")

    # Total (backward-compatible)
    Pload = Pload_a.add(Pload_b, fill_value=0).add(Pload_c, fill_value=0)
    Qload = Qload_a.add(Qload_b, fill_value=0).add(Qload_c, fill_value=0)

    # Production = zeros per phase
    zeros = pd.DataFrame(0.0, index=Pload_a.index, columns=Pload_a.columns)
    Pprod_a = zeros.copy()
    Pprod_b = zeros.copy()
    Pprod_c = zeros.copy()
    Qprod_a = zeros.copy()
    Qprod_b = zeros.copy()
    Qprod_c = zeros.copy()

    # --- Build topology ---
    logger.info("\nBuilding topology parquet...")
    topology = build_topology_parquet(PROJECT_ROOT)
    if topology is None:
        return
    logger.info(f"  Topology: {len(topology)} cable segments")

    # --- Build meter-cabinet mapping ---
    logger.info("\nBuilding meter-cabinet mapping...")
    meter_and_cabinet = build_meter_cabinet_parquet(PROJECT_ROOT)
    logger.info(f"  Meter-cabinet: {len(meter_and_cabinet)} connections")

    # --- Validate: ensure all Pload meters are in meter_and_cabinet ---
    missing_meters = set(Pload.columns) - set(meter_and_cabinet['METER_NUMBER'].values)
    if missing_meters:
        logger.warning(f"  {len(missing_meters)} meters not in meter-cabinet mapping — dropping")
        valid_meters = [m for m in Pload.columns if m not in missing_meters]
        for name in ['Pload_a', 'Pload_b', 'Pload_c', 'Qload_a', 'Qload_b', 'Qload_c',
                      'Pprod_a', 'Pprod_b', 'Pprod_c', 'Qprod_a', 'Qprod_b', 'Qprod_c',
                      'Pload', 'Qload']:
            locals()[name] = locals()[name][valid_meters]

    # --- Save all parquets ---
    logger.info(f"\nSaving to {OUTPUT_DIR}...")

    # Per-phase load matrices
    for phase, df in [('a', Pload_a), ('b', Pload_b), ('c', Pload_c)]:
        df.to_parquet(OUTPUT_DIR / f"Pload_{phase}.parquet")
    for phase, df in [('a', Qload_a), ('b', Qload_b), ('c', Qload_c)]:
        df.to_parquet(OUTPUT_DIR / f"Qload_{phase}.parquet")

    # Per-phase production matrices
    for phase, df in [('a', Pprod_a), ('b', Pprod_b), ('c', Pprod_c)]:
        df.to_parquet(OUTPUT_DIR / f"Pprod_{phase}.parquet")
    for phase, df in [('a', Qprod_a), ('b', Qprod_b), ('c', Qprod_c)]:
        df.to_parquet(OUTPUT_DIR / f"Qprod_{phase}.parquet")

    # Backward-compatible totals
    Pload.to_parquet(OUTPUT_DIR / "Pload.parquet")
    Qload.to_parquet(OUTPUT_DIR / "Qload.parquet")
    Pprod_a.to_parquet(OUTPUT_DIR / "Pprod.parquet")
    Qprod_a.to_parquet(OUTPUT_DIR / "Qprod.parquet")

    # Topology + meter-cabinet
    topology.to_parquet(OUTPUT_DIR / "topology.parquet")
    meter_and_cabinet.to_parquet(OUTPUT_DIR / "meter_and_cabinet.parquet")

    # Phase assignment for traceability
    assignment_rows = []
    for m in sorted(is_3phase_map.keys()):
        assignment_rows.append({
            'meter_number': m,
            'phase': phase_assignment.get(m, 'ABC'),
            'is_3phase': is_3phase_map[m]
        })
    assignment_df = pd.DataFrame(assignment_rows)
    assignment_df.to_csv(OUTPUT_DIR / "meter_phase_assignment.csv", index=False)
    logger.info(f"  Saved meter_phase_assignment.csv ({len(assignment_df)} meters)")

    # Save unmetered load compensation metadata
    import json
    metadata = {
        'substation_id': SUBSTATION_ID,
        'coverage_ratio': coverage_ratio,
        'coverage_ratio_source': cr_source,
        'alpha': round(alpha, 6),
        'load_increase_pct': round((alpha - 1) * 100, 2),
        'n_metered': Pload_a.shape[1],
    }
    meta_path = OUTPUT_DIR / "load_scaling_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved {meta_path.name}")

    logger.info("\n" + "=" * 80)
    logger.info("Power flow input preparation complete (3-phase)!")
    logger.info(f"  Per-phase shape:   {Pload_a.shape}")
    logger.info(f"  Total Pload shape: {Pload.shape}")
    logger.info(f"  Topology:          {len(topology)} rows")
    logger.info(f"  Meter-cabinet:     {len(meter_and_cabinet)} rows")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
