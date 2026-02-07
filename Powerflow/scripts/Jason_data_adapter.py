#!/usr/bin/env python3
"""
Jason_data_adapter.py
=====================
Translates Icelandic ArcGIS topology and Databricks smart meter exports
into the power flow input format.

Writes translated data to Powerflow/output/topology/ files (overwrite mode).

Usage:
    python Jason_data_adapter.py [--dry-run]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
import argparse
from Jason_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# --- Substation-specific values (set in main() from config + CSVs) ---
SUBSTATION_ID = None
TRANSFORMER_CAPACITY_KVA = None
ZIP_CODE = None
LV_FEEDER_ID = None
LV_FEEDER_FUSE_SIZE = None
DEFAULT_SERVICE_FUSE_SIZE = None

# Cable R/X lookup (Ohm/km) from IEC 60228 at 20°C, underground XLPE at 50Hz
CABLE_PROPERTIES = {
    # (size_mm2, material): (R_ohm_per_km, X_ohm_per_km, capacity_A)
    (10, 'Cu'):  (1.830, 0.094, 73),
    (16, 'Cu'):  (1.150, 0.087, 98),
    (25, 'Cu'):  (0.727, 0.083, 129),
    (35, 'Cu'):  (0.524, 0.081, 152),
    (50, 'Cu'):  (0.387, 0.079, 179),
    (50, 'Al'):  (0.641, 0.078, 137),
    (70, 'Al'):  (0.443, 0.075, 169),
    (95, 'Al'):  (0.320, 0.072, 201),
    (120, 'Al'): (0.253, 0.072, 234),
    (150, 'Al'): (0.206, 0.073, 258),
    (185, 'Al'): (0.164, 0.072, 294),
    (240, 'Al'): (0.125, 0.070, 340),
    (300, 'Al'): (0.100, 0.069, 385),
}

# LV line types to include
LV_LINE_TYPES = ['Heimtaug', 'Lágspennudreifilögn', 'Lágspennustrengur', 'Lágspennu- og götuljósalögn']
# Note: 'Lágspennu- og götuljósalögn' = combined LV + street lighting trunk cables.
# These carry 400V power and are essential for network connectivity (not pure street lighting).

# Cable type mapping: material -> cable_type code
MATERIAL_TO_CABLE_TYPE = {
    'Al': 'PEX',   # Metal-sheathed Aluminum
    'Cu': 'PEX',   # Cross-linked polyethylene (copper cables)
}


def parse_gerd_decoded(gerd_decoded: str):
    """
    Parse GERD_DECODED field like '4 x 50 Al' or '4 x 150 Al'.
    Returns (phase_size_mm2, material_code).
    """
    if pd.isna(gerd_decoded) or not isinstance(gerd_decoded, str):
        return 50, 'Al'  # Default fallback

    match = re.search(r'(\d+)\s*x\s*(\d+)\s*(Al|Cu)', gerd_decoded, re.IGNORECASE)
    if match:
        size = int(match.group(2))
        material = match.group(3).capitalize()
        # Normalize: 'al' -> 'Al', 'cu' -> 'Cu'
        if material.lower() == 'al':
            material = 'Al'
        else:
            material = 'Cu'
        return size, material

    return 50, 'Al'  # Default fallback


def get_cable_properties(size_mm2: int, material: str):
    """Look up R, X, capacity for a cable specification."""
    key = (size_mm2, material)
    if key in CABLE_PROPERTIES:
        return CABLE_PROPERTIES[key]

    # Try closest size match for the same material
    same_material = {k: v for k, v in CABLE_PROPERTIES.items() if k[1] == material}
    if same_material:
        closest = min(same_material.keys(), key=lambda k: abs(k[0] - size_mm2))
        logger.warning(f"No exact match for {size_mm2}mm² {material}, using {closest[0]}mm² {closest[1]}")
        return same_material[closest]

    # Ultimate fallback
    logger.warning(f"No cable data for {size_mm2}mm² {material}, using 50mm² Al defaults")
    return (0.641, 0.078, 137)


def translate_topology(lines_df, cabinets_df, transformers_df):
    """
    Translate ArcGIS topology to lv_topology format.

    Returns DataFrame with 16 columns matching lv_topology.csv schema.
    """
    # Filter to LV lines only (exclude Götuljósalögn = street lighting)
    lv_mask = lines_df['HLUTVERK'].isin(LV_LINE_TYPES)
    lv_lines = lines_df[lv_mask].copy()
    logger.info(f"Filtered to {len(lv_lines)} LV lines from {len(lines_df)} total")

    for htype, count in lv_lines['HLUTVERK'].value_counts().items():
        logger.info(f"  {htype}: {count}")

    rows = []
    for _, line in lv_lines.iterrows():
        # Parse cable specification
        phase_size, material = parse_gerd_decoded(line.get('GERD_DECODED', ''))
        r_ohm_km, x_ohm_km, capacity_a = get_cable_properties(phase_size, material)

        cable_length = line.get('SHAPE_LENGD', 0.0)
        if pd.isna(cable_length):
            cable_length = line.get('LENGD', 0.0)
        if pd.isna(cable_length):
            cable_length = 0.0

        # Determine node1 and node2 from FROM/TO columns
        from_node = line.get('FROM', np.nan)
        to_node = line.get('TO', np.nan)
        hlutverk = line.get('HLUTVERK', '')

        # Handle node identification
        # FROM/TO columns contain values like:
        #   - "31832" (cabinet TENGINR)
        #   - "D1416" (transformer/substation connection point)
        #   - "194558" (meter endpoint for Heimtaug)
        #   - numeric values matching cabinet TENGINR or meter numbers

        if pd.isna(from_node) or str(from_node).strip() == '':
            # Skip lines with no connectivity info
            logger.warning(f"Skipping line {line['OBJECTID']}: missing FROM node")
            continue
        if pd.isna(to_node) or str(to_node).strip() == '':
            logger.warning(f"Skipping line {line['OBJECTID']}: missing TO node")
            continue

        from_str = str(from_node).strip()
        to_str = str(to_node).strip()
        # Remove trailing .0 from float-parsed numeric values (e.g., "31832.0" -> "31832")
        if from_str.endswith('.0') and from_str[:-2].isdigit():
            from_str = from_str[:-2]
        if to_str.endswith('.0') and to_str[:-2].isdigit():
            to_str = to_str[:-2]

        # Skip lines with placeholder node "0" (invalid in topology)
        if from_str == '0' or to_str == '0':
            logger.warning(f"Skipping line {line['OBJECTID']}: FROM={from_str}/TO={to_str} is invalid placeholder")
            continue

        # Determine node types
        # "D1416" or just "1416" as FROM typically means transformer/feeder connection
        if from_str.startswith('D') and from_str[1:].isdigit():
            # This is a distribution feeder line FROM transformer
            node1 = f"LvFeeder.{LV_FEEDER_ID}"
            node1_is_feeder = True
        else:
            node1 = f"Cabinet.{from_str}"
            node1_is_feeder = False

        if to_str.startswith('D') and to_str[1:].isdigit():
            node2 = f"LvFeeder.{LV_FEEDER_ID}"
        elif to_str == str(SUBSTATION_ID):
            # TO = 1416 means connecting to the transformer/substation
            node2 = f"Cabinet.{to_str}"
        else:
            node2 = f"Cabinet.{to_str}"

        cable_type = MATERIAL_TO_CABLE_TYPE.get(material, 'MAL')

        rows.append({
            'secondary_substation': f'SecondarySubstation.{SUBSTATION_ID}',
            'zip_code_secondary_substation': ZIP_CODE,
            'transformer': f'Transformer.{SUBSTATION_ID}',
            'transformer_capacity': TRANSFORMER_CAPACITY_KVA,
            'vector_group': VECTOR_GROUP,
            'lv_feeder': f'LvFeeder.{LV_FEEDER_ID}',
            'lv_feeder_fuse_size': LV_FEEDER_FUSE_SIZE,
            'node1': node1,
            'node2': node2,
            'cable_id': f'LvCable.{line["OBJECTID"]}',
            'cable_type': cable_type,
            'cable_length': round(cable_length, 5),
            'phase_size': phase_size,
            'phase_material': material.upper(),
            'cable_capacity': capacity_a,
            'resistance': round(r_ohm_km, 5),
            'reactance': round(x_ohm_km, 5),
        })

    topology_df = pd.DataFrame(rows)
    logger.info(f"Created {len(topology_df)} topology rows")
    return topology_df


def translate_meter_cabinet_connections(sm_df, lines_df, cabinets_df):
    """
    Create meter-cabinet connection mapping.

    Uses husveita_fastanumer as the actual meter ID (200 unique smart meters).
    Each meter maps through numer_heimlagnar (line ID) to its Heimtaug line
    endpoint in the topology (Cabinet.{numer_heimlagnar}).

    Returns DataFrame matching meter_cabinet_connection.csv schema (200 rows).
    """
    # Build unique meter -> line -> cabinet mapping from parquet
    mapping = sm_df.groupby('husveita_fastanumer').agg({
        'numer_heimlagnar': 'first',
        'tengiskapur': 'first'
    }).reset_index()

    logger.info(f"Found {len(mapping)} unique meters (husveita_fastanumer) in smart meter data")
    logger.info(f"  Across {mapping['numer_heimlagnar'].nunique()} lines (numer_heimlagnar)")
    logger.info(f"  Connected to {mapping['tengiskapur'].nunique()} cabinets (tengiskapur)")

    rows = []
    for _, row in mapping.iterrows():
        meter_id = str(row['husveita_fastanumer'])
        line_id = str(int(float(row['numer_heimlagnar'])))

        # The meter's cabinet in the topology is Cabinet.{numer_heimlagnar}
        # This is the Heimtaug line endpoint node where the meter physically connects.
        # The tengiskapur (e.g. 31832) is the upstream junction cabinet, but the
        # meter attaches at the TO end of the Heimtaug line = Cabinet.{line_id}.
        cabinet_node = f"Cabinet.{line_id}"

        rows.append({
            'meter_number': meter_id,
            'delivery_point_id': '',
            'cabinet': cabinet_node,
            'lv_feeder': f'LvFeeder.{LV_FEEDER_ID}',
            'has_heat_pump': 'false',
            'has_solar_panel': 'false',
            'capacity_solar_panel': '',
            'service_fuse_size': DEFAULT_SERVICE_FUSE_SIZE,
        })

    conn_df = pd.DataFrame(rows)
    logger.info(f"Created {len(conn_df)} meter-cabinet connections")
    return conn_df


def translate_phase_measurements(sm_df):
    """
    Translate smart meter parquet data to phase_measurements CSV format.

    Returns dict of {(year, month): DataFrame} for each month of data.
    """
    df = sm_df.copy()

    # Parse timestamp as datetime first (for year/month grouping), then convert to
    # consistent string format for CSV output (dask infers format from first partition;
    # inconsistent formats like "2025-10-01" vs "2025-09-01 00:00:00" cause ValueError)
    ts_parsed = pd.to_datetime(df['ts'])
    df['timestamp_dst'] = ts_parsed.dt.strftime('%Y-%m-%d %H:%M:%S')
    df['_ts_parsed'] = ts_parsed  # keep datetime for year/month grouping

    # Meter number: use husveita_fastanumer (200 unique meters, not numer_heimlagnar which is line ID)
    df['meter_number'] = df['husveita_fastanumer'].astype(str)

    # Voltages
    df['voltage_l1'] = df['V_a']
    df['voltage_l2'] = df['V_b']
    df['voltage_l3'] = df['V_c']

    # Active power import (P14 quadrants 1&4)
    df['active_power_p14_l1'] = df['P_a']
    df['active_power_p14_l2'] = df['P_b']
    df['active_power_p14_l3'] = df['P_c']

    # Active power export (P23 quadrants 2&3) - no export data available
    df['active_power_p23_l1'] = 0.0
    df['active_power_p23_l2'] = 0.0
    df['active_power_p23_l3'] = 0.0

    # Reactive power: split Q_total proportionally across phases by P ratio
    if 'Q_total' in df.columns:
        total_p = df['P_a'].abs() + df['P_b'].abs() + df['P_c'].abs()
        total_p_safe = total_p.replace(0, np.nan)

        df['reactive_power_q12_l1'] = (df['Q_total'] * df['P_a'].abs() / total_p_safe).fillna(0)
        df['reactive_power_q12_l2'] = (df['Q_total'] * df['P_b'].abs() / total_p_safe).fillna(0)
        df['reactive_power_q12_l3'] = (df['Q_total'] * df['P_c'].abs() / total_p_safe).fillna(0)
    else:
        # No reactive power data at all
        df['reactive_power_q12_l1'] = 0.0
        df['reactive_power_q12_l2'] = 0.0
        df['reactive_power_q12_l3'] = 0.0

    # Capacitive reactive power (Q34) - assumed zero (all inductive)
    df['reactive_power_q34_l1'] = 0.0
    df['reactive_power_q34_l2'] = 0.0
    df['reactive_power_q34_l3'] = 0.0

    # Output columns
    output_cols = [
        'meter_number', 'timestamp_dst',
        'voltage_l1', 'voltage_l2', 'voltage_l3',
        'active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3',
        'active_power_p23_l1', 'active_power_p23_l2', 'active_power_p23_l3',
        'reactive_power_q12_l1', 'reactive_power_q12_l2', 'reactive_power_q12_l3',
        'reactive_power_q34_l1', 'reactive_power_q34_l2', 'reactive_power_q34_l3',
    ]

    # Add current columns if available
    if 'I_a' in df.columns:
        df['current_l1'] = df['I_a']
        df['current_l2'] = df['I_b']
        df['current_l3'] = df['I_c']
        output_cols.extend(['current_l1', 'current_l2', 'current_l3'])

    # Group by year-month (use the parsed datetime, not the string version)
    df['year'] = df['_ts_parsed'].dt.year
    df['month'] = df['_ts_parsed'].dt.month

    monthly_dfs = {}
    for (year, month), group in df.groupby(['year', 'month']):
        monthly_dfs[(year, month)] = group[output_cols].copy()
        logger.info(f"  Month {year}-{month}: {len(group)} records")

    return monthly_dfs


def write_csv(df, filepath: Path):
    """Write DataFrame to CSV, overwriting any existing file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Wrote {len(df)} rows to {filepath.name}")


def fuse_size_for_transformer(kva):
    """Standard LV feeder fuse size (A) for a given transformer capacity (kVA)."""
    if kva <= 100:
        return 160.0
    elif kva <= 200:
        return 200.0
    elif kva <= 315:
        return 250.0
    elif kva <= 500:
        return 315.0
    elif kva <= 800:
        return 400.0
    elif kva <= 1000:
        return 630.0
    else:
        return 800.0


def main():
    parser = argparse.ArgumentParser(description='Translate Jason ArcGIS data to power flow format')
    parser.add_argument('--dry-run', action='store_true', help='Preview output without writing files')
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config()
    global SUBSTATION_ID, LV_FEEDER_ID, DEFAULT_SERVICE_FUSE_SIZE
    SUBSTATION_ID = cfg['substation_id']
    LV_FEEDER_ID = cfg['lv_feeder_id']
    DEFAULT_SERVICE_FUSE_SIZE = cfg['default_service_fuse_size']

    input_dir = cfg['topology_dir']
    output_dir = cfg['output_dir']
    smartmeter_path = cfg['smartmeter_dir'] / cfg['smartmeter_file']

    logger.info("=" * 80)
    logger.info("Jason Data Adapter: ArcGIS -> Power Flow format")
    logger.info("=" * 80)
    logger.info(f"Substation: {SUBSTATION_ID}")
    logger.info(f"Input dir:  {input_dir}")
    logger.info(f"Output dir: {output_dir}")

    # --- Validate input paths ---
    if not input_dir.exists():
        logger.error(f"Topology directory not found: {input_dir}")
        logger.error(f"No ArcGIS data export for substation {SUBSTATION_ID}. Skipping.")
        return

    if not smartmeter_path.exists():
        logger.error(f"Smart meter file not found: {smartmeter_path}")
        return

    # --- Load source data ---
    logger.info("\nLoading source data...")

    lines_df = pd.read_csv(input_dir / "lines_clean.csv")
    cabinets_df = pd.read_csv(input_dir / "cabinets.csv")
    transformers_df = pd.read_csv(input_dir / "transformers.csv")

    # --- Dynamic values from CSVs ---
    global TRANSFORMER_CAPACITY_KVA, ZIP_CODE, LV_FEEDER_FUSE_SIZE, VECTOR_GROUP
    TRANSFORMER_CAPACITY_KVA = int(transformers_df['MALRAUN'].iloc[0])
    ZIP_CODE = str(int(cabinets_df['PNR'].iloc[0]))
    LV_FEEDER_FUSE_SIZE = fuse_size_for_transformer(TRANSFORMER_CAPACITY_KVA)
    VECTOR_GROUP = str(transformers_df['TENGIFLOKKUR'].iloc[0])

    logger.info(f"  Transformer capacity: {TRANSFORMER_CAPACITY_KVA} kVA (from MALRAUN)")
    logger.info(f"  Vector group: {VECTOR_GROUP} (from TENGIFLOKKUR)")
    logger.info(f"  ZIP code: {ZIP_CODE} (from PNR)")
    logger.info(f"  LV feeder fuse: {LV_FEEDER_FUSE_SIZE} A")

    parquet_file = smartmeter_path
    sm_df = pd.read_parquet(parquet_file)

    logger.info(f"Loaded: {len(lines_df)} lines, {len(cabinets_df)} cabinets, "
                f"{len(transformers_df)} transformers, {len(sm_df)} SM records")

    # --- Step 1: Translate topology ---
    logger.info("\n--- Step 1: Translating topology ---")
    topology_df = translate_topology(lines_df, cabinets_df, transformers_df)

    if args.dry_run:
        print("\nTopology preview:")
        print(topology_df.to_string())
    else:
        write_csv(topology_df, output_dir / "lv_topology.csv")

    # --- Step 2: Translate meter-cabinet connections ---
    logger.info("\n--- Step 2: Translating meter-cabinet connections ---")
    conn_df = translate_meter_cabinet_connections(sm_df, lines_df, cabinets_df)

    if args.dry_run:
        print("\nMeter-cabinet connections preview:")
        print(conn_df.to_string())
    else:
        write_csv(conn_df, output_dir / "meter_cabinet_connection.csv")

    # --- Step 3: Translate phase measurements ---
    logger.info("\n--- Step 3: Translating phase measurements ---")
    monthly_dfs = translate_phase_measurements(sm_df)

    if args.dry_run:
        for (year, month), mdf in monthly_dfs.items():
            print(f"\nPhase measurements {year}-{month} preview (first 3 rows):")
            print(mdf.head(3).to_string())
    else:
        for (year, month), mdf in monthly_dfs.items():
            write_csv(mdf, output_dir / f"phase_measurements_{year}_{month}.csv")

    # --- Summary ---
    logger.info("\n" + "=" * 80)
    logger.info("Translation complete!")
    logger.info(f"  Topology rows:     {len(topology_df)}")
    logger.info(f"  Meter connections:  {len(conn_df)}")
    logger.info(f"  Monthly files:     {len(monthly_dfs)}")
    logger.info(f"  Total SM records:  {sum(len(df) for df in monthly_dfs.values())}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
