#!/usr/bin/env python3
"""
Jason_visualize_topology.py
============================
Creates a visual map of the network topology for validation.

Reads the translated topology (from Jason_data_adapter.py output) and
cross-references raw ArcGIS files (cabinets.csv, lines_clean.csv,
transformers.csv) for spatial coordinates only.

This validates that the data translation produces a correct, connected network.

Saves output to Powerflow/output/Jason_topology_map.png

Usage:
    python Jason_visualize_topology.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import re
import logging
from Jason_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Powerflow" / "output"

# Path to translated topology data (output from Jason_data_adapter.py)
TOPO_DIR = PROJECT_ROOT / "Powerflow" / "output" / "topology"

# Set from config in main()
SUBSTATION_ID = None
LV_LINE_TYPES = ['Heimtaug', 'Lágspennudreifilögn']


def parse_geometry_first_point(geom_str):
    """Extract first coordinate from WKT LINESTRING or POINT geometry."""
    if pd.isna(geom_str):
        return None, None
    geom_str = str(geom_str)
    match = re.search(r'[\(\s]([\d.]+)\s+([\d.]+)', geom_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def parse_geometry_last_point(geom_str):
    """Extract last coordinate from WKT LINESTRING geometry."""
    if pd.isna(geom_str):
        return None, None
    geom_str = str(geom_str)
    matches = re.findall(r'([\d.]+)\s+([\d.]+)', geom_str)
    if matches:
        return float(matches[-1][0]), float(matches[-1][1])
    return None, None


def clean_node_id(node_val):
    """Clean node ID from float-parsed values (e.g., '31832.0' -> '31832')."""
    s = str(node_val).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    return s


def build_coordinate_lookup(cabinets_df, lines_df, transformers_df):
    """
    Build a mapping from raw ArcGIS IDs to (x, y) coordinates.

    Returns dict: raw_id_str -> (x, y)
    Uses cabinets.csv for junction cabinets, lines_clean.csv geometry for
    line endpoints, and transformers.csv for transformer location.
    """
    coords = {}

    # 1. Cabinet coordinates from cabinets.csv (X, Y columns)
    for _, c in cabinets_df.iterrows():
        tenginr = str(int(c['TENGINR']))
        coords[tenginr] = (c['X'], c['Y'])

    # 2. Transformer location from transformers.csv
    for _, t in transformers_df.iterrows():
        dnr = str(int(t['DNR']))
        if 'X' in t and 'Y' in t and pd.notna(t.get('X')) and pd.notna(t.get('Y')):
            coords[f"D{dnr}"] = (t['X'], t['Y'])

    # 3. Line endpoints from lines_clean.csv geometry
    lv_lines = lines_df[lines_df['HLUTVERK'].isin(LV_LINE_TYPES)].copy()
    for _, line in lv_lines.iterrows():
        from_id = clean_node_id(line.get('FROM', ''))
        to_id = clean_node_id(line.get('TO', ''))

        # FROM = first geometry point
        if from_id and from_id not in coords:
            x, y = parse_geometry_first_point(line.get('geometry'))
            if x is not None:
                coords[from_id] = (x, y)

        # TO = last geometry point
        if to_id and to_id not in coords:
            x, y = parse_geometry_last_point(line.get('geometry'))
            if x is not None:
                coords[to_id] = (x, y)

    # Also get D1416 (transformer busbar) from the FROM of distribution lines
    for _, line in lv_lines.iterrows():
        from_id = clean_node_id(line.get('FROM', ''))
        if from_id.startswith('D') and from_id not in coords:
            x, y = parse_geometry_first_point(line.get('geometry'))
            if x is not None:
                coords[from_id] = (x, y)

    return coords


def resolve_node_coord(node_name, raw_coords):
    """
    Resolve a processed topology node name (e.g., 'Cabinet.31832', 'LvFeeder.1416')
    to (x, y) coordinates using the raw ArcGIS coordinate lookup.
    """
    # Extract the numeric/string ID from the processed node name
    if '.' in node_name:
        raw_id = node_name.split('.')[-1]
    else:
        raw_id = node_name

    # Direct match
    if raw_id in raw_coords:
        return raw_coords[raw_id]

    # LvFeeder.1416 -> try D1416 (transformer busbar)
    if node_name.startswith('LvFeeder.'):
        d_key = f"D{raw_id}"
        if d_key in raw_coords:
            return raw_coords[d_key]

    return None, None


def main():
    # --- Load config ---
    cfg = load_config()
    global SUBSTATION_ID
    SUBSTATION_ID = cfg['substation_id']

    logger.info("Loading topology for visualization...")
    logger.info(f"Substation: {SUBSTATION_ID}")

    # --- Load topology data (from adapter output) ---
    topo_path = TOPO_DIR / "lv_topology.csv"
    conn_path = TOPO_DIR / "meter_cabinet_connection.csv"
    logger.info(f"Topology dir: {TOPO_DIR}")

    topo_df = pd.read_csv(topo_path, dtype=str)
    conn_df = pd.read_csv(conn_path, dtype=str)

    # Filter to our substation
    topo_sub = topo_df[topo_df['secondary_substation'] == f'SecondarySubstation.{SUBSTATION_ID}'].copy()
    conn_sub = conn_df[conn_df['lv_feeder'] == f'LvFeeder.{SUBSTATION_ID}'].copy()

    logger.info(f"Topology: {len(topo_sub)} cable segments for substation {SUBSTATION_ID}")
    logger.info(f"Meter-cabinet connections: {len(conn_sub)} meters")

    if len(topo_sub) == 0:
        logger.error(f"No topology data found for substation {SUBSTATION_ID}. Run the data adapter first.")
        return

    # --- Load raw ArcGIS data for coordinates only ---
    topology_dir = cfg['topology_dir']
    logger.info(f"Loading raw ArcGIS data for coordinates from {topology_dir}...")
    lines_df = pd.read_csv(topology_dir / "lines_clean.csv")
    cabinets_df = pd.read_csv(topology_dir / "cabinets.csv")
    transformers_df = pd.read_csv(topology_dir / "transformers.csv")

    raw_coords = build_coordinate_lookup(cabinets_df, lines_df, transformers_df)
    logger.info(f"Coordinate lookup: {len(raw_coords)} raw node positions")

    # --- Build meter count per cabinet ---
    meter_counts = conn_sub.groupby('cabinet').size().to_dict() if len(conn_sub) > 0 else {}

    # --- Identify junction cabinets from cabinets.csv ---
    cabinet_tenginr = set(str(int(c['TENGINR'])) for _, c in cabinets_df.iterrows())

    # --- Collect all unique nodes and classify them ---
    node_coords = {}  # processed_node_name -> (x, y)
    node_types = {}   # processed_node_name -> 'transformer' | 'junction_cabinet' | 'meter_endpoint'

    all_nodes = set(topo_sub['node1'].unique()) | set(topo_sub['node2'].unique())

    for node_name in all_nodes:
        # Resolve coordinates
        xy = resolve_node_coord(node_name, raw_coords)
        if xy[0] is not None:
            node_coords[node_name] = xy
        else:
            logger.warning(f"No coordinates found for node: {node_name}")

        # Classify node type
        if node_name.startswith('LvFeeder.'):
            node_types[node_name] = 'transformer'
        elif '.' in node_name:
            raw_id = node_name.split('.')[-1]
            if raw_id in cabinet_tenginr:
                node_types[node_name] = 'junction_cabinet'
            else:
                node_types[node_name] = 'meter_endpoint'
        else:
            node_types[node_name] = 'meter_endpoint'

    logger.info(f"Nodes with coordinates: {len(node_coords)} / {len(all_nodes)}")

    # --- Classify edges ---
    # Determine if an edge is a feeder line (LvFeeder -> Cabinet) or service/distribution line
    edge_types = []  # list of (node1, node2, edge_type, attrs_dict)
    for _, row in topo_sub.iterrows():
        n1, n2 = row['node1'], row['node2']
        cable_len = float(row.get('cable_length', 0) or 0)
        cable_type = row.get('cable_type', '')
        phase_size = row.get('phase_size', '')
        resistance = row.get('resistance', '')

        if n1.startswith('LvFeeder.'):
            etype = 'feeder'  # Transformer to cabinet
        elif node_types.get(n2) == 'meter_endpoint':
            etype = 'service'  # Cabinet to meter endpoint
        else:
            etype = 'distribution'  # Cabinet to cabinet

        edge_types.append((n1, n2, etype, {
            'cable_length': cable_len,
            'cable_type': cable_type,
            'phase_size': phase_size,
            'resistance': resistance,
        }))

    # --- Build figure ---
    fig, ax = plt.subplots(1, 1, figsize=(18, 16), dpi=150)

    # Color and size settings
    colors = {
        'transformer': '#D32F2F',        # Red
        'junction_cabinet': "#1976D2",   # Blue
        'meter_endpoint': "#00AA00",     # Green
    }
    sizes = {
        'transformer': 250,
        'junction_cabinet': 150,
        'meter_endpoint': 50,
    }
    markers = {
        'transformer': 's',       # Square
        'junction_cabinet': 'D',  # Diamond
        'meter_endpoint': 'o',    # Circle
    }

    # Edge style by type
    edge_styles = {
        'feeder': {'color': "#9C27B0", 'linewidth': 2.5, 'linestyle': '-'},       # Purple for feeder trunk
        'distribution': {'color': "#00C853", 'linewidth': 2.0, 'linestyle': '-'},  # Green for distribution
        'service': {'color': "#FF6B00", 'linewidth': 1.0, 'linestyle': '-'},       # Orange for service
    }

    # Draw edges from PROCESSED topology
    for n1, n2, etype, attrs in edge_types:
        if n1 in node_coords and n2 in node_coords:
            x1, y1 = node_coords[n1]
            x2, y2 = node_coords[n2]
            style = edge_styles.get(etype, edge_styles['service'])
            ax.plot([x1, x2], [y1, y2], **style, alpha=0.7, zorder=1)

            # Edge label at midpoint
            cable_len = attrs['cable_length']
            cable_type = attrs['cable_type']
            phase_size = attrs['phase_size']
            if cable_len > 0:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                parts = []
                if cable_type:
                    parts.append(str(cable_type))
                if phase_size:
                    parts.append(f"{phase_size}mm2")
                parts.append(f"{cable_len:.0f}m")
                label = " ".join(parts)
                ax.annotate(label, (mx, my), fontsize=3.5, ha='center', va='center',
                            color='#555555', alpha=0.8, zorder=2)

    # Draw nodes
    for node_name, (x, y) in node_coords.items():
        ntype = node_types.get(node_name, 'meter_endpoint')
        ax.scatter(x, y, c=colors[ntype], s=sizes[ntype],
                   marker=markers[ntype], zorder=3, edgecolors='white', linewidths=0.5)

        # Label nodes
        if ntype == 'transformer':
            cap = topo_sub['transformer_capacity'].iloc[0] if 'transformer_capacity' in topo_sub.columns else '800'
            label = f"Transformer\n{node_name}\n{cap} kVA"
            ax.annotate(label, (x, y), fontsize=7, fontweight='bold',
                        ha='center', va='bottom', xytext=(0, 14),
                        textcoords='offset points', color=colors[ntype])
        elif ntype == 'junction_cabinet':
            n_meters = meter_counts.get(node_name, 0)
            label = f"{node_name}\n({n_meters} meters)"
            ax.annotate(label, (x, y), fontsize=5.5, fontweight='bold',
                        ha='center', va='bottom', xytext=(0, 10),
                        textcoords='offset points', color=colors[ntype])
        else:  # meter_endpoint
            n_meters = meter_counts.get(node_name, 0)
            raw_id = node_name.split('.')[-1] if '.' in node_name else node_name
            label = raw_id
            if n_meters > 0:
                label = f"{raw_id}\n({n_meters}m)"
            # Only show labels if not too many endpoints
            n_endpoints = sum(1 for v in node_types.values() if v == 'meter_endpoint')
            if n_endpoints < 50:
                ax.annotate(label, (x, y), fontsize=3.5, ha='center', va='bottom',
                            xytext=(0, 6), textcoords='offset points', color='#333333', alpha=0.7)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['transformer'], label='Transformer (LvFeeder)'),
        mpatches.Patch(facecolor=colors['junction_cabinet'], label='Junction Cabinet'),
        mpatches.Patch(facecolor=colors['meter_endpoint'], label='Meter Endpoint (Cabinet.{line_id})'),
        plt.Line2D([0], [0], **edge_styles['feeder'], label='Feeder Trunk (Transformer to Cabinet)'),
        plt.Line2D([0], [0], **edge_styles['distribution'], label='Distribution (Cabinet to Cabinet)'),
        plt.Line2D([0], [0], **edge_styles['service'], label='Service Cable (Cabinet to Endpoint)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

    # Title with topology stats
    n_trans = sum(1 for v in node_types.values() if v == 'transformer')
    n_junct = sum(1 for v in node_types.values() if v == 'junction_cabinet')
    n_endpt = sum(1 for v in node_types.values() if v == 'meter_endpoint')

    ax.set_title(
        f'Network Topology: Substation {SUBSTATION_ID}\n'
        f'{len(topo_sub)} cables, {len(all_nodes)} nodes '
        f'({n_trans} transformer, {n_junct} junction cabinets, {n_endpt} endpoints) | '
        f'{len(conn_sub)} meters',
        fontsize=12, fontweight='bold')
    ax.set_xlabel('X (ISN93)', fontsize=10)
    ax.set_ylabel('Y (ISN93)', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "Jason_topology_map.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved topology map to {output_path}")
    plt.close(fig)

    # Print summary
    logger.info(f"\nTopology Summary:")
    logger.info(f"  Transformers:        {n_trans}")
    logger.info(f"  Junction cabinets:   {n_junct}")
    logger.info(f"  Meter endpoints:     {n_endpt}")
    logger.info(f"  Total meters:        {len(conn_sub)}")
    logger.info(f"  Cable segments:      {len(topo_sub)}")
    for etype in ['feeder', 'distribution', 'service']:
        count = sum(1 for _, _, et, _ in edge_types if et == etype)
        logger.info(f"  {etype.capitalize()} edges: {count}")

    # Report missing coordinates
    missing = [n for n in all_nodes if n not in node_coords]
    if missing:
        logger.warning(f"\nNodes without coordinates ({len(missing)}):")
        for n in sorted(missing):
            logger.warning(f"  {n}")


if __name__ == '__main__':
    main()
