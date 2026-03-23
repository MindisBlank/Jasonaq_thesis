#!/usr/bin/env python3
"""
Jason_visualize_topology.py
============================
Creates visual maps of the network topology for validation.

Reads the translated topology (from Jason_data_adapter.py output) and
cross-references raw ArcGIS files (cabinets.csv, lines_clean.csv,
transformers.csv) for spatial coordinates only.

Produces three plots per substation:
  1. Geographic layout  (Jason_topology_map.png)
  2. Tree / hierarchical layout (Jason_topology_tree.png)
  3. Force-directed spring layout (Jason_topology_spring.png)

Saves output to Powerflow/output/pf_results/{substation_id}/

Usage:
    python Jason_visualize_topology.py
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import deque
import re
import logging
import networkx as nx
from Jason_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Path to translated topology data (output from Jason_data_adapter.py)
TOPO_DIR = PROJECT_ROOT / "Powerflow" / "output" / "topology"

# Set from config in main()
SUBSTATION_ID = None
LV_LINE_TYPES = ['Heimtaug', 'Lágspennudreifilögn']


# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Coordinate lookup  (raw ArcGIS -> x, y)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Data loading & classification
# ---------------------------------------------------------------------------

def load_topology_data(substation_id=None):
    """
    Load config, topology CSVs, raw ArcGIS data.  Classify nodes and edges.

    Returns
    -------
    dict with keys:
        cfg, topo_sub, conn_sub, node_coords, node_types,
        edge_types, all_nodes, cabinet_tenginr
    """
    cfg = load_config(substation_id)
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
        return None

    # --- Load raw ArcGIS data for coordinates only ---
    topology_dir = cfg['topology_dir']
    logger.info(f"Loading raw ArcGIS data for coordinates from {topology_dir}...")
    lines_df = pd.read_csv(topology_dir / "lines_clean.csv")
    cabinets_df = pd.read_csv(topology_dir / "cabinets.csv")
    transformers_df = pd.read_csv(topology_dir / "transformers.csv")

    raw_coords = build_coordinate_lookup(cabinets_df, lines_df, transformers_df)
    logger.info(f"Coordinate lookup: {len(raw_coords)} raw node positions")

    # --- Identify junction cabinets from cabinets.csv ---
    cabinet_tenginr = set(str(int(c['TENGINR'])) for _, c in cabinets_df.iterrows())

    # --- Collect all unique nodes and classify them ---
    node_coords = {}   # processed_node_name -> (x, y)
    node_types = {}    # processed_node_name -> 'transformer' | 'junction_cabinet' | 'meter_endpoint'

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
    edge_types = []  # list of (node1, node2, edge_type, attrs_dict)
    for _, row in topo_sub.iterrows():
        n1, n2 = row['node1'], row['node2']
        cable_len = float(row.get('cable_length', 0) or 0)
        cable_type = row.get('cable_type', '')
        phase_size = row.get('phase_size', '')
        resistance = row.get('resistance', '')

        if n1.startswith('LvFeeder.'):
            etype = 'feeder'        # Transformer to cabinet
        elif node_types.get(n2) == 'meter_endpoint':
            etype = 'service'       # Cabinet to meter endpoint
        else:
            etype = 'distribution'  # Cabinet to cabinet

        edge_types.append((n1, n2, etype, {
            'cable_length': cable_len,
            'cable_type': cable_type,
            'phase_size': phase_size,
            'resistance': resistance,
        }))

    return {
        'cfg': cfg,
        'topo_sub': topo_sub,
        'conn_sub': conn_sub,
        'node_coords': node_coords,
        'node_types': node_types,
        'edge_types': edge_types,
        'all_nodes': all_nodes,
        'cabinet_tenginr': cabinet_tenginr,
    }


# ---------------------------------------------------------------------------
#  Shared drawing
# ---------------------------------------------------------------------------

def draw_topology(ax, positions, node_types, edge_types, substation_id,trafo_labels,
                  topo_sub, conn_sub, all_nodes, title_suffix=""):
    """
    Draw the topology (nodes + edges) onto a matplotlib Axes using the
    supplied *positions* dict  {node_name: (x, y)}.

    The same colours, markers, and legend are used regardless of layout.
    """
    # Colour / size / marker settings
    colors = {
        'transformer': '#D32F2F',
        'junction_cabinet': '#1976D2',
        'meter_endpoint': '#00AA00',
    }
    sizes = {
        'transformer': 250,
        'junction_cabinet': 150,
        'meter_endpoint': 50,
    }
    markers = {
        'transformer': 's',
        'junction_cabinet': 'D',
        'meter_endpoint': 'o',
    }
    edge_styles = {
        'feeder':       {'color': '#9C27B0', 'linewidth': 2.5, 'linestyle': '-'},
        'distribution': {'color': '#00C853', 'linewidth': 2.0, 'linestyle': '-'},
        'service':      {'color': '#FF6B00', 'linewidth': 1.0, 'linestyle': '-'},
    }

    # --- Draw edges ---
    for n1, n2, etype, attrs in edge_types:
        if n1 in positions and n2 in positions:
            x1, y1 = positions[n1]
            x2, y2 = positions[n2]
            style = edge_styles.get(etype, edge_styles['service'])
            ax.plot([x1, x2], [y1, y2], **style, alpha=0.7, zorder=1)

    # --- Draw nodes ---
    for node_name, (x, y) in positions.items():
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
            ax.annotate(node_name, (x, y), fontsize=5.5, fontweight='bold',
                        ha='center', va='bottom', xytext=(0, 10),
                        textcoords='offset points', color=colors[ntype])
        else:  # meter_endpoint — no label, just the dot
            pass

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor=colors['transformer'], label='Transformer (LvFeeder)'),
        mpatches.Patch(facecolor=colors['junction_cabinet'], label='Junction Cabinet'),
        mpatches.Patch(facecolor=colors['meter_endpoint'], label='Meter Endpoint'),
        plt.Line2D([0], [0], **edge_styles['feeder'], label='Feeder Trunk'),
        plt.Line2D([0], [0], **edge_styles['distribution'], label='Distribution Cable'),
        plt.Line2D([0], [0], **edge_styles['service'], label='Service Cable'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

    # --- Title ---
    n_trans = sum(1 for v in node_types.values() if v == 'transformer')
    n_junct = sum(1 for v in node_types.values() if v == 'junction_cabinet')
    n_endpt = sum(1 for v in node_types.values() if v == 'meter_endpoint')

    sub_label = trafo_labels.get(str(substation_id), f'{substation_id} SP1')
    ax.set_title(
        f'Network Topology: {sub_label} {title_suffix}\n'
        f'{len(topo_sub)} cables, {len(all_nodes)} nodes '
        f'({n_trans} transformer, {n_junct} junction cabinets, {n_endpt} endpoints) | '
        f'{len(conn_sub)} meters',
        fontsize=12, fontweight='bold')

    # --- Clean frame ---
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#CCCCCC')


# ---------------------------------------------------------------------------
#  NetworkX graph + abstract layouts
# ---------------------------------------------------------------------------

def build_networkx_graph(edge_types):
    """Build an undirected networkx Graph from the edge_types list."""
    G = nx.Graph()
    for n1, n2, etype, attrs in edge_types:
        G.add_edge(n1, n2, edge_type=etype, **attrs)
    return G


def compute_tree_layout(G, node_types):
    """
    BFS from the transformer root, assigning y-levels by depth and
    spreading children evenly on x.

    Returns positions dict  {node_name: (x, y)}.
    """
    # Find root (transformer node)
    root = None
    for n, ntype in node_types.items():
        if ntype == 'transformer' and n in G:
            root = n
            break
    if root is None:
        logger.warning("No transformer node found for tree layout — falling back to spring layout.")
        return nx.spring_layout(G, seed=42)

    # BFS to get depth levels
    depths = {root: 0}
    queue = deque([root])
    visited = {root}
    children_order = {}  # parent -> [child, ...]

    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                depths[neighbor] = depths[node] + 1
                queue.append(neighbor)
                children_order.setdefault(node, []).append(neighbor)

    # Include any disconnected nodes at the bottom
    max_depth = max(depths.values()) if depths else 0
    for n in G.nodes():
        if n not in depths:
            max_depth += 1
            depths[n] = max_depth

    # Group nodes by depth level
    levels = {}
    for n, d in depths.items():
        levels.setdefault(d, []).append(n)

    # Sort nodes within each level for consistent ordering
    for d in levels:
        levels[d].sort()

    # Assign positions: x spread evenly, y = -depth (top-down)
    positions = {}
    for d, nodes in levels.items():
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = (i + 0.5) / n_nodes  # normalise to [0, 1]
            y = -d                    # root at top
            positions[node] = (x, y)

    return positions


def compute_spring_layout(G, node_types):
    """
    Force-directed spring layout seeded from the tree layout so the
    structural hierarchy is preserved while the spring relaxation
    smooths spacing and reduces overlaps.

    Returns positions dict  {node_name: (x, y)}.
    """
    # Use tree layout as starting positions — deterministic & structurally aware
    init_pos = compute_tree_layout(G, node_types)

    # Pin the transformer so it stays central
    fixed_nodes = [n for n, t in node_types.items()
                   if t == 'transformer' and n in G]

    k = 1.8 / np.sqrt(max(G.number_of_nodes(), 1))
    positions = nx.spring_layout(
        G,
        pos=init_pos,
        fixed=fixed_nodes if fixed_nodes else None,
        k=k,
        iterations=200,
        seed=42,
    )
    return positions


# ---------------------------------------------------------------------------
#  Saving helper
# ---------------------------------------------------------------------------

def save_figure(fig, filename):
    """Save figure to the per-substation results folder."""
    output_dir = PROJECT_ROOT / "Powerflow" / "output" / "pf_results" / str(SUBSTATION_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize LV topology')
    parser.add_argument('--substation', type=str, default=None,
                        help='Substation ID (overrides config IDs[0])')
    args = parser.parse_args()

    data = load_topology_data(substation_id=args.substation)
    if data is None:
        return

    # Load transformer labels for plot titles
    try:
        from Jason_config import load_transformer_labels
        trafo_labels = load_transformer_labels()
    except Exception:
        trafo_labels = {}

    topo_sub = data['topo_sub']
    conn_sub = data['conn_sub']
    node_coords = data['node_coords']
    node_types = data['node_types']
    edge_types = data['edge_types']
    all_nodes = data['all_nodes']

    # Build networkx graph for abstract layouts
    G = build_networkx_graph(edge_types)

    # ---------------------------------------------------------------
    # Plot 1: Geographic layout (existing behaviour)
    # ---------------------------------------------------------------
    fig1, ax1 = plt.subplots(1, 1, figsize=(18, 16), dpi=150)
    draw_topology(ax1, node_coords, node_types, edge_types,
                  SUBSTATION_ID, trafo_labels, topo_sub, conn_sub, all_nodes,
                  title_suffix="(Geographic)")
    ax1.set_xlabel('X (ISN93)', fontsize=10)
    ax1.set_ylabel('Y (ISN93)', fontsize=10)
    ax1.set_aspect('equal')
    save_figure(fig1, "Jason_topology_map.png")

    # ---------------------------------------------------------------
    # Plot 2: Tree / hierarchical layout
    # ---------------------------------------------------------------
    tree_pos = compute_tree_layout(G, node_types)
    fig2, ax2 = plt.subplots(1, 1, figsize=(18, 12), dpi=150)
    draw_topology(ax2, tree_pos, node_types, edge_types,
                  SUBSTATION_ID, trafo_labels, topo_sub, conn_sub, all_nodes,
                  title_suffix="(Tree Layout)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    save_figure(fig2, "Jason_topology_tree.png")

    # ---------------------------------------------------------------
    # Plot 3: Force-directed spring layout
    # ---------------------------------------------------------------
    spring_pos = compute_spring_layout(G, node_types)
    fig3, ax3 = plt.subplots(1, 1, figsize=(16, 16), dpi=150)
    draw_topology(ax3, spring_pos, node_types, edge_types,
                  SUBSTATION_ID, trafo_labels, topo_sub, conn_sub, all_nodes,
                  title_suffix="(Spring Layout)")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect('equal')
    save_figure(fig3, "Jason_topology_spring.png")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    n_trans = sum(1 for v in node_types.values() if v == 'transformer')
    n_junct = sum(1 for v in node_types.values() if v == 'junction_cabinet')
    n_endpt = sum(1 for v in node_types.values() if v == 'meter_endpoint')

    logger.info(f"\nTopology Summary:")
    logger.info(f"  Transformers:        {n_trans}")
    logger.info(f"  Junction cabinets:   {n_junct}")
    logger.info(f"  Meter endpoints:     {n_endpt}")
    logger.info(f"  Total meters:        {len(conn_sub)}")
    logger.info(f"  Cable segments:      {len(topo_sub)}")
    for etype in ['feeder', 'distribution', 'service']:
        count = sum(1 for _, _, et, _ in edge_types if et == etype)
        logger.info(f"  {etype.capitalize()} edges: {count}")

    # Report missing coordinates (geographic plot only)
    missing = [n for n in all_nodes if n not in node_coords]
    if missing:
        logger.warning(f"\nNodes without coordinates ({len(missing)}):")
        for n in sorted(missing):
            logger.warning(f"  {n}")


if __name__ == '__main__':
    main()
