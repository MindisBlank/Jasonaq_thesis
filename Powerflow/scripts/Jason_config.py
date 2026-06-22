"""
Jason_config.py
================
Shared configuration loader for all Jason_ scripts.

Reads jason_config.yaml and provides a single dict with all
substation-specific settings. To switch substations, edit:
  - IDs in jason_config.yaml
  - Per-substation overrides in the 'substations' section
"""

import yaml
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "Powerflow" / "config" / "jason_config.yaml"


def _derive_topology_dir(smartmeter_file, project_root):
    """
    Derive topology folder from smartmeter filename.
    'smartmeter_15min__substation_1416_transformer_sp1_...' -> '1416_Trafo_1'
    'smartmeter_15min__substation_0670_transformer_sp1_...' -> '670_Trafo_1'
    """
    m = re.search(r'substation_(\d+)_transformer_sp(\d+)', smartmeter_file)
    if not m:
        raise ValueError(f"Cannot parse substation/transformer from '{smartmeter_file}'")
    substation_id = str(int(m.group(1)))  # strip leading zeros
    trafo_num = m.group(2)
    folder_name = f"{substation_id}_Trafo_{trafo_num}"
    return project_root / "notebooks" / "data_exports" / folder_name


def load_substation_ids():
    """Return list of substation IDs (as strings) from config."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return [str(x) for x in cfg.get('IDs', [])]


def load_transformer_labels():
    """Return dict mapping substation ID (str) -> display label like '579 SP1'."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get('defaults', {})
    substations = cfg.get('substations', {})
    labels = {}
    for sub_id, overrides in substations.items():
        trafo = overrides.get('transformer', defaults.get('transformer', 1))
        labels[str(sub_id)] = f'{sub_id} SP{trafo}'
    return labels


def load_config(substation_id=None):
    """
    Load configuration for a specific substation.

    Args:
        substation_id: int or str. If None, uses first entry in IDs list.

    Returns dict with keys:
        substation_id (int), lv_feeder_id (str), smartmeter_file (str),
        default_service_fuse_size (int/float), phase_ratio (list), phase_seed (int),
        coverage_ratio (float or None),
        script_dir (Path), project_root (Path), topology_dir (Path),
        smartmeter_dir (Path), output_dir (Path)
    """
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # --- Resolve substation ID ---
    if substation_id is None:
        substation_id = int(cfg['IDs'][0])
    else:
        substation_id = int(substation_id)

    # --- Merge defaults + per-substation overrides ---
    defaults = cfg.get('defaults', {})
    sub_overrides = cfg.get('substations', {}).get(substation_id, {})

    def _get(key, fallback=None):
        """Per-substation value > defaults > fallback."""
        if key in sub_overrides:
            return sub_overrides[key]
        if key in defaults:
            return defaults[key]
        return fallback

    phase_ratio = _get('phase_ratio', [40, 30, 30])
    if len(phase_ratio) != 3 or abs(sum(phase_ratio) - 100) > 0.01:
        raise ValueError(f"phase_ratio must be 3 values summing to 100, got {phase_ratio}")

    transformer = _get('transformer', 1)
    date_range = _get('date_range', '20250801_20250901')

    # Auto-derive smartmeter filename
    smartmeter_file = (
        f"smartmeter_15min__substation_{substation_id:04d}"
        f"_transformer_sp{transformer}_{date_range}.parquet"
    )

    # --- Path overrides (optional, fall back to auto-derived defaults) ---
    smartmeter_file_override = _get('smartmeter_file', None)
    if smartmeter_file_override:
        smartmeter_file = smartmeter_file_override

    smartmeter_dir_str = _get('smartmeter_dir', None)
    smartmeter_dir = (
        PROJECT_ROOT / smartmeter_dir_str
        if smartmeter_dir_str
        else PROJECT_ROOT / "data" / "smartmeter"
    )

    topology_dir_str = _get('topology_dir', None)
    topology_dir = (
        PROJECT_ROOT / topology_dir_str
        if topology_dir_str
        else _derive_topology_dir(smartmeter_file, PROJECT_ROOT)
    )

    return {
        'substation_id': substation_id,
        'lv_feeder_id': str(substation_id),
        'smartmeter_file': smartmeter_file,
        'default_service_fuse_size': _get('default_service_fuse_size', 25),
        'phase_ratio': phase_ratio,
        'phase_seed': _get('phase_seed', 42),
        'coverage_ratio': _get('coverage_ratio', None),
        'growth_rates': _get('growth_rates', [0.02, 0.04, 0.06]),
        'script_dir': SCRIPT_DIR,
        'project_root': PROJECT_ROOT,
        'topology_dir': topology_dir,
        'smartmeter_dir': smartmeter_dir,
        'output_dir': PROJECT_ROOT / "Powerflow" / "output" / "topology",
    }
