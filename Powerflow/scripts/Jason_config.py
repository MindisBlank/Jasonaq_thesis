"""
Jason_config.py
================
Shared configuration loader for all Jason_ scripts.

Reads jason_config.yaml and provides a single dict with all
substation-specific settings. To switch substations, edit:
  - IDs in jason_config.yaml
  - smartmeter_file in Jason_settings
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


def load_config():
    """
    Load configuration from jason_config.yaml.

    Returns dict with keys:
        substation_id (int), lv_feeder_id (str), smartmeter_file (str),
        default_service_fuse_size (int/float), phase_ratio (list), phase_seed (int),
        script_dir (Path), project_root (Path), topology_dir (Path),
        smartmeter_dir (Path), output_dir (Path)
    """
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    substation_id = int(cfg['IDs'][0])
    jason = cfg.get('Jason_settings', {})

    phase_ratio = jason.get('phase_ratio', [40, 30, 30])
    if len(phase_ratio) != 3 or abs(sum(phase_ratio) - 100) > 0.01:
        raise ValueError(f"phase_ratio must be 3 values summing to 100, got {phase_ratio}")

    smartmeter_file = jason.get('smartmeter_file', '')
    topology_dir = _derive_topology_dir(smartmeter_file, PROJECT_ROOT)

    return {
        'substation_id': substation_id,
        'lv_feeder_id': str(substation_id),
        'smartmeter_file': smartmeter_file,
        'default_service_fuse_size': jason.get('default_service_fuse_size', 25),
        'phase_ratio': phase_ratio,
        'phase_seed': jason.get('phase_seed', 42),
        'script_dir': SCRIPT_DIR,
        'project_root': PROJECT_ROOT,
        'topology_dir': topology_dir,
        'smartmeter_dir': PROJECT_ROOT / "data" / "smartmeter",
        'output_dir': PROJECT_ROOT / "Powerflow" / "output" / "topology",
    }
